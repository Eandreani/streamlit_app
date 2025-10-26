import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value, PULP_CBC_CMD, LpStatus
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin

# --- CONFIGURACIÓN GLOBAL ---
# Puedes ajustar estos parámetros desde la UI de Streamlit más adelante,
# pero aquí definimos valores por defecto para iniciar.
N_WEEKS_OPT = 4
PRICE_GRID_STEP = 0.5
PRICE_MIN_DEFAULT = 10.0
PRICE_MAX_DEFAULT = 50.0
COSTO_UNITARIO_DEFAULT = 5.0
MARGEN_MIN_DIRECTO_DEFAULT = 0.15 # 15% de margen directo mínimo
P_MIN_ABS_DEFAULT = 8.0 # Precio mínimo absoluto para cualquier SKU
P_MAX_ABS_DEFAULT = 60.0 # Precio máximo absoluto para cualquier SKU
UNITS_MIN_POLICY_DEFAULT = "min_abs" # Opciones: "min_abs", "min_factor_hist"
UNITS_MIN_FACTOR_HIST_DEFAULT = 0.2 # Factor del histórico para política "min_factor_hist"
Q_MIN_ABS_DEFAULT = 5 # Cantidad mínima absoluta a vender por SKU

# --- CACHE DE RECURSOS ---
@st.cache_resource
def load_bundle_pkl(uploaded_file):
    """Carga el bundle de modelos desde un archivo .pkl subido."""
    if uploaded_file is not None:
        try:
            bundle = pickle.load(uploaded_file)
            st.success("Bundle de modelos cargado correctamente.")
            return bundle
        except Exception as e:
            st.error(f"Error al cargar el bundle de modelos: {e}")
            return None
    return None

@st.cache_data
def load_df_hist_parquet(uploaded_file):
    """Carga el DataFrame histórico desde un archivo .parquet subido."""
    if uploaded_file is not None:
        try:
            df_hist = pd.read_parquet(uploaded_file)
            # Asegurar que 'fecha' sea datetime y establecer como índice
            df_hist['fecha'] = pd.to_datetime(df_hist['fecha'])
            df_hist = df_hist.set_index('fecha')
            df_hist = df_hist.sort_index() # Asegurar orden cronológico
            st.success("DataFrame histórico cargado correctamente.")
            return df_hist
        except Exception as e:
            st.error(f"Error al cargar el histórico (Parquet): {e}")
            return None
    return None

# Función auxiliar para la predicción (del notebook)
class CustomModel(BaseEstimator, RegressorMixin):
    """
    Clase wrapper para modelos pre-entrenados que encapsula el predict.
    Esto es útil si los modelos no son clases estándar de sklearn
    o si necesitan un preprocesamiento específico antes de la predicción.
    """
    def __init__(self, model, feature_names=None, feature_order=None):
        self.model = model
        self.feature_names = feature_names # Nombres de las features esperadas por el modelo
        self.feature_order = feature_order # Orden específico de las features

    def fit(self, X, y=None):
        # No se entrena en esta clase, ya se asume un modelo entrenado.
        return self

    def predict(self, X):
        # Asegurar que X tiene las columnas correctas en el orden correcto
        if self.feature_order is not None:
            X = X[self.feature_order]
        elif self.feature_names is not None:
            X = X[self.feature_names] # Asegurar solo las features esperadas

        return self.model.predict(X)

def _predict_row_sku(model_by_sku, info_by_sku, row, sku_id):
    """
    Realiza la predicción para una fila/SKU específica utilizando el modelo y la información
    correspondiente a ese SKU.
    """
    if sku_id not in model_by_sku:
        # st.warning(f"No hay modelo para SKU: {sku_id}. Retornando 0.")
        return 0

    model_info = info_by_sku.get(sku_id, {})
    model = model_by_sku[sku_id]

    # Crear un DataFrame a partir de la fila para la predicción
    # Asegurar que el índice es numérico o None para evitar errores de reshape con series
    X_single_row = pd.DataFrame([row.values], columns=row.index)

    # Alinear features para el modelo
    expected_features = model_info.get('features')
    if expected_features:
        # Añadir features que puedan faltar con 0 (o NaN si el modelo los maneja)
        for feature in expected_features:
            if feature not in X_single_row.columns:
                X_single_row[feature] = 0 # O np.nan, dependiendo de cómo el modelo maneja faltantes
        X_single_row = X_single_row[expected_features]
    
    # Asegurar tipos de datos correctos si es necesario
    # (por ejemplo, categorías a dummies si el modelo lo espera y no se hizo antes)
    
    try:
        pred_qty = model.predict(X_single_row).item() # .item() para obtener el valor escalar
        return max(0, pred_qty) # Cantidad no puede ser negativa
    except Exception as e:
        # st.error(f"Error al predecir para SKU {sku_id}: {e}")
        return 0

def create_price_grid(p_min, p_max, step):
    """Genera una rejilla de precios."""
    return np.arange(p_min, p_max + step, step)

def parse_json_input(json_str):
    """Intenta parsear un string JSON."""
    if not json_str.strip():
        return {}
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        st.error(f"Error al parsear JSON: {e}. Asegúrate de que el formato es válido.")
        return None
