import pandas as pd
import numpy as np
import holidays
from datetime import timedelta

# --- HELPERS PARA FECHAS Y CALENDARIO ---
def _create_calendar_features(df_input):
    """
    Crea características de calendario para un DataFrame dado,
    asumiendo que tiene un índice de tipo DatetimeIndex.
    """
    df = df_input.copy()
    df['dayofweek'] = df.index.dayofweek
    df['dayofmonth'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['dayofyear'] = df.index.dayofyear
    df['quarter'] = df.index.quarter
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    return df

def _add_holidays(df_input, country='CL', state=None):
    """
    Agrega columnas binarias para feriados nacionales y el día previo/posterior.
    """
    df = df_input.copy()
    country_holidays = holidays.CountryHoliday(country=country, state=state, years=df.index.year.unique())
    
    df['is_holiday'] = df.index.map(lambda x: 1 if x in country_holidays else 0)
    df['day_before_holiday'] = df.index.map(lambda x: 1 if (x + timedelta(days=1)) in country_holidays else 0)
    df['day_after_holiday'] = df.index.map(lambda x: 1 if (x - timedelta(days=1)) in country_holidays else 0)
    
    return df

def _add_fourier_features(df_input, column, orders):
    """
    Agrega características de Fourier (seno/coseno) para una columna temporal (e.g., semana del año).
    """
    df = df_input.copy()
    for order in orders:
        df[f'sin_{column}_{order}'] = np.sin(2 * np.pi * order * df[column] / df[column].max())
        df[f'cos_{column}_{order}'] = np.cos(2 * np.pi * order * df[column] / df[column].max())
    return df

# --- CREACIÓN DEL ESQUELETO FUTURO ---
def generar_esqueleto_futuro(start_date, n_weeks_opt, df_hist_actual, all_sku_ids, 
                              promos_dict=None, eventos_dict=None,
                              price_min_abs=None, price_max_abs=None):
    """
    Genera un DataFrame esqueleto para el horizonte de optimización.
    Este esqueleto contendrá la estructura básica de fechas, SKUs,
    características de calendario, Fourier, promociones y eventos.
    También inyecta placeholders para lags y rollings.
    """
    
    # 1. Crear rango de fechas
    forecast_dates = pd.date_range(start=start_date, periods=n_weeks_opt * 7, freq='D')
    
    # 2. Crear DataFrame base con todas las combinaciones SKU-Fecha
    df_futuro = pd.DataFrame({'fecha': forecast_dates})
    df_futuro['key'] = 1 # Para merge con SKUs
    df_skus = pd.DataFrame({'sku_id': list(all_sku_ids)})
    df_skus['key'] = 1
    
    df_futuro = pd.merge(df_futuro, df_skus, on='key').drop('key', axis=1)
    df_futuro = df_futuro.set_index('fecha').sort_index()
    
    # 3. Añadir características de calendario
    df_futuro = _create_calendar_features(df_futuro)
    df_futuro = _add_holidays(df_futuro)
    
    # 4. Añadir características de Fourier (ej. para 'weekofyear')
    df_futuro = _add_fourier_features(df_futuro, 'weekofyear', orders=[1, 2, 3])
    
    # 5. Inicializar columnas de precios y cantidades (placeholders)
    # Estos serán llenados durante la optimización
    df_futuro['precio'] = np.nan 
    df_futuro['q_opt'] = np.nan
    df_futuro['promo_semana'] = 0 # Placeholder para promociones
    df_futuro['evento_semana'] = 0 # Placeholder para eventos
    
    # 6. Inyectar promociones y eventos si se proporcionan
    # promos_dict = {'sku_id': {fecha: valor_promo}}
    # eventos_dict = {fecha: valor_evento}
    if promos_dict:
        for sku, promo_dates in promos_dict.items():
            for date_str, value in promo_dates.items():
                date_dt = pd.to_datetime(date_str)
                if date_dt in df_futuro.index:
                    df_futuro.loc[(df_futuro.index == date_dt) & (df_futuro['sku_id'] == sku), 'promo_semana'] = value

    if eventos_dict:
        for date_str, value in eventos_dict.items():
            date_dt = pd.to_datetime(date_str)
            if date_dt in df_futuro.index:
                df_futuro.loc[df_futuro.index == date_dt, 'evento_semana'] = value
    
    # 7. Añadir columnas de lags y rollings como placeholders
    # Estas serán rellenadas por `crear_features_dinamicas`
    # Es importante que existan para evitar KeyErrors
    # Se puede inferir la lista de lags/rollings de un df_hist_actual si es necesario
    
    # Por ahora, agregamos placeholders para las features que suelen ser lags/rollings
    # Esto es una suposición, y lo ideal sería obtener la lista de features dinámicas
    # de los modelos entrenados. Para esta iteración, asumimos algunas comunes.
    
    # Lags de precio, q_opt, promo, evento
    for col in ['precio', 'q_opt', 'promo_semana', 'evento_semana']:
        for lag in [1, 7, 14, 21]: # Ejemplos de lags
            df_futuro[f'{col}_lag_{lag}'] = np.nan
            
    # Rollings de precio, q_opt
    for col in ['precio', 'q_opt']:
        for window in [7, 14, 28]: # Ejemplos de ventanas de rolling
            df_futuro[f'{col}_rolling_mean_{window}'] = np.nan
            df_futuro[f'{col}_rolling_std_{window}'] = np.nan # std si es relevante
            df_futuro[f'{col}_rolling_median_{window}'] = np.nan
            
    # Añadir precio_min_abs y precio_max_abs como columnas si son globales
    if price_min_abs is not None:
        df_futuro['precio_min_abs'] = price_min_abs
    if price_max_abs is not None:
        df_futuro['precio_max_abs'] = price_max_abs

    return df_futuro.reset_index().set_index(['fecha', 'sku_id']).sort_index()

# --- CREACIÓN DE FEATURES DINÁMICAS (Lags y Rollings) ---
def crear_features_dinamicas(df_full, all_sku_ids):
    """
    Calcula lags y rollings para 'precio', 'q_opt', 'promo_semana' y 'evento_semana'
    sobre un DataFrame que combina histórico y futuro.
    
    Args:
        df_full (pd.DataFrame): DataFrame combinado (histórico + futuro) con 'fecha' y 'sku_id' como índice.
                                Debe contener las columnas 'precio', 'q_opt', 'promo_semana', 'evento_semana'.
        all_sku_ids (list): Lista de todos los SKUs para asegurar el groupby correcto.

    Returns:
        pd.DataFrame: DataFrame con las características dinámicas calculadas.
    """
    df = df_full.copy()
    
    # Asegurarse de que el índice sea de tipo DatetimeIndex para las operaciones de tiempo
    if not isinstance(df.index, pd.MultiIndex) or not isinstance(df.index.get_level_values('fecha'), pd.DatetimeIndex):
        raise ValueError("df_full debe tener un MultiIndex con 'fecha' como primer nivel y DatetimeIndex.")

    df_processed = []

    # Se agrupa por SKU para calcular lags y rollings de forma independiente
    # Y se asegura que cada SKU tenga todas las fechas necesarias
    for sku in all_sku_ids:
        df_sku = df.loc[(slice(None), sku), :].droplevel('sku_id').sort_index() # Solo fechas
        
        # Calcular Lags
        for col in ['precio', 'q_opt', 'promo_semana', 'evento_semana']:
            if col in df_sku.columns:
                for lag in [1, 7, 14, 21]: # Lags de 1, 7, 14, 21 días
                    df_sku[f'{col}_lag_{lag}'] = df_sku[col].shift(lag)
        
        # Calcular Rollings (media, std, mediana)
        for col in ['precio', 'q_opt']:
            if col in df_sku.columns:
                for window in [7, 14, 28]: # Ventanas de 7, 14, 28 días
                    # Aplicar rolling y luego shift para evitar data leakage
                    df_sku[f'{col}_rolling_mean_{window}'] = df_sku[col].rolling(window=window, min_periods=1).mean().shift(1)
                    df_sku[f'{col}_rolling_std_{window}'] = df_sku[col].rolling(window=window, min_periods=1).std().shift(1)
                    df_sku[f'{col}_rolling_median_{window}'] = df_sku[col].rolling(window=window, min_periods=1).median().shift(1)

        # Volver a añadir el nivel 'sku_id'
        df_sku['sku_id'] = sku
        df_sku = df_sku.reset_index().set_index(['fecha', 'sku_id'])
        df_processed.append(df_sku)

    return pd.concat(df_processed).sort_index()


def generar_escenarios_semana(df_esqueleto_semana, current_week_prices, price_grid,
                              costo_unitario, margen_min_directo,
                              q_min_abs=0, units_min_policy="min_abs",
                              units_min_factor_hist=0.0):
    """
    Genera un DataFrame con todos los escenarios posibles de precio para cada SKU
    para la semana actual, incluyendo cálculo de margen.
    
    Args:
        df_esqueleto_semana (pd.DataFrame): Esqueleto del DataFrame para la semana actual,
                                             con características de calendario, Fourier, etc.
                                             Debe tener 'fecha' y 'sku_id' como índice.
        current_week_prices (dict): Precios actuales de los SKUs (podrían ser los de la semana anterior).
        price_grid (np.array): Array de precios posibles a probar.
        costo_unitario (float): Costo unitario por SKU. Se asume que es el mismo para todos por ahora.
        margen_min_directo (float): Margen directo mínimo como porcentaje (0 a 1).
        q_min_abs (int): Cantidad mínima absoluta a vender por SKU.
        units_min_policy (str): Política para unidades mínimas ('min_abs', 'min_factor_hist').
        units_min_factor_hist (float): Factor para 'min_factor_hist' si se usa.

    Returns:
        pd.DataFrame: DataFrame de escenarios con 'precio' y 'margen_unitario' calculados.
    """
    
    # Flatten df_esqueleto_semana para facilitar la combinación con la rejilla de precios
    df_temp = df_esqueleto_semana.reset_index()
    
    all_scenarios = []
    
    for sku_id in df_temp['sku_id'].unique():
        df_sku_base = df_temp[df_temp['sku_id'] == sku_id].copy()
        
        # Crear un DataFrame para la rejilla de precios de este SKU
        price_scenarios = pd.DataFrame({'precio': price_grid})
        
        # Combinar el DataFrame base del SKU con la rejilla de precios
        # Esto crea una fila por cada combinación de fecha, SKU y precio en la rejilla
        df_sku_scenarios = pd.merge(df_sku_base, price_scenarios, how='cross')
        
        # Calcular el margen unitario para cada escenario de precio
        df_sku_scenarios['margen_unitario'] = df_sku_scenarios['precio'] - costo_unitario
        
        # Aplicar el filtro de margen mínimo directo
        df_sku_scenarios = df_sku_scenarios[df_sku_scenarios['margen_unitario'] >= (costo_unitario * margen_min_directo)]
        
        # Aplicar límites de precio si existen en las features del esqueleto
        # Se asume que price_min_abs y price_max_abs son columnas si se inyectaron
        if 'precio_min_abs' in df_sku_scenarios.columns and not df_sku_scenarios['precio_min_abs'].isnull().all():
            min_p_val = df_sku_scenarios['precio_min_abs'].iloc[0] # Asumiendo que es constante por sku/semana
            df_sku_scenarios = df_sku_scenarios[df_sku_scenarios['precio'] >= min_p_val]
        
        if 'precio_max_abs' in df_sku_scenarios.columns and not df_sku_scenarios['precio_max_abs'].isnull().all():
            max_p_val = df_sku_scenarios['precio_max_abs'].iloc[0]
            df_sku_scenarios = df_sku_scenarios[df_sku_scenarios['precio'] <= max_p_val]
        
        # Añadir al listado general de escenarios
        all_scenarios.append(df_sku_scenarios)
        
    if not all_scenarios:
        return pd.DataFrame() # Retorna DataFrame vacío si no hay escenarios
        
    df_scenarios = pd.concat(all_scenarios)
    
    # Reestablecer el índice si es necesario para mantener la coherencia
    df_scenarios = df_scenarios.set_index(['fecha', 'sku_id']).sort_index()
    
    return df_scenarios
