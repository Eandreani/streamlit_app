import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import altair as alt # Para visualizaciones

# Importar funciones desde los módulos locales
from utils import (
    load_bundle_pkl, load_df_hist_parquet, parse_json_input,
    N_WEEKS_OPT, PRICE_GRID_STEP, PRICE_MIN_DEFAULT, PRICE_MAX_DEFAULT,
    COSTO_UNITARIO_DEFAULT, MARGEN_MIN_DIRECTO_DEFAULT, P_MIN_ABS_DEFAULT,
    P_MAX_ABS_DEFAULT, UNITS_MIN_POLICY_DEFAULT, UNITS_MIN_FACTOR_HIST_DEFAULT,
    Q_MIN_ABS_DEFAULT
)
from iterative_optimizer import optimizar_precios_iterativo

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Optimizador de Precios Semanal")

st.title("💰 Optimizador de Precios Semanal")
st.markdown("""
Esta aplicación permite optimizar precios de SKUs semana a semana, 
simulando un proceso de decisión basado en modelos de demanda.
""")

# --- CARGA DE ARCHIVOS (BASE HISTÓRICA Y MODELOS) ---
st.header("Carga de Datos")
st.markdown("Por favor, sube los archivos `.parquet` del histórico y `.pkl` del bundle de modelos.")

col1_upload, col2_upload = st.columns(2)

with col1_upload:
    uploaded_df_hist = st.file_uploader("Subir DataFrame Histórico (.parquet)", type="parquet", key="df_hist_uploader")
    if uploaded_df_hist:
        df_hist = load_df_hist_parquet(uploaded_df_hist)
        st.session_state['df_hist'] = df_hist # Guardar en session_state
        st.success("Histórico cargado y guardado en memoria.")
    else:
        st.session_state['df_hist'] = None

with col2_upload:
    uploaded_bundle = st.file_uploader("Subir Bundle de Modelos (.pkl)", type="pkl", key="bundle_uploader")
    if uploaded_bundle:
        bundle = load_bundle_pkl(uploaded_bundle)
        if bundle:
            st.session_state['model_by_sku'] = bundle.get('model_by_sku')
            st.session_state['info_by_sku'] = bundle.get('info_by_sku')
            st.session_state['all_sku_ids'] = list(bundle.get('model_by_sku', {}).keys()) # Asumiendo que las keys son los SKUs
            st.success("Bundle de modelos cargado y guardado en memoria.")
    else:
        st.session_state['model_by_sku'] = None
        st.session_state['info_by_sku'] = None
        st.session_state['all_sku_ids'] = None

# --- VERIFICACIÓN DE DATOS CARGADOS ---
if st.session_state.get('df_hist') is None or st.session_state.get('model_by_sku') is None:
    st.warning("Por favor, sube ambos archivos (Histórico y Bundle de Modelos) para continuar.")
    st.stop() # Detener la ejecución si los datos no están cargados

df_hist = st.session_state['df_hist']
model_by_sku = st.session_state['model_by_sku']
info_by_sku = st.session_state['info_by_sku']
all_sku_ids = st.session_state['all_sku_ids']

st.success("Todos los datos requeridos han sido cargados exitosamente.")
st.markdown(f"**SKUs detectados:** {len(all_sku_ids)}")
st.markdown(f"**Fechas en histórico:** Desde {df_hist.index.min().strftime('%Y-%m-%d')} hasta {df_hist.index.max().strftime('%Y-%m-%d')}")

# --- PARÁMETROS DE OPTIMIZACIÓN ---
st.header("Parámetros de Optimización")

col_params1, col_params2, col_params3 = st.columns(3)

with col_params1:
    st.subheader("Horizonte y Fecha")
    default_start_date = df_hist.index.max() + timedelta(days=1) # Un día después del último día del histórico
    start_date = st.date_input("Fecha de inicio de optimización (primera semana)", 
                                value=default_start_date,
                                min_value=df_hist.index.max() + timedelta(days=1))
    n_weeks_opt = st.number_input("Número de semanas a optimizar", 
                                  min_value=1, max_value=52, value=N_WEEKS_OPT)

with col_params2:
    st.subheader("Rejilla de Precios")
    price_min = st.number_input("Precio Mínimo de Rejilla", value=PRICE_MIN_DEFAULT, step=0.1)
    price_max = st.number_input("Precio Máximo de Rejilla", value=PRICE_MAX_DEFAULT, step=0.1)
    price_step = st.number_input("Paso de Rejilla de Precios", value=PRICE_GRID_STEP, step=0.01)

with col_params3:
    st.subheader("Costos y Márgenes")
    costo_unitario = st.number_input("Costo Unitario (asumido igual para todos los SKUs)", value=COSTO_UNITARIO_DEFAULT, step=0.1)
    margen_min_directo = st.slider("Margen Directo Mínimo (%)", min_value=0, max_value=50, value=int(MARGEN_MIN_DIRECTO_DEFAULT * 100)) / 100.0
    p_min_abs = st.number_input("Precio Mínimo Absoluto por SKU", value=P_MIN_ABS_DEFAULT, step=0.1)
    p_max_abs = st.number_input("Precio Máximo Absoluto por SKU", value=P_MAX_ABS_DEFAULT, step=0.1)

st.subheader("Políticas de Cantidad Mínima")
col_policy1, col_policy2 = st.columns(2)
with col_policy1:
    q_min_abs = st.number_input("Cantidad Mínima Absoluta por SKU/Día (q_min_abs)", value=Q_MIN_ABS_DEFAULT, min_value=0)
with col_policy2:
    units_min_policy = st.selectbox("Política de Unidades Mínimas", options=["min_abs"], index=0, # "min_factor_hist"
                                    help="Por ahora solo 'min_abs' está implementada por simplificación.")
    units_min_factor_hist = st.number_input("Factor Histórico para Unidades Mínimas (si aplica)", value=UNITS_MIN_FACTOR_HIST_DEFAULT, min_value=0.0, max_value=1.0, step=0.01)


st.subheader("Promociones y Eventos (Formato JSON)")
st.info("Introduce las promociones y eventos en formato JSON. Si no hay, déjalo vacío. "
        "Las fechas deben ser YYYY-MM-DD. Los valores son numéricos.")

col_promo_event1, col_promo_event2 = st.columns(2)
with col_promo_event1:
    promo_json = st.text_area("Promociones (JSON)", value="{}", height=150, 
                              help='{"SKU001": {"2023-10-26": 1, "2023-10-27": 0.5}, "SKU002": {"2023-10-28": 1}}')
    promos_dict = parse_json_input(promo_json)
with col_promo_event2:
    event_json = st.text_area("Eventos (JSON)", value="{}", height=150,
                             help='{"2023-10-26": 1, "2023-11-05": 0.8}')
    eventos_dict = parse_json_input(event_json)

# --- BOTÓN DE EJECUCIÓN ---
st.markdown("---")
if st.button("🚀 Iniciar Optimización de Precios", type="primary"):
    if promos_dict is None or eventos_dict is None:
        st.error("Hay un error en el formato JSON de promociones o eventos. Por favor, corrígelo.")
    else:
        st.info("Iniciando el proceso de optimización. Esto puede tardar unos minutos...")
        
        # Convertir start_date a pd.Timestamp para consistencia
        start_date_ts = pd.Timestamp(start_date)

        try:
            df_resumen_opt, df_detalle_opt, optimization_logs = optimizar_precios_iterativo(
                start_date=start_date_ts,
                n_weeks_opt=n_weeks_opt,
                df_hist_inicial=df_hist,
                model_by_sku=model_by_sku,
                info_by_sku=info_by_sku,
                all_sku_ids=all_sku_ids,
                price_min=price_min,
                price_max=price_max,
                price_step=price_step,
                costo_unitario=costo_unitario,
                margen_min_directo=margen_min_directo,
                promos_dict=promos_dict,
                eventos_dict=eventos_dict,
                q_min_abs=q_min_abs,
                units_min_policy=units_min_policy,
                units_min_factor_hist=units_min_factor_hist,
                p_min_abs=p_min_abs,
                p_max_abs=p_max_abs
            )
            
            st.session_state['df_resumen_opt'] = df_resumen_opt
            st.session_state['df_detalle_opt'] = df_detalle_opt
            st.session_state['optimization_logs'] = optimization_logs
            st.success("✅ Optimización completada exitosamente!")

        except Exception as e:
            st.error(f"Se produjo un error durante la optimización: {e}")
            st.exception(e) # Mostrar el traceback completo para depuración

# --- VISUALIZACIÓN DE RESULTADOS ---
if 'df_resumen_opt' in st.session_state and not st.session_state['df_resumen_opt'].empty:
    st.header("Resultados de la Optimización")

    df_resumen_opt = st.session_state['df_resumen_opt']
    df_detalle_opt = st.session_state['df_detalle_opt']
    optimization_logs = st.session_state['optimization_logs']

    st.subheader("Resumen Semanal")
    st.dataframe(df_resumen_opt.style.format({
        'margen_total_opt_semana': '{:,.2f}',
    }), use_container_width=True)

    # Gráfico del margen total por semana
    chart_resumen = alt.Chart(df_resumen_opt.reset_index()).mark_line(point=True).encode(
        x=alt.X('fecha_inicio_semana:T', title='Fecha Inicio Semana'),
        y=alt.Y('margen_total_opt_semana:Q', title='Margen Total Optimizado'),
        tooltip=[alt.Tooltip('fecha_inicio_semana:T', title='Semana'), 
                 alt.Tooltip('margen_total_opt_semana:Q', title='Margen', format='$,.2f'),
                 'status_opt', 'solver_used']
    ).properties(
        title='Margen Total Optimizado por Semana'
    ).interactive()
    st.altair_chart(chart_resumen, use_container_width=True)
    
    st.subheader("Detalle por Día y SKU")

    # Selector de semana
    unique_dates = df_detalle_opt.index.get_level_values('fecha').unique().sort_values()
    week_selection = st.selectbox("Selecciona una fecha para ver el detalle diario/SKU:", 
                                  options=unique_dates.strftime('%Y-%m-%d').unique())
    
    if week_selection:
        selected_date = pd.to_datetime(week_selection)
        df_detalle_semana = df_detalle_opt.loc[selected_date].copy()
        
        st.markdown(f"**Detalle para el día {selected_date.strftime('%Y-%m-%d')}**")
        st.dataframe(df_detalle_semana.style.format({
            'precio_opt': '{:,.2f}',
            'q_opt': '{:,.0f}',
            'margen_total_opt': '{:,.2f}'
        }), use_container_width=True)

        # Gráfico de precios óptimos y cantidades por SKU para la fecha seleccionada
        chart_sku_prices_qty = alt.Chart(df_detalle_semana.reset_index()).mark_bar().encode(
            x=alt.X('sku_id:N', title='SKU ID'),
            y=alt.Y('q_opt:Q', title='Cantidad Óptima'),
            color=alt.Color('sku_id:N', legend=None),
            tooltip=['sku_id', 'precio_opt', 'q_opt', 'margen_total_opt']
        ).properties(
            title=f'Cantidad Óptima y Precio por SKU para {selected_date.strftime("%Y-%m-%d")}'
        )
        
        # Añadir texto de precio óptimo
        text_prices = alt.Chart(df_detalle_semana.reset_index()).mark_text(align='center', baseline='bottom', dy=-5).encode(
            x='sku_id:N',
            y='q_opt:Q',
            text=alt.Text('precio_opt:Q', format='.2f')
        )
        
        st.altair_chart(chart_sku_prices_qty + text_prices, use_container_width=True)
        
        # Mostrar el log del solver para la semana de la fecha seleccionada
        week_start_of_selected_date = selected_date - timedelta(days=selected_date.dayofweek) # Asumiendo inicio de semana Lunes
        log_key = week_start_of_selected_date.strftime('%Y-%m-%d')
        if log_key in optimization_logs:
            st.subheader(f"Log del Solver para la semana del {log_key}")
            log_data = optimization_logs[log_key]
            st.json(log_data)
        else:
            st.info(f"No hay log de optimización para la semana que contiene {selected_date.strftime('%Y-%m-%d')}.")
