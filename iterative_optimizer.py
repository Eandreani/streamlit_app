import pandas as pd
import streamlit as st
from datetime import timedelta
from feature_engineering import generar_esqueleto_futuro, crear_features_dinamicas, generar_escenarios_semana
from optimization_core import optimizar_semana
from utils import create_price_grid

def optimizar_precios_iterativo(start_date, n_weeks_opt, df_hist_inicial, model_by_sku, info_by_sku,
                                all_sku_ids, price_min, price_max, price_step,
                                costo_unitario, margen_min_directo,
                                promos_dict, eventos_dict,
                                q_min_abs, units_min_policy, units_min_factor_hist,
                                p_min_abs, p_max_abs):
    """
    Función principal que orquesta la optimización de precios de forma iterativa
    semana a semana para el horizonte de N_WEEKS_OPT.

    Args:
        start_date (pd.Timestamp): Fecha de inicio de la primera semana a optimizar.
        n_weeks_opt (int): Número de semanas a optimizar.
        df_hist_inicial (pd.DataFrame): DataFrame histórico inicial con 'fecha' y 'sku_id' como índice.
                                        Contiene 'precio', 'q_opt', 'promo_semana', 'evento_semana' históricos.
        model_by_sku (dict): Diccionario de modelos de demanda, uno por SKU.
        info_by_sku (dict): Diccionario de información adicional para cada SKU.
        all_sku_ids (list): Lista de todos los SKUs considerados.
        price_min (float): Precio mínimo para la rejilla de precios.
        price_max (float): Precio máximo para la rejilla de precios.
        price_step (float): Paso de la rejilla de precios.
        costo_unitario (float): Costo unitario para el cálculo del margen.
        margen_min_directo (float): Margen directo mínimo como porcentaje (0 a 1).
        promos_dict (dict): Diccionario de promociones para inyectar en el futuro.
        eventos_dict (dict): Diccionario de eventos para inyectar en el futuro.
        q_min_abs (int): Cantidad mínima absoluta a vender por SKU.
        units_min_policy (str): Política para unidades mínimas ('min_abs', 'min_factor_hist').
        units_min_factor_hist (float): Factor del histórico para política 'min_factor_hist'.
        p_min_abs (float): Precio mínimo absoluto.
        p_max_abs (float): Precio máximo absoluto.

    Returns:
        tuple: (pd.DataFrame, pd.DataFrame, dict)
            - df_resumen_opt: Resumen semanal de la optimización (margen total, status).
            - df_detalle_opt: Detalle diario/SKU de la optimización (precio_opt, q_opt, margen_total_opt).
            - optimization_logs: Log de los detalles de optimización por semana.
    """
    
    # Asegurarse de que df_hist_inicial tiene el formato correcto (MultiIndex)
    if not isinstance(df_hist_inicial.index, pd.MultiIndex) or \
       'fecha' not in df_hist_inicial.index.names or \
       'sku_id' not in df_hist_inicial.index.names:
        st.error("df_hist_inicial debe tener un MultiIndex con 'fecha' y 'sku_id'.")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # Asegurarse de que df_hist_inicial tiene las columnas esperadas para lags/rollings
    required_cols = ['precio', 'q_opt', 'promo_semana', 'evento_semana']
    for col in required_cols:
        if col not in df_hist_inicial.columns:
            df_hist_inicial[col] = np.nan # Añadir con NaN si falta
            st.warning(f"Columna '{col}' no encontrada en df_hist_inicial. Añadiendo con NaN.")

    df_hist_actual = df_hist_inicial.copy()
    
    df_detalle_opt_all_weeks = []
    df_resumen_opt_all_weeks = []
    optimization_logs = {}

    price_grid = create_price_grid(price_min, price_max, price_step)

    progress_text = "Optimizando semana a semana..."
    my_bar = st.progress(0, text=progress_text)

    for i in range(n_weeks_opt):
        my_bar.progress((i + 1) / n_weeks_opt, text=f"{progress_text} (Semana {i+1}/{n_weeks_opt})")

        current_week_start_date = start_date + timedelta(weeks=i)
        
        st.write(f"--- Optimizando semana que comienza el: {current_week_start_date.strftime('%Y-%m-%d')} ---")

        # 1. Generar esqueleto de la semana futura
        # El esqueleto solo se genera para 7 días
        df_esqueleto_futuro_semana = generar_esqueleto_futuro(
            start_date=current_week_start_date,
            n_weeks_opt=1, # Solo una semana a la vez
            df_hist_actual=df_hist_actual, # Se pasa para info de SKUs, aunque no se usa directamente para el esqueleto
            all_sku_ids=all_sku_ids,
            promos_dict=promos_dict,
            eventos_dict=eventos_dict,
            price_min_abs=p_min_abs,
            price_max_abs=p_max_abs
        )

        # 2. Combinar histórico + esqueleto para calcular features dinámicas
        # Esto asegura que los lags/rollings se calculen correctamente usando
        # datos históricos Y los datos ya optimizados de semanas previas.
        
        # Primero, asegurar que el esqueleto tenga las columnas 'precio', 'q_opt', 'promo_semana', 'evento_semana'
        # que serán llenadas por los valores optimizados
        for col in ['precio', 'q_opt', 'promo_semana', 'evento_semana']:
            if col not in df_esqueleto_futuro_semana.columns:
                df_esqueleto_futuro_semana[col] = np.nan # Inicialmente nulo
        
        df_temp_full = pd.concat([df_hist_actual, df_esqueleto_futuro_semana.drop(columns=['q_opt'])], axis=0, ignore_index=False)
        df_temp_full = df_temp_full[~df_temp_full.index.duplicated(keep='last')] # Quitar posibles duplicados por el concat

        df_features_computed = crear_features_dinamicas(df_temp_full, all_sku_ids)
        
        # Filtrar solo la semana actual para la optimización
        week_end_date = current_week_start_date + timedelta(days=6)
        df_semana_para_opt = df_features_computed.loc[current_week_start_date:week_end_date]
        
        # 3. Generar escenarios para la semana actual
        df_escenarios_semana = generar_escenarios_semana(
            df_semana_para_opt,
            current_week_prices={}, # No se usa directamente aquí, los precios se inyectan en el esqueleto
            price_grid=price_grid,
            costo_unitario=costo_unitario,
            margen_min_directo=margen_min_directo,
            q_min_abs=q_min_abs,
            units_min_policy=units_min_policy,
            units_min_factor_hist=units_min_factor_hist
        )
        
        # 4. Optimizar la semana actual
        df_opt_semana, detalle_log_semana = optimizar_semana(
            df_escenarios_semana,
            model_by_sku,
            info_by_sku,
            all_sku_ids,
            costo_unitario,
            margen_min_directo,
            q_min_abs,
            units_min_policy,
            units_min_factor_hist
        )

        optimization_logs[current_week_start_date.strftime('%Y-%m-%d')] = detalle_log_semana

        if df_opt_semana.empty:
            st.warning(f"No se pudieron optimizar los precios para la semana que comienza el {current_week_start_date.strftime('%Y-%m-%d')}.")
            # Si no hay optimización, podríamos rellenar con NaNs o mantener los valores del esqueleto
            # Para fines de demostración, simplemente agregamos un DataFrame vacío.
            continue
        
        df_detalle_opt_all_weeks.append(df_opt_semana)

        # 5. Actualizar df_hist_actual con los resultados de la optimización para la próxima iteración
        # Se necesita un DataFrame que contenga todas las columnas necesarias para `crear_features_dinamicas`
        # ['precio', 'q_opt', 'promo_semana', 'evento_semana']
        
        # Crear un DF con los resultados optimizados en el formato esperado
        optimized_data_for_hist = df_opt_semana.copy()
        optimized_data_for_hist['precio'] = optimized_data_for_hist['precio_opt']
        optimized_data_for_hist['q_opt'] = optimized_data_for_hist['q_opt']
        
        # Inyectar promo_semana y evento_semana desde el df_semana_para_opt (el esqueleto con features)
        # Asegurarse de que estas columnas estén en df_opt_semana antes de merge
        cols_to_merge_from_esqueleto = ['promo_semana', 'evento_semana']
        for col in cols_to_merge_from_esqueleto:
            if col in df_semana_para_opt.columns:
                optimized_data_for_hist[col] = df_semana_para_opt[col].loc[optimized_data_for_hist.index].fillna(0)
            else:
                optimized_data_for_hist[col] = 0 # Default a 0 si no existe
        
        # Solo mantener las columnas relevantes para el histórico
        optimized_data_for_hist = optimized_data_for_hist[['precio', 'q_opt', 'promo_semana', 'evento_semana']]
        
        # Añadir al histórico actual, reemplazando las entradas si ya existen
        # Esto simula un flujo continuo donde el histórico se va actualizando
        df_hist_actual = pd.concat([df_hist_actual, optimized_data_for_hist], axis=0)
        df_hist_actual = df_hist_actual[~df_hist_actual.index.duplicated(keep='last')]
        df_hist_actual = df_hist_actual.sort_index()

        # Resumen semanal
        resumen_semana = {
            'fecha_inicio_semana': current_week_start_date,
            'margen_total_opt_semana': detalle_log_semana['total_margen_semana'],
            'status_opt': detalle_log_semana['status'],
            'solver_used': detalle_log_semana['solver_used']
        }
        df_resumen_opt_all_weeks.append(resumen_semana)
        
    my_bar.empty() # Borrar la barra de progreso

    df_resumen_opt = pd.DataFrame(df_resumen_opt_all_weeks).set_index('fecha_inicio_semana')
    
    if df_detalle_opt_all_weeks:
        df_detalle_opt = pd.concat(df_detalle_opt_all_weeks)
    else:
        df_detalle_opt = pd.DataFrame()

    return df_resumen_opt, df_detalle_opt, optimization_logs
