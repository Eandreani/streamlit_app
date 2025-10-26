import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value, PULP_CBC_CMD, LpStatus
import streamlit as st # Importa Streamlit para usar sus funciones como st.warning
from .utils import _predict_row_sku, CustomModel # Importa desde el mismo paquete

# --- OPTIMIZACIÓN DE UNA SEMANA ---
def optimizar_semana(df_escenarios_semana, model_by_sku, info_by_sku, all_sku_ids, 
                     costo_unitario, margen_min_directo,
                     q_min_abs=5, units_min_policy="min_abs", units_min_factor_hist=0.2):
    """
    Optimiza los precios para una semana específica utilizando los escenarios precalculados
    y los modelos de predicción de demanda.

    Args:
        df_escenarios_semana (pd.DataFrame): DataFrame de escenarios para la semana actual,
                                             con 'fecha', 'sku_id' como índice, 'precio',
                                             'margen_unitario' y todas las features.
        model_by_sku (dict): Diccionario de modelos de demanda, uno por SKU.
        info_by_sku (dict): Diccionario de información adicional para cada SKU (ej. features esperadas).
        all_sku_ids (list): Lista de todos los SKUs considerados.
        costo_unitario (float): Costo unitario para el cálculo del margen.
        margen_min_directo (float): Margen directo mínimo como porcentaje (0 a 1).
        q_min_abs (int): Cantidad mínima absoluta a vender por SKU.
        units_min_policy (str): Política para unidades mínimas ('min_abs', 'min_factor_hist').
        units_min_factor_hist (float): Factor del histórico para política 'min_factor_hist'.

    Returns:
        tuple: (pd.DataFrame, dict) - DataFrame con resultados óptimos (precio, q_opt, margen total)
                                      y un diccionario con el detalle de la optimización.
    """
    
    if df_escenarios_semana.empty:
        return pd.DataFrame(), {"status": "No scenarios to optimize."}

    # Asumiendo que todas las fechas son la misma para esta semana
    current_date = df_escenarios_semana.index.get_level_values('fecha').min()
    week_start_date_str = current_date.strftime('%Y-%m-%d')
    
    # Predecir cantidad para cada escenario
    # Agrupamos por sku_id para predecir eficientemente
    df_escenarios_semana['q_pred'] = df_escenarios_semana.apply(
        lambda row: _predict_row_sku(model_by_sku, info_by_sku, row, row.name[1]), axis=1
    )
    
    # Calcular el margen total esperado para cada escenario
    df_escenarios_semana['margen_total_pred'] = df_escenarios_semana['q_pred'] * df_escenarios_semana['margen_unitario']

    # --- Configurar el problema de optimización con Pulp ---
    prob = LpProblem(f"Optimización_Semana_{week_start_date_str}", LpMaximize)

    # Variables de decisión: x_s_p = 1 si se elige el precio 'p' para el SKU 's'
    # Creamos un índice único para cada escenario (fecha, sku_id, precio)
    escenario_keys = df_escenarios_semana.index.to_frame(index=False).apply(
        lambda r: (r['fecha'], r['sku_id'], r['precio']), axis=1
    )
    
    # Aseguramos que solo haya variables para escenarios con q_pred > 0 y margen_total_pred > 0
    # Esto reduce el espacio de búsqueda y evita escenarios inviables
    df_valid_scenarios = df_escenarios_semana[
        (df_escenarios_semana['q_pred'] > 0) & 
        (df_escenarios_semana['margen_total_pred'] > 0)
    ].copy()
    
    if df_valid_scenarios.empty:
        st.warning(f"No hay escenarios válidos (q_pred > 0 y margen_total_pred > 0) para la semana que empieza el {week_start_date_str}. Retornando resultados vacíos.")
        return pd.DataFrame(), {"status": "No valid scenarios for optimization."}

    x_vars = LpVariable.dicts("x", 
                              df_valid_scenarios.index, 
                              0, 1, cat='Binary')

    # Objetivo: Maximizar el margen total predicho para todos los SKUs
    prob += lpSum([df_valid_scenarios.loc[idx, 'margen_total_pred'] * x_vars[idx] 
                   for idx in df_valid_scenarios.index]), "Maximizar Margen Total Semanal"

    # Restricciones:
    # 1. Para cada SKU y cada día de la semana, solo se puede elegir un precio.
    for (fecha, sku_id), group in df_valid_scenarios.groupby(level=['fecha', 'sku_id']):
        prob += lpSum([x_vars[idx] for idx in group.index]) == 1, f"One_Price_Per_SKU_Day_{fecha.strftime('%Y-%m-%d')}_{sku_id}"

    # 2. Restricción de cantidad mínima por SKU (si aplica)
    # df_hist_actual_sku_avg = df_hist_actual.groupby('sku_id')['q_opt'].mean() if df_hist_actual is not None else pd.Series()
    
    # Asumo que la política de unidades mínimas se aplica a las unidades predichas
    # y puede variar por SKU.
    for sku_id in all_sku_ids:
        # Aquí puedes buscar el histórico de ventas para este SKU, si la política es "min_factor_hist"
        # Por simplicidad ahora, lo haremos a nivel de día, pero podría ser semanal
        for fecha_dia in df_valid_scenarios.index.get_level_values('fecha').unique():
            sku_day_scenarios = df_valid_scenarios.loc[(fecha_dia, sku_id), :]
            if not sku_day_scenarios.empty:
                
                min_units = q_min_abs # Default a la cantidad mínima absoluta
                # if units_min_policy == "min_factor_hist" and sku_id in df_hist_actual_sku_avg:
                #    min_units = max(q_min_abs, int(df_hist_actual_sku_avg[sku_id] * units_min_factor_hist))

                # La restricción es que la suma de q_pred * x_var para un SKU en un día
                # debe ser al menos min_units si se selecciona el SKU para ese día.
                # Esto es más complejo en MILP, una forma simple es asegurar que si se elige un precio,
                # la q_pred asociada debe ser >= min_units. Esto ya está implícito en la filtración de escenarios
                # donde q_pred > 0 y la variable solo se elige si hay demanda.
                
                # Una restricción más directa sería asegurar que el q_opt total de un SKU en el día es al menos min_units
                # Esta es una aproximación, puede ser refinada
                prob += lpSum([df_valid_scenarios.loc[idx, 'q_pred'] * x_vars[idx] 
                               for idx in sku_day_scenarios.index]) >= min_units, \
                        f"Min_Units_Per_SKU_Day_{fecha_dia.strftime('%Y-%m-%d')}_{sku_id}"

    # Resolver el problema
    solver_used = "Pulp (CBC)"
    try:
        prob.solve(PULP_CBC_CMD(msg=0)) # msg=0 para no imprimir salida del solver
    except Exception as e:
        st.warning(f"Error al usar el solver Pulp ({e}). Recurriendo a estrategia argmax.")
        prob.status = LpStatus.Undefined # Marcar como no resuelto por Pulp
        solver_used = "Argmax Fallback"

    resultados_opt = []
    if prob.status == LpStatus.Optimal:
        # Extraer resultados si la optimización fue exitosa
        for (fecha, sku_id, precio) in df_valid_scenarios.index:
            if x_vars[(fecha, sku_id, precio)].varValue == 1:
                row_opt = df_valid_scenarios.loc[(fecha, sku_id, precio)].copy()
                row_opt['precio_opt'] = precio
                row_opt['q_opt'] = row_opt['q_pred']
                row_opt['margen_total_opt'] = row_opt['margen_total_pred']
                resultados_opt.append(row_opt)
    else:
        # Fallback a argmax simple: para cada SKU, elige el escenario con mayor margen_total_pred
        # Esto ocurre si Pulp no encuentra una solución óptima o si falló el solver.
        st.warning(f"La optimización con Pulp no alcanzó el estado óptimo ({LpStatus[prob.status]}). Recurriendo a estrategia argmax para la semana {week_start_date_str}.")
        solver_used = "Argmax Fallback"
        
        # Agrupar por fecha y sku_id, y para cada grupo, seleccionar la fila con el margen_total_pred más alto
        for (fecha, sku_id), group in df_valid_scenarios.groupby(level=['fecha', 'sku_id']):
            if not group.empty:
                best_scenario_idx = group['margen_total_pred'].idxmax()
                row_opt = group.loc[best_scenario_idx].copy()
                row_opt['precio_opt'] = row_opt.name[2] # El precio está en el multiindex
                row_opt['q_opt'] = row_opt['q_pred']
                row_opt['margen_total_opt'] = row_opt['margen_total_pred']
                resultados_opt.append(row_opt)
    
    df_opt = pd.DataFrame(resultados_opt)
    if not df_opt.empty:
        df_opt = df_opt.reset_index().set_index(['fecha', 'sku_id']).sort_index()
    
    detalle_opt = {
        "status": LpStatus[prob.status] if solver_used != "Argmax Fallback" else "Fallback to Argmax",
        "solver_used": solver_used,
        "total_margen_semana": df_opt['margen_total_opt'].sum() if not df_opt.empty else 0
    }

    return df_opt[['precio_opt', 'q_opt', 'margen_total_opt']], detalle_opt
