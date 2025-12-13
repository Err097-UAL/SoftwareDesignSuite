import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

# ==============================================================================
# BLOQUE 1: MOTOR DE FÍSICA Y CÁLCULO (Integración de todos los scripts .m)
# ==============================================================================

# --- A) PROPIEDADES ELÉCTRICAS Y TÉRMICAS (E-Series) ---
def calc_resistance_dc(R20, alpha, T, T_ref=20):
    """ E3_ResistanceTemperatureCorrection.m """
    return R20 * (1 + alpha * (T - T_ref))

def calc_skin_proximity_factors(freq, diameter_mm, sigma, geometry_factor=1.0):
    """ E3b_AC_Resistance_Factor.m """
    radius_m = (diameter_mm / 1000) / 2
    area_m2 = np.pi * (radius_m**2)
    sigma_Sm = sigma * 1e6 
    
    if sigma_Sm <= 0: return 0, 0
    R_dc = 1 / (sigma_Sm * area_m2)
    
    x_s_sq = (8 * np.pi * freq / R_dc) * 1e-7 if R_dc > 0 else 0
    k_skin = (x_s_sq**2) / (192 + 0.8 * (x_s_sq**2))
    k_prox = k_skin * geometry_factor * 0.5 
    return k_skin, k_prox

def calc_heat_dissipation_advanced(T_surf, env_params, r_outer_m):
    """ E7_HeatDissipation.m (Soporta escenarios) """
    T_amb = env_params.get('T_amb', 25)
    scenario = env_params.get('scenario', 'overhead')
    delta_T = T_surf - T_amb
    if delta_T <= 0: return 0
    
    D_m = r_outer_m * 2

    if scenario == 'overhead': # IEEE 738
        wind = env_params.get('wind', 0.6)
        emissivity = env_params.get('emissivity', 0.5)
        solar = env_params.get('solar', 0)
        sigma_sb = 5.67e-8
        
        q_rad = sigma_sb * emissivity * np.pi * D_m * ((T_surf + 273.15)**4 - (T_amb + 273.15)**4)
        
        if wind < 0.1:
            h_conv = 3.0 * (delta_T / D_m)**0.25 
        else:
            k_air = 0.026
            Re = (wind * D_m) / 1.5e-5
            Nu = 0.65 * Re**0.2 + 0.23 * Re**0.61
            h_conv = (Nu * k_air) / D_m
            
        q_conv = h_conv * np.pi * D_m * delta_T
        return max(0, q_conv + q_rad - solar)

    elif scenario == 'underground': # Neher-McGrath simplificado
        rho_soil = env_params.get('rho_soil', 1.0)
        R_ext = rho_soil * np.log(4.0 / D_m) / (2 * np.pi)
        return delta_T / R_ext

    else: # Interior
        h_conv = 2.5 * (delta_T / D_m)**0.25 
        q_conv = h_conv * np.pi * D_m * delta_T
        return q_conv

def solve_thermal_equilibrium(Current, R20, alpha, T_amb, diameter_mm, k_ac_total, env_params):
    """ E4_ThermalEquilibrium.m """
    r_outer_m = (diameter_mm / 1000) / 2
    def heat_balance(T):
        R_t = calc_resistance_dc(R20, alpha, T) * (1 + k_ac_total)
        Q_gen = (Current**2) * R_t 
        Q_diss = calc_heat_dissipation_advanced(T, env_params, r_outer_m)
        return Q_gen - Q_diss

    try:
        T_eq = brentq(heat_balance, T_amb, 300)
    except:
        T_eq = 300 
    return T_eq

def solve_transient_heating(I, time_span_sec, env_params, R20, alpha, T_ref, r_outer_m, mass_per_m, cp_mat):
    """ E6_TransientHeating.m """
    def thermal_ode(t, T):
        T_val = T[0]
        R_t = calc_resistance_dc(R20, alpha, T_val, T_ref)
        P_gen = (I**2) * R_t
        P_diss = calc_heat_dissipation_advanced(T_val, env_params, r_outer_m)
        dTdt = (P_gen - P_diss) / (mass_per_m * cp_mat)
        return dTdt

    t_eval = np.linspace(0, time_span_sec, 100)
    sol = solve_ivp(thermal_ode, [0, time_span_sec], [env_params['T_amb']], t_eval=t_eval)
    return sol.t, sol.y[0]

# --- B) CAÍDA DE TENSIÓN (F-Series) ---
def calc_voltage_drop(method, V_line, L_m, P_kw, cos_phi, R_ohm_m, X_ohm_m=0):
    """ F2_calculateExactVoltageDrop.m y F3 """
    I = (P_kw * 1000) / (np.sqrt(3) * V_line * cos_phi)
    sin_phi = np.sqrt(1 - cos_phi**2)
    
    if method == "Simplificada (Resistiva)":
        dU = np.sqrt(3) * L_m * I * R_ohm_m * cos_phi
    else:
        dU = np.sqrt(3) * L_m * I * (R_ohm_m * cos_phi + X_ohm_m * sin_phi)
    return dU, I

def check_rebt_compliance(dU, V_source, circuit_type):
    """ F4_checkREBTCompliance.m """
    pct = (dU / V_source) * 100
    if circuit_type == "LGA (Línea General)": limit = 0.5 # Aprox LGA contadores centralizados
    elif circuit_type == "Derivación Individual": limit = 1.0 # Aprox
    elif circuit_type == "Alumbrado": limit = 4.5
    else: limit = 6.5 # Fuerza Motriz / Otros
    return pct <= limit, pct, limit

# --- C) DIMENSIONAMIENTO ECONÓMICO (G-Series) ---

def get_standard_cables_db(material):
    """ G3_selectStandardSection.m (Base de datos) """
    # Formato: [Sección(mm2), Ampacidad(A), Coste(€/m)]
    # Valores de coste y ampacidad aproximados para la demo
    if material == "Cobre":
        # Basado en CPR o XLPE típico
        data = [
            (1.5, 24, 0.6), (2.5, 32, 1.0), (4, 42, 1.5), (6, 54, 2.2),
            (10, 75, 3.8), (16, 100, 6.0), (25, 127, 9.5), (35, 158, 13.0),
            (50, 192, 18.0), (70, 246, 26.0), (95, 298, 35.0), (120, 346, 45.0),
            (150, 399, 56.0), (185, 456, 70.0), (240, 538, 92.0)
        ]
    else: # Aluminio
        data = [
            (16, 75, 2.5), (25, 98, 3.8), (35, 120, 5.0), (50, 145, 6.8),
            (70, 185, 9.5), (95, 225, 12.0), (120, 262, 15.0), (150, 300, 19.0),
            (185, 345, 24.0), (240, 410, 32.0)
        ]
    return pd.DataFrame(data, columns=["Section", "Ampacity", "Cost_per_m"])

def calc_lifecycle_cost(section_row, L_m, I_load, hours_year, years, energy_cost, sigma, line_type="Trifásica"):
    """ G4_calculateLifecycleCost.m """
    s = section_row["Section"]
    c_cable = section_row["Cost_per_m"]
    
    # 1. CAPEX (Inversión)
    num_conductors = 3 if line_type == "Trifásica" else 2
    capex = L_m * c_cable * num_conductors
    
    # 2. OPEX (Pérdidas Joule)
    # R = L / (sigma * S) -> sigma en m/(Ohm*mm2)
    R_total = L_m / (sigma * s)
    # P_loss = N * I^2 * R
    P_loss_kW = (num_conductors * (I_load**2) * R_total) / 1000
    
    total_energy_kWh = P_loss_kW * hours_year * years
    opex = total_energy_kWh * energy_cost
    
    return capex, opex, capex + opex


# ==============================================================================
# BLOQUE 2: INTERFACES GRÁFICAS (TABS)
# ==============================================================================

def render_thermal_tab():
    """ 2a) Ampacidad (Basado en E-Series) """
    st.subheader("Simulador Térmico y Ampacidad (IEC 60287 / IEEE 738)")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        mat_type = st.selectbox("Material Conductor", ["Cobre", "Aluminio"], key="thm_mat")
        sigma = 56.0 if mat_type == "Cobre" else 35.0
        alpha = 0.00393 if mat_type == "Cobre" else 0.00403
        section_mm2 = st.selectbox("Sección (mm²)", [10, 16, 25, 35, 50, 70, 95, 120, 150, 185, 240], index=6, key="thm_sec")
        diameter_mm = 2 * np.sqrt(section_mm2 / np.pi) # Aprox compacto

    with c2:
        ins_type = st.selectbox("Aislamiento", ["XLPE (90°C)", "PVC (70°C)", "Desnudo (80°C)"], key="thm_ins")
        T_max = 90.0 if "XLPE" in ins_type else (70.0 if "PVC" in ins_type else 80.0)

    with c3:
        T_amb = st.slider("Temp. Ambiente (°C)", 0, 50, 25, key="thm_tamb")
        wind = st.slider("Viento (m/s)", 0.0, 5.0, 0.6, key="thm_wind")

    # Cálculos
    R_20_per_m = 1 / (sigma * section_mm2)
    k_skin, k_prox = calc_skin_proximity_factors(50, diameter_mm, sigma) # 50Hz
    env = {'scenario': 'overhead', 'T_amb': T_amb, 'wind': wind, 'emissivity': 0.5}

    # Generación de Curvas
    i_limit_sim = section_mm2 * 6
    currents = np.linspace(0, i_limit_sim, 50)
    temps = [solve_thermal_equilibrium(I, R_20_per_m, alpha, T_amb, diameter_mm, k_skin+k_prox, env) for I in currents]

    # Visualización
    col_gr, col_res = st.columns([2,1])
    
    with col_gr:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=currents, y=temps, name='Temp. Equilibrio', line=dict(color='red', width=3)))
        fig.add_hline(y=T_max, line_dash="dash", line_color="orange", annotation_text=f"Límite {ins_type}")
        fig.update_layout(title="Curva de Calentamiento Estacionario", xaxis_title="Corriente (A)", yaxis_title="Temperatura (°C)")
        st.plotly_chart(fig, use_container_width=True)

    with col_res:
        st.info("Resultados")
        try:
            ampacity = np.interp(T_max, temps, currents)
            st.metric("Ampacidad Máxima", f"{ampacity:.1f} A")
        except:
            st.error("Error en convergencia")
        
        st.write(f"R_AC (20°C): `{(R_20_per_m*(1+k_skin+k_prox))*1000:.4f} Ω/km`")
        st.write(f"Efecto Piel: `{(k_skin+k_prox)*100:.2f}%` incremento")


def render_transient_tab():
    """ 2b) Transitorios (Basado en E6) """
    st.subheader("Análisis Térmico Transitorio (Tiempo Real)")
    st.caption("Evolución de la temperatura ante un escalón de carga.")
    
    c1, c2, c3 = st.columns(3)
    I_step = c1.number_input("Corriente del Salto (A)", value=250.0)
    t_min = c2.number_input("Duración Simulación (min)", value=60)
    T_ini = c3.number_input("Temp. Inicial Cable (°C)", value=25.0)

    if st.button("Simular Transitorio", key="btn_trans"):
        # Datos físicos (Hardcoded para demo, idealmente de DB)
        R20 = 0.0003; alpha = 0.00393; mass = 0.8; cp = 385; r_outer = 0.012
        env = {'scenario': 'overhead', 'T_amb': 25, 'wind': 0.6}
        
        t_sec, T_vals = solve_transient_heating(I_step, t_min*60, env, R20, alpha, 20, r_outer, mass, cp)
        
        fig = px.line(x=t_sec/60, y=T_vals, labels={'x':'Tiempo (min)', 'y':'Temperatura (°C)'}, 
                      title=f"Respuesta Transitoria ante {I_step}A")
        fig.add_hline(y=90, line_dash="dot", line_color="red", annotation_text="Límite 90°C")
        st.plotly_chart(fig, use_container_width=True)


def render_voltage_drop_tab():
    """ 2c) Caída de Tensión (Basado en F-Series) """
    st.subheader("Caída de Tensión (Blondel y Perfiles)")
    
    mode = st.radio("Modo", ["Tramo Simple", "Instalación Completa (Perfil F6)"], horizontal=True)
    
    V_nom = st.number_input("Tensión Nominal (V)", 400.0)
    
    if mode == "Tramo Simple":
        c1, c2, c3 = st.columns(3)
        L = c1.number_input("Longitud (m)", 150.0)
        P = c2.number_input("Potencia (kW)", 40.0)
        S = c3.selectbox("Sección Cu (mm²)", [10, 16, 25, 35, 50, 70, 95], index=3)
        
        # Cálculo
        R = (1/56) / S
        X = 0.00008 # Reactancia estimada
        dU, _ = calc_voltage_drop("Exacta", V_nom, L, P, 0.9, R, X)
        ok, pct, limit = check_rebt_compliance(dU, V_nom, "Fuerza Motriz")
        
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Caída Tensión", f"{dU:.2f} V", delta=f"-{pct:.2f}%")
        if ok: col_res2.success(f"Cumple REBT (<{limit}%)")
        else: col_res2.error(f"NO Cumple REBT (>{limit}%)")

    else:
        # F6: Perfil de Instalación
        st.info("Defina los tramos secuenciales (Ej: LGA -> DI -> Circuito Interior)")
        df_input = pd.DataFrame([
            {"Tramo": "LGA", "Longitud (m)": 50, "P (kW)": 100, "Sección (mm2)": 95},
            {"Tramo": "DI", "Longitud (m)": 20, "P (kW)": 10, "Sección (mm2)": 16},
            {"Tramo": "Interior", "Longitud (m)": 15, "P (kW)": 3, "Sección (mm2)": 2.5}
        ])
        df_edited = st.data_editor(df_input, num_rows="dynamic")
        
        if st.button("Generar Perfil de Tensión (F6)"):
            dist_acc = 0
            V_current = V_nom
            x_points = [0]
            y_points = [V_nom]
            names = ["Origen"]
            
            for i, row in df_edited.iterrows():
                R_seg = (1/56)/row["Sección (mm2)"]
                dU_seg, _ = calc_voltage_drop("Exacta", V_nom, row["Longitud (m)"], row["P (kW)"], 0.9, R_seg, 0.00008)
                
                # Puntos para gráfica escalonada (step plot)
                dist_acc += row["Longitud (m)"]
                V_current -= dU_seg
                
                x_points.append(dist_acc)
                y_points.append(V_current)
                names.append(row["Tramo"])

            # Gráfica F6
            
            fig = px.line(x=x_points, y=y_points, markers=True, text=names,
                          title="Perfil de Tensión Acumulado (F6)")
            fig.update_traces(line_shape='linear', textposition="top right")
            
            limit_v = V_nom * (1 - 0.065) # 6.5% global limit example
            fig.add_hline(y=limit_v, line_dash="dot", line_color="red", annotation_text="Límite Global 6.5%")
            st.plotly_chart(fig, use_container_width=True)


def render_sizing_tab():
    """ 2d) Dimensionamiento Económico (Basado en G-Series) """
    st.subheader("Optimización Económica de Sección (G1-G6)")
    st.markdown("Encuentra el equilibrio entre Coste de Inversión (Cable) y Coste Operativo (Pérdidas).")

    # Inputs G1
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        st.markdown("**1. Datos Técnicos**")
        P_load = st.number_input("Carga (kW)", value=100.0)
        L_line = st.number_input("Longitud Línea (m)", value=250.0)
        V_source = st.number_input("Tensión (V)", value=400.0)
        mat_type = st.selectbox("Material", ["Cobre", "Aluminio"], key="eco_mat")
        
    with col_in2:
        st.markdown("**2. Datos Económicos**")
        years = st.number_input("Vida Útil (años)", value=20)
        hours = st.number_input("Horas Uso/año", value=4000)
        energy_cost = st.number_input("Coste Energía (€/kWh)", value=0.15)
        
    if st.button("🚀 Calcular Sección Óptima"):
        sigma = 56.0 if mat_type == "Cobre" else 35.0
        I_load = (P_load * 1000) / (np.sqrt(3) * V_source * 0.9) # CosPhi 0.9 fijo
        
        # 1. Obtener Base de Datos (G3)
        df_cables = get_standard_cables_db(mat_type)
        
        # 2. Iterar y Calcular Costes (G4)
        results = []
        for idx, row in df_cables.iterrows():
            # Check Técnico básico (Ampacidad)
            if row["Ampacity"] < I_load:
                continue # No vale técnicamente por calor
            
            # Costes
            capex, opex, total = calc_lifecycle_cost(row, L_line, I_load, hours, years, energy_cost, sigma)
            results.append({
                "Sección": str(row["Section"]),
                "S_num": row["Section"],
                "CAPEX (€)": capex,
                "OPEX (Pérdidas) (€)": opex,
                "Total (€)": total
            })
            
        df_res = pd.DataFrame(results)
        
        if df_res.empty:
            st.error("Ningún cable cumple la condición de intensidad mínima.")
        else:
            # Encontrar óptimo
            best_row = df_res.loc[df_res["Total (€)"].idxmin()]
            
            # Visualización G5 (Gráfico de Barras Apiladas)
            st.divider()
            c_best, c_plot = st.columns([1, 2])
            
            with c_best:
                st.success(f"Sección Óptima: {best_row['Sección']} mm²")
                st.metric("Coste Total Ciclo de Vida", f"{best_row['Total (€)']:,.2f} €")
                st.write(f"Inversión Inicial: {best_row['CAPEX (€)']:,.2f} €")
                st.write(f"Coste Pérdidas ({years} años): {best_row['OPEX (Pérdidas) (€)']:,.2f} €")
                
            with c_plot:
                fig = px.bar(df_res, x="Sección", y=["CAPEX (€)", "OPEX (Pérdidas) (€)"],
                             title="Análisis de Ciclo de Vida (G5)",
                             labels={"value": "Coste (€)", "Sección": "Sección (mm²)"})
                st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# BLOQUE 3: ESTRUCTURA PRINCIPAL (APP)
# ==============================================================================

def app():
    st.header("2. Advanced Line Calculations")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Amperaje Térmica", 
        "Caída de Tensión (Blondel)", 
        "Transitorios (Tiempo Real)", 
        "Optimización Económica"
    ])

    with tab1:
        render_thermal_tab()

    with tab2:
        render_voltage_drop_tab()

    with tab3:
        render_transient_tab()
        
    with tab4:
        render_sizing_tab()

# --- FIN DEL ARCHIVO ---