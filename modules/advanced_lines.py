import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

# ==============================================================================
# BLOQUE 1: MOTOR DE FÍSICA Y CÁLCULO
# ==============================================================================

def calc_resistance_dc(R20, alpha, T, T_ref=20):
    return R20 * (1 + alpha * (max(T, -273) - T_ref))

def calc_skin_proximity_factors(freq, diameter_mm, sigma, geometry_factor=1.0):
    radius_m = (max(diameter_mm, 1.0) / 1000) / 2
    area_m2 = np.pi * (radius_m**2)
    sigma_Sm = sigma * 1e6 
    if sigma_Sm <= 0: return 0, 0
    R_dc = 1 / (sigma_Sm * area_m2)
    x_s_sq = (8 * np.pi * freq / R_dc) * 1e-7 if R_dc > 0 else 0
    k_skin = (x_s_sq**2) / (192 + 0.8 * (x_s_sq**2))
    k_prox = k_skin * geometry_factor * 0.5 
    return k_skin, k_prox

def calc_heat_dissipation_advanced(T_surf, env_params, r_outer_m):
    T_amb = env_params.get('T_amb', 25)
    scenario = env_params.get('scenario', 'overhead')
    delta_T = T_surf - T_amb
    if delta_T <= 0: return 0.001 # Valor mínimo para estabilidad
    D_m = max(r_outer_m * 2, 0.001)
    
    if scenario == 'overhead':
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
        return max(0.1, q_conv + q_rad - solar)
    return 10.0 * delta_T # Simplificación para otros casos

def solve_transient_heating(I, time_span_sec, env_params, R20, alpha, T_ref, r_outer_m, mass_per_m, cp_mat):
    def thermal_ode(t, T):
        T_val = T[0]
        R_t = calc_resistance_dc(R20, alpha, T_val, T_ref)
        P_gen = (I**2) * R_t
        P_diss = calc_heat_dissipation_advanced(T_val, env_params, r_outer_m)
        dTdt = (P_gen - P_diss) / (max(mass_per_m, 0.1) * max(cp_mat, 100))
        return [dTdt]
    
    t_eval = np.linspace(0, time_span_sec, 100)
    sol = solve_ivp(thermal_ode, [0, time_span_sec], [env_params['T_amb']], t_eval=t_eval)
    return sol.t, sol.y[0]

def calc_voltage_drop(V_line, L_m, P_kw, cos_phi, R_ohm_km, X_ohm_km=0.08):
    I = (P_kw * 1000) / (np.sqrt(3) * V_line * cos_phi)
    # Convertir ohm/km a ohm/m
    r = R_ohm_km / 1000
    x = X_ohm_km / 1000
    sin_phi = np.sqrt(1 - cos_phi**2)
    dU = np.sqrt(3) * L_m * I * (r * cos_phi + x * sin_phi)
    return dU, I

def get_standard_cables_db(material):
    if material == "Cobre":
        data = [(1.5, 24, 0.6), (2.5, 32, 1.0), (4, 42, 1.5), (6, 54, 2.2), (10, 75, 3.8), (16, 100, 6.0), (25, 127, 9.5), (35, 158, 13.0), (50, 192, 18.0), (70, 246, 26.0), (95, 298, 35.0), (120, 346, 45.0), (150, 399, 56.0), (185, 456, 70.0), (240, 538, 92.0)]
    else:
        data = [(16, 75, 2.5), (25, 98, 3.8), (35, 120, 5.0), (50, 145, 6.8), (70, 185, 9.5), (95, 225, 12.0), (120, 262, 15.0), (150, 300, 19.0), (185, 345, 24.0), (240, 410, 32.0)]
    return pd.DataFrame(data, columns=["Section", "Ampacity", "Cost_per_m"])

# ==============================================================================
# BLOQUE 2: INTERFAZ DE USUARIO (TABS)
# ==============================================================================

def render_thermal_tab():
    st.subheader("⚡ Capacidad Térmica y Simulación Dinámica")
    
    col1, col2 = st.columns(2)
    with col1:
        material = st.selectbox("Material Conductor", ["Cobre", "Aluminio"])
        section = st.number_input("Sección (mm²)", min_value=1.5, value=50.0)
        t_amb = st.slider("Temp. Ambiente (°C)", 1, 60, 40)
    
    with col2:
        ins_type = st.selectbox("Aislamiento", ["PVC (70°C)", "XLPE (90°C)"])
        t_max = 70.0 if "PVC" in ins_type else 90.0
        p_load = st.number_input("Potencia de Carga (kW)", min_value=1.0, value=50.0)
        v_sys = st.number_input("Tensión (V)", min_value=1, value=400)

    # Cálculos base
    sigma = 56.0 if material == "Cobre" else 35.0
    alpha = 0.00393 if material == "Cobre" else 0.00403
    i_op = (p_load * 1000) / (np.sqrt(3) * v_sys * 0.85)
    
    # Simulación de calentamiento
    st.markdown("---")
    if st.button("🚀 Simular Curva de Calentamiento (Transitorio)"):
        # Parámetros físicos aproximados para la simulación
        r_outer = np.sqrt(section/np.pi)/1000
        r20 = 1000 / (sigma * section)
        env = {'T_amb': t_amb, 'scenario': 'overhead', 'wind': 0.6, 'emissivity': 0.5}
        
        t, temp_y = solve_transient_heating(i_op, 3600, env, r20, alpha, 20, r_outer, section*0.0089, 385)
        
        fig = px.line(x=t/60, y=temp_y, labels={'x':'Tiempo (min)', 'y':'Temp (°C)'}, title="Evolución Térmica del Conductor")
        fig.add_hline(y=t_max, line_dash="dash", line_color="red", annotation_text="Límite Aislante")
        st.plotly_chart(fig, use_container_width=True)



def render_voltage_drop_tab():
    st.subheader("📏 Verificación REBT - Caída de Tensión")
    
    c1, c2, c3 = st.columns(3)
    v_nom = c1.number_input("V Nominal", min_value=1, value=400)
    length = c2.number_input("Longitud (m)", min_value=1, value=150)
    p_kw = c3.number_input("Potencia (kW)", min_value=1, value=30)
    
    section = st.selectbox("Sección Seleccionada (mm²)", [6, 10, 16, 25, 35, 50, 70, 95, 120])
    
    r_km = 1000 / (56 * section)
    du, i_calc = calc_voltage_drop(v_nom, length, p_kw, 0.85, r_km)
    pct = (du / v_nom) * 100
    
    st.metric("Caída de Tensión", f"{du:.2f} V", f"{pct:.2f}%")
    if pct > 5.0:
        st.error("❌ Excede el límite aconsejado (5%)")
    else:
        st.success("✅ Dentro de límites normativos")

def render_sizing_tab():
    st.subheader("💰 Optimización de Costes (Ciclo de Vida)")
    
    mat = st.selectbox("Material para Optimizar", ["Cobre", "Aluminio"], key="opt_mat")
    p_load = st.number_input("Carga de Diseño (kW)", min_value=1, value=100)
    years = st.number_input("Años de Análisis", min_value=1, value=25)
    
    df = get_standard_cables_db(mat)
    i_load = (p_load * 1000) / (np.sqrt(3) * 400 * 0.9)
    
    # Filtrar solo cables que soportan la corriente
    df = df[df["Ampacity"] > i_load].copy()
    
    # Cálculo simple de pérdidas (OPEX)
    sigma = 56.0 if mat == "Cobre" else 35.0
    df["OPEX"] = ( (3 * i_load**2 * (1000/(sigma * df["Section"]))) / 1000) * 4000 * years * 0.15
    df["CAPEX"] = df["Cost_per_m"] * 1000 # Para 1km
    df["Total"] = df["OPEX"] + df["CAPEX"]
    
    st.plotly_chart(px.bar(df, x="Section", y=["CAPEX", "OPEX"], title="Coste Total a Largo Plazo (€/km)"))
    best_s = df.loc[df["Total"].idxmin(), "Section"]
    st.info(f"La sección económicamente más eficiente es de **{best_s} mm²**")

# ==============================================================================
# BLOQUE 3: APP PRINCIPAL
# ==============================================================================

def app():
    st.title("Advanced Line Calculations v2.0")
    t1, t2, t3 = st.tabs(["⚡ Térmico", "📏 Tensión", "💰 Economía"])
    with t1: render_thermal_tab()
    with t2: render_voltage_drop_tab()
    with t3: render_sizing_tab()

if __name__ == "__main__":
    app()
