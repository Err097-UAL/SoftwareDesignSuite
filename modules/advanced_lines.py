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

    return R20 * (1 + alpha * (T - T_ref))


def calc_skin_proximity_factors(freq, diameter_mm, sigma, geometry_factor=1.0):

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

    T_amb = env_params.get('T_amb', 25)

    scenario = env_params.get('scenario', 'overhead')

    delta_T = T_surf - T_amb

    if delta_T <= 0: return 0

    D_m = r_outer_m * 2

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

        return max(0, q_conv + q_rad - solar)

    elif scenario == 'underground':

        rho_soil = env_params.get('rho_soil', 1.0)

        R_ext = rho_soil * np.log(4.0 / D_m) / (2 * np.pi)

        return delta_T / R_ext

    else:

        h_conv = 2.5 * (delta_T / D_m)**0.25 

        q_conv = h_conv * np.pi * D_m * delta_T

        return q_conv


def solve_thermal_equilibrium(Current, R20, alpha, T_amb, diameter_mm, k_ac_total, env_params):

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


def calc_voltage_drop(method, V_line, L_m, P_kw, cos_phi, R_ohm_m, X_ohm_m=0):

    I = (P_kw * 1000) / (np.sqrt(3) * V_line * cos_phi)

    sin_phi = np.sqrt(1 - cos_phi**2)

    if method == "Simplificada (Resistiva)":

        dU = np.sqrt(3) * L_m * I * R_ohm_m * cos_phi

    else:

        dU = np.sqrt(3) * L_m * I * (R_ohm_m * cos_phi + X_ohm_m * sin_phi)

    return dU, I


def check_rebt_compliance(dU, V_source, circuit_type):

    pct = (dU / V_source) * 100

    limit = 6.5 # Valor general por defecto

    return pct <= limit, pct, limit


def get_standard_cables_db(material):

    if material == "Cobre":

        data = [(1.5, 24, 0.6), (2.5, 32, 1.0), (4, 42, 1.5), (6, 54, 2.2), (10, 75, 3.8), (16, 100, 6.0), (25, 127, 9.5), (35, 158, 13.0), (50, 192, 18.0), (70, 246, 26.0), (95, 298, 35.0), (120, 346, 45.0), (150, 399, 56.0), (185, 456, 70.0), (240, 538, 92.0)]

    else:

        data = [(16, 75, 2.5), (25, 98, 3.8), (35, 120, 5.0), (50, 145, 6.8), (70, 185, 9.5), (95, 225, 12.0), (120, 262, 15.0), (150, 300, 19.0), (185, 345, 24.0), (240, 410, 32.0)]

    return pd.DataFrame(data, columns=["Section", "Ampacity", "Cost_per_m"])


def calc_lifecycle_cost(section_row, L_m, I_load, hours_year, years, energy_cost, sigma, line_type="Trifásica"):

    s = section_row["Section"]

    c_cable = section_row["Cost_per_m"]

    num_conductors = 3

    capex = L_m * c_cable * num_conductors

    R_total = L_m / (sigma * s)

    P_loss_kW = (num_conductors * (I_load**2) * R_total) / 1000

    opex = P_loss_kW * hours_year * years * energy_cost

    return capex, opex, capex + opex


# ==============================================================================

# BLOQUE 2: DEFINIMOS TABS USANDO FUNCIONES ANTERIORES DEL BLOQUE II

# ==============================================================================


def render_thermal_tab():

    st.subheader("⚡ Capacidad de Corriente e Intercambio de Calor")

    

    # --- FUNDAMENTOS TÉCNICOS ---

    with st.expander("📖 Fundamentos Técnicos (García Trasancos & Termodinámica)", expanded=False):

        st.markdown("""

        El intercambio de calor en un conductor se rige por el balance entre la energía generada por carga y la capacidad de disipación del entorno.

        """)

        

        st.markdown("**1. Corriente de Operación ($I_{op}$):**")

        st.latex(r"I_{op} = \frac{P}{\sqrt{3} \cdot V \cdot \cos \phi}")

        

        st.markdown("**2. Generación de Calor (Efecto Joule):**")

        st.latex(r"Q_{gen} = 3 \cdot I_{op}^2 \cdot R_T")

        

        st.markdown("**3. Criterio de Estabilidad:**")

        st.write("Para que el cable no se degrade, la temperatura de equilibrio debe ser menor al límite del aislante ($T_{eq} < T_{max}$).")


    # --- ENTRADA DE DATOS ---

    st.markdown("##### ⚙️ 1. Parámetros del Conductor e Instalación")

    col1, col2 = st.columns(2)

    

    with col1:

        material = st.selectbox("Material Conductor", ["Cobre", "Aluminio"], key="th_mat_unique")

        sigma = 56.0 if material == "Cobre" else 35.0

        alpha = 0.00393 if material == "Cobre" else 0.00403

        ins_type = st.selectbox("Aislamiento", ["PVC (70°C)", "XLPE (90°C)"], key="th_ins_unique")

        t_max = 70.0 if "PVC" in ins_type else 90.0

        

    with col2:

        section = st.number_input("Sección del Conductor (mm²)", value=50.0, step=1.0, key="th_sec_unique")

        t_amb = st.slider("Temperatura Ambiente (°C)", 0, 60, 40, key="th_tamb_unique")

        v_sys = st.number_input("Tensión del Sistema (V)", value=400, key="th_volt_unique")


    st.markdown("##### 🔌 2. Datos de la Carga (Escenario Real)")

    col_c1, col_c2, col_c3 = st.columns(3)

    

    with col_c1:

        p_load = st.number_input("Potencia Activa (kW)", value=50.0)

    with col_c2:

        cos_phi = st.number_input("Factor de Potencia (cos φ)", value=0.85, min_value=0.1, max_value=1.0)

    with col_c3:

        # Cálculo automático de I_op, pero permitimos verla

        i_op = (p_load * 1000) / (np.sqrt(3) * v_sys * cos_phi)

        st.metric("Corriente Resultante (I_op)", f"{i_op:.1f} A")


    st.divider()


    # --- CÁLCULOS DE INTERCAMBIO DE CALOR ---

    # 1. Capacidad Máxima (Admisible)

    t_ref_rebt = 40.0

    k_temp = np.sqrt(max(0.001, (t_max - t_amb) / (t_max - t_ref_rebt)))

    i_base = 5.5 * (section ** 0.8) # Estimación base tablas

    i_adm = i_base * k_temp


    # 2. Resistencia y Calor generado

    r20 = 1000 / (sigma * section)

    # Resistencia estimada a temperatura de operación (proporcional a la carga)

    t_est_op = t_amb + (t_max - t_amb) * (i_op / i_adm)**2 if i_op < i_adm else t_max + 20

    rt_op = r20 * (1 + alpha * (t_est_op - 20))

    q_gen_op = 3 * (i_op**2) * (rt_op / 1000) # kW/km


    # --- RESULTADOS Y COMPARATIVA ---

    res_col1, res_col2 = st.columns([1, 2])

    

    with res_col1:

        st.write("**Balance Térmico:**")

        st.metric("Capacidad Máxima (I_adm)", f"{i_adm:.1f} A")

        

        # Indicador de seguridad

        utilizacion = (i_op / i_adm) * 100

        if i_op > i_adm:

            st.error(f"❌ SOBRECARGA: {utilizacion:.1f}%")

            st.warning("El calor generado excede la capacidad de disipación.")

        else:

            st.success(f"✅ OPERACIÓN SEGURA: {utilizacion:.1f}%")

            st.write(f"Calor generado en carga: **{q_gen_op:.2f} kW/km**")


    with res_col2:

        # Gráfico dinámico: Punto de operación sobre curva de capacidad

        i_range = np.linspace(0, max(i_adm, i_op) * 1.3, 100)

        # Estimación de temperatura según corriente (Ley de Newton/Joule)

        t_curve = t_amb + (t_max - t_amb) * (i_range / i_adm)**2

        

        fig = go.Figure()

        # Curva de temperatura

        fig.add_trace(go.Scatter(x=i_range, y=t_curve, name="Perfil Térmico", line=dict(color="#00ADB5")))

        # Punto de operación real

        fig.add_trace(go.Scatter(x=[i_op], y=[t_est_op], mode="markers+text", 

                                 name="Tu Carga", text=["Punto de Operación"], 

                                 textposition="top center", marker=dict(color="orange", size=12)))

        # Límite

        fig.add_hline(y=t_max, line_dash="dash", line_color="red", annotation_text="Límite Aislante")

        

        fig.update_layout(title="Intercambio de Calor: Corriente vs Temperatura",

                          xaxis_title="Intensidad (A)", yaxis_title="Temp. Estimada Conductor (°C)")

        st.plotly_chart(fig, use_container_width=True)


    # --- BIBLIOGRAFÍA ---

    st.markdown("---")

    st.markdown("### 📚 Bibliografía y Validación de Fórmulas")

    c_l1, c_l2 = st.columns(2)

    with c_l1:

        st.link_button("📘 García Trasancos - Cap. 4 (Cálculo de Líneas)", 

                       "https://www.amazon.es/Instalaciones-el%C3%A9ctricas-media-tensi%C3%B3n-edici%C3%B3n/dp/842834809X")

    with c_l2:

        st.link_button("📜 Verificación Térmica (Manual Prysmian)", 

                       "https://www.prysmianclub.es/sdm_downloads/guia-tecnica-para-baja-tension-2023/")


def render_voltage_drop_tab():

    st.subheader("Algoritmo de comprobación de validez del REBT a partir de la Caída de Tensión (Método Blondel)")

    

    with st.expander("📖 Teoría de la Caída de Tensión"):

        st.markdown("Para líneas de transporte con impedancia $Z = R + jX$, la caída de tensión trifásica se calcula como:")

        st.latex(r"\Delta U = \sqrt{3} \cdot L \cdot I \cdot (r \cdot \cos \phi + x \cdot \sin \phi)")

        st.write("Donde $r$ y $x$ son las resistencia y reactancia unitarias ($\Omega/km$).")


    V_nom = st.number_input("Tensión Nominal (V)", 400.0)

    c1, c2, c3 = st.columns(3)

    L = c1.number_input("Longitud (m)", 150.0)

    P = c2.number_input("Potencia (kW)", 40.0)

    S = c3.selectbox("Sección Cu (mm²)", [10, 16, 25, 35, 50, 70, 95], index=3)

    

    R = (1/56) / S

    X = 0.00008 

    dU, _ = calc_voltage_drop("Exacta", V_nom, L, P, 0.9, R, X)

    ok, pct, limit = check_rebt_compliance(dU, V_nom, "Fuerza")

    

    col_res1, col_res2 = st.columns(2)

    col_res1.metric("Caída Tensión", f"{dU:.2f} V", delta=f"-{pct:.2f}%")

    if ok: col_res2.success(f"Cumple REBT")

    else: col_res2.error(f"Excede límite REBT")

    st.link_button("📜 Consultar REBT (ITC-BT-19)", "https://www.boe.es/buscar/act.php?id=BOE-A-2002-18099")


def render_sizing_tab():

    st.subheader("Optimización Económica")

    

    with st.expander("📖 Análisis de Ciclo de Vida (LCC)"):

        st.markdown("El coste total se optimiza minimizando la suma de la inversión inicial y el coste de las pérdidas energéticas:")

        st.latex(r"Coste_{Total} = CAPEX + \sum_{n=1}^{N} \frac{OPEX_{pérdidas}}{(1+i)^n}")

        st.write("Una sección mayor reduce el OPEX pero aumenta el CAPEX inicial. El software busca el punto mínimo.")


    c_in1, c_in2 = st.columns(2)

    with c_in1:

        P_load = st.number_input("Carga (kW)", value=100.0)

        L_line = st.number_input("Longitud (m)", value=250.0)

        mat_type = st.selectbox("Material", ["Cobre", "Aluminio"], key="eco_mat")

    with c_in2:

        years = st.number_input("Vida Útil (años)", value=20)

        energy_cost = st.number_input("Energía (€/kWh)", value=0.15)

        

    if st.button("🚀 Calcular Sección Óptima"):

        sigma = 56.0 if mat_type == "Cobre" else 35.0

        I_load = (P_load * 1000) / (np.sqrt(3) * 400 * 0.9)

        df_cables = get_standard_cables_db(mat_type)

        results = []

        for idx, row in df_cables.iterrows():

            if row["Ampacity"] < I_load: continue

            capex, opex, total = calc_lifecycle_cost(row, L_line, I_load, 4000, years, energy_cost, sigma)

            results.append({"Sección": str(row["Section"]), "S_num": row["Section"], "CAPEX (€)": capex, "OPEX (€)": opex, "Total (€)": total})

        

        df_res = pd.DataFrame(results)

        best_row = df_res.loc[df_res["Total (€)"].idxmin()]

        

        st.divider()

        c_best, c_plot = st.columns([1, 2])

        with c_best:

            cond_color = "#FFD700" if mat_type == "Cobre" else "#C0C0C0"

            cable_html = f"""

            <div style="display: flex; align-items: center; background-color: #161B22; padding: 20px; border-radius: 15px; border: 1px solid #30363D;">

                <svg width="80" height="80" viewBox="0 0 100 100"><circle cx="50" cy="50" r="45" fill="#30363D" /><circle cx="50" cy="50" r="30" fill="{cond_color}" /></svg>

                <div style="padding-left: 20px;"><div style="color: #8B949E; font-size: 0.8rem;">SECCIÓN ÓPTIMA</div><div style="color: white; font-size: 2rem; font-weight: 800;">{best_row['Sección']} mm²</div></div>

            </div>"""

            st.markdown(cable_html, unsafe_allow_html=True)

            st.metric("Ahorro Potencial", f"{best_row['Total (€)']:,.0f} €")

        with c_plot:

            st.plotly_chart(px.bar(df_res, x="Sección", y=["CAPEX (€)", "OPEX (€)"], title="Análisis de Costes"), use_container_width=True)

    st.link_button("📈 Estándar IEC 60287-3-2", "https://webstore.iec.ch/publication/1233")


# ==============================================================================

# BLOQUE 3: APP PRINCIPAL

# ==============================================================================


def app():

    st.header("Advanced Line Calculations")

    tab1, tab2, tab4 = st.tabs(["⚡ Ampacidad Térmica", "📏 Caída de Tensión", "💰 Optimización Económica"])

    with tab1: render_thermal_tab()

    with tab2: render_voltage_drop_tab()

    with tab4: render_sizing_tab() 
