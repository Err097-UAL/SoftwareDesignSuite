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

# --- A) PROPIEDADES ELÉCTRICAS Y TÉRMICAS ---
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
# BLOQUE 2: INTERFACES GRÁFICAS (TABS)
# ==============================================================================

def render_thermal_tab():
    st.subheader("Simulador Térmico y Ampacidad")
    
    with st.expander("📖 Fundamentos Térmicos e IEC 60287"):
        st.markdown("La capacidad de carga se basa en el equilibrio entre el calor generado por efecto Joule y el disipado al entorno:")
        st.latex(r"P_{gen}(T) = I^2 \cdot R_{DC} \cdot (1 + y_s + y_p) \cdot [1 + \alpha(T - 20)]")
        st.latex(r"P_{gen}(T) = Q_{conv}(T) + Q_{rad}(T)")
        st.write("Donde $y_s$ y $y_p$ son los factores de efecto piel y proximidad según **IEC 60287**.")

    c1, c2, c3 = st.columns(3)
    with c1:
        mat_type = st.selectbox("Material Conductor", ["Cobre", "Aluminio"], key="thm_mat")
        sigma = 56.0 if mat_type == "Cobre" else 35.0
        alpha = 0.00393 if mat_type == "Cobre" else 0.00403
        section_mm2 = st.selectbox("Sección (mm²)", [10, 16, 25, 35, 50, 70, 95, 120, 150, 185, 240], index=6, key="thm_sec")
        diameter_mm = 2 * np.sqrt(section_mm2 / np.pi)

    with c2:
        ins_type = st.selectbox("Aislamiento", ["XLPE (90°C)", "PVC (70°C)", "Desnudo (80°C)"], key="thm_ins")
        T_max = 90.0 if "XLPE" in ins_type else (70.0 if "PVC" in ins_type else 80.0)

    with c3:
        T_amb = st.slider("Temp. Ambiente (°C)", 0, 50, 25, key="thm_tamb")
        wind = st.slider("Viento (m/s)", 0.0, 5.0, 0.6, key="thm_wind")

    R_20_per_m = 1 / (sigma * section_mm2)
    k_skin, k_prox = calc_skin_proximity_factors(50, diameter_mm, sigma)
    env = {'scenario': 'overhead', 'T_amb': T_amb, 'wind': wind, 'emissivity': 0.5}

    currents = np.linspace(0, section_mm2 * 6, 50)
    temps = [solve_thermal_equilibrium(I, R_20_per_m, alpha, T_amb, diameter_mm, k_skin+k_prox, env) for I in currents]

    col_gr, col_res = st.columns([2,1])
    with col_gr:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=currents, y=temps, name='Temp. Equilibrio', line=dict(color='red', width=3)))
        fig.add_hline(y=T_max, line_dash="dash", line_color="orange", annotation_text=f"Límite {ins_type}")
        fig.update_layout(title="Curva de Calentamiento", xaxis_title="Corriente (A)", yaxis_title="Temperatura (°C)")
        st.plotly_chart(fig, use_container_width=True)

    with col_res:
        st.info("Resultados Técnicos")
        ampacity = np.interp(T_max, temps, currents)
        st.metric("Ampacidad Máxima", f"{ampacity:.1f} A")
        st.link_button("🌐 Verificar en IEEE 738", "https://standards.ieee.org/ieee/738/6837/")

def render_voltage_drop_tab():
    st.subheader("Caída de Tensión (Método Blondel)")
    
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

def render_therma_tab():
    st.subheader("⚡ Capacidad de Corriente (Modelo García Trasancos)")
    
    # --- FUNDAMENTOS TEÓRICOS (Basados en el Libro) ---
    with st.expander("📖 Fundamentos Técnicos (García Trasancos, Cap. 4)", expanded=False):
        st.markdown("""
        Según el modelo de García Trasancos, la intensidad máxima que puede transportar un cable 
        en régimen permanente depende de su capacidad para disipar el calor generado por **Efecto Joule**, 
        sin superar la temperatura máxima de servicio del aislante.
        """)
        
        st.markdown("**1. Resistencia corregida por temperatura:**")
        st.latex(r"R_T = R_{20} \cdot [1 + \alpha \cdot (T - 20)]")
        
        st.markdown("**2. Potencia disipada (Pérdidas Joule) en trifásica:**")
        st.latex(r"P_{pérdidas} = 3 \cdot I^2 \cdot R_T")
        
        st.markdown("**3. Intensidad Corregida (Criterio REBT/Trasancos):**")
        st.write("La intensidad máxima admisible ($I_{adm}$) se ajusta mediante factores de corrección ($k$):")
        st.latex(r"I_{máx} = I_{tab} \cdot k_1 \cdot k_2 \cdot k_n")
        
        st.info("Referencia: García Trasancos, J. (2020). Instalaciones Eléctricas en Media y Baja Tensión (8ª Ed.). Ed. Paraninfo. Págs. 102-120.")

    # --- ENTRADA DE DATOS ---
    st.markdown("##### ⚙️ Parámetros de la Instalación")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CAMBIA "simple_mat" por "thermal_mat_trasancos"
        material = st.selectbox("Material Conductor", ["Cobre", "Aluminio"], key="thermal_mat_trasancos")
        sigma = 56.0 if material == "Cobre" else 35.0
        alpha = 0.00393 if material == "Cobre" else 0.00403
        
    with col2:
        # CAMBIA "simple_ins" por "thermal_ins_trasancos"
        ins_type = st.selectbox("Tipo de Aislamiento", ["PVC (70°C)", "XLPE (90°C)"], key="thermal_ins_trasancos")
        t_max = 70.0 if "PVC" in ins_type else 90.0
        
    with col3:
        section = st.number_input("Sección del Conductor (mm²)", value=50.0, step=1.0)
        t_amb = st.slider("Temperatura Ambiente (°C)", 0, 60, 40)

    st.divider()

    # --- CÁLCULOS SEGÚN TRASANCOS ---
    # 1. Factores de Corrección (Simplificado según tablas del libro)
    # k1: Factor por temperatura ambiente (Base 40°C para aire en España)
    t_ref_rebt = 40.0
    k1 = np.sqrt((t_max - t_amb) / (t_max - t_ref_rebt))

    # 2. Resistencia unitaria a 20°C (Ohm/km)
    r20 = 1000 / (sigma * section)
    
    # 3. Resistencia a temperatura de servicio
    rt_max = r20 * (1 + alpha * (t_max - 20))

    # 4. Intensidad Máxima Admissible (Estimación basada en tablas del libro para cable al aire)
    # Valor base aproximado para sección S
    i_base = 5.5 * (section ** 0.8) # Fórmula empírica para tendencia de tablas
    i_final = i_base * k1

    # --- RESULTADOS ---
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric("Intensidad Máxima Admisible", f"{i_final:.1f} A")
        st.metric("Factor de Corrección (Temp)", f"{k1:.3f}")
        
        if t_amb > 40:
            st.warning("⚠️ El factor de corrección reduce la capacidad por alta temperatura ambiente.")

    with res_col2:
        # Gráfico de Pérdidas vs Intensidad
        i_range = np.linspace(0, i_final * 1.2, 50)
        p_losses = 3 * (i_range**2) * (rt_max / 1000) # kW/km
        
        fig = px.area(x=i_range, y=p_losses, 
                      title="Curva de Pérdidas Térmicas Joule (Régimen Permanente)",
                      labels={'x': 'Intensidad de Carga (A)', 'y': 'Pérdidas Joule (kW/km)'})
        
        # Marcador del punto límite
        fig.add_vline(x=i_final, line_dash="dash", line_color="red", annotation_text="Límite Térmico")
        st.plotly_chart(fig, use_container_width=True)

    # --- BIBLIOGRAFÍA ---
    st.markdown("---")
    st.markdown("### 📚 Bibliografía de Referencia")
    st.write("Puede contrastar estos cálculos en las tablas de intensidad admisible del REBT, citadas extensamente en el manual de García Trasancos.")
    
    c_link1, c_link2 = st.columns(2)
    with c_link1:
        st.link_button("📘 García Trasancos (8ª Edición)", 
                       "https://www.paraninfo.es/catalogo/9788428338974/instalaciones-electricas-en-media-y-baja-tension")
    with c_link2:
        st.link_button("📄 Guía de Aplicación REBT (ITC-BT-19)", 
                       "https://www.f2i2.net/documentos/lsi/rbt/guias/guia_bt_19_sep03R1.pdf")

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
    tab1, tab2, tab3, tab4 = st.tabs(["⚡ Ampacidad Térmica", "📏 Caída de Tensión", "⏱️ Transitorios", "💰 Optimización Económica"])
    with tab1: render_thermal_tab()
    with tab2: render_voltage_drop_tab()
    with tab3: render_therma_tab()
    with tab4: render_sizing_tab()