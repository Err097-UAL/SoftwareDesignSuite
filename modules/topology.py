import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# BLOQUE DE FUNCIONES 1: MOTOR DE CÁLCULO (Traducción de H, I, J series)
# ==============================================================================

def get_material_properties(name):
    """ Equivalente a getMaterialProperties.m """
    if name == "Cobre":
        return {"sigma": 56.0, "cost_factor": 1.5}
    elif name == "Aluminio":
        return {"sigma": 36.0, "cost_factor": 1.0}
    else:
        return {"sigma": 56.0, "cost_factor": 1.0}

def solve_radial_network(U_source, nodes_df, phase_type, sigma, section):
    """ Equivalente a H1_Radial_Analysis.m """
    k = 2 if phase_type == "Monofásica" else 1 
    nodes = nodes_df.sort_values("Distancia (m)").to_dict('records')
    total_current = sum(n["Carga (A)"] for n in nodes)
    current_flow = total_current
    
    processed_nodes = []
    prev_dist = 0
    voltage_current = U_source
    
    for n in nodes:
        dist = n["Distancia (m)"]
        load_i = n["Carga (A)"]
        segment_len = dist - prev_dist
        R_seg = segment_len / (sigma * section)
        drop_seg = k * current_flow * R_seg
        voltage_current -= drop_seg
        
        processed_nodes.append({
            "Distancia": dist,
            "Carga": load_i,
            "Corriente_Tramo": current_flow,
            "Caída_Tramo": drop_seg,
            "Tensión": voltage_current
        })
        current_flow -= load_i
        prev_dist = dist
        
    return pd.DataFrame(processed_nodes)

def solve_dual_fed_network(Ua, Ub, L_total, loads_df, sigma, section):
    """ Equivalente a I1_DualFed_Analysis.m y J1_Ring_Analysis.m """
    R_total_line = L_total / (sigma * section)
    loads = loads_df.sort_values("Distancia (m)").to_dict('records')
    moment_sum = sum(load["Carga (A)"] * (L_total - load["Distancia (m)"]) for load in loads)
    
    Ia = (1/L_total) * moment_sum + (Ua - Ub) / R_total_line
    Ib = sum(l["Carga (A)"] for l in loads) - Ia
    
    profile = [{"Distancia": 0, "Tensión": Ua, "Corriente_Acum": Ia, "Tipo": "Fuente A"}]
    current_flow, v_current, prev_dist = Ia, Ua, 0
    min_v, min_v_dist = float('inf'), 0
    
    for load in loads:
        dist, i_load = load["Distancia (m)"], load["Carga (A)"]
        R_seg = (dist - prev_dist) / (sigma * section)
        v_current -= current_flow * R_seg
        profile.append({"Distancia": dist, "Tensión": v_current, "Corriente_Acum": current_flow, "Tipo": "Carga"})
        if v_current < min_v: min_v, min_v_dist = v_current, dist
        current_flow -= i_load
        prev_dist = dist
        
    profile.append({"Distancia": L_total, "Tensión": Ub, "Corriente_Acum": current_flow, "Tipo": "Fuente B"})
    return pd.DataFrame(profile), Ia, Ib, min_v, min_v_dist

def calc_network_cost(topology_type, L_total, loads_df, sigma, section, material_cost_factor):
    base_price = 5.0 + (section * 0.15) * material_cost_factor
    cable_len = loads_df["Distancia (m)"].max() if topology_type == "Radial" else L_total
    capex = cable_len * base_price
    total_amps = loads_df["Carga (A)"].sum()
    R_total = cable_len / (sigma * section)
    opex = ((total_amps**2 * R_total * 0.33) / 1000) * (20 * 365 * 10) * 0.15
    return capex, opex

# ==============================================================================
# BLOQUE DE FUNCIONES 2: VISUALIZACIÓN
# ==============================================================================

def plot_unifilar_ring(L_total, loads_df):
    fig = go.Figure()
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines', line=dict(color='white', width=3)))
    fig.add_trace(go.Scatter(x=[0], y=[1], mode='markers+text', marker=dict(symbol='square', size=15, color='#00ADB5'), text=["Fuente"]))
    for _, row in loads_df.iterrows():
        angle = (np.pi/2) - (row["Distancia (m)"] / L_total) * (2 * np.pi)
        fig.add_trace(go.Scatter(x=[np.cos(angle), np.cos(angle)*0.85], y=[np.sin(angle), np.sin(angle)*0.85], mode='lines', line=dict(color='red')))
    fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
    return fig

def plot_loading_diagram(profile_df, L_total):
    fig = go.Figure()
    dist, curr = profile_df["Distancia"].tolist(), profile_df["Corriente_Acum"].tolist()
    x_s, y_s = [], []
    for i in range(len(dist)-1):
        x_s.extend([dist[i], dist[i+1]])
        y_s.extend([curr[i], curr[i]])
    fig.add_trace(go.Scatter(x=x_s, y=y_s, mode='lines', fill='tozeroy', line=dict(color='#00ADB5')))
    fig.update_layout(title="Distribución de Corrientes", xaxis_title="m", yaxis_title="A")
    return fig

# ==============================================================================
# BLOQUE DE FUNCIONES 3: INTERFAZ DE USUARIO (TABS)
# ==============================================================================

def render_radial_tab():
    st.subheader("1. Análisis de Red Radial (H)")
    
    with st.expander("📖 Fundamentos: Método de los Momentos Eléctricos"):
        st.markdown("En redes radiales con cargas concentradas, la caída de tensión acumulada se calcula mediante la suma de los momentos eléctricos de cada tramo:")
        st.latex(r"\Delta U = \sum_{i=1}^{n} \Delta U_i = \frac{K}{\sigma \cdot S} \sum_{i=1}^{n} (I_{tramo,i} \cdot L_i)")
        st.write("Donde $K$ es el factor de fase (2 para monofásico, 1 para trifásico fase-neutro).")
        st.link_button("📜 Teoría: Momentos Eléctricos (García Trasancos)", "https://www.paraninfo.es/catalogo/9788428338974/instalaciones-electricas-en-media-y-baja-tension")

    

    c1, c2 = st.columns(2)
    U_source = c1.number_input("Tensión Fuente (V)", 230.0)
    phase = c2.selectbox("Sistema", ["Monofásica", "Trifásica"])
    df_loads = st.data_editor(pd.DataFrame([{"Distancia (m)": 50, "Carga (A)": 20}, {"Distancia (m)": 150, "Carga (A)": 15}]), num_rows="dynamic")
    
    if st.button("Calcular Red Radial"):
        res = solve_radial_network(U_source, df_loads, phase, 56, 50)
        st.plotly_chart(px.line(res, x="Distancia", y="Tensión", markers=True, title="Perfil de Tensión"), use_container_width=True)

def render_dualfed_tab():
    st.subheader("2. Red de Doble Alimentación (I)")
    
    with st.expander("📖 Fundamentos: Reparto de Cargas"):
        st.markdown("Para una línea alimentada por ambos extremos ($V_A$ y $V_B$), la corriente de aporte desde la fuente A se determina por:")
        st.latex(r"I_A = \frac{(V_A - V_B) + \sum (I_i \cdot R_{i,B})}{R_{total}}")
        st.write("El punto de mínima tensión es aquel donde las corrientes de ambos sentidos convergen (punto de corte).")
        st.link_button("📘 Schneider: Guía de Diseño de Redes", "https://www.se.com/es/es/download/document/LVPED210007ES/")

    

    c1, c2, c3 = st.columns(3)
    Ua, Ub, L_tot = c1.number_input("V_A (V)", 405.0), c2.number_input("V_B (V)", 400.0), c3.number_input("L (m)", 500.0)
    df_loads = st.data_editor(pd.DataFrame([{"Distancia (m)": 100, "Carga (A)": 40}, {"Distancia (m)": 350, "Carga (A)": 30}]), num_rows="dynamic")
    
    if st.button("Analizar Red"):
        profile, Ia, Ib, mv, mvd = solve_dual_fed_network(Ua, Ub, L_tot, df_loads, 56, 70)
        st.metric("Punto Crítico", f"{mv:.2f} V", f"a {mvd} m")
        st.plotly_chart(plot_loading_diagram(profile, L_tot), use_container_width=True)

def render_ring_tab():
    st.subheader("3. Red en Anillo (J)")
    
    with st.expander("📖 Fundamentos: Topología en Bucle Cerrado"):
        st.markdown("Un anillo es un caso especial de doble alimentación donde $V_A = V_B$. Esto garantiza mayor fiabilidad, ya que cualquier carga puede ser alimentada por dos caminos:")
        st.latex(r"I_{clockwise} = \frac{\sum (I_i \cdot L_{anti-clockwise, i})}{L_{total}}")
        st.link_button("🌐 Manual de Redes de Distribución", "https://es.wikipedia.org/wiki/Red_de_distribuci%C3%B3n_de_energ%C3%ADa_el%C3%A9ctrica#Topolog%C3%ADas_de_redes_de_distribuci%C3%B3n")

    

    c1, c2 = st.columns(2)
    Uf, Lr = c1.number_input("V_Alimentación (V)", 400.0), c2.number_input("Perímetro (m)", 1000.0)
    df_loads = st.data_editor(pd.DataFrame([{"Distancia (m)": 200, "Carga (A)": 50}, {"Distancia (m)": 700, "Carga (A)": 40}]), num_rows="dynamic")
    
    if st.button("Resolver Anillo"):
        profile, Ia, Ib, mv, mvd = solve_dual_fed_network(Uf, Uf, Lr, df_loads, 56, 95)
        col_l, col_r = st.columns([1, 2])
        with col_l: st.success(f"V_min: {mv:.2f}V"); st.info(f"Ia: {Ia:.1f}A / Ib: {Ib:.1f}A")
        with col_r: st.plotly_chart(plot_unifilar_ring(Lr, df_loads), use_container_width=True)

def render_comparison_tab():
    st.subheader("4. Comparativa Técnica/Económica")
    
    with st.expander("📖 Análisis de Coste de Ciclo de Vida (LCC)"):
        st.markdown("La comparativa evalúa el compromiso entre la inversión inicial (CAPEX) y los costes operativos por pérdidas Joule (OPEX):")
        st.latex(r"Coste_{total} = C_{cable} \cdot L + \int_{0}^{20y} (3 \cdot I^2(t) \cdot R \cdot Coste_{kWh}) dt")
        st.link_button("📈 Eficiencia Energética (Leonardo Energy)", "https://leonardo-energy.org/resources/113/sizing-of-conductors-for-energy-efficiency-5807")

    L_t, U_n = st.number_input("Longitud Total (m)", 800.0), st.number_input("Tensión Nom (V)", 400.0)
    df_loads = st.data_editor(pd.DataFrame([{"Distancia (m)": 200, "Carga (A)": 50}, {"Distancia (m)": 600, "Carga (A)": 50}]))
    
    if st.button("Comparar Topologías"):
        res_rad = solve_radial_network(U_n, df_loads, "Trifásica", 56, 95)
        res_ring, _, _, mv_ring, _ = solve_dual_fed_network(U_n, U_n, L_t, df_loads, 56, 95)
        c_r, o_r = calc_network_cost("Radial", L_t, df_loads, 56, 95, 1.5)
        c_an, o_an = calc_network_cost("Anillo", L_t, df_loads, 56, 95, 1.5)
        
        st.plotly_chart(px.bar(pd.DataFrame({
            "Topología": ["Radial", "Radial", "Anillo", "Anillo"],
            "Tipo": ["CAPEX", "OPEX", "CAPEX", "OPEX"],
            "Euros": [c_r, o_r, c_an, o_an]
        }), x="Topología", y="Euros", color="Tipo", barmode="stack"))

def app():
    st.header("Network Topology & Dimensioning")
    t1, t2, t3, t4 = st.tabs(["1. Radial (H)", "2. Doble Alimentación (I)", "3. Anillo (J)", "4. Comparativa"])
    with t1: render_radial_tab()
    with t2: render_dualfed_tab()
    with t3: render_ring_tab()
    with t4: render_comparison_tab()