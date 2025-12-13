import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# BLOQUE 1: MOTOR DE CÁLCULO (Traducción de H, I, J series)
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
    # Phase factor (K): 2 para Monofásico, 1 (o sqrt(3) según fórmula de momento) para Trifásico
    # En cálculo de momentos simplificado para caída de tensión:
    # Mono: 2, Tri: 1 (si es V_fase-neutro) o sqrt(3) (si es V_linea)
    # Usaremos K=2 (Mono) y K=1 (Trifásico por fase) para simplificar momentos
    k = 2 if phase_type == "Monofásica" else 1 
    
    results = []
    cumulative_drop = 0
    
    # Ordenar nodos por distancia
    nodes = nodes_df.sort_values("Distancia (m)").to_dict('records')
    
    # 1. Cálculo de Momentos Eléctricos y Caída
    # El método de momentos calcula la caída hasta el punto X sumando el efecto de todas las cargas posteriores
    # Pero una forma iterativa más exacta es calcular la caída tramo a tramo.
    
    current_node_dist = 0
    voltage_current = U_source
    
    # Calculamos la corriente que pasa por cada tramo
    # Tramo 0->1 lleva TODA la corriente. Tramo 1->2 lleva (Total - I_1), etc.
    total_current = sum(n["Carga (A)"] for n in nodes)
    current_flow = total_current
    
    processed_nodes = []
    prev_dist = 0
    
    for n in nodes:
        dist = n["Distancia (m)"]
        load_i = n["Carga (A)"]
        segment_len = dist - prev_dist
        
        # Resistencia del tramo
        # R = L / (sigma * S)
        R_seg = segment_len / (sigma * section)
        
        # Caída en el tramo: dU = K * I_flow * R_seg
        drop_seg = k * current_flow * R_seg
        voltage_current -= drop_seg
        
        processed_nodes.append({
            "Distancia": dist,
            "Carga": load_i,
            "Corriente_Tramo": current_flow,
            "Caída_Tramo": drop_seg,
            "Tensión": voltage_current
        })
        
        # Preparamos siguiente tramo
        current_flow -= load_i
        prev_dist = dist
        
    return pd.DataFrame(processed_nodes)

def solve_dual_fed_network(Ua, Ub, L_total, loads_df, sigma, section):
    """ 
    Equivalente a I1_DualFed_Analysis.m y J1_Ring_Analysis.m 
    Resuelve la red encontrando el punto de mínima tensión (Punto de corte).
    """
    R_total_line = L_total / (sigma * section)
    loads = loads_df.sort_values("Distancia (m)").to_dict('records')
    
    # 1. Cálculo de Corrientes de Aporte (Ia e Ib)
    # Fórmula: Ia = [ (Va - Vb) + Sum(Ii * R_i_b) ] / R_total
    # Simplificado por momentos: 
    # Ia = (1/L) * Sum(Ii * (L - xi)) + (Va - Vb)/R_linea
    
    moment_sum = 0
    for load in loads:
        moment_sum += load["Carga (A)"] * (L_total - load["Distancia (m)"])
    
    Ia = (1/L_total) * moment_sum + (Ua - Ub) / R_total_line
    
    # Ib es el resto (o balance negativo)
    total_load = sum(l["Carga (A)"] for l in loads)
    Ib = total_load - Ia
    
    # 2. Perfil de Tensión
    profile = []
    # Punto inicial A
    profile.append({"Distancia": 0, "Tensión": Ua, "Corriente_Acum": Ia, "Tipo": "Fuente A"})
    
    current_flow = Ia
    v_current = Ua
    prev_dist = 0
    
    min_v = float('inf')
    min_v_dist = 0
    
    for load in loads:
        dist = load["Distancia (m)"]
        i_load = load["Carga (A)"]
        seg_len = dist - prev_dist
        
        R_seg = seg_len / (sigma * section)
        
        # Caída
        drop = current_flow * R_seg # Asumimos K=1 (por hilo) para simplificar visualización
        v_current -= drop
        
        # Registrar punto (antes de la carga)
        profile.append({"Distancia": dist, "Tensión": v_current, "Corriente_Acum": current_flow, "Tipo": "Carga"})
        
        if v_current < min_v:
            min_v = v_current
            min_v_dist = dist
            
        # Descargar corriente
        current_flow -= i_load
        prev_dist = dist
        
    # Punto final B
    seg_len_final = L_total - prev_dist
    R_final = seg_len_final / (sigma * section)
    drop_final = current_flow * R_final
    v_final = v_current - drop_final
    
    profile.append({"Distancia": L_total, "Tensión": v_final, "Corriente_Acum": current_flow, "Tipo": "Fuente B"})
    
    return pd.DataFrame(profile), Ia, Ib, min_v, min_v_dist

def calc_network_cost(topology_type, L_total, loads_df, sigma, section, material_cost_factor):
    """ Equivalente a J6_CalculateNetworkCost.m """
    # 1. CAPEX (Coste Cable)
    # Precio base ejemplo: 10 €/m para sección base, escalado por sección
    base_price = 5.0 + (section * 0.15) * material_cost_factor
    
    if topology_type == "Radial":
        # Radial: cable llega hasta la última carga
        cable_len = loads_df["Distancia (m)"].max()
    else:
        # Anillo: cable cierra el círculo
        cable_len = L_total
        
    capex = cable_len * base_price
    
    # 2. OPEX (Pérdidas Joule)
    # Simplificación: Energía perdida en 20 años
    # P_loss = I^2 * R
    # (El cálculo exacto requeriría integrar el perfil de corriente calculado en solve_dual...)
    # Hacemos una estimación basada en corriente media para no recalcular todo aquí
    total_amps = loads_df["Carga (A)"].sum()
    R_total = cable_len / (sigma * section)
    
    # Factor de uso (simulado)
    hours_20y = 20 * 365 * 10 # 10 horas al día
    # Estimación de pérdidas distribuida (aprox 1/3 de pérdidas pico en punta)
    loss_factor = 0.33
    P_peak_loss = (total_amps**2) * R_total * loss_factor 
    
    opex = (P_peak_loss / 1000) * hours_20y * 0.15 # 0.15 €/kWh
    
    return capex, opex

# ==============================================================================
# BLOQUE 2: VISUALIZACIÓN AVANZADA (Plotly recreando I2, I3, I4, J3)
# ==============================================================================

def plot_unifilar_ring(L_total, loads_df):
    """ J3_PlotUnifilarDiagram.m (Polar) """
    fig = go.Figure()
    
    # 1. Dibujar el anillo (Círculo)
    theta = np.linspace(0, 2*np.pi, 100)
    r = 1.0
    fig.add_trace(go.Scatter(x=r*np.cos(theta), y=r*np.sin(theta), mode='lines', 
                             line=dict(color='black', width=3), name='Línea Principal'))
    
    # 2. Fuente (arriba, pi/2)
    fig.add_trace(go.Scatter(x=[0], y=[1], mode='markers+text', 
                             marker=dict(symbol='square', size=15, color='green'),
                             text=["Fuente"], textposition="top center", name='Fuente'))
    
    # 3. Cargas
    for _, row in loads_df.iterrows():
        dist = row["Distancia (m)"]
        # Convertir distancia a ángulo (Clockwise desde arriba)
        # angulo = pi/2 - (dist / L_total) * 2pi
        angle = (np.pi/2) - (dist / L_total) * (2 * np.pi)
        
        x_load = np.cos(angle)
        y_load = np.sin(angle)
        
        # Flecha hacia adentro
        fig.add_trace(go.Scatter(x=[x_load, x_load*0.85], y=[y_load, y_load*0.85], 
                                 mode='lines+markers', marker=dict(symbol='arrow-bar-up', size=10),
                                 line=dict(color='red'), showlegend=False))
        
        fig.add_annotation(x=x_load*0.75, y=y_load*0.75, text=f"{row['Carga (A)']}A", showarrow=False)

    fig.update_layout(title="Diagrama Unifilar (Topología Anillo)", xaxis_visible=False, yaxis_visible=False,
                      width=500, height=500, showlegend=False)
    return fig

def plot_loading_diagram(profile_df, L_total):
    """ I3_PlotLoadingDiagram.m / J4 """
    # Gráfico de dientes de sierra para la corriente
    fig = go.Figure()
    
    distances = profile_df["Distancia"].tolist()
    currents = profile_df["Corriente_Acum"].tolist()
    
    # Crear efecto escalón (Step line)
    x_step = []
    y_step = []
    
    for i in range(len(distances)-1):
        x_step.append(distances[i])
        y_step.append(currents[i])
        x_step.append(distances[i+1]) # Mantenemos corriente hasta la siguiente carga
        y_step.append(currents[i]) 
        
    # Añadir último punto
    x_step.append(distances[-1])
    y_step.append(currents[-1])

    fig.add_trace(go.Scatter(x=x_step, y=y_step, mode='lines', fill='tozeroy', 
                             name='Carga Acumulada', line=dict(color='orange')))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    
    # Anotaciones
    max_curr = max(currents)
    min_curr = min(currents) # Será negativo (Ib)
    
    fig.add_annotation(x=0, y=max_curr, text=f"Ia = {max_curr:.1f}A", showarrow=True)
    fig.add_annotation(x=L_total, y=min_curr, text=f"Ib = {abs(min_curr):.1f}A", showarrow=True)

    fig.update_layout(title="Diagrama de Carga (Distribución de Corrientes)",
                      xaxis_title="Distancia (m)", yaxis_title="Corriente en Línea (A)")
    return fig

# ==============================================================================
# BLOQUE 3: INTERFAZ DE USUARIO (TABS)
# ==============================================================================

def render_radial_tab():
    """ H-Series """
    st.subheader("1. Análisis de Red Radial (H)")
    st.caption("Método de los momentos eléctricos para una sola fuente.")
    
    c1, c2 = st.columns(2)
    U_source = c1.number_input("Tensión Fuente (V)", 230.0)
    phase = c2.selectbox("Sistema", ["Monofásica", "Trifásica"])
    
    st.markdown("**Definición de Cargas:**")
    df_template = pd.DataFrame([{"Distancia (m)": 50, "Carga (A)": 20}, 
                                {"Distancia (m)": 120, "Carga (A)": 15},
                                {"Distancia (m)": 200, "Carga (A)": 30}])
    df_loads = st.data_editor(df_template, num_rows="dynamic", key="rad_loads")
    
    c3, c4 = st.columns(2)
    mat = c3.selectbox("Material", ["Cobre", "Aluminio"], key="rad_mat")
    sec = c4.selectbox("Sección (mm²)", [10, 16, 25, 35, 50, 70, 95, 120, 150], index=4, key="rad_sec")
    
    if st.button("Calcular Radial"):
        props = get_material_properties(mat)
        res_df = solve_radial_network(U_source, df_loads, phase, props["sigma"], sec)
        
        st.dataframe(res_df.style.format({"Tensión": "{:.2f} V", "Caída_Tramo": "{:.2f} V"}), use_container_width=True)
        
        # Gráfico Perfil H2
        
        fig = px.line(res_df, x="Distancia", y="Tensión", markers=True, title="Perfil de Tensión Radial (H2)")
        fig.add_scatter(x=[0], y=[U_source], mode='markers', marker_symbol='star', marker_size=15, name='Fuente')
        st.plotly_chart(fig, use_container_width=True)

def render_dualfed_tab():
    """ I-Series """
    st.subheader("2. Red Doble Alimentación (I)")
    st.caption("Análisis de fiabilidad con dos fuentes (A y B) y punto de corte óptimo.")
    
    c1, c2, c3 = st.columns(3)
    Ua = c1.number_input("Tensión Fuente A (V)", 405.0)
    Ub = c2.number_input("Tensión Fuente B (V)", 400.0)
    L_tot = c3.number_input("Longitud Total (m)", 500.0)
    
    st.markdown("**Cargas entre A y B:**")
    df_template = pd.DataFrame([{"Distancia (m)": 100, "Carga (A)": 40}, 
                                {"Distancia (m)": 300, "Carga (A)": 30},
                                {"Distancia (m)": 450, "Carga (A)": 20}])
    df_loads = st.data_editor(df_template, num_rows="dynamic", key="dual_loads")
    
    c4, c5 = st.columns(2)
    mat = c4.selectbox("Material", ["Cobre", "Aluminio"], key="dual_mat")
    sec = c5.selectbox("Sección (mm²)", [25, 35, 50, 70, 95, 120, 150, 185], index=4, key="dual_sec")
    
    if st.button("Analizar Doble Alimentación"):
        props = get_material_properties(mat)
        profile_df, Ia, Ib, min_v, min_v_dist = solve_dual_fed_network(Ua, Ub, L_tot, df_loads, props["sigma"], sec)
        
        # Resultados Texto
        k1, k2, k3 = st.columns(3)
        k1.metric("Aporte Fuente A", f"{Ia:.1f} A")
        k2.metric("Aporte Fuente B", f"{Ib:.1f} A")
        k3.metric("Tensión Mínima", f"{min_v:.2f} V", delta=f"a {min_v_dist} m")
        
        # Gráfico I2 (Perfil)
        tab_g1, tab_g2 = st.tabs(["Perfil Tensión", "Diagrama Carga"])
        
        with tab_g1:
            fig_v = px.line(profile_df, x="Distancia", y="Tensión", markers=True, title="Perfil Tensión (I2)")
            # Punto mínimo
            fig_v.add_annotation(x=min_v_dist, y=min_v, text="Punto Corte", showarrow=True, arrowhead=1)
            # Simulación Fallo (Radial simple desde A)
            # (Simplificado: Trazar línea teórica si B cae)
            st.plotly_chart(fig_v, use_container_width=True)
            
        with tab_g2:
            fig_load = plot_loading_diagram(profile_df, L_tot)
            st.plotly_chart(fig_load, use_container_width=True)

def render_ring_tab():
    """ J-Series """
    st.subheader("3. Red en Anillo (J)")
    st.caption("Topología cerrada. El cálculo es idéntico a Doble Alimentación con Va = Vb.")
    
    c1, c2 = st.columns(2)
    U_feed = c1.number_input("Tensión Alimentación (V)", 400.0)
    L_ring = c2.number_input("Perímetro Anillo (m)", 1000.0)
    
    st.markdown("**Cargas en el Anillo (Distancia desde bornas):**")
    df_template = pd.DataFrame([{"Distancia (m)": 200, "Carga (A)": 50}, 
                                {"Distancia (m)": 500, "Carga (A)": 40},
                                {"Distancia (m)": 800, "Carga (A)": 60}])
    df_loads = st.data_editor(df_template, num_rows="dynamic", key="ring_loads")
    
    mat = st.selectbox("Material", ["Cobre", "Aluminio"], key="ring_mat")
    sec = st.selectbox("Sección (mm²)", [50, 70, 95, 120, 150, 185, 240], index=2, key="ring_sec")
    
    if st.button("Resolver Anillo"):
        props = get_material_properties(mat)
        # Anillo es Doble alimentación con Ua=Ub
        profile_df, Ia, Ib, min_v, min_v_dist = solve_dual_fed_network(U_feed, U_feed, L_ring, df_loads, props["sigma"], sec)
        
        c_res1, c_res2 = st.columns([1, 2])
        
        with c_res1:
            st.success(f"Tensión Mínima: {min_v:.2f} V")
            st.info(f"Corriente Sentido Horario: {Ia:.1f} A")
            st.info(f"Corriente Anti-Horario: {Ib:.1f} A")
        
        with c_res2:
            # Gráfico J3 (Unifilar Polar)
            
            fig_ring = plot_unifilar_ring(L_ring, df_loads)
            st.plotly_chart(fig_ring, use_container_width=True)
            
        # Gráfico Perfil "Desenrollado" (J2)
        st.divider()
        st.markdown("##### Perfil de Tensión Desenrollado (J2)")
        fig_unwrapped = px.line(profile_df, x="Distancia", y="Tensión", markers=True)
        fig_unwrapped.add_vline(x=min_v_dist, line_dash="dash", line_color="red", annotation_text="Punto Abierto")
        st.plotly_chart(fig_unwrapped, use_container_width=True)

def render_comparison_tab():
    """ Comparativa (J5-J7) """
    st.subheader("4. Comparativa: Radial vs Anillo")
    
    st.info("Compararemos alimentar las cargas mediante una línea abierta (Radial) o cerrando el bucle (Anillo).")
    
    # Inputs Simplificados
    col_a, col_b = st.columns(2)
    L_total = col_a.number_input("Longitud Total Línea (m)", 800.0)
    U_nom = col_b.number_input("Tensión Nominal (V)", 400.0)
    
    df_template = pd.DataFrame([{"Distancia (m)": 200, "Carga (A)": 50}, 
                                {"Distancia (m)": 600, "Carga (A)": 50}])
    df_loads = st.data_editor(df_template, key="comp_loads")
    
    sec = st.selectbox("Sección Común (mm²)", [95, 120, 150], key="comp_sec")
    
    if st.button("Ejecutar Comparativa (J5)"):
        props = get_material_properties("Cobre") # Asumimos cobre para comparar
        
        # 1. Caso Radial (H)
        # Asumimos que la radial termina en la última carga
        # Pero para comparar justamente con anillo, usamos la misma longitud total de cable
        # Radial simple: solve_radial...
        res_radial = solve_radial_network(U_nom, df_loads, "Trifásica", props["sigma"], sec)
        min_v_radial = res_radial["Tensión"].min()
        capex_rad, opex_rad = calc_network_cost("Radial", L_total, df_loads, props["sigma"], sec, props["cost_factor"])
        
        # 2. Caso Anillo (J)
        res_ring, _, _, min_v_ring, _ = solve_dual_fed_network(U_nom, U_nom, L_total, df_loads, props["sigma"], sec)
        capex_ring, opex_ring = calc_network_cost("Anillo", L_total, df_loads, props["sigma"], sec, props["cost_factor"])
        
        # Visualización Comparativa
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.markdown("### 📊 Calidad Técnica (Tensión)")
            st.metric("V_min Radial", f"{min_v_radial:.2f} V")
            st.metric("V_min Anillo", f"{min_v_ring:.2f} V", delta=f"+{min_v_ring-min_v_radial:.2f} V")
            if min_v_ring > min_v_radial:
                st.success("El Anillo ofrece mejor tensión.")
        
        with col_res2:
            st.markdown("### 💰 Análisis Económico (J6)")
            # Gráfico Barras Apiladas
            cost_data = pd.DataFrame({
                "Topología": ["Radial", "Radial", "Anillo", "Anillo"],
                "Tipo Coste": ["CAPEX (Cable)", "OPEX (Pérdidas)", "CAPEX (Cable)", "OPEX (Pérdidas)"],
                "Valor (€)": [capex_rad, opex_rad, capex_ring, opex_ring]
            })
            
            fig_cost = px.bar(cost_data, x="Topología", y="Valor (€)", color="Tipo Coste", title="Coste Ciclo de Vida")
            st.plotly_chart(fig_cost, use_container_width=True)

# ==============================================================================
# BLOQUE PRINCIPAL APP
# ==============================================================================

def app():
    st.header("3. Network Topology & Dimensioning")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Radial (H)", 
        "2. Doble Alimentación (I)", 
        "3. Anillo (J)", 
        "4. Comparativa Técnica/Económica"
    ])

    with tab1:
        render_radial_tab()
    with tab2:
        render_dualfed_tab()
    with tab3:
        render_ring_tab()
    with tab4:
        render_comparison_tab()

# --- FIN DEL ARCHIVO ---