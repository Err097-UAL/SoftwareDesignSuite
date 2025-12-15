import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==============================================================================
# HELPER FUNCTIONS: ELECTRICAL CALCULATIONS
# ==============================================================================

def get_k_factor(phase_type):
    """
    Returns the K factor for voltage drop calculations.
    Single-phase (Monofásica): 2
    Three-phase (Trifásica): sqrt(3) -> approx 1.732
    """
    return 2.0 if phase_type == "Single-phase" else np.sqrt(3)

def calculate_current(p_kw, u_volts, cos_phi, phase_type):
    """
    Calculates Current (I) from Power (P in kW).
    I = (P * 1000) / (U * cos_phi)        [Single-phase]
    I = (P * 1000) / (sqrt(3) * U * cos_phi) [Three-phase]
    """
    if cos_phi == 0 or u_volts == 0:
        return 0.0
    if phase_type == "Single-phase":
        return (p_kw * 1000) / (u_volts * cos_phi)
    else:
        return (p_kw * 1000) / (np.sqrt(3) * u_volts * cos_phi)

def suggest_cross_section(moment_sum, u_source, max_drop_percent, sigma, phase_type, cos_phi_avg=1.0):
    """
    Calculates required cross-section S based on max allowed voltage drop.
    Using PDF Eq (8) and (9) generalized for moments:
    S = (K * Sum(I*L) * cos_phi) / (sigma * DeltaU)
    Note: The PDF formulas (8/9) include cos_phi in the numerator for sizing.
    """
    delta_u_max = u_source * (max_drop_percent / 100.0)
    k = get_k_factor(phase_type)
    
    # Formula: S = (K * Moment_Sum * cos_phi) / (sigma * delta_u)
    # If standard moments method (resistive approx) is used, cos_phi might be omitted, 
    # but PDF explicitly includes it in Eq 8 & 9.
    if delta_u_max == 0:
        return 0.0
    
    s_req = (k * moment_sum * cos_phi_avg) / (sigma * delta_u_max)
    return s_req

def solve_radial_profile(u_source, nodes, sigma, section, phase_type):
    """
    Calculates voltage profile segment by segment.
    nodes: List of dicts with 'dist_prev', 'current', 'cos_phi', 'dist_source'
    """
    k = get_k_factor(phase_type)
    current_u = u_source
    profile = [{'Distance': 0, 'Voltage': u_source, 'Drop_Seg': 0, 'Section_Current': 0}]
    
    # Calculate total current flowing through first segment
    total_current = sum(n['current'] for n in nodes)
    current_flow = total_current
    
    cum_dist = 0
    
    for n in nodes:
        # R = L / (sigma * S)
        # DeltaU = K * I * R * cos_phi (Approximation using active component)
        # Using simple resistive drop K*I*R is standard for these approximations unless X is given.
        # PDF Eq 8 implies DeltaU = (K d I cos_phi) / (sigma S).
        
        r_segment = n['dist_prev'] / (sigma * section)
        drop_segment = (k * current_flow * r_segment * n['cos_phi']) # Using specific cos_phi of the load/segment
        
        current_u -= drop_segment
        cum_dist += n['dist_prev']
        
        profile.append({
            'Distance': cum_dist,
            'Voltage': current_u,
            'Drop_Seg': drop_segment,
            'Section_Current': current_flow
        })
        
        current_flow -= n['current']
        
    return pd.DataFrame(profile)

def solve_dual_fed(u_a, u_b, l_total, nodes, sigma, section, phase_type):
    """
    Solves Dual-Fed network (PDF Eq 12 & 13).
    Returns profile, Ia, Ib, and split point info.
    """
    k = get_k_factor(phase_type)
    
    # Calculate moments from Source A
    moment_sum_a = sum(n['current'] * n['dist_source'] for n in nodes)
    sum_currents = sum(n['current'] for n in nodes)
    
    # Resistance of total line
    r_total = l_total / (sigma * section)
    
    # Calculate Ia (Eq 12 modified for unequal voltages if needed)
    # PDF Eq 12: Iy = Sum(i*d)/d (This calculates contribution from B if d is dist from A)
    # Actually, standard formula: Ia = [Sum(I*(L-x)) + (Ua-Ub)/Z] / L  <-- This is confusing in text.
    # Let's use the PDF Eq 12 interpretation:
    # Iy = (Sum i_k * d_k) / d  (where d_k is distance from A, this calculates Current from B theoretically if Ua=Ub)
    # Then Ix = Sum(i_k) - Iy
    
    i_b_contribution = moment_sum_a / l_total # Pure load contribution to Source B current
    
    # Circulating current due to voltage diff: I_circ = (Ua - Ub) / (K * R_total) ?? 
    # Usually drop = I*R. Here we use K factor. 
    # Let's stick to simple superposition.
    
    # If Ua != Ub, we add the circulating term.
    # Voltage drop across line due to circulating current: Ua - Ub = K * I_circ * R_total
    # I_circ = (Ua - Ub) / (K * R_total)
    i_circ = (u_a - u_b) / (k * r_total) if r_total > 0 else 0
    
    i_b = i_b_contribution - i_circ
    i_a = sum_currents - i_b
    
    # Determine Split Point (Point of Minimum Voltage)
    # We walk from A. Current decreases by load I at each node.
    # The point where current crosses zero (changes direction) is the split point.
    
    current_flow = i_a
    current_u = u_a
    min_u = u_a
    split_node_idx = -1
    
    profile = [{'Distance': 0, 'Voltage': u_a, 'Current_Flow': i_a}]
    
    prev_dist = 0
    
    for idx, n in enumerate(nodes):
        dist_seg = n['dist_source'] - prev_dist
        r_seg = dist_seg / (sigma * section)
        
        # Drop = K * I * R * cos_phi
        drop = k * current_flow * r_seg * n['cos_phi']
        current_u -= drop
        
        if current_u < min_u:
            min_u = current_u
            split_node_idx = idx
            
        profile.append({
            'Distance': n['dist_source'],
            'Voltage': current_u,
            'Current_Flow': current_flow
        })
        
        prev_dist = n['dist_source']
        current_flow -= n['current']
        
    # Final segment to B
    dist_seg = l_total - prev_dist
    r_seg = dist_seg / (sigma * section)
    # Note: flow here should be equal to -i_b ideally
    current_u -= k * current_flow * r_seg * 1.0 # Assuming cos=1 for line or prev load
    
    profile.append({'Distance': l_total, 'Voltage': u_b, 'Current_Flow': current_flow})
    
    return pd.DataFrame(profile), i_a, i_b, min_u

# ==============================================================================
# UI COMPONENTS
# ==============================================================================

def app():
    st.title("Week 3: Network Topology & Dimensioning")
    
    # --- Sidebar: Global Configuration ---
    with st.sidebar:
        st.header("Global Parameters")
        topology = st.selectbox("Topology Type", ["Radial", "Dual-Fed (Two Sources)", "Ring (Closed Loop)"])
        
        st.subheader("System Specs")
        u_nom = st.number_input("Source Voltage (V)", value=400.0, step=10.0)
        f_hz = st.number_input("Frequency (Hz)", value=50.0)
        phase_type = st.radio("System Type", ["Three-phase", "Single-phase"])
        
        st.subheader("Cabling")
        material = st.selectbox("Conductor Material", ["Copper (Cu)", "Aluminum (Al)"])
        sigma = 56.0 if material == "Copper (Cu)" else 35.0
        st.caption(f"Conductivity (σ): {sigma} m/(Ω·mm²)")
        
        max_drop = st.slider("Max Voltage Drop (%)", 0.5, 10.0, 5.0)

    # --- Main Input Area ---
    st.markdown("### 1. Network Configuration")
    st.info("Define loads. For 'Distance', enter the length from the **previous node** (or source for the first load).")

    # Dynamic Data Editor for Loads
    # Default data structure
    default_data = [
        {"Dist_Prev (m)": 50, "Power (kW)": 15, "Cos_Phi": 0.9},
        {"Dist_Prev (m)": 100, "Power (kW)": 25, "Cos_Phi": 0.85},
        {"Dist_Prev (m)": 80, "Power (kW)": 10, "Cos_Phi": 0.95}
    ]
    
    df_input = st.data_editor(
        pd.DataFrame(default_data),
        num_rows="dynamic",
        column_config={
            "Dist_Prev (m)": st.column_config.NumberColumn("Dist from Prev (m)", min_value=1, format="%d m"),
            "Power (kW)": st.column_config.NumberColumn("Active Power (kW)", min_value=0),
            "Cos_Phi": st.column_config.NumberColumn("Power Factor", min_value=0.1, max_value=1.0, step=0.01)
        }
    )

    # Additional Inputs based on Topology
    l_total = 0
    u_b = u_nom
    
    if topology in ["Dual-Fed (Two Sources)", "Ring (Closed Loop)"]:
        st.markdown("### 2. Dual-Feed / Ring Parameters")
        c1, c2 = st.columns(2)
        
        # Calculate derived total length from nodes to check consistency
        derived_len = df_input["Dist_Prev (m)"].sum() if not df_input.empty else 0
        
        with c1:
            l_total = st.number_input("Total Line Length (m)", value=float(derived_len + 50), min_value=float(derived_len))
            st.caption(f"Note: Sum of load distances is {derived_len} m. Total length must be >= this.")
        
        if topology == "Dual-Fed (Two Sources)":
            with c2:
                u_b = st.number_input("Voltage Source B (V)", value=u_nom)
        else:
            # Ring: UA = UB
            u_b = u_nom
            st.caption(f"Ring Topology: Source A = Source B = {u_nom} V")

    # --- Processing & Calculation ---
    if st.button("Run Dimensioning & Analysis", type="primary"):
        if df_input.empty:
            st.error("Please add at least one load.")
            return

        # 1. Pre-process Data
        # Calculate Currents and Cumulative Distances
        nodes = []
        cum_dist = 0
        
        for _, row in df_input.iterrows():
            d_prev = row["Dist_Prev (m)"]
            p_kw = row["Power (kW)"]
            cos_phi = row["Cos_Phi"]
            
            # Calculate Current
            i_load = calculate_current(p_kw, u_nom, cos_phi, phase_type)
            
            cum_dist += d_prev
            nodes.append({
                'dist_prev': d_prev,
                'dist_source': cum_dist, # dk
                'power': p_kw,
                'cos_phi': cos_phi,
                'current': i_load
            })

        # Calculate Totals for Moments
        moment_sum = sum(n['current'] * n['dist_source'] for n in nodes) # Sum(i_k * L_k)
        total_load_current = sum(n['current'] for n in nodes)
        weighted_cos_phi = np.average([n['cos_phi'] for n in nodes], weights=[n['current'] for n in nodes]) if nodes else 0.9

        # --- Tab 1: Sizing Recommendation (Eq 8/9) ---
        st.divider()
        t1, t2 = st.tabs(["📐 Sizing & Dimensions", "📊 Analysis & Voltage Profile"])
        
        with t1:
            st.subheader("Cross-Section Sizing")
            st.markdown("Based on **Week 3 Formulas** (Eq. 8 & 9) and Electrical Moments.")
            
            # Calculate required section based on Max Drop
            req_section = suggest_cross_section(moment_sum, u_nom, max_drop, sigma, phase_type, weighted_cos_phi)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Load Current", f"{total_load_current:.2f} A")
            c1.metric("Electrical Moment", f"{moment_sum/1000:.2f} kA·m")
            c2.metric("Target Max Drop", f"{max_drop}% ({u_nom * max_drop/100:.2f} V)")
            c3.metric("Calculated Min Section", f"{req_section:.2f} mm²", delta="Theoretical", delta_color="off")
            
            # Standard Section Selector
            std_sections = [1.5, 2.5, 4, 6, 10, 16, 25, 35, 50, 70, 95, 120, 150, 185, 240]
            
            # Find closest standard upper
            rec_std = next((s for s in std_sections if s >= req_section), std_sections[-1])
            
            st.markdown(f"**Recommendation:** Select standard cable **{rec_std} mm²**")
            selected_section = st.select_slider("Select Cross-Section for Analysis:", options=std_sections, value=rec_std)

        # --- Tab 2: Analysis ---
        with t2:
            st.subheader(f"Analysis: {topology}")
            
            if topology == "Radial":
                # Radial Analysis
                res_df = solve_radial_profile(u_nom, nodes, sigma, selected_section, phase_type)
                
                # Metrics
                v_min = res_df['Voltage'].min()
                v_drop_abs = u_nom - v_min
                v_drop_pct = (v_drop_abs / u_nom) * 100
                
                c_a, c_b = st.columns(2)
                c_a.metric("Min Voltage", f"{v_min:.2f} V")
                c_b.metric("Total Drop", f"{v_drop_pct:.2f}%", delta_color="inverse" if v_drop_pct > max_drop else "normal")
                
                # Plot
                fig = px.line(res_df, x='Distance', y='Voltage', markers=True, title=f"Voltage Profile (Radial, {selected_section} mm²)")
                fig.add_hline(y=u_nom * (1 - max_drop/100), line_dash="dash", line_color="red", annotation_text="Limit")
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Dual-Fed / Ring Analysis
                res_df, ia, ib, v_min = solve_dual_fed(u_nom, u_b, l_total, nodes, sigma, selected_section, phase_type)
                
                v_drop_abs = u_nom - v_min
                v_drop_pct = (v_drop_abs / u_nom) * 100
                
                # Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Source A (Ia)", f"{ia:.2f} A")
                c2.metric("Current Source B (Ib)", f"{ib:.2f} A")
                c3.metric("Min Voltage Point", f"{v_min:.2f} V")
                
                if abs(ia) > selected_section * 5: # Crude ampacity check
                    st.warning(f"⚠️ Warning: Current Ia ({ia:.1f}A) might exceed ampacity for {selected_section}mm²")

                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res_df['Distance'], y=res_df['Voltage'], mode='lines+markers', name='Voltage Profile', line=dict(color='#00ADB5', width=3)))
                
                # Add source markers
                fig.add_trace(go.Scatter(x=[0], y=[u_nom], mode='markers', marker=dict(size=12, color='red'), name='Source A'))
                fig.add_trace(go.Scatter(x=[l_total], y=[u_b], mode='markers', marker=dict(size=12, color='orange'), name='Source B'))
                
                fig.update_layout(title="Voltage Profile (Dual-Fed / Ring)", xaxis_title="Distance (m)", yaxis_title="Voltage (V)")
                fig.add_hline(y=u_nom * (1 - max_drop/100), line_dash="dash", line_color="red", annotation_text="Limit")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Current Distribution Plot
                fig2 = px.area(res_df, x='Distance', y='Current_Flow', title="Current Flow Distribution")
                st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.caption("Reference: University of Almería - Week 3: Network Topology and Dimensioning")
