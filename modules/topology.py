import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
#week 3 attempt 2
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

def calculate_load_currents(p_kw, u_volts, cos_phi, phase_type):
    """
    Calculates Magnitude Current (I) and Active Current (I_active).
    
    Active Current (I_active = I * cos_phi) is used for Voltage Drop on resistive lines.
    Magnitude Current (I) is used for Ampacity (Heating) checks.
    
    Formulas:
    3-Phase: P = sqrt(3) * U * I * cos_phi  => I_active = P / (sqrt(3) * U) * (sqrt(3)?? No)
             I_active = I * cos_phi = P / (sqrt(3) * U) ? No.
             P = sqrt(3) * U * I_active_component_of_line_current? 
             Actually: Drop = sqrt(3) * R * (I * cos_phi).
             And I * cos_phi = (P_kw * 1000) / (sqrt(3) * U).
             
    1-Phase: P = U * I * cos_phi
             I_active = I * cos_phi = (P_kw * 1000) / U.
    """
    if u_volts == 0:
        return 0.0, 0.0
    
    p_watts = p_kw * 1000.0
    
    if phase_type == "Single-phase":
        # I = P / (U * cos_phi)
        i_mag = p_watts / (u_volts * cos_phi) if cos_phi > 0 else 0
        i_active = i_mag * cos_phi
    else:
        # I = P / (sqrt(3) * U * cos_phi)
        i_mag = p_watts / (np.sqrt(3) * u_volts * cos_phi) if cos_phi > 0 else 0
        i_active = i_mag * cos_phi
        
    return i_mag, i_active

def suggest_cross_section(moment_active_sum, u_source, max_drop_percent, sigma, phase_type):
    """
    Calculates required cross-section S based on max allowed voltage drop.
    Uses the Active Moment (Sum of P_k * L_k converted to Active Current moments).
    
    S = (K * Sum(I_active * L)) / (sigma * DeltaU)
    """
    delta_u_max = u_source * (max_drop_percent / 100.0)
    k = get_k_factor(phase_type)
    
    if delta_u_max == 0:
        return 0.0
    
    s_req = (k * moment_active_sum) / (sigma * delta_u_max)
    return s_req

def solve_radial_profile(u_source, nodes, sigma, section, phase_type):
    """
    Calculates voltage profile using Active Currents for drop and Total Currents for flow.
    """
    k = get_k_factor(phase_type)
    current_u = u_source
    
    # Initialize profile
    profile = [{'Distance': 0, 'Voltage': u_source, 'Drop_Seg': 0, 'Section_Current_Mag': 0}]
    
    # Sum of currents downstream
    total_active_current = sum(n['i_active'] for n in nodes)
    total_mag_current = sum(n['i_mag'] for n in nodes) # Conservative approximation (arithmetic sum)
    
    current_flow_active = total_active_current
    current_flow_mag = total_mag_current
    
    cum_dist = 0
    
    for n in nodes:
        # R = L / (sigma * S)
        # Drop = K * I_active * R (Assuming negligible reactance X)
        
        r_segment = n['dist_prev'] / (sigma * section)
        drop_segment = k * current_flow_active * r_segment
        
        current_u -= drop_segment
        cum_dist += n['dist_prev']
        
        profile.append({
            'Distance': cum_dist,
            'Voltage': current_u,
            'Drop_Seg': drop_segment,
            'Section_Current_Mag': current_flow_mag
        })
        
        current_flow_active -= n['i_active']
        current_flow_mag -= n['i_mag']
        
    return pd.DataFrame(profile)

def solve_dual_fed(u_a, u_b, l_total, nodes, sigma, section, phase_type):
    """
    Solves Dual-Fed network using Active Currents for distribution.
    """
    k = get_k_factor(phase_type)
    r_total = l_total / (sigma * section)
    
    # Calculate Moments based on Active Current (for Voltage Drop physics)
    moment_sum_active_a = sum(n['i_active'] * n['dist_source'] for n in nodes)
    sum_i_active = sum(n['i_active'] for n in nodes)
    
    # Calculate Ia (Active Component)
    # Iy = Sum(i_k * d_k) / L
    i_b_active_contrib = moment_sum_active_a / l_total
    
    # Circulating current (Active component due to voltage difference)
    i_circ_active = (u_a - u_b) / (k * r_total) if r_total > 0 else 0
    
    i_b_active = i_b_active_contrib - i_circ_active
    i_a_active = sum_i_active - i_b_active
    
    # For Ampacity checks, we'll estimate Magnitude based on similar ratio 
    # (Simplified: assuming power factors are similar, scalar ratio holds)
    sum_i_mag = sum(n['i_mag'] for n in nodes)
    ratio_mag = sum_i_mag / sum_i_active if sum_i_active != 0 else 1.0
    i_a_mag = i_a_active * ratio_mag
    i_b_mag = i_b_active * ratio_mag
    
    # Build Profile
    current_flow_active = i_a_active
    current_u = u_a
    min_u = u_a
    split_node_idx = -1
    
    profile = [{'Distance': 0, 'Voltage': u_a, 'Current_Flow_Mag': i_a_mag}]
    prev_dist = 0
    
    for idx, n in enumerate(nodes):
        dist_seg = n['dist_source'] - prev_dist
        r_seg = dist_seg / (sigma * section)
        
        # Drop based on Active Current
        drop = k * current_flow_active * r_seg
        current_u -= drop
        
        if current_u < min_u:
            min_u = current_u
            split_node_idx = idx
            
        current_flow_active -= n['i_active']
        # Estimate magnitude flow for plotting
        current_flow_mag = current_flow_active * ratio_mag 
        
        profile.append({
            'Distance': n['dist_source'],
            'Voltage': current_u,
            'Current_Flow_Mag': current_flow_mag
        })
        
        prev_dist = n['dist_source']
        
    # Final segment to B
    dist_seg = l_total - prev_dist
    r_seg = dist_seg / (sigma * section)
    current_u -= k * current_flow_active * r_seg
    
    profile.append({'Distance': l_total, 'Voltage': u_b, 'Current_Flow_Mag': i_b_mag})
    
    return pd.DataFrame(profile), i_a_mag, i_b_mag, min_u

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
    st.info("Define loads using **Active Power (kW)** and **Power Factor (Cos φ)**.")

    # Dynamic Data Editor for Loads
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
        
        derived_len = df_input["Dist_Prev (m)"].sum() if not df_input.empty else 0
        
        with c1:
            l_total = st.number_input("Total Line Length (m)", value=float(derived_len + 50), min_value=float(derived_len))
        
        if topology == "Dual-Fed (Two Sources)":
            with c2:
                u_b = st.number_input("Voltage Source B (V)", value=u_nom)
        else:
            u_b = u_nom
            st.caption(f"Ring Topology: Source A = Source B = {u_nom} V")

    # --- Processing & Calculation ---
    if st.button("Run Dimensioning & Analysis", type="primary"):
        if df_input.empty:
            st.error("Please add at least one load.")
            return

        # 1. Pre-process Data (Calculate I_mag and I_active)
        nodes = []
        cum_dist = 0
        
        for _, row in df_input.iterrows():
            d_prev = row["Dist_Prev (m)"]
            p_kw = row["Power (kW)"]
            cos_phi = row["Cos_Phi"]
            
            i_mag, i_active = calculate_load_currents(p_kw, u_nom, cos_phi, phase_type)
            
            cum_dist += d_prev
            nodes.append({
                'dist_prev': d_prev,
                'dist_source': cum_dist,
                'power': p_kw,
                'cos_phi': cos_phi,
                'i_mag': i_mag,
                'i_active': i_active
            })

        # Totals
        moment_active_sum = sum(n['i_active'] * n['dist_source'] for n in nodes)
        total_load_current_mag = sum(n['i_mag'] for n in nodes)

        # --- Tab 1: Sizing Recommendation ---
        st.divider()
        t1, t2 = st.tabs(["📐 Sizing & Dimensions", "📊 Analysis & Voltage Profile"])
        
        with t1:
            st.subheader("Cross-Section Sizing")
            st.markdown("Based on **Week 3 Formulas** (Eq. 8 & 9) using Active Currents.")
            
            req_section = suggest_cross_section(moment_active_sum, u_nom, max_drop, sigma, phase_type)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Load Current", f"{total_load_current_mag:.2f} A")
            c1.metric("Active Electrical Moment", f"{moment_active_sum/1000:.2f} kA·m")
            c2.metric("Target Max Drop", f"{max_drop}% ({u_nom * max_drop/100:.2f} V)")
            c3.metric("Calculated Min Section", f"{req_section:.2f} mm²", delta="Theoretical", delta_color="off")
            
            std_sections = [1.5, 2.5, 4, 6, 10, 16, 25, 35, 50, 70, 95, 120, 150, 185, 240]
            rec_std = next((s for s in std_sections if s >= req_section), std_sections[-1])
            
            st.markdown(f"**Recommendation:** Select standard cable **{rec_std} mm²**")
            selected_section = st.select_slider("Select Cross-Section for Analysis:", options=std_sections, value=rec_std)

        # --- Tab 2: Analysis ---
        with t2:
            st.subheader(f"Analysis: {topology}")
            
            if topology == "Radial":
                res_df = solve_radial_profile(u_nom, nodes, sigma, selected_section, phase_type)
                
                v_min = res_df['Voltage'].min()
                v_drop_pct = ((u_nom - v_min) / u_nom) * 100
                
                c_a, c_b = st.columns(2)
                c_a.metric("Min Voltage", f"{v_min:.2f} V")
                c_b.metric("Total Drop", f"{v_drop_pct:.2f}%", delta_color="inverse" if v_drop_pct > max_drop else "normal")
                
                fig = px.line(res_df, x='Distance', y='Voltage', markers=True, title=f"Voltage Profile (Radial, {selected_section} mm²)")
                fig.add_hline(y=u_nom * (1 - max_drop/100), line_dash="dash", line_color="red", annotation_text="Limit")
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                res_df, ia, ib, v_min = solve_dual_fed(u_nom, u_b, l_total, nodes, sigma, selected_section, phase_type)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Source A", f"{ia:.2f} A")
                c2.metric("Current Source B", f"{ib:.2f} A")
                c3.metric("Min Voltage Point", f"{v_min:.2f} V")
                
                if abs(ia) > selected_section * 5: 
                    st.warning(f"⚠️ Warning: Current Ia ({ia:.1f}A) might exceed ampacity for {selected_section}mm²")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res_df['Distance'], y=res_df['Voltage'], mode='lines+markers', name='Voltage', line=dict(color='#00ADB5', width=3)))
                fig.add_trace(go.Scatter(x=[0], y=[u_nom], mode='markers', marker=dict(size=12, color='red'), name='Source A'))
                fig.add_trace(go.Scatter(x=[l_total], y=[u_b], mode='markers', marker=dict(size=12, color='orange'), name='Source B'))
                fig.update_layout(title="Voltage Profile", xaxis_title="Distance (m)", yaxis_title="Voltage (V)")
                fig.add_hline(y=u_nom * (1 - max_drop/100), line_dash="dash", line_color="red", annotation_text="Limit")
                st.plotly_chart(fig, use_container_width=True)
                
                fig2 = px.area(res_df, x='Distance', y='Current_Flow_Mag', title="Current Flow Distribution (Approx Magnitude)")
                st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.caption("Reference: University of Almería - Week 3: Network Topology and Dimensioning")
