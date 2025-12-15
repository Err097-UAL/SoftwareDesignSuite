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

def calculate_load_currents(p_kw, u_volts, cos_phi, phase_type):
    """
    Calculates Magnitude Current (I) and Active Current (I_active).
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
    Modified to ensure profile reflects step-changes in current at loads for plotting.
    """
    k = get_k_factor(phase_type)
    current_u = u_source
    
    # Calculate totals
    total_active_current = sum(n['i_active'] for n in nodes)
    total_mag_current = sum(n['i_mag'] for n in nodes) 
    
    current_flow_active = total_active_current
    current_flow_mag = total_mag_current
    
    # Initialize profile with starting conditions at Distance 0
    # Current in the first segment is the Total Load Current
    profile = [{'Distance': 0, 'Voltage': u_source, 'Drop_Seg': 0, 'Section_Current_Mag': current_flow_mag}]
    
    cum_dist = 0
    
    for n in nodes:
        r_segment = n['dist_prev'] / (sigma * section)
        # Drop is calculated based on flow in the segment arriving at this node
        drop_segment = k * current_flow_active * r_segment
        
        current_u -= drop_segment
        cum_dist += n['dist_prev']
        
        # Decrement current (load is tapped here)
        # The profile point at this node represents the state starting the NEXT segment
        current_flow_active -= n['i_active']
        current_flow_mag -= n['i_mag']
        
        profile.append({
            'Distance': cum_dist,
            'Voltage': current_u,
            'Drop_Seg': drop_segment,
            'Section_Current_Mag': current_flow_mag
        })
        
    return pd.DataFrame(profile)

def solve_dual_fed(u_a, u_b, l_total, nodes, sigma, section, phase_type):
    """
    Solves Dual-Fed network using Active Currents for distribution.
    """
    k = get_k_factor(phase_type)
    r_total = l_total / (sigma * section)
    
    # Calculate Moments based on Active Current
    moment_sum_active_a = sum(n['i_active'] * n['dist_source'] for n in nodes)
    sum_i_active = sum(n['i_active'] for n in nodes)
    
    # Calculate Ia (Active Component)
    i_b_active_contrib = moment_sum_active_a / l_total
    i_circ_active = (u_a - u_b) / (k * r_total) if r_total > 0 else 0
    
    i_b_active = i_b_active_contrib - i_circ_active
    i_a_active = sum_i_active - i_b_active
    
    # Estimate Magnitude
    sum_i_mag = sum(n['i_mag'] for n in nodes)
    ratio_mag = sum_i_mag / sum_i_active if sum_i_active != 0 else 1.0
    i_a_mag = i_a_active * ratio_mag
    i_b_mag = i_b_active * ratio_mag
    
    # Build Profile
    current_flow_active = i_a_active
    current_u = u_a
    min_u = u_a
    
    # Initial Point
    profile = [{'Distance': 0, 'Voltage': u_a, 'Current_Flow_Mag': i_a_mag}]
    prev_dist = 0
    
    # Track magnitude for plot continuity
    current_flow_mag = i_a_mag 

    for idx, n in enumerate(nodes):
        dist_seg = n['dist_source'] - prev_dist
        r_seg = dist_seg / (sigma * section)
        
        drop = k * current_flow_active * r_seg
        current_u -= drop
        
        if current_u < min_u:
            min_u = current_u
            
        # Update Flow for next segment
        current_flow_active -= n['i_active']
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
    
    # The final point needs to reflect the current state arriving at B
    profile.append({'Distance': l_total, 'Voltage': u_b, 'Current_Flow_Mag': current_flow_mag})
    
    return pd.DataFrame(profile), i_a_mag, i_b_mag, min_u

# ==============================================================================
# HELPER FUNCTIONS: VISUALIZATION
# ==============================================================================

def draw_ring_schematic_circular(nodes, l_total, u_a):
    """
    Generates a Circular Diagram for Ring Topology with Source A at top.
    """
    fig = go.Figure()
    radius = 1.0

    # 1. Draw the Ring (Circle)
    theta = np.linspace(0, 2*np.pi, 100)
    x_ring = radius * np.cos(theta)
    y_ring = radius * np.sin(theta)

    fig.add_trace(go.Scatter(
        x=x_ring, y=y_ring,
        mode='lines',
        line=dict(color='black', width=3),
        name='Ring Line',
        hoverinfo='skip'
    ))

    # 2. Source A (At top: 90 degrees or pi/2)
    fig.add_trace(go.Scatter(
        x=[0], y=[radius],
        mode='markers+text',
        marker=dict(symbol='square', size=25, color='#FF5733', line=dict(color='black', width=2)),
        text=[f"<b>Source A</b><br>{u_a}V"],
        textposition='top center',
        name='Source A'
    ))

    # 3. Loads
    # Map distance to angle: Start at Top (pi/2) and go Clockwise
    node_x = []
    node_y = []
    text_x = []
    text_y = []
    node_text = []

    for i, n in enumerate(nodes):
        # Calculate angle
        angle = (np.pi / 2) - (n['dist_source'] / l_total) * (2 * np.pi)
        
        # Node coordinates
        nx = radius * np.cos(angle)
        ny = radius * np.sin(angle)
        node_x.append(nx)
        node_y.append(ny)
        
        # Annotation coordinates (slightly outside the circle)
        tx = (radius + 0.2) * np.cos(angle)
        ty = (radius + 0.2) * np.sin(angle)
        text_x.append(tx)
        text_y.append(ty)
        
        node_text.append(f"<b>L{i+1}</b><br>{n['power']}kW")

    # Draw Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(symbol='triangle-down', size=15, color='#00ADB5', line=dict(color='black', width=1)),
        hoverinfo='text',
        hovertext=[f"Load {i+1}: {n['power']}kW" for i, n in enumerate(nodes)],
        name='Loads'
    ))
    
    # Add Text Annotations
    for i in range(len(nodes)):
        fig.add_annotation(
            x=text_x[i], y=text_y[i],
            text=node_text[i],
            showarrow=False,
            font=dict(size=10, color="#333")
        )

    # Layout
    fig.update_layout(
        title="Ring Topology (Circular Schematic)",
        xaxis=dict(visible=False, range=[-1.5, 1.5], scaleanchor="y"),
        yaxis=dict(visible=False, range=[-1.5, 1.5]),
        height=400,
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    return fig

def draw_network_schematic(topology, nodes, l_total, u_a, u_b):
    """
    Generates a Unifilar Diagram (Schematic) of the network.
    For Ring networks, it shows Source A at both ends.
    """
    fig = go.Figure()
    
    # Define End Point
    if topology == "Radial networks":
        max_dist = nodes[-1]['dist_source'] if nodes else 100
    else:
        max_dist = l_total if l_total > 0 else 100

    # 1. Main Feeder Line (Bus)
    fig.add_trace(go.Scatter(
        x=[0, max_dist], y=[0, 0],
        mode='lines',
        line=dict(color='black', width=5),
        name='Main Line',
        hoverinfo='skip'
    ))

    # 2. Source A (Start)
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers+text',
        marker=dict(symbol='square', size=25, color='#FF5733', line=dict(color='black', width=2)),
        text=[f"<b>Source A (Start)</b><br>{u_a}V"],
        textposition='top center',
        name='Source A'
    ))

    # 3. Source B / End Point (if applicable)
    if topology != "Radial networks":
        # Handle label for Ring vs Dual-Fed
        if topology == "Ring networks":
            label = f"<b>Source A (End)</b><br>{u_a}V" # Same source at end
            color = '#FF5733' # Same color as Source A
            name = "Source A (End)"
        else:
            label = f"<b>Source B</b><br>{u_b}V"
            color = '#FFC300'
            name = "Source B"

        fig.add_trace(go.Scatter(
            x=[max_dist], y=[0],
            mode='markers+text',
            marker=dict(symbol='square', size=25, color=color, line=dict(color='black', width=2)),
            text=[label],
            textposition='top center',
            name=name
        ))

    # 4. Loads (Schematic representation)
    node_x = [n['dist_source'] for n in nodes]
    node_y = [0] * len(nodes)
    
    # Load Markers on the line
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(size=10, color='black'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Vertical lines for loads (Standard unifilar style)
    for i, n in enumerate(nodes):
        fig.add_shape(
            type="line",
            x0=n['dist_source'], y0=0, 
            x1=n['dist_source'], y1=-0.5,
            line=dict(color="black", width=2)
        )
        
        fig.add_trace(go.Scatter(
            x=[n['dist_source']], y=[-0.5],
            mode='markers',
            marker=dict(symbol='triangle-down', size=15, color='#00ADB5'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_annotation(
            x=n['dist_source'], y=-0.6,
            text=f"<b>L{i+1}</b><br>{n['power']} kW<br>cosφ {n['cos_phi']}",
            showarrow=False,
            yanchor="top",
            font=dict(size=10, color="#333")
        )

    title_text = "Network Schematic (Linear View)"
    if topology == "Ring networks":
        title_text = "Ring Topology (Linear Representation - Unfolded)"

    fig.update_layout(
        title=title_text,
        xaxis=dict(title="Distance (m)", range=[-max_dist*0.1, max_dist*1.1], showgrid=False, zeroline=False),
        yaxis=dict(visible=False, range=[-1.5, 1]),
        height=350,
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# ==============================================================================
# UI COMPONENTS
# ==============================================================================

def app():
    st.title("Week 3: Network Topology & Dimensioning")
    
    # --- Configuration Parameters (At the top of the page) ---
    with st.expander("⚙️ Network & System Parameters", expanded=True):
        st.markdown("### Global Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Topology")
            topology = st.selectbox("Topology Type", [
                "Radial networks", 
                "Networks fed from both ends", 
                "Ring networks", 
                "Meshed networks"
            ], key="topology_selector")
            
            phase_type = st.radio("System Type", ["Three-phase", "Single-phase"])

        with col2:
            st.subheader("System Specs")
            u_nom = st.number_input("Source Voltage (V)", value=400.0, step=10.0)
            f_hz = st.number_input("Frequency (Hz)", value=50.0)
            
            st.subheader("Cabling")
            material = st.selectbox("Conductor Material", ["Copper (Cu)", "Aluminum (Al)"])
            sigma = 56.0 if material == "Copper (Cu)" else 35.0
            st.caption(f"Conductivity (σ): {sigma} m/(Ω·mm²)")
            
        with col3:
            st.subheader("Dimensioning")
            max_drop = st.slider("Max Voltage Drop (%)", 0.5, 10.0, 5.0)
            
            # Standard Sections
            st.markdown("**Cable Selection**")
            std_sections = [1.5, 2.5, 4, 6, 10, 16, 25, 35, 50, 70, 95, 120, 150, 185, 240]
            selected_section = st.selectbox("Cross-Section (mm²)", options=std_sections, index=5)
            st.info(f"Selected: {selected_section} mm²")

    # --- Main Input Area ---
    st.markdown("### 1. Network Configuration")
    
    if topology == "Meshed networks":
        st.info("Define nodes and branches for the meshed network (conceptual input).")
    else:
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
    
    if topology in ["Networks fed from both ends", "Ring networks"]:
        st.markdown("### 2. Dual-Feed / Ring Parameters")
        c1, c2 = st.columns(2)
        
        derived_len = df_input["Dist_Prev (m)"].sum() if not df_input.empty else 0
        
        with c1:
            l_total = st.number_input("Total Line Length (m)", value=float(derived_len + 50), min_value=float(derived_len))
        
        if topology == "Networks fed from both ends":
            with c2:
                u_b = st.number_input("Voltage Source B (V)", value=u_nom)
        else:
            u_b = u_nom
            st.caption(f"Ring Topology: Source A = Source B = {u_nom} V")

    # --- Processing & Calculation ---
    if st.button("Run Dimensioning & Analysis", type="primary"):
        # Special Check for Meshed Networks
        if topology == "Meshed networks":
            st.warning("This analysis involves more complex software engineering and has been left out to ensure proper code compilation.")
            return

        if df_input.empty:
            st.error("Please add at least one load.")
            return

        # 1. Pre-process Data
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

        # --- Tab 1: Sizing Verification ---
        st.divider()
        t1, t2 = st.tabs(["📐 Sizing Check", "📊 Analysis & Diagrams"])
        
        with t1:
            st.subheader("Cross-Section Verification")
            st.markdown("Comparison between **Selected Section** (Top Panel) and **Theoretical Minimum** (Eq. 8 & 9).")
            
            req_section = suggest_cross_section(moment_active_sum, u_nom, max_drop, sigma, phase_type)
            rec_std = next((s for s in std_sections if s >= req_section), std_sections[-1])
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Selected Section", f"{selected_section} mm²")
            c2.metric("Required Minimum", f"{req_section:.2f} mm²", help="Theoretical value to meet Max Voltage Drop exactly")
            
            if selected_section >= req_section:
                 c3.success(f"✅ Compliant (Rec: {rec_std} mm²)")
            else:
                 c3.error(f"❌ Undersized (Rec: {rec_std} mm²)")
            
            st.markdown(f"""
            **Electrical Details:**
            - **Total Active Moment:** {moment_active_sum/1000:.2f} kA·m
            - **Total Load Current:** {total_load_current_mag:.2f} A
            - **Target Drop:** {max_drop}% ({u_nom * max_drop/100:.2f} V)
            """)

        # --- Tab 2: Analysis ---
        with t2:
            st.subheader(f"Analysis: {topology}")
            
            # DIAGRAM GENERATION LOGIC
            # If Ring: Show both Circular and Linear (with Source A at both ends)
            if topology == "Ring networks":
                st.write("#### 1. Circular Diagram (Ring)")
                circ_fig = draw_ring_schematic_circular(nodes, l_total, u_nom)
                st.plotly_chart(circ_fig, use_container_width=True, key="ring_circ_diag")
                
                st.write("#### 2. Linear Unfolded Diagram (Ring)")
                # Passes u_nom for both u_a and u_b, allowing draw_network_schematic to handle the labels
                linear_fig = draw_network_schematic(topology, nodes, l_total, u_nom, u_nom)
                st.plotly_chart(linear_fig, use_container_width=True, key="ring_linear_diag")
            else:
                # Radial or Dual-Fed
                schematic_fig = draw_network_schematic(topology, nodes, l_total, u_nom, u_b)
                st.plotly_chart(schematic_fig, use_container_width=True, key="std_schematic")
            
            if topology == "Radial networks":
                res_df = solve_radial_profile(u_nom, nodes, sigma, selected_section, phase_type)
                
                v_min = res_df['Voltage'].min()
                v_drop_pct = ((u_nom - v_min) / u_nom) * 100
                
                c_a, c_b = st.columns(2)
                c_a.metric("Min Voltage", f"{v_min:.2f} V")
                c_b.metric("Total Drop", f"{v_drop_pct:.2f}%", delta_color="inverse" if v_drop_pct > max_drop else "normal")
                
                fig = px.line(res_df, x='Distance', y='Voltage', markers=True, title=f"Voltage Profile (Radial, {selected_section} mm²)")
                fig.add_hline(y=u_nom * (1 - max_drop/100), line_dash="dash", line_color="red", annotation_text="Limit")
                st.plotly_chart(fig, use_container_width=True, key="radial_volt_prof")
                
            else:
                # Common logic for Dual-Fed and Ring
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
                # Handle plot marker for Source B/End
                end_name = "Source A (End)" if topology == "Ring networks" else "Source B"
                end_color = 'red' if topology == "Ring networks" else 'orange'
                fig.add_trace(go.Scatter(x=[l_total], y=[u_b], mode='markers', marker=dict(size=12, color=end_color), name=end_name))
                
                fig.update_layout(title="Voltage Profile", xaxis_title="Distance (m)", yaxis_title="Voltage (V)")
                fig.add_hline(y=u_nom * (1 - max_drop/100), line_dash="dash", line_color="red", annotation_text="Limit")
                st.plotly_chart(fig, use_container_width=True, key="dual_volt_prof")
                
                # Updated Current Distribution Plot
                fig2 = px.area(res_df, x='Distance', y='Current_Flow_Mag', title="Current Flow Distribution (Approx Magnitude)")
                fig2.update_traces(line_shape='hv')
                st.plotly_chart(fig2, use_container_width=True, key="curr_dist_prof")

    st.markdown("---")
    st.caption("Reference: University of Almería - Week 3: Network Topology and Dimensioning")
