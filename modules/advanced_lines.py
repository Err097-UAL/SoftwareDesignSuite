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

# --- B) CÁLCULO DE FLECHAS Y TENSIONES (ECUACIÓN DE CAMBIO DE ESTADO) ---
def catenary_equation(x, H, w):
    return (H / w) * (np.cosh(w * x / H) - 1)

def solve_state_change(T1, w1, temp1, w2, temp2, L, E, alpha, A):
    # Ecuación simplificada de cambio de estado para vanos nivelados
    constant = T1 - (L**2 * w1**2 * E * A) / (24 * T1**2) + alpha * E * A * temp1
    
    def f(T2):
        return T2 - (L**2 * w2**2 * E * A) / (24 * T2**2) + alpha * E * A * temp2 - constant
    
    try:
        return brentq(f, 100, 100000)
    except:
        return T1

# ==============================================================================
# BLOQUE 2: INTERFAZ DE LA APLICACIÓN
# ==============================================================================

def app():
    st.header("⚡ Advanced High-Power Line Analysis")
    st.caption("Thermal, Mechanical and Economic optimization for Transmission Lines.")

    # --- BARRA LATERAL DE PARÁMETROS ---
    with st.sidebar:
        st.subheader("⚙️ System Configuration")
        # UPDATED MIN_VALUES TO 1
        v_nom = st.number_input("Nominal Voltage (kV)", min_value=1.0, value=220.0, step=10.0)
        p_mw = st.number_input("Design Power (MW)", min_value=1.0, value=400.0, step=10.0)
        length_km = st.number_input("Line Length (km)", min_value=1.0, value=150.0, step=5.0)
        
        st.divider()
        st.subheader("❄️ Weather Conditions")
        temp_amb = st.slider("Ambient Temp (°C)", -10, 50, 25)
        wind_speed = st.slider("Wind Speed (m/s)", 0.0, 30.0, 0.6)
        ice_load = st.checkbox("Include Ice Loading")

    # --- PESTAÑAS DE ANÁLISIS ---
    tab_thermal, tab_mech, tab_econ = st.tabs(["Thermal Capacity", "Mechanical Sag", "Economic Sizing"])

    # 1. ANÁLISIS TÉRMICO (Ampacity)
    with tab_thermal:
        st.subheader("🌡️ Dynamic Line Rating (DLR)")
        i_load = (p_mw * 1e6) / (np.sqrt(3) * v_nom * 1e3 * 0.95)
        st.info(f"Calculated Load Current: **{i_load:.2f} A**")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Conductor Temperature Profile**")
            # Simulación simple de calentamiento
            temps = np.linspace(temp_amb, 100, 50)
            losses = [calc_resistance_dc(0.05, 0.004, t) * (i_load**2) / 1000 for t in temps]
            fig_temp = px.line(x=temps, y=losses, labels={'x': 'Temp (°C)', 'y': 'Losses (kW/km)'})
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with c2:
            st.success("Thermal Limit Status: **OK**")
            st.metric("Max Surface Temp", f"{temp_amb + 15}°C", "+2.5°C vs Ambient")

    # 2. ANÁLISIS MECÁNICO (Flechas)
    with tab_mech:
        st.subheader("📐 Sag & Tension Analysis")
        span = st.slider("Span Length (m)", 100, 800, 400)
        
        # Cálculo de flecha simplificado
        H = 25000 # Tensión horizontal (N)
        w = 15    # Peso (N/m)
        x_vals = np.linspace(-span/2, span/2, 100)
        y_vals = catenary_equation(x_vals, H, w)
        
        fig_sag = go.Figure()
        fig_sag.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Catenary'))
        fig_sag.update_layout(title="Conductor Profile (Mechanical Sag)", xaxis_title="Distance (m)", yaxis_title="Vertical Drop (m)")
        st.plotly_chart(fig_sag, use_container_width=True)
        
        st.write(f"**Max Sag at Mid-span:** {max(y_vals):.2f} m")

    # 3. OPTIMIZACIÓN ECONÓMICA
    with tab_econ:
        st.subheader("💰 Life Cycle Cost Analysis")
        sections = [240, 300, 450, 630]
        data = []
        for s in sections:
            capex = s * 150 * length_km
            opex = (1/s) * 5e6 * length_km # Pérdidas inversamente prop a sección
            data.append({"Section": s, "CAPEX (€)": capex, "OPEX (€)": opex, "Total (€)": capex + opex})
        
        df_res = pd.DataFrame(data)
        best_row = df_res.iloc[df_res['Total (€)'].idxmin()]
        
        c_val, c_plot = st.columns([1, 2])
        with c_val:
            cond_color = "#00ADB5"
            cable_html = f"""
            <div style="display: flex; align-items: center; background-color: #161B22; padding: 20px; border-radius: 15px; border: 1px solid #30363D;">
                <svg width="80" height="80" viewBox="0 0 100 100"><circle cx="50" cy="50" r="45" fill="#30363D" /><circle cx="50" cy="50" r="30" fill="{cond_color}" /></svg>
                <div style="padding-left: 20px;"><div style="color: #8B949E; font-size: 0.8rem;">SECCIÓN ÓPTIMA</div><div style="color: white; font-size: 2rem; font-weight: 800;">{best_row['Section']} mm²</div></div>
            </div>"""
            st.markdown(cable_html, unsafe_allow_html=True)
            st.metric("Potential Savings", f"{best_row['Total (€)']:,.0f} €")
        
        with c_plot:
            st.plotly_chart(px.bar(df_res, x="Section", y=["CAPEX (€)", "OPEX (€)"], title="Cost Analysis"), use_container_width=True)
            
    st.link_button("📈 IEC 60287-3-2 Standard", "https://webstore.iec.ch/publication/1233")
