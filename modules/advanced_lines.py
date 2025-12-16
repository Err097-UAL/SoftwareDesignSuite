def render_voltage_drop_tab():
    st.subheader("Algoritmo de comprobación de validez del REBT a partir de la Caída de Tensión (Método Blondel)")
    
    with st.expander("📖 Teoría de la Caída de Tensión"):
        st.markdown("Para líneas de transporte con impedancia $Z = R + jX$, la caída de tensión trifásica se calcula como:")
        st.latex(r"\Delta U = \sqrt{3} \cdot L \cdot I \cdot (r \cdot \cos \phi + x \cdot \sin \phi)")
        st.write("Donde $r$ y $x$ son las resistencia y reactancia unitarias ($\Omega/km$).")

    # Corrected: Use 'value' to set default, allowing lower inputs if needed
    V_nom = st.number_input("Tensión Nominal (V)", value=400.0)

    c1, c2, c3 = st.columns(3)
    
    # UPDATE: Set default to 150, but hard limit (min_value) to 5
    L = c1.number_input("Longitud (m)", value=150.0, min_value=5.0)
    
    # Corrected: Use 'value' to prevent 40.0 from becoming the minimum
    P = c2.number_input("Potencia (kW)", value=40.0)
    
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
