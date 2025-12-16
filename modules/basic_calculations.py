# [INDENTATION GUIDE]
    # The 'with' statement below should have 4 spaces of indentation.
    # The content inside it should have 8 spaces.

    with tab_insulation:
        st.subheader("Comparativa Técnica: PVC vs XLPE")
        
        col_text, col_plot = st.columns([1, 1.5])
        
        with col_text:
            st.info("💡 **Contexto:** La elección del aislamiento del cable es crucial " \
                    "para la seguridad y eficiencia de las instalaciones eléctricas.")
            
            st.markdown("""
            ### 🌡️ Diferencias Térmicas
            * **PVC (Termoplástico):** Se ablanda con el calor. Límite **70°C**.
            * **XLPE (Termoestable):** Mantiene estructura. Límite **90°C**.
            
            ### 🔥 Comportamiento al Fuego
            * **PVC:** Emite humo negro y ácido (Corrosivo).
            * **XLPE (Libre de Halógenos):** Humo blanco, no tóxico.
            """)
            
            # --- BLOQUE DE NORMATIVA CON TABLA OFICIAL ---
            with st.expander("📜 Ver Normativa REBT Asociada"):
                st.markdown("""
                **1. ITC-BT-19 (Instalaciones Interiores):**
                * El **XLPE (90°C)** permite aproximadamente un **22% más de capacidad** de carga que el **PVC (70°C)**.
                """)

                rebt_data = {
                    "Sección (mm²)": [1.5, 2.5, 4, 6, 10, 16, 25, 35, 50, 70, 95, 120, 150, 185, 240],
                    "PVC 70°C (A)": [15, 21, 28, 36, 50, 68, 89, 110, 134, 171, 207, 239, 272, 310, 364],
                    "XLPE 90°C (A)": [18, 26, 34, 44, 61, 82, 108, 135, 164, 211, 254, 294, 335, 382, 453]
                }
                df_rebt = pd.DataFrame(rebt_data)
                st.write("**Intensidades Admisibles (A) - Referencia: Cobre, Método C**")
                st.table(df_rebt) 
                st.caption("Valores según norma UNE 20460-5-523.")

        with col_plot:
            # --- EXPLICACIÓN MATEMÁTICA ACTUALIZADA ---
            st.markdown("##### Modelo de Degradación por Umbral (Threshold Breakdown)")
            
            # Updated Text Description
            st.write("This visualization uses a generic negative decay function ($y=y_1 - A \\cdot e^x$), commonly used in engineering to model behaviors where a material's property remains relatively constant until it reaches a critical threshold, after which it rapidly degrades.")
            
            st.latex(r"Integrity(T) = 100\% - A \cdot e^{k \cdot T}")

            # --- NUEVA LÓGICA DE CÁLCULO ---
            # Range definition
            temp_range = np.arange(20, 110, 1) # 1 degree steps for smooth curve
            
            # Parameters tuned for visual "Crash" at specific temps
            # Formula: y = 100 - A * exp(B * T)
            # We want the "knee" of the curve to hit near 70 (PVC) and 90 (XLPE)
            
            # PVC Calculation
            # "Crash" starts becoming visible around 60C and crosses zero near 75C
            R_pvc = 100 - 0.001 * np.exp(0.16 * temp_range)
            
            # XLPE Calculation
            # We shift the curve. XLPE is roughly 20 degrees more resistant.
            # Effectively: y = 100 - A * exp(B * (T - 20))
            R_xlpe = 100 - 0.001 * np.exp(0.16 * (temp_range - 20))
            
            # Clipping data to avoid negative values in the graph (physically impossible)
            R_pvc = np.clip(R_pvc, 0, 100)
            R_xlpe = np.clip(R_xlpe, 0, 100)

            # Dataframe construction
            df_iso = pd.DataFrame({
                "Temperatura (°C)": np.concatenate([temp_range, temp_range]),
                "Integridad del Material (%)": np.concatenate([R_pvc, R_xlpe]),
                "Tipo": ["PVC (70°C Max)", "XLPE (90°C Max)"] * len(temp_range)
            })
            
            # Plotting
            fig_iso = px.line(df_iso, x="Temperatura (°C)", y="Integridad del Material (%)", 
                            color="Tipo", title="Integridad del Aislamiento vs Temperatura",
                            color_discrete_map={"PVC (70°C Max)": "#EF553B", "XLPE (90°C Max)": "#00CC96"})
            
            # Visual Limits
            fig_iso.add_vrect(x0=70, x1=110, fillcolor="red", opacity=0.1, 
                            annotation_text="Zona Fallo PVC", annotation_position="top left")
            fig_iso.add_vline(x=90, line_dash="dash", line_color="green", annotation_text="Límite XLPE")

            # Y-Axis standardization
            fig_iso.update_yaxes(range=[0, 110])
            
            st.plotly_chart(fig_iso, use_container_width=True)
            st.info("💡 **Conclusión:** Observe cómo la curva se mantiene plana (estable) y colapsa repentinamente al acercarse a la temperatura crítica, simulando la pérdida súbita de propiedades dieléctricas.")
