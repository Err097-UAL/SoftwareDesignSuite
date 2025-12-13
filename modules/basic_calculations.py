import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def app():
    st.header("1. Line Classification & Basic Calculations")
    st.caption("Módulo de análisis fundamental, selección de materiales y normativa.")

    # --- ORGANIZACIÓN EN PESTAÑAS (TABS) ---
    # Esto sustituye a los menús antiguos de MATLAB
    tab_projects, tab_materials, tab_insulation, tab_calc, tab_wizard = st.tabs([
        "📊 Proyectos y Estadísticas",
        "⛓️ Análisis de Conductores",
        "🔥 Aislamientos (PVC vs XLPE)",
        "🧮 Laboratorio de Cálculo",
        "🧙‍♂️ Asistente de Diseño"
    ])

    # ==============================================================================
    # TAB 1: PROYECTOS Y ESTADÍSTICAS (Mejorado con Jerarquía Normativa)
    # ==============================================================================
    with tab_projects:
        st.subheader("Base de datos acerca de los proyectos eléctricos planteados")
        
        # 1. DATOS ENRIQUECIDOS
        # Hemos añadido la columna "ITC/Norma" para que sea educativo
        data_proyectos = [
            {
                "Proyecto": "Instalación Industrial", 
                "Tensión (V)": 20000, 
                "Nivel": "MT (Alta Tensión)", 
                "Topología": "Mixta", 
                "Conductor": "Cobre (XLPE)",
                "Norma": "RAT + ITC-LAT 06",
                "Cantidad": 1
            },
            {
                "Proyecto": "Complejo Residencial", 
                "Tensión (V)": 400, 
                "Nivel": "BT (Baja Tensión)", 
                "Topología": "Subterránea", 
                "Conductor": "Cobre (PVC)",
                "Norma": "REBT ITC-BT-07",
                "Cantidad": 1
            },
            {
                "Proyecto": "Centro Comercial (Línea MT)", 
                "Tensión (V)": 20000, 
                "Nivel": "MT (Alta Tensión)", 
                "Topología": "Aérea", 
                "Conductor": "Aluminio (XLPE)",
                "Norma": "RAT + ITC-LAT 07",
                "Cantidad": 1
            },
            {
                "Proyecto": "Centro Comercial (Interior)", 
                "Tensión (V)": 400, 
                "Nivel": "BT (Baja Tensión)", 
                "Topología": "Interior/Entubada", 
                "Conductor": "Cobre (XLPE)",
                "Norma": "REBT ITC-BT-19/28",
                "Cantidad": 1
            }
        ]
        df_projects = pd.DataFrame(data_proyectos)
        
        # Mostrar tabla interactiva
        st.dataframe(
            df_projects, 
            column_config={
                "Tensión (V)": st.column_config.NumberColumn(format="%d V"),
                "Norma": st.column_config.TextColumn(help="Reglamento aplicable")
            },
            use_container_width=True
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("##### 📊 Comparativa de Niveles de Tensión")
            # Gráfico de Barras mejorado con colores de advertencia por nivel
            fig_bar = px.bar(
                df_projects, 
                x="Proyecto", 
                y="Tensión (V)", 
                color="Nivel",
                color_discrete_map={"MT (Alta Tensión)": "#FF4B4B", "BT (Baja Tensión)": "#00CC96"},
                text_auto=True
            )
            fig_bar.update_layout(showlegend=False, xaxis_title=None)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col2:
            st.markdown("##### 🎯 Clasificación de Normativa y Topología ")
            # GRÁFICO SOLAR (SUNBURST)
            # Muestra la jerarquía: Nivel -> Topología -> Conductor
            # Esto ayuda al ingeniero a ver rápidamente qué grupos de normas aplican
            fig_sun = px.sunburst(
                df_projects, 
                path=['Nivel', 'Topología', 'Conductor'], 
                values='Cantidad',
                color='Nivel',
                color_discrete_map={"MT (Alta Tensión)": "#FF4B4B", "BT (Baja Tensión)": "#00CC96"},
            )
            
            # Personalización para hacerlo más profesional
            fig_sun.update_traces(textinfo="label+percent entry")
            fig_sun.update_layout(margin=dict(t=0, l=0, r=0, b=0))
            
            st.plotly_chart(fig_sun, use_container_width=True)
            
        # Nota explicativa sobre el gráfico solar
        st.info("💡 **Este gráfico circular separa visualmente el ámbito de la Alta Tensión de la Baja Tensión y sus respectivas topologías permitidas.")
    # ==============================================================================
    # TAB 2: ANÁLISIS DE MATERIALES (Basado en B1.m, B3.m y B2.m)
    # ==============================================================================
    with tab_materials:
        st.subheader("Propiedades Físicas y Eléctricas")
        
        # Datos de Materiales (B1.m)
        materials_data = {
            "Material": ["Cobre", "Aluminio (AAC)", "Aluminio-Acero (ACSR)"],
            "Conductividad (m/Ωmm²)": [56.0, 36.0, 34.0],
            "Coste Estimado (€/km)": [1200, 700, 850],
            "Resistencia Tracción Max (MPa)": [450, 160, 1500],
            "Módulo Young Max (GPa)": [125, 70, 75]
        }
        df_mat = pd.DataFrame(materials_data)
        
        # Mostrar tabla de propiedades
        st.dataframe(df_mat.style.highlight_max(axis=0, color="#2c5e2e"), use_container_width=True)
        
        # Gráficos Comparativos (B3.m)
        c1, c2 = st.columns(2)
        with c1:
            fig_tensile = px.bar(df_mat, x="Material", y="Resistencia Tracción Max (MPa)", 
                                 color="Material", title="Resistencia a la Tracción (Mecánica)")
            st.plotly_chart(fig_tensile, use_container_width=True)
        with c2:
            fig_cond = px.bar(df_mat, x="Material", y="Conductividad (m/Ωmm²)", 
                              color="Material", title="Conductividad Eléctrica")
            st.plotly_chart(fig_cond, use_container_width=True)

        st.divider()
        
        # --- SIMULACIÓN DE DISTANCIA (B2.m) ---
        st.subheader("Simulación de Rendimiento vs Distancia")
        st.info("Ajuste los parámetros para ver cómo se comportan los materiales a larga distancia.")
        
        # Controles
        col_sim1, col_sim2, col_sim3 = st.columns(3)
        load_va = col_sim1.number_input("Carga Aparente (VA)", value=1000)
        voltage_sys = col_sim2.number_input("Tensión Sistema (V)", value=400)
        section_sim = col_sim3.number_input("Sección (mm²)", value=2.5)
        
        # Cálculos vectores (Numpy)
        dist_km = np.linspace(0, 2, 100) # De 0 a 2 km
        dist_m = dist_km * 1000
        current_load = load_va / (np.sqrt(3) * voltage_sys)
        
        # Dataframe para plotear
        df_sim_list = []
        for index, row in df_mat.iterrows():
            mat_name = row["Material"]
            sigma = row["Conductividad (m/Ωmm²)"]
            cost_unit = row["Coste Estimado (€/km)"]
            
            # Fórmulas B2.m
            R_vec = dist_m / (sigma * section_sim)
            V_drop = np.sqrt(3) * current_load * R_vec
            Power_loss = 3 * (current_load**2) * R_vec
            Cost_total = cost_unit * dist_km
            
            # Crear mini DF temporal
            df_temp = pd.DataFrame({
                "Distancia (km)": dist_km,
                "Caída Tensión (V)": V_drop,
                "Pérdida Potencia (W)": Power_loss,
                "Coste (€)": Cost_total,
                "Material": mat_name
            })
            df_sim_list.append(df_temp)
            
        df_simu_final = pd.concat(df_sim_list)
        
        # Visualización Selector
        plot_type = st.radio("Seleccione variable a analizar:", 
                             ["Caída Tensión (V)", "Pérdida Potencia (W)", "Coste (€)"], 
                             horizontal=True)
        
        fig_sim = px.line(df_simu_final, x="Distancia (km)", y=plot_type, color="Material",
                          title=f"Evolución de {plot_type} según Distancia", markers=False)
        
        # Línea límite REBT (5%) si es caída de tensión
        if plot_type == "Caída Tensión (V)":
            limit_v = voltage_sys * 0.05
            fig_sim.add_hline(y=limit_v, line_dash="dash", line_color="red", 
                              annotation_text=f"Límite REBT 5% ({limit_v:.1f}V)")
            
        st.plotly_chart(fig_sim, use_container_width=True)

   # ==============================================================================
    # TAB 3: AISLAMIENTOS (Mejorada con Normativa REBT)
    # ==============================================================================
    with tab_insulation:
        st.subheader("Comparativa Técnica: PVC vs XLPE")
        
        col_text, col_plot = st.columns([1, 1.5])
        
        with col_text:
            st.markdown("""
            ### 🌡️ Diferencias Térmicas
            * **PVC (Termoplástico):** Se ablanda con el calor. Límite **70°C**.
            * **XLPE (Termoestable):** Mantiene estructura. Límite **90°C**.
            
            ### 🔥 Comportamiento al Fuego
            * **PVC:** Emite humo negro y ácido (Corrosivo).
            * **XLPE (Libre de Halógenos):** Humo blanco, no tóxico.
            """)
            
            # --- NUEVO BLOQUE DE NORMATIVA ---
            with st.expander("📜 Ver Normativa REBT Asociada"):
                st.markdown("""
                **1. ITC-BT-19 (Instalaciones Interiores):**
                * Define las tablas de intensidad admisible.
                * El **XLPE** permite aprox. un **20% más de corriente** que el PVC para la misma sección.
                
                **2. ITC-BT-28 (Pública Concurrencia):**
                * En Hospitales, Hoteles y C.Comerciales es **OBLIGATORIO** usar cables (AS) Libres de Halógenos.
                * ❌ **PVC:** Prohibido (Propaga incendio y humos tóxicos).
                * ✅ **XLPE (RZ1-K):** Permitido (No propagador, baja emisión de humos).
                
                **3. ITC-BT-07 (Redes Subterráneas):**
                * Estándar de facto: Cables **RV-K (XLPE)** por su resistencia hidrófuga y térmica.
                """)

        with col_plot:
            # Simulación C1.m (Degradación resistencia con temperatura)
            temp_range = np.arange(20, 120, 5)
            R0 = 1000 # Valor base
            
            # Fórmulas exponenciales para la simulación visual
            R_pvc = R0 * np.exp(-0.045 * (temp_range - 20))
            R_xlpe = R0 * np.exp(-0.035 * (temp_range - 20))
            
            df_iso = pd.DataFrame({
                "Temperatura (°C)": np.concatenate([temp_range, temp_range]),
                "Resistencia Aislamiento (Relativa)": np.concatenate([R_pvc, R_xlpe]),
                "Tipo": ["PVC (70°C Max)"]*len(temp_range) + ["XLPE (90°C Max)"]*len(temp_range)
            })
            
            fig_iso = px.line(df_iso, x="Temperatura (°C)", y="Resistencia Aislamiento (Relativa)", 
                              color="Tipo", title="Degradación del Aislamiento vs Temperatura")
            
            # Zonas de peligro visuales
            fig_iso.add_vrect(x0=70, x1=120, fillcolor="red", opacity=0.1, 
                              annotation_text="Fallo PVC", annotation_position="top left")
            
            fig_iso.add_vline(x=90, line_dash="dash", line_color="green", annotation_text="Límite XLPE")
            
            st.plotly_chart(fig_iso, use_container_width=True)
            
            st.info("💡 **Conclusión de Ingeniería:** Use XLPE para líneas de alta potencia o locales públicos. Use PVC para cableado doméstico básico o control.")

    # ==============================================================================
    # TAB 4: LABORATORIO DE CÁLCULO (Basado en D1.m)
    # ==============================================================================
    with tab_calc:
        st.subheader("Cálculo de Escenarios (Alumnos A, B, C)")
        
        # Parámetros Globales
        c_glob1, c_glob2 = st.columns(2)
        v_line = c_glob1.number_input("Tensión de Línea (V)", value=400)
        s_cond = c_glob2.number_input("Sección Conductor (mm²)", value=95)
        sigma_cu = 56.0 # Cobre
        
        # Tabla Editable (User Friendly: Puedes cambiar los valores del alumno)
        st.write("Edite los valores de la tabla para recalcular:")
        
        default_data = pd.DataFrame([
            {"Alumno": "A", "Longitud (m)": 500, "Potencia (kW)": 50, "Cos phi": 0.80},
            {"Alumno": "B", "Longitud (m)": 1200, "Potencia (kW)": 150, "Cos phi": 0.90},
            {"Alumno": "C", "Longitud (m)": 2500, "Potencia (kW)": 300, "Cos phi": 0.85},
        ])
        
        edited_df = st.data_editor(default_data, num_rows="dynamic")
        
        if st.button("🚀 Ejecutar Cálculos"):
            # Lógica D1.m Vectorizada
            # 1. Resistencia R = L / (sigma * S)
            edited_df["R (Ω)"] = edited_df["Longitud (m)"] / (sigma_cu * s_cond)
            
            # 2. Corriente I = P / (sqrt(3) * V * cosphi)
            # OJO: P en kW -> *1000 para W
            edited_df["I (A)"] = (edited_df["Potencia (kW)"] * 1000) / (np.sqrt(3) * v_line * edited_df["Cos phi"])
            
            # 3. Caída V fase = I * R
            edited_df["Caída V (Fase)"] = edited_df["I (A)"] * edited_df["R (Ω)"]
            
            # Formato bonito
            st.success("Cálculos realizados con éxito.")
            st.dataframe(edited_df.style.format({
                "R (Ω)": "{:.4f}",
                "I (A)": "{:.2f}",
                "Caída V (Fase)": "{:.2f}"
            }), use_container_width=True)

    # ==============================================================================
    # TAB 5: ASISTENTE DE DISEÑO (Basado en A2.m)
    # ==============================================================================
    with tab_wizard:
        st.subheader("Asistente de Selección de Línea")
        st.markdown("Responda las preguntas para recibir una recomendación preliminar.")
        
        c_wiz1, c_wiz2 = st.columns(2)
        
        # Inputs (Paso 1, 2, 3 de A2.m)
        voltage_wiz = c_wiz1.number_input("Tensión del Proyecto (V)", value=400)
        lifetime_wiz = c_wiz2.number_input("Vida útil estimada (años)", value=30)
        
        app_type = st.radio("Aplicación Principal:", 
                            ["Alimentador Principal (Subestación -> Distribución)", 
                             "Distribución Local (Conexión final a edificios)"])
        
        location_code = "main_feeder" if "Alimentador" in app_type else "local_distribution"
        
        if st.button("Generar Recomendación"):
            rec_material = ""
            rec_topology = ""
            
            # Lógica Conductor (selectConductorMaterial)
            if lifetime_wiz > 30:
                rec_material = "Cobre (Mayor durabilidad y menores pérdidas a largo plazo)"
                icon_mat = "💎"
            else:
                rec_material = "Aluminio (Solución coste-efectiva para proyectos de menor duración)"
                icon_mat = "💰"
                
            # Lógica Topología (selectLineTopology)
            if voltage_wiz > 1000 and location_code == "main_feeder":
                rec_topology = "Línea Aérea MT (Eficiencia en costes y disipación de calor)"
            elif voltage_wiz <= 1000 and location_code == "local_distribution":
                rec_topology = "Línea Subterránea BT (Seguridad y estética en zonas pobladas)"
            else:
                rec_topology = "Caso Especial / Mixto (Requiere análisis detallado)"
                
            # Resultado Visual
            st.success("Recomendación de Diseño:")
            st.markdown(f"""
            * **Material Recomendado:** {icon_mat} **{rec_material}**
            * **Topología Sugerida:** 🏙️ **{rec_topology}**
            """)
            st.caption("Nota: Esta es una recomendación preliminar basada en reglas generales de ingeniería.")

# --- FIN DEL CÓDIGO ---