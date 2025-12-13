import streamlit as st
from modules import basic_calculations, advanced_lines, topology

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Electric Design Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PERSONALIZADOS (MODO TARJETA CLICABLE) ---
st.markdown("""
<style>
    /* Importar fuente moderna (Inter) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    /* Estilos Globales */
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        color: #FAFAFA;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* TRANSFORMAR EL BOTÓN ESTÁNDAR EN UNA "TARJETA INTERACTIVA" */
    /* Apuntamos directamente a los botones dentro de las columnas */
    div.stButton > button {
        width: 100%;
        height: 240px; /* Altura de tarjeta grande */
        
        /* Estética de la Tarjeta (Dark Glass / Neon) */
        background: linear-gradient(145deg, #161B22, #0D1117);
        border: 1px solid #30363D;
        border-radius: 12px;
        color: #C9D1D9;
        
        /* Tipografía y Layout del contenido del botón */
        font-family: 'Inter', sans-serif;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        white-space: pre-wrap; /* Permite saltos de línea (\n) en el texto */
        
        /* Transiciones suaves */
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    /* EFECTO HOVER (AL PASAR EL RATÓN) */
    div.stButton > button:hover {
        transform: translateY(-5px); /* Se eleva */
        border-color: #00ADB5;       /* Borde Neón */
        color: #FFFFFF;              /* Texto blanco brillante */
        box-shadow: 0 8px 20px rgba(0, 173, 181, 0.15); /* Resplandor */
        background: linear-gradient(145deg, #1c2128, #161b22);
    }
    
    /* EFECTO ACTIVE (AL PULSAR) */
    div.stButton > button:active {
        transform: translateY(2px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Separadores */
    hr { border-color: #30363D; }

</style>
""", unsafe_allow_html=True)

# --- ESTADO DE NAVEGACIÓN ---
if 'current_section' not in st.session_state:
    st.session_state.current_section = "Home"

# --- FUNCIONES DE NAVEGACIÓN ---
def go_home():
    st.session_state.current_section = "Home"
    st.rerun()

def set_section(section):
    st.session_state.current_section = section
    st.rerun()

# --- SIDEBAR (Barra Lateral Profesional) ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #00ADB5;'>⚡ EDS Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.8em; color: gray;'>Electric Design Suite v1.0</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.button("🏠 Inicio"):
        go_home()
    
    st.markdown("---")
    st.markdown("**Estado del Sistema:**")
    if st.session_state.current_section == "Home":
        st.success("🟢 Esperando selección")
    else:
        st.info(f"🔵 Módulo Activo: {st.session_state.current_section}")

    st.markdown("---")
    st.caption("© 2025 Ingeniería Eléctrica")

# --- PÁGINA PRINCIPAL (DASHBOARD VISUAL) ---
if st.session_state.current_section == "Home":
    
    # Encabezado Hero
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">Bienvenido, Ingeniero Eléctrico</h1>
        <p style="font-size: 1.2rem; color: #8B949E;">Seleccione el módulo a consultar para su proyecto.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    # Layout de Tarjetas (Que ahora son botones directos)
    col1, col2, col3 = st.columns(3)
    
    # Usamos \n\n para separar Icono, Título y Descripción visualmente gracias al CSS 'white-space: pre-wrap'
    
    with col1:
        # TARJETA 1
        content_1 = "📐\n\nCÁLCULOS BÁSICOS & NORMATIVA\n\nClasificación ITC-BT, Ley de Ohm,\nFactores de potencia y Cables."
        if st.button(content_1):
            set_section("Basic")

    with col2:
        # TARJETA 2
        content_2 = "⚡\n\nLÍNEAS DE ALTA POTENCIA\n\nCálculo avanzado de flechas,\nAnálisis térmico y Transitorios."
        if st.button(content_2):
            set_section("Advanced")

    with col3:
        # TARJETA 3
        content_3 = "🌐\n\nTOPOLOGÍA & DIMENSIONADO\n\nRedes Radiales vs Anillo,\nUnifilares y Optimización."
        if st.button(content_3):
            set_section("Topology")

# --- LÓGICA DE CARGA DE SUBMENÚS ---

elif st.session_state.current_section == "Basic":
    basic_calculations.app()

elif st.session_state.current_section == "Advanced":
    advanced_lines.app()

elif st.session_state.current_section == "Topology":
    topology.app()