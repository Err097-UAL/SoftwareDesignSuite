#LIBRERÍAS EMPLEADAS
#La librería STREAMLIT nos permite crear nuestras web de forma sencilla con animaciones visuales
#sin necesidad de HTML o JavaScript.
import streamlit as st
from modules import basic_calculations, advanced_lines, topology
#Con la línea de "from modules import archivo1, archivo 2, archivo3", estamos importando
#a nuestro código los distintos archivos de la carpeta MODULES. Así los podemos emplear
#para la lógica de nuestro programa. 

#En esta parte definimos lo que aparece en la pestaña del navegador: el nombre
#y el icono del rayo. Con el comando LAYOUT=WIDE definimos que el contenido
#abarque todo el ancho de la pantalla. 
st.set_page_config(
    page_title="Electric Design Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

#En este apartado empleamos el lenguaje CSS para personalizar la apariencia de nuestra
#web. Empleamos la librería Streamlit para ello. 
st.markdown("""
<style>
    /* Importar fuente moderna (Inter) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    /* Estilos Globales */
    .stApp {
        background-color:  #012B1B;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        color: #FAFAFA;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* USAMOS TARJETAS INTERACTIVAS PARA REPRESENTAR LA INFORMACIÓN */
    /* Apuntamos directamente a los botones dentro de las columnas */
    div.stButton > button {
        width: 100%;
        height: 240px; /* Altura de tarjeta grande */
        
        /* Estética de la Tarjeta (Dark Glass / Neon) */
        background: linear-gradient(145deg, #161B22, #00ADB5);
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

    /* ANIMACIÓN EFECTO HOVER (AL PASAR EL RATÓN) */
    div.stButton > button:hover {
        transform: translateY(-5px); /* Se eleva */
        border-color: #00ADB5;       /* Borde Neón */
        color: #FFFFFF;              /* Texto blanco brillante */
        box-shadow: 0 8px 20px rgba(0, 173, 181, 0.15); /* Resplandor */
        background: linear-gradient(145deg, #1c2128, #161b22);
    }
    
    /* ANIMACIÓN EFECTO ACTIVE (AL PULSAR) */
    div.stButton > button:active {
        transform: translateY(2px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Separadores */
    hr { border-color: #30363D; }

</style>
""", unsafe_allow_html=True)

#La función ST.SESSION_STATE ES COMO UNA MEMORIA A CORTO PLAZO INTERNA
#DE LA LIBRERÍA STREAMLIT. 
if 'current_section' not in st.session_state:
    st.session_state.current_section = "Home"

#En esta parte definimos las funciones que nos permiten movernos
#entre secciones. "ST.SESSION_STATE + .CURRENT_SECTION = "Sección" significa que
#si la variable almacenada en session_state es igual a la sección definida en
#current_section, vuelve a ejecutar la página (rerun) para dirigirse ahí.  
def go_home():
    st.session_state.current_section = "Home"
    st.rerun()

def set_section(section):
    st.session_state.current_section = section
    st.rerun()

#Aquí definimos los elementos de la barra lateral deslizante. La función
#MARKDOWN siempre tiene que ver con los elementos estéticos. 
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #00ADB5;'>⚡ EDS Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.8em; color: gray;'>Electric Design Suite v1.0</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.button("🏠 Página de Inicio"):
        go_home()
    elif st.button("📐 Cálculos Básicos & Normativa"):
        set_section("Basic")
    elif st.button("⚡ Líneas de Alta Potencia"):
        set_section("Advanced")
    elif st.button("🌐 Topología & Dimensionado"):
        set_section("Topology")

    st.markdown("---")
    st.caption("© 2025 JaqueSoft")

#Aquí definimos los elementos de la página principal.
if st.session_state.current_section == "Home":
    
    #Aquí volvemos a usar la función MARKDOWN para la estética del ENCABEZADO.
    st.markdown("""
    <style>
    @keyframes cinnamon-glow {
        0% { 
            transform: translateY(0px); 
            color: #D2691E; /* Canela base */
            text-shadow: 0 0 5px rgba(210, 105, 30, 0.2);
        }
        50% { 
            transform: translateY(-8px); 
            color: #E67E22; /* Canela más brillante al subir */
            text-shadow: 0 10px 20px rgba(210, 105, 30, 0.5);
        }
        100% { 
            transform: translateY(0px); 
            color: #D2691E;
            text-shadow: 0 0 5px rgba(210, 105, 30, 0.2);
        }
    }

    .titulo-ingeniero {
        font-size: 3.2rem; /* Un poco más grande para resaltar */
        font-weight: 800;
        color: #D2691E;
        margin-bottom: 0.5rem;
        display: inline-block;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        animation: cinnamon-glow 3.5s ease-in-out infinite;
    }
    </style>
    
    <div style="text-align: center; padding: 2.5rem 0;">
        <h1 class="titulo-ingeniero">Bienvenido, Ingeniero Eléctrico</h1>
        <p style="font-size: 1.2rem; color: #8B949E; font-weight: 300; letter-spacing: 1px;">
            Seleccione el módulo a consultar para su proyecto.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    #Con MARKDOWN además definimos las secciones de la web como botones con sus animaciones propias. 
    st.markdown("---")

    # Layout de Tarjetas (Que ahora son botones directos)
    col1, col2, col3 = st.columns(3)
    
    # Usamos \n\n para separar Icono, Título y Descripción visualmente gracias al CSS 'white-space: pre-wrap'
    
    with col1:
        #TARJETA 1
        content_1 = "📐\n\nCÁLCULOS BÁSICOS & NORMATIVA\n\nClasificación ITC-BT, Ley de Ohm,\nFactores de potencia y Cables."
        if st.button(content_1):
            set_section("Basic")

    with col2:
        #TARJETA 2
        content_2 = "⚡\n\nLÍNEAS DE ALTA POTENCIA\n\nCálculo avanzado de flechas,\nAnálisis térmico y Transitorios."
        if st.button(content_2):
            set_section("Advanced")

    with col3:
        #TARJETA 3
        content_3 = "🌐\n\nTOPOLOGÍA & DIMENSIONADO\n\nRedes Radiales vs Anillo,\nUnifilares y Optimización."
        if st.button(content_3):
            set_section("Topology")

#Ejecutamos las APPS cuando se seleccionen los módulos correspondientes. 

elif st.session_state.current_section == "Basic":
    basic_calculations.app()

elif st.session_state.current_section == "Advanced":
    advanced_lines.app()

elif st.session_state.current_section == "Topology":

    topology.app()



