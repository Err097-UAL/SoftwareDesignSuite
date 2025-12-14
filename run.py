import streamlit.web.cli as stcli
import os, sys

def resolve_path(path):
    # Esta función ayuda al .exe a encontrar los archivos dentro de sí mismo
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, path)
    return os.path.join(os.getcwd(), path)

if __name__ == "__main__":
    # Indicamos la ruta de tu archivo principal
    main_script = resolve_path("main.py")
    
    # Configuramos los argumentos de Streamlit
    sys.argv = [
        "streamlit",
        "run",
        main_script,
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())