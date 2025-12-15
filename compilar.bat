@echo off
echo =========================================
echo   COMPILANDO EDS PRO A EJECUTABLE
echo =========================================
echo.

REM Verificar que Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no encontrado. Instala Python primero.
    pause
    exit /b 1
)

REM Verificar/Instalar PyInstaller
echo Verificando PyInstaller...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo PyInstaller no encontrado. Instalando...
    pip install pyinstaller
    if errorlevel 1 (
        echo ERROR al instalar PyInstaller
        pause
        exit /b 1
    )
)

REM Verificar que existe __init__.py en modules
if not exist "modules\__init__.py" (
    echo Creando modules\__init__.py...
    type nul > "modules\__init__.py"
)

REM Limpiar compilaciones anteriores
echo Limpiando archivos anteriores...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Compilar
echo.
echo Compilando... (esto puede tardar varios minutos)
python -m PyInstaller EDS_Pro.spec

if errorlevel 1 (
    echo.
    echo ERROR durante la compilacion
    pause
    exit /b 1
)

echo.
echo =========================================
echo   COMPILACION EXITOSA
echo =========================================
echo.
echo El ejecutable esta en: dist\EDS_Pro.exe
echo.
pause
