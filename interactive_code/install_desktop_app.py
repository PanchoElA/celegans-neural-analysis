"""
🔧 Instalador Automático - Neural Analysis Desktop App
Auto-installer for C. elegans Neural Analysis Application
"""

import subprocess
import sys
import os

def print_header():
    """Mostrar header del instalador"""
    print("🧠" + "=" * 60 + "🧠")
    print("    C. ELEGANS NEURAL ANALYSIS - AUTO INSTALLER")
    print("         Desktop Application Setup v2.0")
    print("🧠" + "=" * 60 + "🧠")

def check_python():
    """Verificar versión de Python"""
    print("\n🔍 Verificando Python...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor} detected - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} detected - Requiere Python 3.8+")
        return False

def install_requirements():
    """Instalar dependencias desde requirements"""
    print("\n📦 Instalando dependencias...")
    
    req_file = "requirements_desktop.txt"
    
    if not os.path.exists(req_file):
        print(f"❌ No se encuentra {req_file}")
        return False
    
    try:
        print("   Ejecutando: pip install -r requirements_desktop.txt")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", req_file
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Todas las dependencias instaladas correctamente")
            return True
        else:
            print(f"❌ Error instalando dependencias:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error en instalación: {str(e)}")
        return False

def verify_installation():
    """Verificar que todas las dependencias estén instaladas"""
    print("\n🔬 Verificando instalación...")
    
    packages = [
        'numpy', 'pandas', 'plotly', 'sklearn', 'scipy', 'matplotlib'
    ]
    
    failed = []
    
    for package in packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Falló la instalación de: {', '.join(failed)}")
        return False
    else:
        print("\n✅ Todas las dependencias verificadas correctamente")
        return True

def check_data_files():
    """Verificar archivos de datos"""
    print("\n📊 Verificando archivos de datos...")
    
    data_file = "neural_data_dataframe.csv"
    
    if os.path.exists(data_file):
        print(f"✅ Archivo de datos encontrado: {data_file}")
        
        # Verificar tamaño
        size_mb = os.path.getsize(data_file) / (1024 * 1024)
        print(f"   Tamaño: {size_mb:.1f} MB")
        
    else:
        print(f"⚠️  Archivo de datos no encontrado: {data_file}")
        print("   La aplicación generará datos sintéticos automáticamente")
    
    return True

def create_shortcuts():
    """Crear accesos directos"""
    print("\n🔗 Creando accesos directos...")
    
    # Crear launcher batch para Windows
    if sys.platform == "win32":
        batch_content = f'''@echo off
cd /d "{os.getcwd()}"
"{sys.executable}" launcher.py
pause'''
        
        with open("Neural_Analysis_Launcher.bat", "w") as f:
            f.write(batch_content)
        
        print("✅ Creado: Neural_Analysis_Launcher.bat")
        
        # Crear launcher directo para la app
        app_batch_content = f'''@echo off
cd /d "{os.getcwd()}"
"{sys.executable}" neural_desktop_app_v2.py
pause'''
        
        with open("Neural_Analysis_Direct.bat", "w") as f:
            f.write(app_batch_content)
            
        print("✅ Creado: Neural_Analysis_Direct.bat")
    
    return True

def main():
    """Función principal del instalador"""
    
    print_header()
    
    # Verificaciones previas
    if not check_python():
        print("\n💡 Por favor actualiza Python a versión 3.8 o superior")
        input("Presiona Enter para salir...")
        return
    
    # Instalar dependencias
    if not install_requirements():
        print("\n💡 Intenta instalar manualmente con:")
        print("   pip install -r requirements_desktop.txt")
        input("Presiona Enter para salir...")
        return
    
    # Verificar instalación
    if not verify_installation():
        print("\n💡 Algunas dependencias fallaron. Intenta reinstalar.")
        input("Presiona Enter para salir...")
        return
    
    # Verificar archivos
    check_data_files()
    
    # Crear shortcuts
    create_shortcuts()
    
    # Resumen final
    print("\n" + "🎉" + "=" * 60 + "🎉")
    print("         ¡INSTALACIÓN COMPLETADA EXITOSAMENTE!")
    print("🎉" + "=" * 60 + "🎉")
    
    print("\n📋 CÓMO USAR:")
    print("   • Doble-click en 'Neural_Analysis_Launcher.bat' (recomendado)")
    print("   • O ejecuta: python launcher.py")
    print("   • O directamente: python neural_desktop_app_v2.py")
    
    print("\n✨ CARACTERÍSTICAS DISPONIBLES:")
    print("   🎆 Visualizaciones PCA 3D interactivas")
    print("   🌊 Análisis de derivadas del firing rate")
    print("   🎭 Identificación automática de comportamientos")
    print("   ⚙️ Preprocesamiento configurable")
    print("   🎨 Personalización completa de visualizaciones")
    print("   🌐 Exportación a múltiples formatos")
    
    print("\n📖 DOCUMENTACIÓN:")
    print("   • Ver README_DESKTOP_APP.md para guía completa")
    print("   • Casos de uso y solución de problemas incluidos")
    
    print("\n🧠 ¡Disfruta explorando las dinámicas neurales! 🧠")
    
    # Preguntar si lanzar ahora
    response = input("\n¿Deseas lanzar la aplicación ahora? (y/N): ").lower().strip()
    
    if response in ['y', 'yes', 'si', 's']:
        print("\n🚀 Lanzando aplicación...")
        try:
            subprocess.Popen([sys.executable, "launcher.py"])
            print("✅ Aplicación lanzada exitosamente!")
        except Exception as e:
            print(f"❌ Error lanzando aplicación: {str(e)}")
            print("💡 Intenta ejecutar manualmente: python launcher.py")
    
    input("\nPresiona Enter para finalizar...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Instalación cancelada por usuario")
    except Exception as e:
        print(f"\n💥 Error inesperado: {str(e)}")
        input("Presiona Enter para salir...")