"""
🌐 Visualizador de Gráficos 3D Interactivos
Abre y muestra las visualizaciones 3D generadas

Este script:
1. Abre los archivos HTML 3D en el navegador
2. Muestra descripción detallada de cada visualización
3. Proporciona guías de uso interactivo
"""

import webbrowser
import os
from pathlib import Path
import time

def show_banner():
    """Mostrar banner de bienvenida"""
    print("🌐" + "=" * 70 + "🌐")
    print("     VISUALIZACIONES 3D INTERACTIVAS DE DINÁMICAS NEURALES")
    print("            Enhanced Le Cunff et al. 2024 Edition")
    print("🌐" + "=" * 70 + "🌐")

def check_html_files():
    """Verificar que los archivos HTML existan"""
    html_files = {
        'pca': '3D_Trayectoria_PCA_Neural.html',
        'derivatives': '3D_Espacio_Derivadas_Neural.html', 
        'dashboard': '3D_Dashboard_Neural_Integrado.html'
    }
    
    existing_files = {}
    missing_files = []
    
    for key, filename in html_files.items():
        if os.path.exists(filename):
            existing_files[key] = filename
            print(f"✅ {filename}")
        else:
            missing_files.append(filename)
            print(f"❌ {filename}")
    
    return existing_files, missing_files

def describe_visualizations():
    """Describir cada tipo de visualización"""
    print("\n📖 DESCRIPCIÓN DE VISUALIZACIONES 3D")
    print("=" * 60)
    
    print("\n🧠 1. TRAYECTORIA PCA 3D (3D_Trayectoria_PCA_Neural.html)")
    print("   📊 Qué muestra:")
    print("   • Trayectoria neural en espacio de componentes principales")
    print("   • Estados conductuales identificados por colores")
    print("   • Evolución temporal con gradiente de colores")
    print("   • Puntos de inicio (verde) y final (rojo)")
    print("   ")
    print("   🎮 Cómo interactuar:")
    print("   • Arrastra para rotar el espacio 3D")
    print("   • Scroll para hacer zoom in/out")
    print("   • Hover sobre puntos para ver detalles")
    print("   • Click en leyenda para ocultar/mostrar elementos")
    print("   ")
    print("   🔍 Qué buscar:")
    print("   • Loops o ciclos en la trayectoria")
    print("   • Agrupaciones de estados conductuales")
    print("   • Transiciones entre diferentes regiones")
    print("   • Direcciones principales de varianza")
    
    print("\n🌊 2. ESPACIO DE DERIVADAS 3D (3D_Espacio_Derivadas_Neural.html)")
    print("   📊 Qué muestra:")
    print("   • Velocidades de cambio neural en 3D")
    print("   • Dinámicas temporales de activación")
    print("   • Estados conductuales en espacio de derivadas")
    print("   • Patrones de aceleración/desaceleración")
    print("   ")
    print("   🎮 Cómo interactuar:")
    print("   • Mismas interacciones que PCA 3D")
    print("   • Colorbar muestra diferentes estados")
    print("   • Observe patrones de clustering")
    print("   ")
    print("   🔍 Qué buscar:")
    print("   • Regiones de alta/baja actividad derivativa")
    print("   • Transiciones rápidas vs lentas")
    print("   • Separación clara entre estados conductuales")
    print("   • Patrones espirales o cíclicos")
    
    print("\n📊 3. DASHBOARD INTEGRADO (3D_Dashboard_Neural_Integrado.html)")
    print("   📊 Qué muestra:")
    print("   • Vista comparativa lado a lado")
    print("   • PCA (izquierda) vs Derivadas (derecha)")
    print("   • Sincronización temporal entre ambos espacios")
    print("   • Análisis dual de las dinámicas")
    print("   ")
    print("   🎮 Cómo interactuar:")
    print("   • Cada panel es independiente")
    print("   • Compara patrones entre espacios")
    print("   • Rota ambos para encontrar mejores ángulos")
    print("   ")
    print("   🔍 Qué buscar:")
    print("   • Correlaciones entre ambos espacios")
    print("   • Diferentes perspectivas de los mismos datos")
    print("   • Estados que son claros en un espacio pero no en otro")

def show_interaction_guide():
    """Mostrar guía de interacción detallada"""
    print("\n🎮 GUÍA DE INTERACCIÓN DETALLADA")
    print("=" * 50)
    print("🖱️  CONTROLES DEL MOUSE:")
    print("   • Click y arrastra: Rotar vista 3D")
    print("   • Scroll: Zoom in/out")
    print("   • Double-click: Auto-fit vista")
    print("   • Hover: Mostrar información detallada")
    print("")
    print("🎛️  CONTROLES DE INTERFAZ:")
    print("   • Leyenda: Click para ocultar/mostrar elementos")
    print("   • Toolbar (esquina superior derecha):")
    print("     📷 Descargar como imagen")
    print("     🔍 Zoom a región")
    print("     🏠 Reset vista")
    print("     ↕️  Auto escala")
    print("")
    print("🔍 TÉCNICAS DE EXPLORACIÓN:")
    print("   • Busca agrupaciones y separaciones")
    print("   • Identifica direcciones principales")
    print("   • Observa transiciones temporales")
    print("   • Compara diferentes ángulos de vista")
    print("   • Usa zoom para detalles específicos")

def open_visualizations(existing_files):
    """Abrir visualizaciones en el navegador"""
    print("\n🌐 ABRIENDO VISUALIZACIONES EN NAVEGADOR")
    print("=" * 50)
    
    file_descriptions = {
        'pca': '🧠 Trayectoria PCA 3D',
        'derivatives': '🌊 Espacio de Derivadas 3D',
        'dashboard': '📊 Dashboard Integrado'
    }
    
    for key, filename in existing_files.items():
        print(f"\n🔄 Abriendo {file_descriptions[key]}...")
        file_path = Path(filename).absolute().as_uri()
        try:
            webbrowser.open(file_path)
            print(f"✅ Abierto: {filename}")
            time.sleep(1)  # Pequeña pausa entre aperturas
        except Exception as e:
            print(f"❌ Error abriendo {filename}: {str(e)}")
            print(f"💡 Abre manualmente: {filename}")

def show_analysis_tips():
    """Mostrar consejos de análisis"""
    print("\n💡 CONSEJOS DE ANÁLISIS NEUROBIOLÓGICO")
    print("=" * 50)
    print("🧠 INTERPRETACIÓN BIOLÓGICA:")
    print("   • PCA PC1-PC3: Modos principales de actividad neural")
    print("   • Derivadas: Velocidades de cambio de actividad")
    print("   • Estados conductuales: Agrupaciones funcionales")
    print("   • Trayectorias: Secuencias temporales de activación")
    print("")
    print("🔬 QUÉ BUSCAR:")
    print("   • Ciclos: Comportamientos repetitivos")
    print("   • Bifurcaciones: Puntos de decisión conductual") 
    print("   • Atractores: Estados estables del sistema")
    print("   • Transiciones: Cambios entre estados")
    print("")
    print("📈 PATRONES IMPORTANTES:")
    print("   • Loops cerrados: Comportamientos cíclicos")
    print("   • Ramas: Diferentes rutas conductuales")
    print("   • Clusters: Grupos de actividad similar")
    print("   • Gradientes: Cambios progresivos")

def main():
    """Función principal"""
    show_banner()
    
    print("\n🔍 Verificando archivos de visualización...")
    existing_files, missing_files = check_html_files()
    
    if missing_files:
        print(f"\n⚠️  Archivos faltantes: {len(missing_files)}")
        print("💡 Ejecuta Simple_3D_Visualizations.py primero para generar las visualizaciones")
        for file in missing_files:
            print(f"   • {file}")
        return
    
    if not existing_files:
        print("\n❌ No se encontraron visualizaciones 3D")
        print("💡 Ejecuta Simple_3D_Visualizations.py para generar las visualizaciones")
        return
    
    # Mostrar información detallada
    describe_visualizations()
    show_interaction_guide()
    show_analysis_tips()
    
    # Preguntar si abrir archivos
    print(f"\n🌐 Se encontraron {len(existing_files)} visualizaciones 3D")
    response = input("¿Deseas abrirlas en el navegador? (Y/n): ").lower().strip()
    
    if response in ['', 'y', 'yes', 'si', 's']:
        open_visualizations(existing_files)
        print("\n🎉 ¡Visualizaciones abiertas!")
        print("🎮 Usa los controles descritos arriba para explorar")
        print("🔬 Busca los patrones neurobiológicos mencionados")
    else:
        print("\n📁 Los archivos están disponibles para abrir manualmente:")
        for filename in existing_files.values():
            print(f"   • {filename}")
    
    print("\n🧠 ¡Disfruta explorando las dinámicas neurales en 3D! 🧠")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 ¡Hasta luego! Explora las visualizaciones cuando quieras.")
    except Exception as e:
        print(f"\n💥 Error inesperado: {str(e)}")
        print("Por favor reporta este error si persiste.")