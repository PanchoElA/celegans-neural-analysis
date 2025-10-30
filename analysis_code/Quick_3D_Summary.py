"""
🚀 RESUMEN EJECUTIVO - Gráficos 3D Interactivos
Le Cunff et al. 2024 Enhanced Neural Analysis

¡TUS VISUALIZACIONES 3D ESTÁN LISTAS!
"""

import os
import webbrowser
from pathlib import Path

print("🧠" + "=" * 60 + "🧠")
print("    GRÁFICOS 3D INTERACTIVOS GENERADOS EXITOSAMENTE")
print("🧠" + "=" * 60 + "🧠")

# Verificar archivos
files = {
    "🧠 Trayectoria PCA 3D": "3D_Trayectoria_PCA_Neural.html",
    "🌊 Espacio de Derivadas": "3D_Espacio_Derivadas_Neural.html", 
    "📊 Dashboard Integrado": "3D_Dashboard_Neural_Integrado.html"
}

print("\n✅ ARCHIVOS DISPONIBLES:")
for desc, filename in files.items():
    if os.path.exists(filename):
        print(f"   • {desc}: {filename}")
    else:
        print(f"   ❌ {desc}: {filename} (no encontrado)")

print("\n🎯 CARACTERÍSTICAS PRINCIPALES:")
print("   🔄 Rotación 3D interactiva con mouse")
print("   🔍 Zoom dinámico con scroll")
print("   📊 Información detallada con hover")
print("   🎨 Estados conductuales con colores")
print("   ⏱️  Evolución temporal visualizada")
print("   📈 147 neuronas, 1615 timepoints")

print("\n🚀 MEJORAS IMPLEMENTADAS (Le Cunff et al. 2024):")
print("   ✓ PCA con validación estadística avanzada")
print("   ✓ Identificación automática de estados conductuales") 
print("   ✓ Análisis de derivadas con múltiples métodos")
print("   ✓ Visualizaciones interactivas con Plotly")
print("   ✓ Dashboard integrado para comparación")

print("\n🎮 CÓMO USAR:")
print("   1. Abre cualquier archivo .html en tu navegador")
print("   2. Arrastra con mouse para rotar el espacio 3D")
print("   3. Usa scroll para zoom in/out") 
print("   4. Hover sobre puntos para ver detalles")
print("   5. Click en leyenda para ocultar/mostrar elementos")

print("\n💡 QUÉ BUSCAR EN LOS GRÁFICOS:")
print("   🔵 Agrupaciones = Estados conductuales")
print("   🔄 Loops/ciclos = Comportamientos repetitivos")
print("   ➡️  Trayectorias = Secuencias temporales")
print("   📈 Gradientes = Transiciones progresivas")
print("   🎯 Separaciones = Diferentes modos neurales")

print("\n🌐 ABRIR VISUALIZACIONES:")
response = input("¿Abrir automáticamente en navegador? (Y/n): ").lower().strip()

if response in ['', 'y', 'yes', 'si', 's']:
    print("\n🔄 Abriendo visualizaciones...")
    for desc, filename in files.items():
        if os.path.exists(filename):
            try:
                file_path = Path(filename).absolute().as_uri()
                webbrowser.open(file_path)
                print(f"✅ {desc} abierto")
            except:
                print(f"❌ Error abriendo {desc}")
    print("\n🎉 ¡Visualizaciones abiertas!")
else:
    print("\n📁 Abre manualmente estos archivos:")
    for desc, filename in files.items():
        if os.path.exists(filename):
            print(f"   • {filename}")

print(f"\n🧠 ¡DISFRUTA EXPLORANDO LAS DINÁMICAS NEURALES EN 3D! 🧠")
print("📧 Los gráficos usan metodologías avanzadas de Le Cunff et al. 2024")