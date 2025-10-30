"""
Script para probar y ejecutar visualizaciones de C. elegans
"""
import matplotlib.pyplot as plt
import numpy as np

def test_matplotlib():
    """Probar que matplotlib funciona correctamente"""
    print("=== PROBANDO MATPLOTLIB ===")
    
    # Configurar matplotlib
    plt.ion()
    import matplotlib
    matplotlib.use('TkAgg')
    
    # Crear gráfico de prueba
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    ax.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    ax.set_title('Prueba de Matplotlib - ¡Funcionando!')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("✓ Gráfico de prueba mostrado correctamente!")
    plt.pause(1)
    plt.close()
    
def run_pca_analysis():
    """Ejecutar análisis PCA completo"""
    print("\n=== EJECUTANDO ANÁLISIS PCA COMPLETO ===")
    try:
        exec(open('PCAper.py').read())
        print("✓ Análisis PCA completo terminado con éxito!")
    except Exception as e:
        print(f"❌ Error en PCA: {e}")
        
def run_raster_plot():
    """Ejecutar raster plot"""
    print("\n=== EJECUTANDO RASTER PLOT ===")
    try:
        exec(open('Raster_Plot.py').read())
        print("✓ Raster plot completado con éxito!")
    except Exception as e:
        print(f"❌ Error en Raster Plot: {e}")

def generate_pca_final_clean():
    """Generar PCA_Final_Clean.png"""
    print("\n=== GENERANDO PCA_Final_Clean.png ===")
    try:
        exec(open('generate_PCA_Final_Clean.py').read())
        print("✓ PCA_Final_Clean.png regenerado con éxito!")
    except Exception as e:
        print(f"❌ Error en PCA Final Clean: {e}")
        
def generate_pca_trajectory():
    """Generar PCA estilo trayectorias"""
    print("\n=== GENERANDO PCA ESTILO TRAYECTORIAS ===")
    try:
        exec(open('generate_PCA_trajectory.py').read())
        print("✓ PCA Trajectory Style generado con éxito!")
    except Exception as e:
        print(f"❌ Error en PCA Trajectory: {e}")
        
def generate_pca_2d_3d_only():
    """Generar PCA 2D/3D únicamente (Kato 2015 style)"""
    print("\n=== GENERANDO PCA 2D/3D SOLO (KATO 2015) ===")
    try:
        exec(open('PCA_2D_3D_only.py').read())
        print("✓ PCA 2D/3D Only generado con éxito!")
    except Exception as e:
        print(f"❌ Error en PCA 2D/3D Only: {e}")
        
def generate_pca_behavior():
    """Generar PCA por comportamiento (Adelante vs Atrás)"""
    print("\n=== GENERANDO PCA POR COMPORTAMIENTO ===")
    try:
        exec(open('PCA_behavior_analysis.py').read())
        print("✓ PCA por Comportamiento generado con éxito!")
    except Exception as e:
        print(f"❌ Error en PCA por Comportamiento: {e}")

def generate_pca_3d_overlay():
    """Generar Vista 3D Superpuesta (inicio y final alineados)"""
    print("\n=== GENERANDO VISTA 3D SUPERPUESTA ===")
    try:
        exec(open('PCA_3D_overlay_simple.py').read())
        print("✓ Vista 3D Superpuesta generada con éxito!")
    except Exception as e:
        print(f"❌ Error en Vista 3D Superpuesta: {e}")

def generate_derivatives_analysis():
    """Generar Análisis Comparativo FR vs dFR/dt"""
    print("\n=== GENERANDO ANÁLISIS DE DERIVADAS ===")
    try:
        exec(open('derivatives_simple_analysis.py').read())
        print("✓ Análisis de derivadas generado con éxito!")
    except Exception as e:
        print(f"❌ Error en análisis de derivadas: {e}")

def generate_pca_derivatives():
    """Generar PCA con Derivadas Temporales"""
    print("\n=== GENERANDO PCA CON DERIVADAS ===")
    try:
        exec(open('PCA_2D_3D_derivatives.py').read())
        print("✓ PCA con derivadas generado con éxito!")
    except Exception as e:
        print(f"❌ Error en PCA con derivadas: {e}")

def generate_pca_behavior_derivatives():
    """Generar PCA Comportamental con Derivadas"""
    print("\n=== GENERANDO PCA COMPORTAMENTAL CON DERIVADAS ===")
    try:
        # Ejecutar el código del análisis directamente (sin emojis para compatibilidad)
        exec('''
import numpy as np
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')

# Cargar datos
nwb_file_path = "sub-2023-01-09-15-SWF702_ses-20230109_behavior+image+ophys.nwb"

with NWBHDF5IO(nwb_file_path, 'r') as io:
    nwbfile = io.read()
    calcium_module = nwbfile.processing['CalciumActivity']
    roi_response_series = calcium_module['SignalCalciumImResponseSeries']
    traces = roi_response_series.data[:]
    timestamps = roi_response_series.timestamps[:]
    angular_velocity = nwbfile.processing['Behavior']['angular_velocity']['angular_velocity'].data[:]

# Preprocesamiento
F0 = np.percentile(traces, 10, axis=0)
neural_matrix = (traces - F0) / (F0 + 1e-8)
neural_matrix[np.isnan(neural_matrix)] = 0

# Calcular derivadas
dt = np.mean(np.diff(timestamps))
smoothed_neural = gaussian_filter1d(neural_matrix, sigma=2.0, axis=0)
derivatives = np.gradient(smoothed_neural, dt, axis=0)
derivatives[np.isnan(derivatives)] = 0

# Segmentar por comportamiento
min_length = min(len(angular_velocity), derivatives.shape[0])
angular_velocity = angular_velocity[:min_length]
derivatives = derivatives[:min_length]

forward_mask = angular_velocity > 0.02
backward_mask = angular_velocity < -0.02

derivatives_forward = derivatives[forward_mask]
derivatives_backward = derivatives[backward_mask]

# PCA por comportamiento
def perform_pca(data):
    if len(data) < 20:
        return None, None
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)
    pca = PCA()
    pc = pca.fit_transform(data_std)
    var = pca.explained_variance_ratio_
    return pc, var

pc_forward, var_forward = perform_pca(derivatives_forward)
pc_backward, var_backward = perform_pca(derivatives_backward)

if pc_forward is not None and pc_backward is not None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PCA Comportamental con Derivadas Temporales (dFR/dt)', fontsize=16, fontweight='bold')
    
    # Adelante 2D
    colors_f = plt.cm.Reds(np.linspace(0.3, 1, len(pc_forward)))
    axes[0,0].scatter(pc_forward[:, 0], pc_forward[:, 1], c=colors_f, alpha=0.7, s=6)
    axes[0,0].set_title(f'ADELANTE\\nPC1: {var_forward[0]*100:.1f}%, PC2: {var_forward[1]*100:.1f}%')
    axes[0,0].grid(True, alpha=0.3)
    
    # Atras 2D
    colors_b = plt.cm.Blues(np.linspace(0.3, 1, len(pc_backward)))
    axes[0,1].scatter(pc_backward[:, 0], pc_backward[:, 1], c=colors_b, alpha=0.7, s=6)
    axes[0,1].set_title(f'ATRAS\\nPC1: {var_backward[0]*100:.1f}%, PC2: {var_backward[1]*100:.1f}%')
    axes[0,1].grid(True, alpha=0.3)
    
    # Varianza
    pc_range = np.arange(1, 6)
    width = 0.35
    axes[1,0].bar(pc_range - width/2, var_forward[:5]*100, width, label='Adelante', color='red', alpha=0.7)
    axes[1,0].bar(pc_range + width/2, var_backward[:5]*100, width, label='Atras', color='blue', alpha=0.7)
    axes[1,0].set_title('Varianza Explicada')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Distribución
    axes[1,1].hist(derivatives_forward.flatten(), bins=40, alpha=0.6, color='red', density=True, label='Adelante')
    axes[1,1].hist(derivatives_backward.flatten(), bins=40, alpha=0.6, color='blue', density=True, label='Atras')
    axes[1,1].set_title('Distribucion dFR/dt')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('PCA_Behavior_Derivatives_Menu.png', dpi=300, bbox_inches='tight')
    plt.close()
''')
        print("✓ PCA comportamental con derivadas generado con éxito!")
    except Exception as e:
        print(f"❌ Error en PCA comportamental con derivadas: {e}")

if __name__ == "__main__":
    print("🧬 EJECUTOR DE VISUALIZACIONES DE C. ELEGANS 🧬")
    print("=" * 60)
    
    # Probar matplotlib
    test_matplotlib()
    
    # Menú interactivo
    while True:
        print("\n¿Qué visualización deseas generar?")
        print("1. Análisis PCA completo (9 gráficos)")
        print("2. Raster Plot (actividad neural + comportamiento)")
        print("3. PCA_Final_Clean.png (4 gráficos profesionales)")
        print("4. PCA Trayectorias (estilo elegante con rueda de colores)")
        print("5. PCA 2D/3D SOLO (inspirado en Kato 2015)")
        print("6. PCA por Comportamiento (Adelante vs Atrás)")
        print("7. 🎯 Vista 3D Superpuesta (inicio y final alineados)")
        print("8. 📊 Análisis de Derivadas Temporales (dFR/dt vs FR)")
        print("9. 🧪 PCA con Derivadas Temporales (dFR/dt)")
        print("10. 🔬 PCA Comportamental con Derivadas")
        print("11. Todas las visualizaciones")
        print("12. Salir")
        
        choice = input("\nElige una opción (1-12): ").strip()
        
        if choice == '1':
            run_pca_analysis()
        elif choice == '2':
            run_raster_plot()
        elif choice == '3':
            generate_pca_final_clean()
        elif choice == '4':
            generate_pca_trajectory()
        elif choice == '5':
            generate_pca_2d_3d_only()
        elif choice == '6':
            generate_pca_behavior()
        elif choice == '7':
            generate_pca_3d_overlay()
        elif choice == '8':
            generate_derivatives_analysis()
        elif choice == '9':
            generate_pca_derivatives()
        elif choice == '10':
            generate_pca_behavior_derivatives()
        elif choice == '11':
            print("🚀 Ejecutando todas las visualizaciones...")
            run_pca_analysis()
            run_raster_plot()
            generate_pca_final_clean()
            generate_pca_trajectory()
            generate_pca_2d_3d_only()
            generate_pca_behavior()
            generate_pca_3d_overlay()
            generate_derivatives_analysis()
            generate_pca_derivatives()
            generate_pca_behavior_derivatives()
            print("✅ ¡Todas las visualizaciones completadas!")
        elif choice == '12':
            print("¡Hasta luego! 👋")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")
            
        input("\nPresiona Enter para continuar...")

    print("\n📊 ARCHIVOS GENERADOS:")
    print("- PCA_Complete_Analysis.png: Análisis exhaustivo con 9 subgráficos")
    print("- Raster_Plot_Output.png: Raster plot principal")
    print("- Colorbar_Separate.png: Colorbar independiente")
    print("- PCA_Final_Clean.png: 4 gráficos PCA profesionales")
    print("- PCA_Trajectory_Style.png: Trayectorias temporales elegantes")
    print("- PCA_2D_3D_Multiple_Views.png: Solo PCA 2D/3D (estilo Kato 2015)")
    print("- PCA_Behavior_Forward_vs_Backward.png: Análisis por comportamiento")
    print("- PCA_3D_Overlay_MultiView.png: 🎯 Vista superpuesta (inicio/final alineados)")
    print("- FR_vs_Derivatives_Simple.png: 📊 Comparación FR vs dFR/dt")
    print("- PCA_2D_3D_Derivatives.png: 🧪 PCA con derivadas temporales")
    print("- PCA_Behavior_Derivatives.png: 🔬 PCA comportamental con derivadas")
    print("\n🎯 ¡Análisis neural de C. elegans completado!")