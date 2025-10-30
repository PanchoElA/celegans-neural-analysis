"""
PCA COMPORTAMENTAL CON DERIVADAS TEMPORALES
Análisis de dFR/dt durante movimientos hacia adelante vs hacia atrás
Versión mejorada con visualización detallada
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import interpolate, signal
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para estabilidad

print("=== PCA COMPORTAMENTAL CON DERIVADAS TEMPORALES (dFR/dt) ===")
print("Analizando dinámicas neurales: ADELANTE vs ATRÁS")

# ===============================
# CARGA Y PREPARACIÓN DE DATOS
# ===============================

nwb_file_path = "sub-2023-01-09-15-SWF702_ses-20230109_behavior+image+ophys.nwb"

with NWBHDF5IO(nwb_file_path, 'r') as io:
    nwbfile = io.read()
    
    # Datos neurales
    calcium_module = nwbfile.processing['CalciumActivity']
    roi_response_series = calcium_module['SignalCalciumImResponseSeries']
    traces = roi_response_series.data[:]
    timestamps = roi_response_series.timestamps[:]
    
    # Datos comportamentales
    angular_velocity = nwbfile.processing['Behavior']['angular_velocity']['angular_velocity'].data[:]

print(f"Datos cargados:")
print(f"  - Neuronas: {traces.shape[1]}")
print(f"  - Timepoints: {traces.shape[0]}")
print(f"  - Duración: {(timestamps[-1] - timestamps[0])/60:.1f} minutos")

# Resolución temporal
dt = np.mean(np.diff(timestamps))
print(f"  - Resolución temporal: {dt:.3f} segundos ({1/dt:.1f} Hz)")

# ===============================
# PREPROCESAMIENTO OPTIMIZADO
# ===============================

print("\n1. Aplicando preprocesamiento optimizado...")

# Filtrado pasa-bajas para reducir ruido
nyquist = 0.5 / dt
cutoff_freq = 0.1  # Hz - frecuencia de corte
b, a = signal.butter(4, cutoff_freq / nyquist, btype='low')

print(f"   - Filtro pasa-bajas: {cutoff_freq} Hz")

filtered_traces = np.zeros_like(traces)
for i in range(traces.shape[1]):
    filtered_traces[:, i] = signal.filtfilt(b, a, traces[:, i])

# Normalización ΔF/F₀ robusta
F0 = np.percentile(filtered_traces, 10, axis=0)
neural_matrix = (filtered_traces - F0) / (F0 + 1e-8)

# Limpiar datos problemáticos
neural_matrix[np.isnan(neural_matrix)] = 0
neural_matrix[np.isinf(neural_matrix)] = 0

print(f"   - ΔF/F₀ calculado (baseline: percentil 10)")

# ===============================
# CÁLCULO DE DERIVADAS TEMPORALES
# ===============================

print("\n2. Calculando derivadas temporales dFR/dt...")

# Suavizado Gaussiano antes del cálculo de derivadas
sigma_smooth = 2.0
smoothed_neural = gaussian_filter1d(neural_matrix, sigma=sigma_smooth, axis=0)
print(f"   - Suavizado Gaussiano aplicado (σ = {sigma_smooth})")

# Cálculo de derivadas usando gradiente
derivatives_data = np.gradient(smoothed_neural, dt, axis=0)

# Limpiar derivadas
derivatives_data[np.isnan(derivatives_data)] = 0
derivatives_data[np.isinf(derivatives_data)] = 0

print(f"   - Rango de derivadas: {np.min(derivatives_data):.4f} a {np.max(derivatives_data):.4f}")
print(f"   - Derivadas promedio: {np.mean(derivatives_data):.6f} ± {np.std(derivatives_data):.6f}")

# ===============================
# SEGMENTACIÓN COMPORTAMENTAL
# ===============================

print("\n3. Segmentación por comportamiento...")

# Sincronizar longitudes
min_length = min(len(angular_velocity), derivatives_data.shape[0], len(timestamps))
angular_velocity = angular_velocity[:min_length]
derivatives_data = derivatives_data[:min_length]
timestamps = timestamps[:min_length]

print(f"   - Datos sincronizados: {min_length} timepoints")

# Análisis de velocidad angular
print(f"   - Rango velocidad angular: {np.min(angular_velocity):.3f} a {np.max(angular_velocity):.3f} rad/s")
print(f"   - Velocidad angular media: {np.mean(angular_velocity):.3f} rad/s")

# Definir máscaras de comportamiento
forward_mask = angular_velocity > 0.05   # Umbral para reducir ruido
backward_mask = angular_velocity < -0.05  # Umbral para reducir ruido

# Extraer datos por comportamiento
derivatives_forward = derivatives_data[forward_mask]
derivatives_backward = derivatives_data[backward_mask]

print(f"\nSegmentación completada:")
print(f"   - Movimiento ADELANTE: {np.sum(forward_mask)} timepoints ({np.sum(forward_mask)/min_length*100:.1f}%)")
print(f"   - Movimiento ATRÁS: {np.sum(backward_mask)} timepoints ({np.sum(backward_mask)/min_length*100:.1f}%)")
print(f"   - Estados ambiguos: {min_length - np.sum(forward_mask) - np.sum(backward_mask)} timepoints")

# ===============================
# ANÁLISIS PCA POR COMPORTAMIENTO
# ===============================

def perform_behavioral_derivatives_pca(data, behavior_name, color_name):
    """Realizar análisis PCA en derivadas para un comportamiento específico"""
    
    if len(data) < 10:  # Mínimo de datos para PCA
        print(f"   ⚠️ {behavior_name}: Datos insuficientes ({len(data)} timepoints)")
        return None, None, None
    
    print(f"\n   Analizando {behavior_name}:")
    
    # Estadísticas de las derivadas
    mean_abs_deriv = np.mean(np.abs(data))
    std_deriv = np.std(data)
    print(f"     - |dFR/dt| promedio: {mean_abs_deriv:.6f}")
    print(f"     - Desviación estándar: {std_deriv:.6f}")
    
    # Estandarización
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    
    # PCA
    pca = PCA()
    principal_components = pca.fit_transform(data_standardized)
    explained_variance = pca.explained_variance_ratio_
    
    print(f"     - PC1: {explained_variance[0]*100:.1f}% | PC2: {explained_variance[1]*100:.1f}% | PC3: {explained_variance[2]*100:.1f}%")
    print(f"     - Varianza total (3 PCs): {np.sum(explained_variance[:3])*100:.1f}%")
    
    return principal_components, explained_variance, scaler

print("\n4. Análisis PCA por comportamiento:")

# PCA para movimiento adelante
pc_forward, var_forward, scaler_forward = perform_behavioral_derivatives_pca(
    derivatives_forward, "ADELANTE", "red"
)

# PCA para movimiento atrás  
pc_backward, var_backward, scaler_backward = perform_behavioral_derivatives_pca(
    derivatives_backward, "ATRÁS", "blue"
)

# ===============================
# PREPARACIÓN PARA VISUALIZACIÓN
# ===============================

def smooth_pca_trajectory(pc_data, factor=3, sigma=1.5):
    """Suavizar trayectorias PCA para mejor visualización"""
    if pc_data is None or len(pc_data) < 10:
        return None, None, None
    
    # Filtro Gaussiano en componentes principales
    pc1_filtered = gaussian_filter1d(pc_data[:, 0], sigma=sigma)
    pc2_filtered = gaussian_filter1d(pc_data[:, 1], sigma=sigma)
    pc3_filtered = gaussian_filter1d(pc_data[:, 2], sigma=sigma)
    
    # Interpolación cúbica para más puntos
    t_orig = np.arange(len(pc_data))
    t_interp = np.linspace(0, len(pc_data)-1, len(pc_data)*factor)
    
    f1 = interpolate.interp1d(t_orig, pc1_filtered, kind='cubic')
    f2 = interpolate.interp1d(t_orig, pc2_filtered, kind='cubic')
    f3 = interpolate.interp1d(t_orig, pc3_filtered, kind='cubic')
    
    return f1(t_interp), f2(t_interp), f3(t_interp)

# Suavizar trayectorias
print("\n5. Suavizando trayectorias para visualización...")
pc1_f_smooth, pc2_f_smooth, pc3_f_smooth = smooth_pca_trajectory(pc_forward)
pc1_b_smooth, pc2_b_smooth, pc3_b_smooth = smooth_pca_trajectory(pc_backward)

# ===============================
# VISUALIZACIÓN COMPLETA
# ===============================

print("\n6. Generando visualización...")

fig = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor('white')

# === PANEL 1: PCA 2D ===
ax1 = fig.add_subplot(2, 3, 1)

if pc1_f_smooth is not None:
    # Trayectoria adelante (gradiente rojo)
    colors_f = plt.cm.Reds(np.linspace(0.3, 1.0, len(pc1_f_smooth)))
    for i in range(len(pc1_f_smooth) - 1):
        ax1.plot([pc1_f_smooth[i], pc1_f_smooth[i+1]], 
                 [pc2_f_smooth[i], pc2_f_smooth[i+1]], 
                 color=colors_f[i], alpha=0.8, linewidth=1.8)
    
    # Puntos especiales adelante
    ax1.scatter(pc_forward[0, 0], pc_forward[0, 1], color='darkred', s=120, 
               marker='o', edgecolor='white', linewidth=3, alpha=1.0, 
               label=f'Inicio Adelante', zorder=10)
    ax1.scatter(pc_forward[-1, 0], pc_forward[-1, 1], color='red', s=120, 
               marker='s', edgecolor='white', linewidth=3, alpha=1.0, 
               label=f'Final Adelante', zorder=10)

if pc1_b_smooth is not None:
    # Trayectoria atrás (gradiente azul)
    colors_b = plt.cm.Blues(np.linspace(0.3, 1.0, len(pc1_b_smooth)))
    for i in range(len(pc1_b_smooth) - 1):
        ax1.plot([pc1_b_smooth[i], pc1_b_smooth[i+1]], 
                 [pc2_b_smooth[i], pc2_b_smooth[i+1]], 
                 color=colors_b[i], alpha=0.8, linewidth=1.8)
    
    # Puntos especiales atrás
    ax1.scatter(pc_backward[0, 0], pc_backward[0, 1], color='darkblue', s=120, 
               marker='o', edgecolor='white', linewidth=3, alpha=1.0, 
               label=f'Inicio Atrás', zorder=10)
    ax1.scatter(pc_backward[-1, 0], pc_backward[-1, 1], color='blue', s=120, 
               marker='s', edgecolor='white', linewidth=3, alpha=1.0, 
               label=f'Final Atrás', zorder=10)

ax1.set_xlabel(f'PC1 (A: {var_forward[0]*100:.1f}%, T: {var_backward[0]*100:.1f}%)', 
               fontsize=12, fontweight='bold')
ax1.set_ylabel(f'PC2 (A: {var_forward[1]*100:.1f}%, T: {var_backward[1]*100:.1f}%)', 
               fontsize=12, fontweight='bold')
ax1.set_title('PCA 2D - Derivadas dFR/dt\nPor Comportamiento', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.2)
ax1.set_facecolor('#f8f9fa')

# === PANEL 2: PCA 3D ===
ax2 = fig.add_subplot(2, 3, 2, projection='3d')

if pc1_f_smooth is not None and pc3_f_smooth is not None:
    # Trayectoria 3D adelante
    for i in range(len(pc1_f_smooth) - 1):
        ax2.plot([pc1_f_smooth[i], pc1_f_smooth[i+1]], 
                 [pc2_f_smooth[i], pc2_f_smooth[i+1]], 
                 [pc3_f_smooth[i], pc3_f_smooth[i+1]], 
                 color=colors_f[i], alpha=0.8, linewidth=1.8)
    
    ax2.scatter(pc_forward[0, 0], pc_forward[0, 1], pc_forward[0, 2], 
               color='darkred', s=120, alpha=1.0, edgecolor='white')
    ax2.scatter(pc_forward[-1, 0], pc_forward[-1, 1], pc_forward[-1, 2], 
               color='red', s=120, alpha=1.0, edgecolor='white')

if pc1_b_smooth is not None and pc3_b_smooth is not None:
    # Trayectoria 3D atrás
    for i in range(len(pc1_b_smooth) - 1):
        ax2.plot([pc1_b_smooth[i], pc1_b_smooth[i+1]], 
                 [pc2_b_smooth[i], pc2_b_smooth[i+1]], 
                 [pc3_b_smooth[i], pc3_b_smooth[i+1]], 
                 color=colors_b[i], alpha=0.8, linewidth=1.8)
    
    ax2.scatter(pc_backward[0, 0], pc_backward[0, 1], pc_backward[0, 2], 
               color='darkblue', s=120, alpha=1.0, edgecolor='white')
    ax2.scatter(pc_backward[-1, 0], pc_backward[-1, 1], pc_backward[-1, 2], 
               color='blue', s=120, alpha=1.0, edgecolor='white')

ax2.set_xlabel('PC1', fontweight='bold')
ax2.set_ylabel('PC2', fontweight='bold') 
ax2.set_zlabel('PC3', fontweight='bold')
ax2.set_title('PCA 3D - Derivadas dFR/dt', fontsize=14, fontweight='bold')
ax2.view_init(elev=25, azim=45)

# === PANEL 3: Varianza Explicada ===
ax3 = fig.add_subplot(2, 3, 3)

if var_forward is not None and var_backward is not None:
    pc_indices = np.arange(1, 8)
    width = 0.35
    
    ax3.bar(pc_indices - width/2, var_forward[:7]*100, width, 
            label='Adelante', color='red', alpha=0.7)
    ax3.bar(pc_indices + width/2, var_backward[:7]*100, width, 
            label='Atrás', color='blue', alpha=0.7)
    
    ax3.set_xlabel('Componente Principal', fontweight='bold')
    ax3.set_ylabel('Varianza Explicada (%)', fontweight='bold')
    ax3.set_title('Varianza por PC\n(Derivadas)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

# === PANEL 4: Distribución de Derivadas ===
ax4 = fig.add_subplot(2, 3, 4)

if derivatives_forward is not None and len(derivatives_forward) > 0:
    # Histograma de derivadas para adelante
    deriv_flat_f = derivatives_forward.flatten()
    ax4.hist(deriv_flat_f, bins=50, alpha=0.6, color='red', 
             label=f'Adelante (n={len(deriv_flat_f)})', density=True)

if derivatives_backward is not None and len(derivatives_backward) > 0:
    # Histograma de derivadas para atrás
    deriv_flat_b = derivatives_backward.flatten()
    ax4.hist(deriv_flat_b, bins=50, alpha=0.6, color='blue', 
             label=f'Atrás (n={len(deriv_flat_b)})', density=True)

ax4.set_xlabel('dFR/dt', fontweight='bold')
ax4.set_ylabel('Densidad', fontweight='bold')
ax4.set_title('Distribución de Derivadas\npor Comportamiento', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# === PANEL 5: Ejemplos de Trazas Temporales ===
ax5 = fig.add_subplot(2, 3, 5)

time_minutes = (timestamps - timestamps[0]) / 60
example_neurons = [10, 30, 50, 70, 90]  # Neuronas ejemplo

colors_neurons = plt.cm.Set1(np.linspace(0, 1, len(example_neurons)))

for i, neuron_idx in enumerate(example_neurons):
    if neuron_idx < derivatives_data.shape[1]:
        # Mostrar solo una ventana temporal
        window_end = min(500, len(time_minutes))
        offset = i * 0.3
        
        ax5.plot(time_minutes[:window_end], 
                derivatives_data[:window_end, neuron_idx] + offset,
                color=colors_neurons[i], alpha=0.8, linewidth=1.2,
                label=f'Neurona {neuron_idx}')

ax5.set_xlabel('Tiempo (minutos)', fontweight='bold')
ax5.set_ylabel('dFR/dt + offset', fontweight='bold')
ax5.set_title('Ejemplos de Derivadas\nTemporales por Neurona', fontweight='bold')
ax5.legend(fontsize=9, loc='upper right')
ax5.grid(True, alpha=0.3)

# === PANEL 6: Análisis de Separabilidad ===
ax6 = fig.add_subplot(2, 3, 6)

if pc_forward is not None and pc_backward is not None:
    # Scatter plot de primeros dos PCs para ver separabilidad
    ax6.scatter(pc_forward[:, 0], pc_forward[:, 1], 
               alpha=0.6, c='red', s=20, label='Adelante')
    ax6.scatter(pc_backward[:, 0], pc_backward[:, 1], 
               alpha=0.6, c='blue', s=20, label='Atrás')
    
    # Centroides
    centroid_f = np.mean(pc_forward[:, :2], axis=0)
    centroid_b = np.mean(pc_backward[:, :2], axis=0)
    
    ax6.scatter(centroid_f[0], centroid_f[1], 
               color='darkred', s=200, marker='X', 
               edgecolor='white', linewidth=2, label='Centroide A')
    ax6.scatter(centroid_b[0], centroid_b[1], 
               color='darkblue', s=200, marker='X', 
               edgecolor='white', linewidth=2, label='Centroide T')
    
    # Línea entre centroides
    ax6.plot([centroid_f[0], centroid_b[0]], [centroid_f[1], centroid_b[1]], 
            'k--', alpha=0.8, linewidth=2, label='Separación')
    
    separation = np.linalg.norm(centroid_f - centroid_b)
    ax6.text(0.05, 0.95, f'Separación: {separation:.3f}', 
            transform=ax6.transAxes, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax6.set_xlabel('PC1', fontweight='bold')
ax6.set_ylabel('PC2', fontweight='bold')
ax6.set_title('Separabilidad en\nEspacio PCA', fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

# ===============================
# CONFIGURACIÓN FINAL
# ===============================

plt.suptitle('Análisis PCA Comportamental con Derivadas Temporales (dFR/dt)\n' + 
             'C. elegans: Dinámicas Neurales durante Movimiento Adelante vs Atrás', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.95, hspace=0.4, wspace=0.3)

# Guardar archivo
filename = 'PCA_Behavior_Derivatives_Complete.png'
plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print(f"\n✅ Visualización guardada: {filename}")

# ===============================
# REPORTE ESTADÍSTICO DETALLADO
# ===============================

print("\n" + "="*80)
print("REPORTE ESTADÍSTICO: ANÁLISIS COMPORTAMENTAL CON DERIVADAS")
print("="*80)

if derivatives_forward is not None and derivatives_backward is not None:
    # Estadísticas de las derivadas
    mean_deriv_f = np.mean(derivatives_forward)
    mean_abs_deriv_f = np.mean(np.abs(derivatives_forward))
    std_deriv_f = np.std(derivatives_forward)
    
    mean_deriv_b = np.mean(derivatives_backward)
    mean_abs_deriv_b = np.mean(np.abs(derivatives_backward))
    std_deriv_b = np.std(derivatives_backward)
    
    print(f"\n📊 ESTADÍSTICAS DE DERIVADAS:")
    print(f"   ADELANTE:")
    print(f"     - Media dFR/dt: {mean_deriv_f:.6f}")
    print(f"     - Media |dFR/dt|: {mean_abs_deriv_f:.6f}")
    print(f"     - Std dFR/dt: {std_deriv_f:.6f}")
    
    print(f"   ATRÁS:")
    print(f"     - Media dFR/dt: {mean_deriv_b:.6f}")
    print(f"     - Media |dFR/dt|: {mean_abs_deriv_b:.6f}")
    print(f"     - Std dFR/dt: {std_deriv_b:.6f}")
    
    # Comparación de dinamismo
    if mean_abs_deriv_f > mean_abs_deriv_b:
        ratio = mean_abs_deriv_f / mean_abs_deriv_b
        print(f"\n🎯 CONCLUSIÓN DINÁMICA:")
        print(f"   Movimiento ADELANTE es {ratio:.2f}x más dinámico neuralmente")
    else:
        ratio = mean_abs_deriv_b / mean_abs_deriv_f
        print(f"\n🎯 CONCLUSIÓN DINÁMICA:")
        print(f"   Movimiento ATRÁS es {ratio:.2f}x más dinámico neuralmente")

if var_forward is not None and var_backward is not None:
    print(f"\n📈 VARIANZA EXPLICADA (PCA):")
    print(f"   ADELANTE - Top 5 PCs: {', '.join([f'{v*100:.1f}%' for v in var_forward[:5]])}")
    print(f"   ATRÁS - Top 5 PCs: {', '.join([f'{v*100:.1f}%' for v in var_backward[:5]])}")
    
    total_var_f = np.sum(var_forward[:5])
    total_var_b = np.sum(var_backward[:5])
    print(f"   ADELANTE - Total (5 PCs): {total_var_f*100:.1f}%")
    print(f"   ATRÁS - Total (5 PCs): {total_var_b*100:.1f}%")
    
    if total_var_f > total_var_b:
        print(f"   🏆 Movimiento ADELANTE tiene patrones más estructurados")
    else:
        print(f"   🏆 Movimiento ATRÁS tiene patrones más estructurados")

# Análisis de separabilidad
if pc_forward is not None and pc_backward is not None:
    centroid_forward = np.mean(pc_forward[:, :3], axis=0)
    centroid_backward = np.mean(pc_backward[:, :3], axis=0)
    separation_distance = np.linalg.norm(centroid_forward - centroid_backward)
    
    print(f"\n🔍 ANÁLISIS DE SEPARABILIDAD:")
    print(f"   Distancia entre centroides: {separation_distance:.4f}")
    
    if separation_distance > 3.0:
        print("   🎯 SEPARABILIDAD EXCELENTE - Patrones muy distintos")
    elif separation_distance > 2.0:
        print("   🎯 SEPARABILIDAD ALTA - Patrones claramente diferentes")
    elif separation_distance > 1.0:
        print("   🎯 SEPARABILIDAD MODERADA - Patrones diferenciables")
    else:
        print("   🎯 SEPARABILIDAD BAJA - Patrones similares")

print(f"\n📁 ARCHIVOS GENERADOS:")
print(f"   - {filename}: Análisis visual completo")
print(f"   - Script: PCA_behavior_derivatives_complete.py")

print(f"\n💡 INTERPRETACIÓN BIOLÓGICA:")
print("   🔴 Trayectorias ROJAS (adelante): Dinámicas neurales durante avance")
print("   🔵 Trayectorias AZULES (atrás): Dinámicas neurales durante retroceso")
print("   📈 dFR/dt > 0: Neuronas incrementando actividad")
print("   📉 dFR/dt < 0: Neuronas reduciendo actividad")
print("   ⚡ |dFR/dt|: Velocidad del cambio neural")

print("="*80)
print("✅ ANÁLISIS COMPORTAMENTAL CON DERIVADAS COMPLETADO")
print("="*80)