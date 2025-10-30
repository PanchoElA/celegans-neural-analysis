"""
PCA por Comportamiento con DERIVADAS TEMPORALES
Análisis de dFR/dt durante movimientos hacia adelante vs hacia atrás
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from scipy import signal
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo

print("=== PCA COMPORTAMENTAL CON DERIVADAS TEMPORALES (dFR/dt) ===")

# Cargar datos neurales y comportamentales
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

print(f"Datos cargados: {traces.shape[1]} neuronas, {traces.shape[0]} timepoints")

dt = np.mean(np.diff(timestamps))
print(f"Resolución temporal: {dt:.4f} s ({1/dt:.1f} Hz)")

# ===============================
# PREPROCESAMIENTO Y DERIVADAS
# ===============================

# Filtro pasa-bajas
nyquist = 0.5 / dt
cutoff = 0.1  # Hz
b, a = signal.butter(4, cutoff / nyquist, btype='low')

filtered_traces = np.zeros_like(traces)
for i in range(traces.shape[1]):
    filtered_traces[:, i] = signal.filtfilt(b, a, traces[:, i])

# ΔF/F0
F0 = np.percentile(filtered_traces, 10, axis=0)
neural_matrix = (filtered_traces - F0) / (F0 + 1e-8)

# Calcular derivadas temporales
print("Calculando derivadas temporales...")
smoothed_data = gaussian_filter1d(neural_matrix, sigma=2.0, axis=0)
derivatives_data = np.gradient(smoothed_data, dt, axis=0)

# Limpiar datos
derivatives_data[np.isnan(derivatives_data)] = 0
derivatives_data[np.isinf(derivatives_data)] = 0

# ===============================
# SEGMENTACIÓN POR COMPORTAMIENTO
# ===============================

print("Segmentando por comportamiento basado en velocidad angular...")

# Sincronizar longitudes
min_length = min(len(angular_velocity), derivatives_data.shape[0])
angular_velocity = angular_velocity[:min_length]
derivatives_data = derivatives_data[:min_length]
timestamps = timestamps[:min_length]

print(f"Datos sincronizados: {min_length} timepoints")
print(f"Rango velocidad angular: {np.min(angular_velocity):.3f} a {np.max(angular_velocity):.3f}")

# Segmentar comportamientos
forward_mask = angular_velocity > 0  # Movimiento hacia adelante
backward_mask = angular_velocity < 0  # Movimiento hacia atrás

derivatives_forward = derivatives_data[forward_mask]
derivatives_backward = derivatives_data[backward_mask]

print(f"Segmentación completada:")
print(f"  - Hacia adelante: {np.sum(forward_mask)} timepoints ({np.sum(forward_mask)/min_length*100:.1f}%)")
print(f"  - Hacia atrás: {np.sum(backward_mask)} timepoints ({np.sum(backward_mask)/min_length*100:.1f}%)")

# ===============================
# PCA POR COMPORTAMIENTO
# ===============================

def perform_behavioral_pca(data, behavior_name):
    """Realizar PCA en datos de un comportamiento específico"""
    if len(data) == 0:
        return None, None, None
    
    # Estandarizar
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    
    # PCA
    pca = PCA()
    principal_components = pca.fit_transform(data_standardized)
    explained_variance = pca.explained_variance_ratio_
    
    print(f"{behavior_name} - PC1: {explained_variance[0]*100:.1f}%, PC2: {explained_variance[1]*100:.1f}%, PC3: {explained_variance[2]*100:.1f}%")
    
    return principal_components, explained_variance, scaler

# Realizar PCA por comportamiento
pc_forward, var_forward, scaler_forward = perform_behavioral_pca(derivatives_forward, "Hacia adelante (dFR/dt)")
pc_backward, var_backward, scaler_backward = perform_behavioral_pca(derivatives_backward, "Hacia atrás (dFR/dt)")

# ===============================
# SUAVIZADO PARA VISUALIZACIÓN
# ===============================

def smooth_trajectory(pc_data, factor=3):
    """Suavizar trayectoria PCA"""
    if pc_data is None or len(pc_data) == 0:
        return None, None, None
    
    # Filtro Gaussiano
    pc1_filtered = gaussian_filter1d(pc_data[:, 0], sigma=1.5)
    pc2_filtered = gaussian_filter1d(pc_data[:, 1], sigma=1.5)
    pc3_filtered = gaussian_filter1d(pc_data[:, 2], sigma=1.5)
    
    # Interpolación cúbica
    t = np.arange(len(pc_data))
    t_smooth = np.linspace(0, len(pc_data)-1, len(pc_data)*factor)
    
    f1 = interpolate.interp1d(t, pc1_filtered, kind='cubic')
    f2 = interpolate.interp1d(t, pc2_filtered, kind='cubic')
    f3 = interpolate.interp1d(t, pc3_filtered, kind='cubic')
    
    return f1(t_smooth), f2(t_smooth), f3(t_smooth)

pc1_f_smooth, pc2_f_smooth, pc3_f_smooth = smooth_trajectory(pc_forward)
pc1_b_smooth, pc2_b_smooth, pc3_b_smooth = smooth_trajectory(pc_backward)

# ===============================
# CREAR VISUALIZACIÓN
# ===============================

fig = plt.figure(figsize=(16, 8))
fig.patch.set_facecolor('white')

# === GRÁFICO 2D ===
ax1 = fig.add_subplot(1, 2, 1)

# Trayectorias hacia adelante (rojas)
if pc1_f_smooth is not None:
    colors_forward = plt.cm.Reds(np.linspace(0.3, 1.0, len(pc1_f_smooth)))
    for i in range(len(pc1_f_smooth) - 1):
        ax1.plot([pc1_f_smooth[i], pc1_f_smooth[i+1]], 
                 [pc2_f_smooth[i], pc2_f_smooth[i+1]], 
                 color=colors_forward[i], alpha=0.7, linewidth=1.5)
    
    # Puntos inicio/final adelante
    ax1.scatter(pc_forward[0, 0], pc_forward[0, 1], color='darkred', s=100, 
               marker='o', edgecolor='white', linewidth=2, alpha=1.0, label='Inicio Adelante')
    ax1.scatter(pc_forward[-1, 0], pc_forward[-1, 1], color='red', s=100, 
               marker='s', edgecolor='white', linewidth=2, alpha=1.0, label='Final Adelante')

# Trayectorias hacia atrás (azules)
if pc1_b_smooth is not None:
    colors_backward = plt.cm.Blues(np.linspace(0.3, 1.0, len(pc1_b_smooth)))
    for i in range(len(pc1_b_smooth) - 1):
        ax1.plot([pc1_b_smooth[i], pc1_b_smooth[i+1]], 
                 [pc2_b_smooth[i], pc2_b_smooth[i+1]], 
                 color=colors_backward[i], alpha=0.7, linewidth=1.5)
    
    # Puntos inicio/final atrás
    ax1.scatter(pc_backward[0, 0], pc_backward[0, 1], color='darkblue', s=100, 
               marker='o', edgecolor='white', linewidth=2, alpha=1.0, label='Inicio Atrás')
    ax1.scatter(pc_backward[-1, 0], pc_backward[-1, 1], color='blue', s=100, 
               marker='s', edgecolor='white', linewidth=2, alpha=1.0, label='Final Atrás')

ax1.set_xlabel(f'PC1 (Adelante: {var_forward[0]*100:.1f}%, Atrás: {var_backward[0]*100:.1f}%)', 
               fontsize=12, fontweight='bold')
ax1.set_ylabel(f'PC2 (Adelante: {var_forward[1]*100:.1f}%, Atrás: {var_backward[1]*100:.1f}%)', 
               fontsize=12, fontweight='bold')
ax1.set_title('PCA 2D - Derivadas por Comportamiento', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.2)
ax1.set_facecolor('#f8f9fa')

# === GRÁFICO 3D ===
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# Trayectorias 3D hacia adelante
if pc1_f_smooth is not None:
    for i in range(len(pc1_f_smooth) - 1):
        ax2.plot([pc1_f_smooth[i], pc1_f_smooth[i+1]], 
                 [pc2_f_smooth[i], pc2_f_smooth[i+1]], 
                 [pc3_f_smooth[i], pc3_f_smooth[i+1]], 
                 color=colors_forward[i], alpha=0.7, linewidth=1.5)
    
    ax2.scatter(pc_forward[0, 0], pc_forward[0, 1], pc_forward[0, 2], 
               color='darkred', s=100, alpha=1.0)
    ax2.scatter(pc_forward[-1, 0], pc_forward[-1, 1], pc_forward[-1, 2], 
               color='red', s=100, alpha=1.0)

# Trayectorias 3D hacia atrás
if pc1_b_smooth is not None:
    for i in range(len(pc1_b_smooth) - 1):
        ax2.plot([pc1_b_smooth[i], pc1_b_smooth[i+1]], 
                 [pc2_b_smooth[i], pc2_b_smooth[i+1]], 
                 [pc3_b_smooth[i], pc3_b_smooth[i+1]], 
                 color=colors_backward[i], alpha=0.7, linewidth=1.5)
    
    ax2.scatter(pc_backward[0, 0], pc_backward[0, 1], pc_backward[0, 2], 
               color='darkblue', s=100, alpha=1.0)
    ax2.scatter(pc_backward[-1, 0], pc_backward[-1, 1], pc_backward[-1, 2], 
               color='blue', s=100, alpha=1.0)

ax2.set_xlabel(f'PC1', fontweight='bold')
ax2.set_ylabel(f'PC2', fontweight='bold')
ax2.set_zlabel(f'PC3', fontweight='bold')
ax2.set_title('PCA 3D - Derivadas por Comportamiento', fontsize=14, fontweight='bold')
ax2.view_init(elev=25, azim=45)

# Configurar vista 3D
ax2.grid(True, alpha=0.2)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False

# ===============================
# INFORMACIÓN ADICIONAL
# ===============================

# Calcular estadísticas comparativas
mean_deriv_forward = np.mean(np.abs(derivatives_forward))
mean_deriv_backward = np.mean(np.abs(derivatives_backward))
std_deriv_forward = np.std(derivatives_forward)
std_deriv_backward = np.std(derivatives_backward)

info_text = f"""Análisis Comportamental - Derivadas Temporales (dFR/dt)

Estadísticas:
• Adelante: |dFR/dt| medio = {mean_deriv_forward:.4f}, std = {std_deriv_forward:.4f}
• Atrás: |dFR/dt| medio = {mean_deriv_backward:.4f}, std = {std_deriv_backward:.4f}

Varianza Explicada (3 PCs):
• Adelante: {np.sum(var_forward[:3])*100:.1f}%
• Atrás: {np.sum(var_backward[:3])*100:.1f}%

Interpretación:
🔴 Rojo: Cambios neurales durante movimiento hacia adelante
🔵 Azul: Cambios neurales durante movimiento hacia atrás
• Magnitud indica velocidad del cambio neural
• Dirección indica tipo de cambio (activación/desactivación)"""

fig.text(0.02, 0.02, info_text, fontsize=9, 
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
         verticalalignment='bottom')

# Título general
plt.suptitle('Análisis Comportamental PCA: Derivadas Temporales del Firing Rate\n' + 
             f'Adelante vs Atrás - C. elegans ({min_length} timepoints)', 
             fontsize=16, fontweight='bold', y=0.95)

plt.tight_layout()
plt.subplots_adjust(top=0.87, bottom=0.25, left=0.08, right=0.95)

# Guardar archivo
filename = 'PCA_Behavior_Derivatives.png'
plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

print(f"✅ Archivo guardado: {filename}")

# ===============================
# ANÁLISIS ESTADÍSTICO
# ===============================

print("\n" + "="*60)
print("ANÁLISIS COMPORTAMENTAL DE DERIVADAS")
print("="*60)

print(f"Actividad neural durante movimientos:")
print(f"  Adelante - Media |dFR/dt|: {mean_deriv_forward:.6f}")
print(f"  Atrás - Media |dFR/dt|: {mean_deriv_backward:.6f}")

if mean_deriv_forward > mean_deriv_backward:
    ratio = mean_deriv_forward / mean_deriv_backward
    print(f"  🎯 Movimiento ADELANTE {ratio:.1f}x más dinámico neuralmente")
else:
    ratio = mean_deriv_backward / mean_deriv_forward
    print(f"  🎯 Movimiento ATRÁS {ratio:.1f}x más dinámico neuralmente")

print(f"\nVarianza capturada por comportamiento:")
print(f"  Adelante (3 PCs): {np.sum(var_forward[:3])*100:.1f}%")
print(f"  Atrás (3 PCs): {np.sum(var_backward[:3])*100:.1f}%")

# Análisis de separabilidad
if pc_forward is not None and pc_backward is not None:
    # Calcular centroides
    centroid_forward = np.mean(pc_forward[:, :2], axis=0)
    centroid_backward = np.mean(pc_backward[:, :2], axis=0)
    separation_distance = np.linalg.norm(centroid_forward - centroid_backward)
    
    print(f"\nSeparabilidad en espacio PCA:")
    print(f"  Distancia entre centroides: {separation_distance:.4f}")
    
    if separation_distance > 2.0:
        print("  🎯 ALTA separabilidad - Patrones muy distintos")
    elif separation_distance > 1.0:
        print("  🎯 MODERADA separabilidad - Patrones diferenciables")
    else:
        print("  🎯 BAJA separabilidad - Patrones similares")

print("="*60)

print("✅ Visualización comportamental guardada exitosamente")
print("\n🎯 Análisis comportamental con derivadas completado!")