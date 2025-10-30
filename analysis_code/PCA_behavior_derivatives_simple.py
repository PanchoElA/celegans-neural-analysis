"""
Análisis PCA Comportamental con Derivadas - Versión Simplificada
ADELANTE vs ATRÁS usando dFR/dt
"""
import numpy as np
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')

print("=== PCA COMPORTAMENTAL: ADELANTE vs ATRÁS con dFR/dt ===")

# Cargar datos
nwb_file_path = "sub-2023-01-09-15-SWF702_ses-20230109_behavior+image+ophys.nwb"

with NWBHDF5IO(nwb_file_path, 'r') as io:
    nwbfile = io.read()
    
    calcium_module = nwbfile.processing['CalciumActivity']
    roi_response_series = calcium_module['SignalCalciumImResponseSeries']
    traces = roi_response_series.data[:]
    timestamps = roi_response_series.timestamps[:]
    
    angular_velocity = nwbfile.processing['Behavior']['angular_velocity']['angular_velocity'].data[:]

print(f"Datos: {traces.shape[1]} neuronas, {traces.shape[0]} timepoints")

# Preprocesamiento básico
F0 = np.percentile(traces, 10, axis=0)
neural_matrix = (traces - F0) / (F0 + 1e-8)
neural_matrix[np.isnan(neural_matrix)] = 0

# Calcular derivadas
dt = np.mean(np.diff(timestamps))
print(f"Resolución temporal: {dt:.3f} s")

smoothed_neural = gaussian_filter1d(neural_matrix, sigma=2.0, axis=0)
derivatives = np.gradient(smoothed_neural, dt, axis=0)
derivatives[np.isnan(derivatives)] = 0

print(f"Rango derivadas: {np.min(derivatives):.4f} a {np.max(derivatives):.4f}")

# Sincronizar datos
min_length = min(len(angular_velocity), derivatives.shape[0])
angular_velocity = angular_velocity[:min_length]
derivatives = derivatives[:min_length]

# Segmentar por comportamiento
forward_mask = angular_velocity > 0.02
backward_mask = angular_velocity < -0.02

derivatives_forward = derivatives[forward_mask]
derivatives_backward = derivatives[backward_mask]

print(f"Adelante: {np.sum(forward_mask)} puntos ({np.sum(forward_mask)/min_length*100:.1f}%)")
print(f"Atrás: {np.sum(backward_mask)} puntos ({np.sum(backward_mask)/min_length*100:.1f}%)")

# PCA por comportamiento
def pca_behavior(data, name):
    if len(data) < 10:
        return None, None
    
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)
    
    pca = PCA()
    pc = pca.fit_transform(data_std)
    var = pca.explained_variance_ratio_
    
    print(f"{name} - PC1: {var[0]*100:.1f}%, PC2: {var[1]*100:.1f}%, PC3: {var[2]*100:.1f}%")
    return pc, var

pc_forward, var_forward = pca_behavior(derivatives_forward, "ADELANTE")
pc_backward, var_backward = pca_behavior(derivatives_backward, "ATRÁS")

# Crear visualización
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('PCA Comportamental con Derivadas Temporales (dFR/dt)\nAdelante vs Atrás - C. elegans', 
             fontsize=16, fontweight='bold')

# 2D Adelante
if pc_forward is not None:
    colors_f = plt.cm.Reds(np.linspace(0.3, 1, len(pc_forward)))
    axes[0,0].scatter(pc_forward[:, 0], pc_forward[:, 1], c=colors_f, alpha=0.6, s=8)
    axes[0,0].plot(pc_forward[:, 0], pc_forward[:, 1], 'r-', alpha=0.3, linewidth=0.5)
    axes[0,0].scatter(pc_forward[0, 0], pc_forward[0, 1], color='darkred', s=80, marker='o', 
                     edgecolor='white', linewidth=2, label='Inicio')
    axes[0,0].scatter(pc_forward[-1, 0], pc_forward[-1, 1], color='red', s=80, marker='s', 
                     edgecolor='white', linewidth=2, label='Final')

axes[0,0].set_title(f'ADELANTE - PCA 2D\nPC1: {var_forward[0]*100:.1f}%, PC2: {var_forward[1]*100:.1f}%' 
                    if var_forward is not None else 'ADELANTE - Sin datos suficientes')
axes[0,0].set_xlabel('PC1')
axes[0,0].set_ylabel('PC2')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2D Atrás
if pc_backward is not None:
    colors_b = plt.cm.Blues(np.linspace(0.3, 1, len(pc_backward)))
    axes[0,1].scatter(pc_backward[:, 0], pc_backward[:, 1], c=colors_b, alpha=0.6, s=8)
    axes[0,1].plot(pc_backward[:, 0], pc_backward[:, 1], 'b-', alpha=0.3, linewidth=0.5)
    axes[0,1].scatter(pc_backward[0, 0], pc_backward[0, 1], color='darkblue', s=80, marker='o', 
                     edgecolor='white', linewidth=2, label='Inicio')
    axes[0,1].scatter(pc_backward[-1, 0], pc_backward[-1, 1], color='blue', s=80, marker='s', 
                     edgecolor='white', linewidth=2, label='Final')

axes[0,1].set_title(f'ATRÁS - PCA 2D\nPC1: {var_backward[0]*100:.1f}%, PC2: {var_backward[1]*100:.1f}%' 
                    if var_backward is not None else 'ATRÁS - Sin datos suficientes')
axes[0,1].set_xlabel('PC1')
axes[0,1].set_ylabel('PC2')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Comparación varianza
if var_forward is not None and var_backward is not None:
    pc_range = np.arange(1, 8)
    width = 0.35
    axes[1,0].bar(pc_range - width/2, var_forward[:7]*100, width, 
                  label='Adelante', color='red', alpha=0.7)
    axes[1,0].bar(pc_range + width/2, var_backward[:7]*100, width, 
                  label='Atrás', color='blue', alpha=0.7)
    axes[1,0].set_xlabel('Componente Principal')
    axes[1,0].set_ylabel('Varianza Explicada (%)')
    axes[1,0].set_title('Varianza por Comportamiento')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

# Distribución de derivadas
if len(derivatives_forward) > 0:
    axes[1,1].hist(derivatives_forward.flatten(), bins=30, alpha=0.6, color='red', 
                   label=f'Adelante (n={len(derivatives_forward)})', density=True)
if len(derivatives_backward) > 0:
    axes[1,1].hist(derivatives_backward.flatten(), bins=30, alpha=0.6, color='blue', 
                   label=f'Atrás (n={len(derivatives_backward)})', density=True)

axes[1,1].set_xlabel('dFR/dt')
axes[1,1].set_ylabel('Densidad')
axes[1,1].set_title('Distribución de Derivadas')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
filename = 'PCA_Behavior_Derivatives_Simple.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Archivo guardado: {filename}")

# Estadísticas
if len(derivatives_forward) > 0 and len(derivatives_backward) > 0:
    mean_abs_f = np.mean(np.abs(derivatives_forward))
    mean_abs_b = np.mean(np.abs(derivatives_backward))
    
    print(f"\nEstadísticas:")
    print(f"Adelante - |dFR/dt| promedio: {mean_abs_f:.6f}")
    print(f"Atrás - |dFR/dt| promedio: {mean_abs_b:.6f}")
    
    if mean_abs_f > mean_abs_b:
        print(f"🎯 Adelante {mean_abs_f/mean_abs_b:.2f}x más dinámico")
    else:
        print(f"🎯 Atrás {mean_abs_b/mean_abs_f:.2f}x más dinámico")

if var_forward is not None and var_backward is not None:
    print(f"Varianza total (3 PCs):")
    print(f"  Adelante: {np.sum(var_forward[:3])*100:.1f}%")
    print(f"  Atrás: {np.sum(var_backward[:3])*100:.1f}%")

print("✅ Análisis comportamental con derivadas completado!")