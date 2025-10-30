"""
PCA 3D Combinado: Derivadas Forward vs Backward
Un solo grafico 3D mostrando ambas trayectorias de derivadas
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
import matplotlib
matplotlib.use('Agg')

print("=== PCA 3D COMBINADO: DERIVADAS FORWARD vs BACKWARD ===")

# Cargar datos
nwb_file_path = "sub-2023-01-09-15-SWF702_ses-20230109_behavior+image+ophys.nwb"

with NWBHDF5IO(nwb_file_path, 'r') as io:
    nwbfile = io.read()
    
    calcium_module = nwbfile.processing['CalciumActivity']
    roi_response_series = calcium_module['SignalCalciumImResponseSeries']
    traces = roi_response_series.data[:]
    timestamps = roi_response_series.timestamps[:]
    
    angular_velocity = nwbfile.processing['Behavior']['angular_velocity']['angular_velocity'].data[:]

print(f"Datos cargados: {traces.shape[1]} neuronas, {traces.shape[0]} timepoints")

# Preprocesamiento
F0 = np.percentile(traces, 10, axis=0)
neural_matrix = (traces - F0) / (F0 + 1e-8)
neural_matrix[np.isnan(neural_matrix)] = 0

# Calcular derivadas temporales
dt = np.mean(np.diff(timestamps))
print(f"Resolucion temporal: {dt:.3f} segundos")

# Suavizado antes del calculo de derivadas
smoothed_neural = gaussian_filter1d(neural_matrix, sigma=2.0, axis=0)
derivatives = np.gradient(smoothed_neural, dt, axis=0)
derivatives[np.isnan(derivatives)] = 0
derivatives[np.isinf(derivatives)] = 0

print(f"Rango derivadas: {np.min(derivatives):.4f} a {np.max(derivatives):.4f}")

# Sincronizar datos
min_length = min(len(angular_velocity), derivatives.shape[0])
angular_velocity = angular_velocity[:min_length]
derivatives = derivatives[:min_length]

# Segmentacion comportamental con umbrales
threshold = 0.03  # Umbral para evitar ruido
forward_mask = angular_velocity > threshold
backward_mask = angular_velocity < -threshold

derivatives_forward = derivatives[forward_mask]
derivatives_backward = derivatives[backward_mask]

print(f"Forward: {np.sum(forward_mask)} puntos ({np.sum(forward_mask)/min_length*100:.1f}%)")
print(f"Backward: {np.sum(backward_mask)} puntos ({np.sum(backward_mask)/min_length*100:.1f}%)")

# Verificar que tenemos suficientes datos
if len(derivatives_forward) < 50 or len(derivatives_backward) < 50:
    print("ERROR: Datos insuficientes para PCA")
    exit()

# PCA para Forward
print("\nCalculando PCA para Forward...")
scaler_forward = StandardScaler()
forward_standardized = scaler_forward.fit_transform(derivatives_forward)

pca_forward = PCA()
pc_forward = pca_forward.fit_transform(forward_standardized)
var_forward = pca_forward.explained_variance_ratio_

print(f"Forward PCA - PC1: {var_forward[0]*100:.1f}%, PC2: {var_forward[1]*100:.1f}%, PC3: {var_forward[2]*100:.1f}%")

# PCA para Backward
print("Calculando PCA para Backward...")
scaler_backward = StandardScaler()
backward_standardized = scaler_backward.fit_transform(derivatives_backward)

pca_backward = PCA()
pc_backward = pca_backward.fit_transform(backward_standardized)
var_backward = pca_backward.explained_variance_ratio_

print(f"Backward PCA - PC1: {var_backward[0]*100:.1f}%, PC2: {var_backward[1]*100:.1f}%, PC3: {var_backward[2]*100:.1f}%")

# Suavizado para visualizacion 3D
def smooth_trajectory_3d(pc_data, sigma=1.5, interp_factor=3):
    """Suavizar trayectoria 3D para mejor visualizacion"""
    if len(pc_data) < 10:
        return pc_data[:, 0], pc_data[:, 1], pc_data[:, 2]
    
    # Filtro Gaussiano
    pc1_smooth = gaussian_filter1d(pc_data[:, 0], sigma=sigma)
    pc2_smooth = gaussian_filter1d(pc_data[:, 1], sigma=sigma)
    pc3_smooth = gaussian_filter1d(pc_data[:, 2], sigma=sigma)
    
    # Interpolacion cubica para mas puntos
    t_orig = np.arange(len(pc_data))
    t_interp = np.linspace(0, len(pc_data)-1, len(pc_data)*interp_factor)
    
    f1 = interpolate.interp1d(t_orig, pc1_smooth, kind='cubic')
    f2 = interpolate.interp1d(t_orig, pc2_smooth, kind='cubic')
    f3 = interpolate.interp1d(t_orig, pc3_smooth, kind='cubic')
    
    return f1(t_interp), f2(t_interp), f3(t_interp)

# Suavizar ambas trayectorias
print("Suavizando trayectorias para visualizacion...")
pc1_f_smooth, pc2_f_smooth, pc3_f_smooth = smooth_trajectory_3d(pc_forward)
pc1_b_smooth, pc2_b_smooth, pc3_b_smooth = smooth_trajectory_3d(pc_backward)

# CREAR GRAFICO 3D COMBINADO
print("Generando grafico 3D combinado...")

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Colores para las trayectorias
colors_forward = plt.cm.Reds(np.linspace(0.4, 1.0, len(pc1_f_smooth)))
colors_backward = plt.cm.Blues(np.linspace(0.4, 1.0, len(pc1_b_smooth)))

# Dibujar trayectoria Forward (Roja)
print("Dibujando trayectoria Forward (roja)...")
for i in range(len(pc1_f_smooth) - 1):
    ax.plot([pc1_f_smooth[i], pc1_f_smooth[i+1]], 
            [pc2_f_smooth[i], pc2_f_smooth[i+1]], 
            [pc3_f_smooth[i], pc3_f_smooth[i+1]], 
            color=colors_forward[i], alpha=0.8, linewidth=2.0)

# Puntos especiales Forward
ax.scatter(pc_forward[0, 0], pc_forward[0, 1], pc_forward[0, 2], 
          color='darkred', s=200, marker='o', edgecolor='white', 
          linewidth=3, alpha=1.0, label='Forward Start')
ax.scatter(pc_forward[-1, 0], pc_forward[-1, 1], pc_forward[-1, 2], 
          color='red', s=200, marker='s', edgecolor='white', 
          linewidth=3, alpha=1.0, label='Forward End')

# Dibujar trayectoria Backward (Azul)
print("Dibujando trayectoria Backward (azul)...")
for i in range(len(pc1_b_smooth) - 1):
    ax.plot([pc1_b_smooth[i], pc1_b_smooth[i+1]], 
            [pc2_b_smooth[i], pc2_b_smooth[i+1]], 
            [pc3_b_smooth[i], pc3_b_smooth[i+1]], 
            color=colors_backward[i], alpha=0.8, linewidth=2.0)

# Puntos especiales Backward
ax.scatter(pc_backward[0, 0], pc_backward[0, 1], pc_backward[0, 2], 
          color='darkblue', s=200, marker='o', edgecolor='white', 
          linewidth=3, alpha=1.0, label='Backward Start')
ax.scatter(pc_backward[-1, 0], pc_backward[-1, 1], pc_backward[-1, 2], 
          color='blue', s=200, marker='s', edgecolor='white', 
          linewidth=3, alpha=1.0, label='Backward End')

# Configuracion de ejes
ax.set_xlabel(f'PC1 (F:{var_forward[0]*100:.1f}%, B:{var_backward[0]*100:.1f}%)', 
              fontsize=14, fontweight='bold')
ax.set_ylabel(f'PC2 (F:{var_forward[1]*100:.1f}%, B:{var_backward[1]*100:.1f}%)', 
              fontsize=14, fontweight='bold')
ax.set_zlabel(f'PC3 (F:{var_forward[2]*100:.1f}%, B:{var_backward[2]*100:.1f}%)', 
              fontsize=14, fontweight='bold')

# Titulo y leyenda
ax.set_title('PCA 3D Combinado: Derivadas Forward vs Backward\nC. elegans - Dinamicas Neurales Temporales', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper left')

# Configuracion visual del grafico 3D
ax.grid(True, alpha=0.3)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_alpha(0.1)
ax.yaxis.pane.set_alpha(0.1)
ax.zaxis.pane.set_alpha(0.1)

# Vista optimizada
ax.view_init(elev=20, azim=45)

# Informacion estadistica en el grafico
info_text = f"""Analisis Comparativo - Derivadas Temporales (dFR/dt)

FORWARD ({np.sum(forward_mask)} puntos):
  Varianza 3 PCs: {np.sum(var_forward[:3])*100:.1f}%
  |dFR/dt| promedio: {np.mean(np.abs(derivatives_forward)):.4f}

BACKWARD ({np.sum(backward_mask)} puntos):
  Varianza 3 PCs: {np.sum(var_backward[:3])*100:.1f}%
  |dFR/dt| promedio: {np.mean(np.abs(derivatives_backward)):.4f}

Interpretacion:
ROJO = Dinamicas durante avance
AZUL = Dinamicas durante retroceso
Magnitud = Velocidad de cambio neural"""

fig.text(0.02, 0.02, info_text, fontsize=10, 
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9),
         verticalalignment='bottom')

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)

# Guardar archivo
filename = 'PCA_3D_Forward_Backward_Derivatives.png'
plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"ARCHIVO GUARDADO: {filename}")

# Analisis estadistico final
print("\n" + "="*60)
print("ANALISIS ESTADISTICO COMPARATIVO")
print("="*60)

mean_abs_forward = np.mean(np.abs(derivatives_forward))
mean_abs_backward = np.mean(np.abs(derivatives_backward))
std_forward = np.std(derivatives_forward)
std_backward = np.std(derivatives_backward)

print(f"DINAMISMO NEURAL (|dFR/dt|):")
print(f"  Forward:  {mean_abs_forward:.6f} +- {std_forward:.6f}")
print(f"  Backward: {mean_abs_backward:.6f} +- {std_backward:.6f}")

if mean_abs_backward > mean_abs_forward:
    ratio = mean_abs_backward / mean_abs_forward
    print(f"  RESULTADO: Backward {ratio:.2f}x mas dinamico que Forward")
else:
    ratio = mean_abs_forward / mean_abs_backward
    print(f"  RESULTADO: Forward {ratio:.2f}x mas dinamico que Backward")

print(f"\nVARIANZA EXPLICADA:")
print(f"  Forward  - 3 PCs: {np.sum(var_forward[:3])*100:.1f}%")
print(f"  Backward - 3 PCs: {np.sum(var_backward[:3])*100:.1f}%")

print(f"\nCOMPARACION POR COMPONENTE:")
for i in range(3):
    print(f"  PC{i+1}: Forward {var_forward[i]*100:.1f}% vs Backward {var_backward[i]*100:.1f}%")

# Analisis de separabilidad en 3D
centroid_forward = np.mean(pc_forward[:, :3], axis=0)
centroid_backward = np.mean(pc_backward[:, :3], axis=0)
separation_3d = np.linalg.norm(centroid_forward - centroid_backward)

print(f"\nSEPARABILIDAD 3D:")
print(f"  Distancia entre centroides: {separation_3d:.4f}")
if separation_3d > 3.0:
    print("  SEPARABILIDAD: EXCELENTE - Patrones muy distintos")
elif separation_3d > 2.0:
    print("  SEPARABILIDAD: ALTA - Patrones claramente diferentes")
elif separation_3d > 1.0:
    print("  SEPARABILIDAD: MODERADA - Patrones diferenciables")
else:
    print("  SEPARABILIDAD: BAJA - Patrones similares")

print("="*60)
print("ANALISIS 3D COMBINADO COMPLETADO")
print("="*60)