"""
Vista 3D Especial - Superposición de Inicio y Final
Genera una vista donde el punto inicial y final están alineados verticalmente
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')  # Backend sin ventanas para evitar problemas

print("=== VISTA 3D ESPECIAL: INICIO Y FINAL SUPERPUESTOS ===")

# Cargar datos
nwb_file_path = "sub-2023-01-09-15-SWF702_ses-20230109_behavior+image+ophys.nwb"

with NWBHDF5IO(nwb_file_path, 'r') as io:
    nwbfile = io.read()
    
    calcium_module = nwbfile.processing['CalciumActivity']
    roi_response_series = calcium_module['SignalCalciumImResponseSeries']
    traces = roi_response_series.data[:]
    timestamps = roi_response_series.timestamps[:]

print(f"Datos cargados: {traces.shape[1]} neuronas, {traces.shape[0]} timepoints")

# Procesar datos
time_minutes = (timestamps - timestamps[0]) / 60
F0 = np.percentile(traces, 10, axis=0)
neural_matrix = (traces - F0) / F0
neural_matrix[np.isnan(neural_matrix)] = 0
neural_matrix[np.isinf(neural_matrix)] = 0

# PCA
scaler = StandardScaler()
neural_standardized = scaler.fit_transform(neural_matrix)

pca = PCA()
principal_components = pca.fit_transform(neural_standardized)
explained_variance = pca.explained_variance_ratio_

print(f"PC1: {explained_variance[0]*100:.1f}%, PC2: {explained_variance[1]*100:.1f}%, PC3: {explained_variance[2]*100:.1f}%")

# Suavizar datos para trayectorias más naturales
pc1_smooth = gaussian_filter1d(principal_components[:, 0], sigma=2.0)
pc2_smooth = gaussian_filter1d(principal_components[:, 1], sigma=2.0)
pc3_smooth = gaussian_filter1d(principal_components[:, 2], sigma=2.0)

# Interpolar para más puntos
t_orig = np.arange(len(principal_components))
t_interp = np.linspace(0, len(principal_components)-1, len(principal_components)*3)

f1 = interpolate.interp1d(t_orig, pc1_smooth, kind='cubic')
f2 = interpolate.interp1d(t_orig, pc2_smooth, kind='cubic')
f3 = interpolate.interp1d(t_orig, pc3_smooth, kind='cubic')

pc1_final = f1(t_interp)
pc2_final = f2(t_interp)
pc3_final = f3(t_interp)

# Calcular ángulos para vista superpuesta
start_point = np.array([principal_components[0, 0], principal_components[0, 1], principal_components[0, 2]])
end_point = np.array([principal_components[-1, 0], principal_components[-1, 1], principal_components[-1, 2]])

# Vector desde inicio hasta final
vector_inicio_final = end_point - start_point

# Calcular ángulos para que este vector se vea vertical
azim_overlay = np.degrees(np.arctan2(vector_inicio_final[1], vector_inicio_final[0]))
elev_overlay = 85  # Vista casi desde arriba

print(f"Calculando vista superpuesta: azimut={azim_overlay:.1f}°, elevación={elev_overlay}°")

# CREAR FIGURA CON 4 VISTAS
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')

# Colores temporales
colors = plt.cm.viridis(np.linspace(0, 1, len(pc1_final)))

# Vista 1: Estándar
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
for i in range(len(pc1_final)-1):
    ax1.plot([pc1_final[i], pc1_final[i+1]], 
             [pc2_final[i], pc2_final[i+1]], 
             [pc3_final[i], pc3_final[i+1]], 
             color=colors[i], alpha=0.7, linewidth=1.8)

ax1.scatter(start_point[0], start_point[1], start_point[2], 
           color='blue', s=200, marker='o', edgecolor='white', linewidth=3, label='INICIO')
ax1.scatter(end_point[0], end_point[1], end_point[2], 
           color='red', s=200, marker='s', edgecolor='white', linewidth=3, label='FINAL')

ax1.set_title('Vista Estándar 3D\n(Perspectiva Normal)', fontsize=16, fontweight='bold')
ax1.view_init(elev=30, azim=45)
ax1.legend(fontsize=12)

# Vista 2: SUPERPUESTA (Principal)
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
for i in range(len(pc1_final)-1):
    ax2.plot([pc1_final[i], pc1_final[i+1]], 
             [pc2_final[i], pc2_final[i+1]], 
             [pc3_final[i], pc3_final[i+1]], 
             color=colors[i], alpha=0.8, linewidth=2.0)

ax2.scatter(start_point[0], start_point[1], start_point[2], 
           color='darkblue', s=300, marker='o', edgecolor='white', linewidth=4, label='INICIO')
ax2.scatter(end_point[0], end_point[1], end_point[2], 
           color='darkred', s=300, marker='s', edgecolor='yellow', linewidth=4, label='FINAL')

ax2.set_title('🎯 VISTA SUPERPUESTA\nInicio y Final Alineados Verticalmente', 
              fontsize=16, fontweight='bold', color='darkred')
ax2.view_init(elev=elev_overlay, azim=azim_overlay)
ax2.legend(fontsize=12)

# Vista 3: Vista Lateral
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
for i in range(len(pc1_final)-1):
    ax3.plot([pc1_final[i], pc1_final[i+1]], 
             [pc2_final[i], pc2_final[i+1]], 
             [pc3_final[i], pc3_final[i+1]], 
             color=colors[i], alpha=0.7, linewidth=1.8)

ax3.scatter(start_point[0], start_point[1], start_point[2], 
           color='blue', s=200, marker='o', edgecolor='white', linewidth=3)
ax3.scatter(end_point[0], end_point[1], end_point[2], 
           color='red', s=200, marker='s', edgecolor='white', linewidth=3)

ax3.set_title('Vista Lateral\n(Perfil de la Trayectoria)', fontsize=16, fontweight='bold')
ax3.view_init(elev=0, azim=0)

# Vista 4: Vista Superior
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
for i in range(len(pc1_final)-1):
    ax4.plot([pc1_final[i], pc1_final[i+1]], 
             [pc2_final[i], pc2_final[i+1]], 
             [pc3_final[i], pc3_final[i+1]], 
             color=colors[i], alpha=0.7, linewidth=1.8)

ax4.scatter(start_point[0], start_point[1], start_point[2], 
           color='blue', s=200, marker='o', edgecolor='white', linewidth=3)
ax4.scatter(end_point[0], end_point[1], end_point[2], 
           color='red', s=200, marker='s', edgecolor='white', linewidth=3)

ax4.set_title('Vista Superior\n(Proyección desde Arriba)', fontsize=16, fontweight='bold')
ax4.view_init(elev=90, azim=0)

# Configurar todos los ejes
axes = [ax1, ax2, ax3, ax4]
for i, ax in enumerate(axes):
    ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)', fontweight='bold')
    ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)', fontweight='bold')
    ax.set_zlabel(f'PC3 ({explained_variance[2]*100:.1f}%)', fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

# Título general
plt.suptitle('Análisis 3D Multi-Perspectiva: Trayectorias Neurales de C. elegans\n' + 
             'Vista Principal Muestra Superposición de Inicio y Final (Comportamiento Cíclico)', 
             fontsize=18, fontweight='bold', y=0.95)

plt.tight_layout()
plt.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.2)

# Guardar archivo
filename = 'PCA_3D_Overlay_MultiView.png'
plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

print(f"✅ Archivo guardado: {filename}")

# Análisis numérico
distance_inicio_final = np.linalg.norm(end_point - start_point)
total_range = np.ptp(principal_components[:, :3])

print("\n" + "="*60)
print("ANÁLISIS DE CICLICIDAD")
print("="*60)
print(f"Distancia inicio-final: {distance_inicio_final:.4f}")
print(f"Rango total de variación: {total_range:.4f}")
print(f"Ratio (distancia/rango): {distance_inicio_final/total_range:.3f}")

if distance_inicio_final/total_range < 0.15:
    print("🔄 TRAYECTORIA ALTAMENTE CÍCLICA")
elif distance_inicio_final/total_range < 0.35:
    print("🔄 TRAYECTORIA MODERADAMENTE CÍCLICA")
else:
    print("➡️  TRAYECTORIA MÁS LINEAL")

print(f"\nVista superpuesta configurada con:")
print(f"- Elevación: {elev_overlay}° (vista desde arriba)")
print(f"- Azimut: {azim_overlay:.1f}° (rotación horizontal)")
print(f"- Esta perspectiva alinea el vector inicio-final verticalmente")
print("="*60)

plt.close()
print("Proceso completado exitosamente!")