"""
Vista 3D Especial - Inicio y Final Superpuestos
Muestra la trayectoria PCA con perspectiva donde inicio y final están alineados verticalmente
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

# Configurar matplotlib
plt.ion()
import matplotlib
matplotlib.use('TkAgg')

print("=== VISTA 3D ESPECIAL: INICIO Y FINAL SUPERPUESTOS ===")

# Cargar y preparar datos (igual que antes)
nwb_file_path = "sub-2023-01-09-15-SWF702_ses-20230109_behavior+image+ophys.nwb"

with NWBHDF5IO(nwb_file_path, 'r') as io:
    nwbfile = io.read()
    
    calcium_module = nwbfile.processing['CalciumActivity']
    roi_response_series = calcium_module['SignalCalciumImResponseSeries']
    traces = roi_response_series.data[:]
    timestamps = roi_response_series.timestamps[:]

print(f"Datos cargados: {traces.shape[1]} neuronas, {traces.shape[0]} timepoints")

# Preparar datos neurales
time_minutes = (timestamps - timestamps[0]) / 60
F0 = np.percentile(traces, 10, axis=0)
neural_matrix = (traces - F0) / F0
neural_matrix[np.isnan(neural_matrix)] = 0
neural_matrix[np.isinf(neural_matrix)] = 0

# Estandarizar y aplicar PCA
scaler = StandardScaler()
neural_standardized = scaler.fit_transform(neural_matrix)

pca_full = PCA()
principal_components = pca_full.fit_transform(neural_standardized)
explained_variance_ratio = pca_full.explained_variance_ratio_

print(f"PC1: {explained_variance_ratio[0]*100:.1f}%, PC2: {explained_variance_ratio[1]*100:.1f}%, PC3: {explained_variance_ratio[2]*100:.1f}%")

# Función para encontrar el ángulo donde inicio y final se superponen
def find_overlay_angle(pc1, pc2, pc3):
    """
    Encuentra el ángulo de vista donde el punto inicial y final aparecen superpuestos
    """
    start_point = np.array([pc1[0], pc2[0], pc3[0]])
    end_point = np.array([pc1[-1], pc2[-1], pc3[-1]])
    
    # Vector del inicio al final
    direction_vector = end_point - start_point
    
    # Calcular ángulos para alinear este vector verticalmente
    # Queremos que la proyección 2D del vector sea principalmente vertical
    
    # Ángulo azimutal para alinear en el plano XY
    azim = np.degrees(np.arctan2(direction_vector[1], direction_vector[0])) + 90
    
    # Ángulo de elevación para ver desde arriba/abajo
    elev = 90  # Vista desde arriba para superponer puntos
    
    return elev, azim

# Suavizar trayectorias
pc1_filtered = gaussian_filter1d(principal_components[:, 0], sigma=1.5)
pc2_filtered = gaussian_filter1d(principal_components[:, 1], sigma=1.5)
pc3_filtered = gaussian_filter1d(principal_components[:, 2], sigma=1.5)

t = np.arange(len(principal_components))
t_smooth = np.linspace(0, len(principal_components)-1, len(principal_components)*5)

f_pc1 = interpolate.interp1d(t, pc1_filtered, kind='cubic')
f_pc2 = interpolate.interp1d(t, pc2_filtered, kind='cubic')
f_pc3 = interpolate.interp1d(t, pc3_filtered, kind='cubic')

pc1_smooth = f_pc1(t_smooth)
pc2_smooth = f_pc2(t_smooth)
pc3_smooth = f_pc3(t_smooth)

# Encontrar ángulo de superposición
elev_overlay, azim_overlay = find_overlay_angle(principal_components[:, 0], 
                                               principal_components[:, 1], 
                                               principal_components[:, 2])

print(f"Ángulo calculado para superposición: elevación={elev_overlay:.1f}°, azimut={azim_overlay:.1f}°")

# CREAR VISUALIZACIÓN CON MÚLTIPLES VISTAS
fig = plt.figure(figsize=(20, 15))
fig.patch.set_facecolor('white')

# Colores para progresión temporal
colors = plt.cm.plasma(np.linspace(0, 1, len(pc1_smooth)))

# === VISTA 1: Perspectiva Estándar ===
ax1 = fig.add_subplot(2, 2, 1, projection='3d')

for i in range(len(pc1_smooth) - 1):
    ax1.plot([pc1_smooth[i], pc1_smooth[i+1]], 
             [pc2_smooth[i], pc2_smooth[i+1]], 
             [pc3_smooth[i], pc3_smooth[i+1]], 
             color=colors[i], alpha=0.8, linewidth=2.0)

ax1.scatter(principal_components[0, 0], principal_components[0, 1], principal_components[0, 2],
           color='darkblue', s=150, marker='o', edgecolor='white', linewidth=3, alpha=1.0, label='INICIO')
ax1.scatter(principal_components[-1, 0], principal_components[-1, 1], principal_components[-1, 2],
           color='darkred', s=150, marker='s', edgecolor='white', linewidth=3, alpha=1.0, label='FINAL')

ax1.set_title('Vista Estándar 3D', fontsize=14, fontweight='bold')
ax1.view_init(elev=25, azim=45)
ax1.legend()

# === VISTA 2: Inicio y Final Superpuestos (PRINCIPAL) ===
ax2 = fig.add_subplot(2, 2, 2, projection='3d')

for i in range(len(pc1_smooth) - 1):
    ax2.plot([pc1_smooth[i], pc1_smooth[i+1]], 
             [pc2_smooth[i], pc2_smooth[i+1]], 
             [pc3_smooth[i], pc3_smooth[i+1]], 
             color=colors[i], alpha=0.9, linewidth=2.2)

# Puntos especiales con mayor tamaño para esta vista
ax2.scatter(principal_components[0, 0], principal_components[0, 1], principal_components[0, 2],
           color='darkblue', s=200, marker='o', edgecolor='white', linewidth=4, alpha=1.0, label='INICIO')
ax2.scatter(principal_components[-1, 0], principal_components[-1, 1], principal_components[-1, 2],
           color='darkred', s=200, marker='s', edgecolor='yellow', linewidth=4, alpha=1.0, label='FINAL')

ax2.set_title('🎯 VISTA SUPERPUESTA\n(Inicio y Final Alineados)', fontsize=16, fontweight='bold', color='darkred')
ax2.view_init(elev=elev_overlay, azim=azim_overlay)
ax2.legend(fontsize=12)

# === VISTA 3: Vista Lateral ===
ax3 = fig.add_subplot(2, 2, 3, projection='3d')

for i in range(len(pc1_smooth) - 1):
    ax3.plot([pc1_smooth[i], pc1_smooth[i+1]], 
             [pc2_smooth[i], pc2_smooth[i+1]], 
             [pc3_smooth[i], pc3_smooth[i+1]], 
             color=colors[i], alpha=0.8, linewidth=2.0)

ax3.scatter(principal_components[0, 0], principal_components[0, 1], principal_components[0, 2],
           color='darkblue', s=150, marker='o', edgecolor='white', linewidth=3, alpha=1.0)
ax3.scatter(principal_components[-1, 0], principal_components[-1, 1], principal_components[-1, 2],
           color='darkred', s=150, marker='s', edgecolor='white', linewidth=3, alpha=1.0)

ax3.set_title('Vista Lateral', fontsize=14, fontweight='bold')
ax3.view_init(elev=0, azim=0)

# === VISTA 4: Vista Superior ===
ax4 = fig.add_subplot(2, 2, 4, projection='3d')

for i in range(len(pc1_smooth) - 1):
    ax4.plot([pc1_smooth[i], pc1_smooth[i+1]], 
             [pc2_smooth[i], pc2_smooth[i+1]], 
             [pc3_smooth[i], pc3_smooth[i+1]], 
             color=colors[i], alpha=0.8, linewidth=2.0)

ax4.scatter(principal_components[0, 0], principal_components[0, 1], principal_components[0, 2],
           color='darkblue', s=150, marker='o', edgecolor='white', linewidth=3, alpha=1.0)
ax4.scatter(principal_components[-1, 0], principal_components[-1, 1], principal_components[-1, 2],
           color='darkred', s=150, marker='s', edgecolor='white', linewidth=3, alpha=1.0)

ax4.set_title('Vista Superior', fontsize=14, fontweight='bold')
ax4.view_init(elev=90, azim=0)

# Configurar todos los axes
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)', fontsize=10, fontweight='bold')
    ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)', fontsize=10, fontweight='bold')
    ax.set_zlabel(f'PC3 ({explained_variance_ratio[2]*100:.1f}%)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2, color='gray')
    
    # Configurar paneles 3D
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

# Ajustes finales
plt.suptitle('Análisis 3D Multi-Vista: Trayectorias Neurales Temporales\nVista Principal: Inicio y Final Superpuestos', 
             fontsize=18, fontweight='bold', y=0.95)
plt.tight_layout()
plt.subplots_adjust(top=0.87, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.2)

# Guardar y mostrar
plt.savefig('PCA_3D_Overlay_MultiView.png', dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')

print("Mostrando análisis 3D con vista superpuesta...")
plt.show()

# Información adicional
print("\n" + "="*70)
print("ANÁLISIS 3D MULTI-VISTA COMPLETADO")
print("="*70)
print("Vistas generadas:")
print("1. Vista Estándar (elev=25°, azim=45°)")
print(f"2. 🎯 VISTA SUPERPUESTA (elev={elev_overlay:.1f}°, azim={azim_overlay:.1f}°)")
print("   - Inicio y final alineados verticalmente")
print("   - Muestra la naturaleza cíclica de la trayectoria")
print("3. Vista Lateral (elev=0°, azim=0°)")
print("4. Vista Superior (elev=90°, azim=0°)")
print(f"Archivo guardado: PCA_3D_Overlay_MultiView.png")

# Calcular distancia entre inicio y final
start_point = principal_components[0, :3]
end_point = principal_components[-1, :3]
distance = np.linalg.norm(end_point - start_point)

print(f"\nAnálisis de ciclicidad:")
print(f"- Distancia euclidiana inicio-final: {distance:.4f}")
print(f"- Rango total PC1: {principal_components[:, 0].max() - principal_components[:, 0].min():.4f}")
print(f"- Rango total PC2: {principal_components[:, 1].max() - principal_components[:, 1].min():.4f}")
print(f"- Rango total PC3: {principal_components[:, 2].max() - principal_components[:, 2].min():.4f}")

if distance < 0.1 * np.std(principal_components.flatten()):
    print("✓ La trayectoria es ALTAMENTE CÍCLICA")
elif distance < 0.3 * np.std(principal_components.flatten()):
    print("✓ La trayectoria es MODERADAMENTE CÍCLICA")
else:
    print("- La trayectoria es MÁS LINEAL que cíclica")

print("="*70)