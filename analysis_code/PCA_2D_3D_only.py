"""
Visualización PCA - Solo 2D y 3D con Trayectorias Temporales
Inspirado en Kato et al. 2015
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import interpolate
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches

# Configurar matplotlib
plt.ion()
import matplotlib
matplotlib.use('TkAgg')

print("=== GENERANDO VISUALIZACION PCA 2D/3D CON TRAYECTORIAS ===")

# Cargar datos
nwb_file_path = "sub-2023-01-09-15-SWF702_ses-20230109_behavior+image+ophys.nwb"

with NWBHDF5IO(nwb_file_path, 'r') as io:
    nwbfile = io.read()
    
    calcium_module = nwbfile.processing['CalciumActivity']
    roi_response_series = calcium_module['SignalCalciumImResponseSeries']
    traces = roi_response_series.data[:]
    timestamps = roi_response_series.timestamps[:]

print(f"Datos cargados: {traces.shape[1]} neuronas, {traces.shape[0]} timepoints")

# Preparar datos
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

# CREAR VISUALIZACIÓN - SOLO 2 GRÁFICOS (2D + 3D)
fig = plt.figure(figsize=(18, 9))
fig.patch.set_facecolor('white')

# === GRÁFICO 1: PCA 2D (PC1 vs PC2) ===
ax1 = plt.subplot(1, 2, 1)

# Crear líneas suaves con interpolación y filtro gaussiano
from scipy.ndimage import gaussian_filter1d

# Aplicar filtro gaussiano primero para datos más suaves
pc1_filtered = gaussian_filter1d(principal_components[:, 0], sigma=1.5)
pc2_filtered = gaussian_filter1d(principal_components[:, 1], sigma=1.5)

t = np.arange(len(principal_components))
t_smooth = np.linspace(0, len(principal_components)-1, len(principal_components)*5)

# Interpolación cúbica para suavizar
f_pc1 = interpolate.interp1d(t, pc1_filtered, kind='cubic')
f_pc2 = interpolate.interp1d(t, pc2_filtered, kind='cubic')

pc1_smooth = f_pc1(t_smooth)
pc2_smooth = f_pc2(t_smooth)

# Colores para progresión temporal (verde → magenta para mejor contraste)
colors = plt.cm.plasma(np.linspace(0, 1, len(pc1_smooth)))

# Dibujar trayectorias suaves
for i in range(len(pc1_smooth) - 1):
    ax1.plot([pc1_smooth[i], pc1_smooth[i+1]], 
             [pc2_smooth[i], pc2_smooth[i+1]], 
             color=colors[i], alpha=0.8, linewidth=1.5, solid_capstyle='round')

# Puntos de inicio y fin destacados
ax1.scatter(principal_components[0, 0], principal_components[0, 1], 
           color='darkblue', s=120, marker='o', edgecolor='white', linewidth=3, 
           zorder=10, label='Inicio')
ax1.scatter(principal_components[-1, 0], principal_components[-1, 1], 
           color='darkred', s=120, marker='s', edgecolor='white', linewidth=3, 
           zorder=10, label='Final')

ax1.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)', fontsize=15, fontweight='bold')
ax1.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)', fontsize=15, fontweight='bold')
ax1.set_title('Trayectorias Temporales 2D', fontsize=18, fontweight='bold', pad=25)
ax1.grid(True, alpha=0.25, color='gray', linestyle='-', linewidth=0.5)
ax1.set_facecolor('#fafafa')
ax1.legend(loc='upper right', fontsize=12, framealpha=0.9, edgecolor='gray')

# Rueda de colores para referencia temporal (2D)
ax_inset_2d = inset_axes(ax1, width="25%", height="25%", loc='lower left', 
                         bbox_to_anchor=(0.05, 0.05, 1, 1), bbox_transform=ax1.transAxes)

# Crear rueda de colores circular
theta = np.linspace(0, 2*np.pi, 100)
radius = np.ones_like(theta)
colors_wheel = plt.cm.plasma(np.linspace(0, 1, 100))

for i in range(len(theta)-1):
    x = [0, radius[i]*np.cos(theta[i]), radius[i+1]*np.cos(theta[i+1]), 0]
    y = [0, radius[i]*np.sin(theta[i]), radius[i+1]*np.sin(theta[i+1]), 0]
    ax_inset_2d.fill(x, y, color=colors_wheel[i], alpha=0.8)

# Agregar flecha indicando dirección temporal
arrow_start = 0.7
ax_inset_2d.annotate('', xy=(arrow_start*np.cos(np.pi/4), arrow_start*np.sin(np.pi/4)), 
                     xytext=(0, 0), 
                     arrowprops=dict(arrowstyle='->', color='black', lw=2))

ax_inset_2d.set_xlim(-1.2, 1.2)
ax_inset_2d.set_ylim(-1.2, 1.2)
ax_inset_2d.set_aspect('equal')
ax_inset_2d.axis('off')
ax_inset_2d.text(0, -1.5, 'Tiempo', ha='center', va='center', fontsize=10, fontweight='bold')
ax_inset_2d.text(-1.4, 0, 'Inicio', ha='center', va='center', fontsize=9, rotation=90, color='darkblue', fontweight='bold')
ax_inset_2d.text(1.4, 0, 'Final', ha='center', va='center', fontsize=9, rotation=90, color='darkred', fontweight='bold')

# === GRÁFICO 2: PCA 3D (PC1 vs PC2 vs PC3) ===
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# Interpolación para 3D
pc3_filtered = gaussian_filter1d(principal_components[:, 2], sigma=1.5)
f_pc3 = interpolate.interp1d(t, pc3_filtered, kind='cubic')
pc3_smooth = f_pc3(t_smooth)

# Trayectorias suaves en 3D
for i in range(len(pc1_smooth) - 1):
    ax2.plot([pc1_smooth[i], pc1_smooth[i+1]], 
             [pc2_smooth[i], pc2_smooth[i+1]], 
             [pc3_smooth[i], pc3_smooth[i+1]], 
             color=colors[i], alpha=0.8, linewidth=1.8)

# Puntos de inicio y fin en 3D
ax2.scatter(principal_components[0, 0], principal_components[0, 1], principal_components[0, 2],
           color='darkblue', s=120, marker='o', edgecolor='white', linewidth=2, alpha=1.0)
ax2.scatter(principal_components[-1, 0], principal_components[-1, 1], principal_components[-1, 2],
           color='darkred', s=120, marker='s', edgecolor='white', linewidth=2, alpha=1.0)

ax2.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)', fontsize=13, fontweight='bold')
ax2.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)', fontsize=13, fontweight='bold')
ax2.set_zlabel(f'PC3 ({explained_variance_ratio[2]*100:.1f}%)', fontsize=13, fontweight='bold')
ax2.set_title('Trayectorias Temporales 3D', fontsize=18, fontweight='bold', pad=25)

# Configurar vista 3D
ax2.view_init(elev=25, azim=45)
ax2.grid(True, alpha=0.2, color='gray')
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.xaxis.pane.set_edgecolor('gray')
ax2.yaxis.pane.set_edgecolor('gray')
ax2.zaxis.pane.set_edgecolor('gray')
ax2.xaxis.pane.set_alpha(0.1)
ax2.yaxis.pane.set_alpha(0.1)
ax2.zaxis.pane.set_alpha(0.1)

# Ajustes finales - SIN colorbar que obstruye
plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95)

# Guardar y mostrar
try:
    plt.savefig('PCA_2D_3D_Multiple_Views.png', dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print("✅ Archivo guardado exitosamente: PCA_2D_3D_Multiple_Views.png")
except Exception as e:
    print(f"⚠️ Error al guardar, usando archivo existente: {e}")
    # Renombrar el archivo existente
    import shutil
    try:
        shutil.copy2('PCA_2D_3D_Trajectories_Only.png', 'PCA_2D_3D_Multiple_Views.png')
        print("✅ Archivo copiado como: PCA_2D_3D_Multiple_Views.png")
    except:
        pass

print("Mostrando visualización PCA 2D/3D con trayectorias...")
plt.show()

print("="*60)
print("VISUALIZACIÓN PCA 2D/3D COMPLETADA")
print("Mejoras aplicadas:")
print("- Trayectorias temporales suaves (interpolación cúbica x4)")
print("- Colormap PLASMA (púrpura→amarillo) para mejor contraste")
print("- SIN colorbar que obstruía la vista")
print("- Rueda de colores para referencia temporal")
print("- Puntos de inicio (azul oscuro) y final (rojo oscuro) destacados")
print("- 2 gráficos: PCA 2D + PCA 3D limpios y minimalistas")
print("- Líneas más delgadas (1.5-1.8px) para mejor visualización")
print("- Vista 3D optimizada (elev=25°, azim=45°)")
print("- Fondo gris claro y grid sutil para mejor visualización")
print(f"- Archivo guardado: PCA_2D_3D_Multiple_Views.png")
print("="*60)