import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as patches

# Configurar matplotlib para mostrar gráficos
plt.ion()
import matplotlib
matplotlib.use('TkAgg')

print("=== GENERADOR PCA TRAYECTORIAS TEMPORALES ===")
print("Creando visualización estilo trayectorias con rueda de colores")

# Cargar datos
df = pd.read_csv('neural_data_dataframe.csv', index_col=0)
neural_matrix = df.values
scaler = StandardScaler()
neural_matrix_scaled = scaler.fit_transform(neural_matrix)
pca = PCA()
principal_components = pca.fit_transform(neural_matrix_scaled)
explained_variance_ratio = pca.explained_variance_ratio_

# Tiempo corregido
time_minutes = np.linspace(0, 16, 1615)

print(f"Datos cargados: {neural_matrix.shape}")
print(f"PC1: {explained_variance_ratio[0]*100:.1f}%, PC2: {explained_variance_ratio[1]*100:.1f}%")

# Crear figura con el estilo solicitado
fig, ax = plt.subplots(figsize=(12, 10))

# Configurar colormap circular (como rueda de colores)
# Usar HSV para crear efecto de rueda de colores
colors = plt.cm.hsv(np.linspace(0, 1, len(time_minutes)))

# Plotear trayectorias como líneas conectadas
pc1 = principal_components[:, 0]
pc2 = principal_components[:, 1]

# Crear segmentos de líneas coloreadas
for i in range(len(pc1)-1):
    ax.plot([pc1[i], pc1[i+1]], [pc2[i], pc2[i+1]], 
            color=colors[i], linewidth=1.5, alpha=0.8)

# Agregar puntos en posiciones clave para marcar el progreso
key_points = [0, len(pc1)//4, len(pc1)//2, 3*len(pc1)//4, len(pc1)-1]
for i, point in enumerate(key_points):
    ax.scatter(pc1[point], pc2[point], c=colors[point], s=50, 
              edgecolors='black', linewidth=1, zorder=5)

# Agregar flechas naranjas para indicar dirección temporal
arrow_positions = [len(pc1)//6, len(pc1)//3, 2*len(pc1)//3, 5*len(pc1)//6]
for pos in arrow_positions:
    if pos < len(pc1)-10:
        # Calcular dirección de la flecha
        dx = pc1[pos+10] - pc1[pos]
        dy = pc2[pos+10] - pc2[pos]
        
        # Normalizar y escalar
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx_norm = dx / length * 0.03
            dy_norm = dy / length * 0.03
            
            # Crear flecha curva
            arrow = FancyArrowPatch((pc1[pos], pc2[pos]), 
                                  (pc1[pos] + dx_norm, pc2[pos] + dy_norm),
                                  connectionstyle="arc3,rad=0.2", 
                                  arrowstyle='->', 
                                  mutation_scale=20,
                                  color='darkorange', 
                                  linewidth=3,
                                  zorder=10)
            ax.add_patch(arrow)

# Configurar ejes
ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)', fontsize=14, fontweight='bold')
ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)', fontsize=14, fontweight='bold')
ax.set_title('Neural Activity Trajectory in PC Space\nC. elegans Temporal Dynamics', 
             fontsize=16, fontweight='bold', pad=20)

# Configurar grid y estilo
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('white')

# Agregar líneas de referencia en (0,0)
ax.axhline(y=0, color='lightgray', linestyle='-', alpha=0.5, zorder=0)
ax.axvline(x=0, color='lightgray', linestyle='-', alpha=0.5, zorder=0)

# Crear rueda de colores como inset
# Posición de la rueda de colores (esquina superior derecha)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
inset_ax = inset_axes(ax, width="20%", height="20%", loc='upper right', 
                      bbox_to_anchor=(0.02, 0.02, 1, 1), bbox_transform=ax.transAxes)

# Crear rueda de colores
theta = np.linspace(0, 2*np.pi, 100)
radius_outer = 1.0
radius_inner = 0.6

for i in range(len(theta)-1):
    # Crear segmento de la rueda
    theta_seg = np.array([theta[i], theta[i+1], theta[i+1], theta[i]])
    r_seg = np.array([radius_inner, radius_inner, radius_outer, radius_outer])
    
    x_seg = r_seg * np.cos(theta_seg)
    y_seg = r_seg * np.sin(theta_seg)
    
    color_hsv = plt.cm.hsv(i / (len(theta)-1))
    inset_ax.fill(x_seg, y_seg, color=color_hsv, edgecolor='none')

# Agregar punto negro en el centro
inset_ax.scatter(0, 0, c='black', s=50, zorder=10)

# Configurar rueda de colores
inset_ax.set_xlim(-1.2, 1.2)
inset_ax.set_ylim(-1.2, 1.2)
inset_ax.set_aspect('equal')
inset_ax.axis('off')

# Agregar etiquetas de tiempo en la rueda
time_labels = ['0 min', '4 min', '8 min', '12 min', '16 min']
time_angles = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
for label, angle in zip(time_labels[:-1], time_angles[:-1]):  # Excluir el último (mismo que primero)
    x_label = 1.4 * np.cos(angle)
    y_label = 1.4 * np.sin(angle)
    inset_ax.text(x_label, y_label, label, ha='center', va='center', 
                 fontsize=8, fontweight='bold')

# Agregar texto indicativo
ax.text(0.02, 0.98, 'Time Direction →', transform=ax.transAxes, 
        fontsize=12, fontweight='bold', color='darkorange', 
        verticalalignment='top')

plt.tight_layout()
plt.savefig('PCA_Trajectory_Style.png', dpi=300, bbox_inches='tight')
print("\nMostrando visualización PCA estilo trayectorias...")
plt.show()
print("Gráfico guardado como: PCA_Trajectory_Style.png")
plt.pause(4)

print("\n" + "="*60)
print("VISUALIZACIÓN PCA TRAYECTORIAS COMPLETADA!")
print("Características:")
print("- Trayectorias temporales coloreadas")
print("- Rueda de colores para referencia temporal")
print("- Flechas naranjas indicando dirección")
print("- Estilo elegante y profesional")
print("="*60)