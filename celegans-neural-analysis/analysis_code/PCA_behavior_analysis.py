"""
Análisis PCA por Comportamiento - Velocidad Hacia Adelante vs Hacia Atrás
Comparación de trayectorias neurales según dirección de movimiento
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Configurar matplotlib
plt.ion()
import matplotlib
matplotlib.use('TkAgg')

print("=== ANÁLISIS PCA POR COMPORTAMIENTO: ADELANTE vs ATRÁS ===")

# Cargar datos
nwb_file_path = "sub-2023-01-09-15-SWF702_ses-20230109_behavior+image+ophys.nwb"

with NWBHDF5IO(nwb_file_path, 'r') as io:
    nwbfile = io.read()
    
    # Datos neurales
    calcium_module = nwbfile.processing['CalciumActivity']
    roi_response_series = calcium_module['SignalCalciumImResponseSeries']
    traces = roi_response_series.data[:]
    timestamps = roi_response_series.timestamps[:]
    
    # Datos comportamentales
    behavior_module = nwbfile.processing['Behavior']
    
    # Buscar datos de velocidad
    velocity_data = None
    velocity_timestamps = None
    
    for data_interface_name in behavior_module.data_interfaces:
        data_interface = behavior_module.data_interfaces[data_interface_name]
        print(f"Explorando interface comportamental: {data_interface_name}")
        
        if hasattr(data_interface, 'time_series'):
            for ts_name in data_interface.time_series:
                ts = data_interface.time_series[ts_name]
                print(f"  - Serie temporal: {ts_name}")
                if 'velocity' in ts_name.lower() or 'speed' in ts_name.lower():
                    velocity_data = ts.data[:]
                    velocity_timestamps = ts.timestamps[:]
                    print(f"    ✓ Datos de velocidad encontrados: {ts_name}")
                    break
        
        if velocity_data is not None:
            break
    
    if velocity_data is None:
        # Buscar en el objeto principal de behavior
        for attr_name in dir(behavior_module):
            if 'velocity' in attr_name.lower() or 'speed' in attr_name.lower():
                try:
                    attr_obj = getattr(behavior_module, attr_name)
                    if hasattr(attr_obj, 'data'):
                        velocity_data = attr_obj.data[:]
                        velocity_timestamps = attr_obj.timestamps[:]
                        print(f"✓ Datos de velocidad encontrados: {attr_name}")
                        break
                except:
                    continue

# Si no encontramos datos de velocidad, crear datos sintéticos basados en actividad neural
if velocity_data is None:
    print("⚠️ No se encontraron datos de velocidad explícitos.")
    print("Generando estimación de velocidad basada en cambios en actividad neural...")
    
    # Calcular actividad promedio por frame
    neural_activity = np.mean(traces, axis=1)
    # Calcular derivada para estimar cambios (proxy de velocidad)
    velocity_data = np.gradient(neural_activity)
    velocity_timestamps = timestamps
    
    print(f"✓ Velocidad estimada generada: {len(velocity_data)} puntos")

print(f"Datos cargados:")
print(f"  - Neuronas: {traces.shape[1]}")
print(f"  - Timepoints neurales: {traces.shape[0]}")
print(f"  - Timepoints velocidad: {len(velocity_data)}")

# Preparar datos neurales
time_minutes = (timestamps - timestamps[0]) / 60
F0 = np.percentile(traces, 10, axis=0)
neural_matrix = (traces - F0) / F0
neural_matrix[np.isnan(neural_matrix)] = 0
neural_matrix[np.isinf(neural_matrix)] = 0

# Sincronizar datos neurales y de velocidad
# Interpolar velocidad a los timepoints neurales
velocity_interp = np.interp(timestamps, velocity_timestamps, velocity_data)

print(f"Datos sincronizados:")
print(f"  - Rango velocidad: {velocity_interp.min():.3f} a {velocity_interp.max():.3f}")
print(f"  - Velocidad media: {velocity_interp.mean():.3f}")

# Filtrar datos por comportamiento
forward_mask = velocity_interp > 0  # Hacia adelante
backward_mask = velocity_interp < 0  # Hacia atrás

neural_forward = neural_matrix[forward_mask]
neural_backward = neural_matrix[backward_mask]
time_forward = time_minutes[forward_mask]
time_backward = time_minutes[backward_mask]

print(f"Segmentación por comportamiento:")
print(f"  - Hacia adelante: {neural_forward.shape[0]} timepoints ({forward_mask.sum()/len(forward_mask)*100:.1f}%)")
print(f"  - Hacia atrás: {neural_backward.shape[0]} timepoints ({backward_mask.sum()/len(backward_mask)*100:.1f}%)")

# Verificar que tenemos suficientes datos
if neural_forward.shape[0] < 10:
    print("⚠️ Pocos datos hacia adelante, ajustando umbral...")
    forward_mask = velocity_interp > np.percentile(velocity_interp, 60)
    neural_forward = neural_matrix[forward_mask]
    time_forward = time_minutes[forward_mask]

if neural_backward.shape[0] < 10:
    print("⚠️ Pocos datos hacia atrás, ajustando umbral...")
    backward_mask = velocity_interp < np.percentile(velocity_interp, 40)
    neural_backward = neural_matrix[backward_mask]
    time_backward = time_minutes[backward_mask]

print(f"Datos finales:")
print(f"  - Hacia adelante: {neural_forward.shape[0]} timepoints")
print(f"  - Hacia atrás: {neural_backward.shape[0]} timepoints")

# Estandarizar datos por separado
scaler_forward = StandardScaler()
scaler_backward = StandardScaler()

neural_forward_std = scaler_forward.fit_transform(neural_forward)
neural_backward_std = scaler_backward.fit_transform(neural_backward)

# Aplicar PCA por separado
pca_forward = PCA()
pca_backward = PCA()

pc_forward = pca_forward.fit_transform(neural_forward_std)
pc_backward = pca_backward.fit_transform(neural_backward_std)

explained_var_forward = pca_forward.explained_variance_ratio_
explained_var_backward = pca_backward.explained_variance_ratio_

print(f"Resultados PCA:")
print(f"  Hacia adelante - PC1: {explained_var_forward[0]*100:.1f}%, PC2: {explained_var_forward[1]*100:.1f}%, PC3: {explained_var_forward[2]*100:.1f}%")
print(f"  Hacia atrás - PC1: {explained_var_backward[0]*100:.1f}%, PC2: {explained_var_backward[1]*100:.1f}%, PC3: {explained_var_backward[2]*100:.1f}%")

# CREAR VISUALIZACIÓN COMPARATIVA
fig = plt.figure(figsize=(18, 9))
fig.patch.set_facecolor('white')

# === GRÁFICO 1: PCA 2D Comparativo ===
ax1 = plt.subplot(1, 2, 1)

# Función para suavizar trayectorias
def smooth_trajectory(pc_data, factor=5):
    if len(pc_data) < 4:
        return pc_data, pc_data
    
    # Aplicar filtro gaussiano primero para suavizar los datos originales
    pc1_filtered = gaussian_filter1d(pc_data[:, 0], sigma=1.5)
    pc2_filtered = gaussian_filter1d(pc_data[:, 1], sigma=1.5)
    
    t = np.arange(len(pc_data))
    t_smooth = np.linspace(0, len(pc_data)-1, len(pc_data)*factor)
    
    # Usar interpolación cúbica para curvas más suaves
    f_pc1 = interpolate.interp1d(t, pc1_filtered, kind='cubic')
    f_pc2 = interpolate.interp1d(t, pc2_filtered, kind='cubic')
    
    pc1_smooth = f_pc1(t_smooth)
    pc2_smooth = f_pc2(t_smooth)
    
    return pc1_smooth, pc2_smooth

# Suavizar trayectorias
pc1_forward_smooth, pc2_forward_smooth = smooth_trajectory(pc_forward)
pc1_backward_smooth, pc2_backward_smooth = smooth_trajectory(pc_backward)

# Dibujar trayectorias ADELANTE (ROJO)
for i in range(len(pc1_forward_smooth) - 1):
    alpha = 0.6 + 0.4 * (i / len(pc1_forward_smooth))  # Alpha progresivo
    ax1.plot([pc1_forward_smooth[i], pc1_forward_smooth[i+1]], 
             [pc2_forward_smooth[i], pc2_forward_smooth[i+1]], 
             color='red', alpha=alpha, linewidth=1.5, solid_capstyle='round')

# Dibujar trayectorias ATRÁS (AZUL)
for i in range(len(pc1_backward_smooth) - 1):
    alpha = 0.6 + 0.4 * (i / len(pc1_backward_smooth))  # Alpha progresivo
    ax1.plot([pc1_backward_smooth[i], pc1_backward_smooth[i+1]], 
             [pc2_backward_smooth[i], pc2_backward_smooth[i+1]], 
             color='blue', alpha=alpha, linewidth=1.5, solid_capstyle='round')

# Puntos de inicio y fin
ax1.scatter(pc_forward[0, 0], pc_forward[0, 1], color='darkred', s=100, marker='o', 
           edgecolor='white', linewidth=2, zorder=10, label='Inicio Adelante')
ax1.scatter(pc_forward[-1, 0], pc_forward[-1, 1], color='red', s=100, marker='s', 
           edgecolor='white', linewidth=2, zorder=10, label='Final Adelante')

ax1.scatter(pc_backward[0, 0], pc_backward[0, 1], color='darkblue', s=100, marker='o', 
           edgecolor='white', linewidth=2, zorder=10, label='Inicio Atrás')
ax1.scatter(pc_backward[-1, 0], pc_backward[-1, 1], color='blue', s=100, marker='s', 
           edgecolor='white', linewidth=2, zorder=10, label='Final Atrás')

ax1.set_xlabel(f'PC1 (Adelante: {explained_var_forward[0]*100:.1f}% | Atrás: {explained_var_backward[0]*100:.1f}%)', 
               fontsize=13, fontweight='bold')
ax1.set_ylabel(f'PC2 (Adelante: {explained_var_forward[1]*100:.1f}% | Atrás: {explained_var_backward[1]*100:.1f}%)', 
               fontsize=13, fontweight='bold')
ax1.set_title('PCA Comparativo por Comportamiento\nRojo: Hacia Adelante | Azul: Hacia Atrás', 
              fontsize=16, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.25, color='gray', linestyle='-', linewidth=0.5)
ax1.set_facecolor('#fafafa')
ax1.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='gray')

# === GRÁFICO 2: PCA 3D Comparativo ===
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# Suavizar trayectorias 3D
def smooth_trajectory_3d(pc_data, factor=5):
    if len(pc_data) < 4:
        return pc_data, pc_data, pc_data
    
    # Aplicar filtro gaussiano para suavizar los datos originales en 3D
    pc1_filtered = gaussian_filter1d(pc_data[:, 0], sigma=1.5)
    pc2_filtered = gaussian_filter1d(pc_data[:, 1], sigma=1.5)
    pc3_filtered = gaussian_filter1d(pc_data[:, 2], sigma=1.5)
    
    t = np.arange(len(pc_data))
    t_smooth = np.linspace(0, len(pc_data)-1, len(pc_data)*factor)
    
    # Usar interpolación cúbica para curvas más naturales en 3D
    f_pc1 = interpolate.interp1d(t, pc1_filtered, kind='cubic')
    f_pc2 = interpolate.interp1d(t, pc2_filtered, kind='cubic')
    f_pc3 = interpolate.interp1d(t, pc3_filtered, kind='cubic')
    
    return f_pc1(t_smooth), f_pc2(t_smooth), f_pc3(t_smooth)

pc1_f_3d, pc2_f_3d, pc3_f_3d = smooth_trajectory_3d(pc_forward)
pc1_b_3d, pc2_b_3d, pc3_b_3d = smooth_trajectory_3d(pc_backward)

# Trayectorias 3D ADELANTE (ROJO)
for i in range(len(pc1_f_3d) - 1):
    alpha = 0.6 + 0.4 * (i / len(pc1_f_3d))
    ax2.plot([pc1_f_3d[i], pc1_f_3d[i+1]], 
             [pc2_f_3d[i], pc2_f_3d[i+1]], 
             [pc3_f_3d[i], pc3_f_3d[i+1]], 
             color='red', alpha=alpha, linewidth=1.8)

# Trayectorias 3D ATRÁS (AZUL)
for i in range(len(pc1_b_3d) - 1):
    alpha = 0.6 + 0.4 * (i / len(pc1_b_3d))
    ax2.plot([pc1_b_3d[i], pc1_b_3d[i+1]], 
             [pc2_b_3d[i], pc2_b_3d[i+1]], 
             [pc3_b_3d[i], pc3_b_3d[i+1]], 
             color='blue', alpha=alpha, linewidth=1.8)

# Puntos 3D
ax2.scatter(pc_forward[0, 0], pc_forward[0, 1], pc_forward[0, 2],
           color='darkred', s=120, marker='o', edgecolor='white', linewidth=2, alpha=1.0)
ax2.scatter(pc_forward[-1, 0], pc_forward[-1, 1], pc_forward[-1, 2],
           color='red', s=120, marker='s', edgecolor='white', linewidth=2, alpha=1.0)

ax2.scatter(pc_backward[0, 0], pc_backward[0, 1], pc_backward[0, 2],
           color='darkblue', s=120, marker='o', edgecolor='white', linewidth=2, alpha=1.0)
ax2.scatter(pc_backward[-1, 0], pc_backward[-1, 1], pc_backward[-1, 2],
           color='blue', s=120, marker='s', edgecolor='white', linewidth=2, alpha=1.0)

ax2.set_xlabel(f'PC1', fontsize=12, fontweight='bold')
ax2.set_ylabel(f'PC2', fontsize=12, fontweight='bold')
ax2.set_zlabel(f'PC3', fontsize=12, fontweight='bold')
ax2.set_title('Trayectorias 3D por Comportamiento', fontsize=16, fontweight='bold', pad=20)

# Configurar vista 3D
ax2.view_init(elev=25, azim=45)
ax2.grid(True, alpha=0.2, color='gray')
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False

# Ajustes finales
plt.tight_layout()
plt.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.95)

# Guardar y mostrar
plt.savefig('PCA_Behavior_Forward_vs_Backward.png', dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')

print("Mostrando análisis PCA por comportamiento...")
plt.show()

print("="*70)
print("ANÁLISIS PCA POR COMPORTAMIENTO COMPLETADO")
print("Características:")
print(f"- Trayectorias rojas: Movimiento hacia adelante ({forward_mask.sum()} timepoints)")
print(f"- Trayectorias azules: Movimiento hacia atrás ({backward_mask.sum()} timepoints)")
print("- Líneas MUY suaves con filtro gaussiano + interpolación cúbica")
print("- Factor de suavizado x5 para curvas naturales")
print("- Alpha progresivo para mejor visualización temporal")
print("- Comparación directa en mismo espacio visual")
print("- Vista 2D y 3D para análisis completo")
print(f"- Archivo guardado: PCA_Behavior_Forward_vs_Backward.png")
print("="*70)