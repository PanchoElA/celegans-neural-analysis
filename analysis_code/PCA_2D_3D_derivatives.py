"""
PCA 2D/3D con DERIVADAS TEMPORALES (dFR/dt)
Análisis de las derivadas del firing rate en lugar del firing rate directo
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
matplotlib.use('Agg')  # Backend no interactivo

print("=== PCA 2D/3D CON DERIVADAS TEMPORALES (dFR/dt) ===")

# Cargar datos
nwb_file_path = "sub-2023-01-09-15-SWF702_ses-20230109_behavior+image+ophys.nwb"

with NWBHDF5IO(nwb_file_path, 'r') as io:
    nwbfile = io.read()
    
    calcium_module = nwbfile.processing['CalciumActivity']
    roi_response_series = calcium_module['SignalCalciumImResponseSeries']
    traces = roi_response_series.data[:]
    timestamps = roi_response_series.timestamps[:]

print(f"Datos cargados: {traces.shape[1]} neuronas, {traces.shape[0]} timepoints")

# Calcular resolución temporal
dt = np.mean(np.diff(timestamps))
print(f"Resolución temporal: {dt:.4f} s ({1/dt:.1f} Hz)")

# ===============================
# PREPROCESAMIENTO OPTIMIZADO
# ===============================

# 1. Normalización ΔF/F0 con filtrado
from scipy import signal

# Filtro pasa-bajas para reducir ruido antes de calcular derivadas
nyquist = 0.5 / dt
cutoff = 0.1  # Hz
b, a = signal.butter(4, cutoff / nyquist, btype='low')

filtered_traces = np.zeros_like(traces)
for i in range(traces.shape[1]):
    filtered_traces[:, i] = signal.filtfilt(b, a, traces[:, i])

# Calcular ΔF/F0 en datos filtrados
F0 = np.percentile(filtered_traces, 10, axis=0)
neural_matrix = (filtered_traces - F0) / (F0 + 1e-8)

# ===============================
# CÁLCULO DE DERIVADAS TEMPORALES
# ===============================

print("Calculando derivadas temporales dFR/dt...")

# Método: Gradiente con suavizado previo (más robusto)
smoothed_data = gaussian_filter1d(neural_matrix, sigma=2.0, axis=0)
derivatives_data = np.gradient(smoothed_data, dt, axis=0)

print(f"Rango de derivadas: {np.min(derivatives_data):.4f} a {np.max(derivatives_data):.4f}")

# Limpiar datos
derivatives_data[np.isnan(derivatives_data)] = 0
derivatives_data[np.isinf(derivatives_data)] = 0

# ===============================
# ANÁLISIS PCA DE DERIVADAS
# ===============================

# Estandarizar derivadas
scaler = StandardScaler()
derivatives_standardized = scaler.fit_transform(derivatives_data)

# Aplicar PCA
pca_derivatives = PCA()
principal_components = pca_derivatives.fit_transform(derivatives_standardized)
explained_variance_ratio = pca_derivatives.explained_variance_ratio_

print(f"PCA Derivadas - Varianza explicada:")
print(f"PC1: {explained_variance_ratio[0]*100:.1f}%, PC2: {explained_variance_ratio[1]*100:.1f}%, PC3: {explained_variance_ratio[2]*100:.1f}%")

# ===============================
# SUAVIZADO PARA VISUALIZACIÓN
# ===============================

# Suavizar trayectorias PCA para mejor visualización
pc1_filtered = gaussian_filter1d(principal_components[:, 0], sigma=1.5)
pc2_filtered = gaussian_filter1d(principal_components[:, 1], sigma=1.5)
pc3_filtered = gaussian_filter1d(principal_components[:, 2], sigma=1.5)

# Interpolación cúbica para líneas más suaves
t = np.arange(len(principal_components))
t_smooth = np.linspace(0, len(principal_components)-1, len(principal_components)*5)

f_pc1 = interpolate.interp1d(t, pc1_filtered, kind='cubic')
f_pc2 = interpolate.interp1d(t, pc2_filtered, kind='cubic')
f_pc3 = interpolate.interp1d(t, pc3_filtered, kind='cubic')

pc1_smooth = f_pc1(t_smooth)
pc2_smooth = f_pc2(t_smooth)
pc3_smooth = f_pc3(t_smooth)

# ===============================
# CREAR VISUALIZACIÓN
# ===============================

fig = plt.figure(figsize=(16, 8))
fig.patch.set_facecolor('white')

# Colores temporales
colors = plt.cm.plasma(np.linspace(0, 1, len(pc1_smooth)))
time_minutes = (timestamps - timestamps[0]) / 60

# === GRÁFICO 2D ===
ax1 = fig.add_subplot(1, 2, 1)

# Dibujar trayectoria suave
for i in range(len(pc1_smooth) - 1):
    ax1.plot([pc1_smooth[i], pc1_smooth[i+1]], 
             [pc2_smooth[i], pc2_smooth[i+1]], 
             color=colors[i], alpha=0.8, linewidth=1.8)

# Puntos de inicio y final
ax1.scatter(principal_components[0, 0], principal_components[0, 1],
           color='darkblue', s=150, marker='o', edgecolor='white', linewidth=3, 
           alpha=1.0, label='INICIO', zorder=10)
ax1.scatter(principal_components[-1, 0], principal_components[-1, 1],
           color='darkred', s=150, marker='s', edgecolor='white', linewidth=3, 
           alpha=1.0, label='FINAL', zorder=10)

ax1.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)', fontsize=12, fontweight='bold')
ax1.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)', fontsize=12, fontweight='bold')
ax1.set_title('PCA 2D - Derivadas Temporales (dFR/dt)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.2, color='gray')
ax1.set_facecolor('#f8f9fa')

# === GRÁFICO 3D ===
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# Dibujar trayectoria 3D suave
for i in range(len(pc1_smooth) - 1):
    ax2.plot([pc1_smooth[i], pc1_smooth[i+1]], 
             [pc2_smooth[i], pc2_smooth[i+1]], 
             [pc3_smooth[i], pc3_smooth[i+1]], 
             color=colors[i], alpha=0.8, linewidth=1.8)

# Puntos de inicio y final
ax2.scatter(principal_components[0, 0], principal_components[0, 1], principal_components[0, 2],
           color='darkblue', s=150, marker='o', edgecolor='white', linewidth=3, alpha=1.0)
ax2.scatter(principal_components[-1, 0], principal_components[-1, 1], principal_components[-1, 2],
           color='darkred', s=150, marker='s', edgecolor='white', linewidth=3, alpha=1.0)

ax2.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)', fontweight='bold')
ax2.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)', fontweight='bold')
ax2.set_zlabel(f'PC3 ({explained_variance_ratio[2]*100:.1f}%)', fontweight='bold')
ax2.set_title('PCA 3D - Derivadas Temporales (dFR/dt)', fontsize=14, fontweight='bold')

# Configurar vista 3D
ax2.view_init(elev=25, azim=45)
ax2.grid(True, alpha=0.2)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.xaxis.pane.set_alpha(0.1)
ax2.yaxis.pane.set_alpha(0.1)
ax2.zaxis.pane.set_alpha(0.1)

# ===============================
# INFORMACIÓN ADICIONAL
# ===============================

# Agregar información sobre el análisis
info_text = f"""Análisis PCA de Derivadas Temporales
• Datos: {traces.shape[1]} neuronas, {traces.shape[0]} timepoints
• Resolución: {dt:.3f}s ({1/dt:.1f} Hz)
• Preprocesamiento: Filtrado + ΔF/F₀ + Suavizado
• Método derivadas: Gradiente con suavizado previo
• Varianza total (3 PCs): {np.sum(explained_variance_ratio[:3])*100:.1f}%

Interpretación dFR/dt:
• Valores positivos: Activación creciente
• Valores negativos: Activación decreciente  
• Cero: Estado estacionario
• Magnitud: Velocidad del cambio"""

fig.text(0.02, 0.02, info_text, fontsize=9, 
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
         verticalalignment='bottom')

# Título general
plt.suptitle('Análisis de Componentes Principales: Derivadas Temporales del Firing Rate\n' + 
             f'C. elegans - {time_minutes[-1]:.1f} minutos de registro', 
             fontsize=16, fontweight='bold', y=0.95)

plt.tight_layout()
plt.subplots_adjust(top=0.87, bottom=0.20, left=0.08, right=0.95, hspace=0.2, wspace=0.2)

# Guardar archivo
filename = 'PCA_2D_3D_Derivatives.png'
plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

print(f"✅ Archivo guardado: {filename}")

# ===============================
# ANÁLISIS ESTADÍSTICO
# ===============================

print("\n" + "="*60)
print("ANÁLISIS ESTADÍSTICO DE DERIVADAS")
print("="*60)

# Estadísticas básicas
print(f"Rango derivadas: {np.min(derivatives_data):.4f} a {np.max(derivatives_data):.4f}")
print(f"Media derivadas: {np.mean(derivatives_data):.6f}")
print(f"Std derivadas: {np.std(derivatives_data):.4f}")

# Análisis de componentes principales
print(f"\nVarianza explicada por componente:")
for i in range(min(10, len(explained_variance_ratio))):
    print(f"  PC{i+1}: {explained_variance_ratio[i]*100:.2f}%")

print(f"\nVarianza acumulada:")
cumsum_var = np.cumsum(explained_variance_ratio)
for i in [2, 4, 9]:  # 3, 5, 10 componentes
    if i < len(cumsum_var):
        print(f"  Primeros {i+1} PCs: {cumsum_var[i]*100:.1f}%")

# Análisis de ciclicidad
start_point = principal_components[0, :3]
end_point = principal_components[-1, :3]
distance = np.linalg.norm(end_point - start_point)
total_range = np.ptp(principal_components[:, :3])

print(f"\nAnálisis de trayectoria:")
print(f"  Distancia inicio-final: {distance:.4f}")
print(f"  Rango total variación: {total_range:.4f}")
print(f"  Ratio ciclicidad: {distance/total_range:.3f}")

if distance/total_range < 0.15:
    print("  🔄 TRAYECTORIA DERIVADAS: ALTAMENTE CÍCLICA")
elif distance/total_range < 0.35:
    print("  🔄 TRAYECTORIA DERIVADAS: MODERADAMENTE CÍCLICA")  
else:
    print("  ➡️ TRAYECTORIA DERIVADAS: MÁS LINEAL")

print("="*60)

print("✅ Visualización guardada exitosamente")
print("\n🎯 Análisis PCA de derivadas temporales completado!")