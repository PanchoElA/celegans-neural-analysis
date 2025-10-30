"""
Análisis Simple: Comparación FR vs Derivadas Temporales
"""
import numpy as np
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')

print("=== ANÁLISIS COMPARATIVO: FR vs dFR/dt ===")

# Cargar datos
nwb_file_path = "sub-2023-01-09-15-SWF702_ses-20230109_behavior+image+ophys.nwb"

with NWBHDF5IO(nwb_file_path, 'r') as io:
    nwbfile = io.read()
    
    calcium_module = nwbfile.processing['CalciumActivity']
    roi_response_series = calcium_module['SignalCalciumImResponseSeries']
    traces = roi_response_series.data[:]
    timestamps = roi_response_series.timestamps[:]

print(f"Datos: {traces.shape[1]} neuronas, {traces.shape[0]} timepoints")

# Preparar datos FR
F0 = np.percentile(traces, 10, axis=0)
neural_matrix_fr = (traces - F0) / (F0 + 1e-8)
neural_matrix_fr[np.isnan(neural_matrix_fr)] = 0

# Calcular derivadas temporales
dt = np.mean(np.diff(timestamps))
print(f"Resolución temporal: {dt:.3f} s")

smoothed_fr = gaussian_filter1d(neural_matrix_fr, sigma=2.0, axis=0)
derivatives = np.gradient(smoothed_fr, dt, axis=0)
derivatives[np.isnan(derivatives)] = 0

# PCA para FR
scaler_fr = StandardScaler()
fr_std = scaler_fr.fit_transform(neural_matrix_fr)
pca_fr = PCA()
pc_fr = pca_fr.fit_transform(fr_std)
var_fr = pca_fr.explained_variance_ratio_

# PCA para derivadas
scaler_deriv = StandardScaler()
deriv_std = scaler_deriv.fit_transform(derivatives)
pca_deriv = PCA()
pc_deriv = pca_deriv.fit_transform(deriv_std)
var_deriv = pca_deriv.explained_variance_ratio_

print(f"FR - PC1: {var_fr[0]*100:.1f}%, PC2: {var_fr[1]*100:.1f}%, PC3: {var_fr[2]*100:.1f}%")
print(f"dFR/dt - PC1: {var_deriv[0]*100:.1f}%, PC2: {var_deriv[1]*100:.1f}%, PC3: {var_deriv[2]*100:.1f}%")

# Crear visualización
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Comparación: Firing Rate vs Derivadas Temporales (dFR/dt)', fontsize=16, fontweight='bold')

# Suavizar para visualización
pc1_fr_smooth = gaussian_filter1d(pc_fr[:, 0], sigma=1.5)
pc2_fr_smooth = gaussian_filter1d(pc_fr[:, 1], sigma=1.5)
pc1_deriv_smooth = gaussian_filter1d(pc_deriv[:, 0], sigma=1.5)
pc2_deriv_smooth = gaussian_filter1d(pc_deriv[:, 1], sigma=1.5)

colors = plt.cm.viridis(np.linspace(0, 1, len(pc1_fr_smooth)))

# FR 2D
axes[0,0].scatter(pc1_fr_smooth, pc2_fr_smooth, c=colors, alpha=0.6, s=2)
axes[0,0].plot(pc1_fr_smooth, pc2_fr_smooth, 'k-', alpha=0.3, linewidth=0.5)
axes[0,0].scatter(pc1_fr_smooth[0], pc2_fr_smooth[0], color='blue', s=50, marker='o', label='Inicio')
axes[0,0].scatter(pc1_fr_smooth[-1], pc2_fr_smooth[-1], color='red', s=50, marker='s', label='Final')
axes[0,0].set_title(f'FR - PCA 2D\nPC1: {var_fr[0]*100:.1f}%, PC2: {var_fr[1]*100:.1f}%')
axes[0,0].set_xlabel('PC1')
axes[0,0].set_ylabel('PC2')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Derivadas 2D
axes[0,1].scatter(pc1_deriv_smooth, pc2_deriv_smooth, c=colors, alpha=0.6, s=2)
axes[0,1].plot(pc1_deriv_smooth, pc2_deriv_smooth, 'k-', alpha=0.3, linewidth=0.5)
axes[0,1].scatter(pc1_deriv_smooth[0], pc2_deriv_smooth[0], color='blue', s=50, marker='o', label='Inicio')
axes[0,1].scatter(pc1_deriv_smooth[-1], pc2_deriv_smooth[-1], color='red', s=50, marker='s', label='Final')
axes[0,1].set_title(f'dFR/dt - PCA 2D\nPC1: {var_deriv[0]*100:.1f}%, PC2: {var_deriv[1]*100:.1f}%')
axes[0,1].set_xlabel('PC1')
axes[0,1].set_ylabel('PC2')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Comparación varianza
pc_range = np.arange(1, 11)
axes[1,0].bar(pc_range - 0.2, var_fr[:10]*100, width=0.4, label='FR', alpha=0.8)
axes[1,0].bar(pc_range + 0.2, var_deriv[:10]*100, width=0.4, label='dFR/dt', alpha=0.8)
axes[1,0].set_title('Varianza Explicada por PC')
axes[1,0].set_xlabel('Componente Principal')
axes[1,0].set_ylabel('Varianza Explicada (%)')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Ejemplos de trazas
time_min = (timestamps - timestamps[0]) / 60
example_neurons = [0, 25, 50, 75, 100]

for i, neuron in enumerate(example_neurons):
    if neuron < neural_matrix_fr.shape[1]:
        axes[1,1].plot(time_min[:300], neural_matrix_fr[:300, neuron] + i*2, 
                      label=f'N{neuron} FR', alpha=0.7)
        axes[1,1].plot(time_min[:300], derivatives[:300, neuron]*10 + i*2 + 0.5, 
                      label=f'N{neuron} dFR/dt×10', alpha=0.7, linestyle='--')

axes[1,1].set_title('Ejemplos: FR vs dFR/dt (primeros 5 min)')
axes[1,1].set_xlabel('Tiempo (min)')
axes[1,1].set_ylabel('Señal + offset')
axes[1,1].legend(fontsize=8)
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('FR_vs_Derivatives_Simple.png', dpi=300, bbox_inches='tight')

# Estadísticas
print(f"\nEstadísticas comparativas:")
print(f"FR - Varianza total (3 PCs): {np.sum(var_fr[:3])*100:.1f}%")
print(f"dFR/dt - Varianza total (3 PCs): {np.sum(var_deriv[:3])*100:.1f}%")

correlations = []
for i in range(5):
    corr = np.corrcoef(pc_fr[:, i], pc_deriv[:, i])[0, 1]
    correlations.append(abs(corr))
    print(f"Correlación PC{i+1}: |r| = {abs(corr):.3f}")

print(f"Correlación promedio: {np.mean(correlations):.3f}")

print("\n✅ Análisis completado - Archivo guardado: FR_vs_Derivatives_Simple.png")