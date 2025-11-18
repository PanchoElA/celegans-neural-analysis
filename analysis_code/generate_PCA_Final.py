import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import h5py

# Configurar matplotlib para mostrar gráficos
plt.ion()  # Activar modo interactivo
import matplotlib
matplotlib.use('TkAgg')  # Backend interactivo

print("=== GENERADOR DE PCA_Final_Clean.png ===")
print("Este script genera la visualización PCA limpia con nombres reales de neuronas")

# Load data and perform PCA
nwb_file = 'sub-2023-01-09-15-SWF702_ses-20230109_behavior+image+ophys.nwb'
with h5py.File(nwb_file, 'r') as f:
    neuron_labels = f['processing/CalciumActivity/NeuronIDs/labels'][:]
    neuron_names = [name.decode('utf-8') if name.decode('utf-8') != '' else f'Unknown_{i+1:03d}' 
                   for i, name in enumerate(neuron_labels)]

df = pd.read_csv('neural_data_dataframe.csv', index_col=0)
neural_matrix = df.values
scaler = StandardScaler()
neural_matrix_scaled = scaler.fit_transform(neural_matrix)
pca = PCA()
principal_components = pca.fit_transform(neural_matrix_scaled)
explained_variance_ratio = pca.explained_variance_ratio_

# Correct time calculation
time_minutes_correct = np.linspace(0, 16, 1615)

print("Datos cargados y PCA aplicado exitosamente")
print(f"Top 3 PCs: {explained_variance_ratio[0]*100:.1f}%, {explained_variance_ratio[1]*100:.1f}%, {explained_variance_ratio[2]*100:.1f}%")

# === FINAL CLEAN PLOTS ===
fig = plt.figure(figsize=(18, 14))

# Define common colormap
colormap = 'viridis'
vmin, vmax = 0, 16

# 1. PC1 vs PC2
plt.subplot(2, 2, 1)
scatter1 = plt.scatter(principal_components[:, 0], principal_components[:, 1], 
                      c=time_minutes_correct, cmap=colormap, vmin=vmin, vmax=vmax,
                      alpha=0.7, s=25)
plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)', fontsize=12)
plt.title('PC1 vs PC2', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# 2. PC1 vs PC2 vs PC3
ax = fig.add_subplot(222, projection='3d')
ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2], 
          c=time_minutes_correct, cmap=colormap, vmin=vmin, vmax=vmax,
          alpha=0.6, s=20)
ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)', fontsize=10)
ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)', fontsize=10)
ax.set_zlabel(f'PC3 ({explained_variance_ratio[2]*100:.1f}%)', fontsize=10)
ax.set_title('PC1 vs PC2 vs PC3', fontsize=14, fontweight='bold')

# 3. Scree Chart
plt.subplot(2, 2, 3)
bars = plt.bar(range(1, 6), explained_variance_ratio[:5], alpha=0.8, color='steelblue', edgecolor='navy')
plt.xlabel('Principal Component', fontsize=12)
plt.ylabel('Explained Variance', fontsize=12)
plt.title('Scree Chart - Top 5 PCs', fontsize=14, fontweight='bold')
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{explained_variance_ratio[i]*100:.1f}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold', color='darkblue')
plt.grid(True, alpha=0.3, axis='y')

# 4. Linear Combinations - CLEAN (con nombres reales de neuronas)
plt.subplot(2, 2, 4)
y_positions = [0.9, 0.75, 0.6, 0.45, 0.3]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for pc in range(5):
    pc_name = f'PC{pc+1}'
    variance = explained_variance_ratio[pc]*100
    pc_coeffs = pca.components_[pc]
    top_indices = np.argsort(np.abs(pc_coeffs))[-3:][::-1]
    
    plt.text(0.02, y_positions[pc], f'{pc_name} ({variance:.1f}%):', 
             fontsize=14, fontweight='bold', color=colors[pc],
             transform=plt.gca().transAxes)
    
    for i, idx in enumerate(top_indices):
        coeff = pc_coeffs[idx]
        neuron = neuron_names[idx]
        sign = '+' if coeff >= 0 else ''
        neuron_text = f'  {sign}{coeff:.3f} × {neuron}'
        plt.text(0.05, y_positions[pc] - (i+1)*0.025, neuron_text,
                fontsize=11, fontfamily='monospace', color='black',
                transform=plt.gca().transAxes)

plt.title('Linear Combinations PC1-PC5\n(Real Neuron Names)', fontsize=14, fontweight='bold')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')

# Single colorbar
cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.3])
cbar = fig.colorbar(scatter1, cax=cbar_ax, label='Time (minutes)')
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()
plt.subplots_adjust(right=0.90)
plt.savefig('PCA_Final_Clean.png', dpi=300, bbox_inches='tight')
print("Mostrando PCA_Final_Clean...")
plt.show()
print("Gráfico guardado como: PCA_Final_Clean.png")
plt.pause(3)  # Pausa para ver el gráfico

print("\n" + "="*60)
print("PCA_Final_Clean.png REGENERADO EXITOSAMENTE!")
print("Este es el archivo con:")
print("- 4 gráficos profesionales")
print("- Nombres reales de neuronas C. elegans") 
print("- Colormap unificado (viridis)")
print("- Tiempo corregido (0-16 minutos)")
print("="*60)