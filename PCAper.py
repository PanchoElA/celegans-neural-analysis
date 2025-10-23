import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

print("=== ANALISIS PCA DE C. ELEGANS: IDENTIFICANDO NEURONAS IMPORTANTES ===")
print("Siguiendo metodologia de builtin.com/machine-learning/pca-in-python")
print()

# PASO 1: CARGAR Y PREPARAR LOS DATOS
print("PASO 1: Cargando datos neurales...")

nwb_file_path = "sub-2023-01-09-15-SWF702_ses-20230109_behavior+image+ophys.nwb"

with NWBHDF5IO(nwb_file_path, 'r') as io:
    nwbfile = io.read()
    
    # Obtener datos de calcio
    calcium_module = nwbfile.processing['CalciumActivity']
    roi_response_series = calcium_module['SignalCalciumImResponseSeries']
    traces = roi_response_series.data[:]
    timestamps = roi_response_series.timestamps[:]

print(f"✅ Datos cargados: {traces.shape[1]} neuronas, {traces.shape[0]} puntos temporales")

# Crear matriz neurona vs tiempo (como el raster plot)
# Filas = timepoints, Columnas = neuronas (como en el tutorial)
time_minutes = timestamps / 60

# Calcular deltaF/F
F0 = np.percentile(traces, 10, axis=0)
neural_matrix = (traces - F0) / F0

# Limpiar datos
neural_matrix[np.isnan(neural_matrix)] = 0
neural_matrix[np.isinf(neural_matrix)] = 0

print(f"✅ Matriz neural preparada: {neural_matrix.shape[0]} timepoints x {neural_matrix.shape[1]} neuronas")
print(f"   Rango de actividad: {neural_matrix.min():.3f} a {neural_matrix.max():.3f}")

# PASO 2: ESTANDARIZAR LOS DATOS
print("\nPASO 2: Estandarizando los datos...")
print("PCA es sensible a la escala, por lo que necesitamos estandarizar")

# Estandarizar las neuronas (columnas)
scaler = StandardScaler()
neural_standardized = scaler.fit_transform(neural_matrix)

print(f"✅ Datos estandarizados: media ≈ 0, std ≈ 1")
print(f"   Nueva media: {neural_standardized.mean():.6f}")
print(f"   Nueva std: {neural_standardized.std():.6f}")

# PASO 3: APLICAR PCA
print("\nPASO 3: Aplicando PCA...")

# Aplicar PCA para obtener todos los componentes
pca_full = PCA()
principal_components = pca_full.fit_transform(neural_standardized)

# Obtener la varianza explicada
explained_variance_ratio = pca_full.explained_variance_ratio_
explained_variance_cumsum = np.cumsum(explained_variance_ratio)

print(f"✅ PCA aplicado exitosamente")
print(f"   Componentes principales: {len(explained_variance_ratio)}")

# PASO 4: ANALIZAR LA VARIANZA EXPLICADA
print("\nPASO 4: Analizando varianza explicada...")

print("Top 5 Componentes Principales:")
for i in range(5):
    print(f"   PC{i+1}: {explained_variance_ratio[i]:.4f} ({explained_variance_ratio[i]*100:.2f}%)")

print(f"\nVarianza acumulada Top 5: {explained_variance_cumsum[4]:.4f} ({explained_variance_cumsum[4]*100:.2f}%)")

# Encontrar cuántos componentes para 80% y 95% de varianza
n_components_80 = np.argmax(explained_variance_cumsum >= 0.80) + 1
n_components_95 = np.argmax(explained_variance_cumsum >= 0.95) + 1

print(f"Componentes para 80% varianza: {n_components_80}")
print(f"Componentes para 95% varianza: {n_components_95}")

# PASO 5: ANALIZAR LOS LOADINGS (NEURONAS MAS IMPORTANTES)
print("\nPASO 5: Identificando neuronas más importantes...")

# Los loadings nos dicen qué neuronas contribuyen más a cada PC
loadings = pca_full.components_

# Crear DataFrame de loadings para mejor análisis
neuron_names = [f"Neuron_{i+1:03d}" for i in range(traces.shape[1])]
loadings_df = pd.DataFrame(
    loadings[:5].T,  # Solo los primeros 5 PCs
    columns=[f'PC{i+1}' for i in range(5)],
    index=neuron_names
)

print("✅ Loadings calculados para Top 5 PCs")

# Encontrar las neuronas más importantes para cada PC
print("\nNeuronas MÁS importantes por componente:")
for i in range(5):
    pc_name = f'PC{i+1}'
    # Neuronas con mayor contribución absoluta
    top_neurons = loadings_df[pc_name].abs().sort_values(ascending=False).head(3)
    print(f"\n{pc_name} ({explained_variance_ratio[i]*100:.1f}% varianza):")
    for j, (neuron, loading) in enumerate(top_neurons.items()):
        neuron_num = neuron.split('_')[1]
        print(f"   {j+1}. Neurona {neuron_num}: loading = {loadings_df.loc[neuron, pc_name]:.4f}")

# PASO 6: CREAR VISUALIZACIONES
print("\nPASO 6: Creando visualizaciones...")

fig = plt.figure(figsize=(20, 16))

# Plot 1: Varianza explicada por componente
plt.subplot(3, 3, 1)
plt.bar(range(1, 11), explained_variance_ratio[:10], alpha=0.7, color='steelblue')
plt.xlabel('Componente Principal')
plt.ylabel('Varianza Explicada')
plt.title('Varianza Explicada por Componente (Top 10)')
plt.xticks(range(1, 11))
for i in range(5):
    plt.text(i+1, explained_variance_ratio[i] + 0.001, 
             f'{explained_variance_ratio[i]*100:.1f}%', 
             ha='center', va='bottom', fontsize=9)
plt.grid(True, alpha=0.3)

# Plot 2: Varianza acumulada
plt.subplot(3, 3, 2)
plt.plot(range(1, 21), explained_variance_cumsum[:20], 'ro-', linewidth=2, markersize=6)
plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.7, label='80%')
plt.axhline(y=0.95, color='gray', linestyle='--', alpha=0.7, label='95%')
plt.axvline(x=n_components_80, color='red', linestyle=':', alpha=0.7, 
           label=f'{n_components_80} comp.')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada')
plt.title('Varianza Acumulada')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: PC1 vs PC2 (coloreado por tiempo)
plt.subplot(3, 3, 3)
scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], 
                     c=time_minutes, cmap='viridis', alpha=0.6, s=15)
plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)')
plt.title('Proyección PC1 vs PC2')
plt.colorbar(scatter, label='Tiempo (min)')
plt.grid(True, alpha=0.3)

# Plot 4: Evolución temporal de los Top 5 PCs
plt.subplot(3, 3, 4)
colors = ['blue', 'red', 'green', 'orange', 'purple']
for i in range(5):
    plt.plot(time_minutes, principal_components[:, i], 
             color=colors[i], linewidth=1.5, alpha=0.8,
             label=f'PC{i+1} ({explained_variance_ratio[i]*100:.1f}%)')
plt.xlabel('Tiempo (minutos)')
plt.ylabel('Valor del Componente')
plt.title('Evolución Temporal - Top 5 PCs')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Heatmap de loadings (Top 5 PCs, Top 20 neuronas)
plt.subplot(3, 3, 5)
# Seleccionar las 20 neuronas con mayor variabilidad en loadings
neuron_importance = loadings_df.abs().sum(axis=1).sort_values(ascending=False)
top_neurons_indices = neuron_importance.head(20).index
loadings_subset = loadings_df.loc[top_neurons_indices]

sns.heatmap(loadings_subset.T, cmap='RdBu_r', center=0, 
           cbar_kws={'label': 'Loading'}, fmt='.3f')
plt.title('Loadings: Top 20 Neuronas vs Top 5 PCs')
plt.xlabel('Neuronas')
plt.ylabel('Componentes Principales')

# Plot 6: Contribución de neuronas a PC1
plt.subplot(3, 3, 6)
pc1_loadings = loadings_df['PC1'].abs().sort_values(ascending=False)
top_10_pc1 = pc1_loadings.head(10)
neuron_nums = [name.split('_')[1] for name in top_10_pc1.index]
plt.bar(range(len(top_10_pc1)), top_10_pc1.values, alpha=0.7, color='steelblue')
plt.xlabel('Neuronas')
plt.ylabel('|Loading| en PC1')
plt.title('Top 10 Neuronas Importantes para PC1')
plt.xticks(range(len(top_10_pc1)), neuron_nums, rotation=45)
plt.grid(True, alpha=0.3)

# Plot 7: PC2 vs PC3
plt.subplot(3, 3, 7)
scatter2 = plt.scatter(principal_components[:, 1], principal_components[:, 2], 
                      c=time_minutes, cmap='plasma', alpha=0.6, s=15)
plt.xlabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)')
plt.ylabel(f'PC3 ({explained_variance_ratio[2]*100:.1f}%)')
plt.title('Proyección PC2 vs PC3')
plt.colorbar(scatter2, label='Tiempo (min)')
plt.grid(True, alpha=0.3)

# Plot 8: Distribución de loadings para PC1
plt.subplot(3, 3, 8)
plt.hist(loadings_df['PC1'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Loading Value')
plt.ylabel('Número de Neuronas')
plt.title('Distribución de Loadings - PC1')
plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
plt.grid(True, alpha=0.3)

# Plot 9: Importancia global de neuronas (suma de loadings absolutos)
plt.subplot(3, 3, 9)
global_importance = loadings_df.abs().sum(axis=1).sort_values(ascending=False)
top_15_global = global_importance.head(15)
neuron_nums_global = [name.split('_')[1] for name in top_15_global.index]
plt.bar(range(len(top_15_global)), top_15_global.values, alpha=0.7, color='green')
plt.xlabel('Neuronas')
plt.ylabel('Importancia Global')
plt.title('Top 15 Neuronas Más Importantes (Global)')
plt.xticks(range(len(top_15_global)), neuron_nums_global, rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('PCA_Complete_Analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# PASO 7: RESUMEN DE RESULTADOS
print("\n" + "="*70)
print("RESUMEN: NEURONAS MAS IMPORTANTES IDENTIFICADAS POR PCA")
print("="*70)

print(f"\n📊 ESTADISTICAS GENERALES:")
print(f"   • {traces.shape[1]} neuronas analizadas")
print(f"   • {traces.shape[0]} puntos temporales")
print(f"   • {n_components_80} componentes explican 80% de la varianza")
print(f"   • {n_components_95} componentes explican 95% de la varianza")

print(f"\n🏆 TOP 5 COMPONENTES PRINCIPALES:")
for i in range(5):
    print(f"   PC{i+1}: {explained_variance_ratio[i]*100:.2f}% de varianza")

print(f"\n🧠 NEURONAS MAS IMPORTANTES POR COMPONENTE:")

for i in range(5):
    print(f"\n   PC{i+1} ({explained_variance_ratio[i]*100:.1f}% varianza):")
    pc_loadings = loadings_df[f'PC{i+1}'].abs().sort_values(ascending=False)
    for j, (neuron, loading) in enumerate(pc_loadings.head(3).items()):
        neuron_num = neuron.split('_')[1]
        original_loading = loadings_df.loc[neuron, f'PC{i+1}']
        print(f"      {j+1}. Neurona {neuron_num}: loading = {original_loading:+.4f} (|{loading:.4f}|)")

print(f"\n🌟 TOP 10 NEURONAS MAS IMPORTANTES GLOBALMENTE:")
global_importance = loadings_df.abs().sum(axis=1).sort_values(ascending=False)
for i, (neuron, importance) in enumerate(global_importance.head(10).items()):
    neuron_num = neuron.split('_')[1]
    print(f"   {i+1:2d}. Neurona {neuron_num}: importancia = {importance:.4f}")

print(f"\n💡 INTERPRETACION:")
print(f"   • PC1 captura el {explained_variance_ratio[0]*100:.1f}% del patrón más importante")
print(f"   • Las neuronas con loadings altos (+ o -) son las más influyentes")
print(f"   • Loadings positivos y negativos indican direcciones opuestas de variación")
print(f"   • La red neural es {'compleja' if n_components_80 > 20 else 'relativamente simple'} (requiere {n_components_80} PCs para 80%)")

# PASO 8: GUARDAR RESULTADOS
print(f"\n💾 GUARDANDO RESULTADOS...")

# Guardar loadings
loadings_df.to_csv('neuron_loadings_top5_PCs.csv')
print("   ✅ Loadings guardados: neuron_loadings_top5_PCs.csv")

# Guardar componentes principales
pcs_df = pd.DataFrame(
    principal_components[:, :5],
    columns=[f'PC{i+1}' for i in range(5)],
    index=time_minutes
)
pcs_df.reset_index(inplace=True)
pcs_df.rename(columns={'index': 'Time_minutes'}, inplace=True)
pcs_df.to_csv('principal_components_top5.csv', index=False)
print("   ✅ Componentes principales guardados: principal_components_top5.csv")

# Guardar ranking de importancia
importance_df = pd.DataFrame({
    'Neuron': [name.split('_')[1] for name in global_importance.index],
    'Global_Importance': global_importance.values,
    'PC1_Loading': [loadings_df.loc[neuron, 'PC1'] for neuron in global_importance.index],
    'PC2_Loading': [loadings_df.loc[neuron, 'PC2'] for neuron in global_importance.index],
    'PC3_Loading': [loadings_df.loc[neuron, 'PC3'] for neuron in global_importance.index]
})
importance_df.to_csv('neuron_importance_ranking.csv', index=False)
print("   ✅ Ranking de importancia guardado: neuron_importance_ranking.csv")

print("\n" + "="*70)
print("🎯 ANALISIS PCA COMPLETADO - NEURONAS IMPORTANTES IDENTIFICADAS!")
print("="*70)

