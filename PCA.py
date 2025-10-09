import numpy as np
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Abrir archivo NWB
nwb_file_path = "sub-2023-01-09-15-SWF702_ses-20230109_behavior+image+ophys.nwb"

print("=== ANÁLISIS PCA DE DATOS NEURALES C. ELEGANS ===")
print("Cargando datos...")

with NWBHDF5IO(nwb_file_path, 'r') as io:
    nwbfile = io.read()
    
    # Obtener datos de calcio desde CalciumActivity
    calcium_module = nwbfile.processing['CalciumActivity']
    roi_response_series = calcium_module['SignalCalciumImResponseSeries']
    traces = roi_response_series.data[:]
    timestamps = roi_response_series.timestamps[:]
    
print(f"Datos cargados: {traces.shape[1]} neuronas, {traces.shape[0]} puntos temporales")

# Preparar datos para PCA
time_minutes = timestamps / 60  # Convertir a minutos

# Calcular deltaF/F
F0 = np.percentile(traces, 10, axis=0)  # Baseline como percentil 10
deltaF_over_F = (traces - F0) / F0

# Remover valores infinitos o NaN
deltaF_clean = deltaF_over_F.copy()
deltaF_clean[np.isnan(deltaF_clean)] = 0
deltaF_clean[np.isinf(deltaF_clean)] = 0

print(f"Rango deltaF/F: {np.min(deltaF_clean):.3f} a {np.max(deltaF_clean):.3f}")

# PCA: Estandarizar datos
scaler = StandardScaler()
deltaF_scaled = scaler.fit_transform(deltaF_clean)

# Aplicar PCA
pca = PCA()
pca_transformed = pca.fit_transform(deltaF_scaled)

# Calcular varianza explicada
explained_variance = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_variance)

# Encontrar componentes necesarios para 80% y 90%
components_80 = np.where(cumulative_var >= 0.8)[0][0] + 1
components_90 = np.where(cumulative_var >= 0.9)[0][0] + 1

print("\n=== RESULTADOS PCA ===")
print(f"PC1: {explained_variance[0]:.1%} de varianza")
print(f"PC2: {explained_variance[1]:.1%} de varianza") 
print(f"PC3: {explained_variance[2]:.1%} de varianza")
print(f"Top 3 PCs: {np.sum(explained_variance[:3]):.1%} de varianza total")
print(f"Componentes para 80% varianza: {components_80}")
print(f"Componentes para 90% varianza: {components_90}")

# CREAR VISUALIZACIONES
plt.figure(figsize=(18, 12))

# Gráfico 1: Varianza explicada por componente
plt.subplot(2, 3, 1)
plt.plot(range(1, 21), explained_variance[:20] * 100, 'bo-', linewidth=2, markersize=6)
plt.xlabel('Componente Principal')
plt.ylabel('Varianza Explicada (%)')
plt.title('Varianza por Componente (Top 20)')
plt.grid(True, alpha=0.3)

# Gráfico 2: Varianza acumulada
plt.subplot(2, 3, 2)
plt.plot(range(1, 31), cumulative_var[:30] * 100, 'ro-', linewidth=2, markersize=4)
plt.axhline(y=80, color='gray', linestyle='--', alpha=0.7, label='80%')
plt.axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='90%')
plt.axvline(x=components_80, color='gray', linestyle=':', alpha=0.7, label=f'{components_80} comp.')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada (%)')
plt.title('Varianza Acumulada')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 3: PC1 vs PC2 coloreado por tiempo
plt.subplot(2, 3, 3)
scatter = plt.scatter(pca_transformed[:, 0], pca_transformed[:, 1], 
                     c=time_minutes, cmap='viridis', alpha=0.7, s=12)
plt.xlabel(f'PC1 ({explained_variance[0]:.1%})')
plt.ylabel(f'PC2 ({explained_variance[1]:.1%})')
plt.title('Proyección PC1 vs PC2')
cbar = plt.colorbar(scatter, label='Tiempo (min)')
plt.grid(True, alpha=0.3)

# Gráfico 4: Series temporales PC1, PC2, PC3
plt.subplot(2, 3, 4)
plt.plot(time_minutes, pca_transformed[:, 0], 'b-', linewidth=1.5, 
         label=f'PC1 ({explained_variance[0]:.1%})', alpha=0.8)
plt.plot(time_minutes, pca_transformed[:, 1], 'r-', linewidth=1.5, 
         label=f'PC2 ({explained_variance[1]:.1%})', alpha=0.8)
plt.plot(time_minutes, pca_transformed[:, 2], 'g-', linewidth=1.5, 
         label=f'PC3 ({explained_variance[2]:.1%})', alpha=0.8)
plt.xlabel('Tiempo (min)')
plt.ylabel('Valor del Componente')
plt.title('Evolución Temporal de PCs')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 5: PC2 vs PC3
plt.subplot(2, 3, 5)
scatter2 = plt.scatter(pca_transformed[:, 1], pca_transformed[:, 2], 
                      c=time_minutes, cmap='plasma', alpha=0.7, s=12)
plt.xlabel(f'PC2 ({explained_variance[1]:.1%})')
plt.ylabel(f'PC3 ({explained_variance[2]:.1%})')
plt.title('Proyección PC2 vs PC3')
cbar2 = plt.colorbar(scatter2, label='Tiempo (min)')
plt.grid(True, alpha=0.3)

# Gráfico 6: 3D PC1, PC2, PC3
ax = plt.subplot(2, 3, 6, projection='3d')
scatter3d = ax.scatter(pca_transformed[:, 0], pca_transformed[:, 1], pca_transformed[:, 2], 
                      c=time_minutes, cmap='coolwarm', alpha=0.6, s=10)
ax.set_xlabel(f'PC1 ({explained_variance[0]:.1%})')
ax.set_ylabel(f'PC2 ({explained_variance[1]:.1%})')
ax.set_zlabel(f'PC3 ({explained_variance[2]:.1%})')
ax.set_title('Trayectoria 3D Neural')

plt.tight_layout()
plt.savefig('PCA_Analysis_Complete.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("QUE ES EL PCA Y COMO INTERPRETARLO?")
print("="*60)

print("\n1. QUE ES PCA?")
print("   PCA (Principal Component Analysis) es una tecnica que:")
print("   • Reduce la dimensionalidad de datos complejos")
print("   • Encuentra los patrones principales en los datos")
print("   • Convierte 147 neuronas en unos pocos 'componentes principales'")
print("   • Cada componente es una combinacion de todas las neuronas")

print("\n2. INTERPRETACION DE TUS RESULTADOS:")
print(f"   • PC1 ({explained_variance[0]:.1%}): El patron neural mas importante")
print(f"   • PC2 ({explained_variance[1]:.1%}): El segundo patron mas importante") 
print(f"   • PC3 ({explained_variance[2]:.1%}): El tercer patron mas importante")
print(f"   • Necesitas {components_80} componentes para capturar 80% de la actividad")
print("   • Esto indica una red neural MUY COMPLEJA (sin un patron dominante)")

print("\n3. QUE SIGNIFICAN LOS GRAFICOS?")
print("   VARIANZA POR COMPONENTE:")
print("      - Muestra la 'importancia' de cada patron neural")
print("      - PC1 es el mas importante, pero solo explica ~21%")
print("      - La actividad esta muy distribuida (compleja)")

print("\n   PROYECCION PC1 vs PC2:")
print("      - Cada punto = el estado de toda la red en un momento")
print("      - Los colores muestran como evoluciona en el tiempo")
print("      - Puntos cercanos = estados neurales similares")
print("      - La trayectoria muestra como cambia la actividad")

print("\n   EVOLUCION TEMPORAL:")
print("      - Muestra como cambian los componentes en el tiempo")
print("      - Oscilaciones = patrones repetitivos")
print("      - Cambios bruscos = transiciones entre estados")

print("\n   VISTA 3D:")
print("      - Muestra el 'viaje' del estado neural en 3D")
print("      - Revela la complejidad de la dinamica neural")

print("\n4. CONCLUSIONES BIOLOGICAS:")
print("   • Tu red neural de C. elegans es muy compleja")
print("   • No hay un patron dominante (distribuida)")
print("   • Requiere muchos componentes = rica en patrones")
print("   • Esto es tipico de sistemas nerviosos funcionales")

print(f"\n5. DATOS TECNICOS:")
print(f"   • {traces.shape[1]} neuronas analizadas")
print(f"   • {traces.shape[0]} puntos temporales ({time_minutes[-1]:.1f} minutos)")
print(f"   • Reduccion dimensional: 147D -> {components_80}D (80% info)")
print(f"   • Complejidad: Media-Alta (distribuida)")

print("\n" + "="*60)
print("ANALISIS PCA COMPLETO!")
print("="*60)