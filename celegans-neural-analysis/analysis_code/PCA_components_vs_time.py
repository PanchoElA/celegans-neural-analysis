"""
Grafico de PC1, PC2 y PC3 vs Tiempo - PCA Original (Firing Rate)
Visualizacion temporal de los primeros 3 componentes principales
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('Agg')

print("=== GRAFICO PC1, PC2, PC3 vs TIEMPO ===")

# Cargar datos de los componentes principales
csv_file = "principal_components_top5.csv"
df = pd.read_csv(csv_file)

print(f"Datos cargados: {len(df)} puntos temporales")
print(f"Columnas disponibles: {list(df.columns)}")

# Extraer datos
time_minutes = df['Time_minutes'].values
PC1 = df['PC1'].values
PC2 = df['PC2'].values
PC3 = df['PC3'].values

# Convertir tiempo de minutos a segundos para mejor legibilidad
time_seconds = time_minutes * 60

print(f"Rango temporal: {time_seconds[0]:.1f} - {time_seconds[-1]:.1f} segundos")
print(f"Duracion total: {(time_seconds[-1] - time_seconds[0])/60:.1f} minutos")

# Estadisticas de los componentes
print(f"\nEstadisticas de los componentes:")
print(f"PC1: min={np.min(PC1):.2f}, max={np.max(PC1):.2f}, std={np.std(PC1):.2f}")
print(f"PC2: min={np.min(PC2):.2f}, max={np.max(PC2):.2f}, std={np.std(PC2):.2f}")
print(f"PC3: min={np.min(PC3):.2f}, max={np.max(PC3):.2f}, std={np.std(PC3):.2f}")

# Crear figura con subplots
fig, axes = plt.subplots(3, 1, figsize=(16, 12))
fig.suptitle('Componentes Principales vs Tiempo - C. elegans Neural Activity\nPCA Original (Firing Rate)', 
             fontsize=16, fontweight='bold', y=0.98)

# Colores para cada componente
colors = ['#d62728', '#2ca02c', '#1f77b4']  # Rojo, Verde, Azul
component_names = ['PC1', 'PC2', 'PC3']
components = [PC1, PC2, PC3]

# Graficar cada componente
for i, (ax, component, color, name) in enumerate(zip(axes, components, colors, component_names)):
    # Linea principal
    ax.plot(time_seconds, component, color=color, linewidth=1.5, alpha=0.8)
    
    # Sombreado para resaltar variaciones
    ax.fill_between(time_seconds, component, alpha=0.3, color=color)
    
    # Configuracion del subplot
    ax.set_ylabel(f'{name} Amplitude', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time_seconds[0], time_seconds[-1])
    
    # Agregar linea horizontal en cero para referencia
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
    
    # Estadisticas en el grafico
    mean_val = np.mean(component)
    std_val = np.std(component)
    ax.text(0.02, 0.95, f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}', 
            transform=ax.transAxes, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment='top')
    
    # Mejorar apariencia
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

# Solo el ultimo subplot tiene etiqueta de tiempo
axes[-1].set_xlabel('Tiempo (segundos)', fontsize=12, fontweight='bold')

# Ajustar espaciado
plt.tight_layout()
plt.subplots_adjust(top=0.93, hspace=0.3)

# Guardar archivo
filename = 'PCA_Components_vs_Time.png'
plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\nArchivo guardado: {filename}")

# Analisis adicional de periodicidad y tendencias
print("\n" + "="*50)
print("ANALISIS TEMPORAL DETALLADO")
print("="*50)

# Calcular tendencias
from scipy import signal

for i, (component, name) in enumerate(zip(components, component_names)):
    print(f"\n{name} ANALISIS:")
    
    # Tendencia lineal
    slope, intercept = np.polyfit(time_seconds, component, 1)
    print(f"  Tendencia: {slope:.6f} unidades/segundo")
    if abs(slope) > 0.001:
        print(f"  Interpretacion: {'Creciente' if slope > 0 else 'Decreciente'}")
    else:
        print(f"  Interpretacion: Estable (sin tendencia significativa)")
    
    # Rangos dinamicos
    range_val = np.max(component) - np.min(component)
    print(f"  Rango dinamico: {range_val:.2f}")
    
    # Deteccion de picos prominentes
    peaks, properties = signal.find_peaks(component, height=np.std(component))
    troughs, _ = signal.find_peaks(-component, height=np.std(component))
    
    print(f"  Picos prominentes: {len(peaks)}")
    print(f"  Valles prominentes: {len(troughs)}")
    
    if len(peaks) > 1:
        avg_peak_interval = np.mean(np.diff(time_seconds[peaks]))
        print(f"  Intervalo promedio entre picos: {avg_peak_interval:.1f} segundos")

# Correlaciones cruzadas entre componentes
print(f"\nCORRELACIONES ENTRE COMPONENTES:")
corr_12 = np.corrcoef(PC1, PC2)[0, 1]
corr_13 = np.corrcoef(PC1, PC3)[0, 1]
corr_23 = np.corrcoef(PC2, PC3)[0, 1]

print(f"  PC1 vs PC2: {corr_12:.3f}")
print(f"  PC1 vs PC3: {corr_13:.3f}")
print(f"  PC2 vs PC3: {corr_23:.3f}")

# Interpretacion de correlaciones
def interpret_correlation(corr):
    if abs(corr) > 0.7:
        return "Fuerte"
    elif abs(corr) > 0.3:
        return "Moderada"
    else:
        return "Debil"

print(f"\nINTERPRETACION:")
print(f"  PC1-PC2: Correlacion {interpret_correlation(corr_12).lower()} ({'positiva' if corr_12 > 0 else 'negativa'})")
print(f"  PC1-PC3: Correlacion {interpret_correlation(corr_13).lower()} ({'positiva' if corr_13 > 0 else 'negativa'})")
print(f"  PC2-PC3: Correlacion {interpret_correlation(corr_23).lower()} ({'positiva' if corr_23 > 0 else 'negativa'})")

print("="*50)
print("ANALISIS TEMPORAL COMPLETADO")
print("="*50)