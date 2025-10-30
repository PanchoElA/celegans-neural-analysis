"""
ANÁLISIS AVANZADO: PREPROCESAMIENTO Y DERIVADAS TEMPORALES
Investigación de técnicas de preprocesamiento y análisis de dFR/dt
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import interpolate, signal
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

print("=== ANÁLISIS AVANZADO: PREPROCESAMIENTO Y DERIVADAS ===")

# Cargar datos base
nwb_file_path = "sub-2023-01-09-15-SWF702_ses-20230109_behavior+image+ophys.nwb"

with NWBHDF5IO(nwb_file_path, 'r') as io:
    nwbfile = io.read()
    
    calcium_module = nwbfile.processing['CalciumActivity']
    roi_response_series = calcium_module['SignalCalciumImResponseSeries']
    traces = roi_response_series.data[:]
    timestamps = roi_response_series.timestamps[:]

print(f"Datos cargados: {traces.shape[1]} neuronas, {traces.shape[0]} timepoints")

# Tiempo en minutos
time_minutes = (timestamps - timestamps[0]) / 60
dt = np.mean(np.diff(timestamps))  # Delta tiempo para derivadas
print(f"Resolución temporal: {dt:.4f} segundos ({1/dt:.1f} Hz)")

# ===============================
# PARTE 1: TÉCNICAS DE PREPROCESAMIENTO
# ===============================

def apply_preprocessing_techniques(traces, timestamps):
    """
    Aplica diferentes técnicas de preprocesamiento para comparar resultados
    """
    preprocessing_results = {}
    
    # 1. Método básico actual (ΔF/F0)
    print("1. Aplicando normalización ΔF/F0 básica...")
    F0_basic = np.percentile(traces, 10, axis=0)
    delta_f_f0_basic = (traces - F0_basic) / (F0_basic + 1e-8)
    preprocessing_results['delta_f_f0_basic'] = delta_f_f0_basic
    
    # 2. ΔF/F0 con ventana deslizante (más robusta)
    print("2. Aplicando ΔF/F0 con ventana deslizante...")
    window_size = min(300, traces.shape[0] // 10)  # 10% de los datos o máximo 300 puntos
    delta_f_f0_sliding = np.zeros_like(traces)
    
    for i in range(traces.shape[1]):  # Para cada neurona
        neuron_trace = traces[:, i]
        # Calcular F0 con ventana deslizante
        f0_sliding = signal.medfilt(neuron_trace, kernel_size=window_size)
        f0_sliding = gaussian_filter1d(f0_sliding, sigma=window_size//6)
        delta_f_f0_sliding[:, i] = (neuron_trace - f0_sliding) / (f0_sliding + 1e-8)
    
    preprocessing_results['delta_f_f0_sliding'] = delta_f_f0_sliding
    
    # 3. Z-score normalización
    print("3. Aplicando normalización Z-score...")
    z_scored = zscore(traces, axis=0)
    preprocessing_results['z_score'] = z_scored
    
    # 4. Robust scaling (resistente a outliers)
    print("4. Aplicando Robust Scaling...")
    robust_scaler = RobustScaler()
    robust_scaled = robust_scaler.fit_transform(traces)
    preprocessing_results['robust_scaled'] = robust_scaled
    
    # 5. Filtrado pasa-bajas para reducir ruido
    print("5. Aplicando filtrado pasa-bajas...")
    # Diseñar filtro Butterworth
    nyquist = 0.5 / dt  # Frecuencia de Nyquist
    cutoff = 0.1  # Hz - frecuencia de corte
    b, a = signal.butter(4, cutoff / nyquist, btype='low')
    
    filtered_traces = np.zeros_like(traces)
    for i in range(traces.shape[1]):
        filtered_traces[:, i] = signal.filtfilt(b, a, traces[:, i])
    
    # Aplicar ΔF/F0 a los datos filtrados
    F0_filtered = np.percentile(filtered_traces, 10, axis=0)
    delta_f_f0_filtered = (filtered_traces - F0_filtered) / (F0_filtered + 1e-8)
    preprocessing_results['filtered_delta_f_f0'] = delta_f_f0_filtered
    
    # 6. Detrending (remover tendencias lineales)
    print("6. Aplicando detrending...")
    detrended_traces = signal.detrend(traces, axis=0, type='linear')
    F0_detrended = np.percentile(detrended_traces, 10, axis=0)
    delta_f_f0_detrended = (detrended_traces - F0_detrended) / (np.abs(F0_detrended) + 1e-8)
    preprocessing_results['detrended_delta_f_f0'] = delta_f_f0_detrended
    
    return preprocessing_results

# ===============================
# PARTE 2: CÁLCULO DE DERIVADAS TEMPORALES
# ===============================

def calculate_temporal_derivatives(data, dt):
    """
    Calcula derivadas temporales (dFR/dt) usando diferentes métodos
    """
    derivatives = {}
    
    # 1. Diferencias finitas simples
    print("Calculando derivadas - Método 1: Diferencias finitas...")
    simple_diff = np.diff(data, axis=0) / dt
    # Extender para mantener misma longitud
    simple_diff = np.vstack([simple_diff[0:1], simple_diff])
    derivatives['simple_diff'] = simple_diff
    
    # 2. Diferencias centrales (más precisa)
    print("Calculando derivadas - Método 2: Diferencias centrales...")
    central_diff = np.zeros_like(data)
    central_diff[1:-1] = (data[2:] - data[:-2]) / (2 * dt)
    central_diff[0] = (data[1] - data[0]) / dt  # Forward difference en el inicio
    central_diff[-1] = (data[-1] - data[-2]) / dt  # Backward difference al final
    derivatives['central_diff'] = central_diff
    
    # 3. Gradiente de NumPy (suavizado)
    print("Calculando derivadas - Método 3: Gradiente NumPy...")
    numpy_gradient = np.gradient(data, dt, axis=0)
    derivatives['numpy_gradient'] = numpy_gradient
    
    # 4. Derivadas con suavizado previo
    print("Calculando derivadas - Método 4: Con suavizado previo...")
    smoothed_data = gaussian_filter1d(data, sigma=2.0, axis=0)
    smoothed_gradient = np.gradient(smoothed_data, dt, axis=0)
    derivatives['smoothed_gradient'] = smoothed_gradient
    
    return derivatives

print("\nAplicando técnicas de preprocesamiento...")
preprocessing_methods = apply_preprocessing_techniques(traces, timestamps)

print("\nCalculando derivadas temporales...")
# Usaremos el método básico para las derivadas por ahora
base_data = preprocessing_methods['delta_f_f0_basic']
derivatives_methods = calculate_temporal_derivatives(base_data, dt)

# ===============================
# PARTE 3: COMPARACIÓN VISUAL DE PREPROCESAMIENTO
# ===============================

def create_preprocessing_comparison():
    """Crear figura comparando diferentes métodos de preprocesamiento"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comparación de Técnicas de Preprocesamiento\nEjemplo: Neuronas 1-20', 
                 fontsize=16, fontweight='bold')
    
    methods = [
        ('delta_f_f0_basic', 'ΔF/F₀ Básico'),
        ('delta_f_f0_sliding', 'ΔF/F₀ Ventana Deslizante'),
        ('z_score', 'Z-Score'),
        ('robust_scaled', 'Robust Scaling'),
        ('filtered_delta_f_f0', 'Filtrado + ΔF/F₀'),
        ('detrended_delta_f_f0', 'Detrending + ΔF/F₀')
    ]
    
    # Seleccionar subconjunto de neuronas para visualización
    neuron_subset = slice(0, 20)
    time_subset = slice(0, min(1000, len(time_minutes)))  # Primeros 1000 puntos
    
    for idx, (method_key, method_name) in enumerate(methods):
        ax = axes[idx // 3, idx % 3]
        
        data_subset = preprocessing_methods[method_key][time_subset, neuron_subset]
        
        # Calcular percentiles para colormap consistente
        vmin, vmax = np.percentile(data_subset, [1, 99])
        
        im = ax.imshow(data_subset.T, aspect='auto', cmap='RdBu_r',
                      extent=[time_minutes[time_subset][0], time_minutes[time_subset][-1], 0, 20],
                      vmin=vmin, vmax=vmax)
        
        ax.set_title(method_name, fontweight='bold')
        ax.set_xlabel('Tiempo (min)')
        ax.set_ylabel('Neurona #')
        
        # Añadir colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('Preprocessing_Comparison.png', dpi=300, bbox_inches='tight')
    return fig

print("\nGenerando comparación de preprocesamiento...")
fig_prep = create_preprocessing_comparison()

# ===============================
# PARTE 4: ANÁLISIS PCA DE DERIVADAS
# ===============================

def perform_derivatives_pca_analysis():
    """Realizar análisis PCA completo usando derivadas temporales"""
    
    # Seleccionar método de derivadas (suavizado es generalmente mejor)
    derivatives_data = derivatives_methods['smoothed_gradient']
    
    # Limpiar datos
    derivatives_data[np.isnan(derivatives_data)] = 0
    derivatives_data[np.isinf(derivatives_data)] = 0
    
    # Estandarizar
    scaler = StandardScaler()
    derivatives_standardized = scaler.fit_transform(derivatives_data)
    
    # PCA
    pca = PCA()
    principal_components_derivatives = pca.fit_transform(derivatives_standardized)
    explained_variance = pca.explained_variance_ratio_
    
    print(f"\nPCA de Derivadas - Varianza explicada:")
    print(f"PC1: {explained_variance[0]*100:.1f}%")
    print(f"PC2: {explained_variance[1]*100:.1f}%")
    print(f"PC3: {explained_variance[2]*100:.1f}%")
    
    return principal_components_derivatives, explained_variance, derivatives_data

# ===============================
# PARTE 5: COMPARACIÓN FR vs dFR/dt
# ===============================

def create_fr_vs_derivatives_comparison():
    """Crear comparación lado a lado de FR vs dFR/dt"""
    
    # Datos originales (FR)
    base_data = preprocessing_methods['filtered_delta_f_f0']  # Usar versión filtrada
    base_data[np.isnan(base_data)] = 0
    base_data[np.isinf(base_data)] = 0
    
    scaler_base = StandardScaler()
    base_standardized = scaler_base.fit_transform(base_data)
    
    pca_base = PCA()
    pc_base = pca_base.fit_transform(base_standardized)
    var_base = pca_base.explained_variance_ratio_
    
    # Datos de derivadas
    pc_derivatives, var_derivatives, derivatives_data = perform_derivatives_pca_analysis()
    
    # Crear figura comparativa
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Comparación: Firing Rate vs Derivadas Temporales (dFR/dt)', 
                 fontsize=18, fontweight='bold')
    
    # Suavizar trayectorias para mejor visualización
    from scipy.ndimage import gaussian_filter1d
    
    # FR - 2D
    ax1 = plt.subplot(2, 4, 1)
    pc1_smooth = gaussian_filter1d(pc_base[:, 0], sigma=1.5)
    pc2_smooth = gaussian_filter1d(pc_base[:, 1], sigma=1.5)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(pc1_smooth)))
    for i in range(len(pc1_smooth)-1):
        ax1.plot([pc1_smooth[i], pc1_smooth[i+1]], [pc2_smooth[i], pc2_smooth[i+1]], 
                color=colors[i], alpha=0.7, linewidth=1.5)
    
    ax1.scatter(pc1_smooth[0], pc2_smooth[0], color='blue', s=100, marker='o', 
               edgecolor='white', linewidth=2, label='Inicio')
    ax1.scatter(pc1_smooth[-1], pc2_smooth[-1], color='red', s=100, marker='s', 
               edgecolor='white', linewidth=2, label='Final')
    ax1.set_title(f'FR - PCA 2D\nPC1: {var_base[0]*100:.1f}%, PC2: {var_base[1]*100:.1f}%')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # dFR/dt - 2D
    ax2 = plt.subplot(2, 4, 2)
    pc1_deriv_smooth = gaussian_filter1d(pc_derivatives[:, 0], sigma=1.5)
    pc2_deriv_smooth = gaussian_filter1d(pc_derivatives[:, 1], sigma=1.5)
    
    for i in range(len(pc1_deriv_smooth)-1):
        ax2.plot([pc1_deriv_smooth[i], pc1_deriv_smooth[i+1]], 
                [pc2_deriv_smooth[i], pc2_deriv_smooth[i+1]], 
                color=colors[i], alpha=0.7, linewidth=1.5)
    
    ax2.scatter(pc1_deriv_smooth[0], pc2_deriv_smooth[0], color='blue', s=100, marker='o', 
               edgecolor='white', linewidth=2, label='Inicio')
    ax2.scatter(pc1_deriv_smooth[-1], pc2_deriv_smooth[-1], color='red', s=100, marker='s', 
               edgecolor='white', linewidth=2, label='Final')
    ax2.set_title(f'dFR/dt - PCA 2D\nPC1: {var_derivatives[0]*100:.1f}%, PC2: {var_derivatives[1]*100:.1f}%')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # FR - 3D
    ax3 = plt.subplot(2, 4, 3, projection='3d')
    pc3_smooth = gaussian_filter1d(pc_base[:, 2], sigma=1.5)
    
    for i in range(len(pc1_smooth)-1):
        ax3.plot([pc1_smooth[i], pc1_smooth[i+1]], 
                [pc2_smooth[i], pc2_smooth[i+1]], 
                [pc3_smooth[i], pc3_smooth[i+1]], 
                color=colors[i], alpha=0.7, linewidth=1.5)
    
    ax3.scatter(pc1_smooth[0], pc2_smooth[0], pc3_smooth[0], color='blue', s=100)
    ax3.scatter(pc1_smooth[-1], pc2_smooth[-1], pc3_smooth[-1], color='red', s=100)
    ax3.set_title(f'FR - PCA 3D\nPC3: {var_base[2]*100:.1f}%')
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.set_zlabel('PC3')
    
    # dFR/dt - 3D
    ax4 = plt.subplot(2, 4, 4, projection='3d')
    pc3_deriv_smooth = gaussian_filter1d(pc_derivatives[:, 2], sigma=1.5)
    
    for i in range(len(pc1_deriv_smooth)-1):
        ax4.plot([pc1_deriv_smooth[i], pc1_deriv_smooth[i+1]], 
                [pc2_deriv_smooth[i], pc2_deriv_smooth[i+1]], 
                [pc3_deriv_smooth[i], pc3_deriv_smooth[i+1]], 
                color=colors[i], alpha=0.7, linewidth=1.5)
    
    ax4.scatter(pc1_deriv_smooth[0], pc2_deriv_smooth[0], pc3_deriv_smooth[0], color='blue', s=100)
    ax4.scatter(pc1_deriv_smooth[-1], pc2_deriv_smooth[-1], pc3_deriv_smooth[-1], color='red', s=100)
    ax4.set_title(f'dFR/dt - PCA 3D\nPC3: {var_derivatives[2]*100:.1f}%')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    ax4.set_zlabel('PC3')
    
    # Ejemplo de trazas temporales - FR
    ax5 = plt.subplot(2, 4, 5)
    example_neurons = [0, 25, 50, 75, 100]  # Seleccionar neuronas ejemplo
    for i, neuron_idx in enumerate(example_neurons):
        if neuron_idx < base_data.shape[1]:
            ax5.plot(time_minutes[:500], base_data[:500, neuron_idx] + i*2, 
                    label=f'Neurona {neuron_idx}', alpha=0.8)
    ax5.set_title('Ejemplos FR (Filtrado)')
    ax5.set_xlabel('Tiempo (min)')
    ax5.set_ylabel('ΔF/F₀ + offset')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Ejemplo de trazas temporales - dFR/dt
    ax6 = plt.subplot(2, 4, 6)
    for i, neuron_idx in enumerate(example_neurons):
        if neuron_idx < derivatives_data.shape[1]:
            ax6.plot(time_minutes[:500], derivatives_data[:500, neuron_idx] + i*0.5, 
                    label=f'Neurona {neuron_idx}', alpha=0.8)
    ax6.set_title('Ejemplos dFR/dt')
    ax6.set_xlabel('Tiempo (min)')
    ax6.set_ylabel('dFR/dt + offset')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # Distribución de varianza explicada
    ax7 = plt.subplot(2, 4, 7)
    pc_range = np.arange(1, 11)
    ax7.bar(pc_range - 0.2, var_base[:10]*100, width=0.4, label='FR', alpha=0.8)
    ax7.bar(pc_range + 0.2, var_derivatives[:10]*100, width=0.4, label='dFR/dt', alpha=0.8)
    ax7.set_title('Varianza Explicada por PC')
    ax7.set_xlabel('Componente Principal')
    ax7.set_ylabel('Varianza Explicada (%)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Correlación cruzada entre métodos
    ax8 = plt.subplot(2, 4, 8)
    correlations = []
    for i in range(min(10, pc_base.shape[1], pc_derivatives.shape[1])):
        corr = np.corrcoef(pc_base[:, i], pc_derivatives[:, i])[0, 1]
        correlations.append(abs(corr))  # Valor absoluto
    
    ax8.bar(range(1, len(correlations)+1), correlations, alpha=0.8, color='orange')
    ax8.set_title('Correlación |r| entre\nPC(FR) y PC(dFR/dt)')
    ax8.set_xlabel('Componente Principal')
    ax8.set_ylabel('|Correlación|')
    ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('FR_vs_Derivatives_Comparison.png', dpi=300, bbox_inches='tight')
    
    return fig, pc_base, pc_derivatives, var_base, var_derivatives

print("\nGenerando comparación FR vs dFR/dt...")
fig_comparison, pc_fr, pc_deriv, var_fr, var_deriv = create_fr_vs_derivatives_comparison()

# ===============================
# PARTE 6: REPORTE DE RESULTADOS
# ===============================

print("\n" + "="*80)
print("REPORTE DE ANÁLISIS AVANZADO")
print("="*80)

print("\n1. TÉCNICAS DE PREPROCESAMIENTO EVALUADAS:")
print("   ✓ ΔF/F₀ básico (percentil 10)")
print("   ✓ ΔF/F₀ con ventana deslizante (más robusto)")
print("   ✓ Z-score normalización")
print("   ✓ Robust scaling (resistente a outliers)")
print("   ✓ Filtrado pasa-bajas + ΔF/F₀")
print("   ✓ Detrending + ΔF/F₀")

print("\n2. MÉTODOS DE CÁLCULO DE DERIVADAS:")
print("   ✓ Diferencias finitas simples")
print("   ✓ Diferencias centrales (más precisa)")
print("   ✓ Gradiente NumPy")
print("   ✓ Gradiente con suavizado previo (recomendado)")

print(f"\n3. COMPARACIÓN VARIANZA EXPLICADA:")
print(f"   FR - Top 3 PCs: {var_fr[0]*100:.1f}%, {var_fr[1]*100:.1f}%, {var_fr[2]*100:.1f}%")
print(f"   dFR/dt - Top 3 PCs: {var_deriv[0]*100:.1f}%, {var_deriv[1]*100:.1f}%, {var_deriv[2]*100:.1f}%")

total_var_fr = np.sum(var_fr[:3])
total_var_deriv = np.sum(var_deriv[:3])
print(f"   FR - Total 3 PCs: {total_var_fr*100:.1f}%")
print(f"   dFR/dt - Total 3 PCs: {total_var_deriv*100:.1f}%")

# Calcular correlaciones promedio
correlations_pc = []
for i in range(min(5, pc_fr.shape[1], pc_deriv.shape[1])):
    corr = np.corrcoef(pc_fr[:, i], pc_deriv[:, i])[0, 1]
    correlations_pc.append(abs(corr))

print(f"\n4. CORRELACIONES ENTRE FR y dFR/dt:")
for i, corr in enumerate(correlations_pc):
    print(f"   PC{i+1}: |r| = {corr:.3f}")

print("\n5. ARCHIVOS GENERADOS:")
print("   📊 Preprocessing_Comparison.png - Comparación métodos preprocesamiento")
print("   📊 FR_vs_Derivatives_Comparison.png - Comparación FR vs dFR/dt")

print("\n6. RECOMENDACIONES:")
if total_var_deriv > total_var_fr:
    print("   🎯 Las DERIVADAS capturan MÁS varianza que FR original")
    print("   💡 Recomendación: Usar dFR/dt para análisis dinámicos")
else:
    print("   🎯 El FR original captura más varianza que las derivadas")
    print("   💡 Recomendación: Usar FR para análisis de estados")

if np.mean(correlations_pc) > 0.7:
    print("   🔗 Alta correlación entre FR y dFR/dt - patrones similares")
elif np.mean(correlations_pc) > 0.4:
    print("   🔗 Correlación moderada - patrones parcialmente relacionados")
else:
    print("   🔗 Baja correlación - FR y dFR/dt capturan dinámicas diferentes")

print("="*80)

plt.show()
print("\n✅ Análisis avanzado completado!")