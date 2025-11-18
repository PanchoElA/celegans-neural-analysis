import pynwb
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

### TUNING CURVES - GCaMP vs Velocity para RIMR, AIBR, AVAR ###

print("=== CREANDO TUNING CURVES ===")

f = 'sub-2023-01-09-15-SWF702_ses-20230109_behavior+image+ophys.nwb'
nwb = pynwb.NWBHDF5IO(f, mode='r').read()

# Obtener datos neurales
SignalCalciumImResponseSeries = nwb.processing["CalciumActivity"]["SignalRawFluor"]["SignalCalciumImResponseSeries"]
NeuronIDs = nwb.processing["CalciumActivity"]["NeuronIDs"]
neural_data = np.array(SignalCalciumImResponseSeries.data[:])
timestamps = SignalCalciumImResponseSeries.timestamps[:]

# Obtener datos de velocidad
velocity_data = nwb.processing["Behavior"]["velocity"]["velocity"].data[:]

# Convertir a ΔF/F
def calculate_delta_f_over_f(data):
    normalized_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        neuron_signal = data[:, i]
        F0 = np.percentile(neuron_signal, 10)
        normalized_data[:, i] = (neuron_signal - F0) / (F0 + 1e-8)
    return normalized_data

neural_delta_f = calculate_delta_f_over_f(neural_data)

# Encontrar las neuronas de interés (mismo código que GCaMP vs Time)
neuron_names = NeuronIDs.labels[:]
neurons_of_interest = ["RIMR", "AIBR", "AVAR"]
neuron_indices = {}
neuron_data_dict = {}

for neuron_name in neurons_of_interest:
    for i, name in enumerate(neuron_names):
        if name == neuron_name:
            neuron_indices[neuron_name] = i
            neuron_data_dict[neuron_name] = neural_delta_f[:, i]
            break

print(f"Neuronas encontradas:")
for neuron_name in neurons_of_interest:
    if neuron_name in neuron_indices:
        print(f"  {neuron_name}: índice {neuron_indices[neuron_name]}")
    else:
        print(f"  {neuron_name}: NO ENCONTRADA")

# NO convertir velocidad - usar μm/s directamente
velocity_um_s = velocity_data  # Ya está en μm/s

print(f"Rango de velocidad real: {np.min(velocity_um_s):.3f} a {np.max(velocity_um_s):.3f} μm/s")
print(f"Percentiles velocidad: P1={np.percentile(velocity_um_s, 1):.3f}, P99={np.percentile(velocity_um_s, 99):.3f} μm/s")

### CREAR TUNING CURVES ###

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ['blue', 'green', 'red']
neuron_labels = ["RIMR", "AIBR", "AVAR"]

for i, (neuron_name, color) in enumerate(zip(neurons_of_interest, colors)):
    if neuron_name not in neuron_data_dict:
        continue
        
    ax = axes[i]
    
    # Datos para esta neurona
    gcmp_data = neuron_data_dict[neuron_name]
    
    # Crear scatter plot
    ax.scatter(velocity_um_s, gcmp_data, alpha=0.6, s=2, color=color, rasterized=True)
    
    # Agregar línea de tendencia
    # Filtrar valores no finitos
    valid_mask = np.isfinite(velocity_um_s) & np.isfinite(gcmp_data)
    vel_clean = velocity_um_s[valid_mask]
    gcmp_clean = gcmp_data[valid_mask]
    
    if len(vel_clean) > 10:  # Solo si hay suficientes datos
        # Calcular regresión lineal
        slope, intercept, r_value, p_value, std_err = stats.linregress(vel_clean, gcmp_clean)
        
        # Crear línea de regresión
        vel_range = np.linspace(np.min(vel_clean), np.max(vel_clean), 100)
        regression_line = slope * vel_range + intercept
        
        ax.plot(vel_range, regression_line, 'k-', linewidth=2, alpha=0.8)
        
        # Agregar estadísticas en el título
        ax.set_title(f'{neuron_name}\nR² = {r_value**2:.3f}, p = {p_value:.2e}', 
                    fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'{neuron_name}', fontsize=12, fontweight='bold')
    
    # Configurar ejes
    ax.set_xlabel('Velocity (μm/s)', fontsize=11)
    if i == 0:  # Solo en el primer gráfico
        ax.set_ylabel('GCaMP (ΔF/F)', fontsize=11)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.12, 0.12)  # Rango apropiado para μm/s
    
    # Configurar límites Y basado en los datos
    y_min, y_max = np.percentile(gcmp_data[np.isfinite(gcmp_data)], [1, 99])
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)

plt.tight_layout()
plt.show()

### ANÁLISIS ADICIONAL: TUNING CURVES CON BINNING ###

print("\n=== CREANDO TUNING CURVES CON BINNING (ESTILO EJEMPLO) ===")

# Crear bins de velocidad apropiados para μm/s
velocity_bins = np.linspace(-0.12, 0.12, 25)  # 25 bins de -0.12 a 0.12 μm/s
bin_centers = (velocity_bins[:-1] + velocity_bins[1:]) / 2

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

for i, (neuron_name, color) in enumerate(zip(neurons_of_interest, colors)):
    if neuron_name not in neuron_data_dict:
        continue
        
    ax = axes2[i]
    
    gcmp_data = neuron_data_dict[neuron_name]
    
    # Calcular medias y errores estándar por bin
    bin_means = []
    bin_stds = []
    bin_counts = []
    
    for j in range(len(velocity_bins)-1):
        # Encontrar datos en este bin
        mask = (velocity_um_s >= velocity_bins[j]) & (velocity_um_s < velocity_bins[j+1])
        bin_data = gcmp_data[mask]
        
        if len(bin_data) > 0:
            bin_means.append(np.mean(bin_data))
            bin_stds.append(np.std(bin_data) / np.sqrt(len(bin_data)))  # Error estándar
            bin_counts.append(len(bin_data))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
            bin_counts.append(0)
    
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    
    # Filtrar bins con suficientes datos
    valid_bins = np.array(bin_counts) >= 5
    
    # Plot con barras de error
    ax.errorbar(bin_centers[valid_bins], bin_means[valid_bins], 
               yerr=bin_stds[valid_bins], fmt='o', color=color, 
               capsize=3, capthick=1, markersize=4)
    
    # Línea conectando los puntos
    ax.plot(bin_centers[valid_bins], bin_means[valid_bins], 
           '-', color=color, alpha=0.7, linewidth=1.5)
    
    # Configurar ejes
    ax.set_xlabel('Averaged velocity (μm/s)', fontsize=11)
    if i == 0:
        ax.set_ylabel('GCaMP (ΔF/F)', fontsize=11)
    
    ax.set_title(f'{neuron_name}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.12, 0.12)
    
    # Configurar límites Y
    valid_means = bin_means[valid_bins & np.isfinite(bin_means)]
    if len(valid_means) > 0:
        y_min, y_max = np.min(valid_means), np.max(valid_means)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.2*y_range, y_max + 0.2*y_range)

plt.tight_layout()
plt.show()

### ESTADÍSTICAS ###
print(f"\n=== ESTADÍSTICAS DE TUNING CURVES ===")
print(f"Rango de velocidad: {np.min(velocity_um_s):.3f} a {np.max(velocity_um_s):.3f} μm/s")
print(f"Número total de puntos: {len(velocity_um_s)}")

for neuron_name in neurons_of_interest:
    if neuron_name in neuron_data_dict:
        gcmp_data = neuron_data_dict[neuron_name]
        valid_mask = np.isfinite(velocity_um_s) & np.isfinite(gcmp_data)
        
        if np.sum(valid_mask) > 10:
            correlation = np.corrcoef(velocity_um_s[valid_mask], gcmp_data[valid_mask])[0,1]
            print(f"{neuron_name}: correlación = {correlation:.3f}")

print("¡Tuning curves completadas!")