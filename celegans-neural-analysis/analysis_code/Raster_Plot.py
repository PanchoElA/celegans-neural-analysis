import pynwb
import matplotlib.pyplot as plt
import numpy as np

import pynwb
import matplotlib.pyplot as plt
import numpy as np

# Configurar matplotlib para mostrar gráficos
plt.ion()  # Activar modo interactivo
import matplotlib
matplotlib.use('TkAgg')  # Backend interactivo

### RASTER PLOT - VERSIÓN CORREGIDA ###

f = 'sub-2023-01-09-15-SWF702_ses-20230109_behavior+image+ophys.nwb'
nwb = pynwb.NWBHDF5IO(f, mode='r').read()


# Obtener timestamps neurales como REFERENCIA ABSOLUTA - CORREGIDO
neural_timestamps = nwb.processing["CalciumActivity"]["SignalCalciumImResponseSeries"].timestamps[:]
neural_data = np.array(nwb.processing["CalciumActivity"]["SignalCalciumImResponseSeries"].data[:])

# USAR ESTOS TIMESTAMPS COMO REFERENCIA ABSOLUTA
master_time_minutes = (neural_timestamps - neural_timestamps[0]) / 60

print(f"TIEMPO MAESTRO: 0 a {master_time_minutes[-1]:.3f} minutos")
print(f"Puntos: {len(master_time_minutes)}")

# Obtener datos de comportamiento con LOS MISMOS TIMESTAMPS
velocity_data = nwb.processing["Behavior"]["velocity"]["velocity"].data[:]
angular_velocity_data = nwb.processing["Behavior"]["angular_velocity"]["angular_velocity"].data[:]
pumping_data = nwb.processing["Behavior"]["pumping"]["pumping"].data[:]

# Verificar que tienen la misma longitud
print(f"Neural: {len(neural_data)} puntos")
print(f"Velocity: {len(velocity_data)} puntos")
print(f"Angular: {len(angular_velocity_data)} puntos")
print(f"Pumping: {len(pumping_data)} puntos")

# Asegurar que todos tengan la misma longitud
min_length = min(len(neural_data), len(velocity_data), len(angular_velocity_data), len(pumping_data))
print(f"Usando longitud común: {min_length} puntos")


print(f"TIEMPO FINAL COMÚN: 0 a {master_time_minutes[-1]:.3f} minutos")

### NORMALIZACIÓN ΔF/F ###
def calculate_delta_f_over_f(data):
    normalized_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        neuron_signal = data[:, i]
        F0 = np.percentile(neuron_signal, 10)
        normalized_data[:, i] = (neuron_signal - F0) / (F0 + 1e-8)
    return normalized_data

neural_delta_f = calculate_delta_f_over_f(neural_data)

### CREAR FIGURA CON ALINEACIÓN FORZADA - SIN COLORBAR ###
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(4, 1, height_ratios=[4, 0.8, 0.8, 0.8], 
                      hspace=0.3, left=0.08, right=0.95, top=0.95, bottom=0.08)

# Panel neural - SIN COLORBAR
ax_neural = fig.add_subplot(gs[0])
neural_heatmap = neural_delta_f.T
vmin, vmax = np.percentile(neural_heatmap, [2, 98])

im = ax_neural.imshow(neural_heatmap, aspect='auto', cmap='plasma',
                     extent=[master_time_minutes[0], master_time_minutes[-1], 0, neural_data.shape[1]],
                     vmin=vmin, vmax=vmax, interpolation='nearest')

ax_neural.set_ylabel('Neuron #', fontsize=12, fontweight='bold')
ax_neural.set_title('GCaMP Data - C. elegans Neural Activity', fontsize=14, fontweight='bold')
ax_neural.tick_params(axis='x', labelbottom=False)

# NO AGREGAR COLORBAR AQUÍ - esto desalinea las escalas

# Panel velocity
ax_vel = fig.add_subplot(gs[1])
ax_vel.plot(master_time_minutes, velocity_data, 'k-', linewidth=1)
ax_vel.set_ylabel('Velocity\n(um/s)', fontsize=10, fontweight='bold')
ax_vel.tick_params(axis='x', labelbottom=False)
ax_vel.grid(True, alpha=0.3)

# Panel angular velocity
ax_ang = fig.add_subplot(gs[2])
ax_ang.plot(master_time_minutes, angular_velocity_data, 'k-', linewidth=1)
ax_ang.set_ylabel('Angular\nvelocity\n(rad/s)', fontsize=10, fontweight='bold')
ax_ang.tick_params(axis='x', labelbottom=False)
ax_ang.grid(True, alpha=0.3)

# Panel pumping
ax_pump = fig.add_subplot(gs[3])
ax_pump.plot(master_time_minutes, pumping_data, 'k-', linewidth=1)
ax_pump.set_ylabel('Pumping\n(a.u.)', fontsize=10, fontweight='bold')
ax_pump.set_xlabel('Time (min)', fontsize=12, fontweight='bold')
ax_pump.grid(True, alpha=0.3)

# FORZAR QUE TODOS TENGAN EXACTAMENTE LOS MISMOS LÍMITES X
xlim_min = master_time_minutes[0]
xlim_max = master_time_minutes[-1]

print(f"FORZANDO xlim para todos los paneles: ({xlim_min:.3f}, {xlim_max:.3f})")

ax_neural.set_xlim(xlim_min, xlim_max)
ax_vel.set_xlim(xlim_min, xlim_max)
ax_ang.set_xlim(xlim_min, xlim_max)
ax_pump.set_xlim(xlim_min, xlim_max)

# Sincronizar tick marks
tick_positions = np.linspace(xlim_min, xlim_max, 9)  # 0, 2, 4, 6, 8, 10, 12, 14, 16
for ax in [ax_neural, ax_vel, ax_ang, ax_pump]:
    ax.set_xticks(tick_positions)

plt.tight_layout()
plt.savefig('Raster_Plot_Output.png', dpi=300, bbox_inches='tight')
print("Mostrando raster plot principal...")
plt.show()
print("Gráfico guardado como: Raster_Plot_Output.png")
plt.pause(2)  # Pausa para ver el gráfico

### CREAR FIGURA SEPARADA SOLO PARA EL COLORBAR ###
print("\nCreando colorbar en figura separada...")

fig_colorbar, ax_colorbar = plt.subplots(figsize=(2, 6))
ax_colorbar.axis('off')  # Ocultar ejes

# Crear colorbar independiente
cbar = fig_colorbar.colorbar(im, ax=ax_colorbar, shrink=0.8, aspect=20)
cbar.set_label('Delta F/F', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()
plt.savefig('Colorbar_Separate.png', dpi=300, bbox_inches='tight')
print("Mostrando colorbar...")
plt.show()
print("Colorbar guardado como: Colorbar_Separate.png")
plt.pause(2)  # Pausa para ver el colorbar

print("\nRaster plot completado exitosamente!")
print("Archivos generados:")
print("- Raster_Plot_Output.png (figura principal)")
print("- Colorbar_Separate.png (colorbar independiente)")

# TODO: Hacer tuning curves de las tres neuronas graficas en GCaMP vs Time
# TODO: Entender el PCA, Como definir cada matriz, que son las columnas, filas, celdas y que espero ver con la matriz