import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
import os

# --- 1. Definición de la Función de Clasificación ---
def classify_behavior(velocity_data, angular_velocity_data, 
                      fwd_thresh, rev_thresh, turn_thresh):
    """
    Clasifica el comportamiento de C. elegans en 5 estados categóricos 
    basados en los datos de 'velocity' y 'angular_velocity'.
    """
    
    fwd_data = np.asarray(velocity_data)
    ang_vel_data = np.asarray(angular_velocity_data)

    if fwd_data.shape != ang_vel_data.shape:
        print(f"Error: Los arrays de 'velocity' (forma: {fwd_data.shape}) y 'angular_velocity' (forma: {ang_vel_data.shape}) deben tener la misma longitud.")
        return None  # Devuelve None si hay un error de forma

    num_timesteps = len(fwd_data)
    behavioral_states = np.full(num_timesteps, "pause", dtype=object)

    # Definir giros (prioridad)
    is_turning_right = ang_vel_data > turn_thresh
    is_turning_left = ang_vel_data < -turn_thresh
    behavioral_states[is_turning_right] = "turn_right"
    behavioral_states[is_turning_left] = "turn_left"

    # Definir avance y retroceso (solo si no está girando)
    is_straight = (~is_turning_right) & (~is_turning_left)
    is_forward = (fwd_data > fwd_thresh) & is_straight
    is_reverse = (fwd_data < rev_thresh) & is_straight
    behavioral_states[is_forward] = "forward"
    behavioral_states[is_reverse] = "reverse"

    return behavioral_states

# --- 2. Bloque Principal de Ejecución ---

# Define la ruta a tu archivo NWB
nwb_file_path = 'sub-2023-01-09-15-SWF702_ses-20230109_behavior+image+ophys.nwb'
# Define el nombre del archivo CSV que quieres crear
csv_filename = 'classified_behavioral_states.csv'

io = None  # Para asegurar que podamos cerrarlo en 'finally'

try:
    # Verificar si el archivo NWB existe
    if not os.path.exists(nwb_file_path):
        print(f"--- ERROR DE ARCHIVO ---")
        print(f"No se pudo encontrar el archivo NWB en la ruta: {nwb_file_path}")
    else:
        # --- 2a. Cargar el archivo NWB y los datos ---
        print(f"Cargando datos desde: {nwb_file_path}...")
        io = NWBHDF5IO(nwb_file_path, 'r')
        nwb = io.read()

        angular_velocity = nwb.processing["Behavior"]["angular_velocity"]["angular_velocity"].data[:]
        velocity = nwb.processing["Behavior"]["velocity"]["velocity"].data[:]
        
        print("Datos 'velocity' y 'angular_velocity' cargados exitosamente.")

        # --- 2b. Calcular Umbrales Adaptativos ---
        REV_THRESHOLD = np.quantile(velocity, 0.30)
        FWD_THRESHOLD = np.quantile(velocity, 0.70)
        abs_ang_vel = np.abs(angular_velocity)
        TURN_THRESHOLD = np.quantile(abs_ang_vel, 0.80)
        
        print("\nUmbrales adaptativos calculados:")
        print(f"  Umbral 'Reverse' (Percentil 30): {REV_THRESHOLD:.4f}")
        print(f"  Umbral 'Forward' (Percentil 70): {FWD_THRESHOLD:.4f}")
        print(f"  Umbral 'Turn'    (Percentil 80 de |Ang. Vel.|): {TURN_THRESHOLD:.4f}")

        # --- 2c. Ejecutar la Clasificación ---
        behavioral_states = classify_behavior(velocity, angular_velocity,
                                              fwd_thresh=FWD_THRESHOLD,
                                              rev_thresh=REV_THRESHOLD,
                                              turn_thresh=TURN_THRESHOLD)
        
        if behavioral_states is not None:
            print("\nClasificación de estados completada.")

            # --- 2d. Crear DataFrame y Guardar en CSV ---
            # Guardamos un CSV completo con los datos originales y el estado
            df_final = pd.DataFrame({
                'velocity': velocity,
                'angular_velocity': angular_velocity,
                'behavioral_state': behavioral_states
            })
            
            df_final.to_csv(csv_filename, index=False)
            
            print(f"\n¡Archivo CSV '{csv_filename}' guardado exitosamente!")
            print(f"El archivo contiene {len(df_final)} filas y 3 columnas.")
            print("\nVista previa de los datos guardados:")
            print(df_final.head(10))
        
        else:
            print("\nLa clasificación falló. No se guardó el CSV.")

except KeyError as e:
    print(f"--- ERROR DE CLAVE (KeyError) ---")
    print(f"No se pudo encontrar la ruta dentro del archivo NWB: {e}")
    print("Verifica que las rutas como 'nwb.processing[\"Behavior\"]' sean correctas.")
except Exception as e:
    print(f"\nOcurrió un error inesperado: {e}")
finally:
    # --- 2e. Cerrar el archivo NWB ---
    if io:
        io.close()
        print("\nArchivo NWB cerrado.")
        