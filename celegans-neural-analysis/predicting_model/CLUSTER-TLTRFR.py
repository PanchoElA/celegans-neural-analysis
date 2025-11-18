import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- 1. Cargar TUS dos archivos CSV ---
# (Asegúrate de que los nombres de archivo y columnas sean correctos)
try:
    # Cargar los estados de comportamiento que acabamos de crear
    behavior_df = pd.read_csv('predicting_model/classified_behavioral_states.csv')
    
    # Cargar tus componentes principales (PC)
    # Reemplaza 'tus_pcs.csv' con el nombre de tu archivo
    # Reemplaza ['PC1', 'PC2', 'PC3'] con los nombres reales de las columnas
    pc_df = pd.read_csv('predicting_model/neural_plus_behavior_clusters.csv') # Supongo este nombre por tus archivos de GitHub
    pc_columns = ['PC1', 'PC2', 'PC3'] # Solo usaremos los 3 primeros
    
    # --- 2. Alinear y Combinar los Datos ---
    # ¡CRÍTICO! Ambos DataFrames deben tener la misma longitud 
    # y cada fila debe corresponder al mismo punto en el tiempo.
    
    if len(behavior_df) != len(pc_df):
        print(f"Error: Los archivos no tienen la misma longitud. "
              f"Comportamiento: {len(behavior_df)}, PCs: {len(pc_df)}")
        # Aquí deberías detenerte y arreglar los datos si esto pasa
    else:
        print("Datos alineados. Combinando...")
        # Combinamos todo en un solo DataFrame
        df = pd.concat([pc_df[pc_columns], behavior_df], axis=1)

        # --- 3. Preparar los datos para K-Means ---
        # K-Means es sensible a la escala, por lo que es bueno estandarizar los PCs
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[pc_columns])

        # --- 4. Encontrar el número óptimo de clústeres (k) ---
        # Usamos el "Método del Codo" (Elbow Method)
        print("Calculando el 'Método del Codo' para encontrar k...")
        inertias = []
        possible_k = range(1, 11) # Probaremos de 1 a 10 clústeres
        
        for k in possible_k:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Graficar el método del codo
        plt.figure(figsize=(8, 4))
        plt.plot(possible_k, inertias, 'bo-')
        plt.xlabel('Número de Clústeres (k)')
        plt.ylabel('Inercia (Suma de cuadrados intra-clúster)')
        plt.title('Método del Codo para K-Means')
        plt.savefig('kmeans_elbow_plot.png')
        print("Gráfico del codo guardado como 'kmeans_elbow_plot.png'.")
        print("¡Mira el gráfico! El 'codo' (donde la curva se aplana) es tu 'k' óptimo.")
        print("Por ejemplo, si el codo está en k=4 o k=5, usa ese número.")
        
        # --- 5. Ejecutar K-Means con el 'k' elegido ---
        # ***** CAMBIA ESTE NÚMERO BASADO EN TU GRÁFICO DEL CODO *****
        K_OPTIMO = 5  
        # (Uso 5 porque tienes 5 estados de comportamiento, 
        # pero DEBES verificarlo con tu gráfico)
        
        print(f"Ejecutando K-Means con k={K_OPTIMO}...")
        kmeans = KMeans(n_clusters=K_OPTIMO, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        
        # Añadir las nuevas etiquetas de clúster neuronal al DataFrame
        df['neural_cluster'] = kmeans.labels_

        # --- 6. El Análisis: Comparar Clústeres Neuronales vs. Comportamiento ---
        print("\n--- ¡Análisis Completado! ---")
        print("Comparación de Clústeres Neuronales (filas) vs. Estados de Comportamiento (columnas)\n")
        
        # Esta tabla es el resultado principal:
        contingency_table = pd.crosstab(df['neural_cluster'], df['behavioral_state'])
        
        # Mostrar la tabla en porcentajes (más fácil de leer)
        contingency_table_percent = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
        
        print(contingency_table)
        print("\nTabla de contingencia (en % por clúster neuronal):")
        print(contingency_table_percent.round(1))

        # --- 7. Guardar el resultado ---
        df.to_csv('predicting_model/neural_plus_behavior_clusters.csv', index=False)
        print("\nResultados completos guardados en 'predicting_model/neural_plus_behavior_clusters.csv'")
        
        # --- 8. Visualizar (¡El siguiente paso!) ---
        # Ahora puedes hacer tus gráficos 3D (como los de tu repo) 
        # pero coloreando los puntos por la columna 'neural_cluster'.
        # Si esos clústeres se ven bien separados en el espacio 3D,
        # ¡es una gran señal!

except FileNotFoundError as e:
    print(f"--- ERROR DE ARCHIVO ---")
    print(f"No se pudo encontrar el archivo: {e.filename}")
    print("Por favor, verifica que los nombres 'classified_behavioral_states.csv' y 'principal_components_top5.csv' sean correctos.")
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")