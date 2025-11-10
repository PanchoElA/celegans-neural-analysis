"""
🧠 Simplified Interactive 3D Neural Visualizations
Enhanced with Le Cunff et al. 2024 Methodologies - Focus on Key Visualizations

Crea los gráficos 3D interactivos más importantes:
1. Trayectoria PCA 3D con estados conductuales
2. Espacio de derivadas 3D 
3. Dashboard integrado
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings('ignore')
import os

def create_3d_visualizations():
    """Función principal para crear visualizaciones 3D"""
    print("INICIANDO VISUALIZACIONES 3D INTERACTIVAS")
    print("=" * 60)
    
    # Cargar datos
    print("Cargando datos neurales...")
    data_file = 'neural_data_dataframe.csv'
    # prefer cwd, else try parent repo root where data usually lives
    if not os.path.exists(data_file):
        possible = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', data_file))
        if os.path.exists(possible):
            data_file = possible

    data = pd.read_csv(data_file)
    neuron_cols = [col for col in data.columns if col.startswith('Neuron_')]
    neural_data = data[neuron_cols].values
    time_points = np.arange(len(neural_data))
    
    print(f"✅ Datos cargados: {neural_data.shape} (timepoints × neuronas)")
    
    # Preparar datos
    scaler = StandardScaler()
    neural_scaled = scaler.fit_transform(neural_data)
    
    # PCA
    print("🔍 Computando PCA...")
    pca = PCA(n_components=3)
    pca_scores = pca.fit_transform(neural_scaled)
    explained_var = pca.explained_variance_ratio_
    
    print(f"📈 Varianza explicada: PC1={explained_var[0]:.1%}, PC2={explained_var[1]:.1%}, PC3={explained_var[2]:.1%}")
    
    # Estados conductuales
    print("🎭 Identificando estados conductuales...")
    kmeans = KMeans(n_clusters=4, random_state=42)
    behavioral_states = kmeans.fit_predict(pca_scores)
    
    # Colores para estados
    state_colors = ['red', 'blue', 'green', 'orange']
    
    # === VISUALIZACIÓN 1: Trayectoria PCA 3D ===
    print("\n[1/3] 🎨 Creando Trayectoria PCA 3D...")
    
    fig1 = go.Figure()
    
    # Línea de trayectoria
    fig1.add_trace(go.Scatter3d(
        x=pca_scores[:, 0],
        y=pca_scores[:, 1],
        z=pca_scores[:, 2],
        mode='lines',
        line=dict(
            color=time_points,
            colorscale='Viridis',
            width=4,
            colorbar=dict(title="Tiempo", x=1.1)
        ),
        name='Trayectoria Neural',
        hovertemplate='<b>Tiempo:</b> %{text}<br>' +
                     '<b>PC1:</b> %{x:.3f}<br>' +
                     '<b>PC2:</b> %{y:.3f}<br>' +
                     '<b>PC3:</b> %{z:.3f}<extra></extra>',
        text=[f't={t}' for t in time_points]
    ))
    
    # Puntos por estado conductual
    for state in range(4):
        mask = behavioral_states == state
        if np.any(mask):
            fig1.add_trace(go.Scatter3d(
                x=pca_scores[mask, 0],
                y=pca_scores[mask, 1],
                z=pca_scores[mask, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=state_colors[state],
                    opacity=0.7
                ),
                name=f'Estado {state+1}',
                hovertemplate=f'<b>Estado:</b> {state+1}<br>' +
                             '<b>PC1:</b> %{x:.3f}<br>' +
                             '<b>PC2:</b> %{y:.3f}<br>' +
                             '<b>PC3:</b> %{z:.3f}<extra></extra>'
            ))
    
    # Puntos inicio y fin
    fig1.add_trace(go.Scatter3d(
        x=[pca_scores[0, 0]], y=[pca_scores[0, 1]], z=[pca_scores[0, 2]],
        mode='markers',
        marker=dict(size=12, color='lime', symbol='diamond'),
        name='Inicio',
        hovertemplate='<b>INICIO</b><extra></extra>'
    ))
    
    fig1.add_trace(go.Scatter3d(
        x=[pca_scores[-1, 0]], y=[pca_scores[-1, 1]], z=[pca_scores[-1, 2]],
        mode='markers',
        marker=dict(size=12, color='red', symbol='x'),
        name='Final',
        hovertemplate='<b>FINAL</b><extra></extra>'
    ))
    
    fig1.update_layout(
        title={
            'text': '🧠 Trayectoria Neural 3D - Espacio PCA<br>' + 
                   '<sub>Estados Conductuales y Evolución Temporal</sub>',
            'x': 0.5,
            'font': {'size': 16}
        },
        scene=dict(
            xaxis_title=f'PC1 ({explained_var[0]:.1%} varianza)',
            yaxis_title=f'PC2 ({explained_var[1]:.1%} varianza)',
            zaxis_title=f'PC3 ({explained_var[2]:.1%} varianza)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            bgcolor='rgba(240, 240, 240, 0.1)'
        ),
        width=1000,
        height=800
    )
    
    out_dir = os.path.dirname(__file__)
    fname1 = os.path.join(out_dir, '3D_Trayectoria_PCA_Neural.html')
    pyo.plot(fig1, filename=fname1, auto_open=False)
    print(f"💾 Guardado: {fname1}")
    
    # === VISUALIZACIÓN 2: Espacio de Derivadas 3D ===
    print("\n[2/3] 🌊 Creando Espacio de Derivadas 3D...")
    
    # Calcular derivadas
    derivatives = np.array([
        savgol_filter(neural_data[:, i], 5, 3, deriv=1)
        for i in range(neural_data.shape[1])
    ]).T
    
    # PCA en derivadas
    derivatives_scaled = scaler.fit_transform(derivatives)
    pca_derivatives = PCA(n_components=3)
    pca_der_scores = pca_derivatives.fit_transform(derivatives_scaled)
    
    fig2 = go.Figure()
    
    # Trayectoria de derivadas
    fig2.add_trace(go.Scatter3d(
        x=pca_der_scores[:, 0],
        y=pca_der_scores[:, 1],
        z=pca_der_scores[:, 2],
        mode='lines+markers',
        line=dict(color=time_points, colorscale='Plasma', width=3),
        marker=dict(
            size=3,
            color=behavioral_states,
            colorscale='Turbo',
            opacity=0.8,
            colorbar=dict(title="Estado", x=1.1)
        ),
        name='Trayectoria Derivadas',
        hovertemplate='<b>Tiempo:</b> %{text}<br>' +
                     '<b>Der-PC1:</b> %{x:.3f}<br>' +
                     '<b>Der-PC2:</b> %{y:.3f}<br>' +
                     '<b>Der-PC3:</b> %{z:.3f}<extra></extra>',
        text=[f't={t}' for t in time_points]
    ))
    
    fig2.update_layout(
        title={
            'text': '🌊 Espacio de Derivadas 3D<br>' + 
                   '<sub>Dinámicas de Cambio Neural</sub>',
            'x': 0.5,
            'font': {'size': 16}
        },
        scene=dict(
            xaxis_title='Derivada PC1',
            yaxis_title='Derivada PC2',
            zaxis_title='Derivada PC3',
            camera=dict(eye=dict(x=1.3, y=1.3, z=1.3)),
            bgcolor='rgba(240, 240, 240, 0.1)'
        ),
        width=1000,
        height=800
    )
    
    fname2 = os.path.join(out_dir, '3D_Espacio_Derivadas_Neural.html')
    pyo.plot(fig2, filename=fname2, auto_open=False)
    print(f"💾 Guardado: {fname2}")
    
    # === VISUALIZACIÓN 3: Dashboard Combinado ===
    print("\n[3/3] 📊 Creando Dashboard Integrado 3D...")
    
    fig3 = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('🧠 Espacio PCA', '🌊 Espacio Derivadas'),
        horizontal_spacing=0.1
    )
    
    # Panel izquierdo: PCA
    fig3.add_trace(go.Scatter3d(
        x=pca_scores[:, 0],
        y=pca_scores[:, 1],
        z=pca_scores[:, 2],
        mode='lines+markers',
        line=dict(color=time_points, colorscale='Viridis', width=3),
        marker=dict(size=3, opacity=0.6),
        name='PCA Trayectoria',
        showlegend=False
    ), row=1, col=1)
    
    # Panel derecho: Derivadas
    fig3.add_trace(go.Scatter3d(
        x=pca_der_scores[:, 0],
        y=pca_der_scores[:, 1],
        z=pca_der_scores[:, 2],
        mode='lines+markers',
        line=dict(color=time_points, colorscale='Plasma', width=3),
        marker=dict(
            size=3,
            color=behavioral_states,
            colorscale='Rainbow',
            opacity=0.7
        ),
        name='Derivadas',
        showlegend=False
    ), row=1, col=2)
    
    fig3.update_layout(
        title={
            'text': '📊 Dashboard Neural 3D Integrado<br>' + 
                   '<sub>Análisis Dual: PCA y Derivadas</sub>',
            'x': 0.5,
            'font': {'size': 16}
        },
        width=1400,
        height=700
    )
    
    fig3.update_scenes(
        camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)),
        bgcolor='rgba(240, 240, 240, 0.05)'
    )
    
    fname3 = os.path.join(out_dir, '3D_Dashboard_Neural_Integrado.html')
    pyo.plot(fig3, filename=fname3, auto_open=False)
    print(f"💾 Guardado: {fname3}")
    
    print("\n🎉 ¡VISUALIZACIONES 3D COMPLETADAS!")
    print("=" * 60)
    print("📁 Archivos generados:")
    print(f"   • {fname1}")
    print(f"   • {fname2}")
    print(f"   • {fname3}")
    print("\n🌐 Abre estos archivos HTML en tu navegador para explorar!")
    print("🎮 Usa el mouse para rotar, hacer zoom y explorar el espacio 3D")
    
    return fig1, fig2, fig3

if __name__ == "__main__":
    try:
        create_3d_visualizations()
    except FileNotFoundError:
        print("❌ Error: 'neural_data_dataframe.csv' no encontrado!")
        print("Asegúrate de que el archivo esté en el directorio actual.")
    except Exception as e:
        print(f"💥 Error inesperado: {str(e)}")
        print("Por favor revisa los datos y vuelve a intentar.")