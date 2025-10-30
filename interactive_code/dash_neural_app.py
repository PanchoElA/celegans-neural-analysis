"""
Dash Neural App - Interactive visualizations with embedded controls

Provides all the functionality requested:
- Interactive PCA/derivatives plots with live updates
- Behavior clustering with color overlay
- Preprocessing options
- All controls update the plot in real-time within the same interface

Run this and open http://127.0.0.1:8050 in your browser
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go

DEFAULT_DATA = 'neural_data_dataframe.csv'

def load_data(path=None):
    """Load neural data from CSV or generate synthetic data"""
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded data from {path}: {df.shape}")
    elif os.path.exists(DEFAULT_DATA):
        df = pd.read_csv(DEFAULT_DATA)
        print(f"Loaded default data: {df.shape}")
    else:
        # Generate synthetic neural data
        print("Generating synthetic neural data...")
        np.random.seed(42)
        t = np.linspace(0, 100, 1000)
        n_neurons = 25
        
        # Create realistic neural activity patterns
        data = np.zeros((len(t), n_neurons))
        for i in range(n_neurons):
            # Base oscillation with different frequencies
            freq = 0.05 + i * 0.02
            data[:, i] = (
                np.sin(2 * np.pi * freq * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.01 * t)) +
                0.5 * np.random.randn(len(t)) +
                np.exp(-((t - 50)**2) / 200) * np.sin(2 * np.pi * 0.1 * t)
            )
        
        cols = ['Time_minutes'] + [f'Neuron_{i+1:03d}' for i in range(n_neurons)]
        df = pd.DataFrame(np.column_stack((t, data)), columns=cols)
        print(f"Generated synthetic data: {df.shape}")
    
    return df

def preprocess_data(df, method, apply_filter, window):
    """Preprocess neural data with scaling and optional filtering"""
    neuron_cols = [c for c in df.columns if 'neuron' in c.lower()]
    if not neuron_cols:
        raise ValueError("No neuron columns found! Expected columns with 'neuron' in name.")
    
    X = df[neuron_cols].values.astype(float)
    
    # Apply temporal filtering if requested
    if apply_filter and window > 0:
        if window % 2 == 0:
            window += 1  # Ensure odd window size
        window = max(3, min(window, X.shape[0] - 1))  # Bound window size
        
        print(f"Applying Savitzky-Golay filter (window={window})")
        for i in range(X.shape[1]):
            if X.shape[0] > window:
                try:
                    X[:, i] = savgol_filter(X[:, i], window, 3)
                except:
                    pass  # Skip if filter fails
    
    # Apply scaling
    if method == 'StandardScaler':
        X = StandardScaler().fit_transform(X)
    elif method == 'MinMaxScaler':
        X = MinMaxScaler().fit_transform(X)
    elif method == 'RobustScaler':
        X = RobustScaler().fit_transform(X)
    elif method == 'Z-Score':
        X = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-9)
    
    return X, neuron_cols

def compute_derivatives(X):
    """Compute temporal derivatives using Savitzky-Golay filter"""
    if X is None or X.shape[0] < 5:
        return None
    
    try:
        window = min(5, X.shape[0] - 1)
        if window % 2 == 0:
            window -= 1
        if window >= 3:
            D = np.array([savgol_filter(X[:, i], window, 3, deriv=1) 
                         for i in range(X.shape[1])]).T
        else:
            D = np.gradient(X, axis=0)
    except:
        D = np.gradient(X, axis=0)
    
    return D

def identify_behaviors(scores, n_clusters=4):
    """Identify behavioral states using K-means clustering"""
    if scores.shape[1] < 3:
        # Use all available components if less than 3
        data_for_clustering = scores
    else:
        # Use first 3 PCs for clustering
        data_for_clustering = scores[:, :3]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_for_clustering)
    
    return labels, kmeans.cluster_centers_

# Initialize the Dash app
app = dash.Dash(__name__)

# Load data at startup
df_global = load_data()

app.layout = html.Div([
    html.Div([
        html.H1('🧠 Neural Dynamics Interactive Viewer', 
               style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P('Visualización interactiva de PCA neuronal con overlay de comportamientos', 
               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '20px'}),
        html.Div(id='status-display', 
                style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#ecf0f1', 
                       'borderRadius': '5px', 'margin': '10px'})
    ]),
    
    html.Div([
        # Left panel - Controls
        html.Div([
            html.H3('⚙️ Preprocessing', style={'color': '#34495e'}),
            
            html.Label('Scaling Method:'),
            dcc.Dropdown(
                id='preprocessing-method',
                options=[
                    {'label': 'Standard Scaler', 'value': 'StandardScaler'},
                    {'label': 'MinMax Scaler', 'value': 'MinMaxScaler'},
                    {'label': 'Robust Scaler', 'value': 'RobustScaler'},
                    {'label': 'Z-Score', 'value': 'Z-Score'},
                    {'label': 'None', 'value': 'None'}
                ],
                value='StandardScaler',
                style={'marginBottom': '15px'}
            ),
            
            dcc.Checklist(
                id='apply-filter',
                options=[{'label': 'Apply Savitzky-Golay temporal filter', 'value': 'apply'}],
                value=[],
                style={'marginBottom': '10px'}
            ),
            
            html.Label('Filter Window Size (odd numbers):'),
            dcc.Slider(
                id='filter-window',
                min=3, max=21, step=2, value=5,
                marks={i: str(i) for i in range(3, 22, 2)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Hr(style={'margin': '20px 0'}),
            
            html.H3('📊 Visualization', style={'color': '#34495e'}),
            
            html.Label('Plot Type:'),
            dcc.RadioItems(
                id='plot-type',
                options=[
                    {'label': 'PC vs Time (1D)', 'value': 'pc_time'},
                    {'label': 'PC vs PC (2D)', 'value': 'pc_2d'},
                    {'label': 'PCA Space (3D)', 'value': 'pca_3d'},
                    {'label': 'Derivatives Space (3D)', 'value': 'deriv_3d'}
                ],
                value='pca_3d',
                style={'marginBottom': '15px'}
            ),
            
            html.Label('PC Components:'),
            html.Div([
                html.Div([
                    html.Label('X (PC):'),
                    dcc.Input(id='pc-x', type='number', value=1, min=1, max=10, 
                             style={'width': '60px', 'marginLeft': '5px'})
                ], style={'display': 'inline-block', 'marginRight': '10px'}),
                
                html.Div([
                    html.Label('Y (PC):'),
                    dcc.Input(id='pc-y', type='number', value=2, min=1, max=10,
                             style={'width': '60px', 'marginLeft': '5px'})
                ], style={'display': 'inline-block', 'marginRight': '10px'}),
                
                html.Div([
                    html.Label('Z (PC):'),
                    dcc.Input(id='pc-z', type='number', value=3, min=1, max=10,
                             style={'width': '60px', 'marginLeft': '5px'})
                ], style={'display': 'inline-block'})
            ], style={'marginBottom': '15px'}),
            
            html.Hr(style={'margin': '20px 0'}),
            
            html.H3('🎨 Behaviors', style={'color': '#34495e'}),
            
            dcc.Checklist(
                id='show-behaviors',
                options=[{'label': 'Show behavior overlay (colors by cluster)', 'value': 'show'}],
                value=['show'],
                style={'marginBottom': '10px'}
            ),
            
            html.Label('Number of Behavior Clusters:'),
            dcc.Slider(
                id='n-clusters',
                min=2, max=8, step=1, value=4,
                marks={i: str(i) for i in range(2, 9)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Hr(style={'margin': '20px 0'}),
            
            html.Button('🔄 Update Visualization', id='update-button', 
                       style={'width': '100%', 'padding': '10px', 'backgroundColor': '#3498db', 
                              'color': 'white', 'border': 'none', 'borderRadius': '5px',
                              'fontSize': '16px', 'cursor': 'pointer'})
            
        ], style={'width': '300px', 'padding': '20px', 'backgroundColor': '#f8f9fa',
                 'borderRight': '2px solid #dee2e6', 'height': '100vh', 'overflowY': 'auto',
                 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # Right panel - Visualization
        html.Div([
            dcc.Graph(id='main-plot', style={'height': '90vh'})
        ], style={'display': 'inline-block', 'width': 'calc(100% - 320px)', 
                 'padding': '10px', 'verticalAlign': 'top'})
    ])
])

@app.callback(
    [Output('main-plot', 'figure'),
     Output('status-display', 'children')],
    [Input('update-button', 'n_clicks')],
    [State('preprocessing-method', 'value'),
     State('apply-filter', 'value'),
     State('filter-window', 'value'),
     State('plot-type', 'value'),
     State('pc-x', 'value'),
     State('pc-y', 'value'),
     State('pc-z', 'value'),
     State('show-behaviors', 'value'),
     State('n-clusters', 'value')]
)
def update_visualization(n_clicks, preproc_method, apply_filter_list, filter_window, 
                        plot_type, pc_x, pc_y, pc_z, show_behaviors_list, n_clusters):
    """Main callback to update the visualization based on user inputs"""
    
    try:
        # Process parameters
        apply_filter = 'apply' in (apply_filter_list or [])
        show_behaviors = 'show' in (show_behaviors_list or [])
        
        # Ensure valid PC indices
        pc_x = max(1, pc_x or 1)
        pc_y = max(1, pc_y or 2) 
        pc_z = max(1, pc_z or 3)
        
        # Preprocess data
        X, neuron_cols = preprocess_data(df_global, preproc_method, apply_filter, filter_window or 5)
        
        # Compute PCA
        n_components = max(pc_x, pc_y, pc_z, 5)
        pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0]))
        pca_scores = pca.fit_transform(X)
        
        # Status update
        explained_var = pca.explained_variance_ratio_
        status_text = f"✅ PCA computed | Explained variance (PC1-3): {explained_var[:3].round(3)} | Shape: {pca_scores.shape}"
        
        # Identify behaviors if requested
        behavior_labels = None
        if show_behaviors:
            behavior_labels, _ = identify_behaviors(pca_scores, n_clusters or 4)
            status_text += f" | Behaviors: {len(np.unique(behavior_labels))} clusters"
        
        # Create the plot based on selected type
        fig = go.Figure()
        
        if plot_type == 'pc_time':
            # PC vs Time plot
            pc_idx = pc_x - 1
            if pc_idx >= pca_scores.shape[1]:
                pc_idx = 0
                
            if behavior_labels is not None:
                # Color by behavior
                behavior_colors = px.colors.qualitative.Set1
                for i, behavior in enumerate(np.unique(behavior_labels)):
                    mask = behavior_labels == behavior
                    color = behavior_colors[i % len(behavior_colors)]
                    fig.add_trace(go.Scatter(
                        x=df_global['Time_minutes'].iloc[mask] if 'Time_minutes' in df_global.columns else np.arange(len(mask))[mask],
                        y=pca_scores[mask, pc_idx],
                        mode='markers+lines',
                        name=f'Behavior {behavior}',
                        line=dict(color=color),
                        marker=dict(color=color, size=4)
                    ))
            else:
                # Color by time
                fig.add_trace(go.Scatter(
                    x=df_global['Time_minutes'] if 'Time_minutes' in df_global.columns else np.arange(len(pca_scores)),
                    y=pca_scores[:, pc_idx],
                    mode='lines',
                    name=f'PC{pc_x}',
                    line=dict(color='steelblue')
                ))
            
            fig.update_layout(
                title=f'Principal Component {pc_x} vs Time',
                xaxis_title='Time (minutes)' if 'Time_minutes' in df_global.columns else 'Time points',
                yaxis_title=f'PC{pc_x} Score'
            )
            
        elif plot_type == 'pc_2d':
            # PC vs PC plot (2D)
            pc_x_idx = min(pc_x - 1, pca_scores.shape[1] - 1)
            pc_y_idx = min(pc_y - 1, pca_scores.shape[1] - 1)
            
            if behavior_labels is not None:
                # Color by behavior
                fig = px.scatter(
                    x=pca_scores[:, pc_x_idx], 
                    y=pca_scores[:, pc_y_idx],
                    color=behavior_labels.astype(str),
                    color_discrete_sequence=px.colors.qualitative.Set1,
                    title=f'PC{pc_x} vs PC{pc_y} (Colored by Behavior)',
                    labels={'color': 'Behavior Cluster'}
                )
            else:
                # Color by time
                time_data = df_global['Time_minutes'] if 'Time_minutes' in df_global.columns else np.arange(len(pca_scores))
                fig = px.scatter(
                    x=pca_scores[:, pc_x_idx], 
                    y=pca_scores[:, pc_y_idx],
                    color=time_data,
                    color_continuous_scale='Viridis',
                    title=f'PC{pc_x} vs PC{pc_y} (Colored by Time)'
                )
            
            fig.update_layout(
                xaxis_title=f'PC{pc_x} ({explained_var[pc_x_idx]:.1%} var)',
                yaxis_title=f'PC{pc_y} ({explained_var[pc_y_idx]:.1%} var)'
            )
            
        elif plot_type == 'pca_3d':
            # 3D PCA plot
            pc_x_idx = min(pc_x - 1, pca_scores.shape[1] - 1)
            pc_y_idx = min(pc_y - 1, pca_scores.shape[1] - 1) 
            pc_z_idx = min(pc_z - 1, pca_scores.shape[1] - 1)
            
            if behavior_labels is not None:
                # Color by behavior with distinct colors for each cluster
                fig = px.scatter_3d(
                    x=pca_scores[:, pc_x_idx],
                    y=pca_scores[:, pc_y_idx], 
                    z=pca_scores[:, pc_z_idx],
                    color=behavior_labels.astype(str),
                    color_discrete_sequence=px.colors.qualitative.Set1,
                    title='3D PCA Space (Colored by Behavior)',
                    labels={'color': 'Behavior Cluster'}
                )
            else:
                # Color by time
                time_data = df_global['Time_minutes'] if 'Time_minutes' in df_global.columns else np.arange(len(pca_scores))
                fig = px.scatter_3d(
                    x=pca_scores[:, pc_x_idx],
                    y=pca_scores[:, pc_y_idx],
                    z=pca_scores[:, pc_z_idx], 
                    color=time_data,
                    color_continuous_scale='Viridis',
                    title='3D PCA Space (Colored by Time)'
                )
            
            fig.update_layout(
                scene=dict(
                    xaxis_title=f'PC{pc_x} ({explained_var[pc_x_idx]:.1%} var)',
                    yaxis_title=f'PC{pc_y} ({explained_var[pc_y_idx]:.1%} var)', 
                    zaxis_title=f'PC{pc_z} ({explained_var[pc_z_idx]:.1%} var)'
                )
            )
            
        elif plot_type == 'deriv_3d':
            # 3D Derivatives plot
            D = compute_derivatives(X)
            if D is not None:
                pca_deriv = PCA(n_components=min(max(pc_x, pc_y, pc_z, 5), D.shape[1], D.shape[0]))
                deriv_scores = pca_deriv.fit_transform(D)
                
                pc_x_idx = min(pc_x - 1, deriv_scores.shape[1] - 1)
                pc_y_idx = min(pc_y - 1, deriv_scores.shape[1] - 1)
                pc_z_idx = min(pc_z - 1, deriv_scores.shape[1] - 1)
                
                if behavior_labels is not None:
                    # Color by behavior
                    fig = px.scatter_3d(
                        x=deriv_scores[:, pc_x_idx],
                        y=deriv_scores[:, pc_y_idx],
                        z=deriv_scores[:, pc_z_idx],
                        color=behavior_labels.astype(str),
                        color_discrete_sequence=px.colors.qualitative.Set1,
                        title='3D Derivatives PCA Space (Colored by Behavior)',
                        labels={'color': 'Behavior Cluster'}
                    )
                else:
                    # Color by time
                    time_data = df_global['Time_minutes'] if 'Time_minutes' in df_global.columns else np.arange(len(deriv_scores))
                    fig = px.scatter_3d(
                        x=deriv_scores[:, pc_x_idx],
                        y=deriv_scores[:, pc_y_idx], 
                        z=deriv_scores[:, pc_z_idx],
                        color=time_data,
                        color_continuous_scale='Plasma',
                        title='3D Derivatives PCA Space (Colored by Time)'
                    )
                
                deriv_explained_var = pca_deriv.explained_variance_ratio_
                fig.update_layout(
                    scene=dict(
                        xaxis_title=f'Deriv PC{pc_x} ({deriv_explained_var[pc_x_idx]:.1%} var)',
                        yaxis_title=f'Deriv PC{pc_y} ({deriv_explained_var[pc_y_idx]:.1%} var)',
                        zaxis_title=f'Deriv PC{pc_z} ({deriv_explained_var[pc_z_idx]:.1%} var)'
                    )
                )
            else:
                fig.add_annotation(text="Cannot compute derivatives for this dataset", 
                                 showarrow=False, x=0.5, y=0.5)
                status_text += " | ❌ Cannot compute derivatives"
        
        # Update layout
        fig.update_layout(
            margin=dict(l=0, r=0, t=50, b=0),
            height=800,
            font=dict(size=12),
            showlegend=True
        )
        
        return fig, status_text
        
    except Exception as e:
        # Error handling
        error_msg = f"❌ Error: {str(e)}"
        empty_fig = go.Figure()
        empty_fig.add_annotation(text=f"Error: {str(e)}", showarrow=False, x=0.5, y=0.5)
        return empty_fig, error_msg

if __name__ == '__main__':
    print("\n🚀 Starting Neural Dynamics Interactive Viewer...")
    print("📊 Loading data and initializing interface...")
    print(f"🌐 Open your browser to: http://127.0.0.1:8050")
    print("🎯 Use the controls on the left to adjust preprocessing and visualization")
    print("🎨 Toggle 'Show behavior overlay' to see clusters with different colors")
    print("\n" + "="*60)
    
    app.run(debug=False, port=8050, host='127.0.0.1')