"""
Integrated Dash App - Embedded interactive visualizations

Starts a Dash server with controls for:
- PCA: 1D (PC vs time), 2D (PC vs PC), 3D PCA
- Derivatives vs FR toggle
- Behavior clustering overlay (colors per behavior) and filter
- Preprocessing options

Opens the Dash app in a pywebview window so the interface stays as a single desktop window.
"""

import os
import threading
import tempfile
import webbrowser
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter

import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

import webview

DEFAULT_DATA = 'neural_data_dataframe.csv'

def load_data(path=None):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
    elif os.path.exists(DEFAULT_DATA):
        df = pd.read_csv(DEFAULT_DATA)
    else:
        # synthetic
        np.random.seed(0)
        t = np.linspace(0, 100, 500)
        n = 30
        data = np.zeros((len(t), n))
        for i in range(n):
            data[:, i] = np.sin(2 * np.pi * (0.05 + i*0.01) * t) + 0.3*np.random.randn(len(t))
        cols = ['Time_minutes'] + [f'Neuron_{i+1:03d}' for i in range(n)]
        df = pd.DataFrame(np.column_stack((t, data)), columns=cols)
    return df

def preprocess(df, method, apply_filter, window):
    neuron_cols = [c for c in df.columns if 'neuron' in c.lower() or 'Neuron' in c]
    X = df[neuron_cols].values.astype(float)
    if apply_filter:
        if window % 2 == 0:
            window += 1
        for i in range(X.shape[1]):
            if X.shape[0] > window:
                X[:, i] = savgol_filter(X[:, i], window, 3)
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
    if X is None:
        return None
    if X.shape[0] > 5:
        D = np.array([savgol_filter(X[:, i], 5, 3, deriv=1) for i in range(X.shape[1])]).T
    else:
        D = np.gradient(X, axis=0)
    return D

def compute_behaviors(scores, n_states=4):
    k = KMeans(n_clusters=n_states, random_state=0).fit(scores[:, :3])
    return k.labels_, k.cluster_centers_

def make_app(df):
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.Div([
            html.H2('Neural Dynamics - Integrated Viewer', style={'margin-bottom': '5px'}),
            html.Div(id='status', children='Carga datos...')
        ], style={'padding': '10px'}),

        html.Div([
            html.Div([  # left controls
                html.Label('Preprocessing'),
                dcc.Dropdown(id='preproc', options=[
                    {'label':'StandardScaler','value':'StandardScaler'},
                    {'label':'MinMaxScaler','value':'MinMaxScaler'},
                    {'label':'RobustScaler','value':'RobustScaler'},
                    {'label':'Z-Score','value':'Z-Score'},
                    {'label':'None','value':'None'}], value='StandardScaler'),
                dcc.Checklist(id='apply_filter', options=[{'label':'Apply Savitzky-Golay filter','value':'yes'}], value=[]),
                html.Label('Filter window (odd)'),
                dcc.Input(id='filter_window', type='number', value=5, min=3, step=2),

                html.Hr(),
                html.Label('Plot type'),
                dcc.RadioItems(id='plot_type', options=[
                    {'label':'PC vs Time','value':'pc_time'},
                    {'label':'PC vs PC','value':'pc_2d'},
                    {'label':'PCA 3D','value':'pca_3d'},
                    {'label':'Derivatives 3D','value':'deriv_3d'}], value='pca_3d'),

                html.Hr(),
                html.Label('PC components'),
                dcc.Input(id='pc_x', type='number', value=1, min=1),
                dcc.Input(id='pc_y', type='number', value=2, min=1),
                dcc.Input(id='pc_z', type='number', value=3, min=1),

                html.Hr(),
                html.Label('Behavior overlay'),
                dcc.Checklist(id='show_behaviors', options=[{'label':'Show behaviors overlay','value':'yes'}], value=['yes']),
                html.Label('Number of behavior clusters'),
                dcc.Slider(id='n_behaviors', min=2, max=6, step=1, value=4),

                html.Hr(),
                html.Button('Update plot', id='update_btn'),
                html.Div(id='hidden_div', style={'display':'none'})
            ], style={'width':'280px','padding':'10px','display':'inline-block','verticalAlign':'top','border-right':'1px solid #ccc'}),

            html.Div([  # right plot
                dcc.Graph(id='main_graph', style={'height':'82vh'})
            ], style={'display':'inline-block','width':'calc(100% - 300px)','padding':'10px'})
        ])
    ])

    @app.callback(
        Output('main_graph', 'figure'),
        Output('status', 'children'),
        Input('update_btn', 'n_clicks'),
        State('preproc', 'value'),
        State('apply_filter', 'value'),
        State('filter_window', 'value'),
        State('plot_type', 'value'),
        State('pc_x', 'value'),
        State('pc_y', 'value'),
        State('pc_z', 'value'),
        State('show_behaviors', 'value'),
        State('n_behaviors', 'value')
    )
    def update_graph(n_clicks, preproc, apply_filter_val, filter_window, plot_type, pc_x, pc_y, pc_z, show_beh, n_behav):
        try:
            X, neuron_cols = preprocess(df, preproc if preproc!='None' else None, 'yes' in apply_filter_val if apply_filter_val else False, int(filter_window) if filter_window else 5)
            D = compute_derivatives(X)
            pca = PCA(n_components=max(pc_x,pc_y,pc_z,3))
            scores = pca.fit_transform(X)
            status = f'PCA computed (explained: {pca.explained_variance_ratio_[:3].round(3)})'

            fig = go.Figure()

            if show_beh and 'yes' in show_beh:
                labels, centroids = compute_behaviors(scores, int(n_behav))
            else:
                labels = None

            if plot_type == 'pc_time':
                idx = pc_x-1
                if labels is None:
                    fig.add_trace(go.Scatter(x=df['Time_minutes'], y=scores[:,idx], mode='lines'))
                else:
                    for b in np.unique(labels):
                        mask = labels==b
                        fig.add_trace(go.Scatter(x=df['Time_minutes'][mask], y=scores[mask,idx], mode='markers', name=f'Behavior {b}'))
                fig.update_layout(title=f'PC{pc_x} vs Time')

            elif plot_type == 'pc_2d':
                xidx, yidx = pc_x-1, pc_y-1
                if labels is None:
                    fig = px.scatter(x=scores[:,xidx], y=scores[:,yidx], color=df['Time_minutes'], color_continuous_scale='Viridis')
                else:
                    fig = px.scatter(x=scores[:,xidx], y=scores[:,yidx], color=labels.astype(str), color_discrete_sequence=px.colors.qualitative.Set1)
                fig.update_layout(title=f'PC{pc_x} vs PC{pc_y}')

            elif plot_type == 'pca_3d':
                xidx,yidx,zidx = pc_x-1, pc_y-1, pc_z-1
                if labels is None:
                    fig = px.scatter_3d(x=scores[:,xidx], y=scores[:,yidx], z=scores[:,zidx], color=df['Time_minutes'], color_continuous_scale='Viridis')
                else:
                    fig = px.scatter_3d(x=scores[:,xidx], y=scores[:,yidx], z=scores[:,zidx], color=labels.astype(str), color_discrete_sequence=px.colors.qualitative.Set1)
                fig.update_layout(title='PCA 3D')

            elif plot_type == 'deriv_3d':
                if D is None:
                    status = 'Not enough data for derivatives'
                    return go.Figure(), status
                pca_d = PCA(n_components=max(pc_x,pc_y,pc_z,3))
                ds = pca_d.fit_transform(D)
                xidx,yidx,zidx = pc_x-1, pc_y-1, pc_z-1
                if labels is None:
                    fig = px.scatter_3d(x=ds[:,xidx], y=ds[:,yidx], z=ds[:,zidx], color=df['Time_minutes'], color_continuous_scale='Plasma')
                else:
                    fig = px.scatter_3d(x=ds[:,xidx], y=ds[:,yidx], z=ds[:,zidx], color=labels.astype(str), color_discrete_sequence=px.colors.qualitative.Set1)
                fig.update_layout(title='Derivatives PCA 3D')

            else:
                fig = go.Figure()

            fig.update_layout(margin={'l':0,'r':0,'b':0,'t':30})
            return fig, status

        except Exception as e:
            return go.Figure(), f'Error: {str(e)}'

    return app

def run():
    df = load_data()
    app = make_app(df)

    # Run Dash in a thread
    def start_dash():
        # modern Dash uses app.run
        app.run(port=8050, debug=False)

    t = threading.Thread(target=start_dash, daemon=True)
    t.start()

    # open in webview window
    url = 'http://127.0.0.1:8050'
    webview.create_window('Neural Integrated Viewer', url, width=1200, height=800)
    webview.start()

if __name__ == '__main__':
    run()
