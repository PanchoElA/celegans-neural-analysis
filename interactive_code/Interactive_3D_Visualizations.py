"""
Interactive 3D Neural Dynamics Visualizations
Enhanced with Le Cunff et al. 2024 Methodologies

This script creates advanced interactive 3D visualizations for C. elegans neural data:
- Interactive PCA trajectories with temporal evolution
- 3D Derivatives space with behavioral states
- Neural network topology in 3D space
- Real-time trajectory analysis with animations
- Multi-dimensional behavioral state transitions
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
from scipy.spatial.distance import pdist, squareform
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')

class Interactive3DNeuralVisualizer:
    """
    Advanced 3D Interactive Visualization Suite for C. elegans Neural Dynamics
    Based on Le Cunff et al. 2024 enhanced methodologies
    """
    
    def __init__(self, data_file='neural_data_dataframe.csv'):
        """Initialize the visualizer with neural data"""
        print("Initializing Interactive 3D Neural Visualizer...")
        # Load and prepare data
        # Resolve data file: prefer provided path, else fall back to repo root (parent of this script)
        if not os.path.exists(data_file):
            possible = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', data_file))
            if os.path.exists(possible):
                data_file = possible

        self.data = pd.read_csv(data_file)
        print(f"Loaded neural data: {self.data.shape}")
        
        # Prepare neural activity matrix
        neuron_cols = [col for col in self.data.columns if col.startswith('Neuron_')]
        self.neural_data = self.data[neuron_cols].values
        self.time_points = np.arange(len(self.neural_data))
        self.n_neurons = len(neuron_cols)
        self.neuron_names = neuron_cols
        
        # Initialize components
        self.scaler = StandardScaler()
        self.pca = None
        self.pca_scores = None
        self.derivatives = None
        self.pca_derivatives = None
        self.behavioral_states = None
        
        print(f"Neural matrix: {self.neural_data.shape} (timepoints x neurons)")
        print("Ready for 3D visualization")
    
    def compute_enhanced_pca(self, n_components=3):
        """Compute PCA with enhanced statistical validation"""
        print(f"\nComputing Enhanced PCA ({n_components} components)...")
        
        # Standardize data
        neural_scaled = self.scaler.fit_transform(self.neural_data)
        
        # Compute PCA
        self.pca = PCA(n_components=n_components)
        self.pca_scores = self.pca.fit_transform(neural_scaled)
        
        # Calculate explained variance
        explained_var = self.pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        print(f"Explained variance: {explained_var}")
        print(f"Cumulative variance: {cumulative_var}")
        
        return self.pca_scores
    
    def compute_derivatives(self, method='savgol', window=5):
        """Compute neural activity derivatives with multiple methods"""
        print(f"\nComputing derivatives using {method} method...")
        
        if method == 'savgol':
            # Savitzky-Golay filter derivatives
            self.derivatives = np.array([
                savgol_filter(self.neural_data[:, i], window, 3, deriv=1)
                for i in range(self.n_neurons)
            ]).T
        
        elif method == 'gradient':
            # Simple gradient
            self.derivatives = np.gradient(self.neural_data, axis=0)
        
        elif method == 'central':
            # Central differences
            self.derivatives = np.zeros_like(self.neural_data)
            self.derivatives[1:-1] = (self.neural_data[2:] - self.neural_data[:-2]) / 2
            self.derivatives[0] = self.neural_data[1] - self.neural_data[0]
            self.derivatives[-1] = self.neural_data[-1] - self.neural_data[-2]
        
        print(f"Computed derivatives: {self.derivatives.shape}")
        
        # PCA on derivatives for 3D visualization
        derivatives_scaled = self.scaler.fit_transform(self.derivatives)
        pca_derivatives = PCA(n_components=3)
        self.pca_derivatives = pca_derivatives.fit_transform(derivatives_scaled)
        
        return self.derivatives, self.pca_derivatives
    
    def identify_behavioral_states(self, n_states=4):
        """Identify behavioral states using clustering on PCA space"""
        print(f"\nIdentifying {n_states} behavioral states...")
        
        if self.pca_scores is None:
            self.compute_enhanced_pca()
        
        # K-means clustering on PCA scores
        kmeans = KMeans(n_clusters=n_states, random_state=42)
        self.behavioral_states = kmeans.fit_predict(self.pca_scores)
        
        # Define state names and colors
        state_names = [f'State_{i+1}' for i in range(n_states)]
        state_colors = px.colors.qualitative.Set1[:n_states]
        
        self.state_info = {
            'labels': self.behavioral_states,
            'names': state_names,
            'colors': state_colors,
            'n_states': n_states
        }
        
        print(f"Identified behavioral states: {np.unique(self.behavioral_states)}")
        return self.behavioral_states
    
    def create_interactive_pca_trajectory(self):
        """Create interactive 3D PCA trajectory with temporal evolution"""
        print("\nCreating Interactive 3D PCA Trajectory...")
        
        # Ensure PCA is computed
        self.compute_enhanced_pca()
        
        # Ensure behavioral states are identified
        self.identify_behavioral_states()
        
        # Create figure
        fig = go.Figure()
        
        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=self.pca_scores[:, 0],
            y=self.pca_scores[:, 1],
            z=self.pca_scores[:, 2],
            mode='lines',
            line=dict(
                color=self.time_points,
                colorscale='Viridis',
                width=4,
                colorbar=dict(title="Time", x=1.1)
            ),
            name='Neural Trajectory',
            hovertemplate='<b>Time:</b> %{text}<br>' +
                         '<b>PC1:</b> %{x:.3f}<br>' +
                         '<b>PC2:</b> %{y:.3f}<br>' +
                         '<b>PC3:</b> %{z:.3f}<extra></extra>',
            text=[f't={t}' for t in self.time_points]
        ))
        
        # Add behavioral state points
        for state in np.unique(self.behavioral_states):
            mask = self.behavioral_states == state
            fig.add_trace(go.Scatter3d(
                x=self.pca_scores[mask, 0],
                y=self.pca_scores[mask, 1],
                z=self.pca_scores[mask, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color=self.state_info['colors'][state],
                    opacity=0.8
                ),
                name=f'Behavioral {self.state_info["names"][state]}',
                hovertemplate=f'<b>State:</b> {self.state_info["names"][state]}<br>' +
                             '<b>PC1:</b> %{x:.3f}<br>' +
                             '<b>PC2:</b> %{y:.3f}<br>' +
                             '<b>PC3:</b> %{z:.3f}<extra></extra>'
            ))
        
        # Add start and end points
        fig.add_trace(go.Scatter3d(
            x=[self.pca_scores[0, 0]],
            y=[self.pca_scores[0, 1]],
            z=[self.pca_scores[0, 2]],
            mode='markers',
            marker=dict(size=12, color='green', symbol='diamond'),
            name='Start',
            hovertemplate='<b>START</b><br>' +
                         '<b>PC1:</b> %{x:.3f}<br>' +
                         '<b>PC2:</b> %{y:.3f}<br>' +
                         '<b>PC3:</b> %{z:.3f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[self.pca_scores[-1, 0]],
            y=[self.pca_scores[-1, 1]],
            z=[self.pca_scores[-1, 2]],
            mode='markers',
            marker=dict(size=12, color='red', symbol='x'),
            name='End',
            hovertemplate='<b>END</b><br>' +
                         '<b>PC1:</b> %{x:.3f}<br>' +
                         '<b>PC2:</b> %{y:.3f}<br>' +
                         '<b>PC3:</b> %{z:.3f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Interactive 3D Neural Trajectory - PCA Space<br>' + 
                       '<sub>Enhanced with Behavioral State Classification</sub>',
                'x': 0.5,
                'font': {'size': 16}
            },
            scene=dict(
                xaxis_title=f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)',
                yaxis_title=f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)', 
                zaxis_title=f'PC3 ({self.pca.explained_variance_ratio_[2]:.1%} variance)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                bgcolor='rgba(240, 240, 240, 0.1)'
            ),
            width=1000,
            height=800,
            margin=dict(l=0, r=0, t=80, b=0)
        )
        
        # Save interactive plot
        filename = 'Interactive_3D_PCA_Trajectory.html'
        out_path = os.path.join(os.path.dirname(__file__), filename)
        pyo.plot(fig, filename=out_path, auto_open=False)
        print(f"Saved: {out_path}")

        return fig
    
    def create_interactive_derivatives_space(self):
        """Create interactive 3D derivatives space visualization"""
        print("\nCreating Interactive 3D Derivatives Space...")
        
        if self.derivatives is None:
            self.compute_derivatives()
        
        if self.behavioral_states is None:
            self.identify_behavioral_states()
        
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=('Derivatives Trajectory', 'Derivatives Phase Space'),
            horizontal_spacing=0.1
        )
        
        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=self.pca_derivatives[:, 0],
            y=self.pca_derivatives[:, 1],
            z=self.pca_derivatives[:, 2],
            mode='lines+markers',
            line=dict(
                color=self.time_points,
                colorscale='Plasma',
                width=3
            ),
            marker=dict(
                size=4,
                color=self.behavioral_states,
                colorscale='Viridis',
                opacity=0.8
            ),
            name='Derivatives Trajectory',
            hovertemplate='<b>Time:</b> %{text}<br>' +
                         '<b>Der-PC1:</b> %{x:.3f}<br>' +
                         '<b>Der-PC2:</b> %{y:.3f}<br>' +
                         '<b>Der-PC3:</b> %{z:.3f}<extra></extra>',
            text=[f't={t}' for t in self.time_points]
        ), row=1, col=1)
        
        # Right plot: Phase space with velocity vectors
        # Calculate velocity vectors
        velocity = np.gradient(self.pca_derivatives, axis=0)
        
        # Sample points for velocity vectors (every 10th point)
        sample_idx = np.arange(0, len(self.pca_derivatives), 10)
        
        fig.add_trace(go.Scatter3d(
            x=self.pca_derivatives[sample_idx, 0],
            y=self.pca_derivatives[sample_idx, 1],
            z=self.pca_derivatives[sample_idx, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=self.behavioral_states[sample_idx],
                colorscale='Turbo',
                opacity=0.9
            ),
            name='Phase Points',
            hovertemplate='<b>Time:</b> %{text}<br>' +
                         '<b>State:</b> %{customdata}<br>' +
                         '<b>Der-PC1:</b> %{x:.3f}<br>' +
                         '<b>Der-PC2:</b> %{y:.3f}<br>' +
                         '<b>Der-PC3:</b> %{z:.3f}<extra></extra>',
            text=[f't={t}' for t in self.time_points[sample_idx]],
            customdata=[f'State_{s+1}' for s in self.behavioral_states[sample_idx]]
        ), row=1, col=2)
        
        # Add velocity vectors
        for i, idx in enumerate(sample_idx[::2]):  # Every 20th point for clarity
            if idx < len(velocity) - 1:
                fig.add_trace(go.Scatter3d(
                    x=[self.pca_derivatives[idx, 0], 
                       self.pca_derivatives[idx, 0] + velocity[idx, 0] * 0.5],
                    y=[self.pca_derivatives[idx, 1], 
                       self.pca_derivatives[idx, 1] + velocity[idx, 1] * 0.5],
                    z=[self.pca_derivatives[idx, 2], 
                       self.pca_derivatives[idx, 2] + velocity[idx, 2] * 0.5],
                    mode='lines',
                    line=dict(color='red', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=1, col=2)
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Interactive 3D Derivatives Analysis<br>' + 
                       '<sub>Neural Activity Derivatives and Phase Space</sub>',
                'x': 0.5,
                'font': {'size': 16}
            },
            width=1400,
            height=700,
            margin=dict(l=0, r=0, t=100, b=0)
        )
        
        # Update scene properties for both subplots
        fig.update_scenes(
            camera=dict(eye=dict(x=1.3, y=1.3, z=1.3)),
            bgcolor='rgba(240, 240, 240, 0.1)'
        )
        
        # Save interactive plot
        filename = 'Interactive_3D_Derivatives_Space.html'
        out_path = os.path.join(os.path.dirname(__file__), filename)
        pyo.plot(fig, filename=out_path, auto_open=False)
        print(f"Saved: {out_path}")

        return fig
    
    def create_neural_network_topology_3d(self):
        """Create 3D neural network topology visualization"""
        print("\nCreating 3D Neural Network Topology...")
        
        # Calculate correlation matrix for network connections
        correlation_matrix = np.corrcoef(self.neural_data.T)
        
        # Create 3D positions for neurons using PCA of correlation matrix
        pca_neurons = PCA(n_components=3)
        neuron_positions = pca_neurons.fit_transform(correlation_matrix)
        
        # Calculate connection strengths (use correlation threshold)
        threshold = 0.7
        strong_connections = np.abs(correlation_matrix) > threshold
        
        # Create figure
        fig = go.Figure()
        
        # Add neuron nodes
        fig.add_trace(go.Scatter3d(
            x=neuron_positions[:, 0],
            y=neuron_positions[:, 1],
            z=neuron_positions[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=np.mean(self.neural_data, axis=0),  # Color by average activity
                colorscale='Viridis',
                colorbar=dict(title="Avg Activity", x=1.1),
                opacity=0.8
            ),
            text=self.neuron_names,
            name='Neurons',
            hovertemplate='<b>Neuron:</b> %{text}<br>' +
                         '<b>Avg Activity:</b> %{marker.color:.3f}<br>' +
                         '<b>X:</b> %{x:.3f}<br>' +
                         '<b>Y:</b> %{y:.3f}<br>' +
                         '<b>Z:</b> %{z:.3f}<extra></extra>'
        ))
        
        # Add connections (edges)
        edge_trace_x = []
        edge_trace_y = []
        edge_trace_z = []
        
        for i in range(self.n_neurons):
            for j in range(i+1, self.n_neurons):
                if strong_connections[i, j]:
                    # Add edge
                    edge_trace_x.extend([neuron_positions[i, 0], neuron_positions[j, 0], None])
                    edge_trace_y.extend([neuron_positions[i, 1], neuron_positions[j, 1], None])
                    edge_trace_z.extend([neuron_positions[i, 2], neuron_positions[j, 2], None])
        
        if edge_trace_x:  # Only add if there are connections
            fig.add_trace(go.Scatter3d(
                x=edge_trace_x,
                y=edge_trace_y,
                z=edge_trace_z,
                mode='lines',
                line=dict(color='rgba(125, 125, 125, 0.5)', width=2),
                hoverinfo='none',
                showlegend=False,
                name='Connections'
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': '3D Neural Network Topology<br>' + 
                           f'<sub>Strong Connections (|r| > {threshold})</sub>',
                'x': 0.5,
                'font': {'size': 16}
            },
            scene=dict(
                xaxis_title='Neural PC1',
                yaxis_title='Neural PC2',
                zaxis_title='Neural PC3',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                bgcolor='rgba(240, 240, 240, 0.1)'
            ),
            width=1000,
            height=800,
            margin=dict(l=0, r=0, t=80, b=0)
        )
        
        # Save interactive plot
        filename = 'Interactive_3D_Neural_Network.html'
        out_path = os.path.join(os.path.dirname(__file__), filename)
        pyo.plot(fig, filename=out_path, auto_open=False)
        print(f"Saved: {out_path}")

        return fig
    
    def create_animated_trajectory(self):
        """Create animated 3D trajectory showing temporal evolution"""
        print("\nCreating Animated 3D Trajectory...")
        
        if self.pca_scores is None:
            self.compute_enhanced_pca()
        
        # Create frames for animation
        frames = []
        n_frames = min(50, len(self.pca_scores))  # Limit frames for performance
        frame_step = len(self.pca_scores) // n_frames
        
        for i in range(0, len(self.pca_scores), frame_step):
            end_idx = min(i + frame_step, len(self.pca_scores))
            
            frame = go.Frame(
                data=[
                    go.Scatter3d(
                        x=self.pca_scores[:end_idx, 0],
                        y=self.pca_scores[:end_idx, 1],
                        z=self.pca_scores[:end_idx, 2],
                        mode='lines+markers',
                        line=dict(color='blue', width=4),
                        marker=dict(
                            size=4,
                            color=np.arange(end_idx),
                            colorscale='Viridis'
                        ),
                        name=f'Trajectory (t={end_idx})'
                    )
                ],
                name=str(i)
            )
            frames.append(frame)
        
        # Create initial figure
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=[self.pca_scores[0, 0]],
                    y=[self.pca_scores[0, 1]], 
                    z=[self.pca_scores[0, 2]],
                    mode='markers',
                    marker=dict(size=8, color='red'),
                    name='Current Position'
                )
            ],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            title={
                'text': 'Animated Neural Trajectory<br><sub>Temporal Evolution in PCA Space</sub>',
                'x': 0.5,
                'font': {'size': 16}
            },
            scene=dict(
                xaxis_title=f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%})',
                yaxis_title=f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%})',
                zaxis_title=f'PC3 ({self.pca.explained_variance_ratio_[2]:.1%})',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1000,
            height=800
        )
        
        # Save animated plot
        filename = 'Animated_3D_Neural_Trajectory.html'
        out_path = os.path.join(os.path.dirname(__file__), filename)
        pyo.plot(fig, filename=out_path, auto_open=False)
        print(f"Saved: {out_path}")

        return fig
    
    def create_comprehensive_3d_dashboard(self):
        """Create a comprehensive 3D dashboard with multiple views"""
        print("\nCreating Comprehensive 3D Dashboard...")
        
        # Ensure all computations are done
        if self.pca_scores is None:
            self.compute_enhanced_pca()
        if self.derivatives is None:
            self.compute_derivatives()
        if self.behavioral_states is None:
            self.identify_behavioral_states()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{'type': 'scatter3d'}, {'type': 'scatter3d'}],
                [{'type': 'scatter3d'}, {'type': 'scatter3d'}]
            ],
            subplot_titles=(
                'PCA Neural Trajectory',
                'Derivatives Space', 
                'Behavioral States',
                'Neural Correlations'
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        # 1. PCA Trajectory (top-left)
        fig.add_trace(go.Scatter3d(
            x=self.pca_scores[:, 0],
            y=self.pca_scores[:, 1],
            z=self.pca_scores[:, 2],
            mode='lines+markers',
            line=dict(color=self.time_points, colorscale='Viridis', width=3),
            marker=dict(size=3, opacity=0.7),
            name='PCA Trajectory',
            showlegend=False
        ), row=1, col=1)
        
        # 2. Derivatives Space (top-right)  
        fig.add_trace(go.Scatter3d(
            x=self.pca_derivatives[:, 0],
            y=self.pca_derivatives[:, 1],
            z=self.pca_derivatives[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=self.behavioral_states,
                colorscale='Rainbow',
                opacity=0.8
            ),
            name='Derivatives',
            showlegend=False
        ), row=1, col=2)
        
        # 3. Behavioral States (bottom-left)
        for state in np.unique(self.behavioral_states):
            mask = self.behavioral_states == state
            fig.add_trace(go.Scatter3d(
                x=self.pca_scores[mask, 0],
                y=self.pca_scores[mask, 1],
                z=self.pca_scores[mask, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=self.state_info['colors'][state],
                    opacity=0.8
                ),
                name=f'State {state+1}',
                legendgroup=f'state_{state}'
            ), row=2, col=1)
        
        # 4. Neural Correlations (bottom-right)
        correlation_matrix = np.corrcoef(self.neural_data.T)
        pca_neurons = PCA(n_components=3)
        neuron_positions = pca_neurons.fit_transform(correlation_matrix)
        
        fig.add_trace(go.Scatter3d(
            x=neuron_positions[:, 0],
            y=neuron_positions[:, 1],
            z=neuron_positions[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=np.mean(self.neural_data, axis=0),
                colorscale='Plasma',
                opacity=0.8
            ),
            name='Neurons',
            showlegend=False
        ), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Comprehensive 3D Neural Dynamics Dashboard<br>' + 
                           '<sub>Enhanced Multi-View Analysis</sub>',
                'x': 0.5,
                'font': {'size': 18}
            },
            width=1400,
            height=1000,
            margin=dict(l=0, r=0, t=100, b=0)
        )
        
        # Update all scenes
        fig.update_scenes(
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)),
            bgcolor='rgba(240, 240, 240, 0.05)'
        )
        
        # Save dashboard
        filename = 'Comprehensive_3D_Dashboard.html'
        out_path = os.path.join(os.path.dirname(__file__), filename)
        pyo.plot(fig, filename=out_path, auto_open=False)
        print(f"Saved: {out_path}")

        return fig
    
    def run_all_3d_visualizations(self):
        """Execute all 3D visualization methods"""
        print("\nEXECUTING ALL 3D INTERACTIVE VISUALIZATIONS")
        print("=" * 60)

        start_time = datetime.now()
        
        try:
            # 1. PCA Trajectory
            print("\n[1/5] Creating PCA Trajectory...")
            self.create_interactive_pca_trajectory()
            
            # 2. Derivatives Space
            print("\n[2/5] Creating Derivatives Space...")
            self.create_interactive_derivatives_space()
            
            # 3. Neural Network Topology
            print("\n[3/5] Creating Neural Network Topology...")
            self.create_neural_network_topology_3d()
            
            # 4. Animated Trajectory
            print("\n[4/5] Creating Animated Trajectory...")
            self.create_animated_trajectory()
            
            # 5. Comprehensive Dashboard
            print("\n[5/5] Creating Comprehensive Dashboard...")
            self.create_comprehensive_3d_dashboard()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print("\n" + "=" * 60)
            print("ALL 3D VISUALIZATIONS COMPLETED")
            print(f"Total execution time: {duration:.1f} seconds")
            print("Generated Files:")
            print("   - Interactive_3D_PCA_Trajectory.html")
            print("   - Interactive_3D_Derivatives_Space.html") 
            print("   - Interactive_3D_Neural_Network.html")
            print("   - Animated_3D_Neural_Trajectory.html")
            print("   - Comprehensive_3D_Dashboard.html")
            print("Open these HTML files in your browser for interactive exploration")
            
        except Exception as e:
            print(f"\nError during visualization generation: {str(e)}")
            raise


def main():
    """Main execution function"""
    print("NEURAL VISUALIZER" + "=" * 60)
    print("    INTERACTIVE 3D NEURAL DYNAMICS VISUALIZER")
    print("         Enhanced Le Cunff et al. 2024 Edition")
    print("=" * 80)

    try:
        # Initialize visualizer
        visualizer = Interactive3DNeuralVisualizer()

        # Run all visualizations
        visualizer.run_all_3d_visualizations()

        print("\nSUCCESS! Interactive 3D visualizations ready")
        print("Open the generated HTML files in your web browser")
        print("Use mouse to rotate, zoom, and explore the 3D spaces")

    except FileNotFoundError:
        print("Error: 'neural_data_dataframe.csv' not found!")
        print("Please ensure the neural data file is in the current directory.")

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print("Please check your data and try again.")


if __name__ == "__main__":
    main()