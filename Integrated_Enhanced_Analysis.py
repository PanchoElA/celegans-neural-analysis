"""
Integrated Enhanced Analysis Suite
Combines PCA, derivatives, and advanced visualizations
Based on Le Cunff et al. 2024 methodology improvements
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from scipy import stats, signal
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for compatibility
import matplotlib
matplotlib.use('Agg')

class IntegratedNeuralAnalysis:
    """
    Comprehensive neural analysis combining all advanced techniques
    """
    
    def __init__(self, neural_data_file):
        self.neural_data_file = neural_data_file
        self.neural_data = None
        self.pca_results = {}
        self.derivatives_results = {}
        self.behavioral_analysis = {}
        
    def load_data(self):
        """Load and validate neural data"""
        print("Loading neural data for integrated analysis...")
        self.neural_data = pd.read_csv(self.neural_data_file)
        
        neuron_cols = [col for col in self.neural_data.columns if col not in ['Time_minutes']]
        
        print(f"Dataset overview:")
        print(f"  Shape: {self.neural_data.shape}")
        print(f"  Neurons: {len(neuron_cols)}")
        print(f"  Timepoints: {len(self.neural_data)}")
        print(f"  Time range: {self.neural_data['Time_minutes'].min():.2f} - {self.neural_data['Time_minutes'].max():.2f} minutes")
        
        # Data quality check
        missing_data = self.neural_data.isnull().sum().sum()
        if missing_data > 0:
            print(f"  Warning: {missing_data} missing values detected")
        
        return self.neural_data
    
    def perform_integrated_pca(self):
        """Comprehensive PCA with multiple validation techniques"""
        print("\nPerforming Integrated PCA Analysis...")
        
        neuron_cols = [col for col in self.neural_data.columns if col not in ['Time_minutes']]
        X = self.neural_data[neuron_cols].values
        
        # Handle missing values
        if np.any(np.isnan(X)):
            X = pd.DataFrame(X).fillna(pd.DataFrame(X).mean()).values
        
        # Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA with all components first to analyze optimal number
        pca_full = PCA()
        pca_full.fit(X_scaled)
        
        # Determine optimal number of components
        eigenvalues = pca_full.explained_variance_
        n_significant = np.sum(eigenvalues > 1)  # Kaiser criterion
        
        # 80% variance criterion
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_80percent = np.argmax(cumvar >= 0.8) + 1
        
        print(f"  Kaiser criterion suggests {n_significant} components")
        print(f"  80% variance achieved with {n_80percent} components")
        
        n_components = min(max(n_significant, 5), len(neuron_cols))
        
        # Final PCA with selected components
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(X_scaled)
        
        # Store results
        pc_columns = [f'PC{i+1}' for i in range(n_components)]
        pca_scores = pd.DataFrame(
            data=principal_components,
            columns=pc_columns
        )
        pca_scores['Time_minutes'] = self.neural_data['Time_minutes'].values
        
        # Calculate loadings
        loadings = pd.DataFrame(
            data=pca.components_.T,
            columns=pc_columns,
            index=neuron_cols
        )
        
        self.pca_results = {
            'scores': pca_scores,
            'loadings': loadings,
            'model': pca,
            'scaler': scaler,
            'explained_variance': pca.explained_variance_ratio_,
            'n_components': n_components
        }
        
        print(f"  Selected {n_components} components")
        print(f"  Total variance explained: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
        
        return self.pca_results
    
    def analyze_derivatives_and_behavior(self):
        """Comprehensive derivatives analysis with behavioral classification"""
        print("\nAnalyzing neural derivatives and behavioral patterns...")
        
        if not self.pca_results:
            self.perform_integrated_pca()
        
        pca_scores = self.pca_results['scores']
        time = pca_scores['Time_minutes'].values
        
        # Calculate derivatives for top 3 PCs
        derivatives = {}
        for pc in ['PC1', 'PC2', 'PC3']:
            if pc in pca_scores.columns:
                derivatives[f'{pc}_derivative'] = np.gradient(pca_scores[pc].values, time)
        
        derivatives['Time_minutes'] = time
        derivatives_df = pd.DataFrame(derivatives)
        
        # PCA on derivatives
        derivative_cols = [col for col in derivatives_df.columns if col.endswith('_derivative')]
        X_deriv = derivatives_df[derivative_cols].values
        
        scaler_deriv = StandardScaler()
        X_deriv_scaled = scaler_deriv.fit_transform(X_deriv)
        
        pca_deriv = PCA(n_components=3)
        pca_deriv_components = pca_deriv.fit_transform(X_deriv_scaled)
        
        pca_derivatives = pd.DataFrame(
            data=pca_deriv_components,
            columns=['PC1_deriv', 'PC2_deriv', 'PC3_deriv']
        )
        pca_derivatives['Time_minutes'] = time
        
        # Behavioral state identification using multiple methods
        # Method 1: K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        behavioral_states_kmeans = kmeans.fit_predict(pca_deriv_components)
        
        # Sort clusters by activity level
        cluster_activities = []
        for i in range(3):
            mask = behavioral_states_kmeans == i
            activity = np.mean(np.linalg.norm(pca_deriv_components[mask], axis=1))
            cluster_activities.append((i, activity))
        
        sorted_clusters = sorted(cluster_activities, key=lambda x: x[1])
        cluster_mapping = {sorted_clusters[i][0]: i for i in range(3)}
        behavioral_states = np.array([cluster_mapping[state] for state in behavioral_states_kmeans])
        
        # Method 2: Threshold-based classification
        pc1_deriv_magnitude = np.abs(derivatives_df['PC1_derivative'])
        thresholds = np.percentile(pc1_deriv_magnitude, [33, 67])
        
        behavioral_states_threshold = np.zeros(len(pc1_deriv_magnitude))
        behavioral_states_threshold[pc1_deriv_magnitude > thresholds[1]] = 2
        behavioral_states_threshold[(pc1_deriv_magnitude > thresholds[0]) & 
                                  (pc1_deriv_magnitude <= thresholds[1])] = 1
        
        # Create behavioral analysis dataframe
        behavioral_df = pd.DataFrame({
            'Time_minutes': time,
            'behavioral_state_kmeans': behavioral_states,
            'behavioral_state_threshold': behavioral_states_threshold,
            'PC1_derivative_magnitude': pc1_deriv_magnitude,
            'neural_activity_level': np.linalg.norm(pca_deriv_components, axis=1)
        })
        
        # Store results
        self.derivatives_results = {
            'raw_derivatives': derivatives_df,
            'pca_derivatives': pca_derivatives,
            'behavioral_states': behavioral_df,
            'pca_model': pca_deriv,
            'scaler': scaler_deriv
        }
        
        # Behavioral state statistics
        state_labels = ['Low Activity', 'Medium Activity', 'High Activity']
        print(f"\nBehavioral State Distribution (K-means):")
        for i, label in enumerate(state_labels):
            count = np.sum(behavioral_states == i)
            percentage = count / len(behavioral_states) * 100
            print(f"  {label}: {count} timepoints ({percentage:.1f}%)")
        
        return self.derivatives_results
    
    def train_predictive_models(self):
        """Train models to predict behavioral states"""
        print("\nTraining predictive models...")
        
        if not self.derivatives_results:
            self.analyze_derivatives_and_behavior()
        
        # Prepare features
        pca_scores = self.pca_results['scores']
        derivatives_df = self.derivatives_results['raw_derivatives']
        behavioral_df = self.derivatives_results['behavioral_states']
        
        # Comprehensive feature set
        features = []
        feature_names = []
        
        # PCA scores
        for pc in ['PC1', 'PC2', 'PC3']:
            if pc in pca_scores.columns:
                features.append(pca_scores[pc].values)
                feature_names.append(pc)
        
        # Derivatives
        for deriv in ['PC1_derivative', 'PC2_derivative', 'PC3_derivative']:
            if deriv in derivatives_df.columns:
                features.append(derivatives_df[deriv].values)
                feature_names.append(deriv)
        
        # Derivative magnitudes
        for deriv in ['PC1_derivative', 'PC2_derivative']:
            if deriv in derivatives_df.columns:
                magnitude = np.abs(derivatives_df[deriv].values)
                features.append(magnitude)
                feature_names.append(f'{deriv}_magnitude')
        
        # Moving averages
        window = 5
        for pc in ['PC1', 'PC2']:
            if pc in pca_scores.columns:
                ma = pd.Series(pca_scores[pc]).rolling(window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
                features.append(ma.values)
                feature_names.append(f'{pc}_MA')
        
        X = np.column_stack(features)
        y = behavioral_df['behavioral_state_kmeans'].values
        
        # Train models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        trained_models = {}
        
        print(f"\nModel Training Results:")
        print("-" * 40)
        
        for name, model in models.items():
            # Split data
            split_idx = int(0.7 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            trained_models[name] = model
            
            print(f"{name}: {accuracy:.4f}")
        
        self.behavioral_analysis = {
            'models': trained_models,
            'features': X,
            'labels': y,
            'feature_names': feature_names
        }
        
        return trained_models
    
    def create_interactive_3d_visualizations(self):
        """Create interactive 3D visualizations using Plotly"""
        print("\nCreating interactive 3D visualizations...")
        
        if not self.derivatives_results:
            self.analyze_derivatives_and_behavior()
        
        pca_scores = self.pca_results['scores']
        behavioral_df = self.derivatives_results['behavioral_states']
        
        # Create color mapping for behavioral states
        state_colors = {0: 'blue', 1: 'orange', 2: 'red'}
        state_labels = {0: 'Low Activity', 1: 'Medium Activity', 2: 'High Activity'}
        
        colors = [state_colors[state] for state in behavioral_df['behavioral_state_kmeans']]
        labels = [state_labels[state] for state in behavioral_df['behavioral_state_kmeans']]
        
        # 1. 3D PCA trajectory
        fig1 = go.Figure(data=go.Scatter3d(
            x=pca_scores['PC1'],
            y=pca_scores['PC2'],
            z=pca_scores['PC3'],
            mode='markers+lines',
            marker=dict(
                size=5,
                color=colors,
                colorscale='Viridis',
                opacity=0.8
            ),
            line=dict(
                color='gray',
                width=2,
                opacity=0.5
            ),
            text=[f'Time: {t:.2f}min<br>State: {label}' 
                  for t, label in zip(pca_scores['Time_minutes'], labels)],
            hovertemplate='PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<br>%{text}<extra></extra>'
        ))
        
        fig1.update_layout(
            title='Interactive 3D Neural Trajectory (PCA Space)',
            scene=dict(
                xaxis_title='PC1 (Neural Activity)',
                yaxis_title='PC2 (Neural Activity)',
                zaxis_title='PC3 (Neural Activity)',
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='gray', showbackground=True),
                yaxis=dict(gridcolor='gray', showbackground=True),
                zaxis=dict(gridcolor='gray', showbackground=True)
            ),
            width=800,
            height=600,
            font=dict(size=12)
        )
        
        fig1.write_html('Interactive_3D_PCA_Trajectory.html')
        
        # 2. 3D Derivatives space
        pca_derivatives = self.derivatives_results['pca_derivatives']
        
        fig2 = go.Figure(data=go.Scatter3d(
            x=pca_derivatives['PC1_deriv'],
            y=pca_derivatives['PC2_deriv'],
            z=pca_derivatives['PC3_deriv'],
            mode='markers',
            marker=dict(
                size=4,
                color=colors,
                opacity=0.7,
                colorbar=dict(title="Behavioral State")
            ),
            text=[f'Time: {t:.2f}min<br>State: {label}' 
                  for t, label in zip(pca_derivatives['Time_minutes'], labels)],
            hovertemplate='PC1 Deriv: %{x:.4f}<br>PC2 Deriv: %{y:.4f}<br>PC3 Deriv: %{z:.4f}<br>%{text}<extra></extra>'
        ))
        
        fig2.update_layout(
            title='Interactive 3D Neural Derivatives Space',
            scene=dict(
                xaxis_title='PC1 Derivative',
                yaxis_title='PC2 Derivative',
                zaxis_title='PC3 Derivative',
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='gray', showbackground=True),
                yaxis=dict(gridcolor='gray', showbackground=True),
                zaxis=dict(gridcolor='gray', showbackground=True)
            ),
            width=800,
            height=600,
            font=dict(size=12)
        )
        
        fig2.write_html('Interactive_3D_Derivatives_Space.html')
        
        print("✓ Interactive visualizations saved:")
        print("  - Interactive_3D_PCA_Trajectory.html")
        print("  - Interactive_3D_Derivatives_Space.html")
        
        return fig1, fig2
    
    def create_comprehensive_dashboard(self):
        """Create a comprehensive analysis dashboard"""
        print("\nCreating comprehensive analysis dashboard...")
        
        fig = plt.figure(figsize=(24, 20))
        
        # Get data
        pca_scores = self.pca_results['scores']
        derivatives_df = self.derivatives_results['raw_derivatives']
        behavioral_df = self.derivatives_results['behavioral_states']
        explained_var = self.pca_results['explained_variance']
        
        # Color mapping for states
        state_colors = ['blue', 'orange', 'red']
        state_labels = ['Low Activity', 'Medium Activity', 'High Activity']
        
        # 1. PCA Explained Variance
        plt.subplot(4, 6, 1)
        plt.bar(range(1, len(explained_var) + 1), explained_var * 100, color='skyblue', alpha=0.7)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance (%)')
        plt.title('PCA Explained Variance', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 2. PCA Scores Timeline
        plt.subplot(4, 6, 2)
        time = pca_scores['Time_minutes']
        plt.plot(time, pca_scores['PC1'], 'r-', linewidth=2, label='PC1', alpha=0.8)
        plt.plot(time, pca_scores['PC2'], 'g-', linewidth=2, label='PC2', alpha=0.8)
        plt.plot(time, pca_scores['PC3'], 'b-', linewidth=2, label='PC3', alpha=0.8)
        plt.xlabel('Time (minutes)')
        plt.ylabel('PC Score')
        plt.title('PCA Scores Over Time', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 2D PCA Trajectory
        plt.subplot(4, 6, 3)
        for state in range(3):
            mask = behavioral_df['behavioral_state_kmeans'] == state
            if np.sum(mask) > 0:
                plt.scatter(pca_scores.loc[mask, 'PC1'], pca_scores.loc[mask, 'PC2'],
                          c=state_colors[state], label=state_labels[state], alpha=0.6, s=20)
        plt.xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
        plt.title('2D Neural Trajectory', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Derivatives Timeline
        plt.subplot(4, 6, 4)
        plt.plot(time, derivatives_df['PC1_derivative'], 'r-', linewidth=2, label='PC1', alpha=0.8)
        plt.plot(time, derivatives_df['PC2_derivative'], 'g-', linewidth=2, label='PC2', alpha=0.8)
        plt.plot(time, derivatives_df['PC3_derivative'], 'b-', linewidth=2, label='PC3', alpha=0.8)
        plt.xlabel('Time (minutes)')
        plt.ylabel('Derivative')
        plt.title('Neural Derivatives', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Behavioral States Timeline
        plt.subplot(4, 6, 5)
        states = behavioral_df['behavioral_state_kmeans']
        for state in range(3):
            mask = states == state
            if np.sum(mask) > 0:
                plt.scatter(time[mask], states[mask], c=state_colors[state],
                          label=state_labels[state], alpha=0.7, s=15)
        plt.xlabel('Time (minutes)')
        plt.ylabel('Behavioral State')
        plt.title('Behavioral States', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. State Distribution
        plt.subplot(4, 6, 6)
        state_counts = pd.Series(states).value_counts().sort_index()
        bars = plt.bar(range(3), state_counts.values, color=state_colors, alpha=0.7)
        plt.xlabel('State')
        plt.ylabel('Count')
        plt.title('State Distribution', fontweight='bold')
        plt.xticks(range(3), ['Low', 'Med', 'High'])
        
        # Add percentages on bars
        total = sum(state_counts.values)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height/total*100:.1f}%', ha='center', va='bottom')
        
        # 7. 3D Static Plot (matplotlib)
        ax3d = fig.add_subplot(4, 6, 7, projection='3d')
        for state in range(3):
            mask = behavioral_df['behavioral_state_kmeans'] == state
            if np.sum(mask) > 0:
                ax3d.scatter(pca_scores.loc[mask, 'PC1'], 
                           pca_scores.loc[mask, 'PC2'], 
                           pca_scores.loc[mask, 'PC3'],
                           c=state_colors[state], label=state_labels[state], 
                           alpha=0.6, s=20)
        ax3d.set_xlabel('PC1')
        ax3d.set_ylabel('PC2')
        ax3d.set_zlabel('PC3')
        ax3d.set_title('3D Neural Trajectory', fontweight='bold')
        ax3d.legend()
        
        # 8. Derivative Magnitudes
        plt.subplot(4, 6, 8)
        derivative_magnitude = np.sqrt(
            derivatives_df['PC1_derivative']**2 + 
            derivatives_df['PC2_derivative']**2 + 
            derivatives_df['PC3_derivative']**2
        )
        plt.plot(time, derivative_magnitude, 'purple', linewidth=2, alpha=0.8)
        plt.xlabel('Time (minutes)')
        plt.ylabel('Derivative Magnitude')
        plt.title('Neural Activity Velocity', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 9. Top Neuron Loadings Heatmap
        plt.subplot(4, 6, 9)
        loadings = self.pca_results['loadings']
        # Get top neurons for first 3 PCs
        top_neurons = []
        for pc in ['PC1', 'PC2', 'PC3']:
            top_5 = loadings[pc].abs().nlargest(5).index.tolist()
            top_neurons.extend(top_5)
        
        unique_neurons = list(set(top_neurons))[:12]  # Limit for visibility
        if len(unique_neurons) > 0:
            heatmap_data = loadings.loc[unique_neurons, ['PC1', 'PC2', 'PC3']].T
            sns.heatmap(heatmap_data, annot=True, cmap='RdBu_r', center=0,
                       fmt='.2f', cbar_kws={'label': 'Loading'})
            plt.title('Top Neuron Loadings', fontweight='bold')
            plt.ylabel('PC')
            plt.xlabel('Neurons')
            plt.xticks(rotation=45, ha='right')
        
        # 10. Correlation Matrix
        plt.subplot(4, 6, 10)
        corr_data = pca_scores[['PC1', 'PC2', 'PC3']]
        correlation_matrix = corr_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, cbar_kws={'label': 'Correlation'})
        plt.title('PC Correlations', fontweight='bold')
        
        # 11. Derivative Phase Space
        plt.subplot(4, 6, 11)
        for state in range(3):
            mask = behavioral_df['behavioral_state_kmeans'] == state
            if np.sum(mask) > 0:
                plt.scatter(derivatives_df.loc[mask, 'PC1_derivative'],
                          derivatives_df.loc[mask, 'PC2_derivative'],
                          c=state_colors[state], label=state_labels[state], 
                          alpha=0.6, s=20)
        plt.xlabel('PC1 Derivative')
        plt.ylabel('PC2 Derivative')
        plt.title('Derivative Phase Space', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 12. Statistical Summary Box
        plt.subplot(4, 6, 12)
        
        # Calculate key statistics
        stats_text = [
            f"Dataset Overview:",
            f"• Neurons: {len([col for col in self.neural_data.columns if col != 'Time_minutes'])}",
            f"• Timepoints: {len(self.neural_data)}",
            f"• Duration: {time.max() - time.min():.1f} min",
            f"",
            f"PCA Results:",
            f"• Components: {self.pca_results['n_components']}",
            f"• Variance (PC1-3): {sum(explained_var[:3])*100:.1f}%",
            f"",
            f"Behavioral States:",
            f"• Low: {np.sum(states==0)} ({np.sum(states==0)/len(states)*100:.1f}%)",
            f"• Medium: {np.sum(states==1)} ({np.sum(states==1)/len(states)*100:.1f}%)",
            f"• High: {np.sum(states==2)} ({np.sum(states==2)/len(states)*100:.1f}%)",
            f"",
            f"Neural Dynamics:",
            f"• Mean |PC1 deriv|: {np.mean(np.abs(derivatives_df['PC1_derivative'])):.4f}",
            f"• Max velocity: {np.max(derivative_magnitude):.4f}",
        ]
        
        plt.text(0.05, 0.95, '\n'.join(stats_text), transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        plt.axis('off')
        plt.title('Analysis Summary', fontweight='bold')
        
        # Continue with remaining subplots (13-24)
        
        # 13. Autocorrelation
        plt.subplot(4, 6, 13)
        pc1_values = pca_scores['PC1'] - np.mean(pca_scores['PC1'])
        autocorr = np.correlate(pc1_values, pc1_values, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        lags = np.arange(min(50, len(autocorr)))
        plt.plot(lags, autocorr[:len(lags)], 'green', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title('PC1 Autocorrelation', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 14. Moving Averages
        plt.subplot(4, 6, 14)
        window = 20
        pc1_ma = pd.Series(pca_scores['PC1']).rolling(window, center=True).mean()
        plt.plot(time, pca_scores['PC1'], 'lightblue', alpha=0.5, label='Raw')
        plt.plot(time, pc1_ma, 'darkblue', linewidth=2, label=f'MA({window})')
        plt.xlabel('Time (minutes)')
        plt.ylabel('PC1 Score')
        plt.title('PC1 Smoothing', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 15. Derivative Distribution
        plt.subplot(4, 6, 15)
        plt.hist(derivatives_df['PC1_derivative'], bins=30, alpha=0.7, color='red', density=True)
        plt.axvline(np.mean(derivatives_df['PC1_derivative']), color='black', linestyle='--')
        plt.xlabel('PC1 Derivative')
        plt.ylabel('Density')
        plt.title('PC1 Derivative Distribution', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 16. Cumulative Variance
        plt.subplot(4, 6, 16)
        cumulative_var = np.cumsum(self.pca_results['explained_variance'])
        plt.plot(range(1, len(cumulative_var) + 1), cumulative_var * 100, 
                'ro-', linewidth=2, markersize=6)
        plt.axhline(y=80, color='g', linestyle='--', alpha=0.7, label='80%')
        plt.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90%')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Variance (%)')
        plt.title('Cumulative Explained Variance', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 17. State Transition Heatmap
        plt.subplot(4, 6, 17)
        states = behavioral_df['behavioral_state_kmeans'].values
        transition_matrix = np.zeros((3, 3))
        
        for i in range(len(states) - 1):
            transition_matrix[states[i], states[i + 1]] += 1
        
        # Normalize
        row_sums = transition_matrix.sum(axis=1)
        transition_probs = transition_matrix / row_sums[:, np.newaxis]
        
        sns.heatmap(transition_probs, annot=True, cmap='Blues',
                   xticklabels=['Low', 'Med', 'High'],
                   yticklabels=['Low', 'Med', 'High'],
                   cbar_kws={'label': 'Probability'})
        plt.title('State Transitions', fontweight='bold')
        plt.xlabel('To State')
        plt.ylabel('From State')
        
        # 18. Spectral Analysis
        plt.subplot(4, 6, 18)
        from scipy.fft import fft, fftfreq
        
        pc1_fft = np.abs(fft(pca_scores['PC1']))
        freqs = fftfreq(len(pca_scores['PC1']))
        
        pos_mask = freqs > 0
        plt.loglog(freqs[pos_mask], pc1_fft[pos_mask], 'red', linewidth=2)
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.title('PC1 Power Spectrum', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add more sophisticated analyses in remaining subplots...
        
        plt.tight_layout()
        plt.savefig('Integrated_Neural_Analysis_Dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Comprehensive dashboard saved as 'Integrated_Neural_Analysis_Dashboard.png'")
    
    def save_comprehensive_results(self):
        """Save all analysis results"""
        print("\nSaving comprehensive results...")
        
        # PCA results
        if self.pca_results:
            self.pca_results['scores'].to_csv('integrated_pca_scores.csv', index=False)
            self.pca_results['loadings'].to_csv('integrated_pca_loadings.csv')
            print("✓ PCA results saved")
        
        # Derivatives results
        if self.derivatives_results:
            self.derivatives_results['raw_derivatives'].to_csv('integrated_derivatives.csv', index=False)
            self.derivatives_results['behavioral_states'].to_csv('integrated_behavioral_states.csv', index=False)
            print("✓ Derivatives and behavioral results saved")
        
        # Model results
        if self.behavioral_analysis:
            features_df = pd.DataFrame(
                self.behavioral_analysis['features'],
                columns=self.behavioral_analysis['feature_names']
            )
            features_df['behavioral_state'] = self.behavioral_analysis['labels']
            features_df.to_csv('integrated_model_features.csv', index=False)
            print("✓ Model features saved")

def main():
    """
    Run the complete integrated analysis
    """
    print("=" * 90)
    print("INTEGRATED ENHANCED NEURAL ANALYSIS SUITE")
    print("Advanced C. elegans Neural Dynamics Analysis")
    print("Based on Le Cunff et al. 2024 + Enhanced Methodologies")
    print("=" * 90)
    
    # Initialize integrated analyzer
    analyzer = IntegratedNeuralAnalysis('neural_data_dataframe.csv')
    
    try:
        # Complete analysis pipeline
        print("\n🔄 Starting integrated analysis pipeline...")
        
        # 1. Load data
        neural_data = analyzer.load_data()
        
        # 2. PCA analysis
        pca_results = analyzer.perform_integrated_pca()
        
        # 3. Derivatives and behavioral analysis
        derivatives_results = analyzer.analyze_derivatives_and_behavior()
        
        # 4. Predictive modeling
        models = analyzer.train_predictive_models()
        
        # 5. Interactive visualizations
        fig1, fig2 = analyzer.create_interactive_3d_visualizations()
        
        # 6. Comprehensive dashboard
        analyzer.create_comprehensive_dashboard()
        
        # 7. Save results
        analyzer.save_comprehensive_results()
        
        print("\n" + "=" * 90)
        print("✅ INTEGRATED ANALYSIS COMPLETE!")
        print("\nGenerated outputs:")
        print("📊 Static Dashboard: Integrated_Neural_Analysis_Dashboard.png")
        print("🌐 Interactive 3D: Interactive_3D_PCA_Trajectory.html")
        print("🌐 Interactive 3D: Interactive_3D_Derivatives_Space.html")
        print("💾 Data Files: integrated_*.csv")
        print("\n🧠 Advanced neural dynamics analysis with behavioral prediction complete!")
        print("=" * 90)
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()