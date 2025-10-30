"""
Enhanced Neural Derivatives Analysis with Pattern Prediction
Based on Le Cunff et al. 2024 advanced methodology
Incorporates logistic regression for behavioral state prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from scipy import stats, signal
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for compatibility
import matplotlib
matplotlib.use('Agg')

class EnhancedDerivativesAnalysis:
    """
    Enhanced analysis of neural activity derivatives with pattern prediction
    """
    
    def __init__(self, neural_data_file):
        self.neural_data_file = neural_data_file
        self.neural_data = None
        self.derivatives = None
        self.pca_derivatives = None
        self.behavioral_states = None
        self.prediction_model = None
        
    def load_and_process_data(self):
        """Load and preprocess neural data"""
        print("Loading neural data for derivatives analysis...")
        self.neural_data = pd.read_csv(self.neural_data_file)
        
        neuron_cols = [col for col in self.neural_data.columns if col not in ['Time_minutes']]
        print(f"Loaded data: {self.neural_data.shape[0]} timepoints, {len(neuron_cols)} neurons")
        
        return self.neural_data
    
    def calculate_enhanced_derivatives(self, method='gradient'):
        """
        Calculate temporal derivatives with multiple methods
        """
        print(f"Calculating enhanced derivatives using {method} method...")
        
        neuron_cols = [col for col in self.neural_data.columns if col not in ['Time_minutes']]
        derivatives_data = {}
        
        # Add time information
        if 'Time_minutes' in self.neural_data.columns:
            derivatives_data['Time_minutes'] = self.neural_data['Time_minutes'].values
            time = self.neural_data['Time_minutes'].values
        else:
            time = np.arange(len(self.neural_data))
            derivatives_data['Time_minutes'] = time
        
        # Calculate derivatives for each neuron
        for neuron in neuron_cols:
            neural_activity = self.neural_data[neuron].values
            
            if method == 'gradient':
                # Standard gradient method
                derivative = np.gradient(neural_activity, time)
            elif method == 'central_diff':
                # Central difference method
                derivative = np.zeros_like(neural_activity)
                derivative[1:-1] = (neural_activity[2:] - neural_activity[:-2]) / (time[2:] - time[:-2])
                derivative[0] = (neural_activity[1] - neural_activity[0]) / (time[1] - time[0])
                derivative[-1] = (neural_activity[-1] - neural_activity[-2]) / (time[-1] - time[-2])
            elif method == 'savgol':
                # Savitzky-Golay filter derivative
                window_length = min(21, len(neural_activity) // 4 * 2 + 1)  # Ensure odd number
                derivative = signal.savgol_filter(neural_activity, window_length, 3, deriv=1)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            derivatives_data[f'{neuron}_derivative'] = derivative
        
        self.derivatives = pd.DataFrame(derivatives_data)
        
        # Calculate summary statistics
        derivative_cols = [col for col in self.derivatives.columns if col.endswith('_derivative')]
        print(f"\nDerivative Statistics:")
        print(f"Mean absolute derivative: {np.mean(np.abs(self.derivatives[derivative_cols].values)):.6f}")
        print(f"Std of derivatives: {np.std(self.derivatives[derivative_cols].values):.6f}")
        
        return self.derivatives
    
    def perform_pca_on_derivatives(self, n_components=5):
        """
        Perform PCA on derivative data
        """
        print("Performing PCA on derivatives...")
        
        derivative_cols = [col for col in self.derivatives.columns if col.endswith('_derivative')]
        X_derivatives = self.derivatives[derivative_cols].values
        
        # Handle missing values
        if np.any(np.isnan(X_derivatives)):
            X_derivatives = pd.DataFrame(X_derivatives).fillna(0).values
        
        # Standardize derivatives
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_derivatives)
        
        # PCA on derivatives
        pca = PCA(n_components=n_components)
        pca_components = pca.fit_transform(X_scaled)
        
        # Create PCA derivatives dataframe
        pc_columns = [f'PC{i+1}_derivative' for i in range(n_components)]
        self.pca_derivatives = pd.DataFrame(
            data=pca_components,
            columns=pc_columns
        )
        
        # Add time information
        if 'Time_minutes' in self.derivatives.columns:
            self.pca_derivatives['Time_minutes'] = self.derivatives['Time_minutes']
        
        # Print explained variance
        explained_var = pca.explained_variance_ratio_
        print("\nPCA on Derivatives - Explained Variance:")
        for i, var in enumerate(explained_var):
            print(f"PC{i+1}: {var:.4f} ({var*100:.2f}%)")
        
        print(f"Total variance explained: {np.sum(explained_var)*100:.2f}%")
        
        return self.pca_derivatives, pca
    
    def identify_behavioral_states(self, method='kmeans_derivatives'):
        """
        Identify behavioral states using derivatives
        """
        print("Identifying behavioral states...")
        
        if method == 'kmeans_derivatives':
            from sklearn.cluster import KMeans
            
            # Use PCA derivatives for clustering
            if self.pca_derivatives is None:
                self.perform_pca_on_derivatives()
            
            derivative_features = self.pca_derivatives[['PC1_derivative', 'PC2_derivative', 'PC3_derivative']].values
            
            # K-means clustering
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            behavioral_labels = kmeans.fit_predict(derivative_features)
            
            # Map to meaningful labels
            state_mapping = {0: 'low_activity', 1: 'medium_activity', 2: 'high_activity'}
            
            # Sort clusters by mean derivative magnitude
            cluster_means = []
            for i in range(3):
                mask = behavioral_labels == i
                mean_magnitude = np.mean(np.abs(derivative_features[mask]))
                cluster_means.append((i, mean_magnitude))
            
            # Sort by magnitude and reassign labels
            sorted_clusters = sorted(cluster_means, key=lambda x: x[1])
            label_mapping = {sorted_clusters[i][0]: i for i in range(3)}
            behavioral_labels = np.array([label_mapping[label] for label in behavioral_labels])
            
        elif method == 'threshold_based':
            # Threshold-based classification
            pc1_derivative = self.pca_derivatives['PC1_derivative']
            
            # Calculate thresholds
            low_threshold = np.percentile(np.abs(pc1_derivative), 33)
            high_threshold = np.percentile(np.abs(pc1_derivative), 67)
            
            behavioral_labels = np.zeros(len(pc1_derivative))
            behavioral_labels[np.abs(pc1_derivative) > high_threshold] = 2  # high
            behavioral_labels[(np.abs(pc1_derivative) > low_threshold) & 
                            (np.abs(pc1_derivative) <= high_threshold)] = 1  # medium
            # low remains 0
            
            state_mapping = {0: 'low_activity', 1: 'medium_activity', 2: 'high_activity'}
        
        # Create behavioral states dataframe
        self.behavioral_states = pd.DataFrame({
            'Time_minutes': self.pca_derivatives['Time_minutes'],
            'behavioral_state': behavioral_labels,
            'state_label': [state_mapping[label] for label in behavioral_labels]
        })
        
        # Print state distribution
        state_counts = pd.Series(behavioral_labels).value_counts().sort_index()
        print(f"\nBehavioral State Distribution:")
        for state, count in state_counts.items():
            label = state_mapping[state]
            percentage = count / len(behavioral_labels) * 100
            print(f"{label}: {count} timepoints ({percentage:.1f}%)")
        
        return self.behavioral_states
    
    def train_behavioral_predictor(self):
        """
        Train a model to predict behavioral states from neural derivatives
        """
        print("Training behavioral state predictor...")
        
        if self.behavioral_states is None:
            self.identify_behavioral_states()
        
        # Prepare features and labels
        derivative_features = self.pca_derivatives[['PC1_derivative', 'PC2_derivative', 'PC3_derivative']].values
        behavioral_labels = self.behavioral_states['behavioral_state'].values
        
        # Add additional features
        # Moving averages
        window = 5
        pc1_ma = pd.Series(self.pca_derivatives['PC1_derivative']).rolling(window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        pc2_ma = pd.Series(self.pca_derivatives['PC2_derivative']).rolling(window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        
        # Derivative magnitudes
        pc1_magnitude = np.abs(self.pca_derivatives['PC1_derivative'])
        pc2_magnitude = np.abs(self.pca_derivatives['PC2_derivative'])
        
        # Combine features
        enhanced_features = np.column_stack([
            derivative_features,
            pc1_ma.values,
            pc2_ma.values,
            pc1_magnitude,
            pc2_magnitude
        ])
        
        feature_names = ['PC1_deriv', 'PC2_deriv', 'PC3_deriv', 'PC1_MA', 'PC2_MA', 'PC1_mag', 'PC2_mag']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            enhanced_features, behavioral_labels, test_size=0.3, random_state=42, stratify=behavioral_labels
        )
        
        # Train multiple models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = 0
        
        print(f"\nModel Performance Comparison:")
        print("-" * 50)
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, enhanced_features, behavioral_labels, cv=5, scoring='accuracy')
            
            print(f"{name}:")
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                self.prediction_model = model
        
        # Feature importance for best model
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            print(f"\nFeature Importances (Random Forest):")
            for name, importance in zip(feature_names, importances):
                print(f"  {name}: {importance:.4f}")
        elif hasattr(best_model, 'coef_'):
            coef_mean = np.mean(np.abs(best_model.coef_), axis=0)
            print(f"\nFeature Coefficients (Logistic Regression):")
            for name, coef in zip(feature_names, coef_mean):
                print(f"  {name}: {coef:.4f}")
        
        return best_model, enhanced_features, behavioral_labels
    
    def analyze_behavioral_transitions(self):
        """
        Analyze transitions between behavioral states
        """
        print("Analyzing behavioral state transitions...")
        
        if self.behavioral_states is None:
            self.identify_behavioral_states()
        
        states = self.behavioral_states['behavioral_state'].values
        state_labels = ['low_activity', 'medium_activity', 'high_activity']
        
        # Create transition matrix
        transition_matrix = np.zeros((3, 3))
        
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            transition_matrix[current_state, next_state] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_probs = transition_matrix / row_sums[:, np.newaxis]
        
        print(f"\nBehavioral State Transition Probabilities:")
        print("-" * 50)
        
        transition_df = pd.DataFrame(
            transition_probs,
            index=[f"From_{label}" for label in state_labels],
            columns=[f"To_{label}" for label in state_labels]
        )
        
        print(transition_df.round(3))
        
        # Calculate state persistence
        persistence = np.diag(transition_probs)
        print(f"\nState Persistence:")
        for i, (label, pers) in enumerate(zip(state_labels, persistence)):
            print(f"{label}: {pers:.3f}")
        
        return transition_df, transition_matrix
    
    def create_comprehensive_visualizations(self):
        """
        Create comprehensive visualizations
        """
        print("Creating enhanced derivative visualizations...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Derivatives time series
        plt.subplot(3, 4, 1)
        if 'Time_minutes' in self.pca_derivatives.columns:
            time = self.pca_derivatives['Time_minutes']
            plt.plot(time, self.pca_derivatives['PC1_derivative'], 'r-', linewidth=2, label='PC1', alpha=0.8)
            plt.plot(time, self.pca_derivatives['PC2_derivative'], 'g-', linewidth=2, label='PC2', alpha=0.8)
            plt.plot(time, self.pca_derivatives['PC3_derivative'], 'b-', linewidth=2, label='PC3', alpha=0.8)
            plt.xlabel('Time (minutes)')
            plt.ylabel('Derivative Value')
            plt.title('PCA Derivatives Over Time', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 2. Derivative magnitude distribution
        plt.subplot(3, 4, 2)
        pc1_deriv = self.pca_derivatives['PC1_derivative']
        plt.hist(pc1_deriv, bins=50, alpha=0.7, color='red', density=True, label='PC1')
        plt.axvline(np.mean(pc1_deriv), color='red', linestyle='--', linewidth=2)
        plt.xlabel('Derivative Value')
        plt.ylabel('Density')
        plt.title('PC1 Derivative Distribution', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 3D trajectory of derivatives
        ax3d = fig.add_subplot(3, 4, 3, projection='3d')
        if self.behavioral_states is not None:
            colors = ['blue', 'orange', 'red']
            state_labels = ['Low Activity', 'Medium Activity', 'High Activity']
            for state in range(3):
                mask = self.behavioral_states['behavioral_state'] == state
                if np.sum(mask) > 0:
                    ax3d.scatter(
                        self.pca_derivatives.loc[mask, 'PC1_derivative'],
                        self.pca_derivatives.loc[mask, 'PC2_derivative'],
                        self.pca_derivatives.loc[mask, 'PC3_derivative'],
                        c=colors[state], label=state_labels[state], alpha=0.6, s=20
                    )
        else:
            ax3d.scatter(
                self.pca_derivatives['PC1_derivative'],
                self.pca_derivatives['PC2_derivative'],
                self.pca_derivatives['PC3_derivative'],
                alpha=0.6, s=20
            )
        
        ax3d.set_xlabel('PC1 Derivative')
        ax3d.set_ylabel('PC2 Derivative')
        ax3d.set_zlabel('PC3 Derivative')
        ax3d.set_title('3D Derivative Space', fontweight='bold')
        if self.behavioral_states is not None:
            ax3d.legend()
        
        # 4. Behavioral states over time
        plt.subplot(3, 4, 4)
        if self.behavioral_states is not None:
            time = self.behavioral_states['Time_minutes']
            states = self.behavioral_states['behavioral_state']
            
            # Create colored timeline
            colors = ['blue', 'orange', 'red']
            for state in range(3):
                mask = states == state
                if np.sum(mask) > 0:
                    plt.scatter(time[mask], states[mask], c=colors[state], 
                              label=['Low', 'Medium', 'High'][state], alpha=0.7, s=10)
            
            plt.xlabel('Time (minutes)')
            plt.ylabel('Behavioral State')
            plt.title('Behavioral States Over Time', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 5. Derivative correlations
        plt.subplot(3, 4, 5)
        deriv_corr_data = self.pca_derivatives[['PC1_derivative', 'PC2_derivative', 'PC3_derivative']]
        correlation_matrix = deriv_corr_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, cbar_kws={'label': 'Correlation'})
        plt.title('Derivative Correlations', fontweight='bold')
        
        # 6. Moving average of derivative magnitudes
        plt.subplot(3, 4, 6)
        if 'Time_minutes' in self.pca_derivatives.columns:
            time = self.pca_derivatives['Time_minutes']
            pc1_magnitude = np.abs(self.pca_derivatives['PC1_derivative'])
            
            # Calculate moving average
            window = 20
            ma_magnitude = pd.Series(pc1_magnitude).rolling(window, center=True).mean()
            
            plt.plot(time, pc1_magnitude, 'lightblue', alpha=0.5, label='Raw Magnitude')
            plt.plot(time, ma_magnitude, 'darkblue', linewidth=2, label=f'MA ({window})')
            plt.xlabel('Time (minutes)')
            plt.ylabel('|PC1 Derivative|')
            plt.title('Derivative Magnitude Smoothing', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 7. State transition heatmap
        plt.subplot(3, 4, 7)
        if self.behavioral_states is not None:
            _, transition_matrix = self.analyze_behavioral_transitions()
            
            # Normalize transition matrix
            row_sums = transition_matrix.sum(axis=1)
            transition_probs = transition_matrix / row_sums[:, np.newaxis]
            
            sns.heatmap(transition_probs, annot=True, cmap='Blues', 
                       xticklabels=['Low', 'Med', 'High'],
                       yticklabels=['Low', 'Med', 'High'],
                       cbar_kws={'label': 'Transition Probability'})
            plt.title('State Transition Matrix', fontweight='bold')
            plt.xlabel('To State')
            plt.ylabel('From State')
        
        # 8. Derivative phase space (PC1 vs PC2)
        plt.subplot(3, 4, 8)
        if self.behavioral_states is not None:
            colors = ['blue', 'orange', 'red']
            state_labels = ['Low Activity', 'Medium Activity', 'High Activity']
            for state in range(3):
                mask = self.behavioral_states['behavioral_state'] == state
                if np.sum(mask) > 0:
                    plt.scatter(
                        self.pca_derivatives.loc[mask, 'PC1_derivative'],
                        self.pca_derivatives.loc[mask, 'PC2_derivative'],
                        c=colors[state], label=state_labels[state], alpha=0.6, s=20
                    )
            plt.legend()
        else:
            plt.scatter(self.pca_derivatives['PC1_derivative'], 
                       self.pca_derivatives['PC2_derivative'], alpha=0.6, s=20)
        
        plt.xlabel('PC1 Derivative')
        plt.ylabel('PC2 Derivative')
        plt.title('Derivative Phase Space', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 9. Derivative velocity analysis
        plt.subplot(3, 4, 9)
        velocity = np.sqrt(
            self.pca_derivatives['PC1_derivative']**2 + 
            self.pca_derivatives['PC2_derivative']**2 + 
            self.pca_derivatives['PC3_derivative']**2
        )
        
        if 'Time_minutes' in self.pca_derivatives.columns:
            time = self.pca_derivatives['Time_minutes']
            plt.plot(time, velocity, 'purple', linewidth=2, alpha=0.8)
            plt.xlabel('Time (minutes)')
            plt.ylabel('Derivative Velocity')
            plt.title('Neural Activity Velocity', fontweight='bold')
            plt.grid(True, alpha=0.3)
        
        # 10. Autocorrelation of PC1 derivative
        plt.subplot(3, 4, 10)
        pc1_deriv = self.pca_derivatives['PC1_derivative']
        autocorr = np.correlate(pc1_deriv - np.mean(pc1_deriv), 
                               pc1_deriv - np.mean(pc1_deriv), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        lags = np.arange(len(autocorr))
        plt.plot(lags[:min(100, len(lags))], autocorr[:min(100, len(lags))], 
                'green', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title('PC1 Derivative Autocorrelation', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 11. Derivative spectral analysis
        plt.subplot(3, 4, 11)
        from scipy.fft import fft, fftfreq
        
        pc1_deriv = self.pca_derivatives['PC1_derivative']
        fft_vals = np.abs(fft(pc1_deriv))
        freqs = fftfreq(len(pc1_deriv))
        
        # Plot positive frequencies only
        pos_mask = freqs > 0
        plt.loglog(freqs[pos_mask], fft_vals[pos_mask], 'red', linewidth=2)
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.title('Derivative Power Spectrum', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 12. Summary statistics
        plt.subplot(3, 4, 12)
        
        # Calculate summary statistics
        stats_data = {
            'PC1 Mean': [np.mean(self.pca_derivatives['PC1_derivative'])],
            'PC1 Std': [np.std(self.pca_derivatives['PC1_derivative'])],
            'PC1 Skew': [stats.skew(self.pca_derivatives['PC1_derivative'])],
            'PC1 Kurt': [stats.kurtosis(self.pca_derivatives['PC1_derivative'])],
            'PC2 Mean': [np.mean(self.pca_derivatives['PC2_derivative'])],
            'PC2 Std': [np.std(self.pca_derivatives['PC2_derivative'])],
        }
        
        stats_df = pd.DataFrame(stats_data).T
        stats_df.columns = ['Value']
        
        # Create text plot
        plt.text(0.1, 0.9, 'Summary Statistics:', fontsize=14, fontweight='bold',
                transform=plt.gca().transAxes)
        
        y_pos = 0.8
        for stat, value in stats_df.iterrows():
            plt.text(0.1, y_pos, f'{stat}: {value.iloc[0]:.4f}', 
                    fontsize=10, transform=plt.gca().transAxes)
            y_pos -= 0.1
        
        plt.axis('off')
        plt.title('Statistical Summary', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('Enhanced_Derivatives_Analysis_Complete.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Enhanced derivatives visualization saved as 'Enhanced_Derivatives_Analysis_Complete.png'")
    
    def save_results(self, prefix='enhanced_derivatives'):
        """
        Save all analysis results
        """
        print(f"\nSaving enhanced results with prefix '{prefix}'...")
        
        # Save derivatives
        if self.derivatives is not None:
            self.derivatives.to_csv(f'{prefix}_raw_derivatives.csv', index=False)
            print(f"✓ Raw derivatives saved as '{prefix}_raw_derivatives.csv'")
        
        # Save PCA derivatives
        if self.pca_derivatives is not None:
            self.pca_derivatives.to_csv(f'{prefix}_pca_derivatives.csv', index=False)
            print(f"✓ PCA derivatives saved as '{prefix}_pca_derivatives.csv'")
        
        # Save behavioral states
        if self.behavioral_states is not None:
            self.behavioral_states.to_csv(f'{prefix}_behavioral_states.csv', index=False)
            print(f"✓ Behavioral states saved as '{prefix}_behavioral_states.csv'")

def main():
    """
    Main enhanced derivatives analysis pipeline
    """
    print("=" * 80)
    print("ENHANCED NEURAL DERIVATIVES ANALYSIS WITH PATTERN PREDICTION")
    print("Based on Le Cunff et al. 2024 advanced methodology")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = EnhancedDerivativesAnalysis('neural_data_dataframe.csv')
    
    try:
        # Load and process data
        neural_data = analyzer.load_and_process_data()
        
        # Calculate enhanced derivatives
        derivatives = analyzer.calculate_enhanced_derivatives(method='gradient')
        
        # Perform PCA on derivatives
        pca_derivatives, pca_model = analyzer.perform_pca_on_derivatives(n_components=5)
        
        # Identify behavioral states
        behavioral_states = analyzer.identify_behavioral_states(method='kmeans_derivatives')
        
        # Train behavioral predictor
        predictor_model, features, labels = analyzer.train_behavioral_predictor()
        
        # Analyze behavioral transitions
        transition_df, transition_matrix = analyzer.analyze_behavioral_transitions()
        
        # Create comprehensive visualizations
        analyzer.create_comprehensive_visualizations()
        
        # Save all results
        analyzer.save_results(prefix='enhanced_celegans_derivatives')
        
        print("\n" + "=" * 80)
        print("ENHANCED DERIVATIVES ANALYSIS COMPLETE!")
        print("Advanced pattern prediction and behavioral analysis completed")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()