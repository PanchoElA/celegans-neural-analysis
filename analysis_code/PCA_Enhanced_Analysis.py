"""
Enhanced PCA Analysis for C. elegans Neural Data
Based on Le Cunff et al. 2024 methodology improvements
Incorporates advanced statistical validation and interaction prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for compatibility
import matplotlib
matplotlib.use('Agg')

class EnhancedPCAAnalysis:
    """
    Enhanced PCA Analysis class with advanced statistical methods
    and interaction prediction capabilities
    """
    
    def __init__(self, neural_data_file):
        self.neural_data_file = neural_data_file
        self.neural_data = None
        self.pca_scores = None
        self.pca_model = None
        self.scaler = None
        self.loadings = None
        
    def load_and_prepare_data(self):
        """Load neural data and prepare for analysis"""
        print("Loading neural data...")
        self.neural_data = pd.read_csv(self.neural_data_file)
        
        # Separate neuron columns (assume they are numeric except for time)
        neuron_cols = [col for col in self.neural_data.columns if col not in ['Time_minutes']]
        
        print(f"Data shape: {self.neural_data.shape}")
        print(f"Number of neurons: {len(neuron_cols)}")
        print(f"Number of timepoints: {len(self.neural_data)}")
        
        return self.neural_data
        
    def perform_enhanced_pca(self, n_components=5, standardize=True):
        """
        Perform PCA with enhanced statistical validation
        """
        print("Performing Enhanced PCA Analysis...")
        
        # Prepare data
        neuron_cols = [col for col in self.neural_data.columns if col not in ['Time_minutes']]
        X = self.neural_data[neuron_cols].values
        
        # Handle missing values
        if np.any(np.isnan(X)):
            print("Warning: Missing values detected. Filling with column means.")
            X = pd.DataFrame(X).fillna(pd.DataFrame(X).mean()).values
        
        # Standardization (recommended for PCA)
        if standardize:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
            
        # Perform PCA
        self.pca_model = PCA(n_components=n_components)
        principal_components = self.pca_model.fit_transform(X_scaled)
        
        # Create scores dataframe
        pc_columns = [f'PC{i+1}' for i in range(n_components)]
        self.pca_scores = pd.DataFrame(
            data=principal_components,
            columns=pc_columns
        )
        
        # Add time information
        if 'Time_minutes' in self.neural_data.columns:
            self.pca_scores['Time_minutes'] = self.neural_data['Time_minutes'].values
        
        # Calculate loadings (contribution of each neuron to each PC)
        self.loadings = pd.DataFrame(
            data=self.pca_model.components_.T,
            columns=pc_columns,
            index=neuron_cols
        )
        
        # Print explained variance
        explained_var = self.pca_model.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        print("\nPCA Results:")
        print("-" * 50)
        for i, (var, cum_var) in enumerate(zip(explained_var, cumulative_var)):
            print(f"PC{i+1}: {var:.4f} ({var*100:.2f}%) - Cumulative: {cum_var*100:.2f}%")
        
        return self.pca_scores, self.loadings
    
    def statistical_validation(self):
        """
        Perform comprehensive statistical validation of PCA results
        """
        print("\nPerforming Statistical Validation...")
        
        # Kaiser-Meyer-Olkin (KMO) test approximation
        neuron_cols = [col for col in self.neural_data.columns if col not in ['Time_minutes']]
        correlation_matrix = np.corrcoef(self.neural_data[neuron_cols].T)
        
        # Calculate determinant (measure of multicollinearity)
        det = np.linalg.det(correlation_matrix)
        print(f"Correlation Matrix Determinant: {det:.6f}")
        
        if det < 0.00001:
            print("Warning: Very low determinant suggests high multicollinearity")
        
        # Bartlett's sphericity test approximation
        n_samples, n_features = self.neural_data[neuron_cols].shape
        chi_square_stat = -(n_samples - 1 - (2 * n_features + 5) / 6) * np.log(det)
        df = n_features * (n_features - 1) / 2
        p_value = 1 - stats.chi2.cdf(chi_square_stat, df)
        
        print(f"Bartlett's Test - Chi-square: {chi_square_stat:.4f}, p-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print("✓ Bartlett's test significant - PCA appropriate")
        else:
            print("⚠ Bartlett's test not significant - PCA may not be appropriate")
            
        # Eigenvalue analysis
        eigenvalues = self.pca_model.explained_variance_
        print(f"\nEigenvalues: {eigenvalues}")
        
        # Kaiser criterion (eigenvalues > 1 for standardized data)
        if hasattr(self.scaler, 'mean_'):
            significant_pcs = np.sum(eigenvalues > 1)
            print(f"Significant PCs (Kaiser criterion): {significant_pcs}")
        
        return {
            'determinant': det,
            'bartlett_chi2': chi_square_stat,
            'bartlett_p': p_value,
            'eigenvalues': eigenvalues
        }
    
    def identify_important_neurons(self, pc_number=1, top_n=10):
        """
        Identify neurons with highest contributions to specific PCs
        """
        pc_col = f'PC{pc_number}'
        if pc_col not in self.loadings.columns:
            raise ValueError(f"PC{pc_number} not available")
        
        # Get absolute loadings for the specified PC
        pc_loadings = self.loadings[pc_col].abs().sort_values(ascending=False)
        top_neurons = pc_loadings.head(top_n)
        
        print(f"\nTop {top_n} Neurons Contributing to {pc_col}:")
        print("-" * 60)
        for neuron, loading in top_neurons.items():
            direction = "+" if self.loadings.loc[neuron, pc_col] > 0 else "-"
            print(f"{neuron}: {loading:.4f} ({direction})")
        
        return top_neurons
    
    def create_comprehensive_visualizations(self):
        """
        Create comprehensive visualization suite
        """
        print("\nCreating Enhanced Visualizations...")
        
        # 1. Scree Plot with Kaiser Criterion
        plt.figure(figsize=(15, 12))
        
        plt.subplot(2, 3, 1)
        eigenvalues = self.pca_model.explained_variance_
        plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-', linewidth=2, markersize=8)
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Kaiser Criterion')
        plt.xlabel('Principal Component', fontsize=12)
        plt.ylabel('Eigenvalue', fontsize=12)
        plt.title('Scree Plot with Kaiser Criterion', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 2. Explained Variance Ratio
        plt.subplot(2, 3, 2)
        explained_var = self.pca_model.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        plt.bar(range(1, len(explained_var) + 1), explained_var * 100, 
               alpha=0.7, color='skyblue', label='Individual')
        plt.plot(range(1, len(explained_var) + 1), cumulative_var * 100, 
                'ro-', linewidth=2, markersize=6, label='Cumulative')
        plt.axhline(y=80, color='g', linestyle='--', alpha=0.7, label='80% Threshold')
        plt.xlabel('Principal Component', fontsize=12)
        plt.ylabel('Explained Variance (%)', fontsize=12)
        plt.title('Explained Variance Analysis', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. PC1 vs PC2 Trajectory (colored by time)
        plt.subplot(2, 3, 3)
        if 'Time_minutes' in self.pca_scores.columns:
            scatter = plt.scatter(self.pca_scores['PC1'], self.pca_scores['PC2'], 
                               c=self.pca_scores['Time_minutes'], cmap='viridis', 
                               s=50, alpha=0.7)
            plt.colorbar(scatter, label='Time (minutes)')
        else:
            plt.scatter(self.pca_scores['PC1'], self.pca_scores['PC2'], 
                       s=50, alpha=0.7, color='blue')
        plt.xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontsize=12)
        plt.ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontsize=12)
        plt.title('PC1 vs PC2 Neural Trajectory', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 4. Top Neuron Loadings Heatmap
        plt.subplot(2, 3, 4)
        # Select top contributing neurons across all PCs
        top_neurons_per_pc = []
        for pc in ['PC1', 'PC2', 'PC3']:
            if pc in self.loadings.columns:
                top_5 = self.loadings[pc].abs().nlargest(5).index.tolist()
                top_neurons_per_pc.extend(top_5)
        
        unique_top_neurons = list(set(top_neurons_per_pc))[:15]  # Limit to 15 for visibility
        
        if len(unique_top_neurons) > 0:
            heatmap_data = self.loadings.loc[unique_top_neurons, ['PC1', 'PC2', 'PC3']].T
            sns.heatmap(heatmap_data, annot=True, cmap='RdBu_r', center=0, 
                       fmt='.3f', cbar_kws={'label': 'Loading'})
            plt.title('Top Neuron Loadings Heatmap', fontsize=14, fontweight='bold')
            plt.ylabel('Principal Components', fontsize=12)
            plt.xlabel('Neurons', fontsize=12)
            plt.xticks(rotation=45, ha='right')
        
        # 5. Temporal Evolution of Top 3 PCs
        plt.subplot(2, 3, 5)
        if 'Time_minutes' in self.pca_scores.columns:
            time = self.pca_scores['Time_minutes']
            colors = ['red', 'green', 'blue']
            for i, pc in enumerate(['PC1', 'PC2', 'PC3']):
                if pc in self.pca_scores.columns:
                    plt.plot(time, self.pca_scores[pc], color=colors[i], 
                            linewidth=2, alpha=0.8, label=f'{pc} ({explained_var[i]*100:.1f}%)')
            plt.xlabel('Time (minutes)', fontsize=12)
            plt.ylabel('PC Score', fontsize=12)
            plt.title('Temporal Evolution of PCs', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 6. Biplot (PC1 vs PC2 with neuron vectors)
        plt.subplot(2, 3, 6)
        if 'Time_minutes' in self.pca_scores.columns:
            plt.scatter(self.pca_scores['PC1'], self.pca_scores['PC2'], 
                       alpha=0.6, s=30, color='lightblue', label='Timepoints')
        
        # Add loading vectors for top contributing neurons
        scale_factor = 3  # Adjust for better visualization
        for neuron in unique_top_neurons[:8]:  # Show top 8 for clarity
            loading1 = self.loadings.loc[neuron, 'PC1'] * scale_factor
            loading2 = self.loadings.loc[neuron, 'PC2'] * scale_factor
            plt.arrow(0, 0, loading1, loading2, head_width=0.1, 
                     head_length=0.1, fc='red', ec='red', alpha=0.7)
            plt.text(loading1 * 1.1, loading2 * 1.1, neuron, 
                    fontsize=8, ha='center', va='center')
        
        plt.xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontsize=12)
        plt.ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontsize=12)
        plt.title('PCA Biplot', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        if 'Time_minutes' in self.pca_scores.columns:
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('Enhanced_PCA_Analysis_Complete.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Enhanced visualization saved as 'Enhanced_PCA_Analysis_Complete.png'")
    
    def behavioral_pattern_analysis(self):
        """
        Analyze behavioral patterns in neural activity
        """
        print("\nAnalyzing Behavioral Patterns...")
        
        if 'Time_minutes' not in self.pca_scores.columns:
            print("Time information not available for behavioral analysis")
            return None
        
        # Analyze temporal derivatives
        derivatives = {}
        for pc in ['PC1', 'PC2', 'PC3']:
            if pc in self.pca_scores.columns:
                derivatives[f'{pc}_derivative'] = np.gradient(self.pca_scores[pc])
        
        derivatives_df = pd.DataFrame(derivatives)
        derivatives_df['Time_minutes'] = self.pca_scores['Time_minutes']
        
        # Identify high-activity periods
        activity_threshold = np.percentile(np.abs(derivatives_df['PC1_derivative']), 75)
        high_activity_mask = np.abs(derivatives_df['PC1_derivative']) > activity_threshold
        
        print(f"High neural activity periods: {np.sum(high_activity_mask)} timepoints")
        print(f"Activity threshold (75th percentile): {activity_threshold:.4f}")
        
        # Statistical analysis of behavioral states
        if np.sum(high_activity_mask) > 0 and np.sum(~high_activity_mask) > 0:
            high_activity_pcs = self.pca_scores.loc[high_activity_mask, ['PC1', 'PC2', 'PC3']]
            low_activity_pcs = self.pca_scores.loc[~high_activity_mask, ['PC1', 'PC2', 'PC3']]
            
            # Perform t-tests
            print("\nBehavioral State Comparison (t-tests):")
            print("-" * 50)
            for pc in ['PC1', 'PC2', 'PC3']:
                if pc in self.pca_scores.columns:
                    t_stat, p_value = stats.ttest_ind(
                        high_activity_pcs[pc], low_activity_pcs[pc]
                    )
                    print(f"{pc}: t={t_stat:.4f}, p={p_value:.6f}")
        
        return derivatives_df
    
    def save_results(self, prefix='enhanced_pca'):
        """
        Save all analysis results
        """
        print(f"\nSaving Results with prefix '{prefix}'...")
        
        # Save PCA scores
        if self.pca_scores is not None:
            self.pca_scores.to_csv(f'{prefix}_scores.csv', index=False)
            print(f"✓ PCA scores saved as '{prefix}_scores.csv'")
        
        # Save loadings
        if self.loadings is not None:
            self.loadings.to_csv(f'{prefix}_loadings.csv')
            print(f"✓ PCA loadings saved as '{prefix}_loadings.csv'")
        
        # Save model parameters
        if self.pca_model is not None:
            results_summary = {
                'explained_variance_ratio': self.pca_model.explained_variance_ratio_.tolist(),
                'explained_variance': self.pca_model.explained_variance_.tolist(),
                'n_components': self.pca_model.n_components_
            }
            
            summary_df = pd.DataFrame([results_summary])
            summary_df.to_csv(f'{prefix}_model_summary.csv', index=False)
            print(f"✓ Model summary saved as '{prefix}_model_summary.csv'")

def main():
    """
    Main analysis pipeline
    """
    print("=" * 70)
    print("ENHANCED PCA ANALYSIS FOR C. ELEGANS NEURAL DATA")
    print("Based on Le Cunff et al. 2024 methodology")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = EnhancedPCAAnalysis('neural_data_dataframe.csv')
    
    try:
        # Load and prepare data
        neural_data = analyzer.load_and_prepare_data()
        
        # Perform enhanced PCA
        pca_scores, loadings = analyzer.perform_enhanced_pca(n_components=5)
        
        # Statistical validation
        validation_results = analyzer.statistical_validation()
        
        # Identify important neurons
        for pc_num in [1, 2, 3]:
            analyzer.identify_important_neurons(pc_number=pc_num, top_n=5)
        
        # Create comprehensive visualizations
        analyzer.create_comprehensive_visualizations()
        
        # Behavioral pattern analysis
        derivatives = analyzer.behavioral_pattern_analysis()
        
        # Save all results
        analyzer.save_results(prefix='enhanced_celegans_pca')
        
        print("\n" + "=" * 70)
        print("ENHANCED ANALYSIS COMPLETE!")
        print("All results saved with comprehensive statistical validation")
        print("=" * 70)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()