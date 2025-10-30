"""
Enhanced Neural Analysis Suite - Master Executor
Based on Le Cunff et al. 2024 advanced methodologies

This script allows you to choose which enhanced analysis to run:
1. Enhanced PCA Analysis
2. Enhanced Derivatives Analysis  
3. Integrated Analysis (All features)
4. Run All Analyses
"""

import os
import sys
import subprocess
import time

def print_banner():
    """Print welcome banner"""
    print("=" * 80)
    print("🧠 ENHANCED C. ELEGANS NEURAL ANALYSIS SUITE")
    print("Advanced Neural Dynamics Analysis with Pattern Prediction")
    print("Based on Le Cunff et al. 2024 + Enhanced Methodologies")
    print("=" * 80)

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'scikit-learn', 
        'seaborn', 'scipy', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("⚠️  Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_data_file():
    """Check if neural data file exists"""
    data_file = 'neural_data_dataframe.csv'
    if not os.path.exists(data_file):
        print(f"❌ Data file '{data_file}' not found!")
        print("Please ensure the neural data CSV file is in the current directory.")
        return False
    
    print(f"✅ Data file '{data_file}' found")
    return True

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n🔄 Starting {description}...")
    print("-" * 60)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True, 
                              check=True)
        print(f"✅ {description} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {description}:")
        print(f"   Return code: {e.returncode}")
        return False
        
    except FileNotFoundError:
        print(f"❌ Script '{script_name}' not found!")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error in {description}: {str(e)}")
        return False

def show_menu():
    """Display analysis options menu"""
    print("\n📋 Available Analyses:")
    print("-" * 40)
    print("1. 🔍 Enhanced PCA Analysis")
    print("   • Advanced statistical validation")
    print("   • Comprehensive neuron importance ranking")
    print("   • Enhanced visualizations with biplot")
    
    print("\n2. 📈 Enhanced Derivatives Analysis") 
    print("   • Multiple derivative calculation methods")
    print("   • Behavioral state prediction with ML")
    print("   • Transition analysis and spectral features")
    
    print("\n3. 🚀 Integrated Analysis Suite")
    print("   • Complete analysis pipeline")
    print("   • Interactive 3D visualizations")
    print("   • Comprehensive dashboard")
    print("   • Predictive modeling")
    
    print("\n4. 🔄 Run All Analyses")
    print("   • Execute all enhanced analyses sequentially")
    print("   • Complete neural dynamics characterization")
    
    print("\n5. ℹ️  Show Analysis Details")
    print("6. 🚪 Exit")
    print("-" * 40)

def show_analysis_details():
    """Show detailed information about each analysis"""
    print("\n📖 DETAILED ANALYSIS INFORMATION")
    print("=" * 60)
    
    print("\n1. ENHANCED PCA ANALYSIS:")
    print("   🔬 Features:")
    print("   • Kaiser-Meyer-Olkin (KMO) test approximation")
    print("   • Bartlett's sphericity test")
    print("   • Automatic optimal component selection")
    print("   • Advanced loading analysis with heatmaps")
    print("   • Biplot with neuron contribution vectors")
    print("   • Temporal evolution analysis")
    print("   • Statistical validation metrics")
    
    print("\n   📊 Outputs:")
    print("   • Enhanced_PCA_Analysis_Complete.png")
    print("   • enhanced_celegans_pca_scores.csv")
    print("   • enhanced_celegans_pca_loadings.csv")
    print("   • enhanced_celegans_pca_model_summary.csv")
    
    print("\n2. ENHANCED DERIVATIVES ANALYSIS:")
    print("   🔬 Features:")
    print("   • Multiple derivative calculation methods (gradient, central diff, Savgol)")
    print("   • PCA on derivatives for dimensionality reduction")
    print("   • K-means clustering for behavioral state identification")
    print("   • Machine learning models (Logistic Regression, Random Forest)")
    print("   • Behavioral transition matrix analysis")
    print("   • Spectral analysis and autocorrelation")
    print("   • Comprehensive feature engineering")
    
    print("\n   📊 Outputs:")
    print("   • Enhanced_Derivatives_Analysis_Complete.png")
    print("   • enhanced_celegans_derivatives_raw_derivatives.csv")
    print("   • enhanced_celegans_derivatives_pca_derivatives.csv")
    print("   • enhanced_celegans_derivatives_behavioral_states.csv")
    
    print("\n3. INTEGRATED ANALYSIS SUITE:")
    print("   🔬 Features:")
    print("   • Complete neural dynamics analysis pipeline")
    print("   • Interactive 3D visualizations with Plotly")
    print("   • Advanced behavioral state prediction")
    print("   • Comprehensive statistical dashboard")
    print("   • Multi-model ensemble approaches")
    print("   • Real-time trajectory analysis")
    print("   • Cross-validation and model selection")
    
    print("\n   📊 Outputs:")
    print("   • Integrated_Neural_Analysis_Dashboard.png")
    print("   • Interactive_3D_PCA_Trajectory.html")
    print("   • Interactive_3D_Derivatives_Space.html")
    print("   • integrated_pca_scores.csv")
    print("   • integrated_derivatives.csv")
    print("   • integrated_behavioral_states.csv")
    print("   • integrated_model_features.csv")
    
    print("\n💡 RECOMMENDATIONS:")
    print("   • For basic analysis: Start with Enhanced PCA")
    print("   • For behavioral focus: Use Enhanced Derivatives")
    print("   • For complete characterization: Use Integrated Analysis")
    print("   • For comparison: Run All Analyses")

def main():
    """Main execution function"""
    print_banner()
    
    # Check system requirements
    print("\n🔍 Checking system requirements...")
    
    if not check_requirements():
        print("\n❌ System requirements not met. Please install missing packages.")
        return
    
    if not check_data_file():
        print("\n❌ Data file requirements not met. Please ensure data file exists.")
        return
    
    print("✅ All requirements satisfied!")
    
    # Analysis scripts mapping
    scripts = {
        1: ("PCA_Enhanced_Analysis.py", "Enhanced PCA Analysis"),
        2: ("Enhanced_Derivatives_Analysis.py", "Enhanced Derivatives Analysis"),
        3: ("Integrated_Enhanced_Analysis.py", "Integrated Analysis Suite")
    }
    
    while True:
        show_menu()
        
        try:
            choice = input("\n🎯 Select analysis (1-6): ").strip()
            
            if choice == '6':
                print("\n👋 Thank you for using the Enhanced Neural Analysis Suite!")
                print("🧠 Keep exploring neural dynamics! 🧠")
                break
                
            elif choice == '5':
                show_analysis_details()
                input("\n📖 Press Enter to return to menu...")
                continue
                
            elif choice == '4':
                print("\n🔄 Running ALL Enhanced Analyses...")
                print("This will execute all three analysis suites sequentially.")
                confirm = input("Continue? (y/N): ").lower().strip()
                
                if confirm == 'y':
                    success_count = 0
                    
                    for i in [1, 2, 3]:
                        script_name, description = scripts[i]
                        if run_script(script_name, description):
                            success_count += 1
                        print()  # Add spacing between analyses
                    
                    print("=" * 60)
                    print(f"🏁 Batch Analysis Complete!")
                    print(f"✅ Successfully completed: {success_count}/3 analyses")
                    
                    if success_count == 3:
                        print("🎉 All analyses completed successfully!")
                        print("📁 Check the generated files for comprehensive results")
                    else:
                        print("⚠️  Some analyses encountered issues")
                    
                    input("\n📖 Press Enter to return to menu...")
                    
            elif choice in ['1', '2', '3']:
                choice_num = int(choice)
                script_name, description = scripts[choice_num]
                
                print(f"\n🚀 You selected: {description}")
                confirm = input("Proceed with this analysis? (Y/n): ").lower().strip()
                
                if confirm in ['', 'y', 'yes']:
                    success = run_script(script_name, description)
                    
                    if success:
                        print(f"\n🎉 {description} completed!")
                        print("📁 Check the generated files for detailed results")
                    else:
                        print(f"\n⚠️  {description} encountered issues")
                        print("Please check the error messages above")
                    
                    input("\n📖 Press Enter to return to menu...")
                else:
                    print("Analysis cancelled.")
                    
            else:
                print("❌ Invalid choice. Please select 1-6.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Analysis interrupted by user. Goodbye!")
            break
            
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")
            print("Please try again or check your system configuration.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n💥 Critical error: {str(e)}")
        print("Please check your Python installation and try again.")
        input("\nPress Enter to exit...")