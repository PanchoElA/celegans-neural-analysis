import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def run_state_discovery():
    """
    Uses K-Means clustering on neural data (PCs) to discover
    hidden "brain states" and then analyzes the average
    behavior associated with each state.
    """
    
    # 1. Load the Merged Data
    csv_path = "predicting_model/merged_neural_behavior_data.csv"
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: '{csv_path}' not found.")
        print("Please run 'predict_neural_activity.py' first to generate it.")
        return

    print(f"Loaded merged data with {len(data)} timepoints.")

    # 2. Select Neural Data for Clustering
    # We will cluster on the first 3 PCs to find states
    neural_features = ['PC1', 'PC2', 'PC3']
    
    # Also list our behavior columns for the final analysis
    behavior_features = ['Velocity', 'HeadCurvature', 'BodyCurvature', 'Pumping']
    
    # Clean data: drop any rows where we don't have PCs
    data_clean = data.dropna(subset=neural_features)
    print(f"Using {len(data_clean)} timepoints for clustering.")

    # 3. Standardize the Neural Data
    # K-Means is distance-based, so scaling is very important
    scaler = StandardScaler()
    X_neural_scaled = scaler.fit_transform(data_clean[neural_features])

    # 4. Run K-Means Clustering
    
    # How many states? This is a key parameter to explore!
    # Let's start with 4, a common choice in papers (e.g., fwd, rev, turn_L, turn_R)
    n_states = 4 
    
    kmeans = KMeans(
        n_clusters=n_states, 
        n_init='auto',       # Modern default
        random_state=42      # For reproducible results
    )
    
    # Fit the model and get the state labels for each timepoint
    data_clean['Neural_State'] = kmeans.fit_predict(X_neural_scaled)
    
    print(f"\nSuccessfully clustered neural data into {n_states} states.")

    # 5. Interpret the Discovered States
    print("\n--- ANALYSIS OF NEURAL STATES ---")
    print("What is the average behavior for each state?")
    
    # This is the key result: Group by the new 'Neural_State'
    # and find the average of all the behavior columns.
    state_analysis = data_clean.groupby('Neural_State')[behavior_features].mean()
    
    print(state_analysis)

    # 6. Save the Results
    # This new CSV is very useful for your Plotly visualizations
    output_csv = "neural_states_with_behavior.csv"
    data_clean.to_csv(output_csv, index=False)
    print(f"\nSaved new dataset with state labels to: '{output_csv}'")
    print("\nYou can now use this file for your interactive plots!")


if __name__ == "__main__":
    run_state_discovery()