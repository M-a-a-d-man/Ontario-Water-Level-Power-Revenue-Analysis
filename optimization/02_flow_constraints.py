"""
number 2: get flow constraints
Determines min/max feasible flows based on water level limits
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.optimize import brentq

# Constants
DATA_PATH = Path("clean/Merged_hourly_data_with_revenue.csv")
MODEL_PATH = Path("optimization/regression_models.pkl")
OUTPUT_DIR = Path("optimization")

def load_data_and_models():
    """Load dataset and regression models"""
    df = pd.read_csv(DATA_PATH)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    with open(MODEL_PATH, 'rb') as f:
        models = pickle.load(f)
    
    return df, models

def predict_water_levels(Q, models):
    """Predict water levels for given discharge Q"""
    Q_array = np.array([[Q]])
    Q_poly = models['poly_features'].transform(Q_array)
    
    upstream_level = models['model_upstream'].predict(Q_poly)[0]
    downstream_level = models['model_downstream'].predict(Q_poly)[0]
    
    return upstream_level, downstream_level

def extract_level_constraints(df):
    """Extract min/max allowed water levels from historical data"""
    # Use absolute elevations instead of relative water levels
    upstream_col = 'hydraulic_head_LSD'
    downstream_col = 'hydraulic_head_cornwall'
    
    constraints = {
        'upstream_min': df[upstream_col].min(),
        'upstream_max': df[upstream_col].max(),
        'downstream_min': df[downstream_col].min(),
        'downstream_max': df[downstream_col].max(),
        'Q_min_historical': df['Discharge (cubic meters per second) - Débit (mètres cubes par seconde), Q'].min(),
        'Q_max_historical': df['Discharge (cubic meters per second) - Débit (mètres cubes par seconde), Q'].max()
    }
    
    print("=" * 60)
    print("WATER LEVEL AND FLOW CONSTRAINTS")
    print("=" * 60)
    print(f"Upstream Level Range:   [{constraints['upstream_min']:.3f}, {constraints['upstream_max']:.3f}] m")
    print(f"Downstream Level Range: [{constraints['downstream_min']:.3f}, {constraints['downstream_max']:.3f}] m")
    print(f"Historical Flow Range:  [{constraints['Q_min_historical']:.0f}, {constraints['Q_max_historical']:.0f}] m³/s")
    print("=" * 60 + "\n")
    
    return constraints

def compute_feasible_flow_range(models, constraints, safety_margin=0.01):
    """
    Compute feasible Q_min and Q_max that keep levels within bounds
    
    Parameters:
    - models: Regression models
    - constraints: Level constraints dict
    - safety_margin: Safety buffer in meters (default 0.01m = 1cm)
    """
    
    # Add safety margins to constraints
    upstream_min = constraints['upstream_min'] + safety_margin
    upstream_max = constraints['upstream_max'] - safety_margin
    downstream_min = constraints['downstream_min'] + safety_margin
    downstream_max = constraints['downstream_max'] - safety_margin
    
    # Start with historical flow range as initial bounds
    Q_search_min = constraints['Q_min_historical']
    Q_search_max = constraints['Q_max_historical']
    
    # We need to find Q values that satisfy all four constraints:
    # 1. upstream_level >= upstream_min
    # 2. upstream_level <= upstream_max
    # 3. downstream_level >= downstream_min
    # 4. downstream_level <= downstream_max
    
    # Sample many Q values to find feasible range
    Q_test = np.linspace(Q_search_min * 0.5, Q_search_max * 1.5, 1000)
    feasible_Q = []
    
    for Q in Q_test:
        up_level, down_level = predict_water_levels(Q, models)
        
        if (upstream_min <= up_level <= upstream_max and 
            downstream_min <= down_level <= downstream_max):
            feasible_Q.append(Q)
    
    if len(feasible_Q) == 0:
        # If no feasible flows found, use historical range
        print("⚠ Warning: No fully feasible flows found. Using historical range.")
        Q_min_feasible = Q_search_min
        Q_max_feasible = Q_search_max
    else:
        Q_min_feasible = min(feasible_Q)
        Q_max_feasible = max(feasible_Q)
    
    # Also respect turbine capacity (use historical max as proxy)
    Q_max_feasible = min(Q_max_feasible, constraints['Q_max_historical'])
    
    return Q_min_feasible, Q_max_feasible

def main():
    print("\n" + "="*60)
    print("STEP 2: COMPUTING FLOW CONSTRAINTS")
    print("="*60 + "\n")
    
    # Load data and models
    print("Loading data and models...")
    df, models = load_data_and_models()
    print(f"✓ Loaded {len(df)} records and regression models\n")
    
    # Extract level constraints
    constraints = extract_level_constraints(df)
    
    # Compute feasible flow range
    print("Computing feasible flow range...")
    Q_min, Q_max = compute_feasible_flow_range(models, constraints)
    
    print(f"✓ Feasible Flow Range: [{Q_min:.0f}, {Q_max:.0f}] m³/s\n")
    
    # Save constraints
    constraints['Q_min_feasible'] = Q_min
    constraints['Q_max_feasible'] = Q_max
    
    output_path = OUTPUT_DIR / 'flow_constraints.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(constraints, f)
    
    print(f"✓ Constraints saved to {output_path}")
    
    print("\n" + "="*60)
    print("STEP 2 COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()