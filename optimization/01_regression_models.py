"""
Build Regression Models
Predicts upstream and downstream water levels as functions Q
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
from pathlib import Path

# Constants
DATA_PATH = Path("clean/Merged_hourly_data_with_revenue.csv")
OUTPUT_DIR = Path("optimization")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load the master dataset"""
    df = pd.read_csv(DATA_PATH)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    return df

def build_regression_models(df, degree=2):
    """
    Build polynomial regression models for water levels vs discharge
    
    Parameters:
    - df: DataFrame with the master data
    - degree: Polynomial degree (1=linear, 2=quadratic)
    
    Returns:
    - models: Dict containing fitted models and transformers
    """
    
    # Extract relevant columns
    Q = df['Discharge (cubic meters per second) - Débit (mètres cubes par seconde), Q'].values.reshape(-1, 1)
    # Use absolute elevations (hydraulic_head columns) instead of relative water levels
    upstream_level = df['hydraulic_head_LSD'].values
    downstream_level = df['hydraulic_head_cornwall'].values
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    Q_poly = poly.fit_transform(Q)
    
    # Model A: Upstream level = f1(Q)
    model_upstream = LinearRegression()
    model_upstream.fit(Q_poly, upstream_level)
    upstream_pred = model_upstream.predict(Q_poly)
    
    # Model B: Downstream level = f2(Q)
    model_downstream = LinearRegression()
    model_downstream.fit(Q_poly, downstream_level)
    downstream_pred = model_downstream.predict(Q_poly)
    
    # Calculate metrics
    metrics = {
        'upstream': {
            'r2': r2_score(upstream_level, upstream_pred),
            'mae': mean_absolute_error(upstream_level, upstream_pred)
        },
        'downstream': {
            'r2': r2_score(downstream_level, downstream_pred),
            'mae': mean_absolute_error(downstream_level, downstream_pred)
        }
    }
    
    # Store models
    models = {
        'poly_features': poly,
        'model_upstream': model_upstream,
        'model_downstream': model_downstream,
        'degree': degree,
        'metrics': metrics
    }
    
    print("=" * 60)
    print("REGRESSION MODEL RESULTS")
    print("=" * 60)
    print(f"Polynomial Degree: {degree}")
    print(f"\nUpstream Level Model:")
    print(f"  R² Score: {metrics['upstream']['r2']:.4f}")
    print(f"  MAE: {metrics['upstream']['mae']:.4f} meters")
    print(f"\nDownstream Level Model:")
    print(f"  R² Score: {metrics['downstream']['r2']:.4f}")
    print(f"  MAE: {metrics['downstream']['mae']:.4f} meters")
    print("=" * 60)
    
    return models, Q, upstream_level, downstream_level, upstream_pred, downstream_pred

def plot_regression_results(Q, upstream_level, downstream_level, 
                           upstream_pred, downstream_pred, models):
    """Create visualization of regression models"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Sort for plotting
    sort_idx = np.argsort(Q.flatten())
    Q_sorted = Q.flatten()[sort_idx]
    
    # Plot A: Upstream Level vs Q
    ax1 = axes[0]
    ax1.scatter(Q.flatten(), upstream_level, alpha=0.3, s=1, label='Actual Data')
    ax1.plot(Q_sorted, upstream_pred[sort_idx], 'r-', linewidth=2, label='Regression Model')
    ax1.set_xlabel('Discharge Q (m³/s)', fontsize=12)
    ax1.set_ylabel('Upstream Water Level (m)', fontsize=12)
    ax1.set_title(f'Upstream Level = f₁(Q)\nR² = {models["metrics"]["upstream"]["r2"]:.4f}', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot B: Downstream Level vs Q
    ax2 = axes[1]
    ax2.scatter(Q.flatten(), downstream_level, alpha=0.3, s=1, label='Actual Data')
    ax2.plot(Q_sorted, downstream_pred[sort_idx], 'r-', linewidth=2, label='Regression Model')
    ax2.set_xlabel('Discharge Q (m³/s)', fontsize=12)
    ax2.set_ylabel('Downstream Water Level (m)', fontsize=12)
    ax2.set_title(f'Downstream Level = f₂(Q)\nR² = {models["metrics"]["downstream"]["r2"]:.4f}', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'regression_models.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Regression plot saved to {OUTPUT_DIR / 'regression_models.png'}")
    plt.close()

def save_models(models):
    """Save fitted models to disk"""
    with open(OUTPUT_DIR / 'regression_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    print(f"✓ Models saved to {OUTPUT_DIR / 'regression_models.pkl'}")

def main():
    print("\n" + "="*60)
    print("STEP 1: BUILDING REGRESSION MODELS")
    print("="*60 + "\n")
    
    # Load data
    print("Loading data...")
    df = load_data()
    print(f"✓ Loaded {len(df)} hourly records")
    
    # Build models (using linear regression, degree=1)
    print("\nBuilding regression models...")
    models, Q, upstream_level, downstream_level, upstream_pred, downstream_pred = \
        build_regression_models(df, degree=1)
    
    # Plot results
    print("\nCreating visualizations...")
    plot_regression_results(Q, upstream_level, downstream_level, 
                           upstream_pred, downstream_pred, models)
    
    # Save models
    print("\nSaving models...")
    save_models(models)
    
    print("\n" + "="*60)
    print("STEP 1 COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()