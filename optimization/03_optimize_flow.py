"""
Optimize Flow
Maximizes hourly revenue by choosing optimal discharge Q
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Constants
DATA_PATH = Path("clean/Merged_hourly_data_with_revenue.csv")
MODEL_PATH = Path("optimization/regression_models.pkl")
CONSTRAINTS_PATH = Path("optimization/flow_constraints.pkl")
OUTPUT_DIR = Path("optimization")

# Physical constants (matching the original dataset calculations)
RHO = 1000    # Water density (kg/m³)
G = 9.81      # Gravity (m/s²)
ETA = 0.998   # Turbine efficiency (99.8% - from original data)

def load_all_data():
    """Load dataset, models, and constraints"""
    df = pd.read_csv(DATA_PATH)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    with open(MODEL_PATH, 'rb') as f:
        models = pickle.load(f)
    
    with open(CONSTRAINTS_PATH, 'rb') as f:
        constraints = pickle.load(f)
    
    return df, models, constraints

def predict_water_levels(Q, models):
    """Predict upstream and downstream water levels for given Q"""
    Q_array = np.array([[Q]])
    Q_poly = models['poly_features'].transform(Q_array)
    
    upstream_level = models['model_upstream'].predict(Q_poly)[0]
    downstream_level = models['model_downstream'].predict(Q_poly)[0]
    
    return upstream_level, downstream_level

def compute_hydraulic_head(Q, models):
    """Compute hydraulic head = upstream - downstream"""
    upstream, downstream = predict_water_levels(Q, models)
    return upstream - downstream

def compute_power(Q, h):
    """Compute power in Watts: P = η * ρ * g * Q * h"""
    return ETA * RHO * G * Q * h

def compute_revenue(Q, h, price_per_kWh):
    """
    Compute hourly revenue in dollars
    
    Parameters:
    - Q: discharge (m³/s)
    - h: hydraulic head (m)
    - price_per_kWh: price in $/kWh
    
    Returns:
    - revenue in $
    """
    power_W = compute_power(Q, h)
    energy_kWh = (power_W / 1000) * 1  # Power in kW × 1 hour
    revenue = energy_kWh * price_per_kWh
    return revenue

def optimize_single_hour(price_per_kWh, models, Q_min, Q_max):
    """
    Optimize flow for a single hour to maximize revenue
    
    Returns:
    - optimal_Q: Optimized discharge
    - optimal_revenue: Revenue at optimal Q
    - optimal_power: Power at optimal Q (in MW)
    - optimal_head: Hydraulic head at optimal Q
    - optimal_upstream: Upstream level at optimal Q
    - optimal_downstream: Downstream level at optimal Q
    """
    
    # Objective function (negative because we minimize)
    def negative_revenue(Q):
        h = compute_hydraulic_head(Q, models)
        return -compute_revenue(Q, h, price_per_kWh)
    
    # Optimize
    result = minimize_scalar(
        negative_revenue,
        bounds=(Q_min, Q_max),
        method='bounded'
    )
    
    optimal_Q = result.x
    optimal_h = compute_hydraulic_head(optimal_Q, models)
    optimal_power_MW = compute_power(optimal_Q, optimal_h) / 1e6
    optimal_revenue = -result.fun
    optimal_upstream, optimal_downstream = predict_water_levels(optimal_Q, models)
    
    return {
        'optimized_flow': optimal_Q,
        'optimized_head': optimal_h,
        'optimized_power_MW': optimal_power_MW,
        'optimized_revenue': optimal_revenue,
        'optimized_upstream_level': optimal_upstream,
        'optimized_downstream_level': optimal_downstream
    }

def run_optimization(df, models, constraints):
    """
    Run optimization for all hours with water balance constraint.
    The total water used must equal the total available from Lake Ontario.
    
    Key Insight: Hydraulic head is nearly constant (CV=0.66%), actively managed
    by dam operators regardless of flow. We use constant head assumption for
    optimization, which makes Revenue ∝ Q × Price.
    """
    
    print("Running optimization with water balance constraint...")
    print("This will optimize flow distribution across all hours while maintaining")
    print("the same total water volume as actual operations.\n")
    
    Q_min = constraints['Q_min_feasible']
    Q_max = constraints['Q_max_feasible']
    
    # Calculate constant hydraulic head (mean from actual data)
    h_constant = df['hydraulic_head, h'].mean()
    print(f"Using constant hydraulic head: {h_constant:.3f} m (actual mean)")
    print(f"  (Head varies by only ±{df['hydraulic_head, h'].std():.3f}m, CV={df['hydraulic_head, h'].std()/df['hydraulic_head, h'].mean()*100:.2f}%)\n")
    
    # Calculate total water budget
    total_water_budget = df['Discharge (cubic meters per second) - Débit (mètres cubes par seconde), Q'].sum()
    print(f"Total water budget: {total_water_budget:,.0f} m³")
    print(f"Average must be: {total_water_budget / len(df):,.0f} m³/s")
    print(f"Feasible range per hour: [{Q_min:,.0f}, {Q_max:,.0f}] m³/s\n")
    
    # Step 1: Allocate flow proportional to (Price × constant_head) = Price
    print("Step 1: Allocating flow proportional to electricity price...")
    print("  With constant head, Revenue ∝ Q × Price")
    print("  Therefore, allocate more Q to high-price hours\n")
    
    price_weights = df['TOU_price_$per_kWh'].values
    
    # Start with minimum flow everywhere
    optimized_flows = np.full(len(df), Q_min)
    
    # Remaining water to allocate
    water_allocated = Q_min * len(df)
    water_remaining = total_water_budget - water_allocated
    
    print(f"Baseline allocation (Q_min): {water_allocated:,.0f} m³")
    print(f"Remaining to allocate: {water_remaining:,.0f} m³")
    
    if water_remaining > 0:
        # Distribute remaining water proportional to price, up to Q_max
        available_capacity = (Q_max - Q_min) * len(df)
        
        print(f"Available capacity: {available_capacity:,.0f} m³\n")
        
        if water_remaining <= available_capacity:
            # Simple proportional allocation
            flow_increment = water_remaining * (price_weights / price_weights.sum())
            optimized_flows = Q_min + flow_increment
            
            # Clip to [Q_min, Q_max]
            optimized_flows = np.clip(optimized_flows, Q_min, Q_max)
            
            # Adjust to exactly match budget (due to clipping)
            adjustment_factor = total_water_budget / optimized_flows.sum()
            optimized_flows *= adjustment_factor
        else:
            # Need to use max flow for some hours
            print("  ⚠ Warning: Cannot satisfy water budget within flow constraints")
            optimized_flows = np.full(len(df), Q_max)
    
    # Verify water balance
    optimized_total = optimized_flows.sum()
    error = abs(optimized_total - total_water_budget) / total_water_budget * 100
    
    print(f"\nStep 2: Water Balance Verification:")
    print(f"  Target total:     {total_water_budget:,.0f} m³")
    print(f"  Optimized total:  {optimized_total:,.0f} m³")
    print(f"  Error:            {error:.6f}%")
    
    # Step 3: Calculate results for each hour with optimized flows
    print("\nStep 3: Calculating optimized power and revenue (using constant head)...")
    results = []
    
    for idx, row in df.iterrows():
        # Extract current hour data
        price = row['TOU_price_$per_kWh']
        actual_Q = row['Discharge (cubic meters per second) - Débit (mètres cubes par seconde), Q']
        actual_revenue = row['revenue_hourly_$']
        actual_power_W = row['Power (Watts)']
        actual_energy_kWh = row['energy_kWh']
        actual_head = row['hydraulic_head, h']
        # Use absolute elevations instead of relative water levels
        actual_upstream = row['hydraulic_head_LSD']
        actual_downstream = row['hydraulic_head_cornwall']
        
        # Use optimized flow for this hour
        opt_Q = optimized_flows[idx]
        
        # Use CONSTANT head (not regression model!)
        head_opt = h_constant
        
        # For water level reporting, use regression (but not for revenue calc)
        upstream_opt, downstream_opt = predict_water_levels(opt_Q, models)
        
        # Calculate power and revenue with constant head
        power_opt_MW = compute_power(opt_Q, head_opt) / 1e6
        energy_opt_kWh = power_opt_MW * 1000
        revenue_opt = compute_revenue(opt_Q, head_opt, price)
        
        # Calculate differences
        flow_diff = opt_Q - actual_Q
        power_diff_MW = power_opt_MW - (actual_power_W / 1e6)
        revenue_diff = revenue_opt - actual_revenue
        revenue_improvement_pct = (revenue_diff / actual_revenue * 100) if actual_revenue > 0 else 0
        energy_diff_kWh = energy_opt_kWh - actual_energy_kWh
        
        # Check feasibility
        feasible = (
            constraints['upstream_min'] <= upstream_opt <= constraints['upstream_max'] and
            constraints['downstream_min'] <= downstream_opt <= constraints['downstream_max']
        )
        
        # Store results
        results.append({
            'DateTime': row['DateTime'],
            'month': row['month'],
            'TOU_period': row['TOU_period'],
            'TOU_price': price,
            
            # Actual values
            'lake_ontario_outflow': actual_Q,
            'upstream_level': actual_upstream,
            'downstream_level': actual_downstream,
            'hydraulic_head': actual_head,
            'power_MW': actual_power_W / 1e6,
            'energy_kWh': actual_energy_kWh,
            'revenue_actual': actual_revenue,
            
            # Optimized values
            'optimized_flow': opt_Q,
            'optimized_upstream_level': upstream_opt,
            'optimized_downstream_level': downstream_opt,
            'optimized_head': head_opt,
            'optimized_power_MW': power_opt_MW,
            'optimized_energy_kWh': energy_opt_kWh,
            'optimized_revenue': revenue_opt,
            
            # Differences
            'flow_diff': flow_diff,
            'power_diff_MW': power_diff_MW,
            'energy_diff_kWh': energy_diff_kWh,
            'revenue_diff': revenue_diff,
            'revenue_improvement_pct': revenue_improvement_pct,
            'feasible': feasible
        })
    
    results_df = pd.DataFrame(results)
    
    print(f"\n✓ Optimization complete for {len(results_df)} hours")
    
    return results_df

def print_summary(results_df):
    """Print optimization summary statistics"""
    
    total_actual = results_df['revenue_actual'].sum()
    total_optimized = results_df['optimized_revenue'].sum()
    total_improvement = total_optimized - total_actual
    pct_improvement = (total_improvement / total_actual) * 100
    
    feasible_count = results_df['feasible'].sum()
    feasible_pct = (feasible_count / len(results_df)) * 100
    
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*60)
    print(f"Total Actual Revenue:      ${total_actual:,.2f}")
    print(f"Total Optimized Revenue:   ${total_optimized:,.2f}")
    print(f"Total Improvement:         ${total_improvement:,.2f}")
    print(f"Percentage Improvement:    {pct_improvement:.2f}%")
    print(f"\nFeasible Solutions:        {feasible_count}/{len(results_df)} ({feasible_pct:.1f}%)")
    print(f"\nAverage Flow Difference:   {results_df['flow_diff'].mean():.0f} m³/s")
    print(f"Average Power Difference:  {results_df['power_diff_MW'].mean():.2f} MW")
    print("="*60 + "\n")

def main():
    print("\n" + "="*60)
    print("STEP 3: OPTIMIZING FLOW FOR MAXIMUM REVENUE")
    print("="*60 + "\n")
    
    # Load data
    print("Loading data, models, and constraints...")
    df, models, constraints = load_all_data()
    print(f"✓ Loaded {len(df)} hourly records\n")
    
    # Run optimization
    results_df = run_optimization(df, models, constraints)
    
    # Print summary
    print_summary(results_df)
    
    # Save results
    output_path = OUTPUT_DIR / 'optimization_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"✓ Results saved to {output_path}")
    
    print("\n" + "="*60)
    print("STEP 3 COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()