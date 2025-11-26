"""
Generate Reports and Visualizations
Creates plots and summary statistics for optimization results
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Constants
RESULTS_PATH = Path("optimization/optimization_results.csv")
OUTPUT_DIR = Path("optimization")

def load_results():
    """Load optimization results"""
    df = pd.read_csv(RESULTS_PATH)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    return df

def create_revenue_comparison_plot(df):
    """Plot actual vs optimized revenue over time"""
    
    # Aggregate to daily for better visualization
    df_daily = df.groupby(df['DateTime'].dt.date).agg({
        'revenue_actual': 'sum',
        'optimized_revenue': 'sum',
        'revenue_diff': 'sum'
    }).reset_index()
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Daily Revenue Comparison
    ax1 = axes[0]
    ax1.plot(df_daily['DateTime'], df_daily['revenue_actual'], 
             label='Actual Revenue', linewidth=1.5, alpha=0.7)
    ax1.plot(df_daily['DateTime'], df_daily['optimized_revenue'], 
             label='Optimized Revenue', linewidth=1.5, alpha=0.7)
    ax1.fill_between(df_daily['DateTime'], df_daily['revenue_actual'], 
                      df_daily['optimized_revenue'], alpha=0.3, label='Improvement')
    ax1.set_ylabel('Daily Revenue ($)', fontsize=12)
    ax1.set_title('Daily Revenue: Actual vs Optimized', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Daily Revenue Improvement
    ax2 = axes[1]
    ax2.bar(df_daily['DateTime'], df_daily['revenue_diff'], 
            alpha=0.7, color='green', width=1)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Revenue Improvement ($)', fontsize=12)
    ax2.set_title('Daily Revenue Improvement from Optimization', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'revenue_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Revenue comparison plot saved")
    plt.close()

def create_flow_profiles_plot(df):
    """Plot actual vs optimized flow profiles"""
    
    # Sample first 7 days for clarity
    df_week = df[df['DateTime'] < '2024-01-08']
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Flow Comparison
    ax1 = axes[0]
    ax1.plot(df_week['DateTime'], df_week['lake_ontario_outflow'], 
             label='Actual Flow', linewidth=2, alpha=0.7)
    ax1.plot(df_week['DateTime'], df_week['optimized_flow'], 
             label='Optimized Flow', linewidth=2, alpha=0.7)
    ax1.set_ylabel('Discharge (m³/s)', fontsize=12)
    ax1.set_title('Flow Profiles: First Week of January 2024', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Power Generation
    ax2 = axes[1]
    ax2.plot(df_week['DateTime'], df_week['power_MW'], 
             label='Actual Power', linewidth=2, alpha=0.7)
    ax2.plot(df_week['DateTime'], df_week['optimized_power_MW'], 
             label='Optimized Power', linewidth=2, alpha=0.7)
    ax2.set_ylabel('Power (MW)', fontsize=12)
    ax2.set_title('Power Generation Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: TOU Pricing
    ax3 = axes[2]
    colors = {'Off-Peak': 'blue', 'Mid-Peak': 'orange', 'On-Peak': 'red'}
    for period in df_week['TOU_period'].unique():
        mask = df_week['TOU_period'] == period
        ax3.scatter(df_week.loc[mask, 'DateTime'], 
                   df_week.loc[mask, 'TOU_price'], 
                   label=period, alpha=0.6, s=30, color=colors.get(period, 'gray'))
    ax3.set_xlabel('DateTime', fontsize=12)
    ax3.set_ylabel('Price ($/kWh)', fontsize=12)
    ax3.set_title('Time-of-Use Pricing Periods', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'optimized_profiles.png', dpi=300, bbox_inches='tight')
    print(f"✓ Flow profiles plot saved")
    plt.close()

def create_water_level_comparison(df):
    """Plot water level predictions vs actual"""
    
    # Sample data
    df_sample = df.iloc[::24]  # Every 24th hour for clarity
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Upstream levels
    ax1 = axes[0]
    ax1.plot(df_sample['DateTime'], df_sample['upstream_level'], 
             'o-', label='Actual Upstream', alpha=0.7, markersize=3)
    ax1.plot(df_sample['DateTime'], df_sample['optimized_upstream_level'], 
             's-', label='Predicted Upstream (Optimized)', alpha=0.7, markersize=3)
    ax1.set_ylabel('Elevation (m above datum)', fontsize=12)
    ax1.set_title('Upstream Water Elevation (Long Sault Dam)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Downstream levels
    ax2 = axes[1]
    ax2.plot(df_sample['DateTime'], df_sample['downstream_level'], 
             'o-', label='Actual Downstream', alpha=0.7, markersize=3)
    ax2.plot(df_sample['DateTime'], df_sample['optimized_downstream_level'], 
             's-', label='Predicted Downstream (Optimized)', alpha=0.7, markersize=3)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Elevation (m above datum)', fontsize=12)
    ax2.set_title('Downstream Water Elevation (Cornwall)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'water_level_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Water level comparison plot saved")
    plt.close()

def create_improvement_by_period(df):
    """Analyze improvement by TOU period"""
    
    summary = df.groupby('TOU_period').agg({
        'revenue_actual': 'sum',
        'optimized_revenue': 'sum',
        'revenue_diff': 'sum',
        'revenue_improvement_pct': 'mean'
    }).reset_index()
    
    summary['improvement_pct'] = (summary['revenue_diff'] / summary['revenue_actual']) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Revenue by period
    ax1 = axes[0]
    x = np.arange(len(summary))
    width = 0.35
    ax1.bar(x - width/2, summary['revenue_actual'], width, label='Actual', alpha=0.8)
    ax1.bar(x + width/2, summary['optimized_revenue'], width, label='Optimized', alpha=0.8)
    ax1.set_xlabel('TOU Period', fontsize=12)
    ax1.set_ylabel('Total Revenue ($)', fontsize=12)
    ax1.set_title('Revenue by Time-of-Use Period', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary['TOU_period'])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Improvement percentage
    ax2 = axes[1]
    colors = ['red' if period == 'On-Peak' else 'orange' if period == 'Mid-Peak' else 'blue' 
              for period in summary['TOU_period']]
    ax2.bar(summary['TOU_period'], summary['improvement_pct'], alpha=0.8, color=colors)
    ax2.set_xlabel('TOU Period', fontsize=12)
    ax2.set_ylabel('Revenue Improvement (%)', fontsize=12)
    ax2.set_title('Revenue Improvement by TOU Period', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'improvement_by_period.png', dpi=300, bbox_inches='tight')
    print(f"✓ Improvement by period plot saved")
    plt.close()
    
    return summary

def create_summary_json(df):
    """Create JSON summary of optimization results"""
    
    total_actual = df['revenue_actual'].sum()
    total_optimized = df['optimized_revenue'].sum()
    total_improvement = total_optimized - total_actual
    pct_improvement = (total_improvement / total_actual) * 100
    
    # Monthly breakdown
    df['month_name'] = pd.to_datetime(df['DateTime']).dt.strftime('%B')
    monthly = df.groupby('month_name').agg({
        'revenue_actual': 'sum',
        'optimized_revenue': 'sum',
        'revenue_diff': 'sum'
    }).to_dict()
    
    # By TOU period
    by_period = df.groupby('TOU_period').agg({
        'revenue_actual': 'sum',
        'optimized_revenue': 'sum',
        'revenue_diff': 'sum'
    }).to_dict()
    
    summary = {
        'total_revenue_actual_$': float(total_actual),
        'total_revenue_optimized_$': float(total_optimized),
        'total_improvement_$': float(total_improvement),
        'improvement_percentage': float(pct_improvement),
        'feasible_solutions_count': int(df['feasible'].sum()),
        'feasible_solutions_pct': float((df['feasible'].sum() / len(df)) * 100),
        'avg_flow_difference_m3_per_s': float(df['flow_diff'].mean()),
        'avg_power_difference_MW': float(df['power_diff_MW'].mean()),
        'total_hours_analyzed': len(df),
        'by_TOU_period': {
            period: {
                'actual_$': float(by_period['revenue_actual'][period]),
                'optimized_$': float(by_period['optimized_revenue'][period]),
                'improvement_$': float(by_period['revenue_diff'][period])
            }
            for period in by_period['revenue_actual'].keys()
        }
    }
    
    output_path = OUTPUT_DIR / 'optimization_summary.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary JSON saved to {output_path}")
    
    return summary

def main():
    print("\n" + "="*60)
    print("STEP 4: GENERATING REPORTS AND VISUALIZATIONS")
    print("="*60 + "\n")
    
    # Load results
    print("Loading optimization results...")
    df = load_results()
    print(f"✓ Loaded {len(df)} hours of results\n")
    
    # Create visualizations
    print("Creating visualizations...")
    create_revenue_comparison_plot(df)
    create_flow_profiles_plot(df)
    create_water_level_comparison(df)
    period_summary = create_improvement_by_period(df)
    
    # Create summary JSON
    print("\nGenerating summary statistics...")
    summary = create_summary_json(df)
    
    # Print key insights
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print(f"Total Revenue Improvement: ${summary['total_improvement_$']:,.2f} ({summary['improvement_percentage']:.2f}%)")
    print(f"Best improvement period: {period_summary.loc[period_summary['improvement_pct'].idxmax(), 'TOU_period']}")
    print(f"Average flow adjustment: {summary['avg_flow_difference_m3_per_s']:.0f} m³/s")
    print("="*60)
    
    print("\n" + "="*60)
    print("STEP 4 COMPLETE")
    print("All visualizations and reports saved to optimization/")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()