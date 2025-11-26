"""
Master Script - Run Complete Optimization Pipeline
Executes all steps in sequence for Part C optimization
"""

import sys
import subprocess
from pathlib import Path
import os

def cleanup_old_artifacts():
    """Remove old optimization artifacts before running new analysis"""
    optimization_dir = Path("optimization")
    
    # Define artifact patterns to remove
    artifacts = [
        'regression_models.pkl',
        'regression_models.png',
        'flow_constraints.pkl',
        'optimization_results.csv',
        'optimization_summary.json',
        'revenue_comparison.png',
        'optimized_profiles.png',
        'water_level_comparison.png',
        'improvement_by_period.png'
    ]
    
    print("🧹 Cleaning up old artifacts...")
    removed_count = 0
    
    for artifact in artifacts:
        artifact_path = optimization_dir / artifact
        if artifact_path.exists():
            os.remove(artifact_path)
            print(f"  ✓ Removed: {artifact}")
            removed_count += 1
    
    if removed_count == 0:
        print("  ℹ No old artifacts found")
    else:
        print(f"  ✓ Cleaned up {removed_count} old files")
    
    print()

def main():
    print("\n" + "="*70)
    print(" "*15 + "PART C: HYDROPOWER OPTIMIZATION")
    print(" "*10 + "Ontario Water Level & Power Revenue Analysis")
    print("="*70 + "\n")
    
    # Clean up old artifacts first
    cleanup_old_artifacts()
    
    try:
        # Define script paths
        optimization_dir = Path(__file__).parent
        scripts = [
            ("Step 1: Building Regression Models", "01_regression_models.py"),
            ("Step 2: Computing Flow Constraints", "02_flow_constraints.py"),
            ("Step 3: Optimizing Flow for Maximum Revenue", "03_optimize_flow.py"),
            ("Step 4: Generating Reports and Visualizations", "04_generate_reports.py")
        ]
        
        # Run each step
        for step_name, script_name in scripts:
            print(f"▶ Running {step_name}...")
            script_path = optimization_dir / script_name
            
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=optimization_dir.parent
            )
            
            # Print output
            if result.stdout:
                print(result.stdout)
            
            # Check for errors
            if result.returncode != 0:
                print(f"❌ Error in {script_name}:")
                if result.stderr:
                    print(result.stderr)
                sys.exit(1)
            
            print()
        
        print("\n" + "="*70)
        print(" "*20 + "✓ OPTIMIZATION COMPLETE!")
        print(" "*15 + "All results saved to optimization/")
        print("="*70 + "\n")
        
        print("📁 Generated Files:")
        print("  - optimization/regression_models.pkl")
        print("  - optimization/regression_models.png")
        print("  - optimization/flow_constraints.pkl")
        print("  - optimization/optimization_results.csv")
        print("  - optimization/optimization_summary.json")
        print("  - optimization/revenue_comparison.png")
        print("  - optimization/optimized_profiles.png")
        print("  - optimization/water_level_comparison.png")
        print("  - optimization/improvement_by_period.png")
        print()
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

