"""Analyze optimization log file and print recommendations."""
import sys
sys.path.insert(0, r'c:\Users\irisbridge\Desktop\projects\freqtrade\user_data')

from crypto_analysis import LSTMLogAnalyzer

# Run analysis
analyzer = LSTMLogAnalyzer(
    log_dir=r'c:\Users\irisbridge\Desktop\projects\freqtrade\user_data\notebooks\optimization_logs',
    top_percentile=10.0
)

report = analyzer.analyze_all(run_ids=['117271c5'])

# Print summary
print("=" * 60)
print("LSTM OPTIMIZATION LOG ANALYSIS")
print("=" * 60)
print(f"Run IDs: {report.run_ids}")
print(f"Total Individuals: {report.total_individuals}")
print()

# Feature Importance
print("-" * 60)
print("TOP FEATURES BY IMPORTANCE")
print("-" * 60)
fi = report.feature_importance
print("Top 15 features (by correlation with fitness):")
for i, feat in enumerate(fi.top_features[:15], 1):
    corr = fi.feature_correlations.loc[fi.feature_correlations['feature'] == feat, 'correlation'].values[0]
    freq = fi.selection_frequency.get(feat, 0)
    print(f"  {i:2}. {feat:20} corr={corr:+.4f}  sel_freq={freq:.2%}")
print()

# Parameter Analysis
print("-" * 60)
print("PARAMETER ANALYSIS")
print("-" * 60)
for param, result in report.parameter_analysis.items():
    print(f"\n{param}:")
    print(f"  Fitness correlation: {result.fitness_correlation:+.4f}")
    print(f"  Current bounds: [{result.current_bounds[0]:.4f}, {result.current_bounds[1]:.4f}]")
    print(f"  Optimal range (top 10%): [{result.optimal_range[0]:.4f}, {result.optimal_range[1]:.4f}]")
    print(f"  Bound proximity: {result.bound_proximity}")
    print(f"  Utilization: {result.utilization:.1%}")
    if result.recommended_bounds:
        print(f"  RECOMMENDED bounds: [{result.recommended_bounds[0]:.4f}, {result.recommended_bounds[1]:.4f}]")
print()

# Evolution Analysis
print("-" * 60)
print("EVOLUTION ANALYSIS")
print("-" * 60)
evo = report.evolution
print(f"Converged parameters: {evo.converged_params}")
print(f"Diverse parameters (still exploring): {evo.diverse_params}")
print("\nConvergence scores (higher = more converged):")
for param, score in sorted(evo.convergence_scores.items(), key=lambda x: -x[1])[:10]:
    print(f"  {param:25}: {score:.3f}")
print()

# Recommendations
print("-" * 60)
print("RECOMMENDATIONS")
print("-" * 60)
rec = report.recommendations
print(f"\nConfidence: {rec.confidence:.1%}")

print("\nRecommended Hyperparameter Bounds:")
for param, bounds in rec.hyperparam_bounds.items():
    print(f"  {param}: [{bounds[0]:.6f}, {bounds[1]:.6f}]")

print("\nAPO Settings:")
for setting, value in rec.apo_settings.items():
    print(f"  {setting}: {value}")

print("\nFeature Suggestions:")
keep_count = sum(1 for v in rec.feature_suggestions.values() if v == 'keep')
remove_count = sum(1 for v in rec.feature_suggestions.values() if v == 'consider_removing')
print(f"  Keep: {keep_count} features")
print(f"  Consider removing: {remove_count} features")

if remove_count > 0:
    print("\n  Features to consider removing:")
    for feat, suggestion in rec.feature_suggestions.items():
        if suggestion == 'consider_removing':
            print(f"    - {feat}")

print("\nRationale:")
for key, explanation in rec.rationale.items():
    print(f"  {key}: {explanation}")

print("\n" + "=" * 60)
print("END OF ANALYSIS")
print("=" * 60)

# Export config code
config_path = r'c:\Users\irisbridge\Desktop\projects\freqtrade\user_data\recommended_config.py'
analyzer.export_config_code(report.recommendations, config_path)
print("\n\n")
print("=" * 60)
print("GENERATED CONFIG CODE")
print("=" * 60)
with open(config_path, 'r') as f:
    print(f.read())
