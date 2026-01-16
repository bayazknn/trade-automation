#!/usr/bin/env python3
"""Analyze optimization run 7caf7327."""

import sys
sys.path.insert(0, '/workspace/trade-automation')

from crypto_analysis.log_analyzer import LSTMLogAnalyzer
import numpy as np

# Initialize analyzer
analyzer = LSTMLogAnalyzer(
    log_dir='/workspace/trade-automation/notebooks/optimization_logs',
    top_percentile=10.0
)

# Load logs for the specific run
print("=" * 70)
print("LOADING DATA FOR RUN 7caf7327")
print("=" * 70)
df = analyzer.load_logs(run_ids=['7caf7327'])

print(f"Total individuals: {len(df)}")
print(f"Iterations: {df['iteration'].min()} to {df['iteration'].max()}")
print(f"Fitness range: [{df['fitness'].min():.4f}, {df['fitness'].max():.4f}]")
print(f"Best fitness (F1 product): {-df['fitness'].min():.4f}")
print()

# =========================================================================
# 1. Feature Importance Analysis
# =========================================================================
print("=" * 70)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

feature_result = analyzer.analyze_feature_importance(df, top_k=20)

print("\nTop 20 Features by Correlation with Fitness:")
print("-" * 50)
for i, row in feature_result.feature_correlations.head(20).iterrows():
    print(f"  {row['feature']:10s} | corr: {row['correlation']:+.4f} | p-value: {row['p_value']:.4f}")

print("\nFeature Selection Frequency (Top 10 in top performers):")
print("-" * 50)
for feat, freq in feature_result.selection_frequency.head(10).items():
    print(f"  {feat:10s} | freq: {freq:.2%}")

# =========================================================================
# 2. Parameter Analysis
# =========================================================================
print("\n" + "=" * 70)
print("PARAMETER ANALYSIS")
print("=" * 70)

param_results = analyzer.analyze_parameters(df)

print("\nParameter Correlations and Bound Analysis:")
print("-" * 80)
print(f"{'Parameter':25s} | {'Corr':>7s} | {'Util':>6s} | {'Proximity':>12s} | Recommended Bounds")
print("-" * 80)

for param_name, result in sorted(param_results.items(), key=lambda x: abs(x[1].fitness_correlation), reverse=True):
    rec = result.recommended_bounds if result.recommended_bounds else result.current_bounds
    print(f"{param_name:25s} | {result.fitness_correlation:+.4f} | {result.utilization:.2%} | {result.bound_proximity:>12s} | ({rec[0]:.4f}, {rec[1]:.4f})")

# =========================================================================
# 3. Evolution Analysis
# =========================================================================
print("\n" + "=" * 70)
print("EVOLUTION ANALYSIS")
print("=" * 70)

evolution_result = analyzer.analyze_evolution(df)

print("\nFitness Progression (Best F1 product per iteration):")
print("-" * 50)
fitness_prog = evolution_result.fitness_progression
# Show every 10 iterations
for i in range(0, len(fitness_prog), 10):
    iter_num = fitness_prog.index[i]
    print(f"  Iteration {iter_num:3d}: {fitness_prog.iloc[i]:.4f}")
print(f"  Iteration {fitness_prog.index[-1]:3d}: {fitness_prog.iloc[-1]:.4f} (final)")

print(f"\nImprovement: {fitness_prog.iloc[0]:.4f} -> {fitness_prog.iloc[-1]:.4f} ({(fitness_prog.iloc[-1] - fitness_prog.iloc[0]) / fitness_prog.iloc[0] * 100:+.1f}%)")

print("\nConverged Parameters (variance reduction >= 70%):")
print(f"  {evolution_result.converged_params if evolution_result.converged_params else 'None'}")

print("\nDiverse Parameters (still exploring):")
print(f"  {evolution_result.diverse_params if evolution_result.diverse_params else 'None'}")

print("\nConvergence Scores:")
print("-" * 50)
for param, score in sorted(evolution_result.convergence_scores.items(), key=lambda x: x[1], reverse=True):
    status = "CONVERGED" if score >= 0.7 else "exploring" if score >= 0.4 else "diverse"
    print(f"  {param:25s} | {score:.3f} ({status})")

# =========================================================================
# 4. Elite Individual Analysis
# =========================================================================
print("\n" + "=" * 70)
print("ELITE INDIVIDUAL ANALYSIS")
print("=" * 70)

# Find a good threshold
percentile_5 = np.percentile(df['fitness'], 5)
print(f"\nUsing fitness threshold: {percentile_5:.4f} (5th percentile)")

elite_result = analyzer.analyze_elite_individuals(df, fitness_threshold=percentile_5, use_absolute=True)

print(f"Elite individuals: {elite_result.n_elite} / {elite_result.n_total} ({elite_result.elite_fraction:.1%})")
print(f"Best fitness found: {elite_result.best_fitness:.4f} (F1 product: {-elite_result.best_fitness:.4f})")

print("\nTop 10 Influential Parameters:")
print("-" * 80)
for i, param in enumerate(elite_result.ranked_parameters[:10], 1):
    influence = elite_result.parameter_influences[param]
    print(f"{i:2d}. {param:25s} | Score: {influence.influence_score:.3f} | Effect: {influence.effect_size:+.3f}")
    print(f"    Elite: {influence.elite_mean:.4f} +/- {influence.elite_std:.4f} | Range: [{influence.elite_range[0]:.4f}, {influence.elite_range[1]:.4f}]")

print("\nTop Feature Selection in Elite Individuals:")
print("-" * 50)
for feat, freq in elite_result.feature_selection_in_elites.head(15).items():
    print(f"  {feat:10s} | {freq:.2%}")

# =========================================================================
# 5. Configuration Recommendations
# =========================================================================
print("\n" + "=" * 70)
print("CONFIGURATION RECOMMENDATIONS")
print("=" * 70)

recommendations = analyzer.generate_recommendations(
    feature_result, param_results, evolution_result, df
)

print(f"\nConfidence Score: {recommendations.confidence:.2f}")

print("\nRecommended Hyperparameter Bounds:")
print("-" * 80)
for param, bounds in recommendations.hyperparam_bounds.items():
    rationale = recommendations.rationale.get(param, "")
    print(f"  {param:25s}: ({bounds[0]:.4f}, {bounds[1]:.4f}) - {rationale}")

print("\nRecommended APO Settings:")
for setting, value in recommendations.apo_settings.items():
    print(f"  {setting}: {value}")

# =========================================================================
# 6. Best Individual Details
# =========================================================================
print("\n" + "=" * 70)
print("BEST INDIVIDUAL DETAILS")
print("=" * 70)

best_idx = df['fitness'].idxmin()
best = df.loc[best_idx]

print(f"\nIteration: {int(best['iteration'])}, Individual: {int(best['individual_idx'])}")
print(f"Fitness: {best['fitness']:.4f} (F1 product: {-best['fitness']:.4f})")

print("\nHyperparameters:")
hyperparam_cols = ['class_weight_power', 'focal_gamma', 'learning_rate', 'dropout',
                   'hidden_size', 'num_layers', 'weight_decay', 'label_smoothing',
                   'batch_size', 'scheduler_patience', 'input_seq_length', 'kernel_size', 'num_conv_layers']
for col in hyperparam_cols:
    if col in best.index:
        print(f"  {col:25s}: {best[col]:.4f}" if isinstance(best[col], float) else f"  {col:25s}: {int(best[col])}")

# Count selected features
feature_cols = [c for c in df.columns if c.startswith('feat_')]
n_features = int(sum(best[feature_cols]))
print(f"\nSelected features: {n_features} / {len(feature_cols)}")

selected_features = [col for col in feature_cols if best[col] == 1]
print(f"Feature indices: {', '.join([f.replace('feat_', '') for f in selected_features])}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
