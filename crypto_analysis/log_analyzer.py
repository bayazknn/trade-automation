"""
LSTM Optimization Log Analyzer

Analyzes optimization CSV logs to provide insights for configuring
the APO optimizer, including feature importance, parameter value
analysis, evolution tracking, and configuration recommendations.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


@dataclass
class FeatureImportanceResult:
    """Results from feature importance analysis."""
    feature_correlations: pd.DataFrame  # Feature -> fitness correlation
    top_features: List[str]             # Top K features by importance
    selection_frequency: pd.Series      # Selection frequency across top performers
    cooccurrence_matrix: pd.DataFrame   # Feature co-occurrence patterns


@dataclass
class ParameterAnalysisResult:
    """Results from parameter value analysis."""
    param_name: str
    fitness_correlation: float          # Spearman correlation with fitness
    optimal_range: Tuple[float, float]  # Range in top performers (10th-90th percentile)
    current_bounds: Tuple[float, float] # Current lower/upper bounds
    bound_proximity: str                # 'lower_bound', 'upper_bound', 'centered', 'narrow'
    recommended_bounds: Optional[Tuple[float, float]]
    utilization: float                  # Fraction of bound range used


@dataclass
class EvolutionAnalysisResult:
    """Results from evolution analysis over iterations."""
    fitness_progression: pd.Series      # Best fitness per iteration
    convergence_scores: Dict[str, float]# Per-parameter convergence (variance reduction)
    feature_stability: pd.DataFrame     # Feature selection consistency over time
    converged_params: List[str]         # Parameters that converged
    diverse_params: List[str]           # Parameters still exploring


@dataclass
class ConfigurationRecommendation:
    """Synthesized configuration recommendations."""
    hyperparam_bounds: Dict[str, Tuple[float, float]]  # New bounds
    apo_settings: Dict[str, Any]        # pop_size, np_neighbors, pf_max, iterations
    feature_suggestions: Dict[str, str] # Feature -> "keep"/"consider_removing"
    confidence: float                   # Overall confidence score
    rationale: Dict[str, str]           # Explanations


@dataclass
class EliteParameterInfluence:
    """Results from elite individual parameter influence analysis."""
    param_name: str
    elite_mean: float                    # Mean value in elite individuals
    population_mean: float               # Mean value in overall population
    elite_std: float                     # Std dev in elite individuals
    population_std: float                # Std dev in overall population
    effect_size: float                   # Cohen's d effect size
    ks_statistic: float                  # Kolmogorov-Smirnov statistic
    ks_pvalue: float                     # KS test p-value
    mann_whitney_u: float                # Mann-Whitney U statistic
    mann_whitney_pvalue: float           # Mann-Whitney p-value
    influence_score: float               # Combined influence score (0-1)
    elite_range: Tuple[float, float]     # (5th, 95th percentile) in elites
    interpretation: str                  # Human-readable interpretation


@dataclass
class EliteAnalysisResult:
    """Complete results from elite individual analysis."""
    fitness_threshold: float             # Threshold used for elite selection
    n_elite: int                         # Number of elite individuals
    n_total: int                         # Total population size
    elite_fraction: float                # Fraction of population that is elite
    best_fitness: float                  # Best fitness value found
    parameter_influences: Dict[str, EliteParameterInfluence]  # Per-param results
    ranked_parameters: List[str]         # Parameters ranked by influence
    feature_selection_in_elites: pd.Series  # Feature selection rates in elites
    elite_individuals: pd.DataFrame      # The actual elite individuals


@dataclass
class AnalysisReport:
    """Complete analysis report."""
    run_ids: List[str]
    total_individuals: int
    total_iterations: int
    feature_importance: FeatureImportanceResult
    parameter_analysis: Dict[str, ParameterAnalysisResult]
    evolution: EvolutionAnalysisResult
    recommendations: ConfigurationRecommendation
    metadata: Dict[str, Any] = field(default_factory=dict)


class LSTMLogAnalyzer:
    """
    Analyzer for LSTM optimization CSV logs.

    Analyzes optimization runs and provides insights for configuring
    the APO optimizer, including feature importance, parameter value
    analysis, evolution tracking, and configuration recommendations.

    Parameters
    ----------
    log_dir : str or Path
        Directory containing optimization log CSV files
    top_percentile : float
        Percentile threshold for "top performers" (default: 10)
        Lower fitness values are better (negated F1 product)

    Examples
    --------
    >>> analyzer = LSTMLogAnalyzer("optimization_logs")
    >>> report = analyzer.analyze_all()
    >>> analyzer.generate_report(report, "analysis_report.md")
    >>> analyzer.plot_parameter_distributions(report.parameter_analysis)
    """

    # Hyperparameter names from LSTM optimizer
    HYPERPARAM_NAMES = [
        'class_weight_power', 'focal_gamma', 'learning_rate', 'dropout',
        'hidden_size', 'num_layers', 'weight_decay', 'label_smoothing',
        'batch_size', 'all_hold_weight', 'entry_exit_weight',
        'scheduler_patience', 'input_seq_length'
    ]

    # Optimizer settings columns
    OPTIMIZER_SETTINGS = [
        'pop_size', 'iterations', 'n_workers', 'min_features',
        'epochs_per_eval', 'np_neighbors', 'pf_max',
        'elitist_selection', 'elitist_constant'
    ]

    # Feature column pattern
    FEATURE_PATTERN = re.compile(r'^feat_\d+$')

    def __init__(
        self,
        log_dir: str | Path,
        top_percentile: float = 10.0,
    ):
        self.log_dir = Path(log_dir)
        self.top_percentile = top_percentile
        self._df: Optional[pd.DataFrame] = None
        self._feature_columns: Optional[List[str]] = None

    def load_logs(
        self,
        run_ids: Optional[List[str]] = None,
        exclude_initial: bool = True,
    ) -> pd.DataFrame:
        """
        Load optimization logs from CSV files.

        Parameters
        ----------
        run_ids : list of str, optional
            Specific run IDs to load. If None, loads all.
        exclude_initial : bool
            Whether to exclude iteration -1 (initial population)

        Returns
        -------
        pd.DataFrame
            Combined log data from all runs
        """
        if not self.log_dir.exists():
            raise FileNotFoundError(f"Log directory not found: {self.log_dir}")

        # Find CSV files
        if run_ids:
            csv_files = [self.log_dir / f"run_{rid}.csv" for rid in run_ids]
            csv_files = [f for f in csv_files if f.exists()]
        else:
            csv_files = list(self.log_dir.glob("run_*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No log files found in {self.log_dir}")

        # Load and combine
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)

        # Exclude initial population if requested
        if exclude_initial:
            combined_df = combined_df[combined_df['iteration'] >= 0]

        # Identify feature columns
        self._feature_columns = [
            col for col in combined_df.columns
            if self.FEATURE_PATTERN.match(col)
        ]

        self._df = combined_df
        return combined_df

    def get_top_performers(
        self,
        df: Optional[pd.DataFrame] = None,
        percentile: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Get individuals with best fitness (top percentile).

        Lower fitness is better (negative F1 product).
        """
        if df is None:
            df = self._df
        if percentile is None:
            percentile = self.top_percentile

        threshold = np.percentile(df['fitness'], percentile)
        return df[df['fitness'] <= threshold]

    def _get_positive_fitness(self, df: pd.DataFrame) -> pd.Series:
        """Convert negative fitness to positive F1 product."""
        return -df['fitness']

    # =========================================================================
    # Feature Importance Analysis
    # =========================================================================

    def analyze_feature_importance(
        self,
        df: Optional[pd.DataFrame] = None,
        top_k: int = 20,
    ) -> FeatureImportanceResult:
        """
        Analyze which features correlate with higher fitness.

        Performs:
        1. Point-biserial correlation between feature selection and fitness
        2. Feature selection frequency across top performers
        3. Co-occurrence analysis of feature pairs

        Returns
        -------
        FeatureImportanceResult
            Feature importance analysis results
        """
        if df is None:
            df = self._df
        if df is None:
            raise ValueError("No data loaded. Call load_logs() first.")

        feature_cols = self._feature_columns
        positive_fitness = self._get_positive_fitness(df)

        # 1. Point-biserial correlation
        correlations = []
        for col in feature_cols:
            try:
                corr, p_value = stats.pointbiserialr(df[col], positive_fitness)
                correlations.append({
                    'feature': col,
                    'correlation': corr,
                    'p_value': p_value,
                    'abs_correlation': abs(corr),
                })
            except Exception:
                correlations.append({
                    'feature': col,
                    'correlation': 0.0,
                    'p_value': 1.0,
                    'abs_correlation': 0.0,
                })

        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('abs_correlation', ascending=False)

        # 2. Selection frequency in top performers
        top_df = self.get_top_performers(df)
        selection_freq = top_df[feature_cols].mean().sort_values(ascending=False)

        # Top K features by absolute correlation
        top_features = corr_df.head(top_k)['feature'].tolist()

        # 3. Co-occurrence matrix for top features
        top_feat_cols = corr_df.head(top_k)['feature'].tolist()
        cooc_df = df[top_feat_cols]
        cooccurrence = cooc_df.T @ cooc_df
        # Normalize by diagonal (self-occurrence)
        diag = np.diag(cooccurrence.values)
        diag_safe = np.where(diag > 0, diag, 1)
        cooccurrence_norm = cooccurrence / np.sqrt(np.outer(diag_safe, diag_safe))

        return FeatureImportanceResult(
            feature_correlations=corr_df,
            top_features=top_features,
            selection_frequency=selection_freq,
            cooccurrence_matrix=cooccurrence_norm,
        )

    # =========================================================================
    # Parameter Value Analysis
    # =========================================================================

    def analyze_parameters(
        self,
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, ParameterAnalysisResult]:
        """
        Analyze which parameter values correlate with higher fitness.

        For each LSTM hyperparameter:
        1. Spearman correlation of parameter value with fitness
        2. Distribution of values in top performers
        3. Proximity to current bounds (lower/upper)
        4. Recommended new bounds based on analysis

        Returns
        -------
        dict
            Parameter name -> ParameterAnalysisResult
        """
        if df is None:
            df = self._df
        if df is None:
            raise ValueError("No data loaded. Call load_logs() first.")

        top_df = self.get_top_performers(df)
        positive_fitness = self._get_positive_fitness(df)

        results = {}
        for param_name in self.HYPERPARAM_NAMES:
            if param_name not in df.columns:
                continue

            result = self._analyze_single_parameter(
                df, param_name, top_df, positive_fitness
            )
            results[param_name] = result

        return results

    def _analyze_single_parameter(
        self,
        df: pd.DataFrame,
        param_name: str,
        top_df: pd.DataFrame,
        positive_fitness: pd.Series,
    ) -> ParameterAnalysisResult:
        """Analyze a single parameter's relationship to fitness."""
        param_values = df[param_name]
        top_values = top_df[param_name]

        # Get bounds from columns
        lower_col = f"{param_name}_lower"
        upper_col = f"{param_name}_upper"
        current_lower = df[lower_col].iloc[0] if lower_col in df.columns else param_values.min()
        current_upper = df[upper_col].iloc[0] if upper_col in df.columns else param_values.max()

        # Spearman correlation
        try:
            corr, _ = stats.spearmanr(param_values, positive_fitness)
        except Exception:
            corr = 0.0

        # Optimal range (10th-90th percentile of top performers)
        optimal_lower = top_values.quantile(0.1)
        optimal_upper = top_values.quantile(0.9)

        # Bound proximity
        proximity = self._calculate_bound_proximity(
            top_values, current_lower, current_upper
        )

        # Utilization
        utilization = self._calculate_utilization(
            top_values, current_lower, current_upper
        )

        # Recommended bounds
        recommended = self._recommend_new_bounds(
            top_values, current_lower, current_upper, proximity, utilization
        )

        return ParameterAnalysisResult(
            param_name=param_name,
            fitness_correlation=corr,
            optimal_range=(optimal_lower, optimal_upper),
            current_bounds=(current_lower, current_upper),
            bound_proximity=proximity,
            recommended_bounds=recommended,
            utilization=utilization,
        )

    def _calculate_bound_proximity(
        self,
        values: pd.Series,
        lower: float,
        upper: float,
        edge_threshold: float = 0.2,
        cluster_threshold: float = 0.3,
    ) -> str:
        """
        Determine if optimal values cluster near bounds.

        Returns:
        - 'lower_bound': >30% of values in bottom 20% of range
        - 'upper_bound': >30% of values in top 20% of range
        - 'narrow': values clustered in small region (< 30% of range)
        - 'centered': values well-distributed
        """
        if upper <= lower:
            return 'centered'

        range_size = upper - lower
        edge_size = range_size * edge_threshold

        # Calculate fractions in each region
        in_lower = (values < lower + edge_size).mean()
        in_upper = (values > upper - edge_size).mean()

        # Check if values are clustered narrowly
        value_range = values.quantile(0.9) - values.quantile(0.1)
        if value_range < range_size * 0.3:
            return 'narrow'

        if in_lower > cluster_threshold:
            return 'lower_bound'
        if in_upper > cluster_threshold:
            return 'upper_bound'

        return 'centered'

    def _calculate_utilization(
        self,
        values: pd.Series,
        lower: float,
        upper: float,
    ) -> float:
        """Calculate what fraction of bound range is utilized."""
        if upper <= lower:
            return 1.0

        actual_range = values.quantile(0.9) - values.quantile(0.1)
        bound_range = upper - lower
        return min(actual_range / bound_range, 1.0)

    def _recommend_new_bounds(
        self,
        values: pd.Series,
        current_lower: float,
        current_upper: float,
        proximity: str,
        utilization: float,
        margin_factor: float = 0.2,
    ) -> Optional[Tuple[float, float]]:
        """Recommend new bounds based on value distribution."""
        range_size = current_upper - current_lower
        margin = range_size * margin_factor

        optimal_lower = values.quantile(0.05)
        optimal_upper = values.quantile(0.95)

        if proximity == 'lower_bound':
            # Expand lower bound
            new_lower = current_lower - 0.5 * range_size
            return (new_lower, current_upper)

        elif proximity == 'upper_bound':
            # Expand upper bound
            new_upper = current_upper + 0.5 * range_size
            return (current_lower, new_upper)

        elif utilization < 0.3:
            # Narrow bounds to actual range + margin
            new_lower = max(optimal_lower - margin, current_lower * 0.5)
            new_upper = min(optimal_upper + margin, current_upper * 1.5)
            return (new_lower, new_upper)

        return None  # No change recommended

    # =========================================================================
    # Evolution Analysis
    # =========================================================================

    def analyze_evolution(
        self,
        df: Optional[pd.DataFrame] = None,
        convergence_threshold: float = 0.7,
    ) -> EvolutionAnalysisResult:
        """
        Analyze how solutions evolve over iterations.

        Tracks:
        1. Fitness progression over iterations
        2. Parameter convergence (decreasing variance)
        3. Feature selection stability
        4. Identification of converged vs diverse parameters

        Returns
        -------
        EvolutionAnalysisResult
            Evolution analysis results
        """
        if df is None:
            df = self._df
        if df is None:
            raise ValueError("No data loaded. Call load_logs() first.")

        # 1. Fitness progression (best per iteration)
        fitness_progression = df.groupby('iteration')['fitness'].min()
        fitness_progression = -fitness_progression  # Convert to positive

        # 2. Parameter convergence
        convergence_scores = {}
        for param_name in self.HYPERPARAM_NAMES:
            if param_name not in df.columns:
                continue
            convergence_scores[param_name] = self._calculate_convergence(
                df, param_name
            )

        # 3. Feature stability
        feature_stability = self._calculate_feature_stability(df)

        # 4. Identify converged vs diverse
        converged = [p for p, s in convergence_scores.items() if s >= convergence_threshold]
        diverse = [p for p, s in convergence_scores.items() if s < convergence_threshold]

        return EvolutionAnalysisResult(
            fitness_progression=fitness_progression,
            convergence_scores=convergence_scores,
            feature_stability=feature_stability,
            converged_params=converged,
            diverse_params=diverse,
        )

    def _calculate_convergence(
        self,
        df: pd.DataFrame,
        param_name: str,
    ) -> float:
        """
        Calculate convergence score for a parameter.

        Score = 1 - (variance_final / variance_initial)
        High score = converged, Low score = still exploring
        """
        iterations = df['iteration'].unique()
        if len(iterations) < 2:
            return 0.0

        min_iter = iterations.min()
        max_iter = iterations.max()

        initial_var = df[df['iteration'] == min_iter][param_name].var()
        final_var = df[df['iteration'] == max_iter][param_name].var()

        if initial_var == 0:
            return 1.0 if final_var == 0 else 0.0

        return max(0.0, 1.0 - (final_var / initial_var))

    def _calculate_feature_stability(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate feature selection stability over iterations."""
        feature_cols = self._feature_columns

        stability_data = []
        iterations = sorted(df['iteration'].unique())

        for feat in feature_cols:
            selection_rates = []
            for iter_num in iterations:
                iter_df = df[df['iteration'] == iter_num]
                rate = iter_df[feat].mean()
                selection_rates.append(rate)

            # Stability = 1 - variance of selection rate
            stability = 1.0 - np.var(selection_rates) if len(selection_rates) > 1 else 1.0
            stability_data.append({
                'feature': feat,
                'stability': stability,
                'mean_selection': np.mean(selection_rates),
            })

        return pd.DataFrame(stability_data)

    # =========================================================================
    # Elite Individual Analysis
    # =========================================================================

    def analyze_elite_individuals(
        self,
        df: Optional[pd.DataFrame] = None,
        fitness_threshold: float = 0.05,
        use_absolute: bool = True,
    ) -> EliteAnalysisResult:
        """
        Analyze individuals with exceptional fitness (below threshold).

        Investigates which parameters have the most influence on achieving
        elite-level fitness scores. Uses multiple statistical tests to
        determine parameter importance.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Data to analyze. If None, uses loaded data.
        fitness_threshold : float
            Fitness value threshold for elite selection (default: 0.05).
            Individuals with fitness <= threshold are considered elite.
        use_absolute : bool
            If True, use absolute fitness value for threshold.
            If False, use as percentile (e.g., 0.05 = top 5%).

        Returns
        -------
        EliteAnalysisResult
            Complete analysis of elite individuals and parameter influences.

        Examples
        --------
        >>> analyzer = LSTMLogAnalyzer("optimization_logs")
        >>> analyzer.load_logs()
        >>> elite_result = analyzer.analyze_elite_individuals(fitness_threshold=0.05)
        >>> print(f"Found {elite_result.n_elite} elite individuals")
        >>> for param in elite_result.ranked_parameters[:5]:
        ...     influence = elite_result.parameter_influences[param]
        ...     print(f"{param}: influence={influence.influence_score:.3f}")
        """
        if df is None:
            df = self._df
        if df is None:
            raise ValueError("No data loaded. Call load_logs() first.")

        # Determine elite threshold
        if use_absolute:
            actual_threshold = fitness_threshold
        else:
            # Use as percentile
            actual_threshold = np.percentile(df['fitness'], fitness_threshold * 100)

        # Select elite individuals
        elite_df = df[df['fitness'] <= actual_threshold].copy()
        non_elite_df = df[df['fitness'] > actual_threshold].copy()

        if len(elite_df) == 0:
            raise ValueError(
                f"No individuals found with fitness <= {actual_threshold}. "
                f"Fitness range: [{df['fitness'].min():.4f}, {df['fitness'].max():.4f}]"
            )

        # Analyze each parameter
        parameter_influences = {}
        for param_name in self.HYPERPARAM_NAMES:
            if param_name not in df.columns:
                continue

            influence = self._analyze_elite_parameter_influence(
                param_name, elite_df, non_elite_df, df
            )
            parameter_influences[param_name] = influence

        # Rank parameters by influence score
        ranked_params = sorted(
            parameter_influences.keys(),
            key=lambda p: parameter_influences[p].influence_score,
            reverse=True
        )

        # Feature selection rates in elites
        feature_cols = self._feature_columns or []
        feature_selection = elite_df[feature_cols].mean().sort_values(ascending=False)

        return EliteAnalysisResult(
            fitness_threshold=actual_threshold,
            n_elite=len(elite_df),
            n_total=len(df),
            elite_fraction=len(elite_df) / len(df),
            best_fitness=df['fitness'].min(),
            parameter_influences=parameter_influences,
            ranked_parameters=ranked_params,
            feature_selection_in_elites=feature_selection,
            elite_individuals=elite_df,
        )

    def _analyze_elite_parameter_influence(
        self,
        param_name: str,
        elite_df: pd.DataFrame,
        non_elite_df: pd.DataFrame,
        full_df: pd.DataFrame,
    ) -> EliteParameterInfluence:
        """
        Analyze how a single parameter influences elite status.

        Uses multiple statistical measures:
        - Cohen's d effect size
        - Kolmogorov-Smirnov test (distribution difference)
        - Mann-Whitney U test (rank-based comparison)
        """
        elite_values = elite_df[param_name].dropna()
        non_elite_values = non_elite_df[param_name].dropna()
        all_values = full_df[param_name].dropna()

        # Basic statistics
        elite_mean = elite_values.mean()
        elite_std = elite_values.std()
        pop_mean = all_values.mean()
        pop_std = all_values.std()

        # Cohen's d effect size
        pooled_std = np.sqrt(
            ((len(elite_values) - 1) * elite_std**2 +
             (len(non_elite_values) - 1) * non_elite_values.std()**2) /
            (len(elite_values) + len(non_elite_values) - 2)
        )
        if pooled_std > 0:
            effect_size = (elite_mean - non_elite_values.mean()) / pooled_std
        else:
            effect_size = 0.0

        # Kolmogorov-Smirnov test
        try:
            ks_stat, ks_pvalue = stats.ks_2samp(elite_values, non_elite_values)
        except Exception:
            ks_stat, ks_pvalue = 0.0, 1.0

        # Mann-Whitney U test
        try:
            mw_stat, mw_pvalue = stats.mannwhitneyu(
                elite_values, non_elite_values, alternative='two-sided'
            )
        except Exception:
            mw_stat, mw_pvalue = 0.0, 1.0

        # Calculate influence score (0-1)
        # Combine effect size, KS statistic, and significance
        abs_effect = min(abs(effect_size), 2.0) / 2.0  # Normalize to 0-1
        significance = 1.0 - min(ks_pvalue, mw_pvalue)  # Higher = more significant
        influence_score = (abs_effect * 0.4 + ks_stat * 0.3 + significance * 0.3)

        # Elite range
        elite_range = (
            elite_values.quantile(0.05),
            elite_values.quantile(0.95)
        )

        # Generate interpretation
        interpretation = self._interpret_parameter_influence(
            param_name, effect_size, ks_stat, ks_pvalue, elite_mean, pop_mean,
            elite_range
        )

        return EliteParameterInfluence(
            param_name=param_name,
            elite_mean=elite_mean,
            population_mean=pop_mean,
            elite_std=elite_std,
            population_std=pop_std,
            effect_size=effect_size,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pvalue,
            mann_whitney_u=mw_stat,
            mann_whitney_pvalue=mw_pvalue,
            influence_score=influence_score,
            elite_range=elite_range,
            interpretation=interpretation,
        )

    def _interpret_parameter_influence(
        self,
        param_name: str,
        effect_size: float,
        ks_stat: float,
        ks_pvalue: float,
        elite_mean: float,
        pop_mean: float,
        elite_range: Tuple[float, float],
    ) -> str:
        """Generate human-readable interpretation of parameter influence."""
        parts = []

        # Effect size interpretation
        abs_effect = abs(effect_size)
        if abs_effect >= 0.8:
            effect_desc = "very large"
        elif abs_effect >= 0.5:
            effect_desc = "medium"
        elif abs_effect >= 0.2:
            effect_desc = "small"
        else:
            effect_desc = "negligible"

        direction = "higher" if effect_size > 0 else "lower"
        parts.append(f"{effect_desc} effect ({direction} in elites)")

        # Significance
        if ks_pvalue < 0.01:
            parts.append("highly significant (p<0.01)")
        elif ks_pvalue < 0.05:
            parts.append("significant (p<0.05)")
        else:
            parts.append("not significant")

        # Distribution difference
        if ks_stat > 0.3:
            parts.append("distinct distribution")
        elif ks_stat > 0.15:
            parts.append("moderate distribution shift")

        # Optimal range suggestion
        parts.append(f"elite range: [{elite_range[0]:.4f}, {elite_range[1]:.4f}]")

        return "; ".join(parts)

    def plot_elite_analysis(
        self,
        result: EliteAnalysisResult,
        top_k: int = 8,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot comprehensive elite analysis visualization.

        Creates a multi-panel figure showing:
        1. Parameter influence ranking
        2. Top parameter distributions (elite vs population)
        3. Feature selection in elites
        4. Elite fitness distribution
        """
        fig = plt.figure(figsize=figsize)

        # Create grid: 2 rows, 3 columns
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Panel 1: Parameter influence ranking (full height left column)
        ax1 = fig.add_subplot(gs[:, 0])
        params = result.ranked_parameters
        influences = [result.parameter_influences[p].influence_score for p in params]

        colors = ['#2ecc71' if i >= 0.5 else '#f39c12' if i >= 0.3 else '#e74c3c'
                  for i in influences]
        ax1.barh(params, influences, color=colors)
        ax1.set_xlabel('Influence Score')
        ax1.set_title('Parameter Influence on Elite Status')
        ax1.axvline(0.5, color='green', linestyle='--', alpha=0.5, label='High influence')
        ax1.axvline(0.3, color='orange', linestyle='--', alpha=0.5, label='Medium')
        ax1.invert_yaxis()
        ax1.legend(fontsize=8)

        # Panel 2 & 3: Top parameter distributions
        top_params = result.ranked_parameters[:min(top_k, 4)]
        for idx, param in enumerate(top_params):
            row = idx // 2
            col = 1 + idx % 2
            ax = fig.add_subplot(gs[row, col])

            influence = result.parameter_influences[param]
            elite_vals = result.elite_individuals[param]
            all_vals = self._df[param] if self._df is not None else elite_vals

            # Histogram comparison
            ax.hist(all_vals, bins=25, alpha=0.4, label='All', color='gray', density=True)
            ax.hist(elite_vals, bins=25, alpha=0.7, label='Elite', color='green', density=True)

            # Mark elite range
            ax.axvline(influence.elite_range[0], color='darkgreen', linestyle='--', linewidth=2)
            ax.axvline(influence.elite_range[1], color='darkgreen', linestyle='--', linewidth=2)

            ax.set_title(f'{param}\nInfluence: {influence.influence_score:.3f}, Effect: {influence.effect_size:.2f}')
            ax.legend(fontsize=8)

        plt.suptitle(
            f'Elite Analysis: {result.n_elite} individuals with fitness <= {result.fitness_threshold:.4f}\n'
            f'({result.elite_fraction*100:.1f}% of population, best fitness: {result.best_fitness:.4f})',
            fontsize=12, fontweight='bold'
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def print_elite_summary(
        self,
        result: EliteAnalysisResult,
        top_k: int = 10,
    ) -> str:
        """
        Generate and print a text summary of elite analysis.

        Returns the summary string for further use.
        """
        lines = [
            "=" * 70,
            "ELITE INDIVIDUAL ANALYSIS SUMMARY",
            "=" * 70,
            f"Fitness Threshold: <= {result.fitness_threshold:.4f}",
            f"Elite Individuals: {result.n_elite} / {result.n_total} ({result.elite_fraction*100:.1f}%)",
            f"Best Fitness Found: {result.best_fitness:.4f}",
            "",
            "TOP INFLUENTIAL PARAMETERS (ranked by influence score):",
            "-" * 70,
        ]

        for i, param in enumerate(result.ranked_parameters[:top_k], 1):
            influence = result.parameter_influences[param]
            lines.append(
                f"{i:2d}. {param:25s} | Score: {influence.influence_score:.3f} | "
                f"Effect: {influence.effect_size:+.3f} | "
                f"KS: {influence.ks_statistic:.3f} (p={influence.ks_pvalue:.4f})"
            )
            lines.append(f"    Elite: {influence.elite_mean:.4f} +/- {influence.elite_std:.4f} "
                        f"| Pop: {influence.population_mean:.4f} +/- {influence.population_std:.4f}")
            lines.append(f"    Range: [{influence.elite_range[0]:.4f}, {influence.elite_range[1]:.4f}]")
            lines.append(f"    -> {influence.interpretation}")
            lines.append("")

        lines.extend([
            "-" * 70,
            "RECOMMENDED PARAMETER VALUES FOR ELITE PERFORMANCE:",
            "-" * 70,
        ])

        # Show recommended values for top influential parameters
        for param in result.ranked_parameters[:5]:
            influence = result.parameter_influences[param]
            if influence.influence_score >= 0.3:
                lines.append(
                    f"  {param}: target range [{influence.elite_range[0]:.4f}, "
                    f"{influence.elite_range[1]:.4f}]"
                )

        lines.append("=" * 70)

        summary = "\n".join(lines)
        print(summary)
        return summary

    # =========================================================================
    # Configuration Recommendations
    # =========================================================================

    def generate_recommendations(
        self,
        feature_result: FeatureImportanceResult,
        param_results: Dict[str, ParameterAnalysisResult],
        evolution_result: EvolutionAnalysisResult,
        df: Optional[pd.DataFrame] = None,
    ) -> ConfigurationRecommendation:
        """
        Synthesize all analyses into actionable recommendations.

        Returns
        -------
        ConfigurationRecommendation
            Complete configuration recommendations
        """
        if df is None:
            df = self._df

        # 1. Hyperparameter bounds
        hyperparam_bounds = {}
        rationale = {}

        for param_name, result in param_results.items():
            if result.recommended_bounds:
                hyperparam_bounds[param_name] = result.recommended_bounds
                rationale[param_name] = f"Proximity: {result.bound_proximity}, Utilization: {result.utilization:.2f}"
            else:
                hyperparam_bounds[param_name] = result.current_bounds
                rationale[param_name] = "No change needed"

        # 2. APO settings recommendations
        apo_settings = self._recommend_apo_settings(evolution_result, df)

        # 3. Feature suggestions
        feature_suggestions = self._recommend_features(feature_result)

        # 4. Confidence score
        confidence = self._calculate_confidence(
            feature_result, param_results, evolution_result
        )

        return ConfigurationRecommendation(
            hyperparam_bounds=hyperparam_bounds,
            apo_settings=apo_settings,
            feature_suggestions=feature_suggestions,
            confidence=confidence,
            rationale=rationale,
        )

    def _recommend_apo_settings(
        self,
        evolution_result: EvolutionAnalysisResult,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Recommend APO optimizer settings."""
        # Get current settings
        current_settings = {
            'pop_size': int(df['pop_size'].iloc[0]) if 'pop_size' in df.columns else 10,
            'iterations': int(df['iterations'].iloc[0]) if 'iterations' in df.columns else 50,
            'np_neighbors': int(df['np_neighbors'].iloc[0]) if 'np_neighbors' in df.columns else 1,
            'pf_max': float(df['pf_max'].iloc[0]) if 'pf_max' in df.columns else 0.1,
        }

        recommendations = current_settings.copy()

        # Analyze fitness progression
        fitness_prog = evolution_result.fitness_progression
        if len(fitness_prog) > 5:
            # Check if fitness is still improving
            late_improvement = fitness_prog.iloc[-5:].max() - fitness_prog.iloc[-5:].min()
            early_improvement = fitness_prog.iloc[:5].max() - fitness_prog.iloc[:5].min()

            if late_improvement > early_improvement * 0.1:
                # Still improving - suggest more iterations
                recommendations['iterations'] = int(current_settings['iterations'] * 1.5)

        # Check convergence - if many params converged early, might need more diversity
        convergence_rate = len(evolution_result.converged_params) / len(evolution_result.convergence_scores)
        if convergence_rate > 0.8:
            # High convergence - increase exploration
            recommendations['pf_max'] = min(0.2, current_settings['pf_max'] * 1.5)

        return recommendations

    def _recommend_features(
        self,
        feature_result: FeatureImportanceResult,
    ) -> Dict[str, str]:
        """Recommend features to keep or consider removing."""
        suggestions = {}

        corr_df = feature_result.feature_correlations
        freq = feature_result.selection_frequency

        for _, row in corr_df.iterrows():
            feat = row['feature']
            corr = row['abs_correlation']
            sel_freq = freq.get(feat, 0)

            if corr > 0.1 and sel_freq > 0.5:
                suggestions[feat] = 'keep'
            elif corr < 0.02 and sel_freq < 0.2:
                suggestions[feat] = 'consider_removing'
            else:
                suggestions[feat] = 'neutral'

        return suggestions

    def _calculate_confidence(
        self,
        feature_result: FeatureImportanceResult,
        param_results: Dict[str, ParameterAnalysisResult],
        evolution_result: EvolutionAnalysisResult,
    ) -> float:
        """Calculate overall confidence in recommendations."""
        scores = []

        # Feature analysis confidence (based on correlation strength)
        max_corr = feature_result.feature_correlations['abs_correlation'].max()
        scores.append(min(max_corr * 5, 1.0))  # Scale correlation to 0-1

        # Parameter analysis confidence (based on utilization consistency)
        utilizations = [r.utilization for r in param_results.values()]
        util_consistency = 1.0 - np.std(utilizations) if utilizations else 0.5
        scores.append(util_consistency)

        # Evolution confidence (based on fitness improvement)
        fitness_prog = evolution_result.fitness_progression
        if len(fitness_prog) > 1:
            improvement = (fitness_prog.iloc[-1] - fitness_prog.iloc[0]) / abs(fitness_prog.iloc[0] + 1e-10)
            scores.append(min(improvement, 1.0))

        return np.mean(scores)

    # =========================================================================
    # High-Level Analysis Methods
    # =========================================================================

    def analyze_all(
        self,
        run_ids: Optional[List[str]] = None,
    ) -> AnalysisReport:
        """
        Run complete analysis pipeline.

        Parameters
        ----------
        run_ids : list of str, optional
            Run IDs to analyze. If None, analyzes all available.

        Returns
        -------
        AnalysisReport
            Complete analysis report
        """
        df = self.load_logs(run_ids)

        feature_result = self.analyze_feature_importance(df)
        param_results = self.analyze_parameters(df)
        evolution_result = self.analyze_evolution(df)
        recommendations = self.generate_recommendations(
            feature_result, param_results, evolution_result, df
        )

        return AnalysisReport(
            run_ids=df['run_id'].unique().tolist(),
            total_individuals=len(df),
            total_iterations=df['iteration'].nunique(),
            feature_importance=feature_result,
            parameter_analysis=param_results,
            evolution=evolution_result,
            recommendations=recommendations,
            metadata={
                'top_percentile': self.top_percentile,
                'log_dir': str(self.log_dir),
            },
        )

    # =========================================================================
    # Visualization Methods
    # =========================================================================

    def plot_feature_importance(
        self,
        result: FeatureImportanceResult,
        top_k: int = 20,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
    ) -> None:
        """Plot feature importance bar chart."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Correlation bar chart
        top_corr = result.feature_correlations.head(top_k)
        ax1 = axes[0]
        colors = ['green' if c > 0 else 'red' for c in top_corr['correlation']]
        ax1.barh(top_corr['feature'], top_corr['correlation'], color=colors)
        ax1.set_xlabel('Correlation with Fitness')
        ax1.set_title('Feature Correlation with Fitness')
        ax1.invert_yaxis()

        # Selection frequency
        top_freq = result.selection_frequency.head(top_k)
        ax2 = axes[1]
        ax2.barh(top_freq.index, top_freq.values, color='steelblue')
        ax2.set_xlabel('Selection Frequency')
        ax2.set_title('Feature Selection Frequency (Top Performers)')
        ax2.invert_yaxis()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_parameter_distributions(
        self,
        param_results: Dict[str, ParameterAnalysisResult],
        df: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (15, 12),
        save_path: Optional[str] = None,
    ) -> None:
        """Plot parameter value distributions with bounds."""
        if df is None:
            df = self._df

        n_params = len(param_results)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        top_df = self.get_top_performers(df)

        for idx, (param_name, result) in enumerate(param_results.items()):
            ax = axes[idx]

            # Histogram of all values
            ax.hist(df[param_name], bins=30, alpha=0.5, label='All', color='gray')
            # Histogram of top performers
            ax.hist(top_df[param_name], bins=30, alpha=0.7, label='Top', color='green')

            # Mark bounds
            ax.axvline(result.current_bounds[0], color='red', linestyle='--', label='Current Bounds')
            ax.axvline(result.current_bounds[1], color='red', linestyle='--')

            if result.recommended_bounds:
                ax.axvline(result.recommended_bounds[0], color='blue', linestyle=':', label='Recommended')
                ax.axvline(result.recommended_bounds[1], color='blue', linestyle=':')

            ax.set_title(f'{param_name}\nCorr: {result.fitness_correlation:.3f}')
            ax.legend(fontsize=8)

        # Hide unused subplots
        for idx in range(len(param_results), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_fitness_evolution(
        self,
        evolution_result: EvolutionAnalysisResult,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
    ) -> None:
        """Plot fitness progression over iterations."""
        fig, ax = plt.subplots(figsize=figsize)

        fitness = evolution_result.fitness_progression
        ax.plot(fitness.index, fitness.values, marker='o', linewidth=2)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Fitness (F1 Product)')
        ax.set_title('Fitness Progression Over Iterations')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_parameter_convergence(
        self,
        evolution_result: EvolutionAnalysisResult,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
    ) -> None:
        """Plot parameter convergence scores."""
        fig, ax = plt.subplots(figsize=figsize)

        scores = evolution_result.convergence_scores
        params = list(scores.keys())
        values = list(scores.values())

        colors = ['green' if v >= 0.7 else 'orange' if v >= 0.4 else 'red' for v in values]
        ax.barh(params, values, color=colors)

        ax.axvline(0.7, color='green', linestyle='--', alpha=0.5, label='Converged threshold')
        ax.set_xlabel('Convergence Score')
        ax.set_title('Parameter Convergence')
        ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_bound_utilization(
        self,
        param_results: Dict[str, ParameterAnalysisResult],
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
    ) -> None:
        """Plot bound utilization for each parameter."""
        fig, ax = plt.subplots(figsize=figsize)

        params = list(param_results.keys())
        utilizations = [r.utilization for r in param_results.values()]

        colors = ['green' if u > 0.5 else 'orange' if u > 0.3 else 'red' for u in utilizations]
        ax.barh(params, utilizations, color=colors)

        ax.axvline(0.3, color='red', linestyle='--', alpha=0.5, label='Low utilization')
        ax.axvline(0.5, color='orange', linestyle='--', alpha=0.5, label='Medium')
        ax.set_xlabel('Utilization (fraction of bound range used)')
        ax.set_title('Parameter Bound Utilization')
        ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_cooccurrence(
        self,
        result: FeatureImportanceResult,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None,
    ) -> None:
        """Plot feature co-occurrence heatmap."""
        fig, ax = plt.subplots(figsize=figsize)

        if HAS_SEABORN:
            sns.heatmap(
                result.cooccurrence_matrix,
                cmap='RdYlGn',
                center=0.5,
                ax=ax,
                xticklabels=True,
                yticklabels=True,
            )
        else:
            # Fallback to matplotlib imshow
            im = ax.imshow(result.cooccurrence_matrix.values, cmap='RdYlGn', aspect='auto')
            ax.set_xticks(range(len(result.cooccurrence_matrix.columns)))
            ax.set_yticks(range(len(result.cooccurrence_matrix.index)))
            ax.set_xticklabels(result.cooccurrence_matrix.columns, rotation=90)
            ax.set_yticklabels(result.cooccurrence_matrix.index)
            plt.colorbar(im, ax=ax)

        ax.set_title('Feature Co-occurrence (Top Features)')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    # =========================================================================
    # Report Generation
    # =========================================================================

    def generate_report(
        self,
        report: AnalysisReport,
        output_path: str,
        format: str = 'markdown',
    ) -> None:
        """
        Generate human-readable analysis report.

        Parameters
        ----------
        report : AnalysisReport
            Analysis results to format
        output_path : str
            Path to save report
        format : str
            Output format: 'markdown' or 'json'
        """
        if format == 'markdown':
            content = self._generate_markdown_report(report)
        elif format == 'json':
            content = self._generate_json_report(report)
        else:
            raise ValueError(f"Unknown format: {format}")

        with open(output_path, 'w') as f:
            f.write(content)

    def _generate_markdown_report(self, report: AnalysisReport) -> str:
        """Generate markdown-formatted report."""
        lines = [
            "# LSTM Optimization Analysis Report\n",
            f"## Summary",
            f"- **Run IDs**: {', '.join(report.run_ids)}",
            f"- **Total Individuals**: {report.total_individuals}",
            f"- **Total Iterations**: {report.total_iterations}",
            f"- **Confidence Score**: {report.recommendations.confidence:.2f}\n",

            "## Feature Importance",
            "### Top 10 Features by Correlation",
        ]

        top_features = report.feature_importance.feature_correlations.head(10)
        lines.append("| Feature | Correlation | P-Value |")
        lines.append("|---------|-------------|---------|")
        for _, row in top_features.iterrows():
            lines.append(f"| {row['feature']} | {row['correlation']:.4f} | {row['p_value']:.4f} |")

        lines.extend([
            "\n## Parameter Analysis",
            "| Parameter | Correlation | Current Bounds | Recommended | Utilization |",
            "|-----------|-------------|----------------|-------------|-------------|",
        ])

        for param_name, result in report.parameter_analysis.items():
            rec = result.recommended_bounds if result.recommended_bounds else result.current_bounds
            lines.append(
                f"| {param_name} | {result.fitness_correlation:.3f} | "
                f"({result.current_bounds[0]:.4f}, {result.current_bounds[1]:.4f}) | "
                f"({rec[0]:.4f}, {rec[1]:.4f}) | {result.utilization:.2f} |"
            )

        lines.extend([
            "\n## Evolution Analysis",
            f"- **Converged Parameters**: {', '.join(report.evolution.converged_params) or 'None'}",
            f"- **Diverse Parameters**: {', '.join(report.evolution.diverse_params) or 'None'}",
        ])

        lines.extend([
            "\n## Recommendations",
            "### Suggested HYPERPARAM_CONFIGS",
            "```python",
            "HYPERPARAM_CONFIGS = [",
        ])

        for param_name, bounds in report.recommendations.hyperparam_bounds.items():
            param_type = 'int' if param_name in ['hidden_size', 'num_layers', 'batch_size', 'scheduler_patience', 'input_seq_length'] else 'float'
            lines.append(f"    HyperparamConfig('{param_name}', {bounds[0]}, {bounds[1]}, '{param_type}', '{param_name}'),")

        lines.extend([
            "]",
            "```",
            "\n### APO Settings",
        ])

        for setting, value in report.recommendations.apo_settings.items():
            lines.append(f"- **{setting}**: {value}")

        return "\n".join(lines)

    def _generate_json_report(self, report: AnalysisReport) -> str:
        """Generate JSON report for programmatic use."""
        import json

        def serialize(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            if isinstance(obj, pd.Series):
                return obj.to_dict()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return str(obj)

        data = {
            'run_ids': report.run_ids,
            'total_individuals': report.total_individuals,
            'total_iterations': report.total_iterations,
            'recommendations': {
                'hyperparam_bounds': report.recommendations.hyperparam_bounds,
                'apo_settings': report.recommendations.apo_settings,
                'confidence': report.recommendations.confidence,
            },
            'metadata': report.metadata,
        }

        return json.dumps(data, default=serialize, indent=2)

    def export_config_code(
        self,
        recommendations: ConfigurationRecommendation,
        output_path: str,
    ) -> None:
        """
        Export recommendations as Python code snippet.

        Generates code that can be copied into lstm_optimizer.py
        to update HYPERPARAM_CONFIGS.
        """
        lines = [
            "# Recommended HYPERPARAM_CONFIGS based on log analysis",
            "# Generated by LSTMLogAnalyzer",
            "",
            "HYPERPARAM_CONFIGS = [",
        ]

        int_params = {'hidden_size', 'num_layers', 'batch_size', 'scheduler_patience', 'input_seq_length'}

        for param_name, bounds in recommendations.hyperparam_bounds.items():
            param_type = 'int' if param_name in int_params else 'float'
            lines.append(
                f"    HyperparamConfig('{param_name}', {bounds[0]}, {bounds[1]}, "
                f"'{param_type}', '{param_name}'),"
            )

        lines.extend([
            "]",
            "",
            "# Recommended APO settings:",
        ])

        for setting, value in recommendations.apo_settings.items():
            lines.append(f"# {setting} = {value}")

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
