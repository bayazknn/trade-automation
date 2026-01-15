"""
Indicator Association Analyzer Module

Analyzes which technical indicators are most associated with the 'trade' class
using statistical tests and association rule mining. Results can be used as
input for LSTM feature selection.

Methods:
- Chi-Square Test: Measures statistical dependence
- Mutual Information: Quantifies shared information
- Point-Biserial Correlation: Binary-binary correlation
- Fisher's Exact Test: Precise 2x2 contingency test
- Apriori Association Rules: Finds indicator combinations
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

try:
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False


@dataclass
class FeatureStatistics:
    """Statistics for a single feature's association with the target class."""
    feature_name: str
    chi2_statistic: float
    chi2_pvalue: float
    mutual_info: float
    point_biserial_corr: float
    point_biserial_pvalue: float
    fisher_odds_ratio: float
    fisher_pvalue: float
    trade_support: float        # P(feature=1 | trade)
    hold_support: float         # P(feature=1 | hold)
    lift: float                 # trade_support / overall_support
    composite_score: float = 0.0


@dataclass
class AssociationRule:
    """Association rule linking indicator combination to trade class."""
    antecedent: Tuple[str, ...]  # Indicator combination
    consequent: str              # 'trade'
    support: float
    confidence: float
    lift: float
    conviction: float
    leverage: float


@dataclass
class IndicatorAssociationResult:
    """Complete analysis result."""
    feature_statistics: List[FeatureStatistics]
    association_rules: List[AssociationRule]
    top_trade_indicators: List[str]           # Ranked by composite score
    indicator_combinations: List[Tuple[str, ...]]  # Best co-occurring sets
    class_distribution: Dict[str, int]        # hold/trade counts
    feature_columns: List[str]                # Original feature column names


class IndicatorAssociationAnalyzer:
    """
    Analyzes indicator-target associations using statistical tests and
    association rule mining.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with binary indicator columns and target column
    target_col : str
        Name of target column (default: 'tradeable')
    positive_class : str
        Name of positive class to analyze association with (default: 'trade')

    Example
    -------
    >>> analyzer = IndicatorAssociationAnalyzer(df, target_col='tradeable')
    >>> result = analyzer.analyze_all()
    >>> print(result.top_trade_indicators[:10])
    >>> analyzer.plot_feature_importance(top_k=20)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = 'tradeable',
        positive_class: str = 'trade'
    ):
        self.df = df.copy()
        self.target_col = target_col
        self.positive_class = positive_class

        # Identify feature columns (binary indicators)
        self.feature_cols = [
            col for col in df.columns
            if col.endswith(('_gs_entry', '_gs_exit', '_ho_entry', '_ho_exit'))
        ]

        if not self.feature_cols:
            raise ValueError("No indicator columns found (expected *_gs_entry, *_gs_exit, etc.)")

        # Encode target to binary (trade=1, hold=0)
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(self.df[target_col])
        self.positive_idx = list(self.le.classes_).index(positive_class)

        # Ensure y is encoded as positive_class=1
        if self.positive_idx == 0:
            self.y = 1 - self.y

        # Feature matrix
        self.X = self.df[self.feature_cols].values.astype(np.float64)

        # Class distribution
        self.class_distribution = dict(self.df[target_col].value_counts())

        # Cache for computed statistics
        self._chi2_results = None
        self._mi_results = None
        self._pb_results = None
        self._fisher_results = None
        self._feature_stats = None

    def compute_chi_square(self) -> pd.DataFrame:
        """
        Compute Chi-Square test for each feature vs target.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: feature, chi2_statistic, chi2_pvalue
        """
        if self._chi2_results is not None:
            return self._chi2_results

        chi2_stats, p_values = chi2(self.X, self.y)

        self._chi2_results = pd.DataFrame({
            'feature': self.feature_cols,
            'chi2_statistic': chi2_stats,
            'chi2_pvalue': p_values
        })

        return self._chi2_results

    def compute_mutual_information(self, n_neighbors: int = 3, random_state: int = 42) -> pd.DataFrame:
        """
        Compute Mutual Information for each feature vs target.

        Parameters
        ----------
        n_neighbors : int
            Number of neighbors for MI estimation
        random_state : int
            Random seed for reproducibility

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: feature, mutual_info
        """
        if self._mi_results is not None:
            return self._mi_results

        mi_scores = mutual_info_classif(
            self.X, self.y,
            discrete_features=True,
            n_neighbors=n_neighbors,
            random_state=random_state
        )

        self._mi_results = pd.DataFrame({
            'feature': self.feature_cols,
            'mutual_info': mi_scores
        })

        return self._mi_results

    def compute_point_biserial(self) -> pd.DataFrame:
        """
        Compute Point-Biserial correlation for each feature vs target.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: feature, point_biserial_corr, point_biserial_pvalue
        """
        if self._pb_results is not None:
            return self._pb_results

        results = []
        for i, col in enumerate(self.feature_cols):
            corr, pvalue = stats.pointbiserialr(self.X[:, i], self.y)
            results.append({
                'feature': col,
                'point_biserial_corr': corr if not np.isnan(corr) else 0.0,
                'point_biserial_pvalue': pvalue if not np.isnan(pvalue) else 1.0
            })

        self._pb_results = pd.DataFrame(results)
        return self._pb_results

    def compute_fisher_exact(self) -> pd.DataFrame:
        """
        Compute Fisher's Exact test for each feature vs target.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: feature, fisher_odds_ratio, fisher_pvalue
        """
        if self._fisher_results is not None:
            return self._fisher_results

        results = []
        for i, col in enumerate(self.feature_cols):
            # Build 2x2 contingency table
            # [[feature=0 & target=0, feature=0 & target=1],
            #  [feature=1 & target=0, feature=1 & target=1]]
            x = self.X[:, i]
            table = np.array([
                [np.sum((x == 0) & (self.y == 0)), np.sum((x == 0) & (self.y == 1))],
                [np.sum((x == 1) & (self.y == 0)), np.sum((x == 1) & (self.y == 1))]
            ])

            try:
                odds_ratio, pvalue = stats.fisher_exact(table)
                # Handle infinite odds ratio
                if np.isinf(odds_ratio):
                    odds_ratio = 1000.0 if odds_ratio > 0 else 0.001
            except Exception:
                odds_ratio, pvalue = 1.0, 1.0

            results.append({
                'feature': col,
                'fisher_odds_ratio': odds_ratio,
                'fisher_pvalue': pvalue
            })

        self._fisher_results = pd.DataFrame(results)
        return self._fisher_results

    def compute_support_lift(self) -> pd.DataFrame:
        """
        Compute support (conditional probability) and lift for each feature.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: feature, trade_support, hold_support, lift
        """
        results = []
        trade_mask = self.y == 1
        hold_mask = self.y == 0
        overall_support = np.mean(self.X, axis=0)

        for i, col in enumerate(self.feature_cols):
            x = self.X[:, i]
            trade_support = np.mean(x[trade_mask]) if trade_mask.sum() > 0 else 0.0
            hold_support = np.mean(x[hold_mask]) if hold_mask.sum() > 0 else 0.0

            # Lift = P(feature=1|trade) / P(feature=1)
            # Handle edge cases: if feature is always 0 or always 1
            if overall_support[i] > 0 and overall_support[i] < 1:
                lift = trade_support / overall_support[i]
            elif overall_support[i] == 0:
                # Feature is always 0, lift is undefined (set to 1)
                lift = 1.0
            else:
                # Feature is always 1, lift = trade_support (which is 1)
                lift = 1.0

            # Handle NaN/inf
            if np.isnan(lift) or np.isinf(lift):
                lift = 1.0

            results.append({
                'feature': col,
                'trade_support': trade_support,
                'hold_support': hold_support,
                'lift': lift
            })

        return pd.DataFrame(results)

    def compute_all_statistics(self) -> pd.DataFrame:
        """
        Compute all statistical measures for each feature.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with all statistics
        """
        chi2_df = self.compute_chi_square()
        mi_df = self.compute_mutual_information()
        pb_df = self.compute_point_biserial()
        fisher_df = self.compute_fisher_exact()
        support_df = self.compute_support_lift()

        # Merge all results
        result = chi2_df.merge(mi_df, on='feature')
        result = result.merge(pb_df, on='feature')
        result = result.merge(fisher_df, on='feature')
        result = result.merge(support_df, on='feature')

        # Compute composite score
        result['composite_score'] = self._compute_composite_score(result)

        return result.sort_values('composite_score', ascending=False)

    def _compute_composite_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute composite score combining all metrics.

        Score = weighted combination of normalized metrics:
        - Chi-square statistic (normalized)
        - Mutual information (normalized)
        - Lift (capped and normalized)
        - Fisher's test significance (1 - pvalue)
        """
        # Normalize each metric to [0, 1] range, handling NaN and constant arrays
        def normalize(x):
            x = np.array(x, dtype=np.float64)
            # Replace NaN/inf with 0
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            min_val, max_val = x.min(), x.max()
            if max_val - min_val == 0:
                return np.zeros_like(x)
            return (x - min_val) / (max_val - min_val)

        # Cap lift at reasonable values to prevent outliers
        lift_values = np.nan_to_num(df['lift'].values, nan=1.0, posinf=10.0, neginf=0.0)
        lift_capped = np.clip(lift_values, 0, 10)

        # Compute normalized components
        chi2_norm = normalize(df['chi2_statistic'].values)
        mi_norm = normalize(df['mutual_info'].values)
        lift_norm = normalize(lift_capped)

        # Fisher score: lower p-value = higher score
        fisher_pvals = np.nan_to_num(df['fisher_pvalue'].values, nan=1.0)
        fisher_score = 1 - np.clip(fisher_pvals, 0, 1)

        # Weighted composite score
        composite = (
            0.25 * chi2_norm +
            0.25 * mi_norm +
            0.25 * lift_norm +
            0.25 * fisher_score
        )

        return composite

    def mine_association_rules(
        self,
        min_support: float = 0.01,
        min_confidence: float = 0.3,
        min_lift: float = 1.0,
        max_antecedent_len: int = 3,
        use_fpgrowth: bool = True,
        max_rules: int = 1000,
        max_itemsets: int = 1000000,
        top_k_features: Optional[int] = None
    ) -> List[AssociationRule]:
        """
        Mine association rules where consequent is the trade class.

        Parameters
        ----------
        min_support : float
            Minimum support threshold (default: 0.01)
        min_confidence : float
            Minimum confidence threshold (default: 0.3)
        min_lift : float
            Minimum lift threshold (default: 1.0)
        max_antecedent_len : int
            Maximum number of indicators in rule antecedent (default: 3)
        use_fpgrowth : bool
            Use FP-Growth instead of Apriori (faster for large datasets)
        max_rules : int
            Maximum number of rules to return (default: 1000, prevents memory issues)
        max_itemsets : int
            Maximum frequent itemsets to process (default: 200000, prevents memory issues)
        top_k_features : int, optional
            Only use top K features by composite score to reduce memory.
            If None, uses all features. Recommended: 50-100 for large datasets.

        Returns
        -------
        List[AssociationRule]
            List of association rules sorted by lift
        """
        import gc

        if not MLXTEND_AVAILABLE:
            print("Warning: mlxtend not installed. Skipping association rule mining.")
            print("Install with: pip install mlxtend")
            return []

        # Optionally reduce features to top K by composite score (memory optimization)
        if top_k_features is not None and top_k_features < len(self.feature_cols):
            stats_df = self.compute_all_statistics()
            selected_features = stats_df.head(top_k_features)['feature'].tolist()
            print(f"Using top {top_k_features} features for association rule mining")
        else:
            selected_features = self.feature_cols

        # Prepare transaction data with minimal memory footprint
        # Use sparse boolean representation
        trans_df = self.df[selected_features].astype(bool).copy()
        trans_df['trade'] = (self.y == 1)
        trans_df['hold'] = (self.y == 0)

        # Force garbage collection before heavy computation
        gc.collect()

        # Mine frequent itemsets with max_len to prevent memory explosion
        # max_len = max_antecedent_len + 1 (to include 'trade' in itemsets)
        max_len = max_antecedent_len + 1

        try:
            if use_fpgrowth:
                frequent_itemsets = fpgrowth(
                    trans_df,
                    min_support=min_support,
                    use_colnames=True,
                    max_len=max_len
                )
            else:
                frequent_itemsets = apriori(
                    trans_df,
                    min_support=min_support,
                    use_colnames=True,
                    max_len=max_len,
                    low_memory=True
                )
        except MemoryError:
            print("MemoryError: Try increasing min_support or reducing top_k_features")
            del trans_df
            gc.collect()
            return []

        # Clean up transaction DataFrame immediately
        del trans_df
        gc.collect()

        if len(frequent_itemsets) == 0:
            print(f"No frequent itemsets found with min_support={min_support}")
            return []

        print(f"Found {len(frequent_itemsets)} frequent itemsets")

        # Memory guard: if too many itemsets, keep only highest support ones
        # (association_rules needs complete lattice, so we can't filter by content)
        if len(frequent_itemsets) > max_itemsets:
            print(f"Warning: {len(frequent_itemsets)} itemsets exceeds limit of {max_itemsets}")
            print("Keeping only highest support itemsets to prevent memory issues")
            frequent_itemsets = frequent_itemsets.nlargest(max_itemsets, 'support')
            gc.collect()

        # Generate association rules from full itemsets
        # (association_rules needs complete lattice structure to compute metrics)
        try:
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=min_confidence
            )
        except Exception as e:
            print(f"Error generating rules: {e}")
            del frequent_itemsets
            gc.collect()
            return []

        # Clean up itemsets immediately after rule generation
        del frequent_itemsets
        gc.collect()

        if len(rules) == 0:
            print(f"No rules found with min_confidence={min_confidence}")
            return []

        # Filter rules where consequent is exactly 'trade' (not in antecedent)
        # Use vectorized operations where possible for speed
        mask = (
            rules['consequents'].apply(lambda x: x == frozenset({'trade'})) &
            (rules['lift'] >= min_lift) &
            (rules['antecedents'].apply(len) <= max_antecedent_len)
        )
        trade_rules = rules.loc[mask].copy()

        # Clean up full rules DataFrame
        del rules
        gc.collect()

        if len(trade_rules) == 0:
            print("No rules with 'trade' as consequent found")
            return []

        # Sort by lift descending and limit to max_rules
        trade_rules = trade_rules.nlargest(max_rules, 'lift')

        # Convert to AssociationRule objects
        result = []
        for _, row in trade_rules.iterrows():
            conviction_val = float(row['conviction']) if not np.isinf(row['conviction']) else 1000.0
            rule = AssociationRule(
                antecedent=tuple(sorted(row['antecedents'])),
                consequent='trade',
                support=float(row['support']),
                confidence=float(row['confidence']),
                lift=float(row['lift']),
                conviction=conviction_val,
                leverage=float(row['leverage'])
            )
            result.append(rule)

        # Final cleanup
        del trade_rules
        gc.collect()

        return result

    def rank_features(self, method: str = 'composite') -> pd.DataFrame:
        """
        Rank features by association with trade class.

        Parameters
        ----------
        method : str
            Ranking method: 'composite', 'chi2', 'mutual_info', 'lift', 'fisher'

        Returns
        -------
        pd.DataFrame
            Ranked features with scores
        """
        stats_df = self.compute_all_statistics()

        sort_cols = {
            'composite': ('composite_score', False),
            'chi2': ('chi2_statistic', False),
            'mutual_info': ('mutual_info', False),
            'lift': ('lift', False),
            'fisher': ('fisher_pvalue', True)
        }

        col, ascending = sort_cols.get(method, ('composite_score', False))
        return stats_df.sort_values(col, ascending=ascending).reset_index(drop=True)

    def analyze_all(
        self,
        min_support: float = 0.01,
        min_confidence: float = 0.3,
        min_lift: float = 1.0,
        max_antecedent_len: int = 3
    ) -> IndicatorAssociationResult:
        """
        Run complete analysis including statistics and association rules.

        Returns
        -------
        IndicatorAssociationResult
            Complete analysis result
        """
        # Compute all statistics
        stats_df = self.compute_all_statistics()

        # Convert to FeatureStatistics objects
        feature_stats = []
        for _, row in stats_df.iterrows():
            fs = FeatureStatistics(
                feature_name=row['feature'],
                chi2_statistic=row['chi2_statistic'],
                chi2_pvalue=row['chi2_pvalue'],
                mutual_info=row['mutual_info'],
                point_biserial_corr=row['point_biserial_corr'],
                point_biserial_pvalue=row['point_biserial_pvalue'],
                fisher_odds_ratio=row['fisher_odds_ratio'],
                fisher_pvalue=row['fisher_pvalue'],
                trade_support=row['trade_support'],
                hold_support=row['hold_support'],
                lift=row['lift'],
                composite_score=row['composite_score']
            )
            feature_stats.append(fs)

        # Mine association rules
        rules = self.mine_association_rules(
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
            max_antecedent_len=max_antecedent_len
        )

        # Extract top indicators
        top_indicators = stats_df.head(20)['feature'].tolist()

        # Extract best indicator combinations from rules
        combinations = []
        seen = set()
        for rule in rules[:50]:  # Top 50 rules
            if rule.antecedent not in seen:
                combinations.append(rule.antecedent)
                seen.add(rule.antecedent)

        return IndicatorAssociationResult(
            feature_statistics=feature_stats,
            association_rules=rules,
            top_trade_indicators=top_indicators,
            indicator_combinations=combinations[:20],
            class_distribution=self.class_distribution,
            feature_columns=self.feature_cols
        )

    # ===================== LSTM Integration Methods =====================

    def get_feature_mask(self, percentile: float = 50.0) -> np.ndarray:
        """
        Generate binary feature mask for LSTM feature selection.

        Features with composite score above the percentile threshold get 1.

        Parameters
        ----------
        percentile : float
            Percentile threshold (0-100). Features above this percentile get 1.
            Default: 50.0 (top 50% of features)

        Returns
        -------
        np.ndarray
            Binary mask array aligned with feature_cols
        """
        stats_df = self.compute_all_statistics()
        threshold = np.percentile(stats_df['composite_score'], percentile)
        mask = (stats_df['composite_score'] >= threshold).astype(int).values
        return mask

    def get_top_feature_indices(self, top_k: int = 30) -> List[int]:
        """
        Get indices of top-K trade-associated features.

        Parameters
        ----------
        top_k : int
            Number of top features to return

        Returns
        -------
        List[int]
            Indices of top features (aligned with feature_cols order)
        """
        stats_df = self.compute_all_statistics()
        top_features = stats_df.head(top_k)['feature'].tolist()
        return [self.feature_cols.index(f) for f in top_features]

    def export_for_lstm(self, percentile: float = 50.0, top_k: int = 30) -> Dict[str, Any]:
        """
        Export analysis results for LSTM optimizer integration.

        Parameters
        ----------
        percentile : float
            Percentile threshold for feature mask
        top_k : int
            Number of top features to include

        Returns
        -------
        Dict[str, Any]
            Dictionary with feature mask, scores, and recommendations
        """
        stats_df = self.compute_all_statistics()

        # Get feature mask
        mask = self.get_feature_mask(percentile)

        # Get top features
        top_features = stats_df.head(top_k)['feature'].tolist()

        # Get feature scores
        feature_scores = dict(zip(stats_df['feature'], stats_df['composite_score']))

        # Get indicator combinations from rules
        rules = self.mine_association_rules()
        combinations = [rule.antecedent for rule in rules[:20]]

        return {
            'feature_columns': self.feature_cols,
            'feature_mask': mask.tolist(),
            'feature_scores': feature_scores,
            'top_features': top_features,
            'top_feature_indices': self.get_top_feature_indices(top_k),
            'indicator_combinations': combinations,
            'class_distribution': self.class_distribution,
            'percentile_threshold': percentile,
        }

    # ===================== Visualization Methods =====================

    def plot_feature_importance(
        self,
        top_k: int = 20,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance based on composite score.

        Parameters
        ----------
        top_k : int
            Number of top features to show
        figsize : Tuple[int, int]
            Figure size
        save_path : str, optional
            Path to save figure
        """
        import matplotlib.pyplot as plt

        stats_df = self.compute_all_statistics().head(top_k)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Composite score bar chart
        ax1 = axes[0]
        colors = plt.cm.RdYlGn(stats_df['composite_score'] / stats_df['composite_score'].max())
        ax1.barh(range(len(stats_df)), stats_df['composite_score'], color=colors)
        ax1.set_yticks(range(len(stats_df)))
        ax1.set_yticklabels(stats_df['feature'], fontsize=8)
        ax1.set_xlabel('Composite Score')
        ax1.set_title(f'Top {top_k} Trade-Associated Indicators')
        ax1.invert_yaxis()

        # Lift vs Trade Support scatter
        ax2 = axes[1]
        scatter = ax2.scatter(
            stats_df['trade_support'],
            stats_df['lift'],
            c=stats_df['composite_score'],
            cmap='RdYlGn',
            s=100,
            alpha=0.7
        )
        ax2.set_xlabel('Trade Support P(feature=1|trade)')
        ax2.set_ylabel('Lift')
        ax2.set_title('Lift vs Trade Support')
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, ax=ax2, label='Composite Score')

        # Add labels for top 5
        for i, (_, row) in enumerate(stats_df.head(5).iterrows()):
            ax2.annotate(
                row['feature'].replace('_gs_', '\n'),
                (row['trade_support'], row['lift']),
                fontsize=7,
                ha='center'
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, axes

    def plot_chi_square_distribution(
        self,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """Plot Chi-square statistics distribution."""
        import matplotlib.pyplot as plt

        stats_df = self.compute_chi_square()

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Chi2 histogram
        ax1 = axes[0]
        ax1.hist(stats_df['chi2_statistic'], bins=30, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Chi-Square Statistic')
        ax1.set_ylabel('Count')
        ax1.set_title('Chi-Square Distribution')
        ax1.axvline(
            x=stats_df['chi2_statistic'].median(),
            color='red',
            linestyle='--',
            label=f'Median: {stats_df["chi2_statistic"].median():.2f}'
        )
        ax1.legend()

        # P-value histogram (log scale)
        ax2 = axes[1]
        pvals = stats_df['chi2_pvalue'].clip(lower=1e-50)
        ax2.hist(np.log10(pvals), bins=30, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('log10(p-value)')
        ax2.set_ylabel('Count')
        ax2.set_title('Chi-Square P-Value Distribution')
        ax2.axvline(x=np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, axes

    def plot_lift_analysis(
        self,
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[str] = None
    ):
        """Plot lift analysis showing trade vs hold support."""
        import matplotlib.pyplot as plt

        stats_df = self.compute_all_statistics()

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Lift distribution
        ax1 = axes[0]
        lift_capped = stats_df['lift'].clip(upper=5)
        ax1.hist(lift_capped, bins=30, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Lift (capped at 5)')
        ax1.set_ylabel('Count')
        ax1.set_title('Lift Distribution')
        ax1.axvline(x=1.0, color='red', linestyle='--', label='Lift=1 (no association)')
        ax1.legend()

        # Trade support vs Hold support
        ax2 = axes[1]
        ax2.scatter(stats_df['hold_support'], stats_df['trade_support'], alpha=0.6)
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax2.set_xlabel('Hold Support P(feat=1|hold)')
        ax2.set_ylabel('Trade Support P(feat=1|trade)')
        ax2.set_title('Support Comparison')

        # Number of features above lift threshold
        ax3 = axes[2]
        thresholds = [1.0, 1.2, 1.5, 2.0, 3.0]
        counts = [sum(stats_df['lift'] >= t) for t in thresholds]
        ax3.bar([str(t) for t in thresholds], counts, edgecolor='black')
        ax3.set_xlabel('Lift Threshold')
        ax3.set_ylabel('Number of Features')
        ax3.set_title('Features Above Lift Threshold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, axes

    def plot_association_network(
        self,
        min_lift: float = 1.5,
        max_rules: int = 30,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ):
        """
        Plot association rules as a network graph.

        Nodes are indicators, edges connect indicators that appear together
        in high-lift rules.
        """
        import matplotlib.pyplot as plt

        rules = self.mine_association_rules(min_lift=min_lift)
        if not rules:
            print(f"No rules found with min_lift={min_lift}")
            return None, None

        # Build co-occurrence counts
        cooccur = {}
        for rule in rules[:max_rules]:
            if len(rule.antecedent) >= 2:
                for i, ind1 in enumerate(rule.antecedent):
                    for ind2 in rule.antecedent[i+1:]:
                        key = tuple(sorted([ind1, ind2]))
                        if key not in cooccur:
                            cooccur[key] = {'count': 0, 'lift_sum': 0}
                        cooccur[key]['count'] += 1
                        cooccur[key]['lift_sum'] += rule.lift

        if not cooccur:
            print("No indicator co-occurrences found in rules")
            return None, None

        try:
            import networkx as nx
        except ImportError:
            print("networkx not installed. Install with: pip install networkx")
            return None, None

        # Create graph
        G = nx.Graph()

        # Add edges weighted by co-occurrence
        for (ind1, ind2), data in cooccur.items():
            G.add_edge(ind1, ind2, weight=data['count'], lift=data['lift_sum']/data['count'])

        fig, ax = plt.subplots(figsize=figsize)

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Draw edges
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=[w * 0.5 for w in edge_weights],
            alpha=0.5,
            edge_color='gray'
        )

        # Draw nodes
        node_sizes = [300 + G.degree(n) * 100 for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color='lightblue', alpha=0.8)

        # Draw labels
        labels = {n: n.replace('_gs_', '\n') for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7)

        ax.set_title(f'Indicator Co-occurrence Network (min_lift={min_lift})')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax

    # ===================== Report Generation =====================

    def generate_report(
        self,
        output_path: str,
        format: str = 'markdown',
        top_k: int = 30
    ):
        """
        Generate analysis report.

        Parameters
        ----------
        output_path : str
            Path to save report
        format : str
            Report format: 'markdown' or 'json'
        top_k : int
            Number of top features to include
        """
        stats_df = self.compute_all_statistics()
        rules = self.mine_association_rules()

        if format == 'markdown':
            self._generate_markdown_report(output_path, stats_df, rules, top_k)
        elif format == 'json':
            self._generate_json_report(output_path, stats_df, rules, top_k)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _generate_markdown_report(
        self,
        output_path: str,
        stats_df: pd.DataFrame,
        rules: List[AssociationRule],
        top_k: int
    ):
        """Generate markdown report."""
        lines = [
            "# Indicator Association Analysis Report",
            "",
            "## Dataset Summary",
            "",
            f"- **Total samples**: {len(self.df)}",
            f"- **Trade samples**: {self.class_distribution.get('trade', 0)}",
            f"- **Hold samples**: {self.class_distribution.get('hold', 0)}",
            f"- **Number of features**: {len(self.feature_cols)}",
            "",
            "## Top Trade-Associated Indicators",
            "",
            "| Rank | Indicator | Composite Score | Chi² | MI | Lift | Trade Support |",
            "|------|-----------|-----------------|------|-----|------|---------------|",
        ]

        for i, (_, row) in enumerate(stats_df.head(top_k).iterrows()):
            lines.append(
                f"| {i+1} | {row['feature']} | {row['composite_score']:.4f} | "
                f"{row['chi2_statistic']:.2f} | {row['mutual_info']:.4f} | "
                f"{row['lift']:.2f} | {row['trade_support']:.3f} |"
            )

        lines.extend([
            "",
            "## Top Association Rules",
            "",
            "| Antecedent | Support | Confidence | Lift |",
            "|------------|---------|------------|------|",
        ])

        for rule in rules[:20]:
            ant_str = " + ".join(rule.antecedent)
            lines.append(
                f"| {ant_str} | {rule.support:.4f} | "
                f"{rule.confidence:.3f} | {rule.lift:.2f} |"
            )

        lines.extend([
            "",
            "## Statistical Test Summary",
            "",
            f"- Features with Chi² p-value < 0.05: {sum(stats_df['chi2_pvalue'] < 0.05)}",
            f"- Features with Fisher p-value < 0.05: {sum(stats_df['fisher_pvalue'] < 0.05)}",
            f"- Features with Lift > 1.0: {sum(stats_df['lift'] > 1.0)}",
            f"- Features with Lift > 1.5: {sum(stats_df['lift'] > 1.5)}",
            f"- Mean Mutual Information: {stats_df['mutual_info'].mean():.4f}",
            "",
        ])

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"Report saved to {output_path}")

    def _generate_json_report(
        self,
        output_path: str,
        stats_df: pd.DataFrame,
        rules: List[AssociationRule],
        top_k: int
    ):
        """Generate JSON report."""
        import json

        report = {
            'summary': {
                'total_samples': len(self.df),
                'class_distribution': self.class_distribution,
                'num_features': len(self.feature_cols),
            },
            'top_features': stats_df.head(top_k).to_dict('records'),
            'association_rules': [
                {
                    'antecedent': list(r.antecedent),
                    'support': r.support,
                    'confidence': r.confidence,
                    'lift': r.lift,
                }
                for r in rules[:50]
            ],
            'statistics_summary': {
                'chi2_significant_count': int(sum(stats_df['chi2_pvalue'] < 0.05)),
                'fisher_significant_count': int(sum(stats_df['fisher_pvalue'] < 0.05)),
                'high_lift_count': int(sum(stats_df['lift'] > 1.5)),
                'mean_mutual_info': float(stats_df['mutual_info'].mean()),
            },
            'feature_mask': self.get_feature_mask(50.0).tolist(),
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Report saved to {output_path}")

    def print_summary(self, top_k: int = 10):
        """Print quick summary of analysis."""
        stats_df = self.compute_all_statistics()

        print("=" * 60)
        print("INDICATOR ASSOCIATION ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"\nDataset: {len(self.df)} samples")
        print(f"  - Trade: {self.class_distribution.get('trade', 0)}")
        print(f"  - Hold: {self.class_distribution.get('hold', 0)}")
        print(f"\nTotal features: {len(self.feature_cols)}")
        print(f"  - Chi² significant (p<0.05): {sum(stats_df['chi2_pvalue'] < 0.05)}")
        print(f"  - Fisher significant (p<0.05): {sum(stats_df['fisher_pvalue'] < 0.05)}")
        print(f"  - Lift > 1.0: {sum(stats_df['lift'] > 1.0)}")
        print(f"  - Lift > 1.5: {sum(stats_df['lift'] > 1.5)}")

        print(f"\nTop {top_k} Trade-Associated Indicators:")
        print("-" * 60)
        for i, (_, row) in enumerate(stats_df.head(top_k).iterrows()):
            print(f"  {i+1:2d}. {row['feature']:25s} score={row['composite_score']:.4f} lift={row['lift']:.2f}")

        print("=" * 60)
