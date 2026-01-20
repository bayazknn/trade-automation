"""
Output builder for exporting optimization results to CSV and JSON.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class OutputBuilder:
    """Build and export optimization results."""

    def __init__(self, output_dir: Path):
        """
        Initialize OutputBuilder.

        Parameters
        ----------
        output_dir : Path
            Output directory for exported files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_result_dataframe(
        self,
        df: pd.DataFrame,
        indicator_signals: Dict[str, Dict],
        indicator_values: Optional[Dict[str, pd.DataFrame]] = None,
        tradeable: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Build result DataFrame with OHLCV, signals, and indicators.

        Parameters
        ----------
        df : pd.DataFrame
            Base OHLCV DataFrame
        indicator_signals : dict
            Dict of {indicator_name: {"entry": Series, "exit": Series}}
        indicator_values : dict, optional
            Dict of {indicator_name: DataFrame} with indicator output values
        tradeable : pd.Series, optional
            Tradeable column ("trade"/"hold")

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with all columns
        """
        result = df.copy()

        # Add signal columns
        for indicator_name, signals in indicator_signals.items():
            if "entry" in signals:
                result[f"{indicator_name}_entry"] = signals["entry"]
            if "exit" in signals:
                result[f"{indicator_name}_exit"] = signals["exit"]

        # Add indicator value columns
        if indicator_values:
            for indicator_name, values_df in indicator_values.items():
                for col in values_df.columns:
                    col_name = f"{indicator_name}_{col}"
                    if col_name not in result.columns:
                        result[col_name] = values_df[col]

        # Add tradeable column
        if tradeable is not None:
            result["tradeable"] = tradeable

        return result

    def export_csv(
        self,
        symbol: str,
        df: pd.DataFrame,
        suffix: str = "_optimized",
    ) -> Path:
        """
        Export DataFrame to CSV.

        Parameters
        ----------
        symbol : str
            Cryptocurrency symbol
        df : pd.DataFrame
            DataFrame to export
        suffix : str
            Filename suffix

        Returns
        -------
        Path
            Path to exported file
        """
        filename = f"{symbol}{suffix}.csv"
        filepath = self.output_dir / filename

        df.to_csv(filepath, index=False)
        logger.info(f"Exported {filepath}")

        return filepath

    def export_params(
        self,
        symbol: str,
        params: Dict[str, Any],
        suffix: str = "_params",
    ) -> Path:
        """
        Export optimization parameters to JSON.

        Parameters
        ----------
        symbol : str
            Cryptocurrency symbol
        params : dict
            Parameters dict {indicator_name: params_dict}
        suffix : str
            Filename suffix

        Returns
        -------
        Path
            Path to exported file
        """
        filename = f"{symbol}{suffix}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(params, f, indent=2, default=self._json_serializer)

        logger.info(f"Exported {filepath}")
        return filepath

    def export_combined_results(
        self,
        all_results: Dict[str, pd.DataFrame],
        filename: str = "all_results.csv",
    ) -> Optional[Path]:
        """
        Export combined results from all symbols.

        Parameters
        ----------
        all_results : dict
            Dict of {symbol: DataFrame}
        filename : str
            Output filename

        Returns
        -------
        Path
            Path to exported file
        """
        # Combine all DataFrames with symbol column
        dfs = []
        for symbol, df in all_results.items():
            df_copy = df.copy()
            df_copy.insert(0, "symbol", symbol)
            dfs.append(df_copy)

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            filepath = self.output_dir / filename
            combined.to_csv(filepath, index=False)
            logger.info(f"Exported combined results to {filepath}")
            return filepath

        return None

    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for non-serializable objects."""
        if hasattr(obj, "item"):  # numpy types
            return obj.item()
        if hasattr(obj, "tolist"):  # numpy arrays
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def export_optimization_results(
    symbol: str,
    df: pd.DataFrame,
    params: Dict[str, Any],
    output_dir: str = "output",
) -> tuple:
    """
    Convenience function to export results.

    Parameters
    ----------
    symbol : str
        Cryptocurrency symbol
    df : pd.DataFrame
        Result DataFrame
    params : dict
        Optimization parameters
    output_dir : str
        Output directory

    Returns
    -------
    tuple
        (csv_path, params_path)
    """
    builder = OutputBuilder(Path(output_dir))
    csv_path = builder.export_csv(symbol, df)
    params_path = builder.export_params(symbol, params)
    return csv_path, params_path
