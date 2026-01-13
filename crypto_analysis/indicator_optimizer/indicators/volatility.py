"""
Volatility Indicators

Implements volatility-based technical indicators including:
ATR, NATR
"""

from typing import Any, Dict, Optional

import pandas as pd
import talib.abstract as ta

from ..base import BaseIndicatorOptimizer


class ATRIndicator(BaseIndicatorOptimizer):
    """
    Average True Range indicator.

    Note: ATR is typically used for stop-loss calculation,
    not direct entry/exit signals. The signals generated here
    are based on ATR breakouts.
    """

    indicator_name = "ATR"
    talib_function = "ta.ATR"
    category = "volatility"
    outputs = ["atr"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 14, "range": [7, 28], "type": "int"},
            "stop_loss_multiplier": {"default": 2.0, "range": [1.5, 4.0], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["atr"] = ta.ATR(df, timeperiod=params.get("timeperiod", 14))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # ATR-based breakout: price moves more than ATR from previous close
        multiplier = params.get("stop_loss_multiplier", 2.0)
        price_change = df["close"] - df["close"].shift(1)
        return price_change > (df["atr"] * multiplier * 0.5)

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # Exit when price drops more than ATR
        multiplier = params.get("stop_loss_multiplier", 2.0)
        price_change = df["close"] - df["close"].shift(1)
        return price_change < -(df["atr"] * multiplier * 0.5)

    def calculate_stop_loss(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        **params
    ) -> float:
        """
        Calculate stop-loss price based on ATR.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with ATR calculated
        entry_idx : int
            Index of entry signal
        **params
            Must include stop_loss_multiplier

        Returns
        -------
        float
            Stop-loss price
        """
        multiplier = params.get("stop_loss_multiplier", 2.0)
        entry_price = df.loc[entry_idx, "close"]
        atr_value = df.loc[entry_idx, "atr"]
        return entry_price - (atr_value * multiplier)


class NATRIndicator(BaseIndicatorOptimizer):
    """
    Normalized Average True Range indicator.

    NATR expresses ATR as a percentage of close price,
    making it comparable across different price levels.
    """

    indicator_name = "NATR"
    talib_function = "ta.NATR"
    category = "volatility"
    outputs = ["natr"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 14, "range": [7, 28], "type": "int"},
            "entry_constant": {"default": 2.0, "range": [1.0, 5.0], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["natr"] = ta.NATR(df, timeperiod=params.get("timeperiod", 14))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # Enter when volatility is above threshold (high volatility = opportunities)
        constant = params.get("entry_constant", 2.0)
        return df["natr"] > constant

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # Exit when volatility drops significantly
        constant = params.get("entry_constant", 2.0)
        return df["natr"] < (constant * 0.5)


class TRANGEIndicator(BaseIndicatorOptimizer):
    """
    True Range indicator.

    TRANGE measures the greatest of:
    - Current High - Current Low
    - Abs(Current High - Previous Close)
    - Abs(Current Low - Previous Close)
    """

    indicator_name = "TRANGE"
    talib_function = "ta.TRANGE"
    category = "volatility"
    outputs = ["trange"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "entry_multiplier": {"default": 1.5, "range": [1.0, 3.0], "type": "float"},
            "exit_multiplier": {"default": 0.5, "range": [0.3, 1.0], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["trange"] = ta.TRANGE(df)
        # Calculate average true range for comparison
        df["trange_avg"] = df["trange"].rolling(window=14).mean()
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # Entry when true range exceeds average (breakout volatility)
        multiplier = params.get("entry_multiplier", 1.5)
        return df["trange"] > (df["trange_avg"] * multiplier)

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # Exit when true range drops below average (volatility contraction)
        multiplier = params.get("exit_multiplier", 0.5)
        return df["trange"] < (df["trange_avg"] * multiplier)
