"""
Overlap Indicators

Implements overlap (moving average) technical indicators including:
SMA, EMA, DEMA, TEMA, KAMA, WMA, TRIMA, T3, SAR, BBANDS
"""

from typing import Any, Dict, Optional

import pandas as pd
import talib.abstract as ta

from ..base import BaseIndicatorOptimizer


class SMAIndicator(BaseIndicatorOptimizer):
    """Simple Moving Average indicator with dual MA crossover."""

    indicator_name = "SMA"
    talib_function = "ta.SMA"
    category = "overlap"
    outputs = ["sma_fast", "sma_slow"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "fast_period": {"default": 9, "range": [3, 50], "type": "int"},
            "slow_period": {"default": 21, "range": [10, 200], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        fast = params.get("fast_period", 9)
        slow = params.get("slow_period", 21)
        df["sma_fast"] = ta.SMA(df, timeperiod=fast)
        df["sma_slow"] = ta.SMA(df, timeperiod=slow)
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["sma_fast"], df["sma_slow"])

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["sma_fast"], df["sma_slow"])


class EMAIndicator(BaseIndicatorOptimizer):
    """Exponential Moving Average indicator with dual MA crossover."""

    indicator_name = "EMA"
    talib_function = "ta.EMA"
    category = "overlap"
    outputs = ["ema_fast", "ema_slow"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "fast_period": {"default": 9, "range": [3, 50], "type": "int"},
            "slow_period": {"default": 21, "range": [10, 200], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        fast = params.get("fast_period", 9)
        slow = params.get("slow_period", 21)
        df["ema_fast"] = ta.EMA(df, timeperiod=fast)
        df["ema_slow"] = ta.EMA(df, timeperiod=slow)
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["ema_fast"], df["ema_slow"])

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["ema_fast"], df["ema_slow"])


class DEMAIndicator(BaseIndicatorOptimizer):
    """Double Exponential Moving Average indicator."""

    indicator_name = "DEMA"
    talib_function = "ta.DEMA"
    category = "overlap"
    outputs = ["dema"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 20, "range": [5, 100], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["dema"] = ta.DEMA(df, timeperiod=params.get("timeperiod", 20))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["close"], df["dema"])

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["close"], df["dema"])


class TEMAIndicator(BaseIndicatorOptimizer):
    """Triple Exponential Moving Average indicator."""

    indicator_name = "TEMA"
    talib_function = "ta.TEMA"
    category = "overlap"
    outputs = ["tema"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 20, "range": [5, 100], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["tema"] = ta.TEMA(df, timeperiod=params.get("timeperiod", 20))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["close"], df["tema"])

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["close"], df["tema"])


class KAMAIndicator(BaseIndicatorOptimizer):
    """Kaufman Adaptive Moving Average indicator."""

    indicator_name = "KAMA"
    talib_function = "ta.KAMA"
    category = "overlap"
    outputs = ["kama"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 10, "range": [5, 30], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["kama"] = ta.KAMA(df, timeperiod=params.get("timeperiod", 10))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["close"], df["kama"])

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["close"], df["kama"])


class WMAIndicator(BaseIndicatorOptimizer):
    """Weighted Moving Average indicator."""

    indicator_name = "WMA"
    talib_function = "ta.WMA"
    category = "overlap"
    outputs = ["wma"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 20, "range": [5, 100], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["wma"] = ta.WMA(df, timeperiod=params.get("timeperiod", 20))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["close"], df["wma"])

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["close"], df["wma"])


class TRIMAIndicator(BaseIndicatorOptimizer):
    """Triangular Moving Average indicator."""

    indicator_name = "TRIMA"
    talib_function = "ta.TRIMA"
    category = "overlap"
    outputs = ["trima"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 20, "range": [5, 100], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["trima"] = ta.TRIMA(df, timeperiod=params.get("timeperiod", 20))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["close"], df["trima"])

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["close"], df["trima"])


class T3Indicator(BaseIndicatorOptimizer):
    """Triple Exponential Moving Average (T3) indicator."""

    indicator_name = "T3"
    talib_function = "ta.T3"
    category = "overlap"
    outputs = ["t3"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 5, "range": [3, 25], "type": "int"},
            "vfactor": {"default": 0.7, "range": [0.0, 1.0], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["t3"] = ta.T3(
            df,
            timeperiod=params.get("timeperiod", 5),
            vfactor=params.get("vfactor", 0.7)
        )
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["close"], df["t3"])

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["close"], df["t3"])


class SARIndicator(BaseIndicatorOptimizer):
    """Parabolic SAR indicator."""

    indicator_name = "SAR"
    talib_function = "ta.SAR"
    category = "overlap"
    outputs = ["sar"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "acceleration": {"default": 0.02, "range": [0.01, 0.05], "type": "float"},
            "maximum": {"default": 0.2, "range": [0.1, 0.4], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["sar"] = ta.SAR(
            df,
            acceleration=params.get("acceleration", 0.02),
            maximum=params.get("maximum", 0.2)
        )
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["close"], df["sar"])

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["close"], df["sar"])


class BBANDSIndicator(BaseIndicatorOptimizer):
    """Bollinger Bands indicator."""

    indicator_name = "BBANDS"
    talib_function = "ta.BBANDS"
    category = "overlap"
    outputs = ["upperband", "middleband", "lowerband"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 20, "range": [10, 50], "type": "int"},
            "nbdevup": {"default": 2.0, "range": [1.0, 3.0], "type": "float"},
            "nbdevdn": {"default": 2.0, "range": [1.0, 3.0], "type": "float"},
            "entry_factor": {"default": 1.02, "range": [1.0, 1.05], "type": "float"},
            "exit_factor": {"default": 0.98, "range": [0.95, 1.0], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        bbands = ta.BBANDS(
            df,
            timeperiod=params.get("timeperiod", 20),
            nbdevup=params.get("nbdevup", 2.0),
            nbdevdn=params.get("nbdevdn", 2.0)
        )
        df["upperband"] = bbands["upperband"]
        df["middleband"] = bbands["middleband"]
        df["lowerband"] = bbands["lowerband"]
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        factor = params.get("entry_factor", 1.02)
        return df["close"] < (df["lowerband"] * factor)

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        factor = params.get("exit_factor", 0.98)
        return df["close"] > (df["upperband"] * factor)
