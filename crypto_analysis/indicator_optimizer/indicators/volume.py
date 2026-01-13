"""
Volume Indicators

Implements volume-based technical indicators including:
OBV, AD, ADOSC
"""

from typing import Any, Dict, Optional

import pandas as pd
import talib.abstract as ta

from ..base import BaseIndicatorOptimizer


class OBVIndicator(BaseIndicatorOptimizer):
    """On Balance Volume indicator."""

    indicator_name = "OBV"
    talib_function = "ta.OBV"
    category = "volume"
    outputs = ["obv", "obv_sma"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "sma_period": {"default": 20, "range": [5, 50], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["obv"] = ta.OBV(df)
        sma_period = params.get("sma_period", 20)
        df["obv_sma"] = ta.SMA(df, timeperiod=sma_period, price="obv")
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["obv"], df["obv_sma"])

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["obv"], df["obv_sma"])


class ADIndicator(BaseIndicatorOptimizer):
    """Chaikin A/D Line indicator."""

    indicator_name = "AD"
    talib_function = "ta.AD"
    category = "volume"
    outputs = ["ad", "ad_sma"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "sma_period": {"default": 20, "range": [5, 50], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["ad"] = ta.AD(df)
        sma_period = params.get("sma_period", 20)
        df["ad_sma"] = ta.SMA(df, timeperiod=sma_period, price="ad")
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["ad"], df["ad_sma"])

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["ad"], df["ad_sma"])


class ADOSCIndicator(BaseIndicatorOptimizer):
    """Chaikin A/D Oscillator indicator."""

    indicator_name = "ADOSC"
    talib_function = "ta.ADOSC"
    category = "volume"
    outputs = ["adosc"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "fastperiod": {"default": 3, "range": [2, 10], "type": "int"},
            "slowperiod": {"default": 10, "range": [5, 20], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["adosc"] = ta.ADOSC(
            df,
            fastperiod=params.get("fastperiod", 3),
            slowperiod=params.get("slowperiod", 10)
        )
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["adosc"], 0)

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["adosc"], 0)
