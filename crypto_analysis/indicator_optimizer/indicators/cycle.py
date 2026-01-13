"""
Cycle Indicators

Implements Hilbert Transform cycle-based technical indicators including:
HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE
"""

from typing import Dict

import pandas as pd
import talib.abstract as ta

from ..base import BaseIndicatorOptimizer


class HT_DCPERIODIndicator(BaseIndicatorOptimizer):
    """Hilbert Transform - Dominant Cycle Period indicator."""

    indicator_name = "HT_DCPERIOD"
    talib_function = "ta.HT_DCPERIOD"
    category = "cycle"
    outputs = ["ht_dcperiod"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "entry_constant": {"default": 20, "range": [10, 40], "type": "float"},
            "exit_constant": {"default": 30, "range": [20, 50], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["ht_dcperiod"] = ta.HT_DCPERIOD(df)
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # Entry when cycle period is short (fast cycles)
        constant = params.get("entry_constant", 20)
        return df["ht_dcperiod"] < constant

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # Exit when cycle period is long (slow cycles)
        constant = params.get("exit_constant", 30)
        return df["ht_dcperiod"] > constant


class HT_DCPHASEIndicator(BaseIndicatorOptimizer):
    """Hilbert Transform - Dominant Cycle Phase indicator."""

    indicator_name = "HT_DCPHASE"
    talib_function = "ta.HT_DCPHASE"
    category = "cycle"
    outputs = ["ht_dcphase"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "entry_constant": {"default": -45, "range": [-90, 0], "type": "float"},
            "exit_constant": {"default": 45, "range": [0, 90], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["ht_dcphase"] = ta.HT_DCPHASE(df)
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # Entry at specific phase (e.g., beginning of cycle)
        constant = params.get("entry_constant", -45)
        return df["ht_dcphase"] < constant

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # Exit at different phase (e.g., end of cycle)
        constant = params.get("exit_constant", 45)
        return df["ht_dcphase"] > constant


class HT_PHASORIndicator(BaseIndicatorOptimizer):
    """Hilbert Transform - Phasor Components indicator."""

    indicator_name = "HT_PHASOR"
    talib_function = "ta.HT_PHASOR"
    category = "cycle"
    outputs = ["inphase", "quadrature"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {}  # No optimizable parameters

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        phasor = ta.HT_PHASOR(df)
        df["inphase"] = phasor["inphase"]
        df["quadrature"] = phasor["quadrature"]
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # Entry when inphase crosses above quadrature
        return self._crossed_above(df["inphase"], df["quadrature"])

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # Exit when inphase crosses below quadrature
        return self._crossed_below(df["inphase"], df["quadrature"])


class HT_SINEIndicator(BaseIndicatorOptimizer):
    """Hilbert Transform - SineWave indicator."""

    indicator_name = "HT_SINE"
    talib_function = "ta.HT_SINE"
    category = "cycle"
    outputs = ["sine", "leadsine"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {}  # No optimizable parameters

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        sine = ta.HT_SINE(df)
        df["sine"] = sine["sine"]
        df["leadsine"] = sine["leadsine"]
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # Entry when sine crosses above leadsine (cycle turning up)
        return self._crossed_above(df["sine"], df["leadsine"])

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # Exit when sine crosses below leadsine (cycle turning down)
        return self._crossed_below(df["sine"], df["leadsine"])


class HT_TRENDMODEIndicator(BaseIndicatorOptimizer):
    """Hilbert Transform - Trend vs Cycle Mode indicator."""

    indicator_name = "HT_TRENDMODE"
    talib_function = "ta.HT_TRENDMODE"
    category = "cycle"
    outputs = ["ht_trendmode"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {}  # No optimizable parameters (output is 0 or 1)

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["ht_trendmode"] = ta.HT_TRENDMODE(df)
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # Entry when switching to trend mode (0 -> 1)
        prev_mode = df["ht_trendmode"].shift(1)
        return (prev_mode == 0) & (df["ht_trendmode"] == 1)

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # Exit when switching to cycle mode (1 -> 0)
        prev_mode = df["ht_trendmode"].shift(1)
        return (prev_mode == 1) & (df["ht_trendmode"] == 0)
