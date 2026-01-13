"""
Momentum Indicators

Implements momentum-based technical indicators including:
RSI, MACD, STOCH, STOCHRSI, CCI, MFI, WILLR, CMO, ADX, MOM, ROC, TRIX, etc.
"""

from typing import Any, Dict, Optional

import pandas as pd
import talib.abstract as ta

from ..base import BaseIndicatorOptimizer


class RSIIndicator(BaseIndicatorOptimizer):
    """Relative Strength Index indicator."""

    indicator_name = "RSI"
    talib_function = "ta.RSI"
    category = "momentum"
    outputs = ["rsi"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 14, "range": [2, 30], "type": "int"},
            "entry_constant": {"default": 30, "range": [20, 40], "type": "float"},
            "exit_constant": {"default": 70, "range": [60, 80], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        timeperiod = params.get("timeperiod", 14)
        df["rsi"] = ta.RSI(df, timeperiod=timeperiod)
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("entry_constant", 30)
        return df["rsi"] < constant

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("exit_constant", 70)
        return df["rsi"] > constant


class MACDIndicator(BaseIndicatorOptimizer):
    """Moving Average Convergence Divergence indicator."""

    indicator_name = "MACD"
    talib_function = "ta.MACD"
    category = "momentum"
    outputs = ["macd", "macdsignal", "macdhist"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "fastperiod": {"default": 12, "range": [5, 20], "type": "int"},
            "slowperiod": {"default": 26, "range": [15, 40], "type": "int"},
            "signalperiod": {"default": 9, "range": [5, 15], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        macd = ta.MACD(
            df,
            fastperiod=params.get("fastperiod", 12),
            slowperiod=params.get("slowperiod", 26),
            signalperiod=params.get("signalperiod", 9)
        )
        df["macd"] = macd["macd"]
        df["macdsignal"] = macd["macdsignal"]
        df["macdhist"] = macd["macdhist"]
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["macd"], df["macdsignal"])

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["macd"], df["macdsignal"])


class STOCHIndicator(BaseIndicatorOptimizer):
    """Stochastic Oscillator indicator."""

    indicator_name = "STOCH"
    talib_function = "ta.STOCH"
    category = "momentum"
    outputs = ["slowk", "slowd"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "fastk_period": {"default": 14, "range": [5, 21], "type": "int"},
            "slowk_period": {"default": 3, "range": [1, 5], "type": "int"},
            "slowd_period": {"default": 3, "range": [1, 5], "type": "int"},
            "entry_constant": {"default": 20, "range": [10, 30], "type": "float"},
            "exit_constant": {"default": 80, "range": [70, 90], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        stoch = ta.STOCH(
            df,
            fastk_period=params.get("fastk_period", 14),
            slowk_period=params.get("slowk_period", 3),
            slowd_period=params.get("slowd_period", 3)
        )
        df["slowk"] = stoch["slowk"]
        df["slowd"] = stoch["slowd"]
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("entry_constant", 20)
        return df["slowk"] < constant

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("exit_constant", 80)
        return df["slowk"] > constant


class STOCHRSIIndicator(BaseIndicatorOptimizer):
    """Stochastic RSI indicator."""

    indicator_name = "STOCHRSI"
    talib_function = "ta.STOCHRSI"
    category = "momentum"
    outputs = ["fastk", "fastd"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 14, "range": [5, 21], "type": "int"},
            "fastk_period": {"default": 5, "range": [3, 10], "type": "int"},
            "fastd_period": {"default": 3, "range": [1, 5], "type": "int"},
            "entry_constant": {"default": 20, "range": [10, 30], "type": "float"},
            "exit_constant": {"default": 80, "range": [70, 90], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        stochrsi = ta.STOCHRSI(
            df,
            timeperiod=params.get("timeperiod", 14),
            fastk_period=params.get("fastk_period", 5),
            fastd_period=params.get("fastd_period", 3)
        )
        df["fastk"] = stochrsi["fastk"]
        df["fastd"] = stochrsi["fastd"]
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("entry_constant", 20)
        return df["fastk"] < constant

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("exit_constant", 80)
        return df["fastk"] > constant


class CCIIndicator(BaseIndicatorOptimizer):
    """Commodity Channel Index indicator."""

    indicator_name = "CCI"
    talib_function = "ta.CCI"
    category = "momentum"
    outputs = ["cci"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 14, "range": [5, 30], "type": "int"},
            "entry_constant": {"default": -100, "range": [-200, -50], "type": "float"},
            "exit_constant": {"default": 100, "range": [50, 200], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["cci"] = ta.CCI(df, timeperiod=params.get("timeperiod", 14))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("entry_constant", -100)
        return df["cci"] < constant

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("exit_constant", 100)
        return df["cci"] > constant


class MFIIndicator(BaseIndicatorOptimizer):
    """Money Flow Index indicator."""

    indicator_name = "MFI"
    talib_function = "ta.MFI"
    category = "momentum"
    outputs = ["mfi"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 14, "range": [5, 30], "type": "int"},
            "entry_constant": {"default": 20, "range": [10, 30], "type": "float"},
            "exit_constant": {"default": 80, "range": [70, 90], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["mfi"] = ta.MFI(df, timeperiod=params.get("timeperiod", 14))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("entry_constant", 20)
        return df["mfi"] < constant

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("exit_constant", 80)
        return df["mfi"] > constant


class WILLRIndicator(BaseIndicatorOptimizer):
    """Williams %R indicator."""

    indicator_name = "WILLR"
    talib_function = "ta.WILLR"
    category = "momentum"
    outputs = ["willr"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 14, "range": [5, 30], "type": "int"},
            "entry_constant": {"default": -80, "range": [-90, -70], "type": "float"},
            "exit_constant": {"default": -20, "range": [-30, -10], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["willr"] = ta.WILLR(df, timeperiod=params.get("timeperiod", 14))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("entry_constant", -80)
        return df["willr"] < constant

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("exit_constant", -20)
        return df["willr"] > constant


class CMOIndicator(BaseIndicatorOptimizer):
    """Chande Momentum Oscillator indicator."""

    indicator_name = "CMO"
    talib_function = "ta.CMO"
    category = "momentum"
    outputs = ["cmo"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 14, "range": [5, 30], "type": "int"},
            "entry_constant": {"default": -50, "range": [-70, -30], "type": "float"},
            "exit_constant": {"default": 50, "range": [30, 70], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["cmo"] = ta.CMO(df, timeperiod=params.get("timeperiod", 14))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("entry_constant", -50)
        return df["cmo"] < constant

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("exit_constant", 50)
        return df["cmo"] > constant


class ADXIndicator(BaseIndicatorOptimizer):
    """Average Directional Index indicator."""

    indicator_name = "ADX"
    talib_function = "ta.ADX"
    category = "momentum"
    outputs = ["adx"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 14, "range": [7, 28], "type": "int"},
            "entry_constant": {"default": 25, "range": [20, 40], "type": "float"},
            "exit_constant": {"default": 20, "range": [15, 25], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["adx"] = ta.ADX(df, timeperiod=params.get("timeperiod", 14))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("entry_constant", 25)
        return df["adx"] > constant

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("exit_constant", 20)
        return df["adx"] < constant


class MOMIndicator(BaseIndicatorOptimizer):
    """Momentum indicator."""

    indicator_name = "MOM"
    talib_function = "ta.MOM"
    category = "momentum"
    outputs = ["mom"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 10, "range": [5, 30], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["mom"] = ta.MOM(df, timeperiod=params.get("timeperiod", 10))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["mom"], 0)

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["mom"], 0)


class ROCIndicator(BaseIndicatorOptimizer):
    """Rate of Change indicator."""

    indicator_name = "ROC"
    talib_function = "ta.ROC"
    category = "momentum"
    outputs = ["roc"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 10, "range": [5, 30], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["roc"] = ta.ROC(df, timeperiod=params.get("timeperiod", 10))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["roc"], 0)

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["roc"], 0)


class TRIXIndicator(BaseIndicatorOptimizer):
    """Triple Exponential Average indicator."""

    indicator_name = "TRIX"
    talib_function = "ta.TRIX"
    category = "momentum"
    outputs = ["trix"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 15, "range": [5, 30], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["trix"] = ta.TRIX(df, timeperiod=params.get("timeperiod", 15))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["trix"], 0)

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["trix"], 0)


class ULTOSCIndicator(BaseIndicatorOptimizer):
    """Ultimate Oscillator indicator."""

    indicator_name = "ULTOSC"
    talib_function = "ta.ULTOSC"
    category = "momentum"
    outputs = ["ultosc"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod1": {"default": 7, "range": [3, 14], "type": "int"},
            "timeperiod2": {"default": 14, "range": [7, 28], "type": "int"},
            "timeperiod3": {"default": 28, "range": [14, 56], "type": "int"},
            "entry_constant": {"default": 30, "range": [20, 40], "type": "float"},
            "exit_constant": {"default": 70, "range": [60, 80], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["ultosc"] = ta.ULTOSC(
            df,
            timeperiod1=params.get("timeperiod1", 7),
            timeperiod2=params.get("timeperiod2", 14),
            timeperiod3=params.get("timeperiod3", 28)
        )
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("entry_constant", 30)
        return df["ultosc"] < constant

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("exit_constant", 70)
        return df["ultosc"] > constant


class APOIndicator(BaseIndicatorOptimizer):
    """Absolute Price Oscillator indicator."""

    indicator_name = "APO"
    talib_function = "ta.APO"
    category = "momentum"
    outputs = ["apo"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "fastperiod": {"default": 12, "range": [5, 20], "type": "int"},
            "slowperiod": {"default": 26, "range": [15, 40], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["apo"] = ta.APO(
            df,
            fastperiod=params.get("fastperiod", 12),
            slowperiod=params.get("slowperiod", 26)
        )
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["apo"], 0)

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["apo"], 0)


class PPOIndicator(BaseIndicatorOptimizer):
    """Percentage Price Oscillator indicator."""

    indicator_name = "PPO"
    talib_function = "ta.PPO"
    category = "momentum"
    outputs = ["ppo"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "fastperiod": {"default": 12, "range": [5, 20], "type": "int"},
            "slowperiod": {"default": 26, "range": [15, 40], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["ppo"] = ta.PPO(
            df,
            fastperiod=params.get("fastperiod", 12),
            slowperiod=params.get("slowperiod", 26)
        )
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["ppo"], 0)

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["ppo"], 0)


class BOPIndicator(BaseIndicatorOptimizer):
    """Balance of Power indicator."""

    indicator_name = "BOP"
    talib_function = "ta.BOP"
    category = "momentum"
    outputs = ["bop"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {}  # BOP has no parameters

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["bop"] = ta.BOP(df)
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["bop"], 0)

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["bop"], 0)


class AROONIndicator(BaseIndicatorOptimizer):
    """Aroon indicator."""

    indicator_name = "AROON"
    talib_function = "ta.AROON"
    category = "momentum"
    outputs = ["aroondown", "aroonup"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 14, "range": [5, 30], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        aroon = ta.AROON(df, timeperiod=params.get("timeperiod", 14))
        df["aroondown"] = aroon["aroondown"]
        df["aroonup"] = aroon["aroonup"]
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["aroonup"], df["aroondown"])

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["aroonup"], df["aroondown"])


class AROONOSCIndicator(BaseIndicatorOptimizer):
    """Aroon Oscillator indicator."""

    indicator_name = "AROONOSC"
    talib_function = "ta.AROONOSC"
    category = "momentum"
    outputs = ["aroonosc"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 14, "range": [5, 30], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["aroonosc"] = ta.AROONOSC(df, timeperiod=params.get("timeperiod", 14))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["aroonosc"], 0)

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["aroonosc"], 0)
