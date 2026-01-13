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


class ADXRIndicator(BaseIndicatorOptimizer):
    """Average Directional Movement Index Rating indicator."""

    indicator_name = "ADXR"
    talib_function = "ta.ADXR"
    category = "momentum"
    outputs = ["adxr"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 14, "range": [7, 28], "type": "int"},
            "entry_constant": {"default": 25, "range": [20, 40], "type": "float"},
            "exit_constant": {"default": 20, "range": [15, 25], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["adxr"] = ta.ADXR(df, timeperiod=params.get("timeperiod", 14))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("entry_constant", 25)
        return df["adxr"] > constant

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("exit_constant", 20)
        return df["adxr"] < constant


class DXIndicator(BaseIndicatorOptimizer):
    """Directional Movement Index indicator."""

    indicator_name = "DX"
    talib_function = "ta.DX"
    category = "momentum"
    outputs = ["dx"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 14, "range": [5, 30], "type": "int"},
            "entry_constant": {"default": 25, "range": [20, 40], "type": "float"},
            "exit_constant": {"default": 20, "range": [10, 25], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["dx"] = ta.DX(df, timeperiod=params.get("timeperiod", 14))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("entry_constant", 25)
        return df["dx"] > constant

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("exit_constant", 20)
        return df["dx"] < constant


class MACDEXTIndicator(BaseIndicatorOptimizer):
    """MACD with controllable MA type indicator."""

    indicator_name = "MACDEXT"
    talib_function = "ta.MACDEXT"
    category = "momentum"
    outputs = ["macd", "macdsignal", "macdhist"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "fastperiod": {"default": 12, "range": [5, 20], "type": "int"},
            "slowperiod": {"default": 26, "range": [15, 40], "type": "int"},
            "signalperiod": {"default": 9, "range": [5, 15], "type": "int"},
            "fastmatype": {"default": 0, "range": [0, 8], "type": "int"},
            "slowmatype": {"default": 0, "range": [0, 8], "type": "int"},
            "signalmatype": {"default": 0, "range": [0, 8], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        macdext = ta.MACDEXT(
            df,
            fastperiod=params.get("fastperiod", 12),
            slowperiod=params.get("slowperiod", 26),
            signalperiod=params.get("signalperiod", 9),
            fastmatype=params.get("fastmatype", 0),
            slowmatype=params.get("slowmatype", 0),
            signalmatype=params.get("signalmatype", 0)
        )
        df["macd"] = macdext["macd"]
        df["macdsignal"] = macdext["macdsignal"]
        df["macdhist"] = macdext["macdhist"]
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["macd"], df["macdsignal"])

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["macd"], df["macdsignal"])


class MACDFIXIndicator(BaseIndicatorOptimizer):
    """Moving Average Convergence/Divergence Fix 12/26 indicator."""

    indicator_name = "MACDFIX"
    talib_function = "ta.MACDFIX"
    category = "momentum"
    outputs = ["macd", "macdsignal", "macdhist"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "signalperiod": {"default": 9, "range": [5, 15], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        macdfix = ta.MACDFIX(
            df,
            signalperiod=params.get("signalperiod", 9)
        )
        df["macd"] = macdfix["macd"]
        df["macdsignal"] = macdfix["macdsignal"]
        df["macdhist"] = macdfix["macdhist"]
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["macd"], df["macdsignal"])

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["macd"], df["macdsignal"])


class MINUS_DIIndicator(BaseIndicatorOptimizer):
    """Minus Directional Indicator."""

    indicator_name = "MINUS_DI"
    talib_function = "ta.MINUS_DI"
    category = "momentum"
    outputs = ["minus_di"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 14, "range": [5, 30], "type": "int"},
            "entry_constant": {"default": 25, "range": [15, 40], "type": "float"},
            "exit_constant": {"default": 20, "range": [10, 30], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["minus_di"] = ta.MINUS_DI(df, timeperiod=params.get("timeperiod", 14))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # Low minus_di indicates weak downtrend (potential entry)
        constant = params.get("entry_constant", 25)
        return df["minus_di"] < constant

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # High minus_di indicates strong downtrend (exit)
        constant = params.get("exit_constant", 20)
        return df["minus_di"] > constant


class MINUS_DMIndicator(BaseIndicatorOptimizer):
    """Minus Directional Movement indicator."""

    indicator_name = "MINUS_DM"
    talib_function = "ta.MINUS_DM"
    category = "momentum"
    outputs = ["minus_dm"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 14, "range": [5, 30], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["minus_dm"] = ta.MINUS_DM(df, timeperiod=params.get("timeperiod", 14))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # Entry when minus_dm crosses below its previous value (downtrend weakening)
        return self._crossed_below(df["minus_dm"], df["minus_dm"].shift(1))

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # Exit when minus_dm crosses above its previous value (downtrend strengthening)
        return self._crossed_above(df["minus_dm"], df["minus_dm"].shift(1))


class PLUS_DIIndicator(BaseIndicatorOptimizer):
    """Plus Directional Indicator."""

    indicator_name = "PLUS_DI"
    talib_function = "ta.PLUS_DI"
    category = "momentum"
    outputs = ["plus_di"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 14, "range": [5, 30], "type": "int"},
            "entry_constant": {"default": 25, "range": [15, 40], "type": "float"},
            "exit_constant": {"default": 20, "range": [10, 30], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["plus_di"] = ta.PLUS_DI(df, timeperiod=params.get("timeperiod", 14))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # High plus_di indicates strong uptrend (entry)
        constant = params.get("entry_constant", 25)
        return df["plus_di"] > constant

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # Low plus_di indicates weak uptrend (exit)
        constant = params.get("exit_constant", 20)
        return df["plus_di"] < constant


class PLUS_DMIndicator(BaseIndicatorOptimizer):
    """Plus Directional Movement indicator."""

    indicator_name = "PLUS_DM"
    talib_function = "ta.PLUS_DM"
    category = "momentum"
    outputs = ["plus_dm"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 14, "range": [5, 30], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["plus_dm"] = ta.PLUS_DM(df, timeperiod=params.get("timeperiod", 14))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # Entry when plus_dm crosses above its previous value (uptrend strengthening)
        return self._crossed_above(df["plus_dm"], df["plus_dm"].shift(1))

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        # Exit when plus_dm crosses below its previous value (uptrend weakening)
        return self._crossed_below(df["plus_dm"], df["plus_dm"].shift(1))


class ROCPIndicator(BaseIndicatorOptimizer):
    """Rate of Change Percentage indicator."""

    indicator_name = "ROCP"
    talib_function = "ta.ROCP"
    category = "momentum"
    outputs = ["rocp"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 10, "range": [5, 30], "type": "int"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["rocp"] = ta.ROCP(df, timeperiod=params.get("timeperiod", 10))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_above(df["rocp"], 0)

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        return self._crossed_below(df["rocp"], 0)


class ROCRIndicator(BaseIndicatorOptimizer):
    """Rate of Change Ratio indicator."""

    indicator_name = "ROCR"
    talib_function = "ta.ROCR"
    category = "momentum"
    outputs = ["rocr"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 10, "range": [5, 30], "type": "int"},
            "entry_constant": {"default": 1.02, "range": [1.0, 1.05], "type": "float"},
            "exit_constant": {"default": 0.98, "range": [0.95, 1.0], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["rocr"] = ta.ROCR(df, timeperiod=params.get("timeperiod", 10))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("entry_constant", 1.02)
        return df["rocr"] > constant

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("exit_constant", 0.98)
        return df["rocr"] < constant


class ROCR100Indicator(BaseIndicatorOptimizer):
    """Rate of Change Ratio 100 scale indicator."""

    indicator_name = "ROCR100"
    talib_function = "ta.ROCR100"
    category = "momentum"
    outputs = ["rocr100"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "timeperiod": {"default": 10, "range": [5, 30], "type": "int"},
            "entry_constant": {"default": 102, "range": [100, 105], "type": "float"},
            "exit_constant": {"default": 98, "range": [95, 100], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df["rocr100"] = ta.ROCR100(df, timeperiod=params.get("timeperiod", 10))
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("entry_constant", 102)
        return df["rocr100"] > constant

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("exit_constant", 98)
        return df["rocr100"] < constant


class STOCHFIndicator(BaseIndicatorOptimizer):
    """Stochastic Fast indicator."""

    indicator_name = "STOCHF"
    talib_function = "ta.STOCHF"
    category = "momentum"
    outputs = ["fastk", "fastd"]

    def get_optimizable_params(self) -> Dict[str, Dict]:
        return {
            "fastk_period": {"default": 5, "range": [3, 14], "type": "int"},
            "fastd_period": {"default": 3, "range": [1, 5], "type": "int"},
            "entry_constant": {"default": 20, "range": [10, 30], "type": "float"},
            "exit_constant": {"default": 80, "range": [70, 90], "type": "float"},
        }

    def calculate_indicator(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        stochf = ta.STOCHF(
            df,
            fastk_period=params.get("fastk_period", 5),
            fastd_period=params.get("fastd_period", 3)
        )
        df["fastk"] = stochf["fastk"]
        df["fastd"] = stochf["fastd"]
        return df

    def generate_entry_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("entry_constant", 20)
        return df["fastk"] < constant

    def generate_exit_signal(self, df: pd.DataFrame, **params) -> pd.Series:
        constant = params.get("exit_constant", 80)
        return df["fastk"] > constant
