"""
Technical Indicator Implementations

Each indicator class inherits from BaseIndicatorOptimizer and implements
indicator calculation and signal generation logic.
"""

from .momentum import (
    RSIIndicator,
    MACDIndicator,
    STOCHIndicator,
    STOCHRSIIndicator,
    CCIIndicator,
    MFIIndicator,
    WILLRIndicator,
    CMOIndicator,
    ADXIndicator,
    MOMIndicator,
    ROCIndicator,
    TRIXIndicator,
    ULTOSCIndicator,
    APOIndicator,
    PPOIndicator,
    BOPIndicator,
    AROONIndicator,
    AROONOSCIndicator,
)

from .overlap import (
    SMAIndicator,
    EMAIndicator,
    DEMAIndicator,
    TEMAIndicator,
    KAMAIndicator,
    WMAIndicator,
    TRIMAIndicator,
    T3Indicator,
    SARIndicator,
    BBANDSIndicator,
)

from .volume import (
    OBVIndicator,
    ADIndicator,
    ADOSCIndicator,
)

from .volatility import (
    ATRIndicator,
    NATRIndicator,
)

# Registry of all available indicators
INDICATOR_REGISTRY = {
    # Momentum
    "RSI": RSIIndicator,
    "MACD": MACDIndicator,
    "STOCH": STOCHIndicator,
    "STOCHRSI": STOCHRSIIndicator,
    "CCI": CCIIndicator,
    "MFI": MFIIndicator,
    "WILLR": WILLRIndicator,
    "CMO": CMOIndicator,
    "ADX": ADXIndicator,
    "MOM": MOMIndicator,
    "ROC": ROCIndicator,
    "TRIX": TRIXIndicator,
    "ULTOSC": ULTOSCIndicator,
    "APO": APOIndicator,
    "PPO": PPOIndicator,
    "BOP": BOPIndicator,
    "AROON": AROONIndicator,
    "AROONOSC": AROONOSCIndicator,
    # Overlap
    "SMA": SMAIndicator,
    "EMA": EMAIndicator,
    "DEMA": DEMAIndicator,
    "TEMA": TEMAIndicator,
    "KAMA": KAMAIndicator,
    "WMA": WMAIndicator,
    "TRIMA": TRIMAIndicator,
    "T3": T3Indicator,
    "SAR": SARIndicator,
    "BBANDS": BBANDSIndicator,
    # Volume
    "OBV": OBVIndicator,
    "AD": ADIndicator,
    "ADOSC": ADOSCIndicator,
    # Volatility
    "ATR": ATRIndicator,
    "NATR": NATRIndicator,
}


def get_indicator(name: str):
    """Get indicator class by name."""
    if name not in INDICATOR_REGISTRY:
        raise ValueError(f"Unknown indicator: {name}")
    return INDICATOR_REGISTRY[name]


def list_indicators():
    """List all available indicator names."""
    return list(INDICATOR_REGISTRY.keys())
