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
    # New momentum indicators
    ADXRIndicator,
    DXIndicator,
    MACDEXTIndicator,
    MACDFIXIndicator,
    MINUS_DIIndicator,
    MINUS_DMIndicator,
    PLUS_DIIndicator,
    PLUS_DMIndicator,
    ROCPIndicator,
    ROCRIndicator,
    ROCR100Indicator,
    STOCHFIndicator,
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
    # New overlap indicators
    HT_TRENDLINEIndicator,
    MAIndicator,
    MAMAIndicator,
    MIDPOINTIndicator,
    MIDPRICEIndicator,
    SAREXTIndicator,
)

from .volume import (
    OBVIndicator,
    ADIndicator,
    ADOSCIndicator,
)

from .volatility import (
    ATRIndicator,
    NATRIndicator,
    # New volatility indicator
    TRANGEIndicator,
)

from .cycle import (
    HT_DCPERIODIndicator,
    HT_DCPHASEIndicator,
    HT_PHASORIndicator,
    HT_SINEIndicator,
    HT_TRENDMODEIndicator,
)

# Registry of all available indicators
INDICATOR_REGISTRY = {
    # Momentum (30 indicators)
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
    "ADXR": ADXRIndicator,
    "DX": DXIndicator,
    "MACDEXT": MACDEXTIndicator,
    "MACDFIX": MACDFIXIndicator,
    "MINUS_DI": MINUS_DIIndicator,
    "MINUS_DM": MINUS_DMIndicator,
    "PLUS_DI": PLUS_DIIndicator,
    "PLUS_DM": PLUS_DMIndicator,
    "ROCP": ROCPIndicator,
    "ROCR": ROCRIndicator,
    "ROCR100": ROCR100Indicator,
    "STOCHF": STOCHFIndicator,
    # Overlap (16 indicators)
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
    "HT_TRENDLINE": HT_TRENDLINEIndicator,
    "MA": MAIndicator,
    "MAMA": MAMAIndicator,
    "MIDPOINT": MIDPOINTIndicator,
    "MIDPRICE": MIDPRICEIndicator,
    "SAREXT": SAREXTIndicator,
    # Volume (3 indicators)
    "OBV": OBVIndicator,
    "AD": ADIndicator,
    "ADOSC": ADOSCIndicator,
    # Volatility (3 indicators)
    "ATR": ATRIndicator,
    "NATR": NATRIndicator,
    "TRANGE": TRANGEIndicator,
    # Cycle (5 indicators)
    "HT_DCPERIOD": HT_DCPERIODIndicator,
    "HT_DCPHASE": HT_DCPHASEIndicator,
    "HT_PHASOR": HT_PHASORIndicator,
    "HT_SINE": HT_SINEIndicator,
    "HT_TRENDMODE": HT_TRENDMODEIndicator,
}


def get_indicator(name: str):
    """Get indicator class by name."""
    if name not in INDICATOR_REGISTRY:
        raise ValueError(f"Unknown indicator: {name}")
    return INDICATOR_REGISTRY[name]


def list_indicators():
    """List all available indicator names."""
    return list(INDICATOR_REGISTRY.keys())
