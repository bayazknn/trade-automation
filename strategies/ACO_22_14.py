# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_22_14(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['sar'] = ta.SAR(dataframe, acceleration=0.03, maximum=0.3)
        dataframe['ad'] = ta.AD(dataframe)
        dataframe['ad_sma'] = ta.SMA(dataframe, timeperiod=10, price='ad')
        dataframe['adosc'] = ta.ADOSC(dataframe, fastperiod=3, slowperiod=10)
        dataframe['natr'] = ta.NATR(dataframe, timeperiod=14)
        stochrsi = ta.STOCHRSI(dataframe, timeperiod=7, fastk_period=3, fastd_period=3)
        dataframe['fastk'] = stochrsi['fastk']
        dataframe['fastd'] = stochrsi['fastd']
        res = ta.AROON(dataframe, timeperiod=25)
        dataframe['aroondown'] = res.iloc[:, 0]
        dataframe['aroonup'] = res.iloc[:, 1]
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['close'], dataframe['sar'])
        ) & (
            qtpylib.crossed_above(dataframe['ad'], dataframe['ad_sma'])
        ) & (
            qtpylib.crossed_above(dataframe['adosc'], 0)
        ) & (
            (dataframe['natr'] > 3.0)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['fastk'] > 85)
        ) & (
            qtpylib.crossed_below(dataframe['aroonup'], dataframe['aroondown'])
        ),
        'exit_long'] = 1
        return dataframe
