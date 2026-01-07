# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_3_17(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        stochrsi = ta.STOCHRSI(dataframe, timeperiod=7, fastk_period=3, fastd_period=3)
        dataframe['fastk'] = stochrsi['fastk']
        dataframe['fastd'] = stochrsi['fastd']
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=5)
        dataframe['ultosc'] = ta.ULTOSC(dataframe, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=30)
        bbands = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.5, nbdevdn=2.5)
        dataframe['upperband'] = bbands['upperband']
        dataframe['middleband'] = bbands['middleband']
        dataframe['lowerband'] = bbands['lowerband']
        res = ta.AROON(dataframe, timeperiod=10)
        dataframe['aroondown'] = res.iloc[:, 0]
        dataframe['aroonup'] = res.iloc[:, 1]
        dataframe['sar'] = ta.SAR(dataframe, acceleration=0.02, maximum=0.2)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['fastk'] < 15)
        ) & (
            qtpylib.crossed_above(dataframe['roc'], 0)
        ) & (
            (dataframe['ultosc'] < 30)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['kama'])
        ) & (
            (dataframe['close'] < dataframe['lowerband'] * 1.0)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_below(dataframe['aroonup'], dataframe['aroondown'])
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['sar'])
        ),
        'exit_long'] = 1
        return dataframe
