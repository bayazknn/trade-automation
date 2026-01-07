# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_2_15(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['sma_fast'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['sma_slow'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['wma'] = ta.WMA(dataframe, timeperiod=10)
        dataframe['ad'] = ta.AD(dataframe)
        dataframe['ad_sma'] = ta.SMA(dataframe, timeperiod=10, price='ad')
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=14)
        dataframe['cmo'] = ta.CMO(dataframe, timeperiod=14)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=5)
        res = ta.AROON(dataframe, timeperiod=10)
        dataframe['aroondown'] = res.iloc[:, 0]
        dataframe['aroonup'] = res.iloc[:, 1]
        bbands = ta.BBANDS(dataframe, timeperiod=20, nbdevup=1.5, nbdevdn=1.5)
        dataframe['upperband'] = bbands['upperband']
        dataframe['middleband'] = bbands['middleband']
        dataframe['lowerband'] = bbands['lowerband']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['rsi'] < 25)
        ) & (
            qtpylib.crossed_above(dataframe['sma_fast'], dataframe['sma_slow'])
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['wma'])
        ) & (
            qtpylib.crossed_above(dataframe['ad'], dataframe['ad_sma'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['willr'] > -20)
        ) & (
            (dataframe['cmo'] > 40)
        ) & (
            qtpylib.crossed_below(dataframe['roc'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['aroonup'], dataframe['aroondown'])
        ) & (
            (dataframe['close'] > dataframe['upperband'] * 1.0)
        ),
        'exit_long'] = 1
        return dataframe
