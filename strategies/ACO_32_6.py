# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_32_6(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['dema'] = ta.DEMA(dataframe, timeperiod=10)
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=30)
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        res = ta.AROON(dataframe, timeperiod=14)
        dataframe['aroondown'] = res.iloc[:, 0]
        dataframe['aroonup'] = res.iloc[:, 1]
        dataframe['aroonosc'] = ta.AROONOSC(dataframe, timeperiod=14)
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        bbands = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['upperband'] = bbands['upperband']
        dataframe['middleband'] = bbands['middleband']
        dataframe['lowerband'] = bbands['lowerband']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['close'], dataframe['dema'])
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['kama'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])
        ) & (
            qtpylib.crossed_below(dataframe['aroonup'], dataframe['aroondown'])
        ) & (
            qtpylib.crossed_below(dataframe['aroonosc'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['ema_fast'], dataframe['ema_slow'])
        ) & (
            (dataframe['close'] > dataframe['upperband'] * 0.98)
        ),
        'exit_long'] = 1
        return dataframe
