# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_25_6(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['dema'] = ta.DEMA(dataframe, timeperiod=10)
        macd = ta.MACD(dataframe, fastperiod=19, slowperiod=39, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['aroonosc'] = ta.AROONOSC(dataframe, timeperiod=14)
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=30)
        dataframe['trima'] = ta.TRIMA(dataframe, timeperiod=20)
        dataframe['t3'] = ta.T3(dataframe, timeperiod=10, vfactor=0.9)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['adx'] > 30)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['dema'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])
        ) & (
            qtpylib.crossed_below(dataframe['aroonosc'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['kama'])
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['trima'])
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['t3'])
        ),
        'exit_long'] = 1
        return dataframe
