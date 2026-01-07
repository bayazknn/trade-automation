# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_56_9(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['sma_fast'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma_slow'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['dema'] = ta.DEMA(dataframe, timeperiod=10)
        dataframe['wma'] = ta.WMA(dataframe, timeperiod=20)
        dataframe['t3'] = ta.T3(dataframe, timeperiod=5, vfactor=0.7)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['sma_fast'], dataframe['sma_slow'])
        ) & (
            qtpylib.crossed_above(dataframe['ema_fast'], dataframe['ema_slow'])
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['dema'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_below(dataframe['close'], dataframe['wma'])
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['t3'])
        ),
        'exit_long'] = 1
        return dataframe
