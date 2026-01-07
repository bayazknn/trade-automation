# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_38_16(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=20)
        dataframe['trix'] = ta.TRIX(dataframe, timeperiod=9)
        dataframe['dema'] = ta.DEMA(dataframe, timeperiod=10)
        dataframe['natr'] = ta.NATR(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=21)
        dataframe['ppo'] = ta.PPO(dataframe, fastperiod=5, slowperiod=20)
        dataframe['t3'] = ta.T3(dataframe, timeperiod=10, vfactor=0.9)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['adx'] > 30)
        ) & (
            qtpylib.crossed_above(dataframe['mom'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['trix'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['dema'])
        ) & (
            (dataframe['natr'] > 2.0)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['rsi'] > 65)
        ) & (
            qtpylib.crossed_below(dataframe['ppo'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['t3'])
        ),
        'exit_long'] = 1
        return dataframe
