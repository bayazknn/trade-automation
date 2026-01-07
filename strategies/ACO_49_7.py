# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_49_7(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['dema'] = ta.DEMA(dataframe, timeperiod=10)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['natr'] = ta.NATR(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe, fastperiod=8, slowperiod=17, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        stoch = ta.STOCH(dataframe, fastk_period=21, slowk_period=5, slowd_period=5)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']
        dataframe['apo'] = ta.APO(dataframe, fastperiod=5, slowperiod=20)
        dataframe['t3'] = ta.T3(dataframe, timeperiod=5, vfactor=0.7)
        dataframe['sar'] = ta.SAR(dataframe, acceleration=0.03, maximum=0.3)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['close'], dataframe['dema'])
        ) & (
            (dataframe['natr'] > 2.0)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])
        ) & (
            (dataframe['slowk'] > 75)
        ) & (
            qtpylib.crossed_below(dataframe['apo'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['t3'])
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['sar'])
        ),
        'exit_long'] = 1
        return dataframe
