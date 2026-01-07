# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_58_8(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['bop'] = ta.BOP(dataframe)
        dataframe['natr'] = ta.NATR(dataframe, timeperiod=14)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['apo'] = ta.APO(dataframe, fastperiod=5, slowperiod=20)
        dataframe['aroonosc'] = ta.AROONOSC(dataframe, timeperiod=10)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['bop'], 0)
        ) & (
            (dataframe['natr'] > 2.0)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['cci'] > 150)
        ) & (
            qtpylib.crossed_below(dataframe['apo'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['aroonosc'], 0)
        ),
        'exit_long'] = 1
        return dataframe
