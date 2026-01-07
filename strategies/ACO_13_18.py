# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_13_18(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['ultosc'] = ta.ULTOSC(dataframe, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=7)
        dataframe['adosc'] = ta.ADOSC(dataframe, fastperiod=3, slowperiod=10)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['cci'] < -150)
        ) & (
            (dataframe['ultosc'] < 30)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['cci'] > 100)
        ) & (
            (dataframe['willr'] > -25)
        ) & (
            qtpylib.crossed_below(dataframe['adosc'], 0)
        ),
        'exit_long'] = 1
        return dataframe
