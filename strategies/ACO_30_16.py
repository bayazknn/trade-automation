# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_30_16(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=20)
        dataframe['adosc'] = ta.ADOSC(dataframe, fastperiod=2, slowperiod=5)
        dataframe['natr'] = ta.NATR(dataframe, timeperiod=14)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)
        dataframe['ultosc'] = ta.ULTOSC(dataframe, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        dataframe['t3'] = ta.T3(dataframe, timeperiod=10, vfactor=0.9)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['mom'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['adosc'], 0)
        ) & (
            (dataframe['natr'] > 2.0)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_below(dataframe['plus_di'], dataframe['minus_di'])
        ) & (
            (dataframe['ultosc'] > 65)
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['t3'])
        ),
        'exit_long'] = 1
        return dataframe
