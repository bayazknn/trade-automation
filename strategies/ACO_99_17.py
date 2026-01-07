# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_99_17(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['bop'] = ta.BOP(dataframe)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=7)
        dataframe['ultosc'] = ta.ULTOSC(dataframe, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        dataframe['aroonosc'] = ta.AROONOSC(dataframe, timeperiod=10)
        dataframe['trima'] = ta.TRIMA(dataframe, timeperiod=20)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['adx'] > 25)
        ) & (
            qtpylib.crossed_above(dataframe['bop'], 0)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['willr'] > -25)
        ) & (
            (dataframe['ultosc'] > 65)
        ) & (
            qtpylib.crossed_below(dataframe['aroonosc'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['trima'])
        ),
        'exit_long'] = 1
        return dataframe
