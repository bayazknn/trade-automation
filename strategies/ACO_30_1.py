# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_30_1(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['trix'] = ta.TRIX(dataframe, timeperiod=9)
        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['obv_sma'] = ta.SMA(dataframe, timeperiod=10, price='obv')
        dataframe['natr'] = ta.NATR(dataframe, timeperiod=14)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)
        dataframe['t3'] = ta.T3(dataframe, timeperiod=5, vfactor=0.7)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['trix'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['obv'], dataframe['obv_sma'])
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
            qtpylib.crossed_below(dataframe['trix'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['t3'])
        ),
        'exit_long'] = 1
        return dataframe
