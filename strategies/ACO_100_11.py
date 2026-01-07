# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_100_11(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['dema'] = ta.DEMA(dataframe, timeperiod=10)
        dataframe['adosc'] = ta.ADOSC(dataframe, fastperiod=3, slowperiod=10)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['natr'] = ta.NATR(dataframe, timeperiod=14)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=7)
        dataframe['sma_fast'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['sma_slow'] = ta.SMA(dataframe, timeperiod=21)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['adx'] > 25)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['dema'])
        ) & (
            qtpylib.crossed_above(dataframe['adosc'], 0)
        ) & (
            (dataframe['natr'] > 1.5)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['cci'] > 80)
        ) & (
            qtpylib.crossed_below(dataframe['sma_fast'], dataframe['sma_slow'])
        ),
        'exit_long'] = 1
        return dataframe
