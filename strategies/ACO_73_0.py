# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_73_0(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['bop'] = ta.BOP(dataframe)
        dataframe['dema'] = ta.DEMA(dataframe, timeperiod=10)
        dataframe['cmo'] = ta.CMO(dataframe, timeperiod=14)
        res = ta.AROON(dataframe, timeperiod=25)
        dataframe['aroondown'] = res.iloc[:, 0]
        dataframe['aroonup'] = res.iloc[:, 1]
        dataframe['wma'] = ta.WMA(dataframe, timeperiod=20)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['bop'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['dema'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['cmo'] > 50)
        ) & (
            qtpylib.crossed_below(dataframe['aroonup'], dataframe['aroondown'])
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['wma'])
        ),
        'exit_long'] = 1
        return dataframe
