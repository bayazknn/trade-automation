# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_7_2(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['dema'] = ta.DEMA(dataframe, timeperiod=10)
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=10)
        dataframe['ad'] = ta.AD(dataframe)
        dataframe['ad_sma'] = ta.SMA(dataframe, timeperiod=20, price='ad')
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['adx'] > 25)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['dema'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['adx'] < 25)
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['kama'])
        ) & (
            qtpylib.crossed_below(dataframe['ad'], dataframe['ad_sma'])
        ),
        'exit_long'] = 1
        return dataframe
