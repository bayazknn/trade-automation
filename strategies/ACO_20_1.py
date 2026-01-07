# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_20_1(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['cmo'] = ta.CMO(dataframe, timeperiod=14)
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=30)
        dataframe['wma'] = ta.WMA(dataframe, timeperiod=20)
        dataframe['natr'] = ta.NATR(dataframe, timeperiod=14)
        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=5)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['cmo'] < -50)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['kama'])
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['wma'])
        ) & (
            (dataframe['natr'] > 2.0)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['slowk'] > 80)
        ) & (
            (dataframe['adx'] < 20)
        ) & (
            qtpylib.crossed_below(dataframe['mom'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['wma'])
        ),
        'exit_long'] = 1
        return dataframe
