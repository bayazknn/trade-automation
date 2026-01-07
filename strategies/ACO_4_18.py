# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_4_18(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['cmo'] = ta.CMO(dataframe, timeperiod=14)
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=10)
        bbands = ta.BBANDS(dataframe, timeperiod=14, nbdevup=2.0, nbdevdn=2.0)
        dataframe['upperband'] = bbands['upperband']
        dataframe['middleband'] = bbands['middleband']
        dataframe['lowerband'] = bbands['lowerband']
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=10)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['cmo'] < -40)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['kama'])
        ) & (
            (dataframe['close'] < dataframe['lowerband'] * 1.02)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_below(dataframe['mom'], 0)
        ) & (
            (dataframe['close'] > dataframe['upperband'] * 1.0)
        ),
        'exit_long'] = 1
        return dataframe
