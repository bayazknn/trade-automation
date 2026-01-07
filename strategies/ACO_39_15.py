# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_39_15(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=30)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        stoch = ta.STOCH(dataframe, fastk_period=21, slowk_period=5, slowd_period=5)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']
        dataframe['wma'] = ta.WMA(dataframe, timeperiod=20)
        dataframe['t3'] = ta.T3(dataframe, timeperiod=10, vfactor=0.9)
        bbands = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.5, nbdevdn=2.5)
        dataframe['upperband'] = bbands['upperband']
        dataframe['middleband'] = bbands['middleband']
        dataframe['lowerband'] = bbands['lowerband']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['close'], dataframe['kama'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['slowk'] > 75)
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['wma'])
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['t3'])
        ) & (
            (dataframe['close'] > dataframe['upperband'] * 1.0)
        ) & (
            (dataframe['close'] > dataframe['upperband'] * 0.98)
        ),
        'exit_long'] = 1
        return dataframe
