# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_7_8(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        stoch = ta.STOCH(dataframe, fastk_period=5, slowk_period=3, slowd_period=3)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']
        dataframe['sma_fast'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma_slow'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['dema'] = ta.DEMA(dataframe, timeperiod=10)
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=20)
        dataframe['t3'] = ta.T3(dataframe, timeperiod=10, vfactor=0.9)
        stochrsi = ta.STOCHRSI(dataframe, timeperiod=7, fastk_period=3, fastd_period=3)
        dataframe['fastk'] = stochrsi['fastk']
        dataframe['fastd'] = stochrsi['fastd']
        dataframe['bop'] = ta.BOP(dataframe)
        bbands = ta.BBANDS(dataframe, timeperiod=20, nbdevup=1.5, nbdevdn=1.5)
        dataframe['upperband'] = bbands['upperband']
        dataframe['middleband'] = bbands['middleband']
        dataframe['lowerband'] = bbands['lowerband']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['slowk'] < 15)
        ) & (
            qtpylib.crossed_above(dataframe['sma_fast'], dataframe['sma_slow'])
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['dema'])
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['kama'])
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['t3'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['fastk'] > 85)
        ) & (
            qtpylib.crossed_below(dataframe['bop'], 0)
        ) & (
            (dataframe['close'] > dataframe['upperband'] * 1.0)
        ),
        'exit_long'] = 1
        return dataframe
