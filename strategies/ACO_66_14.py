# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_66_14(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['bop'] = ta.BOP(dataframe)
        bbands = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['upperband'] = bbands['upperband']
        dataframe['middleband'] = bbands['middleband']
        dataframe['lowerband'] = bbands['lowerband']
        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['obv_sma'] = ta.SMA(dataframe, timeperiod=10, price='obv')
        dataframe['natr'] = ta.NATR(dataframe, timeperiod=14)
        stochrsi = ta.STOCHRSI(dataframe, timeperiod=7, fastk_period=3, fastd_period=3)
        dataframe['fastk'] = stochrsi['fastk']
        dataframe['fastd'] = stochrsi['fastd']
        dataframe['t3'] = ta.T3(dataframe, timeperiod=10, vfactor=0.9)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['bop'], 0)
        ) & (
            (dataframe['close'] < dataframe['lowerband'] * 1.02)
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
            (dataframe['fastk'] > 85)
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['t3'])
        ),
        'exit_long'] = 1
        return dataframe
