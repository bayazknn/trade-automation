# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_1_4(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['cmo'] = ta.CMO(dataframe, timeperiod=14)
        dataframe['trix'] = ta.TRIX(dataframe, timeperiod=15)
        bbands = ta.BBANDS(dataframe, timeperiod=14, nbdevup=2.0, nbdevdn=2.0)
        dataframe['upperband'] = bbands['upperband']
        dataframe['middleband'] = bbands['middleband']
        dataframe['lowerband'] = bbands['lowerband']
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=7)
        dataframe['apo'] = ta.APO(dataframe, fastperiod=12, slowperiod=26)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['cmo'] < -40)
        ) & (
            qtpylib.crossed_above(dataframe['trix'], 0)
        ) & (
            (dataframe['close'] < dataframe['lowerband'] * 1.02)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['cci'] > 80)
        ) & (
            qtpylib.crossed_below(dataframe['apo'], 0)
        ),
        'exit_long'] = 1
        return dataframe
