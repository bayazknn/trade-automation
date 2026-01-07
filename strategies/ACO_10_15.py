# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_10_15(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=7)
        dataframe['trix'] = ta.TRIX(dataframe, timeperiod=9)
        dataframe['ppo'] = ta.PPO(dataframe, fastperiod=12, slowperiod=26)
        dataframe['aroonosc'] = ta.AROONOSC(dataframe, timeperiod=14)
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=7)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=20)
        dataframe['wma'] = ta.WMA(dataframe, timeperiod=20)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['cci'] < -80)
        ) & (
            qtpylib.crossed_above(dataframe['trix'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['ppo'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['aroonosc'], 0)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['willr'] > -25)
        ) & (
            (dataframe['adx'] < 20)
        ) & (
            qtpylib.crossed_below(dataframe['roc'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['wma'])
        ),
        'exit_long'] = 1
        return dataframe
