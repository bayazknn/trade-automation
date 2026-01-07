# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_5_5(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=20)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=20)
        dataframe['wma'] = ta.WMA(dataframe, timeperiod=20)
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=14)
        dataframe['ultosc'] = ta.ULTOSC(dataframe, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['rsi'] < 30)
        ) & (
            qtpylib.crossed_above(dataframe['roc'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['tema'])
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['wma'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['mfi'] > 75)
        ) & (
            (dataframe['willr'] > -20)
        ) & (
            qtpylib.crossed_below(dataframe['roc'], 0)
        ) & (
            (dataframe['ultosc'] > 70)
        ),
        'exit_long'] = 1
        return dataframe
