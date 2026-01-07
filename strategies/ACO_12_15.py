# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_12_15(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        macd = ta.MACD(dataframe, fastperiod=8, slowperiod=17, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=7)
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=14)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=10)
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        dataframe['trix'] = ta.TRIX(dataframe, timeperiod=15)
        dataframe['ultosc'] = ta.ULTOSC(dataframe, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])
        ) & (
            (dataframe['cci'] < -80)
        ) & (
            (dataframe['willr'] < -80)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['tema'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['mfi'] > 75)
        ) & (
            qtpylib.crossed_below(dataframe['trix'], 0)
        ) & (
            (dataframe['ultosc'] > 65)
        ),
        'exit_long'] = 1
        return dataframe
