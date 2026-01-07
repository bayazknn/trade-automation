# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_83_17(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['obv_sma'] = ta.SMA(dataframe, timeperiod=10, price='obv')
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe, fastperiod=8, slowperiod=17, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        stoch = ta.STOCH(dataframe, fastk_period=21, slowk_period=5, slowd_period=5)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=7)
        dataframe['t3'] = ta.T3(dataframe, timeperiod=10, vfactor=0.9)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['obv'], dataframe['obv_sma'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])
        ) & (
            (dataframe['slowk'] > 75)
        ) & (
            (dataframe['mfi'] > 80)
        ) & (
            (dataframe['willr'] > -25)
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['t3'])
        ),
        'exit_long'] = 1
        return dataframe
