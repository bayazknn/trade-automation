# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_30_14(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=30)
        macd = ta.MACD(dataframe, fastperiod=8, slowperiod=17, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)
        dataframe['t3'] = ta.T3(dataframe, timeperiod=5, vfactor=0.7)
        dataframe['sar'] = ta.SAR(dataframe, acceleration=0.03, maximum=0.3)
        dataframe['adosc'] = ta.ADOSC(dataframe, fastperiod=3, slowperiod=10)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['adx'] > 30)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['kama'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])
        ) & (
            qtpylib.crossed_below(dataframe['plus_di'], dataframe['minus_di'])
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['t3'])
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['sar'])
        ) & (
            qtpylib.crossed_below(dataframe['adosc'], 0)
        ),
        'exit_long'] = 1
        return dataframe
