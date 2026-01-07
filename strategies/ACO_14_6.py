# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_14_6(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        macd = ta.MACD(dataframe, fastperiod=19, slowperiod=39, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=7)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=30)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        bbands = ta.BBANDS(dataframe, timeperiod=14, nbdevup=2.0, nbdevdn=2.0)
        dataframe['upperband'] = bbands['upperband']
        dataframe['middleband'] = bbands['middleband']
        dataframe['lowerband'] = bbands['lowerband']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])
        ) & (
            (dataframe['willr'] < -75)
        ) & (
            (dataframe['adx'] > 30)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['kama'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['cci'] > 150)
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['kama'])
        ) & (
            (dataframe['close'] > dataframe['upperband'] * 0.98)
        ),
        'exit_long'] = 1
        return dataframe
