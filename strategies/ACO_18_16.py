# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_18_16(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        macd = ta.MACD(dataframe, fastperiod=19, slowperiod=39, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=7)
        dataframe['sma_fast'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['sma_slow'] = ta.SMA(dataframe, timeperiod=21)
        bbands = ta.BBANDS(dataframe, timeperiod=20, nbdevup=1.5, nbdevdn=1.5)
        dataframe['upperband'] = bbands['upperband']
        dataframe['middleband'] = bbands['middleband']
        dataframe['lowerband'] = bbands['lowerband']
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=20)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])
        ) & (
            (dataframe['cci'] < -80)
        ) & (
            qtpylib.crossed_above(dataframe['sma_fast'], dataframe['sma_slow'])
        ) & (
            (dataframe['close'] < dataframe['lowerband'] * 1.0)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['adx'] < 25)
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['kama'])
        ) & (
            (dataframe['close'] > dataframe['upperband'] * 1.0)
        ),
        'exit_long'] = 1
        return dataframe
