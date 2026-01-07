# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_6_4(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=10)
        dataframe['adosc'] = ta.ADOSC(dataframe, fastperiod=2, slowperiod=5)
        dataframe['trix'] = ta.TRIX(dataframe, timeperiod=15)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['rsi'] < 30)
        ) & (
            qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['kama'])
        ) & (
            qtpylib.crossed_above(dataframe['adosc'], 0)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['rsi'] > 75)
        ) & (
            qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])
        ) & (
            qtpylib.crossed_below(dataframe['trix'], 0)
        ),
        'exit_long'] = 1
        return dataframe
