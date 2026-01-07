# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_16_3(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['dema'] = ta.DEMA(dataframe, timeperiod=10)
        dataframe['adosc'] = ta.ADOSC(dataframe, fastperiod=3, slowperiod=10)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=21)
        res = ta.AROON(dataframe, timeperiod=25)
        dataframe['aroondown'] = res.iloc[:, 0]
        dataframe['aroonup'] = res.iloc[:, 1]
        dataframe['natr'] = ta.NATR(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['adx'] > 30)
        ) & (
            qtpylib.crossed_above(dataframe['ema_fast'], dataframe['ema_slow'])
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['dema'])
        ) & (
            qtpylib.crossed_above(dataframe['adosc'], 0)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['rsi'] > 65)
        ) & (
            qtpylib.crossed_below(dataframe['aroonup'], dataframe['aroondown'])
        ),
        'exit_long'] = 1
        return dataframe
