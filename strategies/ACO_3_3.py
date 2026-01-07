# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_3_3(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        res = ta.AROON(dataframe, timeperiod=14)
        dataframe['aroondown'] = res.iloc[:, 0]
        dataframe['aroonup'] = res.iloc[:, 1]
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=20)
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=30)
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=5)
        dataframe['trix'] = ta.TRIX(dataframe, timeperiod=9)
        dataframe['ultosc'] = ta.ULTOSC(dataframe, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        dataframe['sma_fast'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['sma_slow'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['wma'] = ta.WMA(dataframe, timeperiod=20)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['aroonup'], dataframe['aroondown'])
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['tema'])
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['kama'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_below(dataframe['mom'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['trix'], 0)
        ) & (
            (dataframe['ultosc'] > 65)
        ) & (
            qtpylib.crossed_below(dataframe['sma_fast'], dataframe['sma_slow'])
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['wma'])
        ),
        'exit_long'] = 1
        return dataframe
