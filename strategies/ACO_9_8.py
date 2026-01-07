# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_9_8(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=21)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['aroonosc'] = ta.AROONOSC(dataframe, timeperiod=14)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=10)
        dataframe['sar'] = ta.SAR(dataframe, acceleration=0.01, maximum=0.1)
        dataframe['cmo'] = ta.CMO(dataframe, timeperiod=14)
        dataframe['trix'] = ta.TRIX(dataframe, timeperiod=9)
        dataframe['sma_fast'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['sma_slow'] = ta.SMA(dataframe, timeperiod=21)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['rsi'] < 35)
        ) & (
            (dataframe['cci'] < -150)
        ) & (
            qtpylib.crossed_above(dataframe['aroonosc'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['tema'])
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['sar'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['cmo'] > 40)
        ) & (
            qtpylib.crossed_below(dataframe['trix'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['sma_fast'], dataframe['sma_slow'])
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['sar'])
        ),
        'exit_long'] = 1
        return dataframe
