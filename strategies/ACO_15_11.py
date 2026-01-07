# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_15_11(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=7)
        dataframe['dema'] = ta.DEMA(dataframe, timeperiod=10)
        dataframe['sar'] = ta.SAR(dataframe, acceleration=0.02, maximum=0.2)
        dataframe['natr'] = ta.NATR(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=21)
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=7)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['cci'] < -80)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['dema'])
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['sar'])
        ) & (
            (dataframe['natr'] > 1.5)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['rsi'] > 65)
        ) & (
            (dataframe['mfi'] > 80)
        ) & (
            qtpylib.crossed_below(dataframe['ema_fast'], dataframe['ema_slow'])
        ),
        'exit_long'] = 1
        return dataframe
