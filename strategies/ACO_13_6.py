# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_13_6(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=5)
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=30)
        dataframe['sar'] = ta.SAR(dataframe, acceleration=0.03, maximum=0.3)
        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['obv_sma'] = ta.SMA(dataframe, timeperiod=20, price='obv')
        dataframe['trix'] = ta.TRIX(dataframe, timeperiod=21)
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=21)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['rsi'] < 30)
        ) & (
            qtpylib.crossed_above(dataframe['mom'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['kama'])
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['sar'])
        ) & (
            qtpylib.crossed_above(dataframe['obv'], dataframe['obv_sma'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_below(dataframe['trix'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['ema_fast'], dataframe['ema_slow'])
        ),
        'exit_long'] = 1
        return dataframe
