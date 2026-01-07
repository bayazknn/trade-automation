# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_2_13(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ppo'] = ta.PPO(dataframe, fastperiod=12, slowperiod=26)
        dataframe['sma_fast'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['sma_slow'] = ta.SMA(dataframe, timeperiod=21)
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=30)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['apo'] = ta.APO(dataframe, fastperiod=5, slowperiod=20)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=20)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['ppo'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['sma_fast'], dataframe['sma_slow'])
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['kama'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['adx'] < 20)
        ) & (
            qtpylib.crossed_below(dataframe['apo'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['tema'])
        ),
        'exit_long'] = 1
        return dataframe
