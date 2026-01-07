# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_15_16(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        stoch = ta.STOCH(dataframe, fastk_period=5, slowk_period=3, slowd_period=3)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']
        dataframe['ppo'] = ta.PPO(dataframe, fastperiod=5, slowperiod=20)
        dataframe['sma_fast'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma_slow'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=7)
        dataframe['cmo'] = ta.CMO(dataframe, timeperiod=14)
        dataframe['natr'] = ta.NATR(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['slowk'] < 15)
        ) & (
            qtpylib.crossed_above(dataframe['ppo'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['sma_fast'], dataframe['sma_slow'])
        ) & (
            qtpylib.crossed_above(dataframe['ema_fast'], dataframe['ema_slow'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['cmo'] > 50)
        ) & (
            qtpylib.crossed_below(dataframe['ema_fast'], dataframe['ema_slow'])
        ),
        'exit_long'] = 1
        return dataframe
