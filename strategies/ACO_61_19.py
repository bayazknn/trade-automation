# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_61_19(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['bop'] = ta.BOP(dataframe)
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=30)
        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['obv_sma'] = ta.SMA(dataframe, timeperiod=10, price='obv')
        dataframe['natr'] = ta.NATR(dataframe, timeperiod=14)
        stoch = ta.STOCH(dataframe, fastk_period=21, slowk_period=5, slowd_period=5)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=7)
        dataframe['ppo'] = ta.PPO(dataframe, fastperiod=5, slowperiod=20)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['adx'] > 25)
        ) & (
            qtpylib.crossed_above(dataframe['bop'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['kama'])
        ) & (
            qtpylib.crossed_above(dataframe['obv'], dataframe['obv_sma'])
        ) & (
            (dataframe['natr'] > 2.0)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['slowk'] > 75)
        ) & (
            (dataframe['willr'] > -25)
        ) & (
            qtpylib.crossed_below(dataframe['ppo'], 0)
        ),
        'exit_long'] = 1
        return dataframe
