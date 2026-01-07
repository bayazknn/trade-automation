# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_3_18(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=7)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['ppo'] = ta.PPO(dataframe, fastperiod=5, slowperiod=20)
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=7)
        dataframe['cmo'] = ta.CMO(dataframe, timeperiod=14)
        dataframe['trix'] = ta.TRIX(dataframe, timeperiod=9)
        dataframe['adosc'] = ta.ADOSC(dataframe, fastperiod=5, slowperiod=20)
        dataframe['natr'] = ta.NATR(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['cci'] < -80)
        ) & (
            (dataframe['adx'] > 25)
        ) & (
            qtpylib.crossed_above(dataframe['ppo'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['ema_fast'], dataframe['ema_slow'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['willr'] > -25)
        ) & (
            (dataframe['cmo'] > 40)
        ) & (
            qtpylib.crossed_below(dataframe['trix'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['adosc'], 0)
        ),
        'exit_long'] = 1
        return dataframe
