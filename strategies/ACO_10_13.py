# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_10_13(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)
        dataframe['trix'] = ta.TRIX(dataframe, timeperiod=21)
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=30)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['ppo'] = ta.PPO(dataframe, fastperiod=12, slowperiod=26)
        dataframe['dema'] = ta.DEMA(dataframe, timeperiod=10)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['mfi'] < 25)
        ) & (
            qtpylib.crossed_above(dataframe['plus_di'], dataframe['minus_di'])
        ) & (
            qtpylib.crossed_above(dataframe['trix'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['kama'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['mfi'] > 75)
        ) & (
            qtpylib.crossed_below(dataframe['ppo'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['ppo'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['dema'])
        ),
        'exit_long'] = 1
        return dataframe
