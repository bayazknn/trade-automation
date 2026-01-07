# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_77_4(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ppo'] = ta.PPO(dataframe, fastperiod=12, slowperiod=26)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=7)
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=7)
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=20)
        dataframe['trix'] = ta.TRIX(dataframe, timeperiod=15)
        dataframe['adosc'] = ta.ADOSC(dataframe, fastperiod=2, slowperiod=5)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['ppo'], 0)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['willr'] > -25)
        ) & (
            qtpylib.crossed_below(dataframe['mom'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['trix'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['adosc'], 0)
        ),
        'exit_long'] = 1
        return dataframe
