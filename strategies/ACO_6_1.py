# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_6_1(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['sar'] = ta.SAR(dataframe, acceleration=0.03, maximum=0.3)
        dataframe['adosc'] = ta.ADOSC(dataframe, fastperiod=3, slowperiod=10)
        stoch = ta.STOCH(dataframe, fastk_period=5, slowk_period=3, slowd_period=3)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']
        dataframe['cmo'] = ta.CMO(dataframe, timeperiod=14)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=10)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=20)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['cci'] < -150)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['sar'])
        ) & (
            qtpylib.crossed_above(dataframe['adosc'], 0)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['slowk'] > 85)
        ) & (
            (dataframe['cmo'] > 40)
        ) & (
            qtpylib.crossed_below(dataframe['roc'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['tema'])
        ),
        'exit_long'] = 1
        return dataframe
