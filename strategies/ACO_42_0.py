# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_42_0(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['obv_sma'] = ta.SMA(dataframe, timeperiod=10, price='obv')
        dataframe['ad'] = ta.AD(dataframe)
        dataframe['ad_sma'] = ta.SMA(dataframe, timeperiod=10, price='ad')
        dataframe['adosc'] = ta.ADOSC(dataframe, fastperiod=2, slowperiod=5)
        dataframe['natr'] = ta.NATR(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=21)
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=7)
        dataframe['cmo'] = ta.CMO(dataframe, timeperiod=14)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=5)
        dataframe['aroonosc'] = ta.AROONOSC(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['obv'], dataframe['obv_sma'])
        ) & (
            qtpylib.crossed_above(dataframe['ad'], dataframe['ad_sma'])
        ) & (
            qtpylib.crossed_above(dataframe['adosc'], 0)
        ) & (
            (dataframe['natr'] > 2.0)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['rsi'] > 65)
        ) & (
            (dataframe['willr'] > -25)
        ) & (
            (dataframe['cmo'] > 50)
        ) & (
            qtpylib.crossed_below(dataframe['roc'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['aroonosc'], 0)
        ),
        'exit_long'] = 1
        return dataframe
