# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_38_19(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=20)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=10)
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=30)
        dataframe['natr'] = ta.NATR(dataframe, timeperiod=14)
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=7)
        dataframe['apo'] = ta.APO(dataframe, fastperiod=5, slowperiod=20)
        res = ta.AROON(dataframe, timeperiod=14)
        dataframe['aroondown'] = res.iloc[:, 0]
        dataframe['aroonup'] = res.iloc[:, 1]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['mom'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['roc'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['kama'])
        ) & (
            (dataframe['natr'] > 2.0)
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
            qtpylib.crossed_below(dataframe['apo'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['aroonup'], dataframe['aroondown'])
        ),
        'exit_long'] = 1
        return dataframe
