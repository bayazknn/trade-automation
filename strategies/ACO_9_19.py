# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_9_19(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        stoch = ta.STOCH(dataframe, fastk_period=5, slowk_period=3, slowd_period=3)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']
        stochrsi = ta.STOCHRSI(dataframe, timeperiod=14, fastk_period=5, fastd_period=3)
        dataframe['fastk'] = stochrsi['fastk']
        dataframe['fastd'] = stochrsi['fastd']
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=30)
        dataframe['t3'] = ta.T3(dataframe, timeperiod=10, vfactor=0.9)
        dataframe['ad'] = ta.AD(dataframe)
        dataframe['ad_sma'] = ta.SMA(dataframe, timeperiod=10, price='ad')
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=20)
        res = ta.AROON(dataframe, timeperiod=10)
        dataframe['aroondown'] = res.iloc[:, 0]
        dataframe['aroonup'] = res.iloc[:, 1]
        dataframe['sar'] = ta.SAR(dataframe, acceleration=0.03, maximum=0.3)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['slowk'] < 15)
        ) & (
            (dataframe['fastk'] < 20)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['kama'])
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['t3'])
        ) & (
            qtpylib.crossed_above(dataframe['ad'], dataframe['ad_sma'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_below(dataframe['roc'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['aroonup'], dataframe['aroondown'])
        ) & (
            qtpylib.crossed_below(dataframe['aroonup'], dataframe['aroondown'])
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['sar'])
        ),
        'exit_long'] = 1
        return dataframe
