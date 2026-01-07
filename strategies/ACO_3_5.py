# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_3_5(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        res = ta.AROON(dataframe, timeperiod=10)
        dataframe['aroondown'] = res.iloc[:, 0]
        dataframe['aroonup'] = res.iloc[:, 1]
        dataframe['aroonosc'] = ta.AROONOSC(dataframe, timeperiod=14)
        dataframe['t3'] = ta.T3(dataframe, timeperiod=5, vfactor=0.7)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=21)
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=14)
        dataframe['ppo'] = ta.PPO(dataframe, fastperiod=5, slowperiod=20)
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=20)
        dataframe['ad'] = ta.AD(dataframe)
        dataframe['ad_sma'] = ta.SMA(dataframe, timeperiod=20, price='ad')
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['aroonup'], dataframe['aroondown'])
        ) & (
            qtpylib.crossed_above(dataframe['aroonosc'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['t3'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['rsi'] > 65)
        ) & (
            (dataframe['willr'] > -20)
        ) & (
            qtpylib.crossed_below(dataframe['ppo'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['kama'])
        ) & (
            qtpylib.crossed_below(dataframe['ad'], dataframe['ad_sma'])
        ),
        'exit_long'] = 1
        return dataframe
