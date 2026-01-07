# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_20_17(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=20)
        dataframe['sar'] = ta.SAR(dataframe, acceleration=0.03, maximum=0.3)
        dataframe['ad'] = ta.AD(dataframe)
        dataframe['ad_sma'] = ta.SMA(dataframe, timeperiod=10, price='ad')
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe, fastperiod=19, slowperiod=39, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['ppo'] = ta.PPO(dataframe, fastperiod=5, slowperiod=20)
        dataframe['aroonosc'] = ta.AROONOSC(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['roc'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['sar'])
        ) & (
            qtpylib.crossed_above(dataframe['ad'], dataframe['ad_sma'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])
        ) & (
            qtpylib.crossed_below(dataframe['ppo'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['aroonosc'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['sar'])
        ),
        'exit_long'] = 1
        return dataframe
