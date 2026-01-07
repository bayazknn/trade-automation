# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_16_6(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=10)
        dataframe['kama'] = ta.KAMA(dataframe, timeperiod=30)
        dataframe['sar'] = ta.SAR(dataframe, acceleration=0.02, maximum=0.2)
        stochrsi = ta.STOCHRSI(dataframe, timeperiod=14, fastk_period=5, fastd_period=3)
        dataframe['fastk'] = stochrsi['fastk']
        dataframe['fastd'] = stochrsi['fastd']
        dataframe['trix'] = ta.TRIX(dataframe, timeperiod=21)
        dataframe['wma'] = ta.WMA(dataframe, timeperiod=20)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['adx'] > 30)
        ) & (
            qtpylib.crossed_above(dataframe['mom'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['kama'])
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['sar'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['fastk'] > 80)
        ) & (
            qtpylib.crossed_below(dataframe['trix'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['wma'])
        ),
        'exit_long'] = 1
        return dataframe
