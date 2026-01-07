# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_2_3(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']
        res = ta.AROON(dataframe, timeperiod=25)
        dataframe['aroondown'] = res.iloc[:, 0]
        dataframe['aroonup'] = res.iloc[:, 1]
        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['obv_sma'] = ta.SMA(dataframe, timeperiod=20, price='obv')
        dataframe['natr'] = ta.NATR(dataframe, timeperiod=14)
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=10)
        dataframe['trix'] = ta.TRIX(dataframe, timeperiod=21)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['slowk'] < 20)
        ) & (
            qtpylib.crossed_above(dataframe['aroonup'], dataframe['aroondown'])
        ) & (
            qtpylib.crossed_above(dataframe['obv'], dataframe['obv_sma'])
        ) & (
            (dataframe['natr'] > 2.0)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_below(dataframe['mom'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['trix'], 0)
        ),
        'exit_long'] = 1
        return dataframe
