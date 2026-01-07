# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_1_1(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=14)
        dataframe['ppo'] = ta.PPO(dataframe, fastperiod=12, slowperiod=26)
        dataframe['sar'] = ta.SAR(dataframe, acceleration=0.01, maximum=0.1)
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=5)
        res = ta.AROON(dataframe, timeperiod=10)
        dataframe['aroondown'] = res.iloc[:, 0]
        dataframe['aroonup'] = res.iloc[:, 1]
        dataframe['ad'] = ta.AD(dataframe)
        dataframe['ad_sma'] = ta.SMA(dataframe, timeperiod=10, price='ad')
        dataframe['adosc'] = ta.ADOSC(dataframe, fastperiod=3, slowperiod=10)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['slowk'] < 20)
        ) & (
            (dataframe['cci'] < -100)
        ) & (
            (dataframe['cci'] < -80)
        ) & (
            qtpylib.crossed_above(dataframe['ppo'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['close'], dataframe['sar'])
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_below(dataframe['mom'], 0)
        ) & (
            qtpylib.crossed_below(dataframe['aroonup'], dataframe['aroondown'])
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['sar'])
        ) & (
            qtpylib.crossed_below(dataframe['ad'], dataframe['ad_sma'])
        ) & (
            qtpylib.crossed_below(dataframe['adosc'], 0)
        ),
        'exit_long'] = 1
        return dataframe
