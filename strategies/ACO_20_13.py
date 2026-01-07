# Source: generated via dynamic_strategy_generator
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ACO_20_13(IStrategy):
    timeframe = '1h'
    
    # Standard ROI and Stoploss
    minimal_roi = {"0": 0.1, "60": 0.05, "120": 0.0}
    stoploss = -0.05
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=10)
        dataframe['ad'] = ta.AD(dataframe)
        dataframe['ad_sma'] = ta.SMA(dataframe, timeperiod=10, price='ad')
        dataframe['adosc'] = ta.ADOSC(dataframe, fastperiod=2, slowperiod=5)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=7)
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        res = ta.AROON(dataframe, timeperiod=25)
        dataframe['aroondown'] = res.iloc[:, 0]
        dataframe['aroonup'] = res.iloc[:, 1]
        dataframe['sar'] = ta.SAR(dataframe, acceleration=0.03, maximum=0.3)
        bbands = ta.BBANDS(dataframe, timeperiod=20, nbdevup=1.5, nbdevdn=1.5)
        dataframe['upperband'] = bbands['upperband']
        dataframe['middleband'] = bbands['middleband']
        dataframe['lowerband'] = bbands['lowerband']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            qtpylib.crossed_above(dataframe['mom'], 0)
        ) & (
            qtpylib.crossed_above(dataframe['ad'], dataframe['ad_sma'])
        ) & (
            qtpylib.crossed_above(dataframe['adosc'], 0)
        ),
        'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
        (
            (dataframe['mfi'] > 80)
        ) & (
            qtpylib.crossed_below(dataframe['aroonup'], dataframe['aroondown'])
        ) & (
            qtpylib.crossed_below(dataframe['close'], dataframe['sar'])
        ) & (
            (dataframe['close'] > dataframe['upperband'] * 1.0)
        ),
        'exit_long'] = 1
        return dataframe
