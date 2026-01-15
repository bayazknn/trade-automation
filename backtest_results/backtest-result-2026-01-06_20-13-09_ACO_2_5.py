from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.vendor.qtpylib.indicators import crossed_above, crossed_below


class ACO_2_5(IStrategy):
    timeframe = '1h'
    stoploss = -0.05
    minimal_roi = {"0": 0.03, "60": 0.02, "180": 0.01}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI Conservative (21)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=21)

        # EMA Fast Crossover (9/21)
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=21)

        # Stochastic Fast (5/3/3)
        stoch = ta.STOCH(dataframe, fastk_period=5, slowk_period=3, slowd_period=3)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']

        # OBV Standard (SMA 20)
        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['obv_sma'] = ta.SMA(dataframe['obv'], timeperiod=20)

        # AD Fast (SMA 10)
        dataframe['ad'] = ta.AD(dataframe)
        dataframe['ad_sma'] = ta.SMA(dataframe['ad'], timeperiod=10)

        # ATR Wide (14, mult 3.0)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Entry conditions (all must be TRUE - AND logic)
        # 1. RSI < 35
        condition1 = dataframe['rsi'] < 35

        # 2. EMA fast crosses above EMA slow
        condition2 = crossed_above(dataframe['ema_fast'], dataframe['ema_slow'])

        dataframe.loc[
            (condition1) & (condition2) & (dataframe['volume'] > 0),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit conditions (all must be TRUE - AND logic)
        # 1. Stochastic slowk > 85
        condition1 = dataframe['slowk'] > 85

        # 2. OBV crosses below OBV SMA
        condition2 = crossed_below(dataframe['obv'], dataframe['obv_sma'])

        # 3. AD crosses below AD SMA
        condition3 = crossed_below(dataframe['ad'], dataframe['ad_sma'])

        dataframe.loc[
            (condition1) & (condition2) & (condition3) & (dataframe['volume'] > 0),
            'exit_long'] = 1

        return dataframe
