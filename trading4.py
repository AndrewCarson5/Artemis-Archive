import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import logging
from typing import List, Tuple, Dict
import talib
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from deap import base, creator, tools, algorithms  # For genetic algorithm
import random

class ForexTradingBot4:
    def __init__(self, symbols: List[str], timeframe: str):
        self.symbols = symbols  # Trade multiple currency pairs
        self.timeframe = timeframe
        self.balance = None
        self.models = {symbol: None for symbol in symbols}
        self.logger = self.setup_logger()
        self.in_position = {symbol: False for symbol in symbols}
        self.running = False

        if not mt5.initialize():
            print("initialize() failed")
            mt5.shutdown()

        # Strategy parameters
        self.short_window = 50
        self.long_window = 100
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.atr_period = 14
        self.risk_per_trade = 0.01  # 1% risk per trade
        self.slippage = 2  # Slippage (pips)
        self.transaction_cost = 0.0002  # Assuming 20 basis points

    def setup_logger(self):
        logger = logging.getLogger("TradingBot")
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler("trading_bot.log")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def fetch_ohlcv(self, symbol: str):
        try:
            timeframe = getattr(mt5.TIMEFRAME_D1, self.timeframe.upper())
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 2000)
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df: pd.DataFrame):
        try:
            df["short_ma"] = talib.SMA(df["close"], timeperiod=self.short_window)
            df["long_ma"] = talib.SMA(df["close"], timeperiod=self.long_window)
            df["rsi"] = talib.RSI(df["close"], timeperiod=self.rsi_period)
            df["upper_bb"], df["middle_bb"], df["lower_bb"] = talib.BBANDS(
                df["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            df["atr"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=self.atr_period)
            df["ema_50"] = talib.EMA(df["close"], timeperiod=50)
            df["ema_200"] = talib.EMA(df["close"], timeperiod=200)
            df["macd"], df["macd_signal"], _ = talib.MACD(df["close"])
            df["adx"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)
            df["volatility"] = df["close"].pct_change().rolling(window=20).std()
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")

    def update_balance(self):
        try:
            account_info = mt5.account_info()
            if account_info is not None:
                self.balance = account_info.balance
                self.logger.info(f"Updated balance: {self.balance}")
            else:
                self.logger.error("Failed to get account info")
        except Exception as e:
            self.logger.error(f"Error updating balance: {e}")

    def build_lstm_model(self):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(50, 12)))  # LSTM for 50 timesteps
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1, activation="linear"))  # Output layer for regression
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    def train_ml_model(self, df: pd.DataFrame, symbol: str):
        features = ["short_ma", "long_ma", "rsi", "upper_bb", "lower_bb", "atr", "ema_50", "ema_200", "macd", "macd_signal", "adx", "volatility"]
        X = df[features].dropna().values
        y = np.where(df["close"].shift(-1) > df["close"], 1, 0)[-len(X):]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = self.build_lstm_model()
        model.fit(X_train.reshape(-1, 50, 12), y_train, epochs=5, batch_size=32)
        self.models[symbol] = model

    def get_ml_signal(self, df: pd.DataFrame, symbol: str):
        model = self.models.get(symbol)
        if model is None:
            return 0
        X = df.iloc[-50:][["short_ma", "long_ma", "rsi", "upper_bb", "lower_bb", "atr", "ema_50", "ema_200", "macd", "macd_signal", "adx", "volatility"]].values
        prediction = model.predict(X.reshape(1, 50, 12))
        return 1 if prediction[0][0] > 0 else -1

    def walk_forward_optimization(self):
        pass  # Implement walk-forward optimization

    def optimize_parameters(self):
        pass  # Implement genetic algorithm for optimization

    def calculate_position_size(self, side: str, symbol: str) -> float:
        account_balance = self.balance
        risk_amount = account_balance * self.risk_per_trade
        last_atr = self.df["atr"].iloc[-1]
        point = mt5.symbol_info(symbol).point
        contract_size = mt5.symbol_info(symbol).trade_contract_size

        if side == "buy":
            entry_price = mt5.symbol_info_tick(symbol).ask
            stop_loss = entry_price - (2 * last_atr)
        else:
            entry_price = mt5.symbol_info_tick(symbol).bid
            stop_loss = entry_price + (2 * last_atr)

        position_size = risk_amount / (abs(entry_price - stop_loss) / point * contract_size)
        return position_size

    def process_signals(self):
        for symbol in self.symbols:
            df = self.fetch_ohlcv(symbol)
            self.calculate_indicators(df)
            signal = self.get_ml_signal(df, symbol)
            self.logger.info(f"Signal for {symbol}: {signal}")
            # Process buy/sell signals for each currency pair independently

    def run(self):
        self.running = True
        while self.running:
            try:
                self.perform_trade_cycle()
                time.sleep(60)
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")

    def perform_trade_cycle(self):
        self.update_balance()
        self.process_signals()

    def stop(self):
        self.running = False
        mt5.shutdown()
