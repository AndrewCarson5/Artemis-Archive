import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
from typing import List, Tuple
import talib
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class ForexTradingBot3:
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.in_position = False
        self.balance = None
        self.df = pd.DataFrame()
        self.logger = self.setup_logger()
        self.running = False
        self.model = None

        if not mt5.initialize():
            self.logger.error("initialize() failed")
            mt5.shutdown()
            raise RuntimeError("MetaTrader5 initialization failed")

        self.short_window = 50
        self.long_window = 100
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.atr_period = 14
        self.risk_per_trade = 0.01
        self.accuracy_threshold = 0.75
        self.cache = {}

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

    def fetch_ohlcv(self):
        try:
            timeframe = getattr(mt5.TIMEFRAME_D1, self.timeframe.upper())
            rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, 2000)
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            self.df = df
            self.logger.info(f"Fetched {len(df)} candles of data")
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data: {e}")

    def calculate_indicators(self):
        try:
            self.df["short_ma"] = talib.SMA(
                self.df["close"], timeperiod=self.short_window
            )
            self.df["long_ma"] = talib.SMA(
                self.df["close"], timeperiod=self.long_window
            )
            self.df["rsi"] = talib.RSI(self.df["close"], timeperiod=self.rsi_period)
            self.df["upper_bb"], self.df["middle_bb"], self.df["lower_bb"] = (
                talib.BBANDS(
                    self.df["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
                )
            )
            self.df["atr"] = talib.ATR(
                self.df["high"],
                self.df["low"],
                self.df["close"],
                timeperiod=self.atr_period,
            )
            self.df["ema_50"] = talib.EMA(self.df["close"], timeperiod=50)
            self.df["ema_200"] = talib.EMA(self.df["close"], timeperiod=200)
            self.df["macd"], self.df["macd_signal"], _ = talib.MACD(self.df["close"])
            self.df["adx"] = talib.ADX(
                self.df["high"], self.df["low"], self.df["close"], timeperiod=14
            )
            self.df["trend_strength"] = (
                abs(self.df["ema_50"] - self.df["ema_200"]) / self.df["ema_200"]
            )
            self.df["volatility"] = (
                self.df["close"].pct_change().rolling(window=20).std()
            )
            self.logger.info("Calculated technical indicators")
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

    def validate_parameters(self):
        if self.short_window < 10:
            self.short_window = 10
            self.logger.warning("short_window adjusted to minimum of 10")
        if self.long_window < 20:
            self.long_window = 20
            self.logger.warning("long_window adjusted to minimum of 20")

    def optimize_parameters(self):
        best_sharpe = -np.inf
        best_params = {
            "rsi_period": self.rsi_period,
            "rsi_overbought": self.rsi_overbought,
            "rsi_oversold": self.rsi_oversold,
        }

        for rsi_period in range(10, 30):
            for rsi_overbought in range(65, 85, 5):
                for rsi_oversold in range(15, 35, 5):
                    cache_key = (rsi_period, rsi_overbought, rsi_oversold)
                    if cache_key in self.cache:
                        sharpe = self.cache[cache_key]
                    else:
                        self.df["rsi"] = talib.RSI(
                            self.df["close"], timeperiod=rsi_period
                        )
                        self.df["signal"] = np.where(
                            self.df["rsi"] > rsi_overbought,
                            -1,
                            np.where(self.df["rsi"] < rsi_oversold, 1, 0),
                        )
                        self.df["returns"] = self.df["close"].pct_change()
                        self.df["strategy_returns"] = (
                            self.df["signal"].shift() * self.df["returns"]
                        )
                        sharpe = (
                            np.sqrt(252)
                            * self.df["strategy_returns"].mean()
                            / self.df["strategy_returns"].std()
                        )
                        self.cache[cache_key] = sharpe

                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = {
                            "rsi_period": rsi_period,
                            "rsi_overbought": rsi_overbought,
                            "rsi_oversold": rsi_oversold,
                        }

        self.rsi_period = best_params["rsi_period"]
        self.rsi_overbought = best_params["rsi_overbought"]
        self.rsi_oversold = best_params["rsi_oversold"]
        self.logger.info(
            f"Optimized RSI parameters: period={self.rsi_period}, overbought={self.rsi_overbought}, oversold={self.rsi_oversold}"
        )

    def train_ml_model(self):
        features = [
            "short_ma",
            "long_ma",
            "rsi",
            "upper_bb",
            "lower_bb",
            "atr",
            "ema_50",
            "ema_200",
            "macd",
            "macd_signal",
            "adx",
            "trend_strength",
            "volatility",
        ]
        X = self.df[features].dropna()
        y = np.where(self.df["close"].shift(-1) > self.df["close"], 1, 0)[
            len(self.df) - len(X) :
        ]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.logger.info(f"ML model accuracy: {accuracy:.2f}")

        if accuracy < self.accuracy_threshold:
            self.logger.info("Retraining ML model due to accuracy drop.")
            self.train_ml_model()

    def get_ml_signal(self) -> int:
        if self.model is None:
            return 0
        last_row = self.df.iloc[-1]
        features = [
            "short_ma",
            "long_ma",
            "rsi",
            "upper_bb",
            "lower_bb",
            "atr",
            "ema_50",
            "ema_200",
            "macd",
            "macd_signal",
            "adx",
            "trend_strength",
            "volatility",
        ]
        X = last_row[features].values.reshape(1, -1)
        prediction = self.model.predict(X)[0]
        return 1 if prediction == 1 else -1

    def calculate_dynamic_confidence(self, last_row) -> float:
        volatility = last_row["volatility"]
        if volatility > 0.02:
            return 0.5
        return 1.0

    def get_signal(self) -> Tuple[str, float]:
        last_row = self.df.iloc[-1]
        signal = "HOLD"
        confidence = 0.0

        if last_row["short_ma"] > last_row["long_ma"]:
            signal = "BUY"
            confidence += 0.2
        elif last_row["short_ma"] < last_row["long_ma"]:
            signal = "SELL"
            confidence += 0.2

        if last_row["rsi"] < self.rsi_oversold:
            signal = "BUY"
            confidence += 0.2
        elif last_row["rsi"] > self.rsi_overbought:
            signal = "SELL"
            confidence += 0.2

        if last_row["close"] < last_row["lower_bb"]:
            signal = "BUY"
            confidence += 0.1
        elif last_row["close"] > last_row["upper_bb"]:
            signal = "SELL"
            confidence += 0.1

        ml_signal = self.get_ml_signal()
        if ml_signal == 1:
            signal = "BUY"
            confidence += 0.3
        elif ml_signal == -1:
            signal = "SELL"
            confidence += 0.3

        confidence = min(confidence, 1.0)
        dynamic_confidence = self.calculate_dynamic_confidence(last_row)
        confidence = max(confidence, dynamic_confidence)

        self.logger.info(
            f"Generated signal: {signal} with confidence: {confidence:.2f}"
        )
        return signal, confidence

    def execute_trade(self, signal: str, confidence: float):
        try:
            if not self.in_position:
                if signal == "BUY":
                    volume = self.calculate_volume()
                    order = mt5.ORDER_BUY
                elif signal == "SELL":
                    volume = self.calculate_volume()
                    order = mt5.ORDER_SELL
                else:
                    self.logger.info("No trading action required.")
                    return

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": volume,
                    "type": order,
                    "price": mt5.symbol_info_tick(self.symbol).last,
                    "sl": 0,
                    "tp": 0,
                    "deviation": 10,
                    "magic": 123456,
                    "comment": "Trading bot execution",
                }

                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    self.logger.error(f"Failed to execute trade: {result.retcode}")
                else:
                    self.in_position = True
                    self.logger.info(f"Trade executed: {signal} at volume {volume}")
            else:
                self.logger.info("Already in position, no new trades executed.")
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")

    def calculate_volume(self) -> float:
        if self.balance is None:
            self.update_balance()
        return self.balance * self.risk_per_trade / 100

    def run(self):
        self.running = True
        self.logger.info("Starting trading bot")
        while self.running:
            self.fetch_ohlcv()
            self.calculate_indicators()
            self.optimize_parameters()
            self.train_ml_model()
            signal, confidence = self.get_signal()
            self.execute_trade(signal, confidence)
            time.sleep(60)  # Run every minute

    def stop(self):
        self.running = False
        mt5.shutdown()
        self.logger.info("Trading bot stopped")

    def plot_gui(self):
        root = tk.Tk()
        root.title("Trading Bot")

        fig, ax = plt.subplots(figsize=(10, 6))
        self.df[["close", "short_ma", "long_ma"]].plot(ax=ax)
        ax.set_title(f"{self.symbol} - {self.timeframe} Chart")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        root.mainloop()

    def start_gui_thread(self):
        gui_thread = threading.Thread(target=self.plot_gui)
        gui_thread.start()


class MultiCurrencyForexTradingBot:
    def __init__(self, symbols: List[str], timeframe: str):
        self.symbols = symbols
        self.timeframe = timeframe
        self.bots = [ForexTradingBot3(symbol, timeframe) for symbol in symbols]

    def run(self):
        self.running = True
        threads = []
        for bot in self.bots:
            thread = threading.Thread(target=bot.run)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def stop(self):
        self.running = False
        for bot in self.bots:
            bot.stop()


if __name__ == "__main__":
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]  # Add more currency pairs as needed
    multi_bot = MultiCurrencyForexTradingBot(symbols, "M15")
    try:
        multi_bot.run()
    except KeyboardInterrupt:
        multi_bot.stop()

""" if __name__ == "__main__":
    bot = ForexTradingBot("EURUSD", "M15")
    try:
        bot.start_gui_thread()
        bot.run()
    except KeyboardInterrupt:
        bot.stop()
 """
