import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import logging
from typing import List, Tuple
import talib
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class OptimizedForexTradingBot:
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
            print("initialize() failed")
            mt5.shutdown()

        # Dynamic strategy parameters
        self.short_window = 50
        self.long_window = 100
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.atr_period = 14
        self.risk_per_trade = 0.01  # Reduced risk per trade to 1%

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
            rates = mt5.copy_rates_from_pos(
                self.symbol, timeframe, 0, 2000
            )  # Increased data points
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            self.df = df
            self.logger.info(f"Fetched {len(df)} candles of data")
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data: {e}")

    def calculate_indicators(self):
        try:
            # Existing indicators
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

            # New indicators
            self.df["ema_50"] = talib.EMA(self.df["close"], timeperiod=50)
            self.df["ema_200"] = talib.EMA(self.df["close"], timeperiod=200)
            self.df["macd"], self.df["macd_signal"], _ = talib.MACD(self.df["close"])
            self.df["adx"] = talib.ADX(
                self.df["high"], self.df["low"], self.df["close"], timeperiod=14
            )

            # Trend strength
            self.df["trend_strength"] = (
                abs(self.df["ema_50"] - self.df["ema_200"]) / self.df["ema_200"]
            )

            # Volatility
            self.df["volatility"] = (
                self.df["close"].pct_change().rolling(window=20).std()
            )

            self.logger.info("Calculated technical indicators")
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")

    def update_balance(self):
        try:
            # Highlight: Changed to use MetaTrader5 for fetching account info
            account_info = mt5.account_info()
            if account_info is not None:
                self.balance = account_info.balance
                self.logger.info(f"Updated balance: {self.balance}")
            else:
                self.logger.error("Failed to get account info")
        except Exception as e:
            self.logger.error(f"Error updating balance: {e}")

    def optimize_parameters(self):
        # Optimize RSI parameters
        best_sharpe = -np.inf
        best_rsi_period = self.rsi_period
        best_rsi_overbought = self.rsi_overbought
        best_rsi_oversold = self.rsi_oversold

        for rsi_period in range(10, 30):
            for rsi_overbought in range(65, 85, 5):
                for rsi_oversold in range(15, 35, 5):
                    self.df["rsi"] = talib.RSI(self.df["close"], timeperiod=rsi_period)
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

                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_rsi_period = rsi_period
                        best_rsi_overbought = rsi_overbought
                        best_rsi_oversold = rsi_oversold

        self.rsi_period = best_rsi_period
        self.rsi_overbought = best_rsi_overbought
        self.rsi_oversold = best_rsi_oversold
        self.logger.info(
            f"Optimized RSI parameters: period={self.rsi_period}, overbought={self.rsi_overbought}, oversold={self.rsi_oversold}"
        )

    def train_ml_model(self):
        # Prepare features and target
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

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.logger.info(f"ML model accuracy: {accuracy:.2f}")

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

    def get_signal(self) -> Tuple[str, float]:
        last_row = self.df.iloc[-1]
        signal = "HOLD"
        confidence = 0.0

        # Moving Average Crossover
        if last_row["short_ma"] > last_row["long_ma"]:
            signal = "BUY"
            confidence += 0.2
        elif last_row["short_ma"] < last_row["long_ma"]:
            signal = "SELL"
            confidence += 0.2

        # RSI
        if last_row["rsi"] < self.rsi_oversold:
            signal = "BUY"
            confidence += 0.2
        elif last_row["rsi"] > self.rsi_overbought:
            signal = "SELL"
            confidence += 0.2

        # Bollinger Bands
        if last_row["close"] < last_row["lower_bb"]:
            signal = "BUY"
            confidence += 0.1
        elif last_row["close"] > last_row["upper_bb"]:
            signal = "SELL"
            confidence += 0.1

        # MACD
        if last_row["macd"] > last_row["macd_signal"]:
            if signal == "BUY":
                confidence += 0.1
            else:
                signal = "BUY"
                confidence = 0.1
        elif last_row["macd"] < last_row["macd_signal"]:
            if signal == "SELL":
                confidence += 0.1
            else:
                signal = "SELL"
                confidence = 0.1

        # ADX (trend strength)
        if last_row["adx"] > 25:
            confidence += 0.1

        # ML model signal
        ml_signal = self.get_ml_signal()
        if (ml_signal == 1 and signal == "BUY") or (
            ml_signal == -1 and signal == "SELL"
        ):
            confidence += 0.2
        else:
            confidence -= 0.1

        return signal, min(confidence, 1.0)

    def calculate_position_size(self, side: str) -> float:
        # Highlight: Adjusted position sizing for forex
        account_balance = self.balance
        risk_amount = account_balance * self.risk_per_trade
        last_atr = self.df["atr"].iloc[-1]

        point = mt5.symbol_info(self.symbol).point
        contract_size = mt5.symbol_info(self.symbol).trade_contract_size

        if side == "buy":
            entry_price = mt5.symbol_info_tick(self.symbol).ask
            stop_loss = entry_price - (2 * last_atr)
        else:
            entry_price = mt5.symbol_info_tick(self.symbol).bid
            stop_loss = entry_price + (2 * last_atr)

        position_size = risk_amount / (
            abs(entry_price - stop_loss) / point * contract_size
        )

        return position_size

    def execute_trade(self, side: str, amount: float):
        try:
            # Highlight: Changed to use MetaTrader5 for order execution
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                self.logger.error(f"{self.symbol} not found")
                return

            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    self.logger.error(f"symbol_select({self.symbol}) failed")
                    return

            lot = amount
            point = mt5.symbol_info(self.symbol).point
            price = (
                mt5.symbol_info_tick(self.symbol).ask
                if side == "buy"
                else mt5.symbol_info_tick(self.symbol).bid
            )
            deviation = 20
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": price - 100 * point if side == "buy" else price + 100 * point,
                "tp": price + 100 * point if side == "buy" else price - 100 * point,
                "deviation": deviation,
                "magic": 234000,
                "comment": "python script open",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            self.logger.info(f"Executed {side} order: {result}")
        except Exception as e:
            self.logger.error(f"Error executing {side} order: {e}")

    def backtest(self, start_date: str, end_date: str) -> pd.DataFrame:
        backtest_df = self.df.loc[start_date:end_date].copy()
        backtest_df["signal"], backtest_df["confidence"] = zip(
            *backtest_df.apply(lambda row: self.get_signal(), axis=1)
        )
        backtest_df["position"] = 0
        backtest_df.loc[backtest_df["signal"] == "BUY", "position"] = 1
        backtest_df.loc[backtest_df["signal"] == "SELL", "position"] = -1
        backtest_df["returns"] = backtest_df["close"].pct_change()
        backtest_df["strategy_returns"] = (
            backtest_df["position"].shift() * backtest_df["returns"]
        )
        backtest_df["cumulative_returns"] = (
            1 + backtest_df["strategy_returns"]
        ).cumprod()

        # Calculate success rate
        backtest_df["correct_prediction"] = (
            backtest_df["position"].shift() * backtest_df["returns"]
        ) > 0
        success_rate = backtest_df["correct_prediction"].mean()
        self.logger.info(f"Backtest success rate: {success_rate:.2%}")

        return backtest_df

    def adjust_parameters(self):
        recent_volatility = (
            self.df["close"].pct_change().rolling(window=30).std().iloc[-1]
        )
        if recent_volatility > 0.02:  # High volatility
            self.short_window = max(10, self.short_window - 5)
            self.long_window = max(20, self.long_window - 10)
        else:  # Low volatility
            self.short_window = min(100, self.short_window + 5)
            self.long_window = min(200, self.long_window + 10)
        self.logger.info(
            f"Adjusted parameters: short_window={self.short_window}, long_window={self.long_window}"
        )

    def perform_trade_cycle(self):
        self.fetch_ohlcv()
        self.calculate_indicators()
        self.update_balance()
        self.adjust_parameters()
        self.optimize_parameters()  # Periodically optimize parameters
        self.train_ml_model()  # Periodically retrain ML model
        self.process_signals()

    def process_signals(self):
        signal, confidence = self.get_signal()
        self.logger.info(f"Signal: {signal}, Confidence: {confidence}")
        if signal == "BUY" and not self.in_position and confidence > 0.7:
            amount = self.calculate_position_size("buy")
            self.execute_trade("buy", amount)
            self.in_position = True
        elif signal == "SELL" and self.in_position and confidence > 0.7:
            amount = self.calculate_position_size("sell")
            self.execute_trade("sell", amount)
            self.in_position = False

    def run(self):
        self.running = True
        while self.running:
            try:
                self.perform_trade_cycle()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait a minute before retrying

    def stop(self):
        self.running = False
        mt5.shutdown()


class TradingBotGUI:
    def __init__(self, bot: AdvancedTradingBot):
        self.bot = bot
        self.root = tk.Tk()
        self.root.title("Forex Trading Bot")
        self.setup_ui()

    def setup_ui(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.start_button = ttk.Button(
            self.control_frame, text="Start", command=self.start_bot
        )
        self.start_button.pack(side=tk.LEFT)

        self.stop_button = ttk.Button(
            self.control_frame, text="Stop", command=self.stop_bot
        )
        self.stop_button.pack(side=tk.LEFT)

        self.update_button = ttk.Button(
            self.control_frame, text="Update Chart", command=self.update_chart
        )
        self.update_button.pack(side=tk.LEFT)

    def start_bot(self):
        self.bot_thread = threading.Thread(target=self.bot.run)
        self.bot_thread.start()

    def stop_bot(self):
        self.bot.stop()
        if self.bot_thread.is_alive():
            self.bot_thread.join()
        self.bot.logger.info("Bot stopped successfully")

    def update_chart(self):
        self.ax.clear()
        self.bot.df["close"].plot(ax=self.ax, label="Close Price")
        self.bot.df["short_ma"].plot(ax=self.ax, label="50-period MA")
        self.bot.df["long_ma"].plot(ax=self.ax, label="100-period MA")
        self.bot.df["upper_bb"].plot(ax=self.ax, label="Upper BB")
        self.bot.df["lower_bb"].plot(ax=self.ax, label="Lower BB")
        self.ax.legend()
        self.ax.set_title("Price and Technical Indicators")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price")
        self.canvas.draw()

    def run(self):
        self.root.mainloop()


# Usage example:
# bot = ForexTradingBot('EURUSD', 'M15')  # Highlight: Changed to use forex pair and MT5 timeframe
# gui = TradingBotGUI(bot)
# gui.run()
