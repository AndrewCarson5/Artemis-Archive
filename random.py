import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
from typing import List, Tuple, Dict
import talib
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import threading

class ForexTradingBot:
    def __init__(self, symbols: List[str], timeframe: str):
        self.symbols = symbols
        self.timeframe = timeframe
        self.models: Dict[str, RandomForestClassifier] = {symbol: None for symbol in symbols}
        self.in_position: Dict[str, bool] = {symbol: False for symbol in symbols}
        self.balance: float = 0.0
        self.df: Dict[str, pd.DataFrame] = {}
        self.logger = self.setup_logger()
        self.running = False

        if not mt5.initialize():
            self.logger.error("MetaTrader5 initialization failed")
            raise RuntimeError("MetaTrader5 initialization failed")

        # Strategy parameters
        self.short_window = 50
        self.long_window = 100
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.atr_period = 14
        self.risk_per_trade = 0.01

    def setup_logger(self):
        logger = logging.getLogger("TradingBot")
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler("trading_bot.log")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def fetch_ohlcv(self, symbol: str) -> pd.DataFrame:
        try:
            timeframe = getattr(mt5, f"TIMEFRAME_{self.timeframe.upper()}")
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 2000)
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            self.df[symbol] = df
            self.logger.info(f"Fetched {len(df)} candles for {symbol}")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df["short_ma"] = talib.SMA(df["close"], timeperiod=self.short_window)
            df["long_ma"] = talib.SMA(df["close"], timeperiod=self.long_window)
            df["rsi"] = talib.RSI(df["close"], timeperiod=self.rsi_period)
            df["upper_bb"], df["middle_bb"], df["lower_bb"] = talib.BBANDS(
                df["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df["atr"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=self.atr_period)
            df["ema_50"] = talib.EMA(df["close"], timeperiod=50)
            df["ema_200"] = talib.EMA(df["close"], timeperiod=200)
            df["macd"], df["macd_signal"], _ = talib.MACD(df["close"])
            df["adx"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)
            df["volatility"] = df["close"].pct_change().rolling(window=20).std()
            self.logger.info("Calculated technical indicators")
            return df
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return df

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

    def train_ml_model(self, symbol: str):
        features = ["short_ma", "long_ma", "rsi", "upper_bb", "lower_bb", "atr",
                    "ema_50", "ema_200", "macd", "macd_signal", "adx", "volatility"]
        
        if symbol not in self.df or len(self.df[symbol]) < 100:
            return
        
        df = self.df[symbol].copy()
        df['target'] = np.where(df["close"].shift(-1) > df["close"], 1, 0)
        df = df.dropna()

        X = df[features].values
        y = df['target'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if accuracy > 0.55:  # Lowered threshold for model acceptance
            self.models[symbol] = model
            self.logger.info(f"Trained ML model for {symbol} with accuracy: {accuracy:.2f}")
        else:
            self.logger.warning(f"ML model for {symbol} did not meet accuracy threshold: {accuracy:.2f}")

    def get_ml_signal(self, symbol: str) -> int:
        model = self.models.get(symbol)

        if model is None or symbol not in self.df or len(self.df[symbol]) < 1:
            return 0
        
        last_row = self.df[symbol].iloc[-1]
        
        features = ["short_ma", "long_ma", "rsi", "upper_bb", "lower_bb", 
                    "atr", "ema_50", "ema_200", "macd", "macd_signal", 
                    "adx", "volatility"]
        
        X_last = last_row[features].values.reshape(1, -1)
        
        prediction = model.predict(X_last)[0]
        
        return 1 if prediction == 1 else -1

    def calculate_position_size(self, symbol: str, side: str) -> float:
        risk_amount = self.balance * self.risk_per_trade
        
        if symbol not in self.df or len(self.df[symbol]) < 1:
            return 0.0

        last_atr_value = self.df[symbol]['atr'].iloc[-1]
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.error(f"Symbol info not found for {symbol}")
            return 0.0

        point_value = symbol_info.point
        contract_size = symbol_info.trade_contract_size
        
        if side == "buy":
            entry_price = mt5.symbol_info_tick(symbol).ask
            stop_loss_price = entry_price - (2 * last_atr_value)
        elif side == "sell":
            entry_price = mt5.symbol_info_tick(symbol).bid
            stop_loss_price = entry_price + (2 * last_atr_value)
        else:
            self.logger.error(f"Invalid trade side: {side}")
            return 0.0
        
        position_size = risk_amount / (abs(entry_price - stop_loss_price) / point_value * contract_size)
        
        return position_size

    def execute_trade(self, symbol: str, side: str, amount: float):
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"{symbol} not found")
                return

            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    self.logger.error(f"symbol_select({symbol}) failed")
                    return

            point = symbol_info.point
            price = mt5.symbol_info_tick(symbol).ask if side == "buy" else mt5.symbol_info_tick(symbol).bid
            deviation = 20

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": amount,
                "type": mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": price - 100 * point if side == "buy" else price + 100 * point,
                "tp": price + 100 * point if side == "buy" else price - 100 * point,
                "deviation": deviation,
                "magic": 123456,
                "comment": 'Trading bot execution',
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC
            }

            result = mt5.order_send(request)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Failed to execute trade: {result.retcode}")
            else:
                self.in_position[symbol] = True
                self.logger.info(f"Trade executed: {side} order for {symbol} at volume {amount}")

        except Exception as e:
            self.logger.error(f"Error executing {side} order for {symbol}: {e}")

    def process_signals(self):
        for symbol in self.symbols:
            signal_ml = self.get_ml_signal(symbol)

            if signal_ml == 1 and not self.in_position[symbol]:
                position_size = self.calculate_position_size(symbol, "buy")
                self.execute_trade(symbol, "buy", position_size)
               
            elif signal_ml == -1 and self.in_position[symbol]:
                position_size = self.calculate_position_size(symbol, "sell")
                self.execute_trade(symbol, "sell", position_size)

    def run(self):
        self.running = True
        while self.running:
            try:
                for symbol in self.symbols:
                    df = self.fetch_ohlcv(symbol)
                    if not df.empty:
                        df = self.calculate_indicators(df)
                        self.train_ml_model(symbol)

                self.update_balance()
                self.process_signals()
                time.sleep(60)  # Wait a minute before the next cycle.
               
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")

    def stop(self):
        self.running = False
        mt5.shutdown()


class TradingBotGUI:
    def __init__(self, bot: ForexTradingBot):
        self.bot = bot 
        self.root = tk.Tk()
        self.root.title("Forex Trading Bot GUI")
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        start_button = ttk.Button(control_frame, text="Start", command=self.start_bot)
        start_button.pack(side=tk.LEFT)

        stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_bot)
        stop_button.pack(side=tk.LEFT)

        update_button = ttk.Button(control_frame, text="Update Chart", command=self.update_chart)
        update_button.pack(side=tk.LEFT)

        self.symbol_var = tk.StringVar(self.root)
        self.symbol_var.set(self.bot.symbols[0])  # default value
        symbol_menu = ttk.OptionMenu(control_frame, self.symbol_var, self.bot.symbols[0], *self.bot.symbols)
        symbol_menu.pack(side=tk.LEFT)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_chart()  # Initial chart update

    def start_bot(self):
        if not self.bot.running:
            bot_thread = threading.Thread(target=self.bot.run)
            bot_thread.start()

    def stop_bot(self):
        if self.bot.running:
            self.bot.stop()

    def update_chart(self):
        symbol = self.symbol_var.get()
        if symbol in self.bot.df and not self.bot.df[symbol].empty:
            df = self.bot.df[symbol]
            self.ax.clear()
            self.ax.plot(df.index, df['close'], label='Close Price')
            self.ax.plot(df.index, df['short_ma'], label='Short MA')
            self.ax.plot(df.index, df['long_ma'], label='Long MA')
            self.ax.set_title(f'{symbol} Price and Moving Averages')
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Price')
            self.ax.legend()
            self.canvas.draw()

    def on_closing(self):
        self.stop_bot()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    trading_bot = ForexTradingBot(symbols, "M15")
    
    gui = TradingBotGUI(trading_bot)
    gui.run()