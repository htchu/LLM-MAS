# -*- coding: utf-8 -*- llmmas.py
# -----------------------------------------------------------------------------
# Author: Edward Cheng
# Date: 2025-09-13 Complete Refactored Version
# Version: 3.0 Production-Ready with Train-Validation-Test Split
# License: MIT
# -----------------------------------------------------------------------------
"""
AI Quantitative Trading Strategy Development Framework (LLM-MAS-DRL Framework)

A comprehensive framework for developing quantitative trading strategies using
reinforcement learning, multi-factor analysis, and advanced risk management.

Key Features:
1. Download/load stock price data (OHLCV) 
2. Load pre-calculated factor scores 
3. Merge OHLCV with factor data 
4. Feature engineering: technical indicators 
5. **Three-stage data splitting (Train/Validation/Test)** 
6. Optuna hyperparameter optimization on validation set 
7. Custom RL environment with risk management 
8. RL model training using Stable Baselines 3 (PPO) 
9. Model validation and final testing 
10. Performance metrics with Buy&Hold comparison 
11. Interactive trading visualization 
12. Comprehensive reporting and logging 

Architecture Improvements:
- Validation set prevents overfitting on test data 
- Hyperparameter tuning on validation set 
- Test set for unbiased final performance 
- Follows academic and industry best practices 
"""

import os
import datetime
import logging
import warnings
import traceback
from functools import partial
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import yfinance as yf
import talib
import gym
from gym import spaces
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import optuna
from sklearn.preprocessing import StandardScaler

# Reinforcement Learning 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as sb3_configure

# Suppress warnings 
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
pd.options.mode.chained_assignment = None

# ============================================================================
# Configuration Management 
# ============================================================================

class TradingConfig:
    """
    Centralized configuration management for trading system
    """

    # Logging configuration 
    LOGGING_LEVEL = logging.INFO
    DATA_DIR = "quantitative_trading_data"
    LOG_FILE_NAME = "trading_strategy.log"

    # Stock configuration - US Stocks 
    STOCK_TICKER = "MSFT"
    # Alternative options: "AAPL"(Apple), "AMZN"(Amazon), "MSFT"(Microsoft), "NVDA"(Nvidia)

    # Date configuration - Three-stage split 
    # CRITICAL: Validation set is used for hyperparameter optimization
    TRAIN_START_DATE = "2000-01-01"           # Training start 
    TRAIN_END_DATE = "2021-12-31"             # Training end 
    VALIDATION_START_DATE = "2022-01-01"      # Validation start 
    VALIDATION_END_DATE = "2024-06-30"        # Validation end 
    TEST_START_DATE = "2024-07-01"            # Test start 
    TEST_END_DATE = "2025-06-30"              # Test end 

    # Financial configuration 
    INITIAL_BALANCE = 1_000_000  # USD for US stocks 
    WINDOW_SIZE = 40  # RL environment observation window size 

    # File paths 
    @classmethod
    def get_raw_data_path(cls):
        """Get raw OHLCV data file path"""
        return os.path.join(cls.DATA_DIR, f"{cls.STOCK_TICKER}_raw_data.csv")

    @classmethod
    def get_factor_data_path(cls):
        """Get factor score data file path"""
        return os.path.join(cls.DATA_DIR, f"{cls.STOCK_TICKER}_score_data_ok.csv")

    @classmethod
    def get_processed_data_path(cls):
        """Get processed merged data file path"""
        return os.path.join(cls.DATA_DIR, f"{cls.STOCK_TICKER}_final_processed_data.csv")

    @classmethod
    def get_model_path(cls):
        """Get trained model file path"""
        return os.path.join(cls.DATA_DIR, f"{cls.STOCK_TICKER}_rl_model_best.zip")

    @classmethod
    def get_trades_path(cls, phase: str = "test"):
        """
        Get trade log file path
        
        Args:
            phase: "validation" or "test"
        """
        return os.path.join(cls.DATA_DIR, f"{cls.STOCK_TICKER}_trades_{phase}.csv")

    @classmethod
    def get_report_path(cls):
        """Get final report file path"""
        return os.path.join(cls.DATA_DIR, f"{cls.STOCK_TICKER}_report_best.md")

    @classmethod
    def get_chart_path(cls, phase: str = "test"):
        """
        Get trading chart file path
        
        Args:
            phase: "validation" or "test"
        """
        return os.path.join(cls.DATA_DIR, f"{cls.STOCK_TICKER}_trading_chart_{phase}.html")

    @classmethod
    def get_sb3_log_dir(cls):
        """Get Stable Baselines 3 log directory"""
        return os.path.join(cls.DATA_DIR, f"{cls.STOCK_TICKER}_sb3_logs")

    @classmethod
    def get_optuna_db_path(cls):
        """Get Optuna study database path"""
        return os.path.join(cls.DATA_DIR, f"{cls.STOCK_TICKER}_optuna_study.db")

    # Expected factor columns 
    EXPECTED_FACTOR_COLUMNS = [
        'fundamental_score',      # Fundamental Score
        'sentiment_score',        # Sentiment Score
        'industry_trend_score',   # Industry Trend Score
        'market_risk_factor',     # Market Risk Factor
        'black_swan_risk'         # Black Swan Risk
    ]

    # RL configuration 
    TOTAL_TRAINING_TIMESTEPS = 100_000  # Final model training steps 
    TRIAL_TRAINING_TIMESTEPS = 10_000   # Optuna optimization training steps 
    RL_ALGORITHM = PPO                  # Reinforcement learning algorithm 

    # Trading configuration 
    COMMISSION_RATE = 0.001 * 2  # US stock commission (buy + sell) 
    SLIPPAGE = 0.001             # Slippage rate 
    MIN_TRADE_SHARES = 100       # Minimum trade size for US stocks 

    # Risk management configuration 
    PROFIT_REWARD_FACTOR = 0.3          # Profit reward factor 
    VOLATILITY_PENALTY_FACTOR = 0.035   # Volatility penalty factor 
    STOP_LOSS_PCT = 0.05                # Stop loss at 5% 
    TAKE_PROFIT_PCT = 0.5               # Take profit at 50% 
    MAX_POSITION_RISK_PCT = 0.1         # Maximum position risk percentage 
    ATR_PERIOD_FOR_SIZING = 14          # ATR period for position sizing 

    # Optuna configuration 
    N_OPTUNA_TRIALS = 20  # Number of optimization trials 


# ============================================================================
# Logging Setup 
# ============================================================================

def setup_logger() -> logging.Logger:
    """
    Setup and configure logging system
    """
    log_file_path = os.path.join(TradingConfig.DATA_DIR, TradingConfig.LOG_FILE_NAME)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logger = logging.getLogger("trading")
    logger.setLevel(TradingConfig.LOGGING_LEVEL)

    # Clear existing handlers 
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )

    # Console handler 
    console_handler = logging.StreamHandler()
    console_handler.setLevel(TradingConfig.LOGGING_LEVEL)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler 
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(TradingConfig.LOGGING_LEVEL)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Trading system logger setup completed")
    logger.debug(f"Log level set to: {logging.getLevelName(TradingConfig.LOGGING_LEVEL)}")

    return logger


# ============================================================================
# Data Loading and Preprocessing 
# ============================================================================

class DataManager:
    """
    Manages data loading, processing, and technical indicator calculation
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def load_raw_stock_data(self, ticker: str, start_date: str, end_date: str,
                           save_path: str) -> Optional[pd.DataFrame]:
        """
        Download or load raw OHLCV data
        """
        self.logger.info(f"Loading raw data for {ticker} from {start_date} to {end_date}")

        # Try loading from local file first 
        if os.path.exists(save_path):
            self.logger.info(f"Attempting to load OHLCV data from {save_path}...")
            try:
                df = pd.read_csv(save_path, index_col='Date', parse_dates=True)
                required_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

                if not all(col in df.columns for col in required_cols):
                    self.logger.warning(f"Loaded data missing required columns, will re-download.")
                    df = None
                elif df.empty:
                    self.logger.warning(f"Loaded data is empty, will re-download.")
                    df = None
                else:
                    self.logger.info(f"Successfully loaded OHLCV data from {save_path}.")
                    return df
            except Exception as e:
                self.logger.exception(f"Error loading data from {save_path}, will re-download: {e}")
                df = None

        # Download from yfinance 
        self.logger.info(f"Downloading {ticker} OHLCV data from yfinance...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False,
                           auto_adjust=False, interval='1d')

            if df.empty:
                self.logger.error(f"Downloaded data for {ticker} is empty.")
                return None

            self.logger.info("Download completed.")

            # Handle MultiIndex columns 
            if isinstance(df.columns, pd.MultiIndex):
                self.logger.debug("Processing MultiIndex columns...")
                df.columns = df.columns.get_level_values(0)

            # Handle duplicate columns 
            if df.columns.has_duplicates:
                self.logger.warning("Processing duplicate columns...")
                df = df.loc[:, ~df.columns.duplicated()]

            # Standardize column names 
            df.columns = [col.capitalize() for col in df.columns]
            if 'Adj close' in df.columns:
                df.rename(columns={'Adj close': 'Adj Close'}, inplace=True)

            # Validate required columns 
            required_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Downloaded data missing required columns: {set(required_cols)-set(df.columns)}")
                return None

            # Clean data 
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
            for col in price_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove invalid data 
            df.dropna(subset=['Volume'] + price_cols, inplace=True)
            df = df[df['Volume'] > 0]

            if df.empty:
                self.logger.error("Data is empty after cleaning.")
                return None

            # Save to local file 
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path)
            self.logger.info(f"Raw OHLCV data saved to {save_path}")

            return df

        except Exception as e:
            self.logger.exception(f"Error downloading or processing OHLCV data: {e}")
            return None

    def load_factor_data(self, factor_path: str, expected_columns: List[str]) -> Optional[pd.DataFrame]:
        """
        Load pre-calculated factor data
        """
        self.logger.info(f"Loading factor data from: {factor_path}")

        if not os.path.exists(factor_path):
            self.logger.error(f"Factor data file not found: {factor_path}")
            return None

        try:
            df_factors = pd.read_csv(factor_path, index_col='Date', parse_dates=True)

            # Check required columns 
            missing_factors = [col for col in expected_columns if col not in df_factors.columns]
            if missing_factors:
                self.logger.error(f"Factor data missing expected columns: {missing_factors}")
                return None

            self.logger.info(f"Successfully loaded factor data with columns: {df_factors.columns.tolist()}")
            return df_factors

        except Exception as e:
            self.logger.exception(f"Error loading factor data from {factor_path}: {e}")
            return None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Calculate technical indicators using TA-Lib
        """
        self.logger.info("Calculating technical indicators...")

        if df is None or df.empty:
            self.logger.error("Input DataFrame is empty or None.")
            return None

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            self.logger.error(f"Missing required columns for technical indicators: {set(required_cols) - set(df.columns)}")
            return None

        df_out = df.copy()

        try:
            # Convert to numpy arrays with float64 dtype 
            open_price = df_out['Open'].values.astype(np.float64)
            high_price = df_out['High'].values.astype(np.float64)
            low_price = df_out['Low'].values.astype(np.float64)
            close_price = df_out['Close'].values.astype(np.float64)
            volume = df_out['Volume'].values.astype(np.float64)

            self.logger.debug("Successfully converted input data to float64 numpy arrays.")

        except Exception as e:
            self.logger.exception(f"Error converting data types to float64: {e}")
            return None

        # Check minimum data requirements 
        min_required_period = 200
        if len(close_price) < min_required_period:
            self.logger.warning(f"Data length ({len(close_price)}) may be insufficient for all technical indicators.")

        try:
            # Overlap studies 
            self.logger.debug("Calculating overlap studies...")
            df_out['SMA_10'] = talib.SMA(close_price, timeperiod=10)
            df_out['SMA_30'] = talib.SMA(close_price, timeperiod=30)
            df_out['SMA_50'] = talib.SMA(close_price, timeperiod=50)
            df_out['SMA_200'] = talib.SMA(close_price, timeperiod=200)
            df_out['EMA_10'] = talib.EMA(close_price, timeperiod=10)
            df_out['EMA_30'] = talib.EMA(close_price, timeperiod=30)

            # Bollinger Bands 
            upper, middle, lower = talib.BBANDS(close_price, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df_out['BB_UPPER'] = upper
            df_out['BB_MIDDLE'] = middle
            df_out['BB_LOWER'] = lower

            # Parabolic SAR 
            df_out['SAR'] = talib.SAR(high_price, low_price, acceleration=0.02, maximum=0.2)

            # Momentum indicators 
            self.logger.debug("Calculating momentum indicators...")
            df_out['RSI'] = talib.RSI(close_price, timeperiod=14)

            # MACD
            macd, macdsignal, macdhist = talib.MACD(close_price, fastperiod=12, slowperiod=26, signalperiod=9)
            df_out['MACD'] = macd
            df_out['MACD_signal'] = macdsignal
            df_out['MACD_hist'] = macdhist

            # Stochastic 
            slowk, slowd = talib.STOCH(high_price, low_price, close_price,
                                     fastk_period=14, slowk_period=3, slowk_matype=0,
                                     slowd_period=3, slowd_matype=0)
            df_out['STOCH_k'] = slowk
            df_out['STOCH_d'] = slowd

            # Other momentum indicators 
            df_out['ADX'] = talib.ADX(high_price, low_price, close_price, timeperiod=14)
            df_out['CCI'] = talib.CCI(high_price, low_price, close_price, timeperiod=14)
            df_out['WILLR'] = talib.WILLR(high_price, low_price, close_price, timeperiod=14)
            df_out['MOM'] = talib.MOM(close_price, timeperiod=10)

            # Volume indicators 
            self.logger.debug("Calculating volume indicators...")
            df_out['OBV'] = talib.OBV(close_price, volume)
            df_out['AD'] = talib.AD(high_price, low_price, close_price, volume)

            # Volatility indicators /
            self.logger.debug("Calculating volatility indicators...")
            df_out['ATR'] = talib.ATR(high_price, low_price, close_price, timeperiod=TradingConfig.ATR_PERIOD_FOR_SIZING)
            df_out['NATR'] = talib.NATR(high_price, low_price, close_price, timeperiod=14)
            df_out['TRANGE'] = talib.TRANGE(high_price, low_price, close_price)

            # Cycle indicators 
            self.logger.debug("Calculating cycle indicators...")
            df_out['HT_DCPERIOD'] = talib.HT_DCPERIOD(close_price)

            # Count added indicators 
            base_cols = set(df.columns)
            num_indicators_added = len(set(df_out.columns) - base_cols)
            self.logger.info(f"Calculated {num_indicators_added} technical indicators.")

            return df_out

        except Exception as e:
            self.logger.exception(f"Error calculating technical indicators: {e}")
            return None

    def merge_and_process_data(self, ohlcv_df: pd.DataFrame, factor_df: pd.DataFrame,
                              output_path: str) -> Optional[pd.DataFrame]:
        """
        Merge price data with factor data, calculate indicators, and handle missing values
        """
        self.logger.info("Merging and processing data...")

        if ohlcv_df is None or factor_df is None:
            self.logger.error("OHLCV or factor data is None, cannot merge.")
            return None

        try:
            # Ensure datetime index 
            ohlcv_df.index = pd.to_datetime(ohlcv_df.index)
            factor_df.index = pd.to_datetime(factor_df.index)

            # Merge data 
            merged_df = pd.merge(
                ohlcv_df,
                factor_df[TradingConfig.EXPECTED_FACTOR_COLUMNS],
                left_index=True,
                right_index=True,
                how='inner'
            )

            self.logger.info(f"Successfully merged price and factor data: {len(merged_df)} records.")

            if merged_df.empty:
                self.logger.error("Merged data is empty. Check date ranges or index matching.")
                return None

        except Exception as e:
            self.logger.exception(f"Error merging price and factor data: {e}")
            return None

        # Calculate technical indicators 
        df_with_indicators = self.calculate_technical_indicators(merged_df)
        if df_with_indicators is None:
            self.logger.error("Technical indicator calculation failed.")
            return None

        # Handle missing values 
        self.logger.info("Processing missing values in final data...")
        initial_len = len(df_with_indicators)

        df_with_indicators.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_final = df_with_indicators.dropna()

        final_len = len(df_final)
        rows_removed = initial_len - final_len
        self.logger.info(f"Missing value processing: removed {rows_removed} rows.")

        if df_final.empty:
            self.logger.error("Data is empty after processing missing values.")
            return None

        # Save processed data 
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            df_final.to_csv(output_path)
            self.logger.info(f"Final merged and processed data saved to {output_path}")
        except Exception as e:
            self.logger.exception(f"Failed to save processed data to {output_path}: {e}")
            return None

        self.logger.info("Data merging and processing completed.")
        return df_final


# ============================================================================
# Reinforcement Learning Environment 
# ============================================================================

class StockTradingEnvironment(gym.Env):
    """
    Custom stock trading environment with advanced risk management
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame,
                 window_size: int = TradingConfig.WINDOW_SIZE,
                 initial_balance: float = TradingConfig.INITIAL_BALANCE,
                 commission_rate: float = TradingConfig.COMMISSION_RATE,
                 slippage: float = TradingConfig.SLIPPAGE,
                 min_trade_shares: int = TradingConfig.MIN_TRADE_SHARES,
                 profit_reward_factor: float = TradingConfig.PROFIT_REWARD_FACTOR,
                 volatility_penalty_factor: float = TradingConfig.VOLATILITY_PENALTY_FACTOR,
                 stop_loss_pct: float = TradingConfig.STOP_LOSS_PCT,
                 take_profit_pct: float = TradingConfig.TAKE_PROFIT_PCT,
                 max_position_risk_pct: float = TradingConfig.MAX_POSITION_RISK_PCT,
                 atr_period: int = TradingConfig.ATR_PERIOD_FOR_SIZING,
                 main_logger: Optional[logging.Logger] = None):
        """
        Initialize trading environment
        """
        super(StockTradingEnvironment, self).__init__()

        # Setup environment-specific logger 
        env_id = id(self)
        if main_logger:
            self.logger = main_logger.getChild(f"Env_{env_id}")
        else:
            self.logger = logging.getLogger(f"trading.Env_{env_id}")

        self.logger.info(f"Initializing StockTradingEnvironment (ID: {env_id})...")

        # Validate input data 
        if df is None or df.empty:
            self.logger.error("Input DataFrame is empty or None.")
            raise ValueError("DataFrame cannot be None or empty for StockTradingEnvironment")

        if 'ATR' not in df.columns:
            self.logger.error("DataFrame missing 'ATR' column.")
            raise ValueError("DataFrame missing 'ATR' column")

        # Get factor columns 
        self.factor_columns = [col for col in TradingConfig.EXPECTED_FACTOR_COLUMNS if col in df.columns]
        if len(self.factor_columns) != len(TradingConfig.EXPECTED_FACTOR_COLUMNS):
            self.logger.warning(f"DataFrame factor columns ({len(self.factor_columns)}) don't match expected ({len(TradingConfig.EXPECTED_FACTOR_COLUMNS)}).")
            self.logger.warning(f"Available factor columns: {self.factor_columns}")

        self.n_factor_features = len(self.factor_columns)

        # Store parameters 
        self.df = df.copy()
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.min_trade_shares = min_trade_shares
        self.profit_reward_factor = profit_reward_factor
        self.volatility_penalty_factor = volatility_penalty_factor
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_position_risk_pct = max_position_risk_pct
        self.atr_period = atr_period

        # Define action space: 0=sell, 1=hold, 2=buy 
        self.action_space = spaces.Discrete(3)

        # Calculate observation space dimensions 
        n_price_volume_features = 6  # Open, High, Low, Close, Adj Close, Volume
        self.technical_indicator_columns = [
            col for col in self.df.columns
            if col not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] + self.factor_columns
        ]
        n_technical_indicators = len(self.technical_indicator_columns)
        n_portfolio_features = 3  # cash ratio, stock ratio, pnl ratio

        self.observation_shape = (
            window_size,
            n_price_volume_features + n_technical_indicators + self.n_factor_features + n_portfolio_features
        )

        # Define observation space 
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.observation_shape,
            dtype=np.float32
        )

        self.logger.info(f"Environment observation space: {self.observation_shape}")
        self.logger.debug(f"Price/volume features: {n_price_volume_features}")
        self.logger.debug(f"Technical indicators: {n_technical_indicators}")
        self.logger.debug(f"Factor features: {self.n_factor_features}")
        self.logger.debug(f"Portfolio features: {n_portfolio_features}")

        # Initialize state variables 
        self.current_step = 0
        self.balance = 0
        self.shares_held = 0
        self.net_worth = 0
        self.max_net_worth = 0
        self.trade_history = []
        self.portfolio_value_history = []
        self.entry_price = 0.0
        self.last_trade_profit = 0.0

        # Setup scaler for feature normalization 
        self.scaler = StandardScaler()
        if not self._fit_scaler():
            raise RuntimeError("Environment initialization failed: Scaler fitting failed.")

        # Reset environment 
        self.reset()
        self.logger.info(f"StockTradingEnvironment (ID: {env_id}) initialization completed.")

    def _fit_scaler(self) -> bool:
        """
        Fit scaler for observation normalization
        """
        self.logger.debug("Fitting scaler for feature normalization...")

        # Determine columns to scale 
        cols_to_scale = (['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] +
                        self.technical_indicator_columns + self.factor_columns)
        available_cols = [col for col in cols_to_scale if col in self.df.columns]

        if len(available_cols) != len(cols_to_scale):
            self.logger.warning("DataFrame missing some expected columns for scaling.")
            missing = set(cols_to_scale) - set(available_cols)
            self.logger.warning(f"Missing: {missing}")
            cols_to_scale = available_cols

        if not cols_to_scale:
            self.logger.error("No columns available for scaling.")
            return False

        # Prepare data for fitting 
        observable_df = self.df[cols_to_scale].copy()
        observable_df.replace([np.inf, -np.inf], 0, inplace=True)
        observable_df.fillna(0, inplace=True)

        try:
            self.scaler.fit(observable_df)
            self.logger.info(f"Scaler fitted successfully on {len(cols_to_scale)} columns.")
            return True
        except Exception as e:
            self.logger.exception(f"Scaler fitting failed: {e}")
            return False

    def _get_current_price(self) -> float:
        """
        Get current step's closing price
        """
        idx = self.current_step + self.window_size - 1
        if idx < len(self.df):
            return self.df['Close'].iloc[idx]
        else:
            self.logger.warning(f"Price index {idx} out of range, returning last price.")
            return self.df['Close'].iloc[-1]

    def _get_current_atr(self) -> float:
        """
        Get current step's ATR value
        """
        idx = self.current_step + self.window_size - 1
        if idx < len(self.df):
            return self.df['ATR'].iloc[idx]
        else:
            self.logger.warning(f"ATR index {idx} out of range, returning NaN.")
            return np.nan

    def _next_observation(self) -> np.ndarray:
        """
        Get next observation state
        """
        self.logger.debug(f"Getting observation for step: {self.current_step}")

        # Get current window data 
        start_idx = self.current_step
        end_idx = self.current_step + self.window_size

        if end_idx > len(self.df):
            self.logger.error(f"Requested end index {end_idx} exceeds DataFrame length {len(self.df)}.")
            return np.zeros(self.observation_shape, dtype=np.float32)

        frame = self.df.iloc[start_idx:end_idx]

        # Determine columns to scale 
        cols_to_scale = (['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] +
                        self.technical_indicator_columns + self.factor_columns)
        available_cols = [col for col in cols_to_scale if col in frame.columns]

        if len(available_cols) != len(cols_to_scale):
            self.logger.warning("DataFrame missing some expected columns for observation.")
            cols_to_scale = available_cols

        if not cols_to_scale:
            self.logger.error("No columns available for observation scaling.")
            return np.zeros(self.observation_shape, dtype=np.float32)

        # Prepare and scale features 
        obs_features = frame[cols_to_scale].values
        obs_features[np.isinf(obs_features)] = 0
        obs_features = np.nan_to_num(obs_features)

        # Scale features 
        try:
            scaled_features = self.scaler.transform(obs_features)
        except Exception as e:
            self.logger.exception(f"Scaler transformation failed: {e}")
            return np.zeros(self.observation_shape, dtype=np.float32)

        # Calculate portfolio state 
        current_price = self._get_current_price()
        total_value = self.balance + self.shares_held * current_price

        balance_ratio = self.balance / total_value if total_value > 0 else 0
        shares_ratio = (self.shares_held * current_price) / total_value if total_value > 0 else 0
        pnl_ratio = (total_value - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0

        portfolio_state = np.array([balance_ratio, shares_ratio, pnl_ratio])
        portfolio_state_expanded = np.tile(portfolio_state, (self.window_size, 1))

        # Validate scaled features shape 
        expected_scaled_shape = (self.window_size, len(cols_to_scale))
        if scaled_features.shape != expected_scaled_shape:
            self.logger.error(f"Scaled features shape mismatch. Expected {expected_scaled_shape}, got {scaled_features.shape}")
            return np.zeros(self.observation_shape, dtype=np.float32)

        # Combine observation 
        try:
            observation = np.hstack((scaled_features, portfolio_state_expanded))
        except ValueError as e:
            self.logger.error(f"Observation concatenation failed: {e}")
            self.logger.error(f"Scaled features shape: {scaled_features.shape}")
            self.logger.error(f"Portfolio state shape: {portfolio_state_expanded.shape}")
            return np.zeros(self.observation_shape, dtype=np.float32)

        # Validate final observation shape 
        if observation.shape != self.observation_shape:
            self.logger.error(f"Final observation shape mismatch. Expected {self.observation_shape}, got {observation.shape}")
            return np.zeros(self.observation_shape, dtype=np.float32)

        self.logger.debug(f"Observation generated successfully, shape: {observation.shape}")
        return observation.astype(np.float32)

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state
        """
        self.logger.info("Resetting environment...")

        # Reset state variables 
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.current_step = 0
        self.trade_history = []
        self.portfolio_value_history = [self.initial_balance]
        self.entry_price = 0.0
        self.last_trade_profit = 0.0

        # Validate data sufficiency 
        if len(self.df) < self.window_size:
            self.logger.error(f"Data length ({len(self.df)}) less than window size ({self.window_size}).")
            raise ValueError(f"Data length ({len(self.df)}) less than window size ({self.window_size})")

        self.logger.info("Environment reset completed.")
        return self._next_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one time step with stop-loss/take-profit and reward shaping
        """
        self.logger.debug(f"Executing step {self.current_step}, original action: {action}")

        # Check if reached end of data 
        is_done = self.current_step >= len(self.df) - self.window_size
        if is_done:
            self.logger.info(f"Reached end of data (step {self.current_step}), environment done.")
            try:
                final_obs = self._next_observation()
            except Exception as e:
                self.logger.exception("Error getting final observation, returning zeros.")
                final_obs = np.zeros(self.observation_shape, dtype=np.float32)

            return final_obs, 0.0, True, {'net_worth': self.net_worth, 'trades': len(self.trade_history)}

        # Get current price and save previous net worth 
        current_price = self._get_current_price()
        prev_net_worth = self.net_worth
        self.last_trade_profit = 0.0

        # Initialize trigger flags 
        stop_loss_triggered = False
        take_profit_triggered = False

        # --- Take-profit logic (priority over stop-loss) ---
        if (self.shares_held > 0 and self.entry_price > 0 and self.take_profit_pct > 0):
            take_profit_price = self.entry_price * (1 + self.take_profit_pct)
            if current_price >= take_profit_price:
                self.logger.info(f"Take-profit triggered! Current {current_price:.2f} >= target {take_profit_price:.2f} (entry {self.entry_price:.2f})")
                action = 0  # Force sell 
                take_profit_triggered = True

        # --- Stop-loss logic (if take-profit not triggered) ---
        if (not take_profit_triggered and self.shares_held > 0 and
            self.entry_price > 0 and self.stop_loss_pct > 0):
            stop_loss_price = self.entry_price * (1 - self.stop_loss_pct)
            if current_price <= stop_loss_price:
                self.logger.info(f"Stop-loss triggered! Current {current_price:.2f} <= stop {stop_loss_price:.2f} (entry {self.entry_price:.2f})")
                action = 0  # Force sell 
                stop_loss_triggered = True

        self.logger.debug(f"Final action: {action} (take-profit: {take_profit_triggered}, stop-loss: {stop_loss_triggered})")

        # Execute trading action 
        self._take_action(action, current_price)

        # --- Update state and calculate reward ---
        self.current_step += 1
        next_price_idx = self.current_step + self.window_size - 1

        if next_price_idx < len(self.df):
            next_price = self.df['Close'].iloc[next_price_idx]
        else:
            self.logger.warning(f"Next price index {next_price_idx} out of range for reward calculation.")
            next_price = current_price

        # Calculate net worth using Close price 
        self.net_worth = self.balance + self.shares_held * next_price
        self.portfolio_value_history.append(self.net_worth)
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # --- Calculate shaped reward ---
        base_reward = (self.net_worth - prev_net_worth) / prev_net_worth if prev_net_worth != 0 else 0

        # Profit bonus for take-profit 
        profit_bonus = 0.0
        if self.trade_history:
            last_trade = self.trade_history[-1]
            # Check if this is a just-completed sell trade 
            if (last_trade['step'] == (self.current_step + self.window_size - 2) and
                last_trade['type'] == 'SELL'):
                if take_profit_triggered:
                    profit_bonus = self.profit_reward_factor
                    self.last_trade_profit = last_trade['cost/proceeds']
                    self.logger.debug(f"Take-profit trade reward: {profit_bonus:.6f}")

        # Volatility penalty 
        volatility_penalty = 0.0
        lookback_period = 20
        if len(self.portfolio_value_history) > lookback_period + 1:
            recent_returns = pd.Series(self.portfolio_value_history[-lookback_period-1:]).pct_change().dropna()
            if not recent_returns.empty:
                returns_std = recent_returns.std()
                volatility_penalty = self.volatility_penalty_factor * returns_std
                self.logger.debug(f"Volatility penalty: std={returns_std:.6f}, penalty={volatility_penalty:.6f}")

        # Stop-loss penalty and take-profit reward 
        stop_loss_penalty = -0.01 if stop_loss_triggered else 0.0
        take_profit_reward = 0.01 if take_profit_triggered else 0.0

        # Total reward 
        reward = base_reward + take_profit_reward - volatility_penalty + stop_loss_penalty

        self.logger.debug(f"Step {self.current_step}: price={next_price:.2f}, net_worth={self.net_worth:.2f}, " +
                         f"Reward breakdown: base={base_reward:.6f}, tp_reward={take_profit_reward:.2f}, " +
                         f"vol_penalty={volatility_penalty:.6f}, sl_penalty={stop_loss_penalty:.2f}, " +
                         f"Total={reward:.6f}")

        # Check if done 
        done = self.current_step >= len(self.df) - self.window_size

        # Get next observation 
        try:
            obs = self._next_observation()
        except Exception as e:
            self.logger.exception("Error getting next observation, returning zeros and ending environment.")
            obs = np.zeros(self.observation_shape, dtype=np.float32)
            done = True

        # Return info 
        info = {
            'net_worth': self.net_worth,
            'trades': len(self.trade_history),
            'last_trade_profit': self.last_trade_profit
        }

        self.logger.debug(f"Step completed, returning: done={done}, info={info}")
        return obs, reward, done, info

    def _take_action(self, action: int, current_price: float) -> None:
        """
        Execute buy, sell, or hold action with dynamic position management
        """
        action_type = action
        trade_executed = False
        trade_type = "HOLD"
        trade_shares = 0
        trade_cost = 0

        buy_price = current_price * (1 + self.slippage)
        sell_price = current_price * (1 - self.slippage)

        if action_type == 2:  # Buy 
            if self.balance > 0:
                # Dynamic position management based on ATR 
                current_atr = self._get_current_atr()

                if np.isnan(current_atr) or current_atr <= 0:
                    self.logger.warning(f"Invalid ATR ({current_atr}), using default position sizing")
                    position_pct = 0.5
                    cash_for_trade = self.balance * 0.95 * position_pct
                    shares_to_buy = int(cash_for_trade / buy_price / self.min_trade_shares) * self.min_trade_shares
                    self.logger.debug(f"Dynamic position (invalid ATR): using {position_pct*100:.1f}% cash, shares={shares_to_buy}")
                else:
                    # Risk-based position calculation 
                    total_asset_value = self.balance + self.shares_held * current_price
                    max_shares_by_risk = (total_asset_value * self.max_position_risk_pct) / current_atr
                    cash_for_trade = self.balance * 0.95
                    max_shares_by_cash = cash_for_trade / buy_price
                    target_shares = min(max_shares_by_risk, max_shares_by_cash)
                    shares_to_buy = int(target_shares / self.min_trade_shares) * self.min_trade_shares
                    self.logger.debug(f"Dynamic position: ATR={current_atr:.2f}, max_risk_shares={max_shares_by_risk:.0f}, "
                                    f"max_cash_shares={max_shares_by_cash:.0f}, target={target_shares:.0f}, buy={shares_to_buy}")

                if shares_to_buy > 0:
                    # Calculate cost including commission 
                    cost = shares_to_buy * buy_price * (1 + self.commission_rate)

                    if self.balance >= cost:
                        self.balance -= cost

                        # Record entry price 
                        if self.shares_held == 0:
                            self.entry_price = buy_price
                            self.logger.debug(f"Recording new entry price: {self.entry_price:.2f}")
                        else:
                            self.logger.debug(f"Adding to position, keeping initial entry price: {self.entry_price:.2f}")

                        self.shares_held += shares_to_buy
                        trade_executed = True
                        trade_type = "BUY"
                        trade_shares = shares_to_buy
                        trade_cost = -cost
                        self.last_trade_cost = cost
                        self.logger.info(f"Executed BUY (dynamic position): {trade_shares} shares @ {current_price:.2f}, cost: {cost:.2f}")
                    else:
                        self.logger.warning(f"Insufficient balance for buy: need {cost:.2f}, have {self.balance:.2f}")
                else:
                    self.logger.debug("Calculated buy shares is 0.")
            else:
                self.logger.debug("Balance is 0, cannot buy.")

        elif action_type == 0:  # Sell 
            if self.shares_held > 0:
                # Calculate sell value and commission 
                sell_value = self.shares_held * sell_price
                commission = sell_value * self.commission_rate
                proceeds = sell_value - commission

                self.balance += proceeds
                trade_executed = True
                trade_type = "SELL"
                trade_shares = self.shares_held
                trade_cost = proceeds

                self.logger.info(f"Executed SELL: {trade_shares} shares @ {current_price:.2f} "
                                f"(actual sell price with slippage: {sell_price:.2f}), proceeds: {proceeds:.2f}")

                self.shares_held = 0
                self.last_trade_cost = 0
                self.entry_price = 0.0  # Clear entry price 
            else:
                self.logger.debug("Attempted to sell but holding no shares.")

        # Record trade 
        if trade_executed:
            step_index = self.current_step + self.window_size - 1
            if step_index < len(self.df):
                trade_info = {
                    'step': step_index,
                    'date': self.df.index[step_index],
                    'type': trade_type,
                    'price': current_price,
                    'shares': trade_shares,
                    'cost/proceeds': trade_cost,
                    'balance': self.balance,
                    'shares_held': self.shares_held,
                    'net_worth': self.balance + self.shares_held * current_price
                }
                self.trade_history.append(trade_info)
                self.logger.debug(f"Recorded trade: {trade_info}")
            else:
                self.logger.warning(f"Trade recording: index {step_index} out of range.")

    def render(self, mode='human', close=False):
        """
        Render environment state
        """
        current_price = self._get_current_price()
        net_worth = self.balance + self.shares_held * current_price
        step_idx = self.current_step + self.window_size - 1

        if step_idx < len(self.df):
            log_message = (
                f"Render Step: {step_idx}, "
                f"Date: {self.df.index[step_idx].strftime('%Y-%m-%d')}, "
                f"Balance: {self.balance:,.2f}, "
                f"Shares: {self.shares_held}, "
                f"Entry Price: {self.entry_price:.2f}, "
                f"Close: {current_price:,.2f}, "
                f"Net Worth: {net_worth:,.2f}, "
                f"Trades: {len(self.trade_history)}"
            )
            self.logger.info(log_message)
        else:
            self.logger.info("Render: Reached end of data.")


# ============================================================================
# Model Training and Optimization 
# ============================================================================

class TensorboardCallback(BaseCallback):
    """
    Training callback for logging metrics during training
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.tb_logger = logging.getLogger("trading.TensorboardCallback")

    def _on_step(self) -> bool:
        """
        Callback function called at each step
        """
        if len(self.locals['infos']) > 0 and 'net_worth' in self.locals['infos'][0]:
            net_worth = self.locals['infos'][0]['net_worth']
            self.logger.record('rollout/net_worth', net_worth)
        return True


class ModelTrainer:
    """
    Handles RL model training and optimization
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def train_model(self, env: gym.Env, hyperparameters: Dict[str, Any],
                   save_path: str = None, log_dir: str = None,
                   total_timesteps: int = TradingConfig.TOTAL_TRAINING_TIMESTEPS) -> Optional[PPO]:
        """
        Train RL model and save if specified
        """
        self.logger.info(f"Starting model training, save path: {save_path}, SB3 log dir: {log_dir}")
        self.logger.info(f"Using hyperparameters: {hyperparameters}")
        self.logger.info(f"Total training timesteps: {total_timesteps}")

        # Setup SB3 logging 
        if log_dir:
            run_log_dir = os.path.join(log_dir, f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            try:
                os.makedirs(run_log_dir, exist_ok=True)
                sb3_output_formats = ["stdout", "csv", "tensorboard"]
                sb3logger = sb3_configure(run_log_dir, sb3_output_formats)
                self.logger.info(f"Stable Baselines 3 logger setup completed, output to: {run_log_dir}")
            except Exception as e:
                self.logger.exception(f"Failed to setup Stable Baselines 3 logger: {e}")
                sb3logger = None
        else:
            sb3logger = None

        try:
            # Create model with hyperparameters 
            model = TradingConfig.RL_ALGORITHM(
                'MlpPolicy',
                env,
                verbose=0,
                learning_rate=hyperparameters.get('learning_rate', 0.0003),
                n_steps=hyperparameters.get('n_steps', 2048),
                batch_size=64,
                n_epochs=10,
                gamma=hyperparameters.get('gamma', 0.99),
                gae_lambda=hyperparameters.get('gae_lambda', 0.95),
                clip_range=hyperparameters.get('clip_range', 0.2),
                policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
                seed=42
            )

            if sb3logger:
                model.set_logger(sb3logger)

            self.logger.info(f"RL model ({TradingConfig.RL_ALGORITHM.__name__}) created successfully.")

        except Exception as e:
            self.logger.exception(f"Failed to create RL model: {e}")
            return None

        self.logger.info("Starting model training...")
        callback = TensorboardCallback()

        try:
            model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=100)
            self.logger.info("Model training completed.")
        except Exception as e:
            self.logger.exception(f"Error during model training: {e}")
            raise e

        # Save only final models 
        if save_path and total_timesteps == TradingConfig.TOTAL_TRAINING_TIMESTEPS:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            try:
                model.save(save_path)
                self.logger.info(f"Final model saved to {save_path}")
            except Exception as e:
                self.logger.exception(f"Error saving final model: {e}")
        elif save_path:
            self.logger.info(f"Non-final training (steps {total_timesteps}), not saving to {save_path}.")

        self.logger.info("Model training function completed.")
        return model


# ============================================================================
# Backtesting and Evaluation 
# ============================================================================

class BacktestEvaluator:
    """
    Handles model evaluation and backtesting
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def evaluate_model(self, df_eval: pd.DataFrame, model=None, model_path: str = None,
                      initial_balance: float = TradingConfig.INITIAL_BALANCE,
                      window_size: int = TradingConfig.WINDOW_SIZE,
                      save_trades_path: str = None,
                      eval_name: str = "Evaluation") -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Evaluate trained model on evaluation data
        
        Args:
            df_eval: Evaluation DataFrame 
            model: Model object (if provided) 
            model_path: Path to saved model 
            initial_balance: Initial capital 
            window_size: Observation window size 
            save_trades_path: Path to save trade log 
            eval_name: Name for this evaluation (e.g., "Validation", "Test") 
        """
        self.logger.info(f"Starting {eval_name}...")

        # Load model 
        if model is not None:
            self.logger.info(f"Using provided model object for {eval_name}.")
            loaded_model = model
        elif model_path and os.path.exists(model_path):
            self.logger.info(f"Loading model from {model_path} for {eval_name}...")
            try:
                loaded_model = TradingConfig.RL_ALGORITHM.load(model_path)
                self.logger.info(f"Model {model_path} loaded successfully.")
            except Exception as e:
                self.logger.exception(f"Error loading model {model_path}: {e}")
                return None, None
        else:
            self.logger.error(f"No model provided and model file not found: {model_path}")
            return None, None

        # Create evaluation environment 
        self.logger.info(f"Creating {eval_name} environment...")
        try:
            eval_env = StockTradingEnvironment(
                df_eval,
                window_size=window_size,
                initial_balance=initial_balance,
                main_logger=self.logger
            )
            self.logger.info(f"{eval_name} environment created successfully.")
        except Exception as e:
            self.logger.exception(f"Error creating {eval_name} environment: {e}")
            return None, None

        # Run backtest 
        obs = eval_env.reset()
        done = False
        self.logger.info(f"Starting {eval_name} backtest...")
        step_count = 0
        max_steps = len(df_eval) - window_size

        while not done:
            if step_count > max_steps + 5:
                self.logger.warning(f"{eval_name} steps ({step_count}) exceeded expected maximum ({max_steps}), force terminating.")
                break

            try:
                action, _states = loaded_model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                step_count += 1

                if step_count % 250 == 0:
                    self.logger.info(f"{eval_name} progress: Step {step_count}/{max_steps}")

            except Exception as e:
                self.logger.exception(f"Error in {eval_name} step {step_count}: {e}")
                done = True

        self.logger.info(f"{eval_name} completed. Total steps: {step_count}")

        # Process trade log 
        trade_log_df = pd.DataFrame(eval_env.trade_history)

        # Process portfolio value history 
        start_idx_portfolio = window_size - 1
        expected_len_portfolio = step_count + 1
        actual_len_portfolio = len(eval_env.portfolio_value_history)

        if actual_len_portfolio != expected_len_portfolio:
            self.logger.warning(f"{eval_name}: Portfolio history length ({actual_len_portfolio}) doesn't match expected ({expected_len_portfolio}).")
            actual_len_portfolio = min(actual_len_portfolio, expected_len_portfolio)
            eval_env.portfolio_value_history = eval_env.portfolio_value_history[:actual_len_portfolio]

        end_idx_portfolio = start_idx_portfolio + actual_len_portfolio
        if end_idx_portfolio > len(df_eval):
            self.logger.warning(f"{eval_name}: Calculated end index ({end_idx_portfolio}) exceeds data length ({len(df_eval)}).")
            end_idx_portfolio = len(df_eval)
            actual_len_portfolio = end_idx_portfolio - start_idx_portfolio
            eval_env.portfolio_value_history = eval_env.portfolio_value_history[:actual_len_portfolio]

        # Create portfolio value Series 
        portfolio_values = pd.Series(dtype=np.float64)
        if start_idx_portfolio < end_idx_portfolio:
            portfolio_values = pd.Series(
                eval_env.portfolio_value_history,
                index=df_eval.index[start_idx_portfolio:end_idx_portfolio]
            )
            self.logger.info(f"{eval_name}: Successfully obtained portfolio value history, length: {len(portfolio_values)}")
        else:
            self.logger.error(f"{eval_name}: Invalid portfolio index range calculated.")

        # Save trade log 
        if save_trades_path and not trade_log_df.empty:
            os.makedirs(os.path.dirname(save_trades_path), exist_ok=True)
            try:
                trade_log_df['date'] = pd.to_datetime(trade_log_df['date']).dt.strftime('%Y-%m-%d')
                trade_log_df.to_csv(save_trades_path, index=False, float_format='%.2f')
                self.logger.info(f"{eval_name}: Trade log saved to {save_trades_path}")
            except Exception as e:
                self.logger.exception(f"{eval_name}: Error saving trade log: {e}")
        elif not trade_log_df.empty:
            self.logger.info(f"{eval_name}: No save path specified, not saving trade log.")
        else:
            self.logger.info(f"{eval_name}: No trades occurred during backtest.")

        self.logger.info(f"{eval_name} completed.")
        return trade_log_df, portfolio_values


# ============================================================================
# Performance Metrics and Analysis 
# ============================================================================

class PerformanceAnalyzer:
    """
    Calculates and analyzes trading performance metrics
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def calculate_cvar(self, returns: pd.Series, alpha: float = 0.05) -> float:
        """
        Calculate Conditional Value at Risk (CVaR)
        """
        if returns is None or returns.empty or returns.isnull().all():
            return np.nan

        returns = returns.dropna()
        if returns.empty or len(returns) < 2:
            return np.nan

        var = returns.quantile(alpha)
        cvar = returns[returns <= var].mean()
        return cvar

    def calculate_metrics(self, portfolio_values: pd.Series, df_eval: pd.DataFrame,
                         trade_log_df: pd.DataFrame, initial_balance: float = TradingConfig.INITIAL_BALANCE,
                         risk_free_rate: float = 0.0, eval_name: str = "Strategy") -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate and return strategy and Buy&Hold performance metrics
        
        Args:
            eval_name: Name for logging purposes (e.g., "Validation", "Test") 
        """
        self.logger.info(f"Starting performance metrics calculation for {eval_name} (including Buy&Hold)...")

        # Initialize metrics dictionaries 
        strategy_metrics = {
            "Annualized Return": np.nan,
            "Sharpe Ratio": np.nan,
            "Maximum Drawdown": np.nan,
            "Conditional VaR (95%)": np.nan,
            "Annualized Volatility": np.nan,
            "Win Rate": np.nan,
            "Total Trades": len(trade_log_df) if trade_log_df is not None else 0,
            "Final Net Worth": initial_balance
        }

        buy_hold_metrics = {
            "Annualized Return": np.nan,
            "Sharpe Ratio": np.nan,
            "Maximum Drawdown": np.nan,
            "Conditional VaR (95%)": np.nan,
            "Annualized Volatility": np.nan,
            "Win Rate": np.nan,
            "Total Trades": 1,
            "Final Net Worth": initial_balance
        }

        # Check portfolio values 
        if portfolio_values is None or portfolio_values.empty:
            self.logger.warning(f"{eval_name}: Strategy portfolio value series is empty.")
            return strategy_metrics, buy_hold_metrics

        if len(portfolio_values) < 2:
            self.logger.warning(f"{eval_name}: Strategy portfolio value series too short (<2).")
            strategy_metrics["Final Net Worth"] = portfolio_values.iloc[-1] if not portfolio_values.empty else initial_balance
            return strategy_metrics, buy_hold_metrics

        # Update strategy final net worth 
        strategy_metrics["Final Net Worth"] = portfolio_values.iloc[-1]
        self.logger.debug(f"{eval_name}: Strategy final net worth: {strategy_metrics['Final Net Worth']:.2f}")

        # Calculate strategy metrics 
        returns = portfolio_values.pct_change().dropna()
        if not returns.empty and not returns.isnull().all():
            annual_factor = 252  # Trading days 
            total_return = (portfolio_values.iloc[-1] / initial_balance) - 1
            num_trading_days = len(returns)

            if num_trading_days > 0:
                if num_trading_days < annual_factor / 4:
                    self.logger.warning(f"{eval_name}: Strategy trading days ({num_trading_days}) too few.")

                # Annualized return 
                annual_return = ((1 + total_return) ** (annual_factor / num_trading_days)) - 1
                strategy_metrics["Annualized Return"] = annual_return
                self.logger.debug(f"{eval_name}: Strategy annualized return: {annual_return:.2%}")

            # Annualized volatility 
            annual_volatility = returns.std() * np.sqrt(annual_factor)
            strategy_metrics["Annualized Volatility"] = 0.0 if np.isnan(annual_volatility) else annual_volatility
            self.logger.debug(f"{eval_name}: Strategy annualized volatility: {strategy_metrics['Annualized Volatility']:.2%}")

            # Sharpe ratio 
            mean_daily_return = returns.mean()
            if annual_volatility != 0 and not np.isnan(annual_volatility):
                sharpe_ratio = (mean_daily_return * annual_factor - risk_free_rate) / annual_volatility
                strategy_metrics["Sharpe Ratio"] = sharpe_ratio
                self.logger.debug(f"{eval_name}: Strategy Sharpe ratio: {sharpe_ratio:.3f}")
            else:
                strategy_metrics["Sharpe Ratio"] = 0.0

            # Maximum drawdown 
            roll_max = portfolio_values.cummax()
            daily_drawdown = portfolio_values / roll_max - 1.0
            max_drawdown = daily_drawdown.min()
            strategy_metrics["Maximum Drawdown"] = 0.0 if np.isnan(max_drawdown) else max_drawdown
            self.logger.debug(f"{eval_name}: Strategy maximum drawdown: {max_drawdown:.2%}")

            # Conditional VaR 
            cvar_95 = self.calculate_cvar(returns, alpha=0.05)
            strategy_metrics["Conditional VaR (95%)"] = cvar_95
            self.logger.debug(f"{eval_name}: Strategy CVaR (95%): {cvar_95:.2%}")

            # Calculate win rate 
            win_rate = np.nan
            if trade_log_df is not None and not trade_log_df.empty:
                profitable_trades = 0
                total_closed_trades = 0
                entry_cost_total = 0
                entry_shares_total = 0

                self.logger.debug(f"{eval_name}: Calculating strategy win rate...")

                for i, trade in trade_log_df.iterrows():
                    if trade['type'] == 'BUY':
                        entry_cost_total += abs(trade['cost/proceeds'])
                        entry_shares_total += trade['shares']
                    elif trade['type'] == 'SELL' and entry_shares_total > 0:
                        exit_proceeds = trade['cost/proceeds']
                        avg_entry_cost_per_share = entry_cost_total / entry_shares_total if entry_shares_total > 0 else 0
                        shares_sold = trade['shares']

                        if shares_sold > entry_shares_total + 1:
                            self.logger.warning(f"Shares sold ({shares_sold}) > shares held ({entry_shares_total}).")
                            shares_sold = entry_shares_total

                        cost_of_sold_shares = avg_entry_cost_per_share * shares_sold
                        profit = exit_proceeds - cost_of_sold_shares
                        total_closed_trades += 1

                        if profit > 0:
                            profitable_trades += 1

                        entry_cost_total = 0
                        entry_shares_total = 0

                if total_closed_trades > 0:
                    win_rate = profitable_trades / total_closed_trades
                    self.logger.info(f"{eval_name}: Strategy win rate calculation: {profitable_trades} / {total_closed_trades} = {win_rate:.2%}")
                elif not trade_log_df[trade_log_df['type']=='SELL'].empty:
                    win_rate = 0.0
                    self.logger.info(f"{eval_name}: Strategy win rate: 0.0%")
                else:
                    self.logger.info(f"{eval_name}: Strategy win rate: No complete buy-sell cycles.")

            strategy_metrics["Win Rate"] = win_rate
        else:
            self.logger.warning(f"{eval_name}: Strategy returns series is empty or all NaN.")

        # Calculate Buy & Hold metrics 
        self.logger.debug(f"{eval_name}: Calculating Buy & Hold metrics (based on Close prices)...")

        if df_eval is not None and not df_eval.empty and 'Close' in df_eval.columns:
            try:
                # Determine Buy & Hold start date 
                first_eval_date = portfolio_values.index.min() if portfolio_values is not None and not portfolio_values.empty else df_eval.index[TradingConfig.WINDOW_SIZE-1]
                df_bh_calc = df_eval.loc[first_eval_date:]

                if not df_bh_calc.empty:
                    start_price_bh = df_bh_calc['Close'].iloc[0]

                    if start_price_bh > 0:
                        # Calculate Buy & Hold portfolio values 
                        buy_hold_values = initial_balance * (df_bh_calc['Close'] / start_price_bh)
                        buy_hold_metrics["Final Net Worth"] = buy_hold_values.iloc[-1]
                        self.logger.debug(f"{eval_name}: Buy&Hold final net worth: {buy_hold_metrics['Final Net Worth']:.2f}")

                        # Calculate Buy & Hold returns 
                        bh_returns = buy_hold_values.pct_change().dropna()

                        if not bh_returns.empty:
                            bh_annual_factor = 252
                            bh_total_return = (buy_hold_values.iloc[-1] / initial_balance) - 1
                            bh_num_trading_days = len(bh_returns)

                            if bh_num_trading_days > 0:
                                if bh_num_trading_days < bh_annual_factor / 4:
                                    self.logger.warning(f"{eval_name}: Buy&Hold trading days ({bh_num_trading_days}) too few.")

                                # Annualized return 
                                bh_annual_return = ((1 + bh_total_return) ** (bh_annual_factor / bh_num_trading_days)) - 1
                                buy_hold_metrics["Annualized Return"] = bh_annual_return
                                self.logger.debug(f"{eval_name}: Buy&Hold annualized return: {bh_annual_return:.2%}")

                            # Annualized volatility 
                            bh_annual_volatility = bh_returns.std() * np.sqrt(bh_annual_factor)
                            buy_hold_metrics["Annualized Volatility"] = 0.0 if np.isnan(bh_annual_volatility) else bh_annual_volatility
                            self.logger.debug(f"{eval_name}: Buy&Hold annualized volatility: {buy_hold_metrics['Annualized Volatility']:.2%}")

                            # Sharpe ratio 
                            bh_mean_daily_return = bh_returns.mean()
                            if bh_annual_volatility != 0 and not np.isnan(bh_annual_volatility):
                                bh_sharpe_ratio = (bh_mean_daily_return * bh_annual_factor - risk_free_rate) / bh_annual_volatility
                                buy_hold_metrics["Sharpe Ratio"] = bh_sharpe_ratio
                                self.logger.debug(f"{eval_name}: Buy&Hold Sharpe ratio: {bh_sharpe_ratio:.3f}")
                            else:
                                buy_hold_metrics["Sharpe Ratio"] = 0.0

                            # Maximum drawdown 
                            bh_roll_max = buy_hold_values.cummax()
                            bh_daily_drawdown = buy_hold_values / bh_roll_max - 1.0
                            bh_max_drawdown = bh_daily_drawdown.min()
                            buy_hold_metrics["Maximum Drawdown"] = 0.0 if np.isnan(bh_max_drawdown) else bh_max_drawdown
                            self.logger.debug(f"{eval_name}: Buy&Hold maximum drawdown: {bh_max_drawdown:.2%}")

                            # Conditional VaR 
                            bh_cvar_95 = self.calculate_cvar(bh_returns, alpha=0.05)
                            buy_hold_metrics["Conditional VaR (95%)"] = bh_cvar_95
                            self.logger.debug(f"{eval_name}: Buy&Hold CVaR (95%): {bh_cvar_95:.2%}")
                        else:
                            self.logger.warning(f"{eval_name}: Buy&Hold returns series is empty.")
                    else:
                        self.logger.warning(f"{eval_name}: Cannot calculate Buy&Hold metrics, invalid starting price (<=0).")
                else:
                    self.logger.warning(f"{eval_name}: Cannot calculate Buy&Hold metrics, calculation DataFrame is empty.")
            except IndexError:
                self.logger.error(f"{eval_name}: IndexError occurred while calculating Buy&Hold metrics.")
            except Exception as e:
                self.logger.exception(f"{eval_name}: Unknown error calculating Buy&Hold metrics: {e}")
        else:
            self.logger.warning(f"{eval_name}: Missing data required for Buy&Hold metrics calculation (df_eval or Close column).")

        self.logger.info(f"{eval_name}: Performance metrics calculation completed.")
        return strategy_metrics, buy_hold_metrics


# ============================================================================
# Visualization and Reporting 
# ============================================================================

class TradingVisualizer:
    """
    Creates trading visualizations and charts
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def plot_trading_results(self, df_plot: pd.DataFrame, portfolio_values: pd.Series,
                           trade_log_df: pd.DataFrame, save_path: str, chart_title: str = None) -> None:
        """
        Create interactive trading results chart using Plotly
        
        Args:
            chart_title: Custom chart title 
        """
        self.logger.info(f"Creating trading results chart, saving to: {save_path}")

        # Check data availability 
        if (df_plot is None or df_plot.empty) and (portfolio_values is None or portfolio_values.empty):
            self.logger.error("Both df_plot and portfolio_values are empty.")
            return

        if portfolio_values is None or portfolio_values.empty:
            self.logger.warning("Portfolio values series is empty, chart will be incomplete.")

        # Create subplots 
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.1,
            subplot_titles=(
                chart_title or f'{TradingConfig.STOCK_TICKER} Trading Strategy Backtest Results',
                'Portfolio Net Worth'
            )
        )

        # Add price line 
        if df_plot is not None and not df_plot.empty and 'Close' in df_plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_plot.index,
                    y=df_plot['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='black', width=1)
                ),
                row=1, col=1
            )
            self.logger.debug("Added Close price line to chart.")
        else:
            self.logger.warning("Cannot plot Close price line.")

        # Add trading points 
        if trade_log_df is not None and not trade_log_df.empty:
            try:
                trade_log_df['date'] = pd.to_datetime(trade_log_df['date'])

                # Buy points 
                buy_trades = trade_log_df[trade_log_df['type'] == 'BUY']
                if not buy_trades.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_trades['date'],
                            y=buy_trades['price'],
                            mode='markers',
                            marker=dict(color='green', size=10, symbol='triangle-up'),
                            name='Buy',
                            hoverinfo='text',
                            hovertext=[
                                f"Buy {s:,.0f} shares @ {p:,.2f}<br>Cost: {abs(c):,.2f}"
                                for s, p, c in zip(buy_trades['shares'], buy_trades['price'], buy_trades['cost/proceeds'])
                            ]
                        ),
                        row=1, col=1
                    )
                    self.logger.debug(f"Added {len(buy_trades)} buy points.")

                # Sell points 
                sell_trades = trade_log_df[trade_log_df['type'] == 'SELL']
                if not sell_trades.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_trades['date'],
                            y=sell_trades['price'],
                            mode='markers',
                            marker=dict(color='red', size=10, symbol='triangle-down'),
                            name='Sell',
                            hoverinfo='text',
                            hovertext=[
                                f"Sell {s:,.0f} shares @ {p:,.2f}<br>Proceeds: {c:,.2f}"
                                for s, p, c in zip(sell_trades['shares'], sell_trades['price'], sell_trades['cost/proceeds'])
                            ]
                        ),
                        row=1, col=1
                    )
                    self.logger.debug(f"Added {len(sell_trades)} sell points.")

            except Exception as e:
                self.logger.exception(f"Error plotting trading points: {e}")
        else:
            self.logger.info("Trade log is empty, no trading points on chart.")

        # Add strategy net worth curve 
        if portfolio_values is not None and not portfolio_values.empty:
            fig.add_trace(
                go.Scatter(
                    x=portfolio_values.index,
                    y=portfolio_values,
                    mode='lines',
                    name='AI Strategy Net Worth',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
            self.logger.debug("Added strategy net worth curve.")
        else:
            self.logger.warning("Portfolio values series is empty.")

        # Calculate and add Buy & Hold baseline 
        self.logger.debug("Calculating Buy & Hold baseline (based on Close prices)...")
        buy_hold_values = None

        if df_plot is not None and not df_plot.empty and 'Close' in df_plot.columns:
            # Ensure datetime index 
            if not isinstance(df_plot.index, pd.DatetimeIndex):
                try:
                    df_plot.index = pd.to_datetime(df_plot.index)
                    self.logger.debug("Converted df_plot index to DatetimeIndex.")
                except Exception as e:
                    self.logger.exception(f"Failed to convert df_plot index to DatetimeIndex: {e}")
                    df_plot = None

            if df_plot is not None:
                try:
                    # Determine Buy & Hold start date 
                    if portfolio_values is not None and not portfolio_values.empty:
                        first_eval_date = portfolio_values.index.min()
                        df_bh_calc = df_plot.loc[first_eval_date:]
                    else:
                        first_eval_date = df_plot.index[TradingConfig.WINDOW_SIZE - 1] if len(df_plot) >= TradingConfig.WINDOW_SIZE else df_plot.index[0]
                        df_bh_calc = df_plot.loc[first_eval_date:]

                    if not df_bh_calc.empty:
                        start_price_bh = df_bh_calc['Close'].iloc[0]

                        if start_price_bh > 0:
                            # Calculate Buy & Hold values 
                            buy_hold_values = TradingConfig.INITIAL_BALANCE * (df_bh_calc['Close'] / start_price_bh)
                            self.logger.debug(f"Buy & Hold calculation completed. Final value: {buy_hold_values.iloc[-1]:.2f}")

                            # Add to chart 
                            fig.add_trace(
                                go.Scatter(
                                    x=buy_hold_values.index,
                                    y=buy_hold_values,
                                    mode='lines',
                                    name='Buy & Hold',
                                    line=dict(color='grey', dash='dash')
                                ),
                                row=2, col=1
                            )
                            self.logger.debug("Added Buy & Hold baseline.")
                        else:
                            self.logger.warning("Cannot calculate Buy & Hold, invalid starting price (<=0).")
                    else:
                        self.logger.warning("Cannot calculate Buy & Hold, calculation DataFrame is empty.")
                except IndexError:
                    self.logger.error("IndexError occurred while calculating Buy & Hold for visualization.")
                except Exception as e:
                    self.logger.exception(f"Unknown error calculating Buy & Hold for visualization: {e}")
        else:
            self.logger.warning("Missing data required for Buy & Hold visualization (df_plot or Close column).")

        # Set chart titles and layout 
        plot_start_date = df_plot.index.min().strftime("%Y-%m-%d") if df_plot is not None and not df_plot.empty else "N/A"
        plot_end_date = df_plot.index.max().strftime("%Y-%m-%d") if df_plot is not None and not df_plot.empty else "N/A"

        fig.update_layout(
            title=chart_title or f'{TradingConfig.STOCK_TICKER} Quantitative Trading Strategy Backtest Results ({plot_start_date} - {plot_end_date})',
            xaxis_title='Date',
            yaxis_title='Closing Price (USD)',
            yaxis2_title='Net Worth (USD)',
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            legend_title_text='Legend'
        )

        fig.update_yaxes(tickformat=',.0f', row=1, col=1)
        fig.update_yaxes(tickformat=',.0f', row=2, col=1)

        # Save chart 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            fig.write_html(save_path)
            if os.path.exists(save_path):
                self.logger.info(f"Interactive chart successfully saved to {save_path}")
            else:
                self.logger.error(f"Chart allegedly saved but file not found at {save_path}!")
        except Exception as e:
            self.logger.exception(f"Error saving chart: {e}")

        self.logger.info("Trading visualization completed.")


class ReportGenerator:
    """
    Generates comprehensive trading reports
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def generate_comprehensive_report(self,
                                    val_strategy_metrics: Dict[str, float],
                                    val_buyhold_metrics: Dict[str, float],
                                    test_strategy_metrics: Dict[str, float],
                                    test_buyhold_metrics: Dict[str, float],
                                    save_path: str,
                                    ticker: str = TradingConfig.STOCK_TICKER,
                                    train_period: Tuple[str, str] = None,
                                    val_period: Tuple[str, str] = None,
                                    test_period: Tuple[str, str] = None,
                                    model_path: str = None,
                                    val_chart_path: str = None,
                                    test_chart_path: str = None,
                                    val_trades_path: str = None,
                                    test_trades_path: str = None) -> None:
        """
        Generate comprehensive backtest report with train-validation-test results
        """
        self.logger.info(f"Generating comprehensive backtest report, saving to: {save_path}")

        if not isinstance(val_strategy_metrics, dict):
            self.logger.warning("val_strategy_metrics is not a dictionary.")
            val_strategy_metrics = {}

        if not isinstance(val_buyhold_metrics, dict):
            self.logger.warning("val_buyhold_metrics is not a dictionary.")
            val_buyhold_metrics = {}

        if not isinstance(test_strategy_metrics, dict):
            self.logger.warning("test_strategy_metrics is not a dictionary.")
            test_strategy_metrics = {}

        if not isinstance(test_buyhold_metrics, dict):
            self.logger.warning("test_buyhold_metrics is not a dictionary.")
            test_buyhold_metrics = {}

        # Helper function for formatting metric values 
        def format_metric(value, format_str):
            if value is None or (isinstance(value, (float, np.number)) and np.isnan(value)):
                return "N/A"
            try:
                return format(value, format_str)
            except (TypeError, ValueError):
                self.logger.warning(f"Failed to format metric value '{value}' with format '{format_str}'.")
                return str(value)

        # Get file names 
        model_filename = os.path.basename(model_path) if model_path else "N/A"
        val_chart_filename = os.path.basename(val_chart_path) if val_chart_path else "N/A"
        test_chart_filename = os.path.basename(test_chart_path) if test_chart_path else "N/A"
        val_trades_filename = os.path.basename(val_trades_path) if val_trades_path else "N/A"
        test_trades_filename = os.path.basename(test_trades_path) if test_trades_path else "N/A"

        # Set default periods if not provided 
        if train_period is None:
            train_period = (TradingConfig.TRAIN_START_DATE, TradingConfig.TRAIN_END_DATE)
        if val_period is None:
            val_period = (TradingConfig.VALIDATION_START_DATE, TradingConfig.VALIDATION_END_DATE)
        if test_period is None:
            test_period = (TradingConfig.TEST_START_DATE, TradingConfig.TEST_END_DATE)

        # Generate report content 
        report_content = f"""
# Quantitative Trading Strategy Backtest Report

**Stock Ticker:** {ticker}
**Model:** Reinforcement Learning ({TradingConfig.RL_ALGORITHM.__name__}) + Multi-Factor Analysis (with pre-calculated factors)
**Model File:** `{model_filename}`

---

## Data Periods 
* **Training Data:** {train_period[0]} to {train_period[1]}
* **Validation Data:** {val_period[0]} to {val_period[1]}
* **Test Data:** {test_period[0]} to {test_period[1]}

---

## Three-Stage Training Approach 

This framework implements the industry-standard **Train-Validation-Test** split:

1. **Training Stage**: Model learns trading patterns from historical data 
2. **Validation Stage**: Hyperparameter optimization using Optuna 
3. **Test Stage**: Final unbiased performance evaluation 

**Key Advantage**: The test set remains completely unseen during training and optimization, providing reliable real-world performance estimates.

---

## Validation Set Performance 

**Period:** {val_period[0]} to {val_period[1]}

### Performance Metrics Comparison 

| Metric | Strategy (RL + Multi-Factor) | Buy & Hold |
|--------|------------------------------|------------|
| Annualized Return | {format_metric(val_strategy_metrics.get('Annualized Return'), '.2%')} | {format_metric(val_buyhold_metrics.get('Annualized Return'), '.2%')} |
| Sharpe Ratio | {format_metric(val_strategy_metrics.get('Sharpe Ratio'), '.3f')} | {format_metric(val_buyhold_metrics.get('Sharpe Ratio'), '.3f')} |
| Maximum Drawdown | {format_metric(val_strategy_metrics.get('Maximum Drawdown'), '.2%')} | {format_metric(val_buyhold_metrics.get('Maximum Drawdown'), '.2%')} |
| Conditional VaR (95%) | {format_metric(val_strategy_metrics.get('Conditional VaR (95%)'), '.2%')} | {format_metric(val_buyhold_metrics.get('Conditional VaR (95%)'), '.2%')} |
| Annualized Volatility | {format_metric(val_strategy_metrics.get('Annualized Volatility'), '.2%')} | {format_metric(val_buyhold_metrics.get('Annualized Volatility'), '.2%')} |
| Win Rate | {format_metric(val_strategy_metrics.get('Win Rate'), '.2%')} | N/A |
| Total Trades | {format_metric(val_strategy_metrics.get('Total Trades'), ',')} | {format_metric(val_buyhold_metrics.get('Total Trades'), ',')} |
| Final Portfolio Net Worth | {format_metric(val_strategy_metrics.get('Final Net Worth'), ',.2f')} USD | {format_metric(val_buyhold_metrics.get('Final Net Worth'), ',.2f')} USD |
| Initial Capital | {format_metric(TradingConfig.INITIAL_BALANCE, ',.2f')} USD | {format_metric(TradingConfig.INITIAL_BALANCE, ',.2f')} USD |

**Validation Set Purpose**: Hyperparameter optimization and model selection 

**Trade Records**: See `{val_trades_filename}` (strategy only)
**Interactive Chart**: See `{val_chart_filename}` (open with browser, includes Buy&Hold comparison)

---

## Test Set Performance 

**Period:** {test_period[0]} to {test_period[1]}

### Performance Metrics Comparison 

| Metric | Strategy (RL + Multi-Factor) | Buy & Hold |
|--------|------------------------------|------------|
| Annualized Return | {format_metric(test_strategy_metrics.get('Annualized Return'), '.2%')} | {format_metric(test_buyhold_metrics.get('Annualized Return'), '.2%')} |
| Sharpe Ratio | {format_metric(test_strategy_metrics.get('Sharpe Ratio'), '.3f')} | {format_metric(test_buyhold_metrics.get('Sharpe Ratio'), '.3f')} |
| Maximum Drawdown | {format_metric(test_strategy_metrics.get('Maximum Drawdown'), '.2%')} | {format_metric(test_buyhold_metrics.get('Maximum Drawdown'), '.2%')} |
| Conditional VaR (95%) | {format_metric(test_strategy_metrics.get('Conditional VaR (95%)'), '.2%')} | {format_metric(test_buyhold_metrics.get('Conditional VaR (95%)'), '.2%')} |
| Annualized Volatility | {format_metric(test_strategy_metrics.get('Annualized Volatility'), '.2%')} | {format_metric(test_buyhold_metrics.get('Annualized Volatility'), '.2%')} |
| Win Rate | {format_metric(test_strategy_metrics.get('Win Rate'), '.2%')} | N/A |
| Total Trades | {format_metric(test_strategy_metrics.get('Total Trades'), ',')} | {format_metric(test_buyhold_metrics.get('Total Trades'), ',')} |
| Final Portfolio Net Worth | {format_metric(test_strategy_metrics.get('Final Net Worth'), ',.2f')} USD | {format_metric(test_buyhold_metrics.get('Final Net Worth'), ',.2f')} USD |
| Initial Capital | {format_metric(TradingConfig.INITIAL_BALANCE, ',.2f')} USD | {format_metric(TradingConfig.INITIAL_BALANCE, ',.2f')} USD |

**Test Set Purpose**: Final unbiased performance evaluation 

**Trade Records**: See `{test_trades_filename}` (strategy only)
**Interactive Chart**: See `{test_chart_filename}` (open with browser, includes Buy&Hold comparison)

---

## Strategy Description 

This strategy uses a reinforcement learning agent ({TradingConfig.RL_ALGORITHM.__name__}) to make trading decisions (buy/sell/hold) based on:
- Historical stock prices (OHLCV) 
- Technical indicators (SMA, EMA, RSI, MACD, etc.) 
- **Pre-calculated** fundamental, news sentiment, industry trend, and market risk factors 

Trading and evaluation are primarily based on **closing prices ('Close')**. The report provides performance comparison with **Buy & Hold** baseline strategy.

**Important Note**: Factor data quality and predictive capability directly impact strategy performance.

---

## Risk Management Mechanisms 

* **Stop Loss:** {TradingConfig.STOP_LOSS_PCT:.0%} (force sell when price drops from entry 
* **Take Profit:** {TradingConfig.TAKE_PROFIT_PCT:.0%} (force sell when price rises from entry 
* **Dynamic Position Management:** Based on ATR ({TradingConfig.ATR_PERIOD_FOR_SIZING} days) and risk ratio ({TradingConfig.MAX_POSITION_RISK_PCT:.2%}) to calculate maximum position size 
* **Reward Shaping:** Includes volatility penalty and profit reward mechanisms 

---

## Performance Analysis 

### Validation vs Test Performance 

The difference between validation and test performance is normal and expected:
- **Validation performance** reflects hyperparameter optimization target 
- **Test performance** represents true real-world capability 

If test performance is significantly lower than validation, this suggests some overfitting to validation data. If they are similar, the model generalizes well.

---

## Future Optimization Directions 

* **Factor Quality Enhancement:** Continuously improve calculation and analysis methods for fundamental, sentiment, trend factors to explore more effective Alpha factors 
* **LLM/MAS Integration:** Explore using LLM to automate factor generation or build Multi-Agent Systems (MAS) for collaborative analysis and decision-making 
* **Feature Engineering:** Explore more or different technical indicators and factor combinations 
* **RL Environment Optimization:** Adjust reward functions and state representations 
* **Hyperparameter Tuning:** Expand Optuna trial count to try more hyperparameter combinations 
* **Risk Management:** Implement more dynamic, multi-dimensional risk controls 
* **Model Ensemble:** Try ensemble multiple models 
* **Data Source Expansion:** Introduce more diverse data sources 

---

## Additional Resources 

* **Detailed Execution Log:** See `{TradingConfig.LOG_FILE_NAME}`
* **Architecture Documentation:** See `ARCHITECTURE_REFACTORING_GUIDE.md`
* **User Guide:** See `README_THREE_STAGE.md`

---

*Report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Framework Version: 3.0 (Three-Stage Training)*
"""

        # Save report
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.logger.info(f"Backtest report saved to {save_path}")
        except Exception as e:
            self.logger.exception(f"Error saving report: {e}")

        self.logger.info("Report generation completed.")


# ============================================================================
# Hyperparameter Optimization 
# ============================================================================

class OptunaOptimizer:
    """
    Handles hyperparameter optimization using Optuna
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def objective(self, trial, train_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
        """
        Optuna objective function for finding optimal hyperparameters using validation set
        
        CRITICAL: Uses validation set for optimization, NOT test set
        """
        self.logger.info(f"--- Optuna Trial {trial.number} ---")

        # Suggest hyperparameters 
        hyperparams = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
            'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99),
            'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
        }

        try:
            # Create environment with TRAINING data 
            trial_train_env = StockTradingEnvironment(
                train_df.copy(),
                window_size=TradingConfig.WINDOW_SIZE,
                initial_balance=TradingConfig.INITIAL_BALANCE,
                main_logger=self.logger
            )
            trial_train_env = DummyVecEnv([lambda: trial_train_env])

            # Train model (using fewer timesteps) 
            trainer = ModelTrainer(self.logger)
            model = trainer.train_model(
                trial_train_env,
                hyperparameters=hyperparams,
                log_dir=os.path.join(TradingConfig.get_sb3_log_dir(), f"trial_{trial.number}"),
                total_timesteps=TradingConfig.TRIAL_TRAINING_TIMESTEPS,
                save_path=None  # Don't save trial models 
            )

            if model is None:
                self.logger.error(f"Trial {trial.number}: Model training failed.")
                return -np.inf

            # Evaluate model on VALIDATION set (NOT test set) 
            evaluator = BacktestEvaluator(self.logger)
            trade_log_df, portfolio_values = evaluator.evaluate_model(
                val_df.copy(),  # CRITICAL: Using validation set 
                model=model,
                initial_balance=TradingConfig.INITIAL_BALANCE,
                window_size=TradingConfig.WINDOW_SIZE,
                save_trades_path=None,  # Don't save trial trades 
                eval_name=f"Trial_{trial.number}_Validation"
            )

            if portfolio_values is None or portfolio_values.empty:
                self.logger.error(f"Trial {trial.number}: Model evaluation failed or produced no results.")
                return -np.inf

            # Calculate target metric (annualized return on validation set) 
            analyzer = PerformanceAnalyzer(self.logger)
            strategy_metrics, _ = analyzer.calculate_metrics(
                portfolio_values,
                val_df,
                trade_log_df,
                initial_balance=TradingConfig.INITIAL_BALANCE,
                eval_name=f"Trial_{trial.number}_Validation"
            )

            annual_return = strategy_metrics.get("Annualized Return", np.nan)

            if np.isnan(annual_return):
                self.logger.error(f"Trial {trial.number}: Cannot calculate annualized return.")
                return -np.inf

            self.logger.info(f"Trial {trial.number}: Validation annualized return = {annual_return:.2%}")
            return annual_return  # Optuna maximizes this value 

        except Exception as e:
            self.logger.exception(f"Trial {trial.number}: Unexpected error during execution: {e}")
            raise optuna.exceptions.TrialPruned(f"Trial failed due to exception: {e}")


# ============================================================================
# Main Execution Pipeline 
# ============================================================================

def main():
    """
    Main execution pipeline integrating all functionality with three-stage training
    """
    # Setup logger 
    logger = setup_logger()

    logger.info("=" * 80)
    logger.info("Starting main trading system pipeline (Three-Stage Training)...")
    logger.info(f"Timestamp: {datetime.datetime.now()}")
    logger.info(f"Stock ticker: {TradingConfig.STOCK_TICKER}")
    logger.info(f"Training period: {TradingConfig.TRAIN_START_DATE} - {TradingConfig.TRAIN_END_DATE}")
    logger.info(f"Validation period: {TradingConfig.VALIDATION_START_DATE} - {TradingConfig.VALIDATION_END_DATE}")
    logger.info(f"Testing period: {TradingConfig.TEST_START_DATE} - {TradingConfig.TEST_END_DATE}")
    logger.info(f"Initial capital: {TradingConfig.INITIAL_BALANCE:,.0f}")
    logger.info(f"Log level: {logging.getLevelName(TradingConfig.LOGGING_LEVEL)}")
    logger.info("=" * 80)

    # Create necessary directories 
    os.makedirs(TradingConfig.DATA_DIR, exist_ok=True)
    os.makedirs(TradingConfig.get_sb3_log_dir(), exist_ok=True)

    # --- Step 1: Load OHLCV and Factor Data ---
    logger.info("=" * 80)
    logger.info("--- Step 1: Load OHLCV and Factor Data ---")
    logger.info("=" * 80)

    data_manager = DataManager(logger)

    # Load or download OHLCV data 
    ohlcv_df = data_manager.load_raw_stock_data(
        TradingConfig.STOCK_TICKER,
        TradingConfig.TRAIN_START_DATE,
        TradingConfig.TEST_END_DATE,
        TradingConfig.get_raw_data_path()
    )

    if ohlcv_df is None:
        logger.critical("Unable to obtain valid OHLCV data, terminating program.")
        return

    logger.info(f"OHLCV data loaded/downloaded successfully: {len(ohlcv_df)} records.")

    # Load factor data 
    factor_df = data_manager.load_factor_data(
        TradingConfig.get_factor_data_path(),
        TradingConfig.EXPECTED_FACTOR_COLUMNS
    )

    if factor_df is None:
        logger.critical("Unable to load valid factor data, terminating program.")
        return

    logger.info(f"Factor data loaded successfully: {len(factor_df)} records.")

    # --- Step 2: Merge Data, Calculate Technical Indicators, and Save Final Data ---
    logger.info("=" * 80)
    logger.info("--- Step 2: Merge Data, Calculate Technical Indicators, and Save Final Data ---")
    logger.info("=" * 80)

    final_processed_df = data_manager.merge_and_process_data(
        ohlcv_df,
        factor_df,
        TradingConfig.get_processed_data_path()
    )

    if final_processed_df is None or final_processed_df.empty:
        logger.critical("Data merging or processing failed, terminating program.")
        return

    logger.info(f"Final data processing completed: {len(final_processed_df)} valid records.")

    # --- Step 3: Split Data into Train/Validation/Test ---
    logger.info("=" * 80)
    logger.info("--- Step 3: Split Data into Train/Validation/Test ---")
    logger.info("=" * 80)

    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()

    try:
        train_df = final_processed_df.loc[TradingConfig.TRAIN_START_DATE:TradingConfig.TRAIN_END_DATE].copy()
        val_df = final_processed_df.loc[TradingConfig.VALIDATION_START_DATE:TradingConfig.VALIDATION_END_DATE].copy()
        test_df = final_processed_df.loc[TradingConfig.TEST_START_DATE:TradingConfig.TEST_END_DATE].copy()

        logger.info(f"Training data: {len(train_df)} records, from {train_df.index.min().strftime('%Y-%m-%d')} to {train_df.index.max().strftime('%Y-%m-%d')}")
        logger.info(f"Validation data: {len(val_df)} records, from {val_df.index.min().strftime('%Y-%m-%d')} to {val_df.index.max().strftime('%Y-%m-%d')}")
        logger.info(f"Testing data: {len(test_df)} records, from {test_df.index.min().strftime('%Y-%m-%d')} to {test_df.index.max().strftime('%Y-%m-%d')}")

    except KeyError as e:
        logger.critical(f"Error splitting data (KeyError): {e}")
        logger.critical(f"Final data index range: {final_processed_df.index.min()} to {final_processed_df.index.max()}")
        return
    except Exception as e:
        logger.exception(f"Unknown error splitting data: {e}")
        return

    if train_df.empty or val_df.empty or test_df.empty:
        logger.critical("Training, validation, or testing dataset is empty, terminating program.")
        return

    # Validate data continuity 
    if train_df.index.max() >= val_df.index.min():
        logger.warning(f"Training data end date ({train_df.index.max().strftime('%Y-%m-%d')}) "
                      f"overlaps or is too close to validation data start date ({val_df.index.min().strftime('%Y-%m-%d')}).")

    if val_df.index.max() >= test_df.index.min():
        logger.warning(f"Validation data end date ({val_df.index.max().strftime('%Y-%m-%d')}) "
                      f"overlaps or is too close to test data start date ({test_df.index.min().strftime('%Y-%m-%d')}).")

    # --- Step 4: Hyperparameter Optimization on Validation Set with Optuna ---
    logger.info("=" * 80)
    logger.info("--- Step 4: Hyperparameter Optimization on Validation Set with Optuna ---")
    logger.info("CRITICAL: Optimization uses VALIDATION set, NOT test set")
    logger.info("=" * 80)

    optimizer = OptunaOptimizer(logger)

    study = optuna.create_study(
        study_name=f"ppo_optimization_{TradingConfig.STOCK_TICKER}",
        direction='maximize',
        storage=f"sqlite:///{TradingConfig.get_optuna_db_path()}",
        load_if_exists=True
    )

    objective_with_data = partial(optimizer.objective, train_df=train_df, val_df=val_df)

    try:
        logger.info(f"Starting Optuna optimization with {TradingConfig.N_OPTUNA_TRIALS} trials...")
        study.optimize(objective_with_data, n_trials=TradingConfig.N_OPTUNA_TRIALS, timeout=3600*3)  # 3 hour timeout
        logger.info("Optuna optimization completed.")
    except Exception as e:
        logger.exception(f"Error during Optuna optimization: {e}")

    # Get best parameters 
    best_params = None
    try:
        best_params = study.best_params
        best_value = study.best_value
        logger.info(f"Best validation annualized return found: {best_value:.2%}")
        logger.info(f"Corresponding best hyperparameters: {best_params}")
    except ValueError:
        logger.warning("Optuna failed to find valid best trial, using default hyperparameters for final training.")
        best_params = {
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2
        }

    # --- Step 5: Train Final Model with Best Parameters on Training Set ---
    logger.info("=" * 80)
    logger.info("--- Step 5: Train Final Model with Best Parameters on Training Set ---")
    logger.info("=" * 80)

    final_model = None
    trainer = ModelTrainer(logger)

    try:
        final_train_env = StockTradingEnvironment(
            train_df,
            window_size=TradingConfig.WINDOW_SIZE,
            initial_balance=TradingConfig.INITIAL_BALANCE,
            main_logger=logger
        )
        final_train_env = DummyVecEnv([lambda: final_train_env])
        logger.info("Final training environment created successfully.")

        final_model = trainer.train_model(
            final_train_env,
            hyperparameters=best_params,
            save_path=TradingConfig.get_model_path(),
            log_dir=os.path.join(TradingConfig.get_sb3_log_dir(), "final_best"),
            total_timesteps=TradingConfig.TOTAL_TRAINING_TIMESTEPS
        )

        if final_model is not None:
            logger.info("Final model training successful.")
        else:
            logger.error("Final model training failed.")

    except Exception as e:
        logger.exception(f"Critical error training final model: {e}")

    # --- Step 6: Evaluate on Validation Set ---
    logger.info("=" * 80)
    logger.info("--- Step 6: Evaluate on Validation Set ---")
    logger.info("=" * 80)

    val_trade_log_df = pd.DataFrame()
    val_portfolio_values = pd.Series(dtype=np.float64)

    evaluator = BacktestEvaluator(logger)

    if final_model is not None and os.path.exists(TradingConfig.get_model_path()):
        val_trade_log_df, val_portfolio_values = evaluator.evaluate_model(
            val_df.copy(),
            model_path=TradingConfig.get_model_path(),
            initial_balance=TradingConfig.INITIAL_BALANCE,
            window_size=TradingConfig.WINDOW_SIZE,
            save_trades_path=TradingConfig.get_trades_path("validation"),
            eval_name="Validation"
        )
    else:
        logger.warning("Final model training failed or not saved, skipping validation evaluation.")

    # --- Step 7: Calculate Validation Performance Metrics ---
    logger.info("=" * 80)
    logger.info("--- Step 7: Calculate Validation Performance Metrics ---")
    logger.info("=" * 80)

    val_strategy_metrics = {}
    val_buyhold_metrics = {}

    analyzer = PerformanceAnalyzer(logger)

    if val_portfolio_values is not None and not val_portfolio_values.empty:
        val_strategy_metrics, val_buyhold_metrics = analyzer.calculate_metrics(
            val_portfolio_values,
            val_df,
            val_trade_log_df,
            initial_balance=TradingConfig.INITIAL_BALANCE,
            eval_name="Validation"
        )

        # Output validation metrics 
        logger.info("Validation performance metrics (Strategy):")
        for k, v in val_strategy_metrics.items():
            formatted_value = "N/A"
            if isinstance(v, (float, np.number)) and not np.isnan(v):
                if any(x in k for x in ['Rate', 'Return', 'Drawdown', '%']):
                    formatted_value = format(v, '.2%')
                elif 'Ratio' in k:
                    formatted_value = format(v, '.3f')
                else:
                    formatted_value = format(v, ',.2f')
            elif isinstance(v, (float, np.number)) and np.isnan(v):
                formatted_value = 'N/A'
            else:
                formatted_value = str(v)

            logger.info(f"{k:<35}: {formatted_value}")

        logger.info("Validation performance metrics (Buy & Hold):")
        for k, v in val_buyhold_metrics.items():
            formatted_value = "N/A"
            if isinstance(v, (float, np.number)) and not np.isnan(v):
                if any(x in k for x in ['Rate', 'Return', 'Drawdown', '%']):
                    formatted_value = format(v, '.2%')
                elif 'Ratio' in k:
                    formatted_value = format(v, '.3f')
                else:
                    formatted_value = format(v, ',.2f')
            elif isinstance(v, (float, np.number)) and np.isnan(v):
                formatted_value = 'N/A'
            else:
                formatted_value = str(v)

            logger.info(f"{k:<35}: {formatted_value}")
    else:
        logger.warning("Validation evaluation failed or produced no portfolio values, cannot calculate metrics.")
        val_strategy_metrics["Total Trades"] = len(val_trade_log_df) if val_trade_log_df is not None else 0
        val_strategy_metrics["Final Net Worth"] = TradingConfig.INITIAL_BALANCE
        val_buyhold_metrics["Final Net Worth"] = TradingConfig.INITIAL_BALANCE

    # --- Step 8: Create Validation Visualization (Optional) ---
    logger.info("=" * 80)
    logger.info("--- Step 8: Create Validation Visualization (Optional) ---")
    logger.info("=" * 80)

    visualizer = TradingVisualizer(logger)

    if val_portfolio_values is not None and not val_portfolio_values.empty:
        val_plot_df = val_df.copy()
        visualizer.plot_trading_results(
            val_plot_df,
            val_portfolio_values,
            val_trade_log_df,
            TradingConfig.get_chart_path("validation"),
            chart_title=f'{TradingConfig.STOCK_TICKER} Validation Set Trading Results'
        )
    else:
        logger.warning("Validation evaluation failed or produced no portfolio values, cannot create visualization.")

    # --- Step 9: Evaluate Final Model on Test Set ---
    logger.info("=" * 80)
    logger.info("--- Step 9: Evaluate Final Model on Test Set ---")
    logger.info("CRITICAL: This is the FINAL evaluation on completely unseen data")
    logger.info("=" * 80)

    test_trade_log_df = pd.DataFrame()
    test_portfolio_values = pd.Series(dtype=np.float64)

    if final_model is not None and os.path.exists(TradingConfig.get_model_path()):
        test_trade_log_df, test_portfolio_values = evaluator.evaluate_model(
            test_df.copy(),
            model_path=TradingConfig.get_model_path(),
            initial_balance=TradingConfig.INITIAL_BALANCE,
            window_size=TradingConfig.WINDOW_SIZE,
            save_trades_path=TradingConfig.get_trades_path("test"),
            eval_name="Test"
        )
    else:
        logger.warning("Final model training failed or not saved, skipping test evaluation.")

    # --- Step 10: Calculate Test Performance Metrics ---
    logger.info("=" * 80)
    logger.info("--- Step 10: Calculate Test Performance Metrics ---")
    logger.info("=" * 80)

    test_strategy_metrics = {}
    test_buyhold_metrics = {}

    if test_portfolio_values is not None and not test_portfolio_values.empty:
        test_strategy_metrics, test_buyhold_metrics = analyzer.calculate_metrics(
            test_portfolio_values,
            test_df,
            test_trade_log_df,
            initial_balance=TradingConfig.INITIAL_BALANCE,
            eval_name="Test"
        )

        # Output test metrics 
        logger.info("Test performance metrics (Strategy):")
        for k, v in test_strategy_metrics.items():
            formatted_value = "N/A"
            if isinstance(v, (float, np.number)) and not np.isnan(v):
                if any(x in k for x in ['Rate', 'Return', 'Drawdown', '%']):
                    formatted_value = format(v, '.2%')
                elif 'Ratio' in k:
                    formatted_value = format(v, '.3f')
                else:
                    formatted_value = format(v, ',.2f')
            elif isinstance(v, (float, np.number)) and np.isnan(v):
                formatted_value = 'N/A'
            else:
                formatted_value = str(v)

            logger.info(f"{k:<35}: {formatted_value}")

        logger.info("Test performance metrics (Buy & Hold):")
        for k, v in test_buyhold_metrics.items():
            formatted_value = "N/A"
            if isinstance(v, (float, np.number)) and not np.isnan(v):
                if any(x in k for x in ['Rate', 'Return', 'Drawdown', '%']):
                    formatted_value = format(v, '.2%')
                elif 'Ratio' in k:
                    formatted_value = format(v, '.3f')
                else:
                    formatted_value = format(v, ',.2f')
            elif isinstance(v, (float, np.number)) and np.isnan(v):
                formatted_value = 'N/A'
            else:
                formatted_value = str(v)

            logger.info(f"{k:<35}: {formatted_value}")
    else:
        logger.warning("Test evaluation failed or produced no portfolio values, cannot calculate metrics.")
        test_strategy_metrics["Total Trades"] = len(test_trade_log_df) if test_trade_log_df is not None else 0
        test_strategy_metrics["Final Net Worth"] = TradingConfig.INITIAL_BALANCE
        test_buyhold_metrics["Final Net Worth"] = TradingConfig.INITIAL_BALANCE

    # --- Step 11: Create Test Visualization ---
    logger.info("=" * 80)
    logger.info("--- Step 11: Create Test Visualization ---")
    logger.info("=" * 80)

    if test_portfolio_values is not None and not test_portfolio_values.empty:
        test_plot_df = test_df.copy()
        visualizer.plot_trading_results(
            test_plot_df,
            test_portfolio_values,
            test_trade_log_df,
            TradingConfig.get_chart_path("test"),
            chart_title=f'{TradingConfig.STOCK_TICKER} Test Set Trading Results'
        )
    else:
        logger.warning("Test evaluation failed or produced no portfolio values, cannot create visualization.")

    # --- Step 12: Generate Comprehensive Report ---
    logger.info("=" * 80)
    logger.info("--- Step 12: Generate Comprehensive Report ---")
    logger.info("=" * 80)

    train_start = train_df.index.min().strftime('%Y-%m-%d') if not train_df.empty else TradingConfig.TRAIN_START_DATE
    train_end = train_df.index.max().strftime('%Y-%m-%d') if not train_df.empty else TradingConfig.TRAIN_END_DATE
    val_start = val_df.index.min().strftime('%Y-%m-%d') if not val_df.empty else TradingConfig.VALIDATION_START_DATE
    val_end = val_df.index.max().strftime('%Y-%m-%d') if not val_df.empty else TradingConfig.VALIDATION_END_DATE
    test_start = test_df.index.min().strftime('%Y-%m-%d') if not test_df.empty else TradingConfig.TEST_START_DATE
    test_end = test_df.index.max().strftime('%Y-%m-%d') if not test_df.empty else TradingConfig.TEST_END_DATE

    model_filename = os.path.basename(TradingConfig.get_model_path()) if os.path.exists(TradingConfig.get_model_path()) else "N/A"

    report_generator = ReportGenerator(logger)
    report_generator.generate_comprehensive_report(
        val_strategy_metrics=val_strategy_metrics,
        val_buyhold_metrics=val_buyhold_metrics,
        test_strategy_metrics=test_strategy_metrics,
        test_buyhold_metrics=test_buyhold_metrics,
        save_path=TradingConfig.get_report_path(),
        train_period=(train_start, train_end),
        val_period=(val_start, val_end),
        test_period=(test_start, test_end),
        model_path=model_filename,
        val_chart_path=TradingConfig.get_chart_path("validation"),
        test_chart_path=TradingConfig.get_chart_path("test"),
        val_trades_path=TradingConfig.get_trades_path("validation"),
        test_trades_path=TradingConfig.get_trades_path("test")
    )

    logger.info("=" * 80)
    logger.info("Quantitative trading strategy development pipeline (Three-Stage Training) completed.")
    logger.info(f"All results saved to '{TradingConfig.DATA_DIR}' directory.")
    logger.info(f"Detailed logs available in: {os.path.join(TradingConfig.DATA_DIR, TradingConfig.LOG_FILE_NAME)}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
