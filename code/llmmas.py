# -*- coding: utf-8 -*- llmmas.py
# -----------------------------------------------------------------------------
# AI 量化交易策略系統開發框架 (改進版)
#
# Author: Edward Cheng
# Date: 2025-05-22 update
#
# 功能:
# 1. 下載/載入 <Stock ID> 原始股價資料 (OHLCV)
# 2. LLM + Multi-Agent 載入預先計算的因子分數 (score_data_ok.csv)，來自 genscore.py 的({STOCK_TICKER}_score_data_ok.csv) 或 genfactor.py 的({STOCK_TICKER}_score_data_optimized.csv) 將檔名自行變更
# 3. 合併 OHLCV 與因子數據
# 4. 特徵工程: 計算技術指標 (基於 'Close')
# 5. 資料分割 (訓練集 / 測試集)
# 6. 使用 Optuna 快速調整 RL 超參數
# 7. 強化學習環境 (Custom Gym Environment) - **加入停利 (50%), 停損 (5%)**
# 8. 強化學習模型訓練 (Stable Baselines 3 - PPO) - 使用優化後的參數
# 9. 模型回測與評估 (基於驗證集)
# 10. 績效指標計算 (包含 Buy&Hold 比較) - 基於 'Close' 價格
# 11. 交易結果視覺化 (Plotly) - 基於 'Close' 價格, 含 Buy&Hold 比較
# 12. 儲存: 資料, 模型, 交易記錄, 報告, 圖表, 日誌
# 13. 可迭代優化的結構
# 14. 增強的日誌記錄和除錯追蹤機制
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 1. 安裝必要的函式庫 (在 Colab 中取消註解並執行)
# -----------------------------------------------------------------------------
# !pip install yfinance pandas numpy matplotlib stable-baselines3[extra] plotly pyfolio gym gymnasium scikit-learn tensorflow optuna
# 安裝 TA-Lib (Colab 特定步驟)
# !wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
# !tar -xzvf ta-lib-0.4.0-src.tar.gz
# %cd ta-lib/
# !./configure --prefix=/usr
# !make
# !make install
# !pip install TA-Lib
# %cd ..

# -----------------------------------------------------------------------------
# 2. 導入函式庫
# -----------------------------------------------------------------------------
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

# 強化學習相關
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as sb3_configure

# 忽略特定警告
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
pd.options.mode.chained_assignment = None  # default='warn'

# -----------------------------------------------------------------------------
# 3. 設定與常數
# -----------------------------------------------------------------------------
class Config:
    """全局配置類，集中管理所有配置參數"""

    # --- 日誌設定 ---
    LOGGING_LEVEL = logging.INFO
    DATA_DIR = "quant_data"
    LOG_FILE_PATH = os.path.join(DATA_DIR, "trading_strategy.log")

    # --- 主要設定 ---
    STOCK_TICKER = "AMZN"     # Yahoo Finance代碼 >> 鴻海："2317.TW", 中信金："2891.TW", 台積電："2330.TW"， APPLE："AAPL"， Amazon："AMZN"， Microsoft："MSFT"
    TRAIN_START_DATE = "2000-01-01"
    TRAIN_END_DATE = "2021-12-31"
    TEST_START_DATE = "2022-01-01"
    TEST_END_DATE = "2023-12-31"    # 使用指定結束日期 # changeto 2023-12-31
    INITIAL_BALANCE = 1_000_000    # 台股初始資金10_000_000(台幣 TWD)，美股初始資金1_000_000(美元 USD)
    WINDOW_SIZE = 55   #60               # RL 環境觀察窗口大小 (過去 N 天的數據)

    # --- 檔案路徑 ---
    RAW_DATA_PATH = os.path.join(DATA_DIR, f"{STOCK_TICKER}_raw_data.csv")
    FACTOR_DATA_PATH = os.path.join(DATA_DIR, "score_data_ok.csv")
    FINAL_PROCESSED_DATA_PATH = os.path.join(DATA_DIR, f"{STOCK_TICKER}_final_processed_data.csv")
    MODEL_SAVE_PATH = os.path.join(DATA_DIR, f"{STOCK_TICKER}_rl_model_best.zip")
    TRADES_SAVE_PATH = os.path.join(DATA_DIR, f"{STOCK_TICKER}_trades_best.csv")
    REPORT_SAVE_PATH = os.path.join(DATA_DIR, f"{STOCK_TICKER}_report_best.md")
    CHART_SAVE_PATH = os.path.join(DATA_DIR, f"{STOCK_TICKER}_trading_chart_best.html")
    SB3_LOG_DIR = os.path.join(DATA_DIR, "sb3_logs")
    OPTUNA_DB_PATH = os.path.join(DATA_DIR, "optuna_study.db")

    # --- 預期因子欄位名稱 ---
    EXPECTED_FACTOR_COLUMNS = [
        'fundamental_score', 'sentiment_score', 'industry_trend_score',
        'market_risk_factor', 'black_swan_risk'
    ]

    # --- RL 設定 ---
    TOTAL_TRAINING_TIMESTEPS = 200_000  # 最終模型訓練步數 200_000
    TRIAL_TRAINING_TIMESTEPS = 20_000   # Optuna 優化時使用的訓練步數 10_000
    RL_ALGORITHM = PPO

    # --- 交易設定 ---
    #COMMISSION_RATE = 0.001425 * 2     # 台股買賣雙向手續費
    COMMISSION_RATE = 0.001 * 2     # 美股買賣雙向手續費
    SLIPPAGE = 0.001   # 0.001                # 滑價率
    MIN_TRADE_SHARES = 100  # 1          # 最小交易股數 (TW) 1000 --> 1(US)

    # --- 風險管理設定 ---
    PROFIT_REWARD_FACTOR = 0.025      # 盈利獎勵因子  0.015 --> 0.02
    VOLATILITY_PENALTY_FACTOR = 0.003  # a 0.003
    STOP_LOSS_PCT = 0.05               # 停損設為 5%
    TAKE_PROFIT_PCT = 0.5             # 停利設為 35% -->50%
    MAX_POSITION_RISK_PCT = 0.025      # 最大倉位風險百分比
    ATR_PERIOD_FOR_SIZING = 21         # 計算ATR的周期 14

    # --- Optuna 設定 ---
    N_OPTUNA_TRIALS = 13              # Optuna 實驗次數 10 --> 20


# -----------------------------------------------------------------------------
# 4. 設定日誌記錄器 (Logger)
# -----------------------------------------------------------------------------
def setup_logger() -> logging.Logger:
    """設置並返回配置好的日誌記錄器"""
    os.makedirs(os.path.dirname(Config.LOG_FILE_PATH), exist_ok=True)

    logger = logging.getLogger("trading")
    logger.setLevel(Config.LOGGING_LEVEL)

    # 檢查是否已經有處理器以避免重複
    if not logger.handlers:
        # 格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

        # 控制台處理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(Config.LOGGING_LEVEL)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 檔案處理器
        file_handler = logging.FileHandler(Config.LOG_FILE_PATH, mode='w', encoding='utf-8')
        file_handler.setLevel(Config.LOGGING_LEVEL)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info("日誌記錄器設定完成")
    logger.debug(f"日誌級別設定為: {logging.getLevelName(Config.LOGGING_LEVEL)}")

    return logger


# -----------------------------------------------------------------------------
# 5. 資料載入與預處理
# -----------------------------------------------------------------------------
def load_raw_data(ticker: str, start: str, end: str, save_path: str, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """下載或載入原始 OHLCV 數據"""
    logger.info(f"開始執行 load_raw_data，股票代碼: {ticker}, 日期: {start} 至 {end}")

    # 嘗試從本地載入
    if os.path.exists(save_path):
        logger.info(f"嘗試從 {save_path} 載入 OHLCV 數據...")
        try:
            df = pd.read_csv(save_path, index_col='Date', parse_dates=True)
            required_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

            if not all(col in df.columns for col in required_cols):
                logger.warning(f"警告：載入的 OHLCV 數據 {save_path} 缺少必要欄位，將重新下載。")
                df = None
            elif df.empty:
                logger.warning(f"警告：載入的 OHLCV 數據 {save_path} 為空，將重新下載。")
                df = None
            else:
                logger.info(f"從 {save_path} 載入 OHLCV 數據成功。")
                return df
        except Exception as e:
            logger.exception(f"載入 OHLCV 數據 {save_path} 時發生錯誤，將重新下載: {e}")
            df = None

    # 如果本地載入失敗，從yfinance下載
    logger.info(f"開始從 yfinance 下載 {ticker} 的 OHLCV 數據...")
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, interval='1d')

        if df.empty:
            logger.error(f"錯誤：下載 {ticker} 的資料為空。")
            return None

        logger.info("下載完成。")

        # 處理MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            logger.debug("處理 MultiIndex...")
            df.columns = df.columns.get_level_values(0)

        # 處理重複欄位
        if df.columns.has_duplicates:
            logger.warning("處理重複欄位...")
            df = df.loc[:, ~df.columns.duplicated()]

        # 標準化欄位名稱
        df.columns = [col.capitalize() for col in df.columns]
        if 'Adj close' in df.columns:
            df.rename(columns={'Adj close': 'Adj Close'}, inplace=True)

        # 檢查必要欄位
        required_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"錯誤：下載數據缺少必要欄位: {set(required_cols)-set(df.columns)}")
            return None

        # 轉換為數值型別並清理數據
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        core_price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in core_price_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 移除缺失值和零成交量
        df.dropna(subset=['Volume'] + core_price_cols, inplace=True)
        df = df[df['Volume'] > 0]

        if df.empty:
            logger.error("錯誤：清理後數據為空。")
            return None

        # 儲存到本地
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
        logger.info(f"原始 OHLCV 資料已成功儲存至 {save_path}")

        return df
    except Exception as e:
        logger.exception(f"下載或處理 OHLCV 資料時發生未預期的錯誤: {e}")
        return None


def load_factor_data(factor_path: str, expected_columns: List[str], logger: logging.Logger) -> Optional[pd.DataFrame]:
    """載入預先計算的因子數據"""
    logger.info(f"開始載入因子數據從: {factor_path}")

    if not os.path.exists(factor_path):
        logger.error(f"錯誤：找不到因子數據檔案 {factor_path}。")
        return None

    try:
        df_factors = pd.read_csv(factor_path, index_col='Date', parse_dates=True)

        # 檢查必要欄位
        missing_factors = [col for col in expected_columns if col not in df_factors.columns]
        if missing_factors:
            logger.error(f"錯誤：因子數據檔案 {factor_path} 缺少預期欄位: {missing_factors}")
            return None

        logger.info(f"成功從 {factor_path} 載入因子數據，包含欄位: {df_factors.columns.tolist()}")
        return df_factors

    except Exception as e:
        logger.exception(f"載入因子數據 {factor_path} 時發生錯誤: {e}")
        return None


def calculate_technical_indicators(df: pd.DataFrame, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """計算 TA-Lib 技術指標"""
    logger.info("開始計算技術指標...")

    if df is None or df.empty:
        logger.error("輸入 calculate_technical_indicators 的 DataFrame 為空或 None。")
        return None

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"計算指標時缺少必要欄位: {set(required_cols) - set(df.columns)}")
        return None

    df_out = df.copy()

    try:
        # 轉換數據為NumPy陣列，並確保dtype為float64
        open_price = df_out['Open'].values.astype(np.float64)
        high_price = df_out['High'].values.astype(np.float64)
        low_price = df_out['Low'].values.astype(np.float64)
        close_price = df_out['Close'].values.astype(np.float64)
        volume = df_out['Volume'].values.astype(np.float64)

        logger.debug("輸入數據已成功轉換為 float64 NumPy 陣列。")
    except Exception as e:
        logger.exception(f"轉換數據類型為 float64 時出錯: {e}")
        return None

    # 檢查數據量是否足夠
    min_required_period = 200
    if len(close_price) < min_required_period:
        logger.warning(f"警告：數據量 ({len(close_price)}) 過少，可能無法計算所有技術指標。")

    try:
        # 計算重疊研究指標
        logger.debug("計算重疊研究指標...")
        df_out['SMA_10'] = talib.SMA(close_price, timeperiod=10)
        df_out['SMA_30'] = talib.SMA(close_price, timeperiod=30)
        df_out['SMA_50'] = talib.SMA(close_price, timeperiod=50)
        df_out['SMA_200'] = talib.SMA(close_price, timeperiod=200)
        df_out['EMA_10'] = talib.EMA(close_price, timeperiod=10)
        df_out['EMA_30'] = talib.EMA(close_price, timeperiod=30)

        # 布林帶
        upper, middle, lower = talib.BBANDS(close_price, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df_out['BB_UPPER'] = upper
        df_out['BB_MIDDLE'] = middle
        df_out['BB_LOWER'] = lower

        # 拋物線SAR
        df_out['SAR'] = talib.SAR(high_price, low_price, acceleration=0.02, maximum=0.2)

        # 計算動量指標
        logger.debug("計算動量指標...")
        df_out['RSI'] = talib.RSI(close_price, timeperiod=14)

        # MACD
        macd, macdsignal, macdhist = talib.MACD(close_price, fastperiod=12, slowperiod=26, signalperiod=9)
        df_out['MACD'] = macd
        df_out['MACD_signal'] = macdsignal
        df_out['MACD_hist'] = macdhist

        # KD指標
        slowk, slowd = talib.STOCH(high_price, low_price, close_price,
                                   fastk_period=14, slowk_period=3, slowk_matype=0,
                                   slowd_period=3, slowd_matype=0)
        df_out['STOCH_k'] = slowk
        df_out['STOCH_d'] = slowd

        # 其他動量指標
        df_out['ADX'] = talib.ADX(high_price, low_price, close_price, timeperiod=14)
        df_out['CCI'] = talib.CCI(high_price, low_price, close_price, timeperiod=14)
        df_out['WILLR'] = talib.WILLR(high_price, low_price, close_price, timeperiod=14)
        df_out['MOM'] = talib.MOM(close_price, timeperiod=10)

        # 計算成交量指標
        logger.debug("計算成交量指標...")
        df_out['OBV'] = talib.OBV(close_price, volume)
        df_out['AD'] = talib.AD(high_price, low_price, close_price, volume)

        # 計算波動性指標
        logger.debug("計算波動性指標...")
        df_out['ATR'] = talib.ATR(high_price, low_price, close_price, timeperiod=Config.ATR_PERIOD_FOR_SIZING)
        df_out['NATR'] = talib.NATR(high_price, low_price, close_price, timeperiod=14)
        df_out['TRANGE'] = talib.TRANGE(high_price, low_price, close_price)

        # 計算週期指標
        logger.debug("計算週期指標...")
        df_out['HT_DCPERIOD'] = talib.HT_DCPERIOD(close_price)

        # 統計新增的指標數量
        base_cols = set(df.columns)
        num_indicators_added = len(set(df_out.columns) - base_cols)
        logger.info(f"已計算 {num_indicators_added} 個技術指標。")

        return df_out

    except Exception as e:
        logger.exception(f"計算技術指標時發生錯誤: {e}")
        return None


def merge_and_process_data(ohlcv_df: pd.DataFrame, factor_df: pd.DataFrame,
                          output_csv_path: str, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """合併價格數據和因子數據，計算技術指標，處理缺失值，並儲存最終數據"""
    logger.info(f"開始執行 merge_and_process_data...")

    if ohlcv_df is None or factor_df is None:
        logger.error("錯誤：OHLCV 或因子數據為 None，無法合併。")
        return None

    try:
        # 確保索引是日期型別
        ohlcv_df.index = pd.to_datetime(ohlcv_df.index)
        factor_df.index = pd.to_datetime(factor_df.index)

        # 合併數據
        merged_df = pd.merge(
            ohlcv_df,
            factor_df[Config.EXPECTED_FACTOR_COLUMNS],
            left_index=True,
            right_index=True,
            how='inner'
        )

        logger.info(f"成功合併價格數據和因子數據，合併後共 {len(merged_df)} 筆。")

        if merged_df.empty:
            logger.error("錯誤：價格數據與因子數據合併後為空，請檢查日期範圍或索引是否匹配。")
            return None

    except Exception as e:
        logger.exception(f"合併價格數據和因子數據時發生錯誤: {e}")
        return None

    # 計算技術指標
    df_with_indicators = calculate_technical_indicators(merged_df, logger)
    if df_with_indicators is None:
        logger.error("技術指標計算失敗。")
        return None

    # 處理缺失值和無限值
    logger.info("開始處理最終數據的缺失值...")
    initial_len = len(df_with_indicators)

    df_with_indicators.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final = df_with_indicators.dropna()

    final_len = len(df_final)
    rows_removed = initial_len - final_len
    logger.info(f"處理缺失值：共移除 {rows_removed} 行。")

    if df_final.empty:
        logger.error("錯誤：處理缺失值後數據為空。")
        return None

    # 儲存處理後的數據
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    try:
        df_final.to_csv(output_csv_path)
        logger.info(f"最終合併和處理後的資料已儲存至 {output_csv_path}")
    except Exception as e:
        logger.exception(f"儲存最終處理數據到 {output_csv_path} 時失敗: {e}")
        return None

    logger.info("merge_and_process_data 函數執行完畢。")
    return df_final


# -----------------------------------------------------------------------------
# 6. 強化學習環境 (Custom Gym Environment)
# -----------------------------------------------------------------------------
class StockTradingEnv(gym.Env):
    """股票交易強化學習環境，含停利停損機制"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame,
                 window_size: int = Config.WINDOW_SIZE,
                 initial_balance: float = Config.INITIAL_BALANCE,
                 commission_rate: float = Config.COMMISSION_RATE,
                 slippage: float = Config.SLIPPAGE,
                 min_trade_shares: int = Config.MIN_TRADE_SHARES,
                 profit_reward_factor: float = Config.PROFIT_REWARD_FACTOR,
                 volatility_penalty_factor: float = Config.VOLATILITY_PENALTY_FACTOR,
                 stop_loss_pct: float = Config.STOP_LOSS_PCT,
                 take_profit_pct: float = Config.TAKE_PROFIT_PCT,
                 max_position_risk_pct: float = Config.MAX_POSITION_RISK_PCT,
                 atr_period: int = Config.ATR_PERIOD_FOR_SIZING,
                 main_logger: Optional[logging.Logger] = None):
        """初始化環境"""
        super(StockTradingEnv, self).__init__()

        # 設置環境專屬的日誌記錄器
        env_id = id(self)
        if main_logger:
            self.logger = main_logger.getChild(f"Env_{env_id}")
        else:
            self.logger = logging.getLogger(f"trading.Env_{env_id}")

        self.logger.info(f"初始化 StockTradingEnv (ID: {env_id})...")

        # 基本驗證
        if df is None or df.empty:
            self.logger.error("錯誤：傳遞給環境的 DataFrame 為空或 None。")
            raise ValueError("DataFrame cannot be None or empty for StockTradingEnv")

        if 'ATR' not in df.columns:
            self.logger.error(f"錯誤：DataFrame 缺少 'ATR' 欄位。")
            raise ValueError(f"DataFrame missing 'ATR' column")

        # 獲取因子列
        self.other_feature_columns = [col for col in Config.EXPECTED_FACTOR_COLUMNS if col in df.columns]
        if len(self.other_feature_columns) != len(Config.EXPECTED_FACTOR_COLUMNS):
            self.logger.warning(f"警告：DataFrame 中實際存在的因子欄位 ({len(self.other_feature_columns)}) 與預期 ({len(Config.EXPECTED_FACTOR_COLUMNS)}) 不符。")
            self.logger.warning(f"  實際存在的因子欄位: {self.other_feature_columns}")

        self.n_other_features = len(self.other_feature_columns)

        # 儲存參數
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

        # 定義動作空間：0=賣出, 1=持有, 2=買入
        self.action_space = spaces.Discrete(3)

        # 計算觀察空間維度
        n_price_volume_features = 6  # Open, High, Low, Close, Adj Close, Volume
        self.technical_indicator_columns = [
            col for col in self.df.columns
            if col not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] + self.other_feature_columns
        ]
        n_technical_indicators = len(self.technical_indicator_columns)
        n_portfolio_features = 3  # 現金比例, 股票比例, 損益比例

        self.observation_shape = (
            window_size,
            n_price_volume_features + n_technical_indicators + self.n_other_features + n_portfolio_features
        )

        # 定義觀察空間
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.observation_shape,
            dtype=np.float32
        )

        self.logger.info(f"環境觀察空間維度: {self.observation_shape}")
        self.logger.debug(f"  基礎價格特徵數: {n_price_volume_features}")
        self.logger.debug(f"  技術指標數: {n_technical_indicators}")
        self.logger.debug(f"  其他因子數: {self.n_other_features}")
        self.logger.debug(f"  投資組合特徵數: {n_portfolio_features}")

        # 初始化狀態變數
        self.current_step = 0
        self.balance = 0
        self.shares_held = 0
        self.net_worth = 0
        self.max_net_worth = 0
        self.trade_history = []
        self.portfolio_value_history = []
        self.entry_price = 0.0
        self.last_trade_profit = 0.0

        # 用於標準化的scaler
        self.scaler = StandardScaler()
        if not self._fit_scaler():
            raise RuntimeError("環境初始化失敗：Scaler 擬合失敗。")

        # 重置環境
        self.reset()
        self.logger.info(f"StockTradingEnv (ID: {env_id}) 初始化完成。")

    def _fit_scaler(self) -> bool:
        """擬合用於觀察值標準化的 Scaler"""
        self.logger.debug("開始執行 _fit_scaler...")

        # 確定需要縮放的列
        cols_to_scale = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] + self.technical_indicator_columns + self.other_feature_columns
        available_cols_to_scale = [col for col in cols_to_scale if col in self.df.columns]

        if len(available_cols_to_scale) != len(cols_to_scale):
            self.logger.warning(f"警告 (_fit_scaler): DataFrame 缺少部分預期用於縮放的欄位。")
            missing = set(cols_to_scale) - set(available_cols_to_scale)
            self.logger.warning(f"  缺少: {missing}")
            cols_to_scale = available_cols_to_scale

        if not cols_to_scale:
            self.logger.error("錯誤 (_fit_scaler): 沒有可用於縮放的欄位。")
            return False

        # 準備用於擬合的數據
        observable_df = self.df[cols_to_scale]
        observable_df.replace([np.inf, -np.inf], 0, inplace=True)
        observable_df.fillna(0, inplace=True)

        try:
            self.scaler.fit(observable_df)
            self.logger.info(f"_fit_scaler: Scaler 擬合完成，基於 {len(cols_to_scale)} 個欄位。")
            return True
        except Exception as e:
            self.logger.exception(f"錯誤 (_fit_scaler): Scaler 擬合失敗: {e}")
            return False

    def _get_current_price(self) -> float:
        """獲取當前步驟的收盤價"""
        idx = self.current_step + self.window_size - 1
        if idx < len(self.df):
            return self.df['Close'].iloc[idx]
        else:
            self.logger.warning(f"_get_current_price: 索引 {idx} 超出範圍，返回最後一個價格。")
            return self.df['Close'].iloc[-1]

    def _get_current_atr(self) -> float:
        """獲取當前步驟的 ATR 值"""
        idx = self.current_step + self.window_size - 1
        if idx < len(self.df):
            return self.df['ATR'].iloc[idx]
        else:
            self.logger.warning(f"_get_current_atr: 索引 {idx} 超出範圍，返回 NaN。")
            return np.nan

    def _next_observation(self) -> np.ndarray:
        """獲取下一個觀察狀態"""
        self.logger.debug(f"執行 _next_observation，當前步驟: {self.current_step}")

        # 獲取當前窗口的數據
        start_idx = self.current_step
        end_idx = self.current_step + self.window_size

        if end_idx > len(self.df):
            self.logger.error(f"錯誤 (_next_observation): 請求的結束索引 {end_idx} 超出 DataFrame 長度 {len(self.df)}。")
            return np.zeros(self.observation_shape, dtype=np.float32)

        frame = self.df.iloc[start_idx:end_idx]

        # 確定需要縮放的列
        cols_to_scale = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] + self.technical_indicator_columns + self.other_feature_columns
        available_cols_to_scale = [col for col in cols_to_scale if col in frame.columns]

        if len(available_cols_to_scale) != len(cols_to_scale):
            self.logger.warning(f"警告 (_next_observation): DataFrame 缺少部分預期用於縮放的欄位。")
            cols_to_scale = available_cols_to_scale

        if not cols_to_scale:
            self.logger.error("錯誤 (_next_observation): 沒有可用於縮放的欄位。返回零數組。")
            return np.zeros(self.observation_shape, dtype=np.float32)

        # 準備特徵並處理無限值和NaN
        obs_features = frame[cols_to_scale].values
        obs_features[np.isinf(obs_features)] = 0
        obs_features = np.nan_to_num(obs_features)

        # 標準化特徵
        try:
            scaled_features = self.scaler.transform(obs_features)
        except Exception as e:
            self.logger.exception(f"錯誤 (_next_observation): Scaler 轉換失敗: {e}")
            return np.zeros(self.observation_shape, dtype=np.float32)

        # 計算投資組合狀態
        current_price = self._get_current_price()
        total_value = self.balance + self.shares_held * current_price

        balance_ratio = self.balance / total_value if total_value > 0 else 0
        shares_ratio = (self.shares_held * current_price) / total_value if total_value > 0 else 0
        initial_investment = self.initial_balance
        pnl_ratio = (total_value - initial_investment) / initial_investment if initial_investment > 0 else 0

        portfolio_state = np.array([balance_ratio, shares_ratio, pnl_ratio])
        portfolio_state_expanded = np.tile(portfolio_state, (self.window_size, 1))

        # 檢查縮放後特徵的形狀
        expected_scaled_shape = (self.window_size, len(cols_to_scale))
        if scaled_features.shape != expected_scaled_shape:
            self.logger.error(f"錯誤 (_next_observation): scaled_features 形狀不符。預期 {expected_scaled_shape}, 實際 {scaled_features.shape}")
            return np.zeros(self.observation_shape, dtype=np.float32)

        # 組合觀察
        try:
            observation = np.hstack((scaled_features, portfolio_state_expanded))
        except ValueError as e:
            self.logger.error(f"錯誤 (_next_observation): hstack 失敗: {e}")
            self.logger.error(f"  scaled_features shape: {scaled_features.shape}")
            self.logger.error(f"  portfolio_state_expanded shape: {portfolio_state_expanded.shape}")
            return np.zeros(self.observation_shape, dtype=np.float32)

        # 檢查最終觀察形狀
        if observation.shape != self.observation_shape:
            self.logger.error(f"錯誤 (_next_observation): 最終觀察形狀不符。預期 {self.observation_shape}, 實際 {observation.shape}")
            return np.zeros(self.observation_shape, dtype=np.float32)

        self.logger.debug(f"_next_observation 完成，返回觀察 shape: {observation.shape}")
        return observation.astype(np.float32)

    def reset(self) -> np.ndarray:
        """重置環境到初始狀態"""
        self.logger.info("重置環境 (reset)...")

        # 重置狀態變數
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.current_step = 0
        self.trade_history = []
        self.portfolio_value_history = [self.initial_balance]
        self.entry_price = 0.0
        self.last_trade_profit = 0.0

        # 檢查數據是否足夠
        if len(self.df) < self.window_size:
            self.logger.error(f"錯誤 (reset): 數據長度 ({len(self.df)}) 小於窗口大小 ({self.window_size})。")
            raise ValueError(f"Data length ({len(self.df)}) is less than window size ({self.window_size}). Cannot reset environment.")

        self.logger.info("環境重置完成。")
        return self._next_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """執行一個時間步 (包含停利、停損和獎勵塑形)"""
        self.logger.debug(f"執行 step，當前步驟: {self.current_step}, 原始動作: {action}")

        # 檢查是否已到達數據末尾
        is_done = self.current_step >= len(self.df) - self.window_size
        if is_done:
            self.logger.info(f"到達數據末尾 (step {self.current_step})，環境結束。")
            try:
                final_obs = self._next_observation()
            except Exception as e:
                self.logger.exception("獲取最終觀察時出錯，返回零數組。")
                final_obs = np.zeros(self.observation_shape, dtype=np.float32)

            return final_obs, 0.0, True, {'net_worth': self.net_worth, 'trades': len(self.trade_history)}

        # 獲取當前價格和保存上一步的淨值
        current_price = self._get_current_price()
        prev_net_worth = self.net_worth
        self.last_trade_profit = 0.0

        # 初始化觸發標記
        stop_loss_triggered = False
        take_profit_triggered = False

        # --- 停利邏輯 (優先於停損檢查) ---
        if self.shares_held > 0 and self.entry_price > 0 and self.take_profit_pct > 0:
            take_profit_price = self.entry_price * (1 + self.take_profit_pct)
            if current_price >= take_profit_price:
                self.logger.info(f"停利觸發! 當前價格 {current_price:.2f} >= 停利價格 {take_profit_price:.2f} (入場價 {self.entry_price:.2f})")
                action = 0  # 強制賣出
                take_profit_triggered = True

        # --- 停損邏輯 (如果未觸發停利) ---
        if not take_profit_triggered and self.shares_held > 0 and self.entry_price > 0 and self.stop_loss_pct > 0:
            stop_loss_price = self.entry_price * (1 - self.stop_loss_pct)
            if current_price <= stop_loss_price:
                self.logger.info(f"停損觸發! 當前價格 {current_price:.2f} <= 停損價格 {stop_loss_price:.2f} (入場價 {self.entry_price:.2f})")
                action = 0  # 強制賣出
                stop_loss_triggered = True

        self.logger.debug(f"  最終執行動作: {action} (停利觸發: {take_profit_triggered}, 停損觸發: {stop_loss_triggered})")

        # 執行交易動作
        self._take_action(action, current_price)

        # --- 更新狀態和計算獎勵 ---
        self.current_step += 1
        next_price_idx = self.current_step + self.window_size - 1

        if next_price_idx < len(self.df):
            next_price = self.df['Close'].iloc[next_price_idx]
        else:
            self.logger.warning(f"警告 (step): 計算獎勵時 next_price_idx ({next_price_idx}) 超出範圍。")
            next_price = current_price

        # 使用 Close 計算淨值
        self.net_worth = self.balance + self.shares_held * next_price
        self.portfolio_value_history.append(self.net_worth)
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # --- 計算塑形後的獎勵 ---
        base_reward = (self.net_worth - prev_net_worth) / prev_net_worth if prev_net_worth != 0 else 0
        profit_bonus = 0.0

        if self.trade_history:
            last_trade = self.trade_history[-1]
            # 檢查是否是剛剛完成的賣出交易
            if last_trade['step'] == (self.current_step + self.window_size - 2) and last_trade['type'] == 'SELL':
                if take_profit_triggered:
                    profit_bonus = self.profit_reward_factor
                    self.last_trade_profit = last_trade['cost/proceeds']
                    self.logger.debug(f"  給予停利交易獎勵: {profit_bonus:.6f}")

        # 計算波動性懲罰
        volatility_penalty = 0.0
        lookback_period = 20
        if len(self.portfolio_value_history) > lookback_period + 1:
            recent_returns = pd.Series(self.portfolio_value_history[-lookback_period-1:]).pct_change().dropna()
            if not recent_returns.empty:
                returns_std_dev = recent_returns.std()
                volatility_penalty = self.volatility_penalty_factor * returns_std_dev
                self.logger.debug(f"  計算波動性懲罰: std_dev={returns_std_dev:.6f}, penalty={volatility_penalty:.6f}")

        # 停損懲罰
        stop_loss_penalty = -0.01 if stop_loss_triggered else 0.0

        # 停利獎勵
        take_profit_reward = 0.01 if take_profit_triggered else 0.0

        # 總獎勵
        reward = base_reward + take_profit_reward - volatility_penalty + stop_loss_penalty

        self.logger.debug(f"  步驟 {self.current_step}: 價格={next_price:.2f}, 淨值={self.net_worth:.2f}, " +
                         f"Reward Breakdown: base={base_reward:.6f}, tp_reward={take_profit_reward:.2f}, " +
                         f"vol_penalty={volatility_penalty:.6f}, sl_penalty={stop_loss_penalty:.2f}, " +
                         f"Total Reward={reward:.6f}")

        # 檢查是否完成
        done = self.current_step >= len(self.df) - self.window_size

        # 獲取下一個觀察
        try:
            obs = self._next_observation()
        except Exception as e:
            self.logger.exception("獲取下一步觀察時出錯，返回零數組並結束環境。")
            obs = np.zeros(self.observation_shape, dtype=np.float32)
            done = True

        # 返回信息
        info = {
            'net_worth': self.net_worth,
            'trades': len(self.trade_history),
            'last_trade_profit': self.last_trade_profit
        }

        self.logger.debug(f"step 完成，返回: done={done}, info={info}")
        return obs, reward, done, info

    def _take_action(self, action: int, current_price: float) -> None:
        """執行買入、賣出或持有動作 (使用 Close 價格, 包含動態倉位管理)"""
        action_type = action
        trade_executed = False
        trade_type = "HOLD"
        trade_shares = 0
        trade_cost = 0

        buy_price = current_price * (1 + self.slippage)
        sell_price = current_price * (1 - self.slippage)

        if action_type == 2:  # Buy
            if self.balance > 0:
                # 動態倉位管理 (基於ATR)
                current_atr = self._get_current_atr()

                if np.isnan(current_atr) or current_atr <= 0:
                    self.logger.warning(f"ATR 無效 ({current_atr})，使用預設倉位比例 0.5")
                    position_pct = 0.5
                    cash_for_trade = self.balance * 0.95 * position_pct
                    shares_to_buy = int(cash_for_trade / buy_price / self.min_trade_shares) * self.min_trade_shares
                    self.logger.debug(f"  動態倉位計算 (ATR無效): 使用 {position_pct*100:.1f}% 現金, SharesToBuy={shares_to_buy}")
                else:
                    # 基於風險的倉位計算
                    total_asset_value = self.balance + self.shares_held * current_price
                    max_shares_by_risk = (total_asset_value * self.max_position_risk_pct) / current_atr
                    cash_for_trade = self.balance * 0.95
                    max_shares_by_cash = cash_for_trade / buy_price
                    target_shares = min(max_shares_by_risk, max_shares_by_cash)
                    shares_to_buy = int(target_shares / self.min_trade_shares) * self.min_trade_shares
                    self.logger.debug(f"  動態倉位計算: ATR={current_atr:.2f}, MaxSharesRisk={max_shares_by_risk:.0f}, "
                                    f"MaxSharesCash={max_shares_by_cash:.0f}, TargetShares={target_shares:.0f}, "
                                    f"SharesToBuy={shares_to_buy}")

                if shares_to_buy > 0:
                    # 計算成本包含手續費
                    cost = shares_to_buy * buy_price * (1 + self.commission_rate)

                    if self.balance >= cost:
                        self.balance -= cost

                        # 記錄入場價
                        if self.shares_held == 0:
                            self.entry_price = buy_price
                            self.logger.debug(f"  記錄新入場價: {self.entry_price:.2f}")
                        else:
                            self.logger.debug(f"  加倉操作，保持初始入場價: {self.entry_price:.2f}")

                        self.shares_held += shares_to_buy
                        trade_executed = True
                        trade_type = "BUY"
                        trade_shares = shares_to_buy
                        trade_cost = -cost
                        self.last_trade_cost = cost
                        self.logger.info(f"執行買入 (動態倉位): {trade_shares} 股 @ {current_price:.2f}, 成本: {cost:.2f}")
                    else:
                        self.logger.warning(f"嘗試買入但餘額不足: 需要 {cost:.2f}, 餘額 {self.balance:.2f}")
                else:
                    self.logger.debug("計算可買入股數為 0。")
            else:
                self.logger.debug("餘額為 0，無法買入。")

        elif action_type == 0:  # Sell
            if self.shares_held > 0:
                # 計算賣出價值與手續費
                sell_value = self.shares_held * sell_price
                commission = sell_value * self.commission_rate
                proceeds = sell_value - commission

                self.balance += proceeds
                trade_executed = True
                trade_type = "SELL"
                trade_shares = self.shares_held
                trade_cost = proceeds

                self.logger.info(f"執行賣出: {trade_shares} 股 @ {current_price:.2f} "
                                f"(實際賣價含滑價: {sell_price:.2f}), 收入: {proceeds:.2f}")

                self.shares_held = 0
                self.last_trade_cost = 0
                self.entry_price = 0.0  # 清空入場價
            else:
                self.logger.debug("嘗試賣出但未持有任何股份。")

        # 記錄交易
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
                self.logger.debug(f"  記錄交易: {trade_info}")
            else:
                self.logger.warning(f"警告: 嘗試記錄交易時索引 {step_index} 超出範圍。")

    def render(self, mode='human', close=False):
        """渲染環境狀態"""
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


# -----------------------------------------------------------------------------
# 7. 模型訓練相關
# -----------------------------------------------------------------------------
class TensorboardCallback(BaseCallback):
    """訓練回調，用於記錄訓練過程中的指標"""
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.tb_logger = logging.getLogger("trading.TensorboardCallback")

    def _on_step(self) -> bool:
        """每步調用的回調函數"""
        if len(self.locals['infos']) > 0 and 'net_worth' in self.locals['infos'][0]:
            net_worth = self.locals['infos'][0]['net_worth']
            self.logger.record('rollout/net_worth', net_worth)
        return True


def train_model(env: gym.Env,
                hyperparameters: Dict[str, Any],
                save_path: str = Config.MODEL_SAVE_PATH,
                log_dir: str = Config.SB3_LOG_DIR,
                total_timesteps: int = Config.TOTAL_TRAINING_TIMESTEPS,
                logger: Optional[logging.Logger] = None) -> Optional[PPO]:
    """訓練 RL 模型並儲存"""
    if logger is None:
        logger = logging.getLogger("trading.train_model")

    logger.info(f"開始執行 train_model 函數，儲存路徑: {save_path}, SB3 日誌路徑: {log_dir}")
    logger.info(f"使用超參數: {hyperparameters}")
    logger.info(f"總訓練步數: {total_timesteps}")

    # 設置SB3日誌路徑
    run_log_dir = os.path.join(log_dir, f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    try:
        os.makedirs(run_log_dir, exist_ok=True)
        sb3_output_formats = ["stdout", "csv", "tensorboard"]
        sb3logger = sb3_configure(run_log_dir, sb3_output_formats)
        logger.info(f"Stable Baselines 3 logger 設定完成，輸出至: {run_log_dir}")
    except Exception as e:
        logger.exception(f"設定 Stable Baselines 3 logger 時失敗: {e}")
        return None

    try:
        # 創建模型，使用傳入的超參數
        model = Config.RL_ALGORITHM(
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
        model.set_logger(sb3logger)
        logger.info(f"RL 模型 ({Config.RL_ALGORITHM.__name__}) 創建成功。")
    except Exception as e:
        logger.exception(f"創建 RL 模型失敗: {e}")
        return None

    logger.info(f"開始訓練模型...")
    callback = TensorboardCallback()

    try:
        model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=100)
        logger.info("模型訓練完成。")
    except Exception as e:
        logger.exception(f"模型訓練過程中發生錯誤: {e}")
        raise e

    # 只有最終模型才保存
    if save_path and total_timesteps == Config.TOTAL_TRAINING_TIMESTEPS:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            model.save(save_path)
            logger.info(f"最終模型已儲存至 {save_path}")
        except Exception as e:
            logger.exception(f"儲存最終模型時發生錯誤: {e}")
    elif save_path:
        logger.info(f"非最終訓練 (步數 {total_timesteps})，不儲存模型到 {save_path}。")

    logger.info("train_model 函數執行完畢。")
    return model


# -----------------------------------------------------------------------------
# 8. 回測/評估
# -----------------------------------------------------------------------------
def evaluate_model(df_test: pd.DataFrame,
                  model=None,
                  model_path: str = Config.MODEL_SAVE_PATH,
                  initial_balance: float = Config.INITIAL_BALANCE,
                  window_size: int = Config.WINDOW_SIZE,
                  save_trades_path: str = Config.TRADES_SAVE_PATH,
                  logger: Optional[logging.Logger] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """評估訓練好的模型"""
    if logger is None:
        logger = logging.getLogger("trading.evaluate_model")

    logger.info(f"開始執行 evaluate_model...")

    # 載入模型
    loaded_model = None
    if model is not None:
        logger.info("使用傳入的模型對象進行評估。")
        loaded_model = model
    elif model_path and os.path.exists(model_path):
        logger.info(f"從路徑 {model_path} 載入模型...")
        try:
            loaded_model = Config.RL_ALGORITHM.load(model_path)
            logger.info(f"模型 {model_path} 載入成功。")
        except Exception as e:
            logger.exception(f"載入模型 {model_path} 時發生錯誤: {e}")
            return None, None
    else:
        logger.error(f"錯誤：未提供模型對象，且找不到模型檔案 {model_path}。")
        return None, None

    # 創建評估環境
    logger.info("創建評估環境...")
    try:
        eval_env = StockTradingEnv(df_test, window_size=window_size, initial_balance=initial_balance, main_logger=logger)
        logger.info("評估環境創建成功。")
    except Exception as e:
        logger.exception(f"創建評估環境時發生錯誤: {e}")
        return None, None

    # 開始回測
    obs = eval_env.reset()
    done = False
    logger.info("開始回測...")
    step_count = 0
    max_steps = len(df_test) - window_size

    while not done:
        if step_count > max_steps + 5:
            logger.warning(f"警告：回測步驟 ({step_count}) 超出預期最大值 ({max_steps})，強制終止。")
            break

        try:
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            step_count += 1

            if step_count % 250 == 0:
                logger.info(f"  回測進度: Step {step_count}/{max_steps}")
        except Exception as e:
            logger.exception(f"回測步驟 {step_count} 發生錯誤: {e}")
            done = True

    logger.info(f"回測完成。總步數: {step_count}")

    # 整理交易記錄
    trade_log_df = pd.DataFrame(eval_env.trade_history)

    # 處理投資組合價值歷史
    start_idx_portfolio = window_size - 1
    expected_len_portfolio = step_count + 1
    actual_len_portfolio = len(eval_env.portfolio_value_history)

    if actual_len_portfolio != expected_len_portfolio:
        logger.warning(f"警告: 投資組合歷史長度 ({actual_len_portfolio}) 與預期 ({expected_len_portfolio}) 不符。")
        actual_len_portfolio = min(actual_len_portfolio, expected_len_portfolio)
        eval_env.portfolio_value_history = eval_env.portfolio_value_history[:actual_len_portfolio]

    end_idx_portfolio = start_idx_portfolio + actual_len_portfolio
    if end_idx_portfolio > len(df_test):
        logger.warning(f"警告: 計算出的結束索引 ({end_idx_portfolio}) 超出測試數據長度 ({len(df_test)})。")
        end_idx_portfolio = len(df_test)
        actual_len_portfolio = end_idx_portfolio - start_idx_portfolio
        eval_env.portfolio_value_history = eval_env.portfolio_value_history[:actual_len_portfolio]

    # 創建投資組合價值Series
    portfolio_values = pd.Series(dtype=np.float64)
    if start_idx_portfolio < end_idx_portfolio:
        portfolio_values = pd.Series(
            eval_env.portfolio_value_history,
            index=df_test.index[start_idx_portfolio:end_idx_portfolio]
        )
        logger.info(f"成功獲取投資組合價值歷史，長度: {len(portfolio_values)}")
    else:
        logger.error("錯誤: 計算出的投資組合索引範圍無效。")

    # 儲存交易記錄
    if save_trades_path and not trade_log_df.empty:
        os.makedirs(os.path.dirname(save_trades_path), exist_ok=True)
        try:
            trade_log_df['date'] = pd.to_datetime(trade_log_df['date']).dt.strftime('%Y-%m-%d')
            trade_log_df.to_csv(save_trades_path, index=False, float_format='%.2f')
            logger.info(f"交易記錄已儲存至 {save_trades_path}")
        except Exception as e:
            logger.exception(f"儲存交易記錄時發生錯誤: {e}")
    elif not trade_log_df.empty:
        logger.info("未指定儲存路徑，不儲存交易記錄。")
    else:
        logger.info("回測期間沒有發生任何交易。")

    logger.info("evaluate_model 函數執行完畢。")
    return trade_log_df, portfolio_values


# -----------------------------------------------------------------------------
# 9. 績效指標計算
# -----------------------------------------------------------------------------
def calculate_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """計算條件風險價值 (CVaR)"""
    if returns is None or returns.empty or returns.isnull().all():
        return np.nan

    returns = returns.dropna()
    if returns.empty or len(returns) < 2:
        return np.nan

    var = returns.quantile(alpha)
    cvar = returns[returns <= var].mean()

    return cvar


def calculate_metrics(portfolio_values: pd.Series,
                     df_test: pd.DataFrame,
                     trade_log_df: pd.DataFrame,
                     initial_balance: float = Config.INITIAL_BALANCE,
                     risk_free_rate: float = 0.0,
                     logger: Optional[logging.Logger] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """計算並返回策略和 Buy&Hold 的績效指標字典"""
    if logger is None:
        logger = logging.getLogger("trading.calculate_metrics")

    logger.info("開始計算績效指標 (包含 Buy&Hold)...")

    # 初始化指標字典
    strategy_metrics = {
        "年化報酬率 (Annualized Return)": np.nan,
        "夏普比率 (Sharpe Ratio)": np.nan,
        "最大回撤 (Maximum Drawdown)": np.nan,
        "條件風險價值 (CVaR 95%)": np.nan,
        "年化波動率 (Annualized Volatility)": np.nan,
        "勝率 (Win Rate)": np.nan,
        "總交易次數 (Total Trades)": len(trade_log_df) if trade_log_df is not None else 0,
        "最終淨值 (Final Net Worth)": initial_balance
    }

    buy_hold_metrics = {
        "年化報酬率 (Annualized Return)": np.nan,
        "夏普比率 (Sharpe Ratio)": np.nan,
        "最大回撤 (Maximum Drawdown)": np.nan,
        "條件風險價值 (CVaR 95%)": np.nan,
        "年化波動率 (Annualized Volatility)": np.nan,
        "勝率 (Win Rate)": np.nan,
        "總交易次數 (Total Trades)": 1,
        "最終淨值 (Final Net Worth)": initial_balance
    }

    # 檢查投資組合價值序列
    if portfolio_values is None or portfolio_values.empty:
        logger.warning("警告：策略的投資組合價值序列為空。")
        return strategy_metrics, buy_hold_metrics

    if len(portfolio_values) < 2:
        logger.warning("警告：策略的投資組合價值序列長度不足 (<2)。")
        strategy_metrics["最終淨值 (Final Net Worth)"] = portfolio_values.iloc[-1] if not portfolio_values.empty else initial_balance
        return strategy_metrics, buy_hold_metrics

    # 更新策略最終淨值
    strategy_metrics["最終淨值 (Final Net Worth)"] = portfolio_values.iloc[-1]
    logger.debug(f"策略最終淨值: {strategy_metrics['最終淨值 (Final Net Worth)']:.2f}")

    # 計算策略指標
    returns = portfolio_values.pct_change().dropna()
    if not returns.empty and not returns.isnull().all():
        annual_factor = 252  # 交易日數
        total_return = (portfolio_values.iloc[-1] / initial_balance) - 1
        num_trading_days = len(returns)

        if num_trading_days > 0:
            if num_trading_days < annual_factor / 4:
                logger.warning(f"警告: 策略交易日數 ({num_trading_days}) 過少。")

            # 年化報酬率
            annual_return = ((1 + total_return) ** (annual_factor / num_trading_days)) - 1
            strategy_metrics["年化報酬率 (Annualized Return)"] = annual_return
            logger.debug(f"策略年化報酬率: {annual_return:.2%}")

        # 年化波動率
        annual_volatility = returns.std() * np.sqrt(annual_factor)
        strategy_metrics["年化波動率 (Annualized Volatility)"] = 0.0 if np.isnan(annual_volatility) else annual_volatility
        logger.debug(f"策略年化波動率: {strategy_metrics['年化波動率 (Annualized Volatility)']:.2%}")

        # 夏普比率
        mean_daily_return = returns.mean()
        if annual_volatility != 0 and not np.isnan(annual_volatility):
            sharpe_ratio = (mean_daily_return * annual_factor - risk_free_rate) / annual_volatility
            strategy_metrics["夏普比率 (Sharpe Ratio)"] = sharpe_ratio
            logger.debug(f"策略夏普比率: {sharpe_ratio:.3f}")
        else:
            strategy_metrics["夏普比率 (Sharpe Ratio)"] = 0.0

        # 最大回撤
        roll_max = portfolio_values.cummax()
        daily_drawdown = portfolio_values / roll_max - 1.0
        max_drawdown = daily_drawdown.min()
        strategy_metrics["最大回撤 (Maximum Drawdown)"] = 0.0 if np.isnan(max_drawdown) else max_drawdown
        logger.debug(f"策略最大回撤: {max_drawdown:.2%}")

        # 條件風險價值 (CVaR)
        cvar_95 = calculate_cvar(returns, alpha=0.05)
        strategy_metrics["條件風險價值 (CVaR 95%)"] = cvar_95
        logger.debug(f"策略條件風險價值 (CVaR 95%): {cvar_95:.2%}")

        # 計算勝率
        win_rate = np.nan
        if trade_log_df is not None and not trade_log_df.empty:
            profitable_trades = 0
            total_closed_trades = 0
            entry_cost_total = 0
            entry_shares_total = 0

            logger.debug("開始計算策略勝率...")

            for i, trade in trade_log_df.iterrows():
                if trade['type'] == 'BUY':
                    entry_cost_total += abs(trade['cost/proceeds'])
                    entry_shares_total += trade['shares']
                elif trade['type'] == 'SELL' and entry_shares_total > 0:
                    exit_proceeds = trade['cost/proceeds']
                    avg_entry_cost_per_share = entry_cost_total / entry_shares_total if entry_shares_total > 0 else 0
                    shares_sold = trade['shares']

                    if shares_sold > entry_shares_total + 1:
                        logger.warning(f"警告 (勝率計算): 賣出股數 ({shares_sold}) > 持有股數 ({entry_shares_total})。")
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
                logger.info(f"策略勝率計算完成: {profitable_trades} / {total_closed_trades} = {win_rate:.2%}")
            elif not trade_log_df[trade_log_df['type']=='SELL'].empty:
                win_rate = 0.0
                logger.info("策略勝率計算完成: 勝率為 0.0%")
            else:
                logger.info("策略勝率計算完成: 無完整買賣週期。")

        strategy_metrics["勝率 (Win Rate)"] = win_rate
    else:
        logger.warning("警告：策略的收益率序列為空或全為 NaN。")

    # 計算 Buy & Hold 指標
    logger.debug("計算買入持有策略 (Buy & Hold) 指標 (基於 Close)...")

    if df_test is not None and not df_test.empty and 'Close' in df_test.columns:
        try:
            # 確定 Buy & Hold 的開始日期
            first_eval_date = portfolio_values.index.min() if portfolio_values is not None and not portfolio_values.empty else df_test.index[window_size-1]
            df_bh_calc = df_test.loc[first_eval_date:]

            if not df_bh_calc.empty:
                start_price_bh = df_bh_calc['Close'].iloc[0]

                if start_price_bh > 0:
                    # 計算 Buy & Hold 的投資組合價值
                    buy_hold_values = initial_balance * (df_bh_calc['Close'] / start_price_bh)
                    buy_hold_metrics["最終淨值 (Final Net Worth)"] = buy_hold_values.iloc[-1]
                    logger.debug(f"Buy&Hold 最終淨值: {buy_hold_metrics['最終淨值 (Final Net Worth)']:.2f}")

                    # 計算 Buy & Hold 的收益率
                    bh_returns = buy_hold_values.pct_change().dropna()

                    if not bh_returns.empty:
                        bh_annual_factor = 252
                        bh_total_return = (buy_hold_values.iloc[-1] / initial_balance) - 1
                        bh_num_trading_days = len(bh_returns)

                        if bh_num_trading_days > 0:
                            if bh_num_trading_days < bh_annual_factor / 4:
                                logger.warning(f"警告: Buy&Hold 交易日數 ({bh_num_trading_days}) 過少。")

                            # 年化報酬率
                            bh_annual_return = ((1 + bh_total_return) ** (bh_annual_factor / bh_num_trading_days)) - 1
                            buy_hold_metrics["年化報酬率 (Annualized Return)"] = bh_annual_return
                            logger.debug(f"Buy&Hold 年化報酬率: {bh_annual_return:.2%}")

                        # 年化波動率
                        bh_annual_volatility = bh_returns.std() * np.sqrt(bh_annual_factor)
                        buy_hold_metrics["年化波動率 (Annualized Volatility)"] = 0.0 if np.isnan(bh_annual_volatility) else bh_annual_volatility
                        logger.debug(f"Buy&Hold 年化波動率: {buy_hold_metrics['年化波動率 (Annualized Volatility)']:.2%}")

                        # 夏普比率
                        bh_mean_daily_return = bh_returns.mean()
                        if bh_annual_volatility != 0 and not np.isnan(bh_annual_volatility):
                            bh_sharpe_ratio = (bh_mean_daily_return * bh_annual_factor - risk_free_rate) / bh_annual_volatility
                            buy_hold_metrics["夏普比率 (Sharpe Ratio)"] = bh_sharpe_ratio
                            logger.debug(f"Buy&Hold 夏普比率: {bh_sharpe_ratio:.3f}")
                        else:
                            buy_hold_metrics["夏普比率 (Sharpe Ratio)"] = 0.0

                        # 最大回撤
                        bh_roll_max = buy_hold_values.cummax()
                        bh_daily_drawdown = buy_hold_values / bh_roll_max - 1.0
                        bh_max_drawdown = bh_daily_drawdown.min()
                        buy_hold_metrics["最大回撤 (Maximum Drawdown)"] = 0.0 if np.isnan(bh_max_drawdown) else bh_max_drawdown
                        logger.debug(f"Buy&Hold 最大回撤: {bh_max_drawdown:.2%}")

                        # 條件風險價值
                        bh_cvar_95 = calculate_cvar(bh_returns, alpha=0.05)
                        buy_hold_metrics["條件風險價值 (CVaR 95%)"] = bh_cvar_95
                        logger.debug(f"Buy&Hold 條件風險價值 (CVaR 95%): {bh_cvar_95:.2%}")
                    else:
                        logger.warning("警告：Buy&Hold 的收益率序列為空。")
                else:
                    logger.warning("無法計算 Buy&Hold 指標，起始價格無效 (<=0)。")
            else:
                logger.warning("無法計算 Buy&Hold 指標，用於計算的 DataFrame 為空。")
        except Exception as e:
            logger.exception(f"計算 Buy&Hold 指標時發生錯誤: {e}")
    else:
        logger.warning("缺少計算 Buy&Hold 指標所需的數據 (df_test 或 Close)。")

    logger.info("績效指標計算完成。")
    return strategy_metrics, buy_hold_metrics


# -----------------------------------------------------------------------------
# 10. 視覺化
# -----------------------------------------------------------------------------
def plot_trades(df_plot: pd.DataFrame,
               portfolio_values: pd.Series,
               trade_log_df: pd.DataFrame,
               save_path: str = Config.CHART_SAVE_PATH,
               logger: Optional[logging.Logger] = None) -> None:
    """使用 Plotly 繪製包含交易點和 Buy&Hold 基準的互動式圖表"""
    if logger is None:
        logger = logging.getLogger("trading.plot_trades")

    logger.info(f"開始生成交易結果圖表，儲存至: {save_path}")

    # 檢查數據是否可用
    if (df_plot is None or df_plot.empty) and (portfolio_values is None or portfolio_values.empty):
        logger.error("錯誤 (plot_trades): df_plot 和 portfolio_values 均為空。")
        return

    if portfolio_values is None or portfolio_values.empty:
        logger.warning("警告 (plot_trades)：投資組合價值序列為空，圖表將不完整。")

    # 創建子圖
    fig = make_subplots(
        rows=2,           # 建立 2 行的子圖（上下排列）
        cols=1,           # 每行 1 列，所以總共 2 個子圖
        shared_xaxes=True,      # 共享 X 軸，方便對齊時間軸
        row_heights=[0.7, 0.3],   # 上方子圖佔整體高度的 70%，下方子圖佔 30%
        vertical_spacing=0.1,    # 子圖間垂直間距（單位為相對比例）0.05
        subplot_titles=(f'{Config.STOCK_TICKER} 交易策略回測 (Backtesting of Trading Strategy)', '投資組合淨值 (Net Portfolio Value)')
    )

    # 添加價格線
    if df_plot is not None and not df_plot.empty and 'Close' in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot['Close'],
                mode='lines',
                name='Close'
            ),
            row=1,
            col=1
        )
        logger.debug("已添加 Close 價格線到圖表。")
    else:
        logger.warning("警告 (plot_trades): 無法繪製 Close 價格線。")

    # 添加交易點
    if trade_log_df is not None and not trade_log_df.empty:
        try:
            trade_log_df['date'] = pd.to_datetime(trade_log_df['date'])

            # 買入點
            buy_trades = trade_log_df[trade_log_df['type'] == 'BUY']
            if not buy_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_trades['date'],
                        y=buy_trades['price'],
                        mode='markers',
                        marker=dict(color='green', size=10, symbol='triangle-up'),
                        name='買入 (Buy)',
                        hoverinfo='text',
                        hovertext=[
                            f"買入 {s:,.0f} 股 @ {p:,.2f}<br>成本: {abs(c):,.2f}"
                            for s, p, c in zip(buy_trades['shares'], buy_trades['price'], buy_trades['cost/proceeds'])
                        ]
                    ),
                    row=1,
                    col=1
                )
                logger.debug(f"已添加 {len(buy_trades)} 個買入點。")

            # 賣出點
            sell_trades = trade_log_df[trade_log_df['type'] == 'SELL']
            if not sell_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_trades['date'],
                        y=sell_trades['price'],
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='triangle-down'),
                        name='賣出 (Sell)',
                        hoverinfo='text',
                        hovertext=[
                            f"賣出 {s:,.0f} 股 @ {p:,.2f}<br>收益: {c:,.2f}"
                            for s, p, c in zip(sell_trades['shares'], sell_trades['price'], sell_trades['cost/proceeds'])
                        ]
                    ),
                    row=1,
                    col=1
                )
                logger.debug(f"已添加 {len(sell_trades)} 個賣出點。")
        except Exception as e:
            logger.exception(f"繪製交易點時發生錯誤: {e}")
    else:
        logger.info("交易記錄為空，圖表上無交易點。")

    # 添加策略淨值曲線
    if portfolio_values is not None and not portfolio_values.empty:
        fig.add_trace(
            go.Scatter(
                x=portfolio_values.index,
                y=portfolio_values,
                mode='lines',
                name='策略淨值 (AI Strategy)',
                line=dict(color='blue', width=2)
            ),
            row=2,
            col=1
        )
        logger.debug("已添加策略淨值曲線。")
    else:
        logger.warning("警告 (plot_trades): 投資組合價值序列為空。")

    # 計算 Buy & Hold 基準
    logger.debug("計算買入持有策略 (Buy & Hold) 基準 (基於 Close)...")
    buy_hold_values = None

    if df_plot is not None and not df_plot.empty and 'Close' in df_plot.columns:
        # 確保索引是 DatetimeIndex
        if not isinstance(df_plot.index, pd.DatetimeIndex):
            try:
                df_plot.index = pd.to_datetime(df_plot.index)
                logger.debug("已將 df_plot 索引轉換為 DatetimeIndex。")
            except Exception as e:
                logger.exception(f"轉換 df_plot 索引為 DatetimeIndex 失敗: {e}")
                df_plot = None

        if df_plot is not None:
            try:
                # 確定 Buy & Hold 開始日期
                if portfolio_values is not None and not portfolio_values.empty:
                    first_eval_date = portfolio_values.index.min()
                    df_bh_calc = df_plot.loc[first_eval_date:]
                else:
                    first_eval_date = df_plot.index[Config.WINDOW_SIZE - 1] if len(df_plot) >= Config.WINDOW_SIZE else df_plot.index[0]
                    df_bh_calc = df_plot.loc[first_eval_date:]

                if not df_bh_calc.empty:
                    start_price_bh = df_bh_calc['Close'].iloc[0]

                    if start_price_bh > 0:
                        # 計算 Buy & Hold 價值
                        buy_hold_values = Config.INITIAL_BALANCE * (df_bh_calc['Close'] / start_price_bh)
                        logger.debug(f"買入持有策略計算完成。最終價值: {buy_hold_values.iloc[-1]:.2f}")

                        # 添加到圖表
                        fig.add_trace(
                            go.Scatter(
                                x=buy_hold_values.index,
                                y=buy_hold_values,
                                mode='lines',
                                name='買入持有 (Buy & Hold)',
                                line=dict(color='grey', dash='dash')
                            ),
                            row=2,
                            col=1
                        )
                        logger.debug("已添加買入持有策略線。")
                    else:
                        logger.warning("無法計算買入持有策略，起始價格無效 (<=0)。")
                else:
                    logger.warning("無法計算買入持有策略，用於計算的 DataFrame 為空。")
            except IndexError:
                logger.error("計算買入持有策略時發生 IndexError。")
            except Exception as e:
                logger.exception(f"計算買入持有策略時發生未知錯誤: {e}")
    else:
        logger.warning("缺少繪製買入持有策略所需的數據 (df_plot 或 Close)。")

    # 設置圖表標題和佈局
    plot_start_date = df_plot.index.min().strftime("%Y-%m-%d") if df_plot is not None and not df_plot.empty else "N/A"
    plot_end_date = df_plot.index.max().strftime("%Y-%m-%d") if df_plot is not None and not df_plot.empty else "N/A"

    fig.update_layout(
        title=f'{Config.STOCK_TICKER} 量化交易策略回測結果 (Backtesting Results of Quantitative Trading Strategy) ({plot_start_date} - {plot_end_date})',
        xaxis_title='日期 (Date)',
        yaxis_title='收盤價 (Closing Price) (USD)',
        yaxis2_title='淨值 (Net Worth) (USD)',
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        legend_title_text='圖例 (Chart)'
    )

    fig.update_yaxes(tickformat=',.0f', row=1, col=1)
    fig.update_yaxes(tickformat=',.0f', row=2, col=1)

    # 儲存圖表
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        fig.write_html(save_path)
        if os.path.exists(save_path):
            logger.info(f"互動式圖表已成功儲存至 {save_path}")
        else:
            logger.error(f"錯誤：聲稱已儲存圖表，但在 {save_path} 找不到檔案！")
    except Exception as e:
        logger.exception(f"儲存圖表時發生錯誤: {e}")

    logger.info("plot_trades 函數執行完畢。")


# -----------------------------------------------------------------------------
# 11. 報告生成
# -----------------------------------------------------------------------------
def generate_report(strategy_metrics: Dict[str, float],
                   buy_hold_metrics: Dict[str, float],
                   save_path: str = Config.REPORT_SAVE_PATH,
                   ticker: str = Config.STOCK_TICKER,
                   train_period: Tuple[str, str] = (Config.TRAIN_START_DATE, Config.TRAIN_END_DATE),
                   test_period: Tuple[str, str] = (Config.TEST_START_DATE, Config.TEST_END_DATE),
                   model_path: str = Config.MODEL_SAVE_PATH,
                   chart_path: str = Config.CHART_SAVE_PATH,
                   trades_path: str = Config.TRADES_SAVE_PATH,
                   logger: Optional[logging.Logger] = None) -> None:
    """生成回測報告"""
    if logger is None:
        logger = logging.getLogger("trading.generate_report")

    logger.info(f"開始生成回測報告，儲存至: {save_path}")

    if not isinstance(strategy_metrics, dict):
        logger.warning("警告 (generate_report): strategy_metrics 不是字典。")
        strategy_metrics = {}

    if not isinstance(buy_hold_metrics, dict):
        logger.warning("警告 (generate_report): buy_hold_metrics 不是字典。")
        buy_hold_metrics = {}

    # 格式化指標值的輔助函數
    def format_metric(value, format_str):
        if value is None or (isinstance(value, (float, np.number)) and np.isnan(value)):
            return "N/A"
        try:
            return format(value, format_str)
        except (TypeError, ValueError):
            logger.warning(f"警告: 格式化指標值 '{value}' 使用格式 '{format_str}' 時失敗。")
            return str(value)

    # 獲取檔案名
    model_filename = os.path.basename(model_path) if model_path else "N/A"
    chart_filename = os.path.basename(chart_path) if chart_path else "N/A"
    trades_filename = os.path.basename(trades_path) if trades_path else "N/A"

    # 生成報告內容
    report_content = f"""
# 量化交易策略回測報告
**股票代碼:** {ticker}
**模型:** 強化學習 ({Config.RL_ALGORITHM.__name__}) + 多因子分析 (含預計算因子)
**模型檔案:** `{model_filename}`

## 數據期間
* **訓練數據:** {train_period[0]} 至 {train_period[1]}
* **回測 (驗證) 數據:** {test_period[0]} 至 {test_period[1]}

## 回測績效指標比較
| 指標名稱                         | 策略 (RL + 多因子) | 買入持有 (Buy & Hold) |
| -------------------------------- | ------------------- | --------------------- |
| 年化報酬率 (Annualized Return)   | {format_metric(strategy_metrics.get('年化報酬率 (Annualized Return)'), '.2%')}   | {format_metric(buy_hold_metrics.get('年化報酬率 (Annualized Return)'), '.2%')}      |
| 夏普比率 (Sharpe Ratio)          | {format_metric(strategy_metrics.get('夏普比率 (Sharpe Ratio)'), '.3f')}     | {format_metric(buy_hold_metrics.get('夏普比率 (Sharpe Ratio)'), '.3f')}        |
| 最大回撤 (Maximum Drawdown)      | {format_metric(strategy_metrics.get('最大回撤 (Maximum Drawdown)'), '.2%')}   | {format_metric(buy_hold_metrics.get('最大回撤 (Maximum Drawdown)'), '.2%')}      |
| 條件風險價值 (CVaR 95%)          | {format_metric(strategy_metrics.get('條件風險價值 (CVaR 95%)'), '.2%')}   | {format_metric(buy_hold_metrics.get('條件風險價值 (CVaR 95%)'), '.2%')}      |
| 年化波動率 (Annualized Volatility)| {format_metric(strategy_metrics.get('年化波動率 (Annualized Volatility)'), '.2%')} | {format_metric(buy_hold_metrics.get('年化波動率 (Annualized Volatility)'), '.2%')}    |
| 勝率 (Win Rate)                  | {format_metric(strategy_metrics.get('勝率 (Win Rate)'), '.2%')}       | N/A                   |
| 總交易次數 (Total Trades)        | {format_metric(strategy_metrics.get('總交易次數 (Total Trades)'), ',')}       | {format_metric(buy_hold_metrics.get('總交易次數 (Total Trades)'), ',')}           |
| 最終投資組合淨值 (Final Net Worth)| {format_metric(strategy_metrics.get('最終淨值 (Final Net Worth)'), ',.2f')} USD | {format_metric(buy_hold_metrics.get('最終淨值 (Final Net Worth)'), ',.2f')} USD |
| 初始資金 (Initial Balance)       | {format_metric(Config.INITIAL_BALANCE, ',.2f')} USD | {format_metric(Config.INITIAL_BALANCE, ',.2f')} USD |

## 交易記錄與圖表
* **詳細交易記錄:** 請參見 `{trades_filename}` (僅適用於策略)
* **互動式交易圖表:** 請參見 `{chart_filename}` (請用瀏覽器開啟，包含 Buy&Hold 比較)
* **詳細執行日誌:** 請參見 `{os.path.basename(Config.LOG_FILE_PATH)}`

## 策略說明
本策略使用強化學習代理 ({Config.RL_ALGORITHM.__name__}) 根據歷史股價 (OHLCV)、技術指標以及**預先計算**的基本面、新聞情緒、產業趨勢和市場風險因子進行交易決策 (買入/賣出/持有)。交易和評估主要基於 **收盤價 ('Close')**。報告中提供了與 **買入持有 (Buy & Hold)** 基準策略的績效比較。

**重要提示:** 因子數據的質量和預測能力直接影響策略表現。

## 風險管理機制
* **停損設置:** {Config.STOP_LOSS_PCT:.0%} (入場價下跌 5% 時強制賣出)
* **停利設置:** {Config.TAKE_PROFIT_PCT:.0%} (入場價上漲 50% 時強制賣出)
* **動態倉位管理:** 基於 ATR ({Config.ATR_PERIOD_FOR_SIZING}日) 和風險比例 ({Config.MAX_POSITION_RISK_PCT:.2%}) 計算最大倉位
* **獎勵塑形:** 包含波動性懲罰和利潤獎勵機制

## 後續優化方向
* **因子質量提升:** 持續改進基本面、情緒、趨勢等因子的計算和分析方法，探索更有效的 Alpha 因子。
* **LLM/MAS 整合:** 探索使用 LLM 自動化因子生成或構建 MAS 進行協同分析與決策。
* **特徵工程:** 探索更多或不同的技術指標、因子組合。
* **RL 環境優化:** 調整獎勵函數、狀態表示。
* **超參數調優:** 擴大 Optuna 試驗次數，嘗試更多超參數組合。
* **風險管理:** 實現更動態、多維度的風險控制。
* **模型集成:** 嘗試集成多個模型。
* **數據源擴展:** 引入更多元的數據。
"""

    # 儲存報告
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        logger.info(f"回測報告已儲存至 {save_path}")
    except Exception as e:
        logger.exception(f"儲存報告時發生錯誤: {e}")

    logger.info("generate_report 函數執行完畢。")


# -----------------------------------------------------------------------------
# 12. Optuna 目標函數定義
# -----------------------------------------------------------------------------
def objective(trial, train_df: pd.DataFrame, test_df: pd.DataFrame, logger: logging.Logger) -> float:
    """Optuna 目標函數，用於尋找最佳超參數組合"""
    logger.info(f"--- Optuna Trial {trial.number} ---")

    # 1. 建議超參數
    hyperparams = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
        'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
    }

    try:
        # 2. 創建環境
        trial_train_env = StockTradingEnv(
            train_df.copy(),
            window_size=Config.WINDOW_SIZE,
            initial_balance=Config.INITIAL_BALANCE,
            main_logger=logger
        )
        trial_train_env = DummyVecEnv([lambda: trial_train_env])

        # 3. 訓練模型 (使用較少的步數)
        model = train_model(
            trial_train_env,
            hyperparameters=hyperparams,
            log_dir=os.path.join(Config.SB3_LOG_DIR, f"trial_{trial.number}"),
            total_timesteps=Config.TRIAL_TRAINING_TIMESTEPS,
            save_path=None,  # 不儲存試驗模型
            logger=logger
        )

        if model is None:
            logger.error(f"Trial {trial.number}: 模型訓練失敗。")
            return -np.inf

        # 4. 評估模型
        trade_log_df, portfolio_values = evaluate_model(
            test_df.copy(),
            model=model,  # 直接傳遞模型對象
            initial_balance=Config.INITIAL_BALANCE,
            window_size=Config.WINDOW_SIZE,
            save_trades_path=None,  # 不儲存試驗交易
            logger=logger
        )

        if portfolio_values is None or portfolio_values.empty:
            logger.error(f"Trial {trial.number}: 模型評估失敗或未產生結果。")
            return -np.inf

        # 5. 計算目標指標 (年化報酬率)
        strategy_metrics, _ = calculate_metrics(
            portfolio_values,
            test_df,
            trade_log_df,
            initial_balance=Config.INITIAL_BALANCE,
            logger=logger
        )

        annual_return = strategy_metrics.get("年化報酬率 (Annualized Return)", np.nan)

        if np.isnan(annual_return):
            logger.error(f"Trial {trial.number}: 無法計算年化報酬率。")
            return -np.inf

        logger.info(f"Trial {trial.number}: 年化報酬率 = {annual_return:.2%}")
        return annual_return  # Optuna 預設最大化此值

    except Exception as e:
        logger.exception(f"Trial {trial.number}: 執行過程中發生未預期的錯誤: {e}")
        raise optuna.exceptions.TrialPruned(f"Trial failed due to exception: {e}")


# -----------------------------------------------------------------------------
# 13. 主執行流程
# -----------------------------------------------------------------------------
def main():
    """主執行流程，整合所有功能"""
    # 設置日誌記錄器
    logger = setup_logger()

    logger.info("="*50)
    logger.info("開始執行主流程...")
    logger.info(f"時間: {datetime.datetime.now()}")
    logger.info(f"股票代碼: {Config.STOCK_TICKER}")
    logger.info(f"訓練期間: {Config.TRAIN_START_DATE} - {Config.TRAIN_END_DATE}")
    logger.info(f"測試期間: {Config.TEST_START_DATE} - {Config.TEST_END_DATE}")
    logger.info(f"初始資金: {Config.INITIAL_BALANCE:,.0f}")
    logger.info(f"日誌級別: {logging.getLevelName(Config.LOGGING_LEVEL)}")
    logger.info("="*50)

    # 創建必要的目錄
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.SB3_LOG_DIR, exist_ok=True)

    # --- 步驟 1: 載入 OHLCV 和 因子數據 ---
    logger.info(f"--- 步驟 1: 載入 OHLCV 和 因子數據 ---")

    # 載入或下載 OHLCV 數據
    ohlcv_df = load_raw_data(
        Config.STOCK_TICKER,
        Config.TRAIN_START_DATE,
        Config.TEST_END_DATE,
        Config.RAW_DATA_PATH,
        logger
    )

    if ohlcv_df is None:
        logger.critical("無法獲取有效的 OHLCV 數據，程式終止。")
        return

    logger.info(f"OHLCV 數據載入/下載成功，共 {len(ohlcv_df)} 筆。")

    # 載入因子數據
    factor_df = load_factor_data(
        Config.FACTOR_DATA_PATH,
        Config.EXPECTED_FACTOR_COLUMNS,
        logger
    )

    if factor_df is None:
        logger.critical("無法載入有效的因子數據，程式終止。")
        return

    logger.info(f"因子數據載入成功，共 {len(factor_df)} 筆。")

    # --- 步驟 2: 合併數據、計算技術指標並儲存最終數據 ---
    logger.info("--- 步驟 2: 合併數據、計算技術指標並儲存最終數據 ---")

    final_processed_df = merge_and_process_data(
        ohlcv_df,
        factor_df,
        Config.FINAL_PROCESSED_DATA_PATH,
        logger
    )

    if final_processed_df is None or final_processed_df.empty:
        logger.critical("數據合併或處理失敗，程式終止。")
        return

    logger.info(f"最終數據處理完成，共 {len(final_processed_df)} 筆有效數據。")

    # --- 步驟 3: 分割數據 ---
    logger.info("--- 步驟 3: 分割數據 ---")

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    try:
        train_df = final_processed_df.loc[Config.TRAIN_START_DATE:Config.TRAIN_END_DATE].copy()
        test_df = final_processed_df.loc[Config.TEST_START_DATE:Config.TEST_END_DATE].copy()

        logger.info(f"訓練數據: {len(train_df)} 筆, 從 {train_df.index.min().strftime('%Y-%m-%d')} 到 {train_df.index.max().strftime('%Y-%m-%d')}")
        logger.info(f"測試數據: {len(test_df)} 筆, 從 {test_df.index.min().strftime('%Y-%m-%d')} 到 {test_df.index.max().strftime('%Y-%m-%d')}")
    except KeyError as e:
        logger.critical(f"錯誤：分割數據時發生 KeyError: {e}。")
        logger.critical(f"  最終數據索引範圍: {final_processed_df.index.min()} 到 {final_processed_df.index.max()}")
        logger.critical(f"  請求範圍: {Config.TEST_START_DATE} 到 {Config.TEST_END_DATE}")
        return
    except Exception as e:
        logger.exception(f"分割數據時發生未知錯誤: {e}")
        return

    if train_df.empty or test_df.empty:
        logger.critical("錯誤：訓練或測試數據集為空，程式終止。")
        return

    if not train_df.empty and not test_df.empty and train_df.index.max() >= test_df.index.min():
        logger.warning(f"警告：訓練數據結束日期 ({train_df.index.max().strftime('%Y-%m-%d')}) "
                      f"與測試數據開始日期 ({test_df.index.min().strftime('%Y-%m-%d')}) 重疊或過近。")

    # --- 步驟 4: 使用 Optuna 進行超參數優化 ---
    logger.info("--- 步驟 4: 使用 Optuna 進行超參數優化 ---")

    study = optuna.create_study(
        study_name=f"ppo_optimization_{Config.STOCK_TICKER}",
        direction='maximize',
        storage=f"sqlite:///{Config.OPTUNA_DB_PATH}",
        load_if_exists=True
    )

    objective_with_data = partial(objective, train_df=train_df, test_df=test_df, logger=logger)

    try:
        logger.info(f"開始 Optuna 優化，總共嘗試 {Config.N_OPTUNA_TRIALS} 次...")
        study.optimize(objective_with_data, n_trials=Config.N_OPTUNA_TRIALS, timeout=3600*2)  # 2小時超時
        logger.info("Optuna 優化完成。")
    except Exception as e:
        logger.exception(f"Optuna 優化過程中發生錯誤: {e}")

    # 獲取最佳參數
    best_params = None
    try:
        best_params = study.best_params
        best_value = study.best_value
        logger.info(f"找到的最佳年化報酬率 (優化期間): {best_value:.2%}")
        logger.info(f"對應的最佳超參數: {best_params}")
    except ValueError:
        logger.warning("警告：Optuna 未能找到有效的最佳試驗，將使用預設超參數進行最終訓練。")
        best_params = {
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2
        }

    # --- 步驟 5: 使用最佳參數訓練最終模型 ---
    logger.info("--- 步驟 5: 使用最佳參數訓練最終模型 ---")

    final_model = None
    try:
        final_train_env = StockTradingEnv(
            train_df,
            window_size=Config.WINDOW_SIZE,
            initial_balance=Config.INITIAL_BALANCE,
            main_logger=logger
        )
        final_train_env = DummyVecEnv([lambda: final_train_env])
        logger.info("最終訓練環境創建成功。")

        final_model = train_model(
            final_train_env,
            hyperparameters=best_params,
            save_path=Config.MODEL_SAVE_PATH,
            log_dir=os.path.join(Config.SB3_LOG_DIR, "final_best"),
            total_timesteps=Config.TOTAL_TRAINING_TIMESTEPS,
            logger=logger
        )

        if final_model is not None:
            logger.info("最終模型訓練成功。")
        else:
            logger.error("最終模型訓練失敗。")
    except Exception as e:
        logger.exception(f"訓練最終模型時發生嚴重錯誤: {e}")

    # --- 步驟 6: 評估最終模型 ---
    logger.info("--- 步驟 6: 評估最終模型 ---")

    trade_log_df = pd.DataFrame()
    portfolio_values = pd.Series(dtype=np.float64)

    if final_model is not None and os.path.exists(Config.MODEL_SAVE_PATH):
        trade_log_df, portfolio_values = evaluate_model(
            test_df.copy(),
            model_path=Config.MODEL_SAVE_PATH,
            initial_balance=Config.INITIAL_BALANCE,
            window_size=Config.WINDOW_SIZE,
            save_trades_path=Config.TRADES_SAVE_PATH,
            logger=logger
        )
    else:
        logger.warning("最終模型訓練失敗或未儲存，跳過評估。")

    # --- 步驟 7: 計算最終績效指標 ---
    logger.info("--- 步驟 7: 計算最終績效指標 ---")

    strategy_metrics = {}
    buy_hold_metrics = {}

    if portfolio_values is not None and not portfolio_values.empty:
        strategy_metrics, buy_hold_metrics = calculate_metrics(
            portfolio_values,
            test_df,
            trade_log_df,
            initial_balance=Config.INITIAL_BALANCE,
            logger=logger
        )

        # 輸出策略指標
        logger.info("最終回測績效指標 (策略):")
        for k, v in strategy_metrics.items():
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

            logger.info(f"  {k:<35}: {formatted_value}")

        # 輸出買入持有指標
        logger.info("最終回測績效指標 (Buy & Hold):")
        for k, v in buy_hold_metrics.items():
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

            logger.info(f"  {k:<35}: {formatted_value}")
    else:
        logger.warning("最終回測失敗或未產生投資組合價值，無法計算指標。")
        strategy_metrics["總交易次數 (Total Trades)"] = len(trade_log_df) if trade_log_df is not None else 0
        strategy_metrics["最終淨值 (Final Net Worth)"] = Config.INITIAL_BALANCE
        buy_hold_metrics["最終淨值 (Final Net Worth)"] = Config.INITIAL_BALANCE

    # --- 步驟 8: 繪製最終圖表 ---
    logger.info("--- 步驟 8: 繪製最終圖表 ---")

    if portfolio_values is not None and not portfolio_values.empty:
        plot_df = test_df.copy()
        plot_trades(
            plot_df,
            portfolio_values,
            trade_log_df,
            save_path=Config.CHART_SAVE_PATH,
            logger=logger
        )
    else:
        logger.warning("最終回測失敗或未產生投資組合價值，無法繪製圖表。")

    # --- 步驟 9: 生成最終報告 ---
    logger.info("--- 步驟 9: 生成最終報告 ---")

    train_start = train_df.index.min().strftime('%Y-%m-%d') if not train_df.empty else Config.TRAIN_START_DATE
    train_end = train_df.index.max().strftime('%Y-%m-%d') if not train_df.empty else Config.TRAIN_END_DATE
    test_start = test_df.index.min().strftime('%Y-%m-%d') if not test_df.empty else Config.TEST_START_DATE
    test_end = test_df.index.max().strftime('%Y-%m-%d') if not test_df.empty else Config.TEST_END_DATE

    model_filename_for_report = os.path.basename(Config.MODEL_SAVE_PATH) if os.path.exists(Config.MODEL_SAVE_PATH) else "N/A"

    generate_report(
        strategy_metrics,
        buy_hold_metrics,
        save_path=Config.REPORT_SAVE_PATH,
        train_period=(train_start, train_end),
        test_period=(test_start, test_end),
        model_path=model_filename_for_report,
        chart_path=Config.CHART_SAVE_PATH,
        trades_path=Config.TRADES_SAVE_PATH,
        logger=logger
    )

    logger.info("="*50)
    logger.info("量化交易策略開發流程 (含 Optuna 優化) 執行完畢。")
    logger.info(f"所有結果已儲存至 '{Config.DATA_DIR}' 目錄下。")
    logger.info(f"詳細日誌請查看檔案: {Config.LOG_FILE_PATH}")
    logger.info("="*50)


if __name__ == "__main__":
    main()
