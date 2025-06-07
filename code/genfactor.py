# -*- coding: utf-8 -*- genfactor.py
# -----------------------------------------------------------------------------
# LLM + Multi-Agent 金融因子分數生成框架 (擴充模板)
#
# Author: Edward Cheng
# Date: 2025-04-23 (優化版提供日期: 2025-05-27)
#
# 在 Yahoo Finance 的代碼 鴻海："2317.TW" , 中信金："2891.TW" , 台積電 TSMC："2330.TW" ， APPLE："AAPL" ， Amazon："AMZN" ， Microsoft："MSFT"
# 目標:
# 1. 獲取 Microsoft："MSFT" 2000-2024 年的交易日期。
# 2. **提供一個詳細的框架模板**，演示如何整合真實數據源 (yfinance, NewsAPI,
#    FRED, NLTK[有限], Google Trends, RSS, Web Scraping) 和 AI 分析
#    (LLM: gpt-4o, Multi-Agent: CrewAI/AutoGen 概念) 來生成五個核心因子分數。
# 3. 輸出包含 Date 和五個因子分數的 score_data_ok.csv 文件。
#
# **重要提示**:
# - 本代碼是一個需要使用者自行填充實現細節的模板。
# - 不包含實際的 API 金鑰或完整的 Agent 邏輯。
# - 執行真實的 API 調用和 Agent 任務需要配置環境、處理依賴、
#   並可能產生費用和較長的執行時間。
# - Twitter 數據獲取受限。
# - 網頁爬蟲需注意網站條款和反爬機制。
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 1. 安裝與導入函式庫
# -----------------------------------------------------------------------------
# 在 Colab 中取消註解並執行 (根據需要選擇安裝)
# !pip install yfinance pandas numpy tqdm beautifulsoup4 feedparser requests openai newsapi-python fredapi pytrends nltk crewai[tools] pyautogen python-dotenv
# !pip install --upgrade lxml # beautifulsoup 可能需要
# !python -m nltk.downloader vader_lexicon # 下載 NLTK 情緒分析資源 (如果使用)

import yfinance as yf
import pandas as pd
import numpy as np
import os
import datetime
import logging
from tqdm import tqdm # 使用 tqdm 的標準導入方式
import time
import random
import json # 用於處理 API 回應
from functools import lru_cache # 用於緩存 API 或計算結果，減少重複請求

# --- 導入您計劃使用的外部服務庫 (取消註解並配置) ---
# import openai
# from newsapi import NewsApiClient
# from fredapi import Fred
# import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from pytrends.request import TrendReq
# import feedparser
# import requests
# from bs4 import BeautifulSoup
# from crewai import Agent, Task, Crew, Process
# import autogen
# from dotenv import load_dotenv # 用於加載環境變數中的 API KEY

# -----------------------------------------------------------------------------
# 2. 設定與常數
# -----------------------------------------------------------------------------
# --- 日誌設定 ---
LOGGING_LEVEL = logging.INFO
DATA_DIR = "quant_data"
LOG_FILE_PATH = os.path.join(DATA_DIR, "factor_generation_optimized.log") # 修改日誌檔名以區分

# --- 主要設定 ---
'''
STOCK_TICKER = "AAPL"
YAHOO_TICKER = "AAPL"
COMPANY_NAME = "Apple"
INDUSTRY_KEYWORDS = [
    # 核心產品與服務
    "iPhone 16 Pro Max", "iPad Pro", "MacBook Air", "MacBook Pro", "Apple Watch Series 10",
    "AirPods Pro", "Apple Vision Pro", "Apple TV 4K", "HomePod Mini", "Apple Pencil",

    # 軟體與平台
    "iOS 18.1", "macOS 15", "iCloud", "Apple Music", "Apple Arcade",

    # 人工智慧與創新
    "Apple Intelligence", "Siri improvements", "ChatGPT integration", "AI hardware", "OpenAI partnership",

    # 財務與市場表現
    "AAPL stock price", "Apple earnings report", "services revenue", "market capitalization", "share buyback program",

    # 永續發展與企業責任
    "carbon neutral by 2030", "recycled materials", "renewable energy initiatives", "Apple Trade-In program", "environmental sustainability"
]
'''
STOCK_TICKER = "AMZN"
YAHOO_TICKER = "AMZN"
COMPANY_NAME = "Amazon"
INDUSTRY_KEYWORDS = [
    # 核心業務與電商
    "Amazon e-commerce", "Amazon Prime", "Amazon Marketplace", "Amazon Business", "Amazon Fresh",

    # 雲端服務與人工智慧
    "Amazon Web Services", "AWS", "Amazon Bedrock", "Amazon Q", "Amazon CodeWhisperer",

    # 人工智慧與機器人技術
    "Amazon AI initiatives", "Amazon robotics", "Proteus robot", "Sequoia sorting system", "Trainium chips",

    # 物流與配送
    "Amazon fulfillment centers", "Amazon delivery drones", "Amazon electric delivery vehicles", "Amazon supply chain",

    # 財務與市場表現
    "AMZN stock", "Amazon earnings report", "Amazon revenue growth", "Amazon market capitalization", "Amazon stock performance",

    # 永續發展與企業責任
    "The Climate Pledge", "Amazon sustainability initiatives", "Amazon renewable energy", "Amazon carbon neutrality", "Climate Pledge Fund"
]

'''
STOCK_TICKER = "MSFT"
YAHOO_TICKER = "MSFT" # yfinance 使用的代碼
# GOODINFO_TICKER = "MSFT" # GoodInfo 可能有不同的代碼格式，此處僅為示例
COMPANY_NAME = "Microsoft"
INDUSTRY_KEYWORDS = [
    # 核心業務與產品
    "Microsoft 365", "Windows 11", "Surface devices", "Outlook", "OneDrive",
    # 雲端與AI服務
    "Azure cloud", "OpenAI partnership", "Copilot AI", "Power Platform", "GitHub Copilot",
    # 遊戲與娛樂
    "Xbox Series X", "Xbox Game Pass", "Activision Blizzard acquisition", "Cloud gaming", "ZeniMax Media",
    # 財務與市場表現
    "MSFT earnings", "Microsoft stock price", "Satya Nadella", "Microsoft market cap", "Intelligent Cloud revenue",
    # 企業解決方案與產業應用
    "Dynamics 365", "Microsoft Industry Clouds", "Power BI", "Teams collaboration", "LinkedIn integration",
    # 永續發展與社會責任
    "Microsoft sustainability", "carbon negative by 2030", "AI ethics", "cybersecurity initiatives", "digital inclusion"
] # 用於新聞/趨勢搜索
'''
START_DATE = "2000-01-01"
END_DATE = "2024-12-31"

# --- 輸出檔案 ---
OUTPUT_CSV_PATH = os.path.join(DATA_DIR, f"{STOCK_TICKER}_score_data_optimized.csv") # 輸出後複製副本在執行 llmmas.py 前須變更檔名為 score_data_ok.csv

# --- API 金鑰與配置 (!!! 強烈建議使用環境變數 !!!) ---
# load_dotenv() # 從 .env 文件加載環境變數 (需要在 Colab 上傳 .env 文件)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# NEWS_API_KEY = os.getenv("NEWS_API_KEY")
# FRED_API_KEY = os.getenv("FRED_API_KEY")

# --- LLM 設定 ---
LLM_MODEL = "gpt-4o" # 或其他您想使用的模型
LLM_TEMPERATURE = 0.3 # 控制創意程度，分析任務通常較低

# --- 預期因子欄位名稱 ---
FACTOR_COLUMNS = [
    'fundamental_score', 'sentiment_score', 'industry_trend_score',
    'market_risk_factor', 'black_swan_risk'
]

# --- 模擬參數 ---
BLACK_SWAN_DAILY_PROB = 0.0005
SENTIMENT_NOISE = 0.1 # 減少模擬噪音，因為我們期望從真實分析獲得信號
FUNDAMENTAL_NOISE = 0.05
INDUSTRY_NOISE = 0.05
RISK_NOISE = 0.1
BLACK_SWAN_LOW_RISK_MAX = 0.1

# --- 緩存設定 ---
#@lru_cache(maxsize=128)
def cached_api_call(func, *args, **kwargs):
    """通用緩存裝飾器，用於 API 調用或其他耗時計算"""
    logger.debug(f"調用緩存函數: {func.__name__} with args: {args}, kwargs: {kwargs}")
    return func(*args, **kwargs)

# -----------------------------------------------------------------------------
# 3. 設定日誌記錄器
# -----------------------------------------------------------------------------
os.makedirs(DATA_DIR, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

# 清除已存在的 handlers，避免重複記錄 (尤其在 Jupyter 環境中多次執行)
if logger.hasHandlers():
    logger.handlers.clear()

ch = logging.StreamHandler()
ch.setLevel(LOGGING_LEVEL)
ch.setFormatter(formatter)
logger.addHandler(ch)

fh = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
fh.setLevel(LOGGING_LEVEL)
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info("因子生成日誌記錄器設定完成。")
# 檢查 API Key 是否已設置 (移至日誌設定完成後)
# if not OPENAI_API_KEY or not NEWS_API_KEY or not FRED_API_KEY:
#     logger.warning("警告：部分 API 金鑰未在環境變數中設置，相關功能將無法運作。")


# -----------------------------------------------------------------------------
# 4. 數據獲取與分析函數 (詳細模板 - **使用者需填充 API 調用和 Agent 邏輯**)
# -----------------------------------------------------------------------------

# --- 4.1 新聞與情緒分析 ---
# 如果啟用真實 API，可以考慮啟用緩存
def fetch_news_from_newsapi(api_key, query, from_date, to_date, page_size=20):
    if not api_key: logger.warning("NewsAPI 金鑰未設置，跳過 NewsAPI 新聞獲取。"); return []
    logger.info(f"模擬從 NewsAPI 獲取 '{query}' 的新聞 ({from_date} to {to_date})...")
    # --- 實際 NewsAPI 調用邏輯 (需實現) ---
    time.sleep(0.01) # 模擬 API 延遲
    return [{"title": f"Simulated NewsAPI Title {i} for {query}", "description": f"Desc {i}", "publishedAt": datetime.datetime.now().isoformat()} for i in range(random.randint(0,2))]


def fetch_news_from_rss(rss_feeds):
    logger.info(f"模擬從 RSS feeds 獲取新聞: {rss_feeds}")
    articles = []
    # --- 實際 RSS 抓取邏輯 (需實現) ---
    if random.random() < 0.7: # 模擬有時能抓到
        articles.append({'title': 'Simulated RSS Title', 'summary': 'Simulated RSS Summary'})
    return articles


def analyze_sentiment_openai(api_key, text_list):
    if not api_key: logger.warning("OpenAI API 金鑰未設置，跳過 LLM 情緒分析。"); return 0.0
    if not text_list: logger.debug("情緒分析文本列表為空，返回 0.0"); return 0.0
    logger.info(f"模擬使用 OpenAI 分析 {len(text_list)} 條文本的情緒...")
    # --- 實際 OpenAI API 調用邏輯 (需實現) ---
    time.sleep(0.02) # 模擬 API 延遲
    return np.clip(np.random.normal(loc=0.0, scale=SENTIMENT_NOISE), -1, 1)

# --- 4.2 基本面分析 ---

def scrape_goodinfo_fundamentals(ticker_id):
    url = f"https://goodinfo.tw/tw/StockBzPerformance.asp?STOCK_ID={ticker_id}" # 示例 URL
    logger.info(f"模擬從 GoodInfo 爬取基本面數據 (目標: {ticker_id})...")
    # --- 實際爬蟲邏輯 (需實現，且注意網站政策) ---
    data = {"GrossMargin": random.uniform(30, 75), "ROE": random.uniform(10, 50)} # MSFT通常較高
    logger.debug(f"爬取/模擬的基本面數據: {data}")
    return data


def analyze_fundamentals_with_llm(api_key, company_name, financial_data):
    if not api_key: logger.warning("OpenAI API 金鑰未設置，跳過 LLM 基本面分析。"); return 0.5 # 返回中性分數
    if not financial_data: logger.debug("基本面數據為空，返回中性分數 0.5"); return 0.5
    logger.info(f"模擬使用 OpenAI 分析 {company_name} 的基本面數據: {financial_data}")
    # --- 實際 OpenAI API 調用邏輯 (需實現) ---
    time.sleep(0.02)
    # 模擬 Microsoft 這樣的大型科技公司基本面通常較好且穩定增長
    base_fundamental = 0.6 + (datetime.datetime.now().year - 2000) * 0.01 # 調整基線和增長率
    return np.clip(np.random.normal(loc=base_fundamental, scale=FUNDAMENTAL_NOISE), 0, 1)

# --- 4.3 產業趨勢分析 ---

def fetch_trends_pytrends(keywords, timeframe='today 3-m'):
    logger.info(f"模擬獲取 Google Trends 數據 for {keywords}...")
    # --- 實際 pytrends 調用邏輯 (需實現) ---
    time.sleep(0.01)
    data = {kw: [random.randint(40, 100) for _ in range(5)] for kw in keywords} # 科技關鍵字熱度通常較高
    return pd.DataFrame(data, index=pd.date_range(end=datetime.datetime.now(), periods=5))


def analyze_industry_trend_with_llm(api_key, industry_news, trends_data):
    if not api_key: logger.warning("OpenAI API 金鑰未設置，跳過 LLM 產業趨勢分析。"); return 0.5
    logger.info(f"模擬使用 OpenAI 分析產業趨勢...")
    # --- 實際 OpenAI API 調用邏輯 (需實現) ---
    time.sleep(0.02)
    month_cycle_effect = 0.01 * np.sin(2 * np.pi * datetime.datetime.now().dayofyear / 365.25)
    # 科技產業趨勢通常偏向正面，但有波動
    return np.clip(0.55 + month_cycle_effect + np.random.normal(0, INDUSTRY_NOISE), 0, 1)

# --- 4.4 市場風險分析 ---

def fetch_fred_macro_data(api_key, series_ids, start_date, end_date):
    if not api_key: logger.warning("FRED API 金鑰未設置，跳過 FRED 數據獲取。"); return None
    logger.info(f"模擬從 FRED 獲取宏觀數據: {series_ids}")
    # --- 實際 FRED API 調用邏輯 (需實現) ---
    time.sleep(0.01)
    idx = pd.date_range(start=start_date, end=end_date, freq='D')
    data = {sid: np.random.rand(len(idx))*100 for sid in series_ids}
    return pd.DataFrame(data, index=idx)


def fetch_vix_data(start_date, end_date):
    logger.info("模擬獲取 VIX 指數數據...")
    try:
        # 模擬返回 DataFrame 結構，與 yfinance 相似
        idx = pd.date_range(start=start_date, end=end_date, freq='B') # Business days
        if idx.empty: return pd.DataFrame(columns=['Close']) # 如果日期範圍無效
        vix_values = np.random.uniform(10, 30, size=len(idx)) # 模擬 VIX 值
        vix_df = pd.DataFrame(vix_values, index=idx, columns=['Close'])
        logger.debug(f"模擬 VIX 數據生成完畢，共 {len(vix_df)} 筆。")
        return vix_df
    except Exception as e:
        logger.error(f"模擬獲取 VIX 數據失敗: {e}")
        return pd.DataFrame(columns=['Close']) # 返回空的 DataFrame


def analyze_market_risk_with_llm(api_key, macro_data, vix_data, recent_news):
    if not api_key: logger.warning("OpenAI API 金鑰未設置，跳過 LLM 市場風險分析。"); return 0.5
    logger.info("模擬使用 OpenAI 分析市場風險...")
    # --- 實際 OpenAI API 調用邏輯 (需實現) ---
    time.sleep(0.02)
    return np.clip(np.random.normal(loc=0.5, scale=RISK_NOISE), 0, 1)

# --- 4.5 黑天鵝風險評估 (模擬) ---
def assess_black_swan_risk(date_obj, daily_prob=BLACK_SWAN_DAILY_PROB, low_risk_max=BLACK_SWAN_LOW_RISK_MAX):
    """
    模擬黑天鵝風險評估。
    :param date_obj: 當前日期 (datetime.date or datetime.datetime object)
    :param daily_prob: 黑天鵝事件每日發生概率
    :param low_risk_max: 平時低背景風險的最大值
    :return: 黑天鵝風險分數 (0.0 到 1.0)
    """
    if random.random() < daily_prob:
        risk_score = random.uniform(0.5, 1.0) # 事件發生時，風險較高
        logger.warning(f"模擬的黑天鵝事件觸發於 {date_obj.strftime('%Y-%m-%d')}，風險評分: {risk_score:.4f}")
    else:
        risk_score = random.uniform(0, low_risk_max) # 平時的低背景風險
    return risk_score

# --- 4.6 模擬 LLM 與 Multi-Agent 分析函式 (新增的佔位符函式) ---
def simulate_llm_multi_agent_analysis(date_obj, ticker, previous_scores=None):
    """
    模擬 LLM 和 Multi-Agent 分析來生成每日的五個因子分數。
    這是主要的模擬函式，用於替代真實的複雜分析流程。
    :param date_obj: 當前日期 (datetime.datetime object)
    :param ticker: 股票代碼
    :param previous_scores: 前一天的分數 (可選，用於潛在的時序依賴模擬)
    :return: 包含五個因子分數的字典
    """
    logger.info(f"開始為 {date_obj.strftime('%Y-%m-%d')} 的 {ticker} 模擬因子分數...")

    # 1. 基本面分數 (Fundamental Score)
    # 模擬邏輯：假設基本面隨時間緩慢改善，並加入隨機噪聲
    # 參考 analyze_fundamentals_with_llm 模擬部分，但在此處直接計算
    base_fundamental = 0.6 + (date_obj.year - 2000) * 0.01 # Microsoft 基線較高
    fundamental_score = np.clip(np.random.normal(loc=base_fundamental, scale=FUNDAMENTAL_NOISE), 0, 1)
    logger.debug(f"  模擬 Fundamental Score: {fundamental_score:.4f}")

    # 2. 情緒分數 (Sentiment Score)
    # 模擬邏輯：每日隨機波動，中心為0
    # 參考 analyze_sentiment_openai 模擬部分
    sentiment_score = np.clip(np.random.normal(loc=0.0, scale=SENTIMENT_NOISE * 2), -1, 1) # 稍微放大波動範圍
    logger.debug(f"  模擬 Sentiment Score: {sentiment_score:.4f}")

    # 3. 產業趨勢分數 (Industry Trend Score)
    # 模擬邏輯：基礎趨勢上加入年度週期性波動和隨機噪聲
    # 參考 analyze_industry_trend_with_llm 模擬部分
    month_cycle_effect = 0.05 * np.sin(2 * np.pi * date_obj.dayofyear / 365.25)
    base_industry_trend = 0.55 # 科技業趨勢基線
    industry_trend_score = np.clip(base_industry_trend + month_cycle_effect + np.random.normal(0, INDUSTRY_NOISE), 0, 1)
    logger.debug(f"  模擬 Industry Trend Score: {industry_trend_score:.4f}")

    # 4. 市場風險因子 (Market Risk Factor)
    # 模擬邏輯：每日隨機波動，中心為0.5
    # 參考 analyze_market_risk_with_llm 模擬部分
    market_risk_factor = np.clip(np.random.normal(loc=0.4, scale=RISK_NOISE), 0, 1) # 市場風險基線可調整
    logger.debug(f"  模擬 Market Risk Factor: {market_risk_factor:.4f}")

    # 5. 黑天鵝風險 (Black Swan Risk)
    # 使用獨立的 assess_black_swan_risk 函式進行模擬
    black_swan_risk = assess_black_swan_risk(date_obj)
    logger.debug(f"  模擬 Black Swan Risk: {black_swan_risk:.4f}")

    scores = {
        FACTOR_COLUMNS[0]: fundamental_score,
        FACTOR_COLUMNS[1]: sentiment_score,
        FACTOR_COLUMNS[2]: industry_trend_score,
        FACTOR_COLUMNS[3]: market_risk_factor,
        FACTOR_COLUMNS[4]: black_swan_risk
    }

    logger.info(f"為 {date_obj.strftime('%Y-%m-%d')} 的 {ticker} 模擬因子分數完成。")
    return scores


# --- 4.7 Multi-Agent 協作 (概念性，需自行實現詳細 Agent 和 Task) ---
def run_multi_agent_factor_generation(date_obj, ticker_symbol, company_name_str, industry_kw_list, fred_series_ids_list, rss_feeds_list):
    """
    **概念性函數**: 模擬或調用一個完整的多智能體系統 (如 CrewAI 或 AutoGen) 來生成所有因子。
    **需實現**:
        1. 定義各個 Agent (Fundamental, Sentiment, Industry, Risk, BlackSwanDetector, Coordinator)。
        2. 為每個 Agent 定義詳細的 Goal 和 Tools (調用前面實現的數據獲取和 LLM 分析函數)。
        3. 定義 Tasks，將分析流程分解給不同的 Agent。
        4. 設計 Agent 間的協作流程。
        5. 創建 Crew 或 Agent Network 並執行。
        6. 解析最終結果，提取五個因子分數。
    """
    logger.info(f"--- [概念] 開始多智能體分析流程 for {date_obj.strftime('%Y-%m-%d')} ---")

    # --- 此處為 Multi-Agent 框架的預期整合點 ---
    # 例如:
    # Initialize agents and tasks
    # data_collector_agent = Agent(...)
    # sentiment_agent = Agent(...)
    # ... other agents

    # data_collection_task = Task(agent=data_collector_agent, description=f"Collect data for {company_name_str} on {date_obj}")
    # sentiment_task = Task(agent=sentiment_agent, description="Analyze sentiment from news")
    # ... other tasks

    # crew = Crew(agents=[...], tasks=[...], verbose=1)
    # daily_scores_from_crew = crew.kickoff(inputs={'date': date_obj.strftime('%Y-%m-%d'), 'ticker': ticker_symbol, ...})
    # return daily_scores_from_crew
    # --- ------------------------------------ ---

    # 目前，此概念性函數將直接調用上面的 `simulate_llm_multi_agent_analysis` 作為替代方案
    # 在真實應用中，你應該讓 Multi-Agent 系統內部調用各個數據獲取和單一LLM分析函數，然後匯總結果
    logger.warn(f"Multi-Agent 框架 (run_multi_agent_factor_generation) 未完全實現，將使用 simulate_llm_multi_agent_analysis 進行模擬。")

    simulated_scores = simulate_llm_multi_agent_analysis(date_obj, ticker_symbol)

    logger.info(f"--- [概念] 多智能體分析流程結束 for {date_obj.strftime('%Y-%m-%d')} ---")
    return simulated_scores

# -----------------------------------------------------------------------------
# 5. 主執行流程
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("="*60)
    logger.info("開始生成因子分數 (優化版 - 使用模擬數據分析)...")
    logger.info(f"股票代碼: {STOCK_TICKER}")
    logger.info(f"數據期間: {START_DATE} 至 {END_DATE}")
    logger.info(f"日誌檔案: {LOG_FILE_PATH}")
    logger.info(f"輸出檔案: {OUTPUT_CSV_PATH}")
    logger.info("="*60)

    # 1. 獲取交易日期序列
    logger.info(f"正在下載 {STOCK_TICKER} 的歷史數據以獲取交易日期...")
    try:
        stock_data = yf.download(YAHOO_TICKER, start=START_DATE, end=END_DATE, progress=True)
        if stock_data.empty:
            logger.error(f"下載 {STOCK_TICKER} 數據為空，無法獲取交易日期序列。請檢查代碼、日期範圍或網路連線。")
            exit()
        trading_dates = stock_data.index
        logger.info(f"獲取到 {len(trading_dates)} 個交易日期，從 {trading_dates.min().strftime('%Y-%m-%d')} 到 {trading_dates.max().strftime('%Y-%m-%d')}")
    except Exception as e:
        logger.exception(f"下載 yfinance 數據時發生嚴重錯誤: {e}")
        exit()

    # 2. 初始化 Agent 和工具 (如果使用 CrewAI/AutoGen)
    # logger.info("初始化 CrewAI/AutoGen Agents and Tools (佔位符)...")
    # --- 在此處進行 CrewAI/AutoGen 的 Agent 和 Task 定義與初始化 ---

    # 3. 為每個交易日生成因子分數
    logger.info("開始為每個交易日循環生成因子分數...")
    results_list = []
    failed_dates_log = [] # 新增：用於記錄處理失敗的日期

    # 使用 tqdm 顯示進度條
    for current_date in tqdm(trading_dates, desc="生成因子分數進度"):
        try:
            logger.info(f"--- 開始處理日期: {current_date.strftime('%Y-%m-%d')} ---")

            # *** 調用模擬分析函式 ***
            # 在真實應用中，你可能會調用 run_multi_agent_factor_generation(...)
            # 或一個更複雜的協調器函式來調用真實的數據獲取和LLM分析。
            # previous_result = results_list[-1] if results_list else None # 若需要前一天結果
            daily_scores = simulate_llm_multi_agent_analysis(current_date, STOCK_TICKER)

            if daily_scores is None or not isinstance(daily_scores, dict):
                logger.error(f"為日期 {current_date.strftime('%Y-%m-%d')} 生成的分數無效 (None 或非字典)。跳過此日期。")
                failed_dates_log.append({'date': current_date.strftime('%Y-%m-%d'), 'error': 'Invalid scores generated (None or not a dict)'})
                continue

            # 驗證分數是否包含所有預期因子
            if not all(key in daily_scores for key in FACTOR_COLUMNS):
                logger.error(f"為日期 {current_date.strftime('%Y-%m-%d')} 生成的分數缺少預期因子。獲得的鍵: {list(daily_scores.keys())}。跳過此日期。")
                failed_dates_log.append({'date': current_date.strftime('%Y-%m-%d'), 'error': 'Missing factor keys in scores'})
                continue

            results_list.append({
                'Date': current_date.strftime('%Y-%m-%d'),
                **daily_scores # 合併分數字典
            })
            logger.debug(f"已成功為日期 {current_date.strftime('%Y-%m-%d')} 生成並記錄分數。")

        except Exception as e:
            logger.exception(f"處理日期 {current_date.strftime('%Y-%m-%d')} 時發生未預期錯誤: {e}")
            failed_dates_log.append({'date': current_date.strftime('%Y-%m-%d'), 'error': str(e)})
            # 根據需求決定是否要中斷，目前設定為繼續處理下一個日期

        # 模擬真實API呼叫的延遲，避免過快請求 (如果未來啟用真實API)
        # time.sleep(random.uniform(0.1, 0.3)) # 僅在真實API調用時啟用

    logger.info("所有日期的因子分數生成循環處理完畢。")

    # 4. 處理失敗日誌
    if failed_dates_log:
        logger.warning(f"在因子生成過程中，共有 {len(failed_dates_log)} 個日期處理失敗。")
        for failed_entry in failed_dates_log:
            logger.warning(f"  - 日期: {failed_entry['date']}, 錯誤: {failed_entry['error']}")
    else:
        logger.info("所有日期均已成功處理（或按設計跳過）。")

    # 5. 將結果轉換為 DataFrame
    if not results_list:
        logger.error("未能生成任何有效的因子分數。無法創建 CSV 檔案。")
        exit()

    scores_df = pd.DataFrame(results_list)
    try:
        scores_df['Date'] = pd.to_datetime(scores_df['Date'])
        scores_df.set_index('Date', inplace=True)
    except Exception as e:
        logger.exception(f"將結果轉換為 DataFrame 或設定日期索引時出錯: {e}")
        exit()

    logger.info("因子分數已成功轉換為 DataFrame。")
    logger.debug(f"生成的分數數據預覽 (前5行):\n{scores_df.head()}")
    logger.debug(f"生成的分數數據預覽 (後5行):\n{scores_df.tail()}")


    # 6. 儲存為 CSV 檔案
    logger.info(f"正在將結果儲存到 {OUTPUT_CSV_PATH}...")
    try:
        scores_df.to_csv(OUTPUT_CSV_PATH, encoding='utf-8-sig') # utf-8-sig 確保 Excel 正確讀取中文
        logger.info(f"因子分數已成功儲存至 {OUTPUT_CSV_PATH}")
    except Exception as e:
        logger.exception(f"儲存 CSV 檔案時發生錯誤: {e}")

    logger.info("="*60)
    logger.info("因子分數生成流程執行完畢。")
    logger.info("="*60)
	