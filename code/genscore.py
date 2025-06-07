# -*- coding: utf-8 -*- genscore.py
# -----------------------------------------------------------------------------
# LLM + Multi-Agent 金融因子分數生成框架 (多LLM支援 + 優化提示詞 + V6錯誤修正版)
#
# Author: Edward Cheng (Original Concept), AI Assistant (Modifications)
# Date: 2025-05-28
#
# **重要提示 (V6錯誤修正版)**:
# - Google Trends: 移除了 pytrends 的內部重試設定，以避免 'method_whitelist' 錯誤。
# - Pandas FutureWarning: 修正了對 Series 單元素調用 float() 的警告。
# - NewsAPI 日期限制仍需注意。
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 1. 安裝與導入函式庫
# -----------------------------------------------------------------------------
import yfinance as yf
import pandas as pd
import numpy as np
import os
import datetime
import logging
from tqdm import tqdm
import time
import random
import json
from dotenv import load_dotenv
import re

import openai
from newsapi import NewsApiClient
from fredapi import Fred
from pytrends.request import TrendReq # pytrends
import feedparser
import requests
from bs4 import BeautifulSoup

import anthropic
import google.generativeai as genai
# -----------------------------------------------------------------------------
# 2. 基本設定與日誌記錄器初始化
# -----------------------------------------------------------------------------
LOGGING_LEVEL = logging.INFO
DATA_DIR = "quant_data_real_multillm_prompt_v6_colab" # 更新版本號
LOG_FILE_PATH = os.path.join(DATA_DIR, "factor_generation_real_multillm_prompt_v6_colab.log")

os.makedirs(DATA_DIR, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
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
logger.info("日誌記錄器初始化完成。")

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
FRED_API_KEY = os.environ.get("FRED_API_KEY")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()
LLM_TEMPERATURE = 0.1
LLM_DEFAULT_MAX_TOKENS_RESPONSE = 200

LLM_MODEL_NAME = None
# (LLM_MODEL_NAME 設定邏輯與前一版本相同)
if LLM_PROVIDER == "openai":
    LLM_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")
elif LLM_PROVIDER == "anthropic":
    LLM_MODEL_NAME = os.environ.get("ANTHROPIC_MODEL_NAME", "claude-3-haiku-20240307")
    logger.info(f"設定使用 Anthropic 模型: {LLM_MODEL_NAME}.")
elif LLM_PROVIDER == "google":
    LLM_MODEL_NAME = os.environ.get("GOOGLE_MODEL_NAME", "gemini-1.5-flash-latest")
    logger.info(f"設定使用 Google Gemini 模型: {LLM_MODEL_NAME}.")
else:
    logger.error(f"不支援的 LLM_PROVIDER: {LLM_PROVIDER}. 將預設嘗試 OpenAI。")
    LLM_PROVIDER = "openai"; LLM_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")
logger.info(f"最終選用 LLM Provider: {LLM_PROVIDER}, Model: {LLM_MODEL_NAME}")


STOCK_TICKER = "MSFT"; YAHOO_TICKER = "MSFT"; COMPANY_NAME = "Microsoft"
INDUSTRY_KEYWORDS = ["Microsoft Azure", "OpenAI Microsoft", "Windows OS", "Microsoft Copilot", "Xbox gaming", "Activision Blizzard", "MSFT earnings", "cloud computing trends", "AI software", "Surface devices", "Satya Nadella strategy"]

DEFAULT_START_DATE = datetime.date(2025, 5, 5)
DEFAULT_END_DATE = datetime.date(2025, 5, 16)

START_DATE = os.environ.get("ANALYSIS_START_DATE", DEFAULT_START_DATE.strftime('%Y-%m-%d'))
END_DATE = os.environ.get("ANALYSIS_END_DATE", DEFAULT_END_DATE.strftime('%Y-%m-%d'))
logger.info(f"腳本設定的目標日期範圍: {START_DATE} 至 {END_DATE}")
# (NewsAPI 日期檢查日誌與前一版本相同)
one_month_ago_from_today = pd.to_datetime(datetime.date.today()) - pd.DateOffset(months=1)
if pd.to_datetime(START_DATE) < one_month_ago_from_today:
    logger.warning(f"警告：腳本設定的開始日期 {START_DATE} 可能超出了 NewsAPI 免費方案的回溯限制（通常為過去一個月）。")
else:
    logger.info(f"腳本設定的日期範圍 {START_DATE} 至 {END_DATE} 可能在 NewsAPI 近期數據範圍內 (視乎API方案)。")


OUTPUT_CSV_PATH = os.path.join(DATA_DIR, f"{STOCK_TICKER}_score_data_ok.csv") # 輸出後複製副本在執行 llmmas.py 前須變更檔名為 score_data_ok.csv

FACTOR_COLUMNS = ['fundamental_score', 'sentiment_score', 'industry_trend_score', 'market_risk_factor', 'black_swan_risk']
API_CALL_DELAY_SECONDS = 3.5

# -----------------------------------------------------------------------------
# 3. LLM 客戶端初始化
# -----------------------------------------------------------------------------
# (LLM 客戶端初始化邏輯與前一版本相同)
openai_client, anthropic_client, google_model_client = None, None, None
if LLM_PROVIDER == "openai":
    if OPENAI_API_KEY: openai_client = openai.OpenAI(api_key=OPENAI_API_KEY); logger.info("OpenAI 客戶端進一步確認初始化成功。")
    else: logger.error("OpenAI API 金鑰仍未設定！")
elif LLM_PROVIDER == "anthropic":
    if ANTHROPIC_API_KEY: anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY); logger.info("Anthropic 客戶端進一步確認初始化成功。")
    else: logger.error("Anthropic API 金鑰仍未設定！")
elif LLM_PROVIDER == "google":
    if GOOGLE_API_KEY:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            model_name_for_google = LLM_MODEL_NAME if LLM_MODEL_NAME.startswith("models/") else f"models/{LLM_MODEL_NAME}"
            if not model_name_for_google.startswith("models/"): model_name_for_google = f"models/{LLM_MODEL_NAME}"
            google_model_client = genai.GenerativeModel(model_name_for_google)
            logger.info(f"Google Gemini 客戶端進一步確認初始化成功 (使用模型: {model_name_for_google})。")
        except Exception as e: logger.error(f"Google Gemini 客戶端進一步確認初始化失敗 for model {LLM_MODEL_NAME} (嘗試 {model_name_for_google}): {e}")
    else: logger.error("Google API 金鑰仍未設定！")

if not NEWS_API_KEY: logger.warning("NewsAPI 金鑰未設定！新聞獲取功能將受限。")
if not FRED_API_KEY: logger.warning("FRED API 金鑰未設定！宏觀數據獲取功能將受限。")

# -----------------------------------------------------------------------------
# 4. 統一的 LLM 調用函數
# -----------------------------------------------------------------------------
# (get_llm_response, extract_score_from_llm_response 函數與前一版本相同)
def get_llm_response(prompt, max_tokens_response=LLM_DEFAULT_MAX_TOKENS_RESPONSE, temperature=LLM_TEMPERATURE):
    logger.debug(f"調用 LLM ({LLM_PROVIDER} - {LLM_MODEL_NAME}). Prompt (前100字): {prompt[:100]}...")
    response_text = None
    client_ready = False
    if LLM_PROVIDER == "openai" and openai_client: client_ready = True
    elif LLM_PROVIDER == "anthropic" and anthropic_client: client_ready = True
    elif LLM_PROVIDER == "google" and google_model_client: client_ready = True

    if not client_ready:
        logger.error(f"LLM 客戶端 ({LLM_PROVIDER}) 未初始化或 API 金鑰缺失。無法進行 API 調用。")
        return None

    try:
        if LLM_PROVIDER == "openai":
            response = openai_client.chat.completions.create(model=LLM_MODEL_NAME, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens_response, temperature=temperature)
            response_text = response.choices[0].message.content.strip()
        elif LLM_PROVIDER == "anthropic":
            response = anthropic_client.messages.create(model=LLM_MODEL_NAME, max_tokens=max_tokens_response, temperature=temperature, messages=[{"role": "user", "content": prompt}])
            if response.content and isinstance(response.content, list) and hasattr(response.content[0], 'text'): response_text = response.content[0].text.strip()
            else: logger.error(f"Anthropic API 返回格式非預期: {response}")
        elif LLM_PROVIDER == "google":
            response = google_model_client.generate_content(prompt, generation_config=genai.types.GenerationConfig(candidate_count=1, max_output_tokens=max_tokens_response, temperature=temperature))
            if hasattr(response, 'text') and response.text: response_text = response.text.strip()
            elif response.parts: response_text = "".join([part.text for part in response.parts if hasattr(part, 'text')]).strip()
            else: logger.error(f"Google Gemini API 返回格式非預期或無有效文本: {response}")
    except Exception as e:
        logger.error(f"{LLM_PROVIDER} API ({LLM_MODEL_NAME}) 調用失敗: {e}")

    if response_text: logger.debug(f"LLM ({LLM_PROVIDER}) 返回: {response_text[:100]}...")
    else: logger.warning(f"LLM ({LLM_PROVIDER}) 未返回有效內容。")
    return response_text

def extract_score_from_llm_response(score_text, default_score=0.0, scale_min=-1.0, scale_max=1.0):
    if score_text is not None:
        match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", score_text)
        if match:
            try:
                score = float(match.group(0))
                return np.clip(score, scale_min, scale_max)
            except ValueError:
                logger.error(f"LLM ({LLM_PROVIDER}) 提取文本 '{match.group(0)}' 但轉換為數字失敗。原始回應: '{score_text[:200]}'")
                return default_score
        else:
            logger.error(f"LLM ({LLM_PROVIDER}) 未能在回應中找到可解析的數字: '{score_text[:200]}'")
            return default_score
    return default_score
# -----------------------------------------------------------------------------
# 5. 數據獲取與分析函數
# -----------------------------------------------------------------------------
# --- 5.1 新聞與情緒分析 ---
# (fetch_news_from_newsapi, analyze_text_sentiment_llm 函數與前一版本 (V5) 相同)
def fetch_news_from_newsapi(query, from_date_str, to_date_str, page_size=20):
    if not NEWS_API_KEY: logger.warning("NewsAPI 金鑰未提供..."); return []
    logger.info(f"NewsAPI 獲取 '{query}' ({from_date_str} to {to_date_str})...")
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        all_articles = newsapi.get_everything(
            q=query,
            from_param=from_date_str,
            to=to_date_str,
            language='en',
            sort_by='relevancy',
            page_size=page_size
        )
        articles = all_articles.get('articles', [])
        logger.info(f"NewsAPI 返回 {len(articles)} 篇。")
        return articles
    except Exception as e:
        logger.error(f"NewsAPI 獲取 '{query}' 失敗: {e}")
        if isinstance(e, Exception) :
            try:
                msg_details = str(e)
                if hasattr(e, 'get_message') and callable(e.get_message):
                     msg_details = e.get_message()
                elif hasattr(e, 'args') and e.args:
                     msg_details = str(e.args[0])

                if "too far in the past" in msg_details or "maximum age" in msg_details or "dateMismatch" in msg_details:
                    logger.error(f"NewsAPI 錯誤提示日期超出範圍或格式問題。請檢查您的API方案和請求日期。詳細: {msg_details}")
            except:
                pass
        return []

def analyze_text_sentiment_llm(text_content, company_context):
    if not LLM_MODEL_NAME: logger.warning(f"LLM 模型未設定。跳過情緒分析。"); return 0.0
    if not text_content or not isinstance(text_content, str) or len(text_content.strip()) < 10: logger.debug("情緒分析文本過短或為空..."); return 0.0
    prompt = f"""As a seasoned Wall Street quantitative fund manager with 30 years of experience at a firm like Blackstone, meticulously analyze the sentiment of the following financial text concerning {company_context}.
Your focus is on identifying alpha-generating signals or significant risk factors that could impact high-frequency trading algorithms or short-term portfolio adjustments.
Disregard generic statements; pinpoint sentiment that implies actionable, quantifiable market impact, considering potential effects on volatility or momentum.
Consider the source's likely credibility and typical market reaction to such news.
Your final line of response MUST BE the score and nothing else.
Return ONLY a single floating-point number between -1.0 (strong sell signal / high risk, likely to trigger algorithmic selling) and 1.0 (strong buy signal / significant opportunity, likely to trigger algorithmic buying).
A score of 0.0 indicates no discernible tradable edge, purely factual information, or neutral impact on quantitative models.
Text: "{text_content[:2500]}"
Sentiment Score (for quantitative trading model input):"""
    score_text = get_llm_response(prompt, max_tokens_response=15, temperature=LLM_TEMPERATURE)
    return extract_score_from_llm_response(score_text, default_score=0.0, scale_min=-1.0, scale_max=1.0)

# --- 5.2 基本面分析 ---
# (fetch_yfinance_fundamentals, analyze_fundamentals_llm 函數與前一版本 (V5) 相同)
def fetch_yfinance_fundamentals(ticker_symbol):
    logger.info(f"yfinance 獲取 {ticker_symbol} 基本面...")
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        info = ticker_obj.info
        def process_financial_dict_key(fin_dict_raw_key):
            if fin_dict_raw_key is None: return None
            return fin_dict_raw_key.isoformat() if isinstance(fin_dict_raw_key, pd.Timestamp) else str(fin_dict_raw_key)
        financials_raw = ticker_obj.financials.transpose().to_dict(orient='index')
        quarterly_financials_raw = ticker_obj.quarterly_financials.transpose().to_dict(orient='index')
        latest_financials_year_date_str = None
        if financials_raw: latest_financials_year_date_str = process_financial_dict_key(next(iter(financials_raw)))
        latest_financials_quarter_date_str = None
        if quarterly_financials_raw: latest_financials_quarter_date_str = process_financial_dict_key(next(iter(quarterly_financials_raw)))
        key_fundamentals = {"marketCap": info.get("marketCap"),"enterpriseValue": info.get("enterpriseValue"),"trailingPE": info.get("trailingPE"),"forwardPE": info.get("forwardPE"),"profitMargins": info.get("profitMargins"),"returnOnEquity": info.get("returnOnEquity"),"revenueGrowth": info.get("revenueGrowth"),"earningsGrowth": info.get("earningsQuarterlyGrowth"),"debtToEquity": info.get("debtToEquity"),"shortRatio": info.get("shortRatio"),"beta": info.get("beta"),"sector": info.get("sector"),"industry": info.get("industry"),"longBusinessSummary": info.get("longBusinessSummary", "")[:1000],"latest_financials_year_date": latest_financials_year_date_str,"latest_financials_quarter_date": latest_financials_quarter_date_str,}
        logger.info(f"成功獲取 {ticker_symbol} yfinance 基本面。")
        return key_fundamentals
    except Exception as e: logger.error(f"yfinance 獲取 {ticker_symbol} 基本面失敗: {e}"); return {}

def analyze_fundamentals_llm(company_name, financial_data_dict):
    if not LLM_MODEL_NAME: logger.warning(f"LLM 模型未設定。跳過基本面分析。"); return 0.5
    if not financial_data_dict: logger.warning(f"{company_name} 基本面數據為空..."); return 0.5
    relevant_data_str = json.dumps({ k: v for k, v in financial_data_dict.items() if v is not None and not isinstance(v, (dict, list)) or k == "longBusinessSummary" }, indent=2, default=str)
    prompt = f"""Drawing upon your 30 years of experience as a quantitative portfolio manager at a leading Wall Street firm like Blackstone, critically evaluate the fundamental strength and alpha potential of {company_name} using the provided financial data.
Your objective is to derive a score reflecting the company's intrinsic value and its capacity for sustained outperformance, identifying any mispricing opportunities or hidden risks not yet reflected in market consensus. Analyze:
1. Profitability & Efficiency: Scrutinize profit margins (gross, operating, net), ROE, ROA, ROIC. Assess their sustainability and competitive advantages.
2. Valuation: Compare key multiples (P/E, P/S, P/B, EV/EBITDA, FCF yield) against historical averages, industry peers, and growth prospects. Is it under/overvalued?
3. Growth Trajectory: Evaluate revenue and earnings growth rates (historical and forward-looking). Assess the quality and durability of this growth (organic vs. inorganic, market share gains, TAM penetration).
4. Financial Health & Risk: Examine leverage (Debt/Equity, Debt/EBITDA), liquidity (Current/Quick Ratios), cash flow generation (Operating, Free Cash Flow), and any potential covenant risks.
5. Competitive Moat & Business Summary: Assess the company's competitive advantages, market position, and any strategic insights from its business summary.
Your final line of response MUST BE the score and nothing else. Synthesize these factors into a single score between 0.0 (critically flawed, high-risk, potential short) and 1.0 (exceptionally robust, high alpha potential, strong long candidate).
Data: {relevant_data_str[:3500]}
Quantitative Fundamental Score (0.0 to 1.0):"""
    score_text = get_llm_response(prompt, max_tokens_response=15, temperature=LLM_TEMPERATURE)
    return extract_score_from_llm_response(score_text, default_score=0.5, scale_min=0.0, scale_max=1.0)

# --- 5.3 產業趨勢分析 ---
def fetch_google_trends_data(keywords_list, timeframe='today 1-m'):
    if not keywords_list: return None
    logger.info(f"Google Trends 獲取 {keywords_list} (timeframe: {timeframe})...")
    try:
        # 修正：從 TrendReq 中移除 retries 和 backoff_factor，以避免 method_whitelist 錯誤
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25))
        pytrends.build_payload(keywords_list, cat=0, timeframe=timeframe, geo='', gprop='')
        trends_df = pytrends.interest_over_time()
        if trends_df.empty: logger.warning(f"Google Trends for {keywords_list} 返回空數據。"); return None
        if 'isPartial' in trends_df.columns: trends_df = trends_df.drop(columns=['isPartial'])
        logger.info(f"成功獲取 Google Trends for {keywords_list}。")
        return trends_df
    except Exception as e:
        logger.error(f"Google Trends 獲取失敗 for {keywords_list}: {e}")
        # 檢查是否為 urllib3.exceptions.MaxRetryError 或 requests.exceptions.RetryError
        # 如果 pytrends 內部仍然嘗試使用舊的 Retry 方式，這個錯誤可能還是會出現
        # 但移除 TrendReq 的 retries 參數是第一步
        if "Retry.__init__() got an unexpected keyword argument 'method_whitelist'" in str(e):
             logger.error("Google Trends 仍然遇到 'method_whitelist' 錯誤。這可能表示 pytrends 版本與 urllib3 不兼容。請嘗試更新 pytrends 或調整 urllib3 版本。")
        elif "response with code 429" in str(e).lower():
            logger.warning("Google Trends 429錯誤: 請求過於頻繁。請增加延遲或減少請求頻率。")
        return None

# (analyze_industry_trend_llm 函數與前一版本 (V5) 相同)
def analyze_industry_trend_llm(industry_name, company_name, industry_news_titles, trends_data_summary):
    if not LLM_MODEL_NAME: logger.warning(f"LLM 模型未設定。跳過產業趨勢分析。"); return 0.5
    news_str = " ".join(industry_news_titles[:10])
    prompt = f"""Leveraging your three decades of experience as a quantitative strategist at a major Wall Street institution like Blackstone, assess the prevailing trend, its momentum, and potential inflection points for the {industry_name} industry, particularly as it pertains to companies like {company_name}.
Your analysis should feed into dynamic sector allocation models and identify thematic investment opportunities or risks. Focus on:
1. Trend Strength & Momentum: Is the trend accelerating, decelerating, or stable? Are there leading indicators from news or Google Trends suggesting shifts?
2. Key Drivers & Catalysts: What are the current macroeconomic, technological, regulatory, or competitive factors driving the trend? Identify any recent catalysts from news.
3. Investor Sentiment: Gauge overall investor sentiment towards the industry based on news flow and search trends.
4. Relative Strength: Implicitly consider if this industry is outperforming/underperforming broader market or related sectors.
Your final line of response MUST BE the score and nothing else.
Provide a score between 0.0 (strong deteriorating trend, significant headwinds, underweight recommendation) and 1.0 (strong accelerating positive trend, significant tailwinds, overweight recommendation).
A score of 0.5 indicates a neutral, mixed, or non-trending environment.
Recent Industry-Relevant News Headlines (max 10): {news_str[:1500]}
Latest Google Trends Summary for relevant keywords (if available): {str(trends_data_summary)[:1000]}
Quantitative Industry Trend & Momentum Score (0.0 to 1.0):"""
    score_text = get_llm_response(prompt, max_tokens_response=15, temperature=LLM_TEMPERATURE)
    return extract_score_from_llm_response(score_text, default_score=0.5, scale_min=0.0, scale_max=1.0)

# --- 5.4 市場風險分析 ---
# (fetch_fred_macro_data_series, fetch_vix_data_daily, analyze_market_risk_llm 函數與前一版本 (V5) 相同)
def fetch_fred_macro_data_series(series_ids_list, start_date_str, end_date_str):
    if not FRED_API_KEY: logger.warning("FRED API 金鑰未提供..."); return None
    if not series_ids_list: return None
    logger.info(f"FRED 獲取宏觀數據: {series_ids_list} ({start_date_str} to {end_date_str})")
    try:
        fred = Fred(api_key=FRED_API_KEY); data_frames = []
        for series_id in series_ids_list:
            s_start = (pd.to_datetime(start_date_str) - datetime.timedelta(days=90)).strftime('%Y-%m-%d')
            series_data = fred.get_series(series_id, observation_start=s_start, observation_end=end_date_str)
            data_frames.append(series_data.rename(series_id)); time.sleep(0.3)
        if not data_frames: return None
        df = pd.concat(data_frames, axis=1); df = df.ffill().dropna()
        logger.info(f"成功從 FRED 獲取宏觀數據: {list(df.columns)}。")
        return df
    except Exception as e: logger.error(f"FRED 獲取數據失敗 for {series_ids_list}: {e}"); return None

def fetch_vix_data_daily(start_date_str, end_date_str):
    logger.info(f"獲取 VIX 指數 ({start_date_str} to {end_date_str})...")
    try:
        if pd.to_datetime(start_date_str) >= pd.to_datetime(end_date_str):
            logger.warning(f"VIX 下載：開始日期 {start_date_str} 不早於結束日期 {end_date_str}。將調整開始日期。")
            start_date_str = (pd.to_datetime(end_date_str) - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            if pd.to_datetime(start_date_str) >= pd.to_datetime(end_date_str): # 如果仍然無效 (例如只差一天)
                 start_date_str = (pd.to_datetime(end_date_str) - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
                 # 確保 start_date 真的在 end_date 之前
                 if pd.to_datetime(start_date_str) >= pd.to_datetime(end_date_str):
                     logger.error(f"無法為VIX下載設定有效的日期範圍: {start_date_str} to {end_date_str}")
                     return None
        vix = yf.download('^VIX', start=start_date_str, end=end_date_str, progress=False, auto_adjust=True)
        if vix.empty: logger.warning(f"VIX 數據下載為空 for range {start_date_str} to {end_date_str}。"); return None
        logger.info(f"成功獲取 VIX 數據，共 {len(vix)} 筆。")
        return vix[['Close']]
    except Exception as e: logger.error(f"獲取 VIX 數據失敗: {e}"); return None

def analyze_market_risk_llm(macro_data_summary, vix_latest, market_news_titles):
    if not LLM_MODEL_NAME: logger.warning(f"LLM 模型未設定。跳過市場風險分析。"); return 0.5
    news_str = " ".join(market_news_titles[:10])
    prompt = f"""As a highly experienced quantitative risk manager from a Blackstone-caliber firm, provide a concise, actionable assessment of the current overall market risk level.
This assessment is critical for adjusting portfolio beta, determining optimal cash allocation, and implementing hedging overlays. Consider:
1. Macroeconomic Climate: Analyze key indicators (e.g., yield curve inversion, inflation trajectory, PMI, credit spreads) for signs of stress or overheating.
2. Market Volatility Regime: Evaluate VIX levels, its term structure (contango/backwardation), and implied vs. realized volatility. Are we in a low-vol or high-vol regime?
3. Intermarket Correlations: Note any significant shifts in correlations between major asset classes.
4. Sentiment & Liquidity: Gauge broad market sentiment from news and consider potential liquidity conditions.
Your final line of response MUST BE the score and nothing else.
Synthesize these into a risk score between 0.0 (extremely benign, 'risk-on' encouraged, minimal hedging) and 1.0 (extremely high risk, 'risk-off' imperative, maximum hedging/de-leveraging).
A score of 0.5 represents a typical, balanced market risk environment.
Latest Macro Data Summary: {str(macro_data_summary)[:1500]}
Latest VIX Close and its recent trend: {vix_latest if vix_latest is not None else 'N/A'}
Recent General Market News Summary (highlighting risk factors): {news_str[:2000]}
Quantitative Market Risk Factor (0.0 to 1.0):"""
    score_text = get_llm_response(prompt, max_tokens_response=15, temperature=LLM_TEMPERATURE)
    return extract_score_from_llm_response(score_text, default_score=0.5, scale_min=0.0, scale_max=1.0)

# --- 5.5 黑天鵝風險分析 ---
# (analyze_black_swan_event_llm 函數與前一版本 (V5) 相同)
def analyze_black_swan_event_llm(date_obj, company_name, company_specific_news, general_market_news, vix_value, stock_daily_change_pct):
    if not LLM_MODEL_NAME: logger.warning(f"LLM 模型未設定。跳過黑天鵝風險分析。"); return 0.01
    news_summary = f"Company ({company_name}) News: " + " ".join([n.get('title', '') for n in company_specific_news[:3]]) + \
                   f" | General Market News: " + " ".join([n.get('title', '') for n in general_market_news[:3]])
    heuristic_flag = ""
    if vix_value is not None and isinstance(vix_value, (int, float)) and vix_value > 40:
        heuristic_flag += " Extremely High VIX (>40)."
    stock_change_display = 'N/A'
    if stock_daily_change_pct is not None and isinstance(stock_daily_change_pct, (int, float)):
        stock_change_display = f"{stock_daily_change_pct*100:.2f}{'%'}"
        if abs(stock_daily_change_pct) > 0.12:
             heuristic_flag += f" Extreme Stock Price Change for {company_name} ({stock_change_display})."
    prompt = f"""Harnessing your 30-year experience in navigating Wall Street's complexities, including multiple crises, at a premier institution like Blackstone, assess the *imminent* likelihood of a 'black swan' event materializing.
This is not about general market volatility but about truly unforeseen, high-impact, and systemic risks that current quantitative models may not adequately capture.
Analyze the provided information for {date_obj.strftime('%Y-%m-%d')} concerning {company_name} or the broader market. Focus on:
1. Anomalous Signals: Any faint or unusual signals in news (geopolitical, systemic tech failures, pandemics, unprecedented policy shifts) or market data.
2. Contagion Risk: Potential for localized shocks to cascade into systemic crises.
3. Unmodelable Unknowns: Are there narratives emerging that point to risks outside the realm of standard financial modeling?
Your final line of response MUST BE the score and nothing else.
Output a score between 0.0 (negligible black swan risk) and 1.0 (credible signals of an impending black swan event).
Maintain EXTREME conservatism: high scores (>0.5) should only be assigned if there's exceptionally strong, novel, and corroborated evidence pointing towards a potential systemic discontinuity. A typical score should be very close to 0.0 (e.g., 0.0 to 0.05).
News Summary (Company & Market): {news_summary[:2500]}
Market Indicators: VIX: {vix_value if vix_value is not None else 'N/A'}, Stock Daily Change % for {company_name}: {stock_change_display}. Heuristics: {heuristic_flag if heuristic_flag else 'None'}
Quantitative Black Swan Imminence Score (0.0 to 1.0, with extreme caution for scores > 0.1):"""
    score_text = get_llm_response(prompt, max_tokens_response=15, temperature=0.05)
    extracted_score = extract_score_from_llm_response(score_text, default_score=0.01, scale_min=0.0, scale_max=1.0)
    if extracted_score > 0.2: logger.warning(f"LLM ({LLM_PROVIDER}) 指出高黑天鵝風險 ({extracted_score:.4f}) for {date_obj.strftime('%Y-%m-%d')}.")
    return extracted_score
# -----------------------------------------------------------------------------
# 6. Multi-Agent 協作流程
# -----------------------------------------------------------------------------
def run_multi_agent_factor_generation(date_obj, ticker_data_for_day):
    logger.info(f"--- [Log V6 - GoogleTrends/Float Fix] 開始因子生成 for {date_obj.strftime('%Y-%m-%d')} for {STOCK_TICKER} ---")
    date_str = date_obj.strftime('%Y-%m-%d')
    news_to_date_str = date_str
    news_from_date_str = (date_obj - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
    if pd.to_datetime(news_from_date_str) > pd.to_datetime(news_to_date_str):
        news_from_date_str = (pd.to_datetime(news_to_date_str) - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        logger.warning(f"NewsAPI from_date ({news_from_date_str}) 調整為早於或等於 to_date ({news_to_date_str})")

    scores = {factor: 0.5 for factor in FACTOR_COLUMNS}
    raw_fundamentals_cache = {}
    company_news, market_news_cache = [], []

    # --- Fundamental Analyst ---
    # (與 V5 版本相同)
    try:
        logger.info("[Agent:Fundamental] 分析...")
        raw_fundamentals = fetch_yfinance_fundamentals(YAHOO_TICKER)
        raw_fundamentals_cache = raw_fundamentals if raw_fundamentals else {}
        time.sleep(API_CALL_DELAY_SECONDS / 2 if raw_fundamentals else 0.1)
        if raw_fundamentals: scores['fundamental_score'] = analyze_fundamentals_llm(COMPANY_NAME, raw_fundamentals)
        else: scores['fundamental_score'] = 0.3
        logger.info(f"[Agent:Fundamental] 完成. Score: {scores['fundamental_score']:.4f}")
    except Exception as e: logger.error(f"[Agent:Fundamental] 失敗: {e}"); scores['fundamental_score'] = 0.3
    time.sleep(API_CALL_DELAY_SECONDS)

    # --- Sentiment Analyst ---
    # (與 V5 版本相同)
    try:
        logger.info("[Agent:Sentiment] 分析...")
        company_query = f'"{COMPANY_NAME}" OR "{STOCK_TICKER}"'
        company_news = fetch_news_from_newsapi(company_query, news_from_date_str, news_to_date_str, page_size=10)
        time.sleep(API_CALL_DELAY_SECONDS / 2 if company_news else 0.1)
        if company_news:
            sentiment_scores_list = []
            for article_idx, article in enumerate(company_news[:3]):
                text_to_analyze = article.get('title', '') + ". " + article.get('description', article.get('summary',''))
                if text_to_analyze.strip() and len(text_to_analyze) > 20 :
                    sentiment_score = analyze_text_sentiment_llm(text_to_analyze, COMPANY_NAME)
                    if sentiment_score is not None: sentiment_scores_list.append(sentiment_score)
                    if article_idx < 2 : time.sleep(API_CALL_DELAY_SECONDS)
            if sentiment_scores_list:
                valid_sentiments = [s for s in sentiment_scores_list if isinstance(s, (int,float))]
                scores['sentiment_score'] = np.mean(valid_sentiments) if valid_sentiments else 0.0
            else: scores['sentiment_score'] = 0.0
        else: logger.info("[Agent:Sentiment] 未找到公司新聞。"); scores['sentiment_score'] = 0.0
        logger.info(f"[Agent:Sentiment] 完成. Score: {scores['sentiment_score']:.4f}")
    except Exception as e: logger.error(f"[Agent:Sentiment] 失敗: {e}"); scores['sentiment_score'] = 0.0
    time.sleep(API_CALL_DELAY_SECONDS)

    # --- Industry Trend Analyst ---
    try:
        logger.info("[Agent:IndustryTrend] 分析...")
        industry_news_query = f"({INDUSTRY_KEYWORDS[0]}) OR ({INDUSTRY_KEYWORDS[7]}) OR (semiconductor industry trends)" # Example
        industry_news_articles = fetch_news_from_newsapi(industry_news_query, news_from_date_str, news_to_date_str, page_size=5)
        time.sleep(API_CALL_DELAY_SECONDS / 2 if industry_news_articles else 0.1)
        industry_news_titles = [n.get('title','') for n in industry_news_articles]

        trends_keywords = [COMPANY_NAME] + INDUSTRY_KEYWORDS[:2]
        trends_df = fetch_google_trends_data(trends_keywords, timeframe=f"{(date_obj - datetime.timedelta(days=30)).strftime('%Y-%m-%d')} {date_str}")
        # 增加 Google Trends 調用後的延遲，即使它失敗了，也避免立即重試或觸發其他 API 的速率問題
        time.sleep(API_CALL_DELAY_SECONDS * 1.5) # 延長此處延遲
        trends_summary = trends_df.iloc[-1].to_dict() if trends_df is not None and not trends_df.empty else "N/A"

        current_industry = raw_fundamentals_cache.get('industry', "Cloud Computing and AI Software")
        scores['industry_trend_score'] = analyze_industry_trend_llm(current_industry, COMPANY_NAME, industry_news_titles, trends_summary)
        logger.info(f"[Agent:IndustryTrend] 完成. Score: {scores['industry_trend_score']:.4f}")
    except Exception as e: logger.error(f"[Agent:IndustryTrend] 失敗: {e}"); scores['industry_trend_score'] = 0.5
    time.sleep(API_CALL_DELAY_SECONDS)

    # --- Market Risk Analyst ---
    vix_close_on_date = None
    try:
        logger.info("[Agent:MarketRisk] 分析...")
        fred_series = ['T10Y2Y', 'VIXCLS', 'SOFR']
        macro_df = fetch_fred_macro_data_series(fred_series, (date_obj - datetime.timedelta(days=60)).strftime('%Y-%m-%d'), date_str)
        time.sleep(API_CALL_DELAY_SECONDS / 2 if macro_df is not None else 0.1)
        macro_summary = "N/A"
        if macro_df is not None and not macro_df.empty:
            relevant_macro_data = macro_df[macro_df.index <= pd.to_datetime(date_str)]
            if not relevant_macro_data.empty: macro_summary = relevant_macro_data.iloc[-1].to_dict()

        vix_start_date = (date_obj - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
        vix_df = fetch_vix_data_daily(vix_start_date, date_str)
        time.sleep(API_CALL_DELAY_SECONDS / 2 if vix_df is not None else 0.1)
        if vix_df is not None and not vix_df.empty:
            # 修正 FutureWarning: 直接使用 .iloc[-1] 的結果，它應該是純量
            if date_str in vix_df.index:
                vix_close_on_date = vix_df.loc[date_str, 'Close']
            elif not vix_df.empty: # 確保 Series 不為空才用 iloc
                vix_close_on_date = vix_df['Close'].iloc[-1]
            # 確保是 float
            if vix_close_on_date is not None: vix_close_on_date = float(vix_close_on_date)

        market_news_query = "global financial markets OR market volatility OR economic recession risk OR interest rate policy"
        market_news_cache = fetch_news_from_newsapi(market_news_query, news_from_date_str, news_to_date_str, page_size=5)
        time.sleep(API_CALL_DELAY_SECONDS / 2 if market_news_cache else 0.1)
        market_news_titles = [n.get('title','') for n in market_news_cache]

        scores['market_risk_factor'] = analyze_market_risk_llm(macro_summary, vix_close_on_date, market_news_titles)
        logger.info(f"[Agent:MarketRisk] 完成. Score: {scores['market_risk_factor']:.4f}")
    except Exception as e: logger.error(f"[Agent:MarketRisk] 失敗: {e}"); scores['market_risk_factor'] = 0.5
    time.sleep(API_CALL_DELAY_SECONDS)

    # --- Black Swan Risk Analyst ---
    # (與 V5 版本相同)
    try:
        logger.info("[Agent:BlackSwan] 分析...")
        stock_daily_change = None
        if ticker_data_for_day:
            open_p, close_p = ticker_data_for_day.get('Open'), ticker_data_for_day.get('Close')
            if open_p and close_p and isinstance(open_p, (int, float)) and isinstance(close_p, (int, float)) and open_p != 0:
                stock_daily_change = (close_p - open_p) / open_p
        scores['black_swan_risk'] = analyze_black_swan_event_llm(date_obj, COMPANY_NAME, company_news, market_news_cache, vix_close_on_date, stock_daily_change)
        logger.info(f"[Agent:BlackSwan] 完成. Score: {scores['black_swan_risk']:.4f}")
    except Exception as e: logger.error(f"[Agent:BlackSwan] 失敗: {e}"); scores['black_swan_risk'] = 0.05

    logger.info(f"--- 因子生成結束 for {date_str}. Scores: {scores} ---")
    return scores
# -----------------------------------------------------------------------------
# 7. 主執行流程
# -----------------------------------------------------------------------------
# (主執行流程與前一版本 (V5) 相同)
if __name__ == "__main__":
    logger.info("="*70)
    logger.info(f"開始生成因子分數 (V6 - GoogleTrends/Float Fix - Provider: {LLM_PROVIDER}, Model: {LLM_MODEL_NAME})...")
    logger.info(f"股票代碼: {STOCK_TICKER}, 公司: {COMPANY_NAME}")
    logger.info(f"腳本目標日期範圍: {START_DATE} 至 {END_DATE}")
    logger.info(f"日誌檔案: {LOG_FILE_PATH}")
    logger.info(f"輸出檔案: {OUTPUT_CSV_PATH}")
    logger.info("="*70)

    llm_ready = (LLM_PROVIDER == "openai" and openai_client) or \
                  (LLM_PROVIDER == "anthropic" and anthropic_client) or \
                  (LLM_PROVIDER == "google" and google_model_client)

    if not llm_ready: logger.critical(f"LLM Provider '{LLM_PROVIDER}' 的 API 金鑰未設定或客戶端初始化失敗。LLM 分析功能無法運作。")
    # (其他 API 金鑰檢查 ...)

    logger.info(f"正在下載 {STOCK_TICKER} 的歷史股價數據 (目標範圍: {START_DATE} to {END_DATE})...")
    try:
        yf_end_date = (pd.to_datetime(END_DATE) + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        stock_history_data = yf.download(YAHOO_TICKER, start=START_DATE, end=yf_end_date, progress=True, auto_adjust=True)

        if stock_history_data.empty:
            logger.error(f"下載 {STOCK_TICKER} 股價數據為空 (請求範圍: {START_DATE} to {yf_end_date})。請檢查代碼、日期範圍（確保是過去的交易日且Yahoo Finance有數據）或網路。程式終止。");
            exit()

        stock_history_data = stock_history_data[(stock_history_data.index >= pd.to_datetime(START_DATE)) & (stock_history_data.index <= pd.to_datetime(END_DATE))]
        if stock_history_data.empty:
            logger.error(f"在 {START_DATE} to {END_DATE} 過濾後，無有效股價數據。程式終止。");
            exit()

        trading_dates = stock_history_data.index
        actual_start_date = stock_history_data.index.min().strftime('%Y-%m-%d')
        actual_end_date = stock_history_data.index.max().strftime('%Y-%m-%d')
        logger.info(f"獲取到 {len(trading_dates)} 個交易日期。yfinance 數據實際範圍: {actual_start_date} 至 {actual_end_date}")

    except Exception as e:
        logger.exception(f"下載 yfinance 股價數據時發生嚴重錯誤: {e}");
        exit()

    results_list = []
    failed_dates_log = []

    logger.info(f"開始為每個交易日循環生成因子分數 (Provider: {LLM_PROVIDER}, Model: {LLM_MODEL_NAME}, 無快取)...")
    for current_date_dt in tqdm(trading_dates, desc=f"生成因子分數 ({LLM_PROVIDER} - 無快取)"):
        try:
            logger.info(f"---+++--- 開始處理日期: {current_date_dt.strftime('%Y-%m-%d')} ---+++---")
            ticker_data_for_current_day = None
            if not stock_history_data.empty and current_date_dt in stock_history_data.index:
                 ticker_data_for_current_day = stock_history_data.loc[current_date_dt].to_dict()

            daily_scores = run_multi_agent_factor_generation(current_date_dt, ticker_data_for_current_day)

            if daily_scores is None or not isinstance(daily_scores, dict) or not all(key in daily_scores for key in FACTOR_COLUMNS):
                logger.error(f"為日期 {current_date_dt.strftime('%Y-%m-%d')} 生成的分數無效或不完整。跳過。")
                failed_dates_log.append({'date': current_date_dt.strftime('%Y-%m-%d'), 'error': 'Invalid or incomplete scores dictionary'})
                continue
            results_list.append({'Date': current_date_dt.strftime('%Y-%m-%d'), **daily_scores})
            logger.debug(f"已成功為日期 {current_date_dt.strftime('%Y-%m-%d')} 生成並記錄分數: {daily_scores}")
        except KeyboardInterrupt:
            logger.warning("接收到用戶中斷請求 (KeyboardInterrupt)。正在停止因子生成...")
            break
        except Exception as e:
            logger.exception(f"處理日期 {current_date_dt.strftime('%Y-%m-%d')} 時發生未預期錯誤: {e}")
            failed_dates_log.append({'date': current_date_dt.strftime('%Y-%m-%d'), 'error': str(e)})
        finally:
            logger.info(f"完成日期 {current_date_dt.strftime('%Y-%m-%d')} 的處理。主要循環延遲...")
            time.sleep(API_CALL_DELAY_SECONDS)

    logger.info("所有日期的因子分數生成循環處理完畢。")

    if failed_dates_log:
        logger.warning(f"因子生成過程中，共有 {len(failed_dates_log)} 個日期處理失敗或不完整。詳情如下:")
        for entry in failed_dates_log: logger.warning(f"  - 日期: {entry['date']}, 錯誤: {entry['error']}")
    else:
        logger.info("所有日期均已成功處理（或按設計跳過）。")

    if not results_list:
        logger.error("未能生成任何有效的因子分數。無法創建 CSV 檔案。")
    else:
        scores_df = pd.DataFrame(results_list)
        try:
            scores_df['Date'] = pd.to_datetime(scores_df['Date'])
            scores_df.set_index('Date', inplace=True)
            logger.info(f"因子分數已成功轉換為 DataFrame ({len(scores_df)} 行)。")
            if not scores_df.empty: logger.debug(f"生成的分數數據預覽 (前3行):\n{scores_df.head(3)}")

            logger.info(f"正在將結果儲存到 {OUTPUT_CSV_PATH}...")
            scores_df.to_csv(OUTPUT_CSV_PATH, encoding='utf-8-sig')
            logger.info(f"因子分數已成功儲存至 {OUTPUT_CSV_PATH}")
        except Exception as e:
            logger.exception(f"處理 DataFrame 或儲存 CSV 檔案時出錯: {e}")

    logger.info("="*70); logger.info(f"因子分數生成流程 (V6 - GoogleTrends/Float Fix - Provider: {LLM_PROVIDER}) 執行完畢。"); logger.info("="*70)
	