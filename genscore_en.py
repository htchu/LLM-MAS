# -*- coding: utf-8 -*- genscore.py
# -----------------------------------------------------------------------------
# Author: Edward Cheng
# Date: 2025-05-22
# Version: Enhanced Edition
# -----------------------------------------------------------------------------
"""
Multi-Agent LLM Financial Factor Scoring Framework
==================================================

A comprehensive framework for generating quantitative trading factors using multiple
Large Language Models (LLMs) and various financial data sources.

Features:
- Multi-LLM support (OpenAI, Anthropic, Google)
- 5 specialized analysis agents
- Real-time data integration from multiple sources
- Robust error handling and logging
- Modular, extensible architecture
"""
# -----------------------------------------------------------------------------
# 1. Installing and importing libraries
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
# 2. Basic settings and logger initialization
# -----------------------------------------------------------------------------
LOGGING_LEVEL = logging.INFO
DATA_DIR = "quant_data_real_multillm_prompt_v6_colab" # Update version number
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
logger.info("Logger initialization completed。")

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
# (LLM_MODEL_NAME The setting logic is the same as the previous version)
if LLM_PROVIDER == "openai":
    LLM_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")
elif LLM_PROVIDER == "anthropic":
    LLM_MODEL_NAME = os.environ.get("ANTHROPIC_MODEL_NAME", "claude-3-7-sonnet-20250219")
    logger.info(f"Setting up the Anthropic model: {LLM_MODEL_NAME}.")
elif LLM_PROVIDER == "google":
    LLM_MODEL_NAME = os.environ.get("GOOGLE_MODEL_NAME", "gemini-1.5-flash")
    logger.info(f"Setting up the Google Gemini model: {LLM_MODEL_NAME}.")
else:
    logger.error(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. Try OpenAI by default!")
    LLM_PROVIDER = "openai"; LLM_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")
logger.info(f"Final selection LLM Provider: {LLM_PROVIDER}, Model: {LLM_MODEL_NAME}")


STOCK_TICKER = "MSFT"; YAHOO_TICKER = "MSFT"; COMPANY_NAME = "Microsoft"
INDUSTRY_KEYWORDS = ["Microsoft Azure", "OpenAI Microsoft", "Windows OS", "Microsoft Copilot", "Xbox gaming", "Activision Blizzard", "MSFT earnings", "cloud computing trends", "AI software", "Surface devices", "Satya Nadella strategy"]

DEFAULT_START_DATE = datetime.date(2025, 5, 5)
DEFAULT_END_DATE = datetime.date(2025, 5, 16)

START_DATE = os.environ.get("ANALYSIS_START_DATE", DEFAULT_START_DATE.strftime('%Y-%m-%d'))
END_DATE = os.environ.get("ANALYSIS_END_DATE", DEFAULT_END_DATE.strftime('%Y-%m-%d'))
logger.info(f"Target date range set by script: {START_DATE} to {END_DATE}")
# (NewsAPI Date check log is same as previous version)
one_month_ago_from_today = pd.to_datetime(datetime.date.today()) - pd.DateOffset(months=1)
if pd.to_datetime(START_DATE) < one_month_ago_from_today:
    logger.warning(f"Warning: Script set start date {START_DATE} May exceed NewsAPI Free plan has a look-back limit (usually the past month).")
else:
    logger.info(f"Scripted date range {START_DATE} to {END_DATE} Probably within the NewsAPI recent data range (depending on the API plan).")


OUTPUT_CSV_PATH = os.path.join(DATA_DIR, f"{STOCK_TICKER}_score_data_ok.csv") # After the output is copied, execute "llmmas.py" and use

FACTOR_COLUMNS = ['fundamental_score', 'sentiment_score', 'industry_trend_score', 'market_risk_factor', 'black_swan_risk']
API_CALL_DELAY_SECONDS = 3.5

# -----------------------------------------------------------------------------
# 3. LLM Client initialization
# -----------------------------------------------------------------------------
# (LLM The client initialization logic is the same as the previous version)
openai_client, anthropic_client, google_model_client = None, None, None
if LLM_PROVIDER == "openai":
    if OPENAI_API_KEY: openai_client = openai.OpenAI(api_key=OPENAI_API_KEY); logger.info("OpenAI The client further confirms that the initialization was successful.")
    else: logger.error("OpenAI API Keys not set yet!")
elif LLM_PROVIDER == "anthropic":
    if ANTHROPIC_API_KEY: anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY); logger.info("Anthropic The client further confirms that the initialization was successful.")
    else: logger.error("Anthropic API Keys not set yet!")
elif LLM_PROVIDER == "google":
    if GOOGLE_API_KEY:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            model_name_for_google = LLM_MODEL_NAME if LLM_MODEL_NAME.startswith("models/") else f"models/{LLM_MODEL_NAME}"
            if not model_name_for_google.startswith("models/"): model_name_for_google = f"models/{LLM_MODEL_NAME}"
            google_model_client = genai.GenerativeModel(model_name_for_google)
            logger.info(f"The Google Gemini client further confirms that the initialization was successful (using model: {model_name_for_google}).")
        except Exception as e: logger.error(f"Google Gemini client further confirms initialization failure for model {LLM_MODEL_NAME} (trying {model_name_for_google}): {e}")
    else: logger.error("Google API Keys not set yet!")

if not NEWS_API_KEY: logger.warning("NewsAPI key not set! News retrieval functionality will be limited.")
if not FRED_API_KEY: logger.warning("FRED API key not set! Macro data acquisition functionality will be limited.")

# -----------------------------------------------------------------------------
# 4. Unified LLM call function
# -----------------------------------------------------------------------------
# (get_llm_response, extract_score_from_llm_response The function is the same as the previous version)
def get_llm_response(prompt, max_tokens_response=LLM_DEFAULT_MAX_TOKENS_RESPONSE, temperature=LLM_TEMPERATURE):
    logger.debug(f"Call LLM ({LLM_PROVIDER} - {LLM_MODEL_NAME}). Prompt (first 100 words): {prompt[:100]}...")
    response_text = None
    client_ready = False
    if LLM_PROVIDER == "openai" and openai_client: client_ready = True
    elif LLM_PROVIDER == "anthropic" and anthropic_client: client_ready = True
    elif LLM_PROVIDER == "google" and google_model_client: client_ready = True

    if not client_ready:
        logger.error(f"The LLM client ({LLM_PROVIDER}) is not initialized or the API key is missing. Unable to make API calls.")
        return None

    try:
        if LLM_PROVIDER == "openai":
            response = openai_client.chat.completions.create(model=LLM_MODEL_NAME, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens_response, temperature=temperature)
            response_text = response.choices[0].message.content.strip()
        elif LLM_PROVIDER == "anthropic":
            response = anthropic_client.messages.create(model=LLM_MODEL_NAME, max_tokens=max_tokens_response, temperature=temperature, messages=[{"role": "user", "content": prompt}])
            if response.content and isinstance(response.content, list) and hasattr(response.content[0], 'text'): response_text = response.content[0].text.strip()
            else: logger.error(f"Anthropic API Unexpected return format: {response}")
        elif LLM_PROVIDER == "google":
            response = google_model_client.generate_content(prompt, generation_config=genai.types.GenerationConfig(candidate_count=1, max_output_tokens=max_tokens_response, temperature=temperature))
            if hasattr(response, 'text') and response.text: response_text = response.text.strip()
            elif response.parts: response_text = "".join([part.text for part in response.parts if hasattr(part, 'text')]).strip()
            else: logger.error(f"Google Gemini API The return format is unexpected or there is no valid text: {response}")
    except Exception as e:
        logger.error(f"{LLM_PROVIDER} API ({LLM_MODEL_NAME}) Call failed: {e}")

    if response_text: logger.debug(f"LLM ({LLM_PROVIDER}) Return: {response_text[:100]}...")
    else: logger.warning(f"LLM ({LLM_PROVIDER}) No valid content was returned.")
    return response_text

def extract_score_from_llm_response(score_text, default_score=0.0, scale_min=-1.0, scale_max=1.0):
    if score_text is not None:
        match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", score_text)
        if match:
            try:
                score = float(match.group(0))
                return np.clip(score, scale_min, scale_max)
            except ValueError:
                logger.error(f"LLM ({LLM_PROVIDER}) Extracted text '{match.group(0)}' but conversion to number failed. Original response: '{score_text[:200]}'")
                return default_score
        else:
            logger.error(f"LLM ({LLM_PROVIDER}) Could not find a parsable number in the response: '{score_text[:200]}'")
            return default_score
    return default_score
# -----------------------------------------------------------------------------
# 5. Data acquisition and analysis functions
# -----------------------------------------------------------------------------
# --- 5.1 News and Sentiment Analysis ---
# (fetch_news_from_newsapi, analyze_text_sentiment_llm Functionality is the same as the previous version (V5))
def fetch_news_from_newsapi(query, from_date_str, to_date_str, page_size=20):
    if not NEWS_API_KEY: logger.warning("NewsAPI Key not provided..."); return []
    logger.info(f"NewsAPI Get '{query}' ({from_date_str} to {to_date_str})...")
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
        logger.info(f"NewsAPI returns {len(articles)} articles.")
        return articles
    except Exception as e:
        logger.error(f"NewsAPI failed to get '{query}': {e}")
        if isinstance(e, Exception) :
            try:
                msg_details = str(e)
                if hasattr(e, 'get_message') and callable(e.get_message):
                     msg_details = e.get_message()
                elif hasattr(e, 'args') and e.args:
                     msg_details = str(e.args[0])

                if "too far in the past" in msg_details or "maximum age" in msg_details or "dateMismatch" in msg_details:
                    logger.error(f"NewsAPI error message: date is out of range or format error. Please check your API plan and request date. Details: {msg_details}")
            except:
                pass
        return []

def analyze_text_sentiment_llm(text_content, company_context):
    if not LLM_MODEL_NAME: logger.warning(f"LLM model not specified. Skipping sentiment analysis."); return 0.0
    if not text_content or not isinstance(text_content, str) or len(text_content.strip()) < 10: logger.debug("Sentiment analysis text is too short or empty..."); return 0.0
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

# --- 5.2 Fundamental analysis ---
# (fetch_yfinance_fundamentals, analyze_fundamentals_llm Functionality is the same as the previous version (V5))
def fetch_yfinance_fundamentals(ticker_symbol):
    logger.info(f"yfinance Get {ticker_symbol} fundamentals...")
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
        logger.info(f"Successfully obtained {ticker_symbol} yfinance fundamentals.")
        return key_fundamentals
    except Exception as e: logger.error(f"yfinance failed to get {ticker_symbol} fundamentals: {e}"); return {}

def analyze_fundamentals_llm(company_name, financial_data_dict):
    if not LLM_MODEL_NAME: logger.warning(f"LLM model not set. Skipping fundamental analysis."); return 0.5
    if not financial_data_dict: logger.warning(f"{company_name} fundamental data is empty..."); return 0.5
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

# --- 5.3 Industry Trend Analysis ---
def fetch_google_trends_data(keywords_list, timeframe='today 1-m'):
    if not keywords_list: return None
    logger.info(f"Google Trends Get {keywords_list} (timeframe: {timeframe})...")
    try:
        # Fix: Removed retries and backoff_factor from TrendReq to avoid method_whitelist errors
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25))
        pytrends.build_payload(keywords_list, cat=0, timeframe=timeframe, geo='', gprop='')
        trends_df = pytrends.interest_over_time()
        if trends_df.empty: logger.warning(f"Google Trends for {keywords_list} Returns empty data."); return None
        if 'isPartial' in trends_df.columns: trends_df = trends_df.drop(columns=['isPartial'])
        logger.info(f"Successfully obtained Google Trends for {keywords_list}。")
        return trends_df
    except Exception as e:
        logger.error(f"Google Trends failed to retrieve for {keywords_list}: {e}")
        # Check if it is urllib3.exceptions.MaxRetryError or requests.exceptions.RetryError
        # If pytrends still tries to use the old Retry method internally, this error may still occur.
        # But removing the retries parameter of TrendReq is the first step
        if "Retry.__init__() got an unexpected keyword argument 'method_whitelist'" in str(e):
             logger.error("Google Trends still encounters 'method_whitelist' errors. This may indicate that the pytrends version is incompatible with urllib3. Please try updating pytrends or adjusting the urllib3 version.")
        elif "response with code 429" in str(e).lower():
            logger.warning("Google Trends 429 Error: Requests are being made too frequently. Please increase the delay or request less frequently.")
        return None

# (analyze_industry_trend_llm Functionality is the same as the previous version (V5))
def analyze_industry_trend_llm(industry_name, company_name, industry_news_titles, trends_data_summary):
    if not LLM_MODEL_NAME: logger.warning(f"LLM Model not specified. Skipping industry trend analysis."); return 0.5
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

# --- 5.4 Market risk analysis ---
# (fetch_fred_macro_data_series, fetch_vix_data_daily, analyze_market_risk_llm Functionality is the same as the previous version (V5))
def fetch_fred_macro_data_series(series_ids_list, start_date_str, end_date_str):
    if not FRED_API_KEY: logger.warning("FRED API Key not provided..."); return None
    if not series_ids_list: return None
    logger.info(f"FRED Get Macro Data: {series_ids_list} ({start_date_str} to {end_date_str})")
    try:
        fred = Fred(api_key=FRED_API_KEY); data_frames = []
        for series_id in series_ids_list:
            s_start = (pd.to_datetime(start_date_str) - datetime.timedelta(days=90)).strftime('%Y-%m-%d')
            series_data = fred.get_series(series_id, observation_start=s_start, observation_end=end_date_str)
            data_frames.append(series_data.rename(series_id)); time.sleep(0.3)
        if not data_frames: return None
        df = pd.concat(data_frames, axis=1); df = df.ffill().dropna()
        logger.info(f"Successfully acquired macroscopic data from FRED: {list(df.columns)}。")
        return df
    except Exception as e: logger.error(f"FRED failed to obtain data for {series_ids_list}: {e}"); return None

def fetch_vix_data_daily(start_date_str, end_date_str):
    logger.info(f"Get the VIX Index ({start_date_str} to {end_date_str})...")
    try:
        if pd.to_datetime(start_date_str) >= pd.to_datetime(end_date_str):
            logger.warning(f"VIX Download: Start date {start_date_str} is not before end date {end_date_str}. Start date will be adjusted.")
            start_date_str = (pd.to_datetime(end_date_str) - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            if pd.to_datetime(start_date_str) >= pd.to_datetime(end_date_str): # If it still doesn't work (e.g. only one day left)
                 start_date_str = (pd.to_datetime(end_date_str) - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
                 # Make sure start_date is actually before end_date
                 if pd.to_datetime(start_date_str) >= pd.to_datetime(end_date_str):
                     logger.error(f"Unable to set valid date range for VIX download: {start_date_str} to {end_date_str}")
                     return None
        vix = yf.download('^VIX', start=start_date_str, end=end_date_str, progress=False, auto_adjust=True)
        if vix.empty: logger.warning(f"VIX Data download is empty for range {start_date_str} to {end_date_str}。"); return None
        logger.info(f"Successfully obtained VIX data, totaling {len(vix)} records.")
        return vix[['Close']]
    except Exception as e: logger.error(f"Failed to obtain VIX data: {e}"); return None

def analyze_market_risk_llm(macro_data_summary, vix_latest, market_news_titles):
    if not LLM_MODEL_NAME: logger.warning(f"LLM model not specified. Skipping market risk analysis."); return 0.5
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

# --- 5.5 Black Swan Risk Analysis ---
# (analyze_black_swan_event_llm Functionality is the same as the previous version (V5))
def analyze_black_swan_event_llm(date_obj, company_name, company_specific_news, general_market_news, vix_value, stock_daily_change_pct):
    if not LLM_MODEL_NAME: logger.warning(f"LLM model is not set. Skip black swan risk analysis."); return 0.01
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
    if extracted_score > 0.2: logger.warning(f"LLM ({LLM_PROVIDER}) Pointing out high black swan risk ({extracted_score:.4f}) for {date_obj.strftime('%Y-%m-%d')}.")
    return extracted_score
# -----------------------------------------------------------------------------
# 6. Multi-Agent Collaboration Process
# -----------------------------------------------------------------------------
def run_multi_agent_factor_generation(date_obj, ticker_data_for_day):
    logger.info(f"--- [Log V6 - GoogleTrends/Float Fix] Start factor generation for {date_obj.strftime('%Y-%m-%d')} for {STOCK_TICKER} ---")
    date_str = date_obj.strftime('%Y-%m-%d')
    news_to_date_str = date_str
    news_from_date_str = (date_obj - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
    if pd.to_datetime(news_from_date_str) > pd.to_datetime(news_to_date_str):
        news_from_date_str = (pd.to_datetime(news_to_date_str) - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        logger.warning(f"NewsAPI from_date ({news_from_date_str}) Adjust to be earlier than or equal to to_date ({news_to_date_str})")

    scores = {factor: 0.5 for factor in FACTOR_COLUMNS}
    raw_fundamentals_cache = {}
    company_news, market_news_cache = [], []

    # --- Fundamental Analyst ---
    # (Same as version V5)
    try:
        logger.info("[Agent:Fundamental] Analyze...")
        raw_fundamentals = fetch_yfinance_fundamentals(YAHOO_TICKER)
        raw_fundamentals_cache = raw_fundamentals if raw_fundamentals else {}
        time.sleep(API_CALL_DELAY_SECONDS / 2 if raw_fundamentals else 0.1)
        if raw_fundamentals: scores['fundamental_score'] = analyze_fundamentals_llm(COMPANY_NAME, raw_fundamentals)
        else: scores['fundamental_score'] = 0.3
        logger.info(f"[Agent:Fundamental] Finish. Score: {scores['fundamental_score']:.4f}")
    except Exception as e: logger.error(f"[Agent:Fundamental] Fail: {e}"); scores['fundamental_score'] = 0.3
    time.sleep(API_CALL_DELAY_SECONDS)

    # --- Sentiment Analyst ---
    # (Same as version V5)
    try:
        logger.info("[Agent:Sentiment] Analyze...")
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
        else: logger.info("[Agent:Sentiment] No company news found."); scores['sentiment_score'] = 0.0
        logger.info(f"[Agent:Sentiment] Finish. Score: {scores['sentiment_score']:.4f}")
    except Exception as e: logger.error(f"[Agent:Sentiment] Fail: {e}"); scores['sentiment_score'] = 0.0
    time.sleep(API_CALL_DELAY_SECONDS)

    # --- Industry Trend Analyst ---
    try:
        logger.info("[Agent:IndustryTrend] Analyze...")
        industry_news_query = f"({INDUSTRY_KEYWORDS[0]}) OR ({INDUSTRY_KEYWORDS[7]}) OR (semiconductor industry trends)" # Example
        industry_news_articles = fetch_news_from_newsapi(industry_news_query, news_from_date_str, news_to_date_str, page_size=5)
        time.sleep(API_CALL_DELAY_SECONDS / 2 if industry_news_articles else 0.1)
        industry_news_titles = [n.get('title','') for n in industry_news_articles]

        trends_keywords = [COMPANY_NAME] + INDUSTRY_KEYWORDS[:2]
        trends_df = fetch_google_trends_data(trends_keywords, timeframe=f"{(date_obj - datetime.timedelta(days=30)).strftime('%Y-%m-%d')} {date_str}")
        # Increased delay after calling Google Trends, even if it fails, to avoid retrying immediately or triggering rate issues on other APIs
        time.sleep(API_CALL_DELAY_SECONDS * 1.5) # Extend the delay here
        trends_summary = trends_df.iloc[-1].to_dict() if trends_df is not None and not trends_df.empty else "N/A"

        current_industry = raw_fundamentals_cache.get('industry', "Cloud Computing and AI Software")
        scores['industry_trend_score'] = analyze_industry_trend_llm(current_industry, COMPANY_NAME, industry_news_titles, trends_summary)
        logger.info(f"[Agent:IndustryTrend] Finish. Score: {scores['industry_trend_score']:.4f}")
    except Exception as e: logger.error(f"[Agent:IndustryTrend] Fail: {e}"); scores['industry_trend_score'] = 0.5
    time.sleep(API_CALL_DELAY_SECONDS)

    # --- Market Risk Analyst ---
    vix_close_on_date = None
    try:
        logger.info("[Agent:MarketRisk] Analyze...")
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
            # Fix FutureWarning: use result of .iloc[-1] directly, it should be a scalar
            if date_str in vix_df.index:
                vix_close_on_date = vix_df.loc[date_str, 'Close']
            elif not vix_df.empty: # Make sure the Series is not empty before using iloc
                vix_close_on_date = vix_df['Close'].iloc[-1]
            # Make sure it is a float
            if vix_close_on_date is not None: vix_close_on_date = float(vix_close_on_date)

        market_news_query = "global financial markets OR market volatility OR economic recession risk OR interest rate policy"
        market_news_cache = fetch_news_from_newsapi(market_news_query, news_from_date_str, news_to_date_str, page_size=5)
        time.sleep(API_CALL_DELAY_SECONDS / 2 if market_news_cache else 0.1)
        market_news_titles = [n.get('title','') for n in market_news_cache]

        scores['market_risk_factor'] = analyze_market_risk_llm(macro_summary, vix_close_on_date, market_news_titles)
        logger.info(f"[Agent:MarketRisk] Finish. Score: {scores['market_risk_factor']:.4f}")
    except Exception as e: logger.error(f"[Agent:MarketRisk] Fail: {e}"); scores['market_risk_factor'] = 0.5
    time.sleep(API_CALL_DELAY_SECONDS)

    # --- Black Swan Risk Analyst ---
    # (Same as version V5)
    try:
        logger.info("[Agent:BlackSwan] Analyze...")
        stock_daily_change = None
        if ticker_data_for_day:
            open_p, close_p = ticker_data_for_day.get('Open'), ticker_data_for_day.get('Close')
            if open_p and close_p and isinstance(open_p, (int, float)) and isinstance(close_p, (int, float)) and open_p != 0:
                stock_daily_change = (close_p - open_p) / open_p
        scores['black_swan_risk'] = analyze_black_swan_event_llm(date_obj, COMPANY_NAME, company_news, market_news_cache, vix_close_on_date, stock_daily_change)
        logger.info(f"[Agent:BlackSwan] Finish. Score: {scores['black_swan_risk']:.4f}")
    except Exception as e: logger.error(f"[Agent:BlackSwan] Fail: {e}"); scores['black_swan_risk'] = 0.05

    logger.info(f"--- Factor generation ends for {date_str}. Scores: {scores} ---")
    return scores
# -----------------------------------------------------------------------------
# 7. Main execution process
# -----------------------------------------------------------------------------
# (Same as version V5)
if __name__ == "__main__":
    logger.info("="*70)
    logger.info(f"Start generating factor scores (V6 - GoogleTrends/Float Fix - Provider: {LLM_PROVIDER}, Model: {LLM_MODEL_NAME})...")
    logger.info(f"Stock code: {STOCK_TICKER}, Company: {COMPANY_NAME}")
    logger.info(f"Script target date range: {START_DATE} to {END_DATE}")
    logger.info(f"Logfile: {LOG_FILE_PATH}")
    logger.info(f"Output File: {OUTPUT_CSV_PATH}")
    logger.info("="*70)

    llm_ready = (LLM_PROVIDER == "openai" and openai_client) or \
                  (LLM_PROVIDER == "anthropic" and anthropic_client) or \
                  (LLM_PROVIDER == "google" and google_model_client)

    if not llm_ready: logger.critical(f"The API key for LLM Provider '{LLM_PROVIDER}' is not set or client initialization failed. LLM analysis functionality cannot be performed.")
    # (Other API key checks ...)

    logger.info(f"Downloading historical stock price data for {STOCK_TICKER} (target range: {START_DATE} to {END_DATE})...")
    try:
        yf_end_date = (pd.to_datetime(END_DATE) + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        stock_history_data = yf.download(YAHOO_TICKER, start=START_DATE, end=yf_end_date, progress=True, auto_adjust=True)

        if stock_history_data.empty:
            logger.error(f"Downloading {STOCK_TICKER} stock ticker data is empty (request range: {START_DATE} to {yf_end_date}). Please check the ticker, date range (make sure it is in the past trading day and Yahoo Finance has data) or network. Program terminated.");
            exit()

        stock_history_data = stock_history_data[(stock_history_data.index >= pd.to_datetime(START_DATE)) & (stock_history_data.index <= pd.to_datetime(END_DATE))]
        if stock_history_data.empty:
            logger.error(f"No valid stock price data after filtering from {START_DATE} to {END_DATE}. Program terminated.");
            exit()

        trading_dates = stock_history_data.index
        actual_start_date = stock_history_data.index.min().strftime('%Y-%m-%d')
        actual_end_date = stock_history_data.index.max().strftime('%Y-%m-%d')
        logger.info(f"Get {len(trading_dates)} trading dates. Actual range of yfinance data: {actual_start_date} to {actual_end_date}")

    except Exception as e:
        logger.exception(f"A serious error occurred while downloading yfinance stock price data: {e}");
        exit()

    results_list = []
    failed_dates_log = []

    logger.info(f"Start looping over factor scores for each trading day (Provider: {LLM_PROVIDER}, Model: {LLM_MODEL_NAME}, no cache)...")
    for current_date_dt in tqdm(trading_dates, desc=f"Generate factor scores ({LLM_PROVIDER} - No cache)"):
        try:
            logger.info(f"---+++--- Start processing date: {current_date_dt.strftime('%Y-%m-%d')} ---+++---")
            ticker_data_for_current_day = None
            if not stock_history_data.empty and current_date_dt in stock_history_data.index:
                 ticker_data_for_current_day = stock_history_data.loc[current_date_dt].to_dict()

            daily_scores = run_multi_agent_factor_generation(current_date_dt, ticker_data_for_current_day)

            if daily_scores is None or not isinstance(daily_scores, dict) or not all(key in daily_scores for key in FACTOR_COLUMNS):
                logger.error(f"Invalid or incomplete fraction generated for date {current_date_dt.strftime('%Y-%m-%d')}. Skipping.")
                failed_dates_log.append({'date': current_date_dt.strftime('%Y-%m-%d'), 'error': 'Invalid or incomplete scores dictionary'})
                continue
            results_list.append({'Date': current_date_dt.strftime('%Y-%m-%d'), **daily_scores})
            logger.debug(f"Successfully generated and recorded a score for the date {current_date_dt.strftime('%Y-%m-%d')}: {daily_scores}")
        except KeyboardInterrupt:
            logger.warning("Received a user interrupt request (KeyboardInterrupt). Stopping factor generation...")
            break
        except Exception as e:
            logger.exception(f"An unexpected error occurred while processing the date {current_date_dt.strftime('%Y-%m-%d')}: {e}")
            failed_dates_log.append({'date': current_date_dt.strftime('%Y-%m-%d'), 'error': str(e)})
        finally:
            logger.info(f"Finished processing of date {current_date_dt.strftime('%Y-%m-%d')}. Main loop delay...")
            time.sleep(API_CALL_DELAY_SECONDS)

    logger.info("The factor score generation loop for all dates is completed.")

    if failed_dates_log:
        logger.warning(f"During factor generation, a total of {len(failed_dates_log)} dates failed to be processed or were incomplete. Details are as follows:")
        for entry in failed_dates_log: logger.warning(f"  - Date: {entry['date']}, Error: {entry['error']}")
    else:
        logger.info("All dates are processed successfully (or skipped by design).")

    if not results_list:
        logger.error("Failed to generate any valid factor scores. Unable to create CSV archive.")
    else:
        scores_df = pd.DataFrame(results_list)
        try:
            scores_df['Date'] = pd.to_datetime(scores_df['Date'])
            scores_df.set_index('Date', inplace=True)
            logger.info(f"Factor scores have been successfully converted to DataFrame ({len(scores_df)} rows). ")
if not scores_df.empty: logger.debug(f"Preview of generated score data (first 3 rows):\n{scores_df.head(3)}")

            logger.info(f"Storing results in {OUTPUT_CSV_PATH}...")
            scores_df.to_csv(OUTPUT_CSV_PATH, encoding='utf-8-sig')
            logger.info(f"Factor scores have been successfully stored to {OUTPUT_CSV_PATH}")
        except Exception as e:
            logger.exception(f"Error processing DataFrame or saving CSV file: {e}")

    logger.info("="*70); logger.info(f"Factor score generation process (V6 - GoogleTrends/Float Fix - Provider: {LLM_PROVIDER}) has completed execution."); logger.info("="*70)
	