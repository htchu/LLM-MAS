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

import os
import json
import time
import logging
import datetime
import random
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from dotenv import load_dotenv

# LLM Client Libraries
import openai
import anthropic
import google.generativeai as genai

# Data Source Libraries
from newsapi import NewsApiClient
from fredapi import Fred
from pytrends.request import TrendReq
import feedparser
import requests
from bs4 import BeautifulSoup


@dataclass
class Config:
    """Configuration class for the financial factor scoring system."""
    
    # Data directories and files
    DATA_DIR: str = "financial_factor_data"
    LOG_FILE: str = "factor_generation.log"
    OUTPUT_CSV: str = "factor_scores.csv"
    
    # Target company configuration
    STOCK_TICKER: str = "MSFT"
    COMPANY_NAME: str = "Microsoft Corporation"
    INDUSTRY_KEYWORDS: List[str] = None
    
    # Date range
    START_DATE: str = "2025-05-05"
    END_DATE: str = "2025-05-16"
    
    # LLM configuration
    LLM_PROVIDER: str = "openai"  # openai, anthropic, google
    LLM_MODEL: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 200
    
    # API settings
    API_DELAY_SECONDS: float = 3.5
    NEWS_LOOKBACK_DAYS: int = 2
    
    # Factor columns
    FACTOR_COLUMNS: List[str] = None
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.INDUSTRY_KEYWORDS is None:
            self.INDUSTRY_KEYWORDS = [
                "Microsoft Azure", "OpenAI Microsoft", "Windows OS", 
                "Microsoft Copilot", "Xbox gaming", "Activision Blizzard",
                "MSFT earnings", "cloud computing trends", "AI software",
                "Surface devices", "Satya Nadella strategy"
            ]
        
        if self.FACTOR_COLUMNS is None:
            self.FACTOR_COLUMNS = [
                'fundamental_score', 'sentiment_score', 'industry_trend_score',
                'market_risk_factor', 'black_swan_risk'
            ]


class Logger:
    """Enhanced logging utility for the financial factor system."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        # Create data directory
        Path(self.config.DATA_DIR).mkdir(exist_ok=True)
        
        # Configure logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_path = Path(self.config.DATA_DIR) / self.config.LOG_FILE
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)


class LLMClient:
    """Unified LLM client supporting multiple providers."""
    
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client based on configuration."""
        load_dotenv()
        
        provider = self.config.LLM_PROVIDER.lower()
        
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
                self.logger.info(f"OpenAI client initialized with model: {self.config.LLM_MODEL}")
            else:
                self.logger.error("OpenAI API key not found!")
        
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
                self.logger.info(f"Anthropic client initialized with model: {self.config.LLM_MODEL}")
            else:
                self.logger.error("Anthropic API key not found!")
        
        elif provider == "google":
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                model_name = self.config.LLM_MODEL
                if not model_name.startswith("models/"):
                    model_name = f"models/{model_name}"
                self.client = genai.GenerativeModel(model_name)
                self.logger.info(f"Google Gemini client initialized with model: {model_name}")
            else:
                self.logger.error("Google API key not found!")
        
        else:
            self.logger.error(f"Unsupported LLM provider: {provider}")
    
    def generate_response(self, prompt: str, max_tokens: Optional[int] = None, 
                         temperature: Optional[float] = None) -> Optional[str]:
        """Generate response from the LLM."""
        if not self.client:
            self.logger.error("LLM client not initialized")
            return None
        
        max_tokens = max_tokens or self.config.LLM_MAX_TOKENS
        temperature = temperature or self.config.LLM_TEMPERATURE
        
        try:
            provider = self.config.LLM_PROVIDER.lower()
            
            if provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.config.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            
            elif provider == "anthropic":
                response = self.client.messages.create(
                    model=self.config.LLM_MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                if response.content and hasattr(response.content[0], 'text'):
                    return response.content[0].text.strip()
            
            elif provider == "google":
                response = self.client.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        candidate_count=1,
                        max_output_tokens=max_tokens,
                        temperature=temperature
                    )
                )
                if hasattr(response, 'text') and response.text:
                    return response.text.strip()
        
        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            return None
    
    def extract_numeric_score(self, response: str, default: float = 0.0, 
                            min_val: float = -1.0, max_val: float = 1.0) -> float:
        """Extract numeric score from LLM response."""
        if not response:
            return default
        
        import re
        match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", response)
        if match:
            try:
                score = float(match.group(0))
                return np.clip(score, min_val, max_val)
            except ValueError:
                self.logger.error(f"Failed to convert '{match.group(0)}' to float")
        
        self.logger.error(f"No numeric value found in response: {response[:200]}")
        return default


class DataFetcher:
    """Handles fetching data from various financial data sources."""
    
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self._initialize_apis()
    
    def _initialize_apis(self):
        """Initialize API clients for data sources."""
        load_dotenv()
        
        # NewsAPI
        self.news_api_key = os.getenv("NEWS_API_KEY")
        if self.news_api_key:
            self.news_client = NewsApiClient(api_key=self.news_api_key)
        else:
            self.logger.warning("NewsAPI key not found - news functionality limited")
            self.news_client = None
        
        # FRED
        self.fred_api_key = os.getenv("FRED_API_KEY")
        if self.fred_api_key:
            self.fred_client = Fred(api_key=self.fred_api_key)
        else:
            self.logger.warning("FRED API key not found - macro data functionality limited")
            self.fred_client = None
    
    def fetch_stock_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch stock price data from Yahoo Finance."""
        self.logger.info(f"Fetching stock data for {ticker} from {start_date} to {end_date}")
        
        try:
            # Add one day to end date for yfinance
            end_date_adj = (pd.to_datetime(end_date) + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            
            data = yf.download(ticker, start=start_date, end=end_date_adj, 
                             progress=False, auto_adjust=True)
            
            if data.empty:
                self.logger.error(f"No stock data found for {ticker}")
                return pd.DataFrame()
            
            # Filter to exact date range
            data = data[(data.index >= pd.to_datetime(start_date)) & 
                       (data.index <= pd.to_datetime(end_date))]
            
            self.logger.info(f"Successfully fetched {len(data)} trading days of data")
            return data
        
        except Exception as e:
            self.logger.error(f"Failed to fetch stock data: {e}")
            return pd.DataFrame()
    
    def fetch_company_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Fetch company fundamental data from Yahoo Finance."""
        self.logger.info(f"Fetching fundamentals for {ticker}")
        
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            fundamentals = {
                "marketCap": info.get("marketCap"),
                "enterpriseValue": info.get("enterpriseValue"),
                "trailingPE": info.get("trailingPE"),
                "forwardPE": info.get("forwardPE"),
                "profitMargins": info.get("profitMargins"),
                "returnOnEquity": info.get("returnOnEquity"),
                "revenueGrowth": info.get("revenueGrowth"),
                "earningsGrowth": info.get("earningsQuarterlyGrowth"),
                "debtToEquity": info.get("debtToEquity"),
                "beta": info.get("beta"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "businessSummary": info.get("longBusinessSummary", "")[:1000]
            }
            
            self.logger.info("Successfully fetched fundamental data")
            return fundamentals
        
        except Exception as e:
            self.logger.error(f"Failed to fetch fundamentals: {e}")
            return {}
    
    def fetch_news(self, query: str, from_date: str, to_date: str, 
                   page_size: int = 20) -> List[Dict[str, Any]]:
        """Fetch news articles from NewsAPI."""
        if not self.news_client:
            self.logger.warning("NewsAPI client not available")
            return []
        
        self.logger.info(f"Fetching news for query: {query}")
        
        try:
            response = self.news_client.get_everything(
                q=query,
                from_param=from_date,
                to=to_date,
                language='en',
                sort_by='relevancy',
                page_size=page_size
            )
            
            articles = response.get('articles', [])
            self.logger.info(f"Found {len(articles)} news articles")
            return articles
        
        except Exception as e:
            self.logger.error(f"Failed to fetch news: {e}")
            return []
    
    def fetch_google_trends(self, keywords: List[str], 
                           timeframe: str = 'today 1-m') -> Optional[pd.DataFrame]:
        """Fetch Google Trends data."""
        if not keywords:
            return None
        
        self.logger.info(f"Fetching Google Trends for: {keywords}")
        
        try:
            # Initialize pytrends without retries parameter to avoid method_whitelist error
            pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
            pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='', gprop='')
            
            trends_df = pytrends.interest_over_time()
            
            if trends_df.empty:
                self.logger.warning("Google Trends returned empty data")
                return None
            
            # Remove 'isPartial' column if it exists
            if 'isPartial' in trends_df.columns:
                trends_df = trends_df.drop(columns=['isPartial'])
            
            self.logger.info("Successfully fetched Google Trends data")
            return trends_df
        
        except Exception as e:
            self.logger.error(f"Failed to fetch Google Trends: {e}")
            return None
    
    def fetch_macro_data(self, series_ids: List[str], start_date: str, 
                        end_date: str) -> Optional[pd.DataFrame]:
        """Fetch macroeconomic data from FRED."""
        if not self.fred_client or not series_ids:
            return None
        
        self.logger.info(f"Fetching FRED data for: {series_ids}")
        
        try:
            data_frames = []
            for series_id in series_ids:
                # Start earlier to ensure we have data
                extended_start = (pd.to_datetime(start_date) - 
                                datetime.timedelta(days=90)).strftime('%Y-%m-%d')
                
                series_data = self.fred_client.get_series(
                    series_id, 
                    observation_start=extended_start,
                    observation_end=end_date
                )
                
                data_frames.append(series_data.rename(series_id))
                time.sleep(0.3)  # Rate limiting
            
            if not data_frames:
                return None
            
            df = pd.concat(data_frames, axis=1)
            df = df.ffill().dropna()
            
            self.logger.info(f"Successfully fetched FRED data: {list(df.columns)}")
            return df
        
        except Exception as e:
            self.logger.error(f"Failed to fetch FRED data: {e}")
            return None
    
    def fetch_vix_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch VIX volatility index data."""
        self.logger.info(f"Fetching VIX data from {start_date} to {end_date}")
        
        try:
            vix_data = yf.download('^VIX', start=start_date, end=end_date, 
                                 progress=False, auto_adjust=True)
            
            if vix_data.empty:
                self.logger.warning("VIX data is empty")
                return None
            
            self.logger.info(f"Successfully fetched VIX data: {len(vix_data)} records")
            return vix_data[['Close']]
        
        except Exception as e:
            self.logger.error(f"Failed to fetch VIX data: {e}")
            return None


class FactorAnalyst:
    """Base class for specialized factor analysis agents."""
    
    def __init__(self, config: Config, logger: Logger, llm_client: LLMClient, 
                 data_fetcher: DataFetcher):
        self.config = config
        self.logger = logger
        self.llm_client = llm_client
        self.data_fetcher = data_fetcher
    
    def analyze(self, date_obj: datetime.date, **kwargs) -> float:
        """Analyze and return a factor score. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the analyze method")


class FundamentalAnalyst(FactorAnalyst):
    """Analyst for fundamental analysis using company financial data."""
    
    def analyze(self, date_obj: datetime.date, **kwargs) -> float:
        """Analyze fundamental strength of the company."""
        self.logger.info("[Agent:Fundamental] Starting analysis...")
        
        try:
            fundamentals = self.data_fetcher.fetch_company_fundamentals(
                self.config.STOCK_TICKER
            )
            
            if not fundamentals:
                self.logger.warning("No fundamental data available")
                return 0.3
            
            score = self._analyze_fundamentals_with_llm(fundamentals)
            self.logger.info(f"[Agent:Fundamental] Completed. Score: {score:.4f}")
            return score
        
        except Exception as e:
            self.logger.error(f"[Agent:Fundamental] Failed: {e}")
            return 0.3
    
    def _analyze_fundamentals_with_llm(self, fundamentals: Dict[str, Any]) -> float:
        """Use LLM to analyze fundamental data."""
        # Filter out None values and convert to JSON
        clean_data = {k: v for k, v in fundamentals.items() if v is not None}
        data_str = json.dumps(clean_data, indent=2, default=str)
        
        prompt = f"""As a senior quantitative portfolio manager with 30 years of Wall Street experience, 
evaluate the fundamental strength of {self.config.COMPANY_NAME} using the provided financial data.

Analyze these key aspects:
1. Profitability & Efficiency: Profit margins, ROE, ROA
2. Valuation: P/E ratios, enterprise value multiples
3. Growth: Revenue and earnings growth sustainability
4. Financial Health: Debt levels, liquidity, cash flow
5. Competitive Position: Market position and moat strength

Return ONLY a score between 0.0 (weak fundamentals, potential short) and 1.0 (strong fundamentals, attractive long).

Financial Data:
{data_str[:3500]}

Fundamental Score (0.0 to 1.0):"""
        
        response = self.llm_client.generate_response(prompt, max_tokens=15)
        return self.llm_client.extract_numeric_score(response, default=0.5, 
                                                   min_val=0.0, max_val=1.0)


class SentimentAnalyst(FactorAnalyst):
    """Analyst for news sentiment analysis."""
    
    def analyze(self, date_obj: datetime.date, **kwargs) -> float:
        """Analyze sentiment from recent news about the company."""
        self.logger.info("[Agent:Sentiment] Starting analysis...")
        
        try:
            from_date = (date_obj - datetime.timedelta(
                days=self.config.NEWS_LOOKBACK_DAYS)).strftime('%Y-%m-%d')
            to_date = date_obj.strftime('%Y-%m-%d')
            
            # Fetch company-specific news
            query = f'"{self.config.COMPANY_NAME}" OR "{self.config.STOCK_TICKER}"'
            news_articles = self.data_fetcher.fetch_news(query, from_date, to_date, 
                                                       page_size=10)
            
            if not news_articles:
                self.logger.info("No company news found")
                return 0.0
            
            # Analyze sentiment of top articles
            sentiment_scores = []
            for article in news_articles[:3]:
                text = article.get('title', '') + ". " + article.get('description', '')
                if len(text.strip()) > 20:
                    score = self._analyze_sentiment_with_llm(text)
                    if score is not None:
                        sentiment_scores.append(score)
                    time.sleep(self.config.API_DELAY_SECONDS)
            
            if sentiment_scores:
                final_score = np.mean(sentiment_scores)
            else:
                final_score = 0.0
            
            self.logger.info(f"[Agent:Sentiment] Completed. Score: {final_score:.4f}")
            return final_score
        
        except Exception as e:
            self.logger.error(f"[Agent:Sentiment] Failed: {e}")
            return 0.0
    
    def _analyze_sentiment_with_llm(self, text: str) -> float:
        """Use LLM to analyze sentiment of news text."""
        prompt = f"""As an experienced Wall Street quantitative analyst, analyze the sentiment 
of this financial news concerning {self.config.COMPANY_NAME}.

Focus on identifying actionable trading signals that could impact algorithmic trading decisions.
Ignore generic statements and focus on material events affecting stock price momentum.

Return ONLY a single number between -1.0 (strong negative/sell signal) and 1.0 (strong positive/buy signal).
0.0 indicates neutral or no tradeable impact.

News Text: "{text[:2500]}"

Sentiment Score (-1.0 to 1.0):"""
        
        response = self.llm_client.generate_response(prompt, max_tokens=15)
        return self.llm_client.extract_numeric_score(response, default=0.0, 
                                                   min_val=-1.0, max_val=1.0)


class IndustryTrendAnalyst(FactorAnalyst):
    """Analyst for industry trend analysis."""
    
    def analyze(self, date_obj: datetime.date, **kwargs) -> float:
        """Analyze industry trends using news and Google Trends data."""
        self.logger.info("[Agent:IndustryTrend] Starting analysis...")
        
        try:
            from_date = (date_obj - datetime.timedelta(
                days=self.config.NEWS_LOOKBACK_DAYS)).strftime('%Y-%m-%d')
            to_date = date_obj.strftime('%Y-%m-%d')
            
            # Fetch industry news
            industry_query = f"({self.config.INDUSTRY_KEYWORDS[0]}) OR ({self.config.INDUSTRY_KEYWORDS[1]})"
            industry_news = self.data_fetcher.fetch_news(industry_query, from_date, 
                                                       to_date, page_size=5)
            
            # Fetch Google Trends data
            trends_keywords = [self.config.COMPANY_NAME] + self.config.INDUSTRY_KEYWORDS[:2]
            trends_timeframe = f"{(date_obj - datetime.timedelta(days=30)).strftime('%Y-%m-%d')} {to_date}"
            trends_data = self.data_fetcher.fetch_google_trends(trends_keywords, 
                                                               trends_timeframe)
            
            # Prepare data for LLM analysis
            news_titles = [article.get('title', '') for article in industry_news]
            trends_summary = "N/A"
            if trends_data is not None and not trends_data.empty:
                trends_summary = trends_data.iloc[-1].to_dict()
            
            # Get industry from fundamentals (fallback to default)
            industry = kwargs.get('industry', "Cloud Computing and AI Software")
            
            score = self._analyze_industry_trend_with_llm(industry, news_titles, 
                                                        trends_summary)
            
            self.logger.info(f"[Agent:IndustryTrend] Completed. Score: {score:.4f}")
            return score
        
        except Exception as e:
            self.logger.error(f"[Agent:IndustryTrend] Failed: {e}")
            return 0.5
    
    def _analyze_industry_trend_with_llm(self, industry: str, news_titles: List[str], 
                                       trends_summary: Any) -> float:
        """Use LLM to analyze industry trends."""
        news_str = " ".join(news_titles[:10])
        
        prompt = f"""As a senior quantitative strategist at a major Wall Street firm, assess the 
current trend momentum for the {industry} industry, particularly for companies like {self.config.COMPANY_NAME}.

Focus on:
1. Trend Strength & Momentum: Is the trend accelerating or decelerating?
2. Key Drivers: What macroeconomic, technological, or regulatory factors are driving trends?
3. Investor Sentiment: Overall market sentiment toward this industry
4. Relative Performance: How is this industry performing vs. broader market?

Return ONLY a score between 0.0 (strong negative trend, significant headwinds) and 1.0 
(strong positive trend, significant tailwinds). 0.5 indicates neutral/mixed trends.

Recent Industry News: {news_str[:1500]}
Google Trends Summary: {str(trends_summary)[:1000]}

Industry Trend Score (0.0 to 1.0):"""
        
        response = self.llm_client.generate_response(prompt, max_tokens=15)
        return self.llm_client.extract_numeric_score(response, default=0.5, 
                                                   min_val=0.0, max_val=1.0)


class MarketRiskAnalyst(FactorAnalyst):
    """Analyst for overall market risk assessment."""
    
    def analyze(self, date_obj: datetime.date, **kwargs) -> float:
        """Analyze overall market risk using macro data and VIX."""
        self.logger.info("[Agent:MarketRisk] Starting analysis...")
        
        try:
            date_str = date_obj.strftime('%Y-%m-%d')
            
            # Fetch macro data
            fred_series = ['T10Y2Y', 'VIXCLS', 'SOFR']  # Yield curve, VIX, rates
            macro_start = (date_obj - datetime.timedelta(days=60)).strftime('%Y-%m-%d')
            macro_data = self.data_fetcher.fetch_macro_data(fred_series, macro_start, date_str)
            
            # Fetch VIX data
            vix_start = (date_obj - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
            vix_data = self.data_fetcher.fetch_vix_data(vix_start, date_str)
            
            # Fetch market news
            from_date = (date_obj - datetime.timedelta(
                days=self.config.NEWS_LOOKBACK_DAYS)).strftime('%Y-%m-%d')
            market_query = "global financial markets OR market volatility OR economic recession"
            market_news = self.data_fetcher.fetch_news(market_query, from_date, 
                                                     date_str, page_size=5)
            
            # Prepare data for analysis
            macro_summary = "N/A"
            if macro_data is not None and not macro_data.empty:
                relevant_data = macro_data[macro_data.index <= pd.to_datetime(date_str)]
                if not relevant_data.empty:
                    macro_summary = relevant_data.iloc[-1].to_dict()
            
            vix_latest = None
            if vix_data is not None and not vix_data.empty:
                if date_str in vix_data.index:
                    vix_latest = float(vix_data.loc[date_str, 'Close'])
                else:
                    vix_latest = float(vix_data['Close'].iloc[-1])
            
            news_titles = [article.get('title', '') for article in market_news]
            
            score = self._analyze_market_risk_with_llm(macro_summary, vix_latest, 
                                                     news_titles)
            
            self.logger.info(f"[Agent:MarketRisk] Completed. Score: {score:.4f}")
            return score
        
        except Exception as e:
            self.logger.error(f"[Agent:MarketRisk] Failed: {e}")
            return 0.5
    
    def _analyze_market_risk_with_llm(self, macro_summary: Any, vix_latest: Optional[float], 
                                    news_titles: List[str]) -> float:
        """Use LLM to analyze market risk."""
        news_str = " ".join(news_titles[:10])
        
        prompt = f"""As an experienced quantitative risk manager at a premier Wall Street firm, 
assess the current overall market risk level for portfolio positioning and hedging decisions.

Consider:
1. Macroeconomic Indicators: Yield curve, inflation, credit conditions
2. Volatility Regime: VIX levels and volatility patterns
3. Market Sentiment: Broad market sentiment and liquidity conditions
4. Systemic Risks: Any emerging systemic or correlation risks

Return ONLY a score between 0.0 (benign risk environment, risk-on) and 1.0 
(high risk environment, risk-off/defensive positioning required).

Macro Data: {str(macro_summary)[:1500]}
Latest VIX: {vix_latest if vix_latest is not None else 'N/A'}
Market News: {news_str[:2000]}

Market Risk Factor (0.0 to 1.0):"""
        
        response = self.llm_client.generate_response(prompt, max_tokens=15)
        return self.llm_client.extract_numeric_score(response, default=0.5, 
                                                   min_val=0.0, max_val=1.0)


class BlackSwanAnalyst(FactorAnalyst):
    """Analyst for black swan risk assessment."""
    
    def analyze(self, date_obj: datetime.date, **kwargs) -> float:
        """Analyze potential black swan event risks."""
        self.logger.info("[Agent:BlackSwan] Starting analysis...")
        
        try:
            company_news = kwargs.get('company_news', [])
            market_news = kwargs.get('market_news', [])
            vix_value = kwargs.get('vix_value')
            stock_daily_change = kwargs.get('stock_daily_change')
            
            score = self._analyze_black_swan_with_llm(date_obj, company_news, 
                                                    market_news, vix_value, 
                                                    stock_daily_change)
            
            if score > 0.2:
                self.logger.warning(f"High black swan risk detected: {score:.4f}")
            
            self.logger.info(f"[Agent:BlackSwan] Completed. Score: {score:.4f}")
            return score
        
        except Exception as e:
            self.logger.error(f"[Agent:BlackSwan] Failed: {e}")
            return 0.05
    
    def _analyze_black_swan_with_llm(self, date_obj: datetime.date, 
                                   company_news: List[Dict], market_news: List[Dict],
                                   vix_value: Optional[float], 
                                   stock_daily_change: Optional[float]) -> float:
        """Use LLM to analyze black swan risks."""
        news_summary = (
            f"Company News: " + " ".join([n.get('title', '') for n in company_news[:3]]) +
            f" | Market News: " + " ".join([n.get('title', '') for n in market_news[:3]])
        )
        
        # Identify anomalous conditions
        anomalies = []
        if vix_value and vix_value > 40:
            anomalies.append(f"Extremely High VIX ({vix_value:.1f})")
        
        stock_change_str = 'N/A'
        if stock_daily_change and abs(stock_daily_change) > 0.12:
            stock_change_str = f"{stock_daily_change*100:.2f}%"
            anomalies.append(f"Extreme Price Movement ({stock_change_str})")
        
        prompt = f"""As a veteran Wall Street risk manager who has navigated multiple crises, 
assess the likelihood of an imminent 'black swan' event for {date_obj.strftime('%Y-%m-%d')}.

Focus on truly unforeseen, high-impact systemic risks that standard models don't capture:
1. Anomalous Signals: Unusual patterns in news or market data
2. Contagion Risk: Potential for localized shocks to become systemic
3. Unmodeled Risks: Emerging risks outside standard financial modeling

MAINTAIN EXTREME CONSERVATISM: High scores (>0.5) only for exceptionally strong evidence 
of potential systemic discontinuity. Typical scores should be 0.0-0.05.

Return ONLY a score between 0.0 (negligible black swan risk) and 1.0 (credible signals 
of impending black swan event).

News Summary: {news_summary[:2500]}
Market Indicators: VIX: {vix_value if vix_value else 'N/A'}, Stock Change: {stock_change_str}
Anomalies: {'; '.join(anomalies) if anomalies else 'None detected'}

Black Swan Risk Score (0.0 to 1.0, extreme caution for >0.1):"""
        
        response = self.llm_client.generate_response(prompt, max_tokens=15, temperature=0.05)
        return self.llm_client.extract_numeric_score(response, default=0.01, 
                                                   min_val=0.0, max_val=1.0)


class FactorGenerationEngine:
    """Main engine that orchestrates the multi-agent factor generation process."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger(config)
        self.llm_client = LLMClient(config, self.logger)
        self.data_fetcher = DataFetcher(config, self.logger)
        
        # Initialize analysts
        self.analysts = {
            'fundamental': FundamentalAnalyst(config, self.logger, self.llm_client, self.data_fetcher),
            'sentiment': SentimentAnalyst(config, self.logger, self.llm_client, self.data_fetcher),
            'industry_trend': IndustryTrendAnalyst(config, self.logger, self.llm_client, self.data_fetcher),
            'market_risk': MarketRiskAnalyst(config, self.logger, self.llm_client, self.data_fetcher),
            'black_swan': BlackSwanAnalyst(config, self.logger, self.llm_client, self.data_fetcher)
        }
    
    def generate_daily_factors(self, date_obj: datetime.date, 
                              stock_data: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Generate all factor scores for a specific date."""
        self.logger.info(f"=== Generating factors for {date_obj.strftime('%Y-%m-%d')} ===")
        
        scores = {factor: 0.5 for factor in self.config.FACTOR_COLUMNS}
        
        try:
            # Fundamental Analysis
            scores['fundamental_score'] = self.analysts['fundamental'].analyze(date_obj)
            time.sleep(self.config.API_DELAY_SECONDS)
            
            # Sentiment Analysis
            scores['sentiment_score'] = self.analysts['sentiment'].analyze(date_obj)
            time.sleep(self.config.API_DELAY_SECONDS)
            
            # Industry Trend Analysis
            scores['industry_trend_score'] = self.analysts['industry_trend'].analyze(date_obj)
            time.sleep(self.config.API_DELAY_SECONDS)
            
            # Market Risk Analysis
            scores['market_risk_factor'] = self.analysts['market_risk'].analyze(date_obj)
            time.sleep(self.config.API_DELAY_SECONDS)
            
            # Black Swan Analysis (needs additional context)
            # Fetch news for black swan analysis
            from_date = (date_obj - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
            to_date = date_obj.strftime('%Y-%m-%d')
            
            company_query = f'"{self.config.COMPANY_NAME}" OR "{self.config.STOCK_TICKER}"'
            company_news = self.data_fetcher.fetch_news(company_query, from_date, to_date, 5)
            
            market_query = "global financial markets OR crisis OR systemic risk"
            market_news = self.data_fetcher.fetch_news(market_query, from_date, to_date, 5)
            
            # Calculate stock daily change if data available
            stock_daily_change = None
            if stock_data and stock_data.get('Open') and stock_data.get('Close'):
                open_price = stock_data['Open']
                close_price = stock_data['Close']
                if open_price != 0:
                    stock_daily_change = (close_price - open_price) / open_price
            
            # Get VIX value (simplified - could be enhanced)
            vix_data = self.data_fetcher.fetch_vix_data(
                (date_obj - datetime.timedelta(days=3)).strftime('%Y-%m-%d'),
                date_obj.strftime('%Y-%m-%d')
            )
            vix_value = None
            if vix_data is not None and not vix_data.empty:
                vix_value = float(vix_data['Close'].iloc[-1])
            
            scores['black_swan_risk'] = self.analysts['black_swan'].analyze(
                date_obj,
                company_news=company_news,
                market_news=market_news,
                vix_value=vix_value,
                stock_daily_change=stock_daily_change
            )
            
            self.logger.info(f"Generated scores: {scores}")
            return scores
        
        except Exception as e:
            self.logger.error(f"Error generating factors for {date_obj}: {e}")
            return scores
    
    def run_factor_generation(self) -> pd.DataFrame:
        """Run the complete factor generation process."""
        self.logger.info("="*70)
        self.logger.info(f"Starting Multi-Agent Factor Generation")
        self.logger.info(f"Provider: {self.config.LLM_PROVIDER}, Model: {self.config.LLM_MODEL}")
        self.logger.info(f"Company: {self.config.COMPANY_NAME} ({self.config.STOCK_TICKER})")
        self.logger.info(f"Date Range: {self.config.START_DATE} to {self.config.END_DATE}")
        self.logger.info("="*70)
        
        # Fetch stock data
        stock_data = self.data_fetcher.fetch_stock_data(
            self.config.STOCK_TICKER, 
            self.config.START_DATE, 
            self.config.END_DATE
        )
        
        if stock_data.empty:
            self.logger.error("No stock data available. Terminating.")
            return pd.DataFrame()
        
        # Generate factors for each trading day
        results = []
        failed_dates = []
        
        trading_dates = stock_data.index
        self.logger.info(f"Processing {len(trading_dates)} trading days...")
        
        for date_dt in tqdm(trading_dates, desc="Generating Factor Scores"):
            try:
                # Get stock data for the day
                day_stock_data = None
                if date_dt in stock_data.index:
                    day_stock_data = stock_data.loc[date_dt].to_dict()
                
                # Generate factor scores
                daily_scores = self.generate_daily_factors(date_dt.date(), day_stock_data)
                
                # Add to results
                result_row = {'Date': date_dt.strftime('%Y-%m-%d')}
                result_row.update(daily_scores)
                results.append(result_row)
                
                # API rate limiting
                time.sleep(self.config.API_DELAY_SECONDS)
                
            except KeyboardInterrupt:
                self.logger.warning("Process interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Failed to process {date_dt.strftime('%Y-%m-%d')}: {e}")
                failed_dates.append(date_dt.strftime('%Y-%m-%d'))
        
        if failed_dates:
            self.logger.warning(f"Failed to process {len(failed_dates)} dates: {failed_dates}")
        
        # Create DataFrame and save results
        if results:
            results_df = pd.DataFrame(results)
            results_df['Date'] = pd.to_datetime(results_df['Date'])
            results_df.set_index('Date', inplace=True)
            
            # Save to CSV
            output_path = Path(self.config.DATA_DIR) / self.config.OUTPUT_CSV
            results_df.to_csv(output_path, encoding='utf-8-sig')
            
            self.logger.info(f"Factor scores saved to: {output_path}")
            self.logger.info(f"Generated {len(results_df)} factor score records")
            
            return results_df
        else:
            self.logger.error("No factor scores generated")
            return pd.DataFrame()


def main():
    """Main execution function."""
    # Load configuration
    config = Config()
    
    # Override with environment variables if available
    load_dotenv()
    config.LLM_PROVIDER = os.getenv("LLM_PROVIDER", config.LLM_PROVIDER)
    config.START_DATE = os.getenv("ANALYSIS_START_DATE", config.START_DATE)
    config.END_DATE = os.getenv("ANALYSIS_END_DATE", config.END_DATE)
    
    # Create and run the factor generation engine
    engine = FactorGenerationEngine(config)
    results_df = engine.run_factor_generation()
    
    if not results_df.empty:
        print(f"\n✅ Factor generation completed successfully!")
        print(f"📊 Generated scores for {len(results_df)} trading days")
        print(f"📁 Results saved to: {Path(config.DATA_DIR) / config.OUTPUT_CSV}")
        print(f"\n📈 Sample of generated scores:")
        print(results_df.head())
    else:
        print("❌ Factor generation failed - no results generated")


if __name__ == "__main__":
    main()
