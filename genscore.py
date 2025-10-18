# -*- coding: utf-8 -*- genscore.py
# --------------------------------------------------------------------------------------
# Author: Edward Cheng and Advanced Financial AI Research Team
# Date: 2025-09-02 Update
# Version: Production-Ready Enhanced Edition
# License: MIT
# --------------------------------------------------------------------------------------
"""
Advanced Multi-Agent System (MAS) Financial Factor Scoring Framework
=========================================================================================

A comprehensive framework for generating quantitative trading factors using 
multi-agent systems with Large Language Models (LLMs) and various financial data sources.

Features:
- Multi-LLM support (OpenAI, Anthropic, Google)
- 5 specialized analysis agents
- Real-time data integration from multiple sources
- Robust error handling and logging
- Modular, extensible architecture
- Fetch news from Google News RSS
- MAS (Multi-Agent System) architecture
- CrewAI investment strategy task coordination
- AutoGen system process control and supervision
- MCP (Model Context Provider) for LLM coordination
- Deep integration of economic and news data analysis
- Inter-agent context communication and collaboration

"""

import os
import json
import time
import logging
import datetime
import asyncio
import threading
import traceback
import warnings
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
from queue import Queue, Empty, PriorityQueue
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import inspect
import hashlib

import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# LLM Client Libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI library not available")
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    print("Warning: Anthropic library not available")
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    print("Warning: Google Generative AI library not available")
    GOOGLE_AVAILABLE = False

# Multi-Agent Framework Libraries
try:
    from crewai import Agent, Task, Crew, Process
    CREWAI_AVAILABLE = True
except ImportError:
    print("Warning: CrewAI framework not available. Using fallback implementations.")
    CREWAI_AVAILABLE = False

try:
    from autogen import ConversableAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    print("Warning: AutoGen framework not available. Using fallback implementations.")
    AUTOGEN_AVAILABLE = False

# Data Source Libraries
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    print("Warning: NewsAPI client not available")
    NEWSAPI_AVAILABLE = False

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    print("Warning: FRED API client not available")
    FRED_AVAILABLE = False

try:
    from pytrends.request import TrendReq
    TRENDS_AVAILABLE = True
except ImportError:
    print("Warning: Google Trends client not available")
    TRENDS_AVAILABLE = False

# Additional imports for robust operation
import feedparser
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from urllib.parse import quote, urlencode, urlparse
import re


# ============================================
# Core Enums and Data Structures 
# ============================================

class AgentRole(Enum):
    """Agent role definitions"""
    # Factor Analysis Agents
    FUNDAMENTAL_ANALYST = "fundamental_analyst"         # Fundamental analyst 
    SENTIMENT_ANALYST = "sentiment_analyst"             # Sentiment analyst 
    INDUSTRY_TREND_ANALYST = "industry_trend_analyst"   # Industry trend analyst 
    MARKET_RISK_ANALYST = "market_risk_analyst"         # Market risk analyst 
    BLACK_SWAN_ANALYST = "black_swan_analyst"           # Black swan event analyst 
    
    # CrewAI Strategic Agents
    INVESTMENT_STRATEGIST = "investment_strategist"     # Investment strategist 
    RISK_MANAGER = "risk_manager"                       # Risk manager 
    PORTFOLIO_MANAGER = "portfolio_manager"             # Portfolio manager 
    TRADE_EXECUTOR = "trade_executor"                   # Trade executor 
    
    # AutoGen Supervisory Agents
    SYSTEM_SUPERVISOR = "system_supervisor"             # System supervisor 
    QUALITY_CONTROLLER = "quality_controller"           # Quality controller 
    PROCESS_COORDINATOR = "process_coordinator"         # Process coordinator 
    
    # MCP Communication Agents 
    CONTEXT_MANAGER = "context_manager"                 # Context manager 
    MESSAGE_BROKER = "message_broker"                   # Message broker 


class MessageType(Enum):
    """Message type definitions for inter-agent communication"""
    FACTOR_ANALYSIS = "factor_analysis"                # Factor analysis message
    INVESTMENT_SIGNAL = "investment_signal"            # Investment signal message
    RISK_ALERT = "risk_alert"                          # Risk alert message
    TRADE_ORDER = "trade_order"                        # Trade order message
    CONTEXT_UPDATE = "context_update"                  # Context update message
    SYSTEM_CONTROL = "system_control"                  # System control message
    QUALITY_CHECK = "quality_check"                    # Quality check message 
    CONSENSUS_REQUEST = "consensus_request"            # Consensus request message 
    DATA_SYNC = "data_sync"                            # Data sync message 
    ERROR_REPORT = "error_report"                      # Error report message 


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1        # Critical tasks
    HIGH = 2            # High priority 
    MEDIUM = 3          # Medium priority 
    LOW = 4             # Low priority 
    BACKGROUND = 5      # Background tasks 


class AgentState(Enum):
    """Agent state tracking"""
    INITIALIZING = "initializing"   # Initializing 
    READY = "ready"                 # Ready state 
    WORKING = "working"             # Working 
    WAITING = "waiting"             # Waiting
    ERROR = "error"                 # Error state 
    OFFLINE = "offline"             # Offline 


class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"     # Excellent quality
    GOOD = "good"               # Good quality 
    FAIR = "fair"               # Fair quality 
    POOR = "poor"               # Poor quality 
    UNAVAILABLE = "unavailable" # Data unavailable 


# ============================================
# Stock Configuration System 
# ============================================

@dataclass
class StockConfig:
    """Stock-specific configuration class"""
    ticker: str                    # Stock ticker 
    yahoo_ticker: str              # Yahoo Finance ticker 
    company_name: str              # Company name 
    industry_keywords: List[str]   # Industry keyword list 


def get_nvidia_config() -> StockConfig:
    """Nvidia stock configuration"""
    return StockConfig(
        ticker="NVDA",
        yahoo_ticker="NVDA", 
        company_name="Nvidia Corporation",
        industry_keywords=[
            # AI and Data Center 
            "Nvidia H100", "Nvidia A100", "AI training chips", "GPU computing", "CUDA platform",
            "Nvidia data center revenue", "AI chip demand", "generative AI hardware",
            # Gaming and Graphics 
            "GeForce RTX", "RTX 40 series", "gaming GPU", "ray tracing", "DLSS technology", 
            "gaming revenue", "graphics cards", "PC gaming market",
            # Automotive and AV 
            "Nvidia DRIVE platform", "autonomous vehicle chips", "automotive AI",
            "self-driving car technology", "automotive partnerships",
            # Financial Performance 
            "NVDA stock price", "Nvidia earnings", "Jensen Huang CEO", "Nvidia market cap",
            "semiconductor industry", "chip shortage impact"
        ]
    )


def get_microsoft_config() -> StockConfig:
    """Microsoft stock configuration"""
    return StockConfig(
        ticker="MSFT",
        yahoo_ticker="MSFT",
        company_name="Microsoft Corporation", 
        industry_keywords=[
            # Cloud and AI Services
            "Microsoft Azure", "Azure cloud", "OpenAI partnership", "Copilot AI", 
            "Power Platform", "GitHub Copilot",
            # Core Business 
            "Microsoft 365", "Windows 11", "Surface devices", "Outlook", "OneDrive",
            # Gaming and Entertainment 
            "Xbox Series X", "Xbox Game Pass", "Activision Blizzard acquisition", "Cloud gaming",
            # Financial Performance
            "MSFT earnings", "Microsoft stock price", "Satya Nadella", "Microsoft market cap"
        ]
    )


def get_stock_config(ticker: str) -> StockConfig:
    """
    Get stock configuration by ticker 
    
    Args:
        ticker: Stock ticker (case-insensitive) 
        
    Returns:
        Corresponding StockConfig object 
        
    Raises:
        ValueError: When ticker is not supported 
    """
    ticker = ticker.upper()
    
    config_map = {
        "NVDA": get_nvidia_config,
        "MSFT": get_microsoft_config,
        # Additional stock configurations can be added here 
    }
    
    if ticker not in config_map:
        supported_tickers = ", ".join(config_map.keys())
        raise ValueError(f"Ticker '{ticker}' not supported. Supported: {supported_tickers}")
    
    return config_map[ticker]()


# ============================================
# Advanced Message and Context System 
# ============================================

@dataclass
class AgentMessage:
    """Agent message structure for inter-agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Unique message ID 
    sender_id: str = ""                                         # Sender ID 
    receiver_id: str = ""                                       # Receiver ID (empty for broadcast) 
    message_type: MessageType = MessageType.FACTOR_ANALYSIS     # Message type 
    content: Dict[str, Any] = field(default_factory=dict)       # Message content 
    timestamp: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    priority: TaskPriority = TaskPriority.MEDIUM                # Message priority 
    requires_response: bool = False                             # Requires response 
    context: Dict[str, Any] = field(default_factory=dict)       # Context information 
    metadata: Dict[str, Any] = field(default_factory=dict)      # Metadata 


@dataclass 
class TaskDefinition:
    """Task definition structure for CrewAI tasks"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Task ID 
    title: str = ""                                                  # Task title 
    description: str = ""                                            # Task description  
    assigned_agent: str = ""                                         # Assigned agent ID 
    priority: TaskPriority = TaskPriority.MEDIUM                     # Task priority 
    dependencies: List[str] = field(default_factory=list)            # Dependent task ID list 
    expected_output: str = ""                                        # Expected output format 
    deadline: Optional[datetime.datetime] = None                     # Deadline 
    context_requirements: List[str] = field(default_factory=list)    # Required context information 
    tools_required: List[str] = field(default_factory=list)          # Required tools list 


# ============================================
# Configuration Management System 
# ============================================

@dataclass
class SystemConfig:
    """System configuration class"""
    
    # Basic System Configuration
    DATA_DIR: str = "advanced_mas_data"                    # Data directory
    LOG_FILE: str = "mas_system.log"                       # Log file 
    OUTPUT_CSV: str = "NVDA_score_data.csv"              # Output CSV file 
	# OUTPUT_CSV_PATH = os.path.join(DATA_DIR, f"{STOCK_TICKER}_score_data.csv") 
    
    # Stock Analysis Configuration
    STOCK_TICKER: str = "NVDA"                             # Target stock ticker
    COMPANY_NAME: str = ""                                 # Company name (auto-filled)
    INDUSTRY_KEYWORDS: List[str] = field(default_factory=list)  # Industry keywords (auto-filled) 
    
    # Analysis Period Configuration 
    START_DATE: str = "2024-07-01"                         # Start date 
    END_DATE: str = "2025-06-30"                           # End date 
    
    # LLM Configuration 
    PRIMARY_LLM_PROVIDER: str = "openai"                   # Primary LLM provider 
    SECONDARY_LLM_PROVIDER: str = "anthropic"              # Backup LLM provider 
    TERTIARY_LLM_PROVIDER: str = "google"                  # Third LLM provider 
    LLM_MODEL: str = "gpt-4o"                              # Model name 
    LLM_TEMPERATURE: float = 0.3                           # Temperature parameter 
    LLM_MAX_TOKENS: int = 500                              # Maximum tokens 
    
    # MAS System Configuration 
    MAX_CONCURRENT_AGENTS: int = 15                        # Max concurrent agents 
    MESSAGE_QUEUE_SIZE: int = 2000                         # Message queue size 
    AGENT_TIMEOUT_SECONDS: int = 120                       # Agent response timeout (increased) 
    MAX_RETRY_ATTEMPTS: int = 3                            # Max retry attempts 
    
    # CrewAI Configuration 
    CREW_MAX_ITERATIONS: int = 5                           # Max iterations 
    CREW_VERBOSE_MODE: bool = True                         # Verbose mode 
    CREW_MEMORY_ENABLED: bool = True                       # Memory function 
    CREW_ASYNC_EXECUTION: bool = True                      # Async execution 
    
    # AutoGen Configuration 
    AUTOGEN_MAX_ROUNDS: int = 5                            # Max dialog rounds 
    AUTOGEN_SPEAKER_SELECTION: str = "auto"                # Speaker selection mode 
    AUTOGEN_HUMAN_INPUT_MODE: str = "NEVER"                # Human input mode 
    
    # API Configuration 
    API_RATE_LIMIT_DELAY: float = 2.0                      # API rate limit delay (increased) 
    NEWS_LOOKBACK_DAYS: int = 7                            # News lookback days 
    ECONOMIC_DATA_REFRESH_HOURS: int = 6                   # Economic data refresh hours 
    
    # Performance Configuration 
    ENABLE_CACHING: bool = True                            # Enable caching 
    CACHE_EXPIRY_HOURS: int = 4                            # Cache expiry hours (increased) 
    PARALLEL_PROCESSING: bool = True                       # Enable parallel processing 
    THREAD_POOL_SIZE: int = 8                              # Thread pool size 
    
    # Factor Analysis Configuration 
    FACTOR_COLUMNS: List[str] = field(default_factory=lambda: [
        'fundamental_score', 'sentiment_score', 'industry_trend_score',
        'market_risk_factor', 'black_swan_risk'
    ])
    
    def __post_init__(self):
        """Post-initialization configuration"""
        try:
            # Load stock-specific configuration 
            stock_config = get_stock_config(self.STOCK_TICKER)
            self.COMPANY_NAME = stock_config.company_name
            self.INDUSTRY_KEYWORDS = stock_config.industry_keywords
            
            # Update output file names
            ticker_safe = self.STOCK_TICKER.replace('.', '_')
            self.OUTPUT_CSV = f"mas_factors_{ticker_safe}.csv"
            self.LOG_FILE = f"mas_system_{ticker_safe}.log"
            
            print(f"✓ Loaded configuration for {self.STOCK_TICKER}: {self.COMPANY_NAME}")
            print(f"✓ Industry keywords: {len(self.INDUSTRY_KEYWORDS)} specialized terms")
            
        except ValueError as e:
            print(f"✗ Stock configuration error: {e}")
            # Fallback to NVDA as default
            fallback_config = get_nvidia_config()
            self.STOCK_TICKER = fallback_config.ticker
            self.COMPANY_NAME = fallback_config.company_name
            self.INDUSTRY_KEYWORDS = fallback_config.industry_keywords
            print(f"✓ Fallback to default: {self.STOCK_TICKER}")
    
    def validate_configuration(self) -> bool:
        """Validate configuration integrity"""
        validations = [
            (bool(self.STOCK_TICKER), "Stock ticker cannot be empty"),
            (bool(self.COMPANY_NAME), "Company name cannot be empty"),
            (len(self.INDUSTRY_KEYWORDS) > 0, "Industry keywords cannot be empty"),
            (self.MAX_CONCURRENT_AGENTS > 0, "Max concurrent agents must be positive"),
            (self.AGENT_TIMEOUT_SECONDS > 0, "Agent timeout must be positive"),
            (len(self.FACTOR_COLUMNS) > 0, "Factor columns cannot be empty")
        ]
        
        for condition, error_msg in validations:
            if not condition:
                print(f"✗ Configuration validation failed: {error_msg}")
                return False
        
        print("✓ Configuration validation passed")
        return True


# ============================================
# Enhanced Logging System 
# ============================================

class MASLogger:
    """MAS specialized logging system with hierarchical logging"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.agent_logs = {}           # Agent-specific logs
        self.message_logs = []         # Message communication logs
        self.performance_logs = {}     # Performance statistics logs
        self.error_logs = []           # Error logs 
        
    def _setup_logger(self) -> logging.Logger:
        """Setup multi-level logging system"""
        # Create data directory 
        Path(self.config.DATA_DIR).mkdir(exist_ok=True)
        
        # Setup main logger 
        logger = logging.getLogger("MAS_Financial_System")
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers 
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # Create formatters 
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(name)s] - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Console handler 
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
        
        # File handler 
        log_path = Path(self.config.DATA_DIR) / self.config.LOG_FILE
        try:
            file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create file handler: {e}")
        
        # Error file handler 
        error_path = Path(self.config.DATA_DIR) / f"errors_{self.config.STOCK_TICKER}.log"
        try:
            error_handler = logging.FileHandler(error_path, mode='w', encoding='utf-8')
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(detailed_formatter)
            logger.addHandler(error_handler)
        except Exception as e:
            print(f"Warning: Could not create error handler: {e}")
        
        return logger
    
    def log_agent_activity(self, agent_id: str, activity: str, level: str = "info"):
        """Record agent activity"""
        timestamp = datetime.datetime.now()
        log_entry = {
            'timestamp': timestamp,
            'agent_id': agent_id,
            'activity': activity,
            'level': level
        }
        
        if agent_id not in self.agent_logs:
            self.agent_logs[agent_id] = []
        self.agent_logs[agent_id].append(log_entry)
        
        # Also log to main logger 
        message = f"[Agent:{agent_id}] {activity}"
        try:
            if level == "error":
                self.logger.error(message)
            elif level == "warning":
                self.logger.warning(message)
            else:
                self.logger.info(message)
        except Exception as e:
            print(f"Logging error: {e}")
    
    def log_message_exchange(self, sender_id: str, receiver_id: str, 
                           message_type: MessageType, success: bool = True):
        """Record message exchange between agents"""
        log_entry = {
            'timestamp': datetime.datetime.now(),
            'sender': sender_id,
            'receiver': receiver_id,
            'message_type': message_type.value,
            'success': success
        }
        self.message_logs.append(log_entry)
        
        status = "SUCCESS" if success else "FAILED"
        try:
            self.logger.info(f"Message Exchange [{status}]: {sender_id} → {receiver_id} ({message_type.value})")
        except Exception as e:
            print(f"Logging error: {e}")
    
    def log_performance_metric(self, component: str, metric_name: str, value: float):
        """Record performance metrics"""
        timestamp = datetime.datetime.now()
        
        if component not in self.performance_logs:
            self.performance_logs[component] = []
        
        self.performance_logs[component].append({
            'timestamp': timestamp,
            'metric': metric_name,
            'value': value
        })
        
        try:
            self.logger.debug(f"Performance [{component}] {metric_name}: {value}")
        except Exception as e:
            print(f"Logging error: {e}")
    
    def log_system_error(self, error_source: str, error_message: str, stack_trace: str = ""):
        """Record system errors"""
        error_entry = {
            'timestamp': datetime.datetime.now(),
            'source': error_source,
            'message': error_message,
            'stack_trace': stack_trace
        }
        self.error_logs.append(error_entry)
        
        try:
            self.logger.error(f"System Error [{error_source}]: {error_message}")
            if stack_trace:
                self.logger.error(f"Stack Trace: {stack_trace}")
        except Exception as e:
            print(f"Error logging error: {e}")
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate system health report"""
        total_messages = len(self.message_logs)
        failed_messages = sum(1 for msg in self.message_logs if not msg['success'])
        
        active_agents = len(self.agent_logs)
        total_errors = len(self.error_logs)
        
        return {
            'timestamp': datetime.datetime.now(),
            'active_agents': active_agents,
            'total_messages': total_messages,
            'failed_messages': failed_messages,
            'message_success_rate': (total_messages - failed_messages) / max(total_messages, 1),
            'total_errors': total_errors,
            'performance_metrics': len(self.performance_logs)
        }


# =========================================================
# Model Context Provider (MCP) Implementation 
# =========================================================

class MCPContextManager:
    """MCP context manager for multi-LLM coordination"""
    
    def __init__(self, config: SystemConfig, logger: MASLogger):
        self.config = config
        self.logger = logger
        self.contexts = {}                    # Context storage 
        self.context_history = []             # Context history 
        self.active_sessions = {}             # Active sessions 
        self.llm_clients = {}                 # LLM client pool 
        self._initialize_llm_clients()
        
    def _initialize_llm_clients(self):
        """Initialize multiple LLM clients"""
        load_dotenv()
        
        # OpenAI Client 
        if OPENAI_AVAILABLE:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                try:
                    genai.configure(api_key=openai_key)
                    model_name = self.config.LLM_MODEL
                    if not model_name.startswith("gpt-4o"):
                        model_name = "gpt-4o"                    
                    self.llm_clients['openai'] = openai.OpenAI(api_key=openai_key)
                    self.logger.log_agent_activity("MCP", "OpenAI client initialized")
                except Exception as e:
                    self.logger.log_system_error("MCP", f"OpenAI client initialization failed: {e}")
        
        # Anthropic Client 
        if ANTHROPIC_AVAILABLE:
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")  
            if anthropic_key:
                try:
                    genai.configure(api_key=anthropic_key)
                    model_name = self.config.LLM_MODEL
                    if not model_name.startswith("claude"):
                        model_name = "claude-3-7-sonnet-20250219"
                    self.llm_clients['anthropic'] = anthropic.Anthropic(api_key=anthropic_key)
                    self.logger.log_agent_activity("MCP", "Anthropic client initialized")
                except Exception as e:
                    self.logger.log_system_error("MCP", f"Anthropic client initialization failed: {e}")
        
        # Google Client 
        if GOOGLE_AVAILABLE:
            google_key = os.getenv("GOOGLE_API_KEY")
            if google_key:
                try:
                    genai.configure(api_key=google_key)
                    model_name = self.config.LLM_MODEL
                    if not model_name.startswith("gemini"):
                        model_name = "gemini-2.5-pro"
                    self.llm_clients['google'] = genai.GenerativeModel(model_name)
                    self.logger.log_agent_activity("MCP", "Google Gemini client initialized")
                except Exception as e:
                    self.logger.log_system_error("MCP", f"Google client initialization failed: {e}")
        
        if not self.llm_clients:
            self.logger.log_system_error("MCP", "No LLM clients available. Please check API keys.")
    
    def create_context_session(self, agent_id: str, task_type: str, 
                             initial_context: Dict[str, Any]) -> str:
        """Create new context session"""
        session_id = f"{agent_id}_{task_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        context_session = {
            'session_id': session_id,
            'agent_id': agent_id,
            'task_type': task_type,
            'created_at': datetime.datetime.now(),
            'last_accessed': datetime.datetime.now(),
            'context_data': initial_context,
            'message_history': [],
            'llm_responses': []
        }
        
        self.contexts[session_id] = context_session
        self.context_history.append(session_id)
        self.active_sessions[agent_id] = session_id
        
        self.logger.log_agent_activity(agent_id, f"Created context session: {session_id}")
        return session_id
    
    def update_context(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update context session"""
        if session_id not in self.contexts:
            return False
        
        try:
            self.contexts[session_id]['context_data'].update(updates)
            self.contexts[session_id]['last_accessed'] = datetime.datetime.now()
            return True
        except Exception as e:
            self.logger.log_system_error("MCP", f"Context update failed: {e}")
            return False
    
    def get_llm_response(self, session_id: str, prompt: str, 
                        preferred_provider: str = None) -> Optional[str]:
        """Get LLM response through MCP  with retry mechanism"""
        if session_id not in self.contexts:
            self.logger.log_system_error("MCP", f"Context session not found: {session_id}")
            return None
        
        if not self.llm_clients:
            self.logger.log_system_error("MCP", "No LLM clients available")
            return "Error: No LLM clients available"
        
        context_session = self.contexts[session_id]
        
        # Determine LLM provider to use 
        provider = preferred_provider or self.config.PRIMARY_LLM_PROVIDER
        if provider not in self.llm_clients:
            available_providers = list(self.llm_clients.keys())
            if available_providers:
                provider = available_providers[0]  # Fallback to first available
            else:
                return "Error: No LLM providers available"
        
        # Enhance prompt with context 
        enhanced_prompt = self._enhance_prompt_with_context(prompt, context_session['context_data'])
        
        # Retry mechanism for LLM calls 
        max_retries = self.config.MAX_RETRY_ATTEMPTS
        for attempt in range(max_retries):
            try:
                response = self._call_llm_api(provider, enhanced_prompt)
                
                # Store response in context history 
                context_session['llm_responses'].append({
                    'timestamp': datetime.datetime.now(),
                    'provider': provider,
                    'prompt': prompt,
                    'response': response,
                    'attempt': attempt + 1
                })
                
                context_session['last_accessed'] = datetime.datetime.now()
                return response
                
            except Exception as e:
                self.logger.log_system_error("MCP", f"LLM API call failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    # Wait before retry 
                    time.sleep(self.config.API_RATE_LIMIT_DELAY * (attempt + 1))
                    
                    # Try different provider if available 
                    available_providers = [p for p in self.llm_clients.keys() if p != provider]
                    if available_providers:
                        provider = available_providers[0]
                        self.logger.log_agent_activity("MCP", f"Switching to provider: {provider}")
                else:
                    return f"Error: LLM API call failed after {max_retries} attempts - {str(e)}"
        
        return "Error: All LLM API attempts failed"
    
    def _enhance_prompt_with_context(self, prompt: str, context_data: Dict[str, Any]) -> str:
        """Enhance prompt with context data"""
        try:
            context_str = json.dumps(context_data, indent=2, default=str, ensure_ascii=False)
            
            enhanced_prompt = f"""Context Information:
{context_str[:2000]}

Task Request:
{prompt}

Please provide a response that considers the context information above. Focus on accuracy and relevance to the financial analysis task.
"""
            
            return enhanced_prompt
        except Exception as e:
            self.logger.log_system_error("MCP", f"Prompt enhancement failed: {e}")
            return prompt  # Return original prompt on error
    
    def _call_llm_api(self, provider: str, prompt: str) -> str:
        """Call specific LLM API with timeout handling"""
        client = self.llm_clients[provider]
        
        try:
            if provider == "openai":
                response = client.chat.completions.create(
                    model=self.config.LLM_MODEL if "gpt-4o" in self.config.LLM_MODEL else "gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config.LLM_MAX_TOKENS,
                    temperature=self.config.LLM_TEMPERATURE,
                    timeout=60  # Add timeout 
                )
                return response.choices[0].message.content.strip()
            
            elif provider == "anthropic":
                # Updated for new Anthropic API format
                response = client.messages.create(
                    model=self.config.LLM_MODEL if "claude-3-7-sonnet-20250219" in self.config.LLM_MODEL else "claude-3-5-haiku-20241022",
                    max_tokens=self.config.LLM_MAX_TOKENS,
                    temperature=self.config.LLM_TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=60  # Add timeout 
                )
                return response.content[0].text.strip()
            
            elif provider == "google":
                response = client.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        candidate_count=1,
                        max_output_tokens=self.config.LLM_MAX_TOKENS,
                        temperature=self.config.LLM_TEMPERATURE
                    )
                )
                return response.text.strip()
            
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
                
        except Exception as e:
            raise Exception(f"API call failed for {provider}: {str(e)}")
    
    def extract_numeric_score(self, response: str, default: float = 0.0,
                            min_val: float = -1.0, max_val: float = 1.0) -> float:
        """Extract numeric score from LLM response with improved parsing"""
        if not response:
            return default
        
        try:
            # Clean response text 
            response_clean = response.replace(',', '.').lower()
            
            # Try to find numeric values in response 
            patterns = [
                r"score[:\s]*([+-]?\d*\.?\d+)",  # "score: 0.75"
                r"([+-]?\d*\.?\d+)\s*(?:out of|/)\s*(?:10|100|1)",  # "7.5 out of 10"
                r"rating[:\s]*([+-]?\d*\.?\d+)",  # "rating: 0.8"
                r"([+-]?\d*\.?\d+)",  # Any number
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, response_clean)
                if matches:
                    for match in matches:
                        try:
                            score = float(match)
                            
                            # Handle different scales
                            if abs(score) > 10:  # Assume percentage (0-100)
                                score = score / 100.0
                            elif abs(score) > 1 and abs(score) <= 10:  # Assume 1-10 scale
                                score = (score - 5.5) / 4.5  # Convert to -1 to 1
                            
                            # Clamp to valid range 
                            score = np.clip(score, min_val, max_val)
                            
                            if min_val <= score <= max_val:
                                return score
                        except (ValueError, TypeError):
                            continue
        except Exception as e:
            self.logger.log_system_error("MCP", f"Score extraction failed: {e}")
        
        return default
    
    def cleanup_expired_contexts(self, max_age_hours: int = 24) -> int:
        """Clean up expired context sessions"""
        current_time = datetime.datetime.now()
        expired_sessions = []
        
        for session_id, context in self.contexts.items():
            try:
                age = current_time - context['last_accessed']
                if age.total_seconds() > max_age_hours * 3600:
                    expired_sessions.append(session_id)
            except Exception as e:
                self.logger.log_system_error("MCP", f"Context cleanup error: {e}")
                expired_sessions.append(session_id)  # Remove problematic contexts
        
        for session_id in expired_sessions:
            try:
                del self.contexts[session_id]
                self.logger.log_agent_activity("MCP", f"Cleaned up expired context: {session_id}")
            except Exception as e:
                self.logger.log_system_error("MCP", f"Context deletion failed: {e}")
        
        return len(expired_sessions)


# ============================================
# Enhanced Data Fetching System 
# ============================================

class DataFetchingEngine:
    """Data fetching engine with multi-source integration and improved error handling"""
    
    def __init__(self, config: SystemConfig, logger: MASLogger):
        self.config = config
        self.logger = logger
        self.cache = {}                       # Data cache
        self.cache_timestamps = {}            # Cache timestamps 
        self.api_clients = {}                 # API client pool 
        self.session = self._setup_session()  # HTTP session 
        self._initialize_api_clients()
        
    def _setup_session(self) -> requests.Session:
        """Setup HTTP session with retry strategy"""
        session = requests.Session()
        
        # Configure retry strategy 
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,  # Exponential backoff 
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]  # Updated parameter name
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set timeout 
        session.timeout = 30
        
        return session
        
    def _initialize_api_clients(self):
        """Initialize various data API clients"""
        load_dotenv()
        
        # NewsAPI Client 
        if NEWSAPI_AVAILABLE:
            news_key = os.getenv("NEWS_API_KEY")
            if news_key:
                try:
                    self.api_clients['news'] = NewsApiClient(api_key=news_key)
                    self.logger.log_agent_activity("DataEngine", "NewsAPI client initialized")
                except Exception as e:
                    self.logger.log_system_error("DataEngine", f"NewsAPI initialization failed: {e}")
            else:
                self.logger.log_agent_activity("DataEngine", "NewsAPI key not found", "warning")
        
        # FRED Client 
        if FRED_AVAILABLE:
            fred_key = os.getenv("FRED_API_KEY")
            if fred_key:
                try:
                    self.api_clients['fred'] = Fred(api_key=fred_key)
                    self.logger.log_agent_activity("DataEngine", "FRED API client initialized")
                except Exception as e:
                    self.logger.log_system_error("DataEngine", f"FRED initialization failed: {e}")
            else:
                self.logger.log_agent_activity("DataEngine", "FRED API key not found", "warning")
        
        # Google Trends Client 
        if TRENDS_AVAILABLE:
            try:
                self.api_clients['trends'] = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
                self.logger.log_agent_activity("DataEngine", "Google Trends client initialized")
            except Exception as e:
                self.logger.log_system_error("DataEngine", f"Google Trends initialization failed: {e}")
    
    def fetch_stock_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch stock price data with improved error handling"""
        cache_key = f"stock_{ticker}_{start_date}_{end_date}"
        
        # Check cache first 
        if self._is_cache_valid(cache_key):
            self.logger.log_agent_activity("DataEngine", f"Using cached stock data for {ticker}")
            return self.cache[cache_key]
        
        self.logger.log_agent_activity("DataEngine", f"Fetching stock data: {ticker} ({start_date} to {end_date})")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Fetch from Yahoo Finance 
                end_date_adj = (pd.to_datetime(end_date) + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                
                data = yf.download(
                    ticker, 
                    start=start_date, 
                    end=end_date_adj, 
                    progress=False, 
                    auto_adjust=True,
                    threads=True,
                    timeout=60  # Add timeout 
                )
                
                if data.empty:
                    raise ValueError(f"No stock data found for {ticker}")
                
                # Filter to exact date range 
                data = data[(data.index >= pd.to_datetime(start_date)) & 
                           (data.index <= pd.to_datetime(end_date))]
                
                # Validate data quality 
                if len(data) == 0:
                    raise ValueError(f"No data in specified date range for {ticker}")
                
                # Cache the result 
                self._update_cache(cache_key, data)
                
                self.logger.log_agent_activity("DataEngine", f"Successfully fetched {len(data)} trading days for {ticker}")
                return data
                
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.log_agent_activity("DataEngine", f"Stock data fetch attempt {attempt + 1} failed for {ticker}: {e}", "warning")
                    time.sleep(2 ** attempt)  # Exponential backoff 
                else:
                    self.logger.log_system_error("DataEngine", f"Stock data fetch failed for {ticker} after {max_retries} attempts: {e}")
                    return pd.DataFrame()
        
        return pd.DataFrame()
    
    def fetch_company_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Fetch company fundamental data with improved error handling"""
        cache_key = f"fundamentals_{ticker}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        self.logger.log_agent_activity("DataEngine", f"Fetching fundamentals for {ticker}")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info
                
                if not info or len(info) == 0:
                    raise ValueError(f"No fundamental data available for {ticker}")
                
                # Extract key fundamental metrics
                fundamentals = {
                    "marketCap": info.get("marketCap"),
                    "enterpriseValue": info.get("enterpriseValue"), 
                    "trailingPE": info.get("trailingPE"),
                    "forwardPE": info.get("forwardPE"),
                    "priceToBook": info.get("priceToBook"),
                    "profitMargins": info.get("profitMargins"),
                    "returnOnEquity": info.get("returnOnEquity"),
                    "returnOnAssets": info.get("returnOnAssets"),
                    "revenueGrowth": info.get("revenueGrowth"),
                    "earningsGrowth": info.get("earningsQuarterlyGrowth"),
                    "debtToEquity": info.get("debtToEquity"),
                    "currentRatio": info.get("currentRatio"),
                    "beta": info.get("beta"),
                    "dividendYield": info.get("dividendYield"),
                    "payoutRatio": info.get("payoutRatio"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "businessSummary": info.get("longBusinessSummary", "")[:2000] if info.get("longBusinessSummary") else "",
                    "fullTimeEmployees": info.get("fullTimeEmployees"),
                    "website": info.get("website"),
                    "ebitdaMargins": info.get("ebitdaMargins"),
                    "operatingMargins": info.get("operatingMargins"),
                    "grossMargins": info.get("grossMargins")
                }
                
                # Filter out None values 
                fundamentals = {k: v for k, v in fundamentals.items() if v is not None}
                
                # Cache the result
                self._update_cache(cache_key, fundamentals)
                
                self.logger.log_agent_activity("DataEngine", f"Successfully fetched fundamentals for {ticker}")
                return fundamentals
                
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.log_agent_activity("DataEngine", f"Fundamentals fetch attempt {attempt + 1} failed for {ticker}: {e}", "warning")
                    time.sleep(2 ** attempt)  # Exponential backoff 
                else:
                    self.logger.log_system_error("DataEngine", f"Fundamentals fetch failed for {ticker} after {max_retries} attempts: {e}")
                    return {}
        
        return {}
    
    def fetch_news_data(self, query: str, from_date: str, to_date: str, 
                       max_articles: int = 50) -> List[Dict[str, Any]]:
        """Fetch news data with improved rate limit handling"""
        cache_key = f"news_{hash(query)}_{from_date}_{to_date}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        self.logger.log_agent_activity("DataEngine", f"Fetching news: {query}")
        
        all_articles = []
        
        # Fetch from NewsAPI if available (with rate limit handling)
        if 'news' in self.api_clients:
            try:
                # Add delay to respect rate limits 
                time.sleep(self.config.API_RATE_LIMIT_DELAY)
                
                response = self.api_clients['news'].get_everything(
                    q=query,
                    from_param=from_date,
                    to=to_date,
                    language='en',
                    sort_by='relevancy',
                    page_size=min(max_articles, 50)  # Reduced to avoid rate limits
                )
                
                articles = response.get('articles', [])
                if articles:
                    all_articles.extend(articles)
                    self.logger.log_agent_activity("DataEngine", f"Found {len(articles)} NewsAPI articles")
                
            except Exception as e:
                self.logger.log_system_error("DataEngine", f"NewsAPI fetch failed: {e}")
                # Continue with other sources 
        
        # Also fetch from Google News RSS 
        try:
            rss_articles = self._fetch_google_news_rss(query, days_back=7)
            if rss_articles:
                all_articles.extend(rss_articles)
                self.logger.log_agent_activity("DataEngine", f"Found {len(rss_articles)} Google News articles")
        except Exception as e:
            self.logger.log_system_error("DataEngine", f"Google News RSS fetch failed: {e}")
        
        # Remove duplicates and sort by relevance 
        unique_articles = self._deduplicate_articles(all_articles)[:max_articles]
        
        # Cache the result 
        self._update_cache(cache_key, unique_articles)
        
        self.logger.log_agent_activity("DataEngine", f"Processed {len(unique_articles)} unique articles")
        return unique_articles
    
    def fetch_economic_data(self, series_ids: List[str], start_date: str, 
                          end_date: str) -> Optional[pd.DataFrame]:
        """Fetch economic data with improved error handling"""
        if 'fred' not in self.api_clients or not series_ids:
            return None
        
        cache_key = f"econ_{'_'.join(series_ids)}_{start_date}_{end_date}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        self.logger.log_agent_activity("DataEngine", f"Fetching economic data: {series_ids}")
        
        try:
            data_frames = []
            for series_id in series_ids:
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        # Extend start date to ensure data availability 
                        extended_start = (pd.to_datetime(start_date) - datetime.timedelta(days=90)).strftime('%Y-%m-%d')
                        
                        series_data = self.api_clients['fred'].get_series(
                            series_id,
                            observation_start=extended_start,
                            observation_end=end_date
                        )
                        
                        if not series_data.empty:
                            data_frames.append(series_data.rename(series_id))
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        if attempt < max_retries - 1:
                            self.logger.log_agent_activity("DataEngine", f"FRED fetch attempt {attempt + 1} failed for {series_id}: {e}", "warning")
                            time.sleep(1)
                        else:
                            self.logger.log_system_error("DataEngine", f"Failed to fetch series {series_id}: {e}")
                            continue
                
                time.sleep(0.5)  # Rate limiting
            
            if not data_frames:
                return None
            
            # Combine all series
            df = pd.concat(data_frames, axis=1)
            df = df.ffill().dropna()  # Forward fill and drop NaN
            
            # Cache the result 
            self._update_cache(cache_key, df)
            
            self.logger.log_agent_activity("DataEngine", f"Successfully fetched economic data: {list(df.columns)}")
            return df
            
        except Exception as e:
            self.logger.log_system_error("DataEngine", f"Economic data fetch failed: {e}")
            return None
    
    def _fetch_google_news_rss(self, query: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """Fetch news from Google News RSS with improved error handling"""
        base_url = "https://news.google.com/rss/search"
        params = {
            'q': query,
            'hl': 'en-US',
            'gl': 'US',
            'ceid': 'US:en'
        }
        
        url = f"{base_url}?{urlencode(params)}"
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            articles = []
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_back)
            
            for entry in feed.entries:
                try:
                    # Parse publication date 
                    pub_date = datetime.datetime(*entry.published_parsed[:6])
                    
                    if pub_date < cutoff_date:
                        continue
                    
                    # Extract article information 
                    article = {
                        'title': self._clean_text(entry.title),
                        'description': self._clean_text(entry.get('summary', '')),
                        'url': entry.link,
                        'publishedAt': pub_date.isoformat(),
                        'source': {'name': self._extract_source_name(entry)},
                        'content': entry.get('summary', '')
                    }
                    
                    articles.append(article)
                    
                except Exception as e:
                    self.logger.log_system_error("DataEngine", f"Failed to parse RSS entry: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            self.logger.log_system_error("DataEngine", f"Google News RSS fetch failed: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean text content"""
        if not text:
            return ""
        
        try:
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            # Remove extra whitespace 
            text = ' '.join(text.split())
            # Remove special characters but keep basic punctuation 
            text = re.sub(r'[^\w\s\-.,!?;:]', '', text)
            
            return text.strip()
        except Exception as e:
            self.logger.log_system_error("DataEngine", f"Text cleaning failed: {e}")
            return str(text)  # Return original text on error
    
    def _extract_source_name(self, entry) -> str:
        """Extract news source name"""
        try:
            if hasattr(entry, 'source') and hasattr(entry.source, 'title'):
                return entry.source.title
            else:
                # Extract from URL
                parsed = urlparse(entry.link)
                return parsed.netloc.replace('www.', '')
        except Exception:
            return 'Unknown Source'
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles with improved similarity detection"""
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            try:
                title = article.get('title', '').lower().strip()
                if title and len(title) > 10:
                    # Create a normalized version for duplicate detection
                    normalized_title = re.sub(r'[^\w\s]', '', title)
                    title_words = set(normalized_title.split())
                    
                    # Check for similarity with existing titles
                    is_duplicate = False
                    for seen_title in seen_titles:
                        seen_words = set(re.sub(r'[^\w\s]', '', seen_title).split())
                        if len(title_words & seen_words) / max(len(title_words), len(seen_words)) > 0.7:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        seen_titles.add(title)
                        unique_articles.append(article)
                        
            except Exception as e:
                self.logger.log_system_error("DataEngine", f"Deduplication error: {e}")
                continue
        
        # Sort by publication date (most recent first)
        try:
            unique_articles.sort(
                key=lambda x: x.get('publishedAt', ''), 
                reverse=True
            )
        except Exception as e:
            self.logger.log_system_error("DataEngine", f"Sorting failed: {e}")
        
        return unique_articles
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache is valid"""
        if not self.config.ENABLE_CACHING or cache_key not in self.cache:
            return False
        
        timestamp = self.cache_timestamps.get(cache_key)
        if not timestamp:
            return False
        
        try:
            age_hours = (datetime.datetime.now() - timestamp).total_seconds() / 3600
            return age_hours < self.config.CACHE_EXPIRY_HOURS
        except Exception:
            return False
    
    def _update_cache(self, cache_key: str, data: Any):
        """Update cache"""
        if self.config.ENABLE_CACHING:
            try:
                self.cache[cache_key] = data
                self.cache_timestamps[cache_key] = datetime.datetime.now()
            except Exception as e:
                self.logger.log_system_error("DataEngine", f"Cache update failed: {e}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            total_size = sum(len(str(v)) for v in self.cache.values()) / (1024 * 1024)  # MB
            oldest_cache = 0
            if self.cache_timestamps:
                oldest_cache = min([
                    (datetime.datetime.now() - ts).total_seconds() / 3600 
                    for ts in self.cache_timestamps.values()
                ])
            
            return {
                'total_cached_items': len(self.cache),
                'cache_memory_usage_mb': round(total_size, 2),
                'oldest_cache_hours': round(oldest_cache, 2)
            }
        except Exception as e:
            self.logger.log_system_error("DataEngine", f"Cache statistics failed: {e}")
            return {
                'total_cached_items': 0,
                'cache_memory_usage_mb': 0,
                'oldest_cache_hours': 0
            }


# ============================================
# Base Agent System 
# ============================================

class BaseAgent(ABC):
    """Base agent abstract class with improved error handling and timeout management"""
    
    def __init__(self, agent_id: str, role: AgentRole, config: SystemConfig,
                 logger: MASLogger, mcp_manager: MCPContextManager,
                 data_engine: DataFetchingEngine):
        self.agent_id = agent_id
        self.role = role
        self.config = config
        self.logger = logger
        self.mcp_manager = mcp_manager
        self.data_engine = data_engine
        
        # Agent state management
        self.state = AgentState.INITIALIZING
        self.current_task = None
        self.message_queue = PriorityQueue(maxsize=self.config.MESSAGE_QUEUE_SIZE)
        
        # Performance metrics 
        self.performance_metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_response_time': 0.0,
            'accuracy_score': 0.5,
            'last_activity': datetime.datetime.now()
        }
        
        # Context session management 
        self.context_session_id = None
        
        # Initialize agent 
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize agent with improved error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Create MCP context session 
                initial_context = {
                    'agent_id': self.agent_id,
                    'role': self.role.value,
                    'company': self.config.COMPANY_NAME,
                    'ticker': self.config.STOCK_TICKER,
                    'initialization_time': datetime.datetime.now().isoformat(),
                    'attempt': attempt + 1
                }
                
                self.context_session_id = self.mcp_manager.create_context_session(
                    self.agent_id, 
                    'financial_analysis',
                    initial_context
                )
                
                self.state = AgentState.READY
                self.logger.log_agent_activity(self.agent_id, "Agent initialized successfully")
                return
                
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.log_agent_activity(self.agent_id, f"Agent initialization attempt {attempt + 1} failed: {e}", "warning")
                    time.sleep(1)
                else:
                    self.state = AgentState.ERROR
                    self.logger.log_system_error(self.agent_id, f"Agent initialization failed after {max_retries} attempts: {e}")
    
    @abstractmethod
    async def process_task(self, task: TaskDefinition) -> Dict[str, Any]:
        """Process assigned task - implemented by subclasses"""
        pass
    
    def send_message(self, message: AgentMessage) -> bool:
        """Send message to other agents"""
        try:
            message.sender_id = self.agent_id
            message.timestamp = datetime.datetime.now(datetime.timezone.utc)
            
            # Log the message exchange 
            self.logger.log_message_exchange(
                self.agent_id, 
                message.receiver_id, 
                message.message_type,
                True
            )
            
            return True
            
        except Exception as e:
            self.logger.log_system_error(self.agent_id, f"Failed to send message: {e}")
            self.logger.log_message_exchange(
                self.agent_id,
                message.receiver_id,
                message.message_type, 
                False
            )
            return False
    
    def receive_message(self, message: AgentMessage) -> bool:
        """Receive message from other agents"""
        try:
            # Add to priority queue based on message priority
            priority_value = message.priority.value
            if not self.message_queue.full():
                self.message_queue.put((priority_value, message))
                
                self.logger.log_agent_activity(
                    self.agent_id, 
                    f"Received {message.message_type.value} message from {message.sender_id}"
                )
                return True
            else:
                self.logger.log_system_error(self.agent_id, "Message queue is full")
                return False
            
        except Exception as e:
            self.logger.log_system_error(self.agent_id, f"Failed to receive message: {e}")
            return False
    
    def update_performance_metrics(self, task_successful: bool, response_time: float):
        """Update performance metrics"""
        try:
            if task_successful:
                self.performance_metrics['tasks_completed'] += 1
            else:
                self.performance_metrics['tasks_failed'] += 1
            
            # Update average response time
            total_tasks = (self.performance_metrics['tasks_completed'] + 
                          self.performance_metrics['tasks_failed'])
            
            if total_tasks > 0:
                current_avg = self.performance_metrics['average_response_time']
                new_avg = ((current_avg * (total_tasks - 1)) + response_time) / total_tasks
                self.performance_metrics['average_response_time'] = new_avg
                
                # Update accuracy score
                self.performance_metrics['accuracy_score'] = (
                    self.performance_metrics['tasks_completed'] / total_tasks
                )
            
            self.performance_metrics['last_activity'] = datetime.datetime.now()
            
            # Log performance metric 
            self.logger.log_performance_metric(
                self.agent_id,
                'accuracy_score', 
                self.performance_metrics['accuracy_score']
            )
        except Exception as e:
            self.logger.log_system_error(self.agent_id, f"Performance metrics update failed: {e}")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        try:
            return {
                'agent_id': self.agent_id,
                'role': self.role.value,
                'state': self.state.value,
                'current_task': self.current_task.task_id if self.current_task else None,
                'queue_size': self.message_queue.qsize(),
                'performance_metrics': self.performance_metrics.copy(),
                'context_session_id': self.context_session_id
            }
        except Exception as e:
            self.logger.log_system_error(self.agent_id, f"Failed to get agent status: {e}")
            return {
                'agent_id': self.agent_id,
                'role': self.role.value,
                'state': 'error',
                'error': str(e)
            }


# ============================================
# Specialized Agent Implementations 
# ============================================

class FundamentalAnalysisAgent(BaseAgent):
    """Fundamental analysis agent with improved analysis logic"""
    
    async def process_task(self, task: TaskDefinition) -> Dict[str, Any]:
        """Process fundamental analysis task with timeout and retry"""
        start_time = time.time()
        self.state = AgentState.WORKING
        self.current_task = task
        
        try:
            self.logger.log_agent_activity(self.agent_id, "Starting fundamental analysis")
            
            # Fetch company fundamental data with timeout
            fundamentals = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self.data_engine.fetch_company_fundamentals, self.config.STOCK_TICKER
                ),
                timeout=60  # 60 seconds timeout
            )
            
            if not fundamentals:
                raise ValueError("No fundamental data available")
            
            # Update MCP context with fundamental data
            context_update = {
                'fundamental_data': fundamentals,
                'analysis_timestamp': datetime.datetime.now().isoformat(),
                'data_quality': self._assess_data_quality(fundamentals)
            }
            self.mcp_manager.update_context(self.context_session_id, context_update)
            
            # Generate LLM analysis with timeout 
            analysis_prompt = self._create_fundamental_analysis_prompt(fundamentals)
            
            llm_response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self.mcp_manager.get_llm_response,
                    self.context_session_id, analysis_prompt, self.config.PRIMARY_LLM_PROVIDER
                ),
                timeout=90  # 90 seconds timeout
            )
            
            if not llm_response or "Error:" in llm_response:
                raise ValueError(f"Invalid LLM response: {llm_response}")
            
            # Extract numeric score from LLM response 
            fundamental_score = self.mcp_manager.extract_numeric_score(
                llm_response, 
                default=0.5, 
                min_val=0.0, 
                max_val=1.0
            )
            
            # Validate score 
            if not (0.0 <= fundamental_score <= 1.0):
                fundamental_score = 0.5  # Fallback to neutral
            
            # Prepare result 
            result = {
                'agent_id': self.agent_id,
                'task_id': task.task_id,
                'fundamental_score': fundamental_score,
                'analysis_summary': llm_response[:1000],  # Truncate for storage
                'data_quality': self._assess_data_quality(fundamentals),
                'key_metrics': self._extract_key_metrics(fundamentals),
                'confidence_level': self._calculate_confidence(fundamentals),
                'timestamp': datetime.datetime.now().isoformat(),
                'processing_time': time.time() - start_time
            }
            
            # Update performance metrics 
            self.update_performance_metrics(True, time.time() - start_time)
            self.state = AgentState.READY
            
            self.logger.log_agent_activity(
                self.agent_id, 
                f"Fundamental analysis completed. Score: {fundamental_score:.4f}"
            )
            
            return result
            
        except asyncio.TimeoutError:
            self.update_performance_metrics(False, time.time() - start_time)
            self.state = AgentState.ERROR
            self.logger.log_system_error(self.agent_id, "Fundamental analysis timed out")
            
            return {
                'agent_id': self.agent_id,
                'task_id': task.task_id,
                'fundamental_score': 0.5,  # Neutral fallback score 
                'error': 'Analysis timed out',
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            self.update_performance_metrics(False, time.time() - start_time)
            self.state = AgentState.ERROR
            self.logger.log_system_error(self.agent_id, f"Fundamental analysis failed: {e}")
            
            return {
                'agent_id': self.agent_id,
                'task_id': task.task_id,
                'fundamental_score': 0.3,  # Conservative fallback score 
                'error': str(e),
                'timestamp': datetime.datetime.now().isoformat()
            }
        finally:
            self.current_task = None
    
    def _create_fundamental_analysis_prompt(self, fundamentals: Dict[str, Any]) -> str:
        """Create fundamental analysis prompt with improved structure"""
        # Filter out None values and format data 
        clean_data = {k: v for k, v in fundamentals.items() if v is not None}
        
        # Format key metrics for better readability 
        key_metrics = []
        for key, value in clean_data.items():
            if isinstance(value, (int, float)):
                if abs(value) < 0.01:
                    key_metrics.append(f"{key}: {value:.6f}")
                elif abs(value) < 1:
                    key_metrics.append(f"{key}: {value:.4f}")
                else:
                    key_metrics.append(f"{key}: {value:,.2f}")
            else:
                key_metrics.append(f"{key}: {str(value)[:100]}")  # Truncate long strings
        
        data_str = "\n".join(key_metrics[:30])  # Limit to top 30 metrics
        
        prompt = f"""As a senior quantitative portfolio manager with 20+ years of Wall Street experience, 
conduct a comprehensive fundamental analysis of {self.config.COMPANY_NAME} ({self.config.STOCK_TICKER}).

Analyze these key dimensions:
1. Financial Health: Profitability, liquidity, solvency ratios
2. Valuation: P/E ratios, price-to-book, enterprise value multiples  
3. Growth Prospects: Revenue growth sustainability, earnings trajectory
4. Operational Efficiency: ROE, ROA, margin trends
5. Capital Structure: Debt levels, dividend policy, share buybacks
6. Competitive Position: Market share, moat strength, industry dynamics

Financial Data:
{data_str}

Provide a score from 0.0 (weak fundamentals, potential short candidate) to 1.0 (strong fundamentals, attractive long position).
Focus on quantitative metrics and provide specific reasoning. Consider both absolute values and relative industry position.

Fundamental Score (0.0 to 1.0):"""
        
        return prompt
    
    def _assess_data_quality(self, fundamentals: Dict[str, Any]) -> float:
        """Assess data quality with improved metrics"""
        try:
            if not fundamentals:
                return 0.0
            
            total_fields = len(fundamentals)
            non_null_fields = sum(1 for v in fundamentals.values() if v is not None and v != "")
            
            if total_fields == 0:
                return 0.0
            
            completeness_score = non_null_fields / total_fields
            
            # Check for key financial metrics 
            critical_metrics = ['marketCap', 'trailingPE', 'profitMargins', 'returnOnEquity']
            important_metrics = ['debtToEquity', 'currentRatio', 'revenueGrowth', 'beta']
            
            critical_present = sum(1 for metric in critical_metrics 
                                 if fundamentals.get(metric) is not None)
            important_present = sum(1 for metric in important_metrics 
                                  if fundamentals.get(metric) is not None)
            
            critical_score = critical_present / len(critical_metrics)
            important_score = important_present / len(important_metrics)
            
            # Weighted combination 
            overall_quality = (
                completeness_score * 0.4 + 
                critical_score * 0.4 + 
                important_score * 0.2
            )
            
            return round(min(overall_quality, 1.0), 3)
        except Exception as e:
            self.logger.log_system_error(self.agent_id, f"Data quality assessment failed: {e}")
            return 0.0
    
    def _extract_key_metrics(self, fundamentals: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics with improved categorization"""
        key_metrics = {}
        
        try:
            # Valuation metrics 
            valuation_metrics = {}
            if fundamentals.get('trailingPE'):
                valuation_metrics['trailing_pe'] = fundamentals['trailingPE']
            if fundamentals.get('forwardPE'):
                valuation_metrics['forward_pe'] = fundamentals['forwardPE']
            if fundamentals.get('priceToBook'):
                valuation_metrics['price_to_book'] = fundamentals['priceToBook']
            
            # Profitability metrics 
            profitability_metrics = {}
            for metric, key in [('profitMargins', 'profit_margins'), 
                               ('grossMargins', 'gross_margins'),
                               ('operatingMargins', 'operating_margins'),
                               ('returnOnEquity', 'roe'), 
                               ('returnOnAssets', 'roa')]:
                if fundamentals.get(metric):
                    profitability_metrics[key] = fundamentals[metric]
            
            # Growth metrics 
            growth_metrics = {}
            if fundamentals.get('revenueGrowth'):
                growth_metrics['revenue_growth'] = fundamentals['revenueGrowth']
            if fundamentals.get('earningsGrowth'):
                growth_metrics['earnings_growth'] = fundamentals['earningsGrowth']
            
            # Financial strength 
            financial_strength = {}
            if fundamentals.get('debtToEquity'):
                financial_strength['debt_to_equity'] = fundamentals['debtToEquity']
            if fundamentals.get('currentRatio'):
                financial_strength['current_ratio'] = fundamentals['currentRatio']
            if fundamentals.get('beta'):
                financial_strength['beta'] = fundamentals['beta']
                
            key_metrics = {
                'valuation': valuation_metrics,
                'profitability': profitability_metrics,
                'growth': growth_metrics,
                'financial_strength': financial_strength
            }
                
        except Exception as e:
            self.logger.log_system_error(self.agent_id, f"Key metrics extraction failed: {e}")
        
        return key_metrics
    
    def _calculate_confidence(self, fundamentals: Dict[str, Any]) -> float:
        """Calculate analysis confidence level with improved logic"""
        try:
            data_quality = self._assess_data_quality(fundamentals)
            
            # Factor in data completeness 
            confidence = data_quality
            
            # Adjust based on company size (larger companies typically have more reliable data)
            market_cap = fundamentals.get('marketCap')
            if market_cap:
                if market_cap > 100_000_000_000:  # > $100B
                    confidence += 0.15
                elif market_cap > 10_000_000_000:  # > $10B  
                    confidence += 0.10
                elif market_cap > 1_000_000_000:  # > $1B
                    confidence += 0.05
            
            # Adjust for data consistency / 
            pe_ratio = fundamentals.get('trailingPE')
            if pe_ratio and 0 < pe_ratio < 100:  # Reasonable P/E range
                confidence += 0.05
            
            profit_margins = fundamentals.get('profitMargins')
            if profit_margins and profit_margins > 0:  # Positive margins
                confidence += 0.05
            
            return round(min(confidence, 1.0), 3)
        except Exception as e:
            self.logger.log_system_error(self.agent_id, f"Confidence calculation failed: {e}")
            return 0.5


class SentimentAnalysisAgent(BaseAgent):
    """Sentiment analysis agent with improved NLP and error handling"""
    
    async def process_task(self, task: TaskDefinition) -> Dict[str, Any]:
        """Process sentiment analysis task with improved timeout handling"""
        start_time = time.time()
        self.state = AgentState.WORKING
        self.current_task = task
        
        try:
            self.logger.log_agent_activity(self.agent_id, "Starting sentiment analysis")
            
            # Fetch recent news data with timeout
            news_articles = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self.data_engine.fetch_news_data,
                    f'"{self.config.COMPANY_NAME}" OR "{self.config.STOCK_TICKER}"',
                    (datetime.datetime.now() - datetime.timedelta(days=self.config.NEWS_LOOKBACK_DAYS)).strftime('%Y-%m-%d'),
                    datetime.datetime.now().strftime('%Y-%m-%d'),
                    30
                ),
                timeout=120  # 2 minutes timeout
            )
            
            if not news_articles:
                self.logger.log_agent_activity(self.agent_id, "No news articles found", "warning")
                return self._create_fallback_result(task, 0.0, "No news data available")
            
            # Update MCP context with news data
            context_update = {
                'news_articles': news_articles[:5],  # Store top 5 articles in context
                'total_articles_analyzed': len(news_articles),
                'analysis_timestamp': datetime.datetime.now().isoformat()
            }
            self.mcp_manager.update_context(self.context_session_id, context_update)
            
            # Analyze sentiment of top articles with timeout 
            sentiment_scores = []
            sentiment_details = []
            
            # Limit analysis to prevent timeout 
            articles_to_analyze = min(5, len(news_articles))
            
            for i, article in enumerate(news_articles[:articles_to_analyze]):
                try:
                    article_text = f"{article.get('title', '')} {article.get('description', '')}"
                    
                    if len(article_text.strip()) > 20:
                        sentiment_prompt = self._create_sentiment_analysis_prompt(article_text, article)
                        
                        # Analyze with timeout 
                        llm_response = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                None, self.mcp_manager.get_llm_response,
                                self.context_session_id, sentiment_prompt, self.config.PRIMARY_LLM_PROVIDER
                            ),
                            timeout=60  # 60 seconds per article
                        )
                        
                        if llm_response and "Error:" not in llm_response:
                            article_sentiment = self.mcp_manager.extract_numeric_score(
                                llm_response,
                                default=0.0,
                                min_val=-1.0,
                                max_val=1.0
                            )
                            
                            sentiment_scores.append(article_sentiment)
                            sentiment_details.append({
                                'article_title': article.get('title', '')[:100],
                                'sentiment_score': article_sentiment,
                                'analysis': llm_response[:500],  # Truncate for storage
                                'source': article.get('source', {}).get('name', 'Unknown')
                            })
                        
                        # Rate limiting 
                        await asyncio.sleep(self.config.API_RATE_LIMIT_DELAY)
                        
                except asyncio.TimeoutError:
                    self.logger.log_agent_activity(self.agent_id, f"Sentiment analysis timed out for article {i+1}", "warning")
                    break  # Stop processing remaining articles
                except Exception as e:
                    self.logger.log_system_error(self.agent_id, f"Failed to analyze article {i+1}: {e}")
                    continue
            
            # Calculate overall sentiment score 
            if sentiment_scores:
                # Use weighted average (more recent articles have higher weight)
                weights = [0.4, 0.3, 0.2, 0.1, 0.05][:len(sentiment_scores)]
                total_weight = sum(weights)
                weighted_sentiment = sum(score * weight for score, weight in zip(sentiment_scores, weights)) / total_weight
                
                # Apply smoothing to prevent extreme values 
                weighted_sentiment = np.tanh(weighted_sentiment * 0.8) * 0.9
            else:
                weighted_sentiment = 0.0
            
            # Prepare result 
            result = {
                'agent_id': self.agent_id,
                'task_id': task.task_id,
                'sentiment_score': weighted_sentiment,
                'articles_analyzed': len(sentiment_scores),
                'total_articles_found': len(news_articles),
                'sentiment_details': sentiment_details,
                'confidence_level': self._calculate_sentiment_confidence(news_articles, sentiment_scores),
                'market_sentiment_indicators': self._extract_market_indicators(news_articles),
                'timestamp': datetime.datetime.now().isoformat(),
                'processing_time': time.time() - start_time
            }
            
            # Update performance metrics 
            self.update_performance_metrics(True, time.time() - start_time)
            self.state = AgentState.READY
            
            self.logger.log_agent_activity(
                self.agent_id,
                f"Sentiment analysis completed. Score: {weighted_sentiment:.4f} (analyzed {len(sentiment_scores)} articles)"
            )
            
            return result
            
        except asyncio.TimeoutError:
            self.update_performance_metrics(False, time.time() - start_time)
            self.state = AgentState.ERROR
            self.logger.log_system_error(self.agent_id, "Sentiment analysis timed out")
            
            return self._create_fallback_result(task, 0.0, "Analysis timed out")
            
        except Exception as e:
            self.update_performance_metrics(False, time.time() - start_time)
            self.state = AgentState.ERROR
            self.logger.log_system_error(self.agent_id, f"Sentiment analysis failed: {e}")
            
            return self._create_fallback_result(task, 0.0, str(e))
        finally:
            self.current_task = None
    
    def _create_sentiment_analysis_prompt(self, article_text: str, article: Dict[str, Any]) -> str:
        """Create sentiment analysis prompt with improved structure"""
        source = article.get('source', {}).get('name', 'Unknown')
        pub_date = article.get('publishedAt', 'Unknown')
        
        prompt = f"""As an experienced Wall Street quantitative analyst specializing in sentiment analysis, 
evaluate the market sentiment of this financial news regarding {self.config.COMPANY_NAME} ({self.config.STOCK_TICKER}).

Focus on actionable trading signals that could impact algorithmic trading decisions:
1. Direct impact on stock price momentum 
2. Investor confidence indicators 
3. Institutional sentiment shifts 
4. Market perception changes 
5. Risk factors emergence 

News Details:
- Source: {source}
- Published: {pub_date}
- Content: "{article_text[:1500]}"

Analysis Framework:
- News credibility and source reliability
- Magnitude of potential market impact  
- Time sensitivity of the information
- Correlation with historical price movements
- Forward-looking implications

Provide a sentiment score from -1.0 (strong negative/sell signal) to +1.0 (strong positive/buy signal).
0.0 indicates neutral or no tradeable impact.

Sentiment Score (-1.0 to 1.0):"""
        
        return prompt
    
    def _calculate_sentiment_confidence(self, news_articles: List[Dict], sentiment_scores: List[float]) -> float:
        """Calculate sentiment analysis confidence level with improved metrics"""
        if not sentiment_scores or not news_articles:
            return 0.0
        
        try:
            # Factor 1: Number of articles analyzed 
            article_count_factor = min(len(sentiment_scores) / 5.0, 1.0)
            
            # Factor 2: Sentiment consistency 
            if len(sentiment_scores) > 1:
                sentiment_std = np.std(sentiment_scores)
                consistency_factor = max(0.0, 1.0 - sentiment_std * 2)  # Penalize high volatility
            else:
                consistency_factor = 0.3  # Lower confidence for single article
            
            # Factor 3: Source diversity 
            sources = set(article.get('source', {}).get('name', 'Unknown') for article in news_articles[:5])
            source_diversity_factor = min(len(sources) / 3.0, 1.0)
            
            # Factor 4: News recency 
            now = datetime.datetime.now()
            recent_articles = 0
            for article in news_articles[:5]:
                try:
                    pub_date_str = article.get('publishedAt', '')
                    if pub_date_str:
                        pub_date = datetime.datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                        if (now - pub_date.replace(tzinfo=None)).days <= 1:
                            recent_articles += 1
                except Exception:
                    pass
            
            recency_factor = recent_articles / min(len(news_articles), 5)
            
            # Factor 5: Article quality (length and completeness) 
            quality_scores = []
            for article in news_articles[:5]:
                title_len = len(article.get('title', ''))
                desc_len = len(article.get('description', ''))
                quality_score = min((title_len + desc_len) / 200.0, 1.0)
                quality_scores.append(quality_score)
            
            quality_factor = np.mean(quality_scores) if quality_scores else 0.0
            
            # Combine factors with weights 
            overall_confidence = (
                article_count_factor * 0.25 +
                consistency_factor * 0.25 +
                source_diversity_factor * 0.2 +
                recency_factor * 0.15 +
                quality_factor * 0.15
            )
            
            return round(min(overall_confidence, 1.0), 3)
        except Exception as e:
            self.logger.log_system_error(self.agent_id, f"Confidence calculation failed: {e}")
            return 0.0
    
    def _extract_market_indicators(self, news_articles: List[Dict]) -> Dict[str, Any]:
        """Extract market sentiment indicators with improved keyword analysis"""
        indicators = {
            'positive_keywords': 0,
            'negative_keywords': 0,
            'earnings_related': 0,
            'analyst_upgrades': 0,
            'analyst_downgrades': 0,
            'regulatory_mentions': 0,
            'growth_mentions': 0,
            'ai_mentions': 0,           # Specific to tech companies
            'market_share_mentions': 0
        }
        
        try:
            # Enhanced keyword categories 
            positive_keywords = ['growth', 'profit', 'gain', 'increase', 'beat', 'exceed', 'strong', 'positive', 
                               'upgrade', 'buy', 'outperform', 'bullish', 'expansion', 'success', 'breakthrough']
            negative_keywords = ['loss', 'decline', 'fall', 'weak', 'miss', 'cut', 'downgrade', 'sell', 
                               'underperform', 'concern', 'risk', 'bearish', 'layoffs', 'recession', 'crisis']
            earnings_keywords = ['earnings', 'revenue', 'guidance', 'forecast', 'outlook', 'results', 'eps']
            growth_keywords = ['growth', 'expand', 'scaling', 'increasing', 'rising', 'surge']
            ai_keywords = ['artificial intelligence', 'ai', 'machine learning', 'neural', 'algorithm']
            
            for article in news_articles:
                text = f"{article.get('title', '')} {article.get('description', '')}".lower()
                
                # Count keyword occurrences with better normalization 
                indicators['positive_keywords'] += sum(1 for keyword in positive_keywords if keyword in text)
                indicators['negative_keywords'] += sum(1 for keyword in negative_keywords if keyword in text)
                indicators['earnings_related'] += sum(1 for keyword in earnings_keywords if keyword in text)
                indicators['growth_mentions'] += sum(1 for keyword in growth_keywords if keyword in text)
                indicators['ai_mentions'] += sum(1 for keyword in ai_keywords if keyword in text)
                
                # Specific pattern detection with improved regex 
                if any(phrase in text for phrase in ['upgrade', 'raised target', 'increased rating', 'price target raised']):
                    indicators['analyst_upgrades'] += 1
                if any(phrase in text for phrase in ['downgrade', 'lowered target', 'reduced rating', 'price target cut']):
                    indicators['analyst_downgrades'] += 1
                if any(phrase in text for phrase in ['regulation', 'sec', 'investigation', 'lawsuit', 'compliance']):
                    indicators['regulatory_mentions'] += 1
                if any(phrase in text for phrase in ['market share', 'competitive position', 'market leadership']):
                    indicators['market_share_mentions'] += 1
            
            # Normalize by number of articles 
            total_articles = max(len(news_articles), 1)
            normalized_indicators = {
                key: value / total_articles 
                for key, value in indicators.items()
            }
            
            return normalized_indicators
            
        except Exception as e:
            self.logger.log_system_error(self.agent_id, f"Market indicators extraction failed: {e}")
            return indicators
    
    def _create_fallback_result(self, task: TaskDefinition, score: float, reason: str) -> Dict[str, Any]:
        """Create fallback result with improved error handling"""
        return {
            'agent_id': self.agent_id,
            'task_id': task.task_id,
            'sentiment_score': score,
            'articles_analyzed': 0,
            'total_articles_found': 0,
            'sentiment_details': [],
            'confidence_level': 0.0,
            'market_sentiment_indicators': {},
            'fallback_reason': reason,
            'timestamp': datetime.datetime.now().isoformat()
        }


# ============================================
# Main MAS System Controller 
# ============================================

class MASFinancialSystem:
    """MAS Financial System main controller with improved error handling and performance"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = MASLogger(config)
        self.mcp_manager = MCPContextManager(config, self.logger)
        self.data_engine = DataFetchingEngine(config, self.logger)
        
        # Agent management 
        self.agents = {}
        self.active_tasks = {}
        self.system_metrics = {}
        
        # Initialize system 
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize MAS system with improved error handling"""
        try:
            self.logger.log_agent_activity("MAS_System", "Initializing MAS Financial System")
            
            # Validate configuration
            if not self.config.validate_configuration():
                raise RuntimeError("Configuration validation failed")
            
            # Initialize core agents 
            self._initialize_agents()
            
            self.logger.log_agent_activity("MAS_System", f"System initialized with {len(self.agents)} agents")
            
        except Exception as e:
            self.logger.log_system_error("MAS_System", f"System initialization failed: {e}")
            raise
    
    def _initialize_agents(self):
        """Initialize agents with improved error handling"""
        # Factor analysis agents
        agent_configs = [
            (AgentRole.FUNDAMENTAL_ANALYST, FundamentalAnalysisAgent),
            (AgentRole.SENTIMENT_ANALYST, SentimentAnalysisAgent),
            # Additional agents can be added here 
        ]
        
        for role, agent_class in agent_configs:
            agent_id = f"{role.value}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    agent = agent_class(
                        agent_id=agent_id,
                        role=role,
                        config=self.config,
                        logger=self.logger,
                        mcp_manager=self.mcp_manager,
                        data_engine=self.data_engine
                    )
                    
                    if agent.state == AgentState.READY:
                        self.agents[agent_id] = agent
                        self.logger.log_agent_activity("MAS_System", f"Initialized agent: {agent_id}")
                        break
                    else:
                        raise Exception(f"Agent not ready after initialization: {agent.state}")
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.log_agent_activity("MAS_System", f"Agent initialization attempt {attempt + 1} failed for {role.value}: {e}", "warning")
                        time.sleep(1)
                    else:
                        self.logger.log_system_error("MAS_System", f"Failed to initialize agent {role.value} after {max_retries} attempts: {e}")
    
    async def generate_factor_scores(self, analysis_date: datetime.date) -> Dict[str, float]:
        """Generate factor scores for specific date with improved error handling"""
        self.logger.log_agent_activity("MAS_System", f"Generating factor scores for {analysis_date}")
        
        # Create tasks for each factor analysis agent
        tasks = []
        
        # Fundamental Analysis Task
        fundamental_agent_id = self._get_agent_by_role(AgentRole.FUNDAMENTAL_ANALYST)
        if fundamental_agent_id:
            fundamental_task = TaskDefinition(
                title="Fundamental Analysis",
                description=f"Analyze fundamental metrics for {self.config.COMPANY_NAME} ({self.config.STOCK_TICKER})",
                assigned_agent=fundamental_agent_id,
                priority=TaskPriority.HIGH,
                expected_output="Fundamental score between 0.0 and 1.0",
                deadline=datetime.datetime.now() + datetime.timedelta(seconds=self.config.AGENT_TIMEOUT_SECONDS)
            )
            tasks.append(fundamental_task)
        
        # Sentiment Analysis Task
        sentiment_agent_id = self._get_agent_by_role(AgentRole.SENTIMENT_ANALYST)
        if sentiment_agent_id:
            sentiment_task = TaskDefinition(
                title="Sentiment Analysis", 
                description=f"Analyze market sentiment for {self.config.COMPANY_NAME} ({self.config.STOCK_TICKER})",
                assigned_agent=sentiment_agent_id,
                priority=TaskPriority.HIGH,
                expected_output="Sentiment score between -1.0 and 1.0",
                deadline=datetime.datetime.now() + datetime.timedelta(seconds=self.config.AGENT_TIMEOUT_SECONDS)
            )
            tasks.append(sentiment_task)
        
        if not tasks:
            self.logger.log_system_error("MAS_System", "No agents available for task execution")
            return self._get_default_factor_scores()
        
        # Execute tasks with improved error handling 
        try:
            if self.config.PARALLEL_PROCESSING and len(tasks) > 1:
                results = await self._execute_tasks_parallel(tasks)
            else:
                results = await self._execute_tasks_sequential(tasks)
            
            # Compile factor scores 
            factor_scores = self._compile_factor_scores(results)
            
            self.logger.log_agent_activity("MAS_System", f"Factor scores generated: {factor_scores}")
            
            return factor_scores
            
        except Exception as e:
            self.logger.log_system_error("MAS_System", f"Factor score generation failed: {e}")
            return self._get_default_factor_scores()
    
    def _get_agent_by_role(self, role: AgentRole) -> Optional[str]:
        """Get agent ID by role with state checking"""
        for agent_id, agent in self.agents.items():
            if agent.role == role and agent.state in [AgentState.READY, AgentState.WAITING]:
                return agent_id
        return None
    
    async def _execute_tasks_parallel(self, tasks: List[TaskDefinition]) -> List[Dict[str, Any]]:
        """Execute tasks in parallel with improved timeout handling"""
        async def execute_task_with_timeout(task):
            agent_id = task.assigned_agent
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                try:
                    # Check agent state before execution 
                    if agent.state != AgentState.READY:
                        return {'error': f'Agent not ready: {agent.state}', 'agent_id': agent_id, 'task_id': task.task_id}
                    
                    # Execute task with timeout 
                    result = await asyncio.wait_for(
                        agent.process_task(task),
                        timeout=self.config.AGENT_TIMEOUT_SECONDS
                    )
                    return result
                    
                except asyncio.TimeoutError:
                    self.logger.log_system_error("MAS_System", f"Task execution timeout for {agent_id}")
                    agent.state = AgentState.ERROR  # Mark agent as error state
                    return {'error': 'Timeout', 'agent_id': agent_id, 'task_id': task.task_id}
                    
                except Exception as e:
                    self.logger.log_system_error("MAS_System", f"Task execution failed for {agent_id}: {e}")
                    agent.state = AgentState.ERROR  # Mark agent as error state
                    return {'error': f'Execution failed: {e}', 'agent_id': agent_id, 'task_id': task.task_id}
            else:
                return {'error': f'Agent not found: {agent_id}', 'agent_id': agent_id, 'task_id': task.task_id}
        
        # Execute all tasks concurrently 
        try:
            results = await asyncio.gather(
                *[execute_task_with_timeout(task) for task in tasks],
                return_exceptions=True
            )
            
            # Handle exceptions in results 
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        'error': f'Task execution exception: {result}', 
                        'agent_id': tasks[i].assigned_agent,
                        'task_id': tasks[i].task_id
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            self.logger.log_system_error("MAS_System", f"Parallel execution failed: {e}")
            return [{'error': f'Parallel execution failed: {e}', 'agent_id': 'unknown', 'task_id': 'unknown'}] * len(tasks)
    
    async def _execute_tasks_sequential(self, tasks: List[TaskDefinition]) -> List[Dict[str, Any]]:
        """Execute tasks sequentially with improved error handling"""
        results = []
        
        for task in tasks:
            agent_id = task.assigned_agent
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                try:
                    # Check agent state before execution
                    if agent.state != AgentState.READY:
                        self.logger.log_agent_activity("MAS_System", f"Agent {agent_id} not ready: {agent.state}", "warning")
                        results.append({'error': f'Agent not ready: {agent.state}', 'agent_id': agent_id, 'task_id': task.task_id})
                        continue
                    
                    result = await asyncio.wait_for(
                        agent.process_task(task),
                        timeout=self.config.AGENT_TIMEOUT_SECONDS
                    )
                    results.append(result)
                    
                except asyncio.TimeoutError:
                    self.logger.log_system_error("MAS_System", f"Agent {agent_id} timed out")
                    agent.state = AgentState.ERROR
                    results.append({'error': 'Timeout', 'agent_id': agent_id, 'task_id': task.task_id})
                    
                except Exception as e:
                    self.logger.log_system_error("MAS_System", f"Task execution failed for {agent_id}: {e}")
                    agent.state = AgentState.ERROR
                    results.append({'error': str(e), 'agent_id': agent_id, 'task_id': task.task_id})
            else:
                results.append({'error': f'Agent not found: {agent_id}', 'agent_id': agent_id, 'task_id': task.task_id})
        
        return results
    
    def _compile_factor_scores(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compile factor scores from agent results with improved validation"""
        factor_scores = self._get_default_factor_scores()
        
        try:
            for result in results:
                if 'error' not in result:
                    # Map agent results to factor scores with validation
                    if 'fundamental_score' in result:
                        score = float(result['fundamental_score'])
                        if 0.0 <= score <= 1.0:
                            factor_scores['fundamental_score'] = score
                        else:
                            self.logger.log_system_error("MAS_System", f"Invalid fundamental score: {score}")
                    
                    if 'sentiment_score' in result:
                        score = float(result['sentiment_score'])
                        if -1.0 <= score <= 1.0:
                            # Convert sentiment score to 0-1 range for consistency 
                            normalized_score = (score + 1.0) / 2.0
                            factor_scores['sentiment_score'] = normalized_score
                        else:
                            self.logger.log_system_error("MAS_System", f"Invalid sentiment score: {score}")
                else:
                    self.logger.log_system_error("MAS_System", f"Agent error in results: {result['error']}")
        except Exception as e:
            self.logger.log_system_error("MAS_System", f"Factor score compilation failed: {e}")
        
        return factor_scores
    
    def _get_default_factor_scores(self) -> Dict[str, float]:
        """Get default factor scores"""
        return {
            'fundamental_score': 0.5,        # Default neutral score 
            'sentiment_score': 0.01,         # Default neutral sentiment 
            'industry_trend_score': 0.5,     # Placeholder for future implementation 
            'market_risk_factor': 0.01,      # Placeholder for future implementation 
            'black_swan_risk': 0.01          # Default low black swan risk 
        }
    
    def run_analysis_pipeline(self) -> pd.DataFrame:
        """Run complete analysis pipeline with improved error handling"""
        self.logger.log_agent_activity("MAS_System", "Starting analysis pipeline")
        
        try:
            # Fetch stock data
            stock_data = self.data_engine.fetch_stock_data(
                self.config.STOCK_TICKER,
                self.config.START_DATE,
                self.config.END_DATE
            )
            
            if stock_data.empty:
                self.logger.log_system_error("MAS_System", "No stock data available")
                return pd.DataFrame()
            
            # Process each trading day 
            results = []
            trading_dates = stock_data.index
            
            self.logger.log_agent_activity("MAS_System", f"Processing {len(trading_dates)} trading days")
            
            # Create and run async tasks            
            try:
                loop = asyncio.get_running_loop()
                # We're in an existing event loop, use thread executor 
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_pipeline_in_new_loop, trading_dates)
                    results = future.result()
                    
            except RuntimeError:
                # No event loop running, use asyncio.run 
                results = asyncio.run(self._process_trading_dates_async(trading_dates))
            
            # Show progress 
            print(f"Processed {len(results)} trading days")
            
            # Create results DataFrame
            if results:
                results_df = pd.DataFrame(results)
                results_df['Date'] = pd.to_datetime(results_df['Date'])
                results_df.set_index('Date', inplace=True)
                
                # Ensure data directory exists 
                Path(self.config.DATA_DIR).mkdir(exist_ok=True)
                
                # Save results 
                output_path = Path(self.config.DATA_DIR) / self.config.OUTPUT_CSV
                results_df.to_csv(output_path, encoding='utf-8-sig')
                
                self.logger.log_agent_activity("MAS_System", f"Results saved to: {output_path}")
                self.logger.log_agent_activity("MAS_System", f"Generated {len(results_df)} factor score records")
                
                return results_df
            else:
                self.logger.log_system_error("MAS_System", "No factor scores generated")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.log_system_error("MAS_System", f"Analysis pipeline failed: {e}")
            return pd.DataFrame()
    
    def _run_pipeline_in_new_loop(self, trading_dates):
        """Run pipeline in a new event loop"""
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(self._process_trading_dates_async(trading_dates))
            finally:
                new_loop.close()
        
        import threading
        result_container = []
        exception_container = []
        
        def thread_target():
            try:
                result = run_in_thread()
                result_container.append(result)
            except Exception as e:
                exception_container.append(e)
        
        thread = threading.Thread(target=thread_target)
        thread.start()
        thread.join()
        
        if exception_container:
            raise exception_container[0]
        
        return result_container[0] if result_container else []
    
    async def _process_trading_dates_async(self, trading_dates):
        """Process trading dates asynchronously"""
        results = []
        
        for date_dt in trading_dates:
            try:
                self.logger.log_agent_activity("MAS_System", f"Generating factor scores for {date_dt.strftime('%Y-%m-%d')}")
                
                factor_scores = await self.generate_factor_scores(date_dt.date())
                result_row = {'Date': date_dt.strftime('%Y-%m-%d')}
                result_row.update(factor_scores)
                results.append(result_row)
                
                # Rate limiting between dates
                await asyncio.sleep(self.config.API_RATE_LIMIT_DELAY)
                
            except Exception as e:
                self.logger.log_system_error("MAS_System", f"Failed to process {date_dt.strftime('%Y-%m-%d')}: {e}")
                result_row = {
                    'Date': date_dt.strftime('%Y-%m-%d'),
                    **self._get_default_factor_scores(),
                    'error': str(e)
                }
                results.append(result_row)
        
        return results
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get system health report with improved metrics"""
        try:
            health_report = self.logger.get_system_health_report()
            
            # Add agent-specific metrics 
            agent_metrics = {}
            healthy_agents = 0
            error_agents = 0
            
            for agent_id, agent in self.agents.items():
                try:
                    status = agent.get_agent_status()
                    agent_metrics[agent_id] = status
                    
                    if status.get('state') == 'ready':
                        healthy_agents += 1
                    elif status.get('state') == 'error':
                        error_agents += 1
                        
                except Exception as e:
                    agent_metrics[agent_id] = {'error': str(e)}
                    error_agents += 1
            
            health_report['agents'] = agent_metrics
            health_report['healthy_agents'] = healthy_agents
            health_report['error_agents'] = error_agents
            health_report['cache_stats'] = self.data_engine.get_cache_statistics()
            health_report['mcp_contexts'] = len(self.mcp_manager.contexts)
            health_report['system_status'] = 'healthy' if error_agents == 0 else 'degraded'
            
            return health_report
        except Exception as e:
            self.logger.log_system_error("MAS_System", f"Health report generation failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.datetime.now(),
                'status': 'unhealthy'
            }
    
    def cleanup_system(self):
        """Cleanup system resources with improved error handling"""
        try:
            # Cleanup expired MCP contexts 
            expired_contexts = self.mcp_manager.cleanup_expired_contexts(24)
            if expired_contexts > 0:
                self.logger.log_agent_activity("MAS_System", f"Cleaned up {expired_contexts} expired contexts")
            
            # Reset agent states 
            reset_agents = 0
            for agent_id, agent in self.agents.items():
                try:
                    if agent.state == AgentState.ERROR:
                        agent.state = AgentState.READY
                        reset_agents += 1
                        self.logger.log_agent_activity("MAS_System", f"Reset agent {agent_id} from error state")
                except Exception as e:
                    self.logger.log_system_error("MAS_System", f"Failed to reset agent {agent_id}: {e}")
            
            if reset_agents > 0:
                self.logger.log_agent_activity("MAS_System", f"Reset {reset_agents} agents from error state")
            
            self.logger.log_agent_activity("MAS_System", "System cleanup completed")
            
        except Exception as e:
            self.logger.log_system_error("MAS_System", f"System cleanup failed: {e}")


# ============================================
# Main Execution Function 
# ============================================

def main():
    """Main execution function - MAS Financial Factor Scoring System entry point"""
    try:
        print("\n🚀 Starting Advanced MAS Financial Factor Scoring System")
        print("="*80)
        
        # Initialize system configuration 
        config = SystemConfig()
        
        # Override with environment variables if available 
        load_dotenv()
        config.PRIMARY_LLM_PROVIDER = os.getenv("LLM_PROVIDER", config.PRIMARY_LLM_PROVIDER)
        config.START_DATE = os.getenv("ANALYSIS_START_DATE", config.START_DATE)
        config.END_DATE = os.getenv("ANALYSIS_END_DATE", config.END_DATE)
        
        # Display system information 
        print(f"📊 Analysis Target: {config.COMPANY_NAME} ({config.STOCK_TICKER})")
        print("-"*80)
        print(f"📅 Analysis Period: {config.START_DATE} to {config.END_DATE}")
        print("-"*80)
        print(f"🤖 Primary LLM: {config.PRIMARY_LLM_PROVIDER.upper()}")
        print("-"*80)
        print(f"🔧 System Version: MAS Production-Ready Enhanced Edition")
        print("-"*80)
        print(f"🏭 Industry Keywords: {len(config.INDUSTRY_KEYWORDS)} specialized terms")
        print("-"*80)
        
        # Display available components 
        print(f"\n🔌 Available Components:")
        print(f"   • OpenAI: {'✓' if OPENAI_AVAILABLE else '✗'}")
        print(f"   • Anthropic: {'✓' if ANTHROPIC_AVAILABLE else '✗'}")
        print(f"   • Google AI: {'✓' if GOOGLE_AVAILABLE else '✗'}")
        print(f"   • NewsAPI: {'✓' if NEWSAPI_AVAILABLE else '✗'}")
        print(f"   • FRED API: {'✓' if FRED_AVAILABLE else '✗'}")
        print(f"   • CrewAI: {'✓' if CREWAI_AVAILABLE else '✗'}")
        print(f"   • AutoGen: {'✓' if AUTOGEN_AVAILABLE else '✗'}")
        print()
        
        # Initialize and run MAS system 
        mas_system = MASFinancialSystem(config)
        
        # Run the analysis pipeline 
        print("\n🔄 Starting factor analysis pipeline...")
        print("="*80)
        
        results_df = mas_system.run_analysis_pipeline()
        
        if not results_df.empty:
            print(f"\n✅ Analysis completed successfully!")
            print(f"📈 Generated factor scores for {len(results_df)} trading days")
            print(f"💾 Results saved to: {Path(config.DATA_DIR) / config.OUTPUT_CSV}")
            
            # Display sample results 
            print(f"\n📊 Sample Factor Scores for {config.COMPANY_NAME}:")
            print(results_df.head().round(4))
            
            # Display statistical summary 
            print(f"\n📈 Statistical Summary:")
            for column in config.FACTOR_COLUMNS:
                if column in results_df.columns:
                    mean_val = results_df[column].mean()
                    std_val = results_df[column].std()
                    min_val = results_df[column].min()
                    max_val = results_df[column].max()
                    print(f"   • {column}: Mean={mean_val:.4f}, Std={std_val:.4f}, Range=[{min_val:.4f}, {max_val:.4f}]")
            
            # Display system health report
            health_report = mas_system.get_system_health_report()
            print(f"\n🤖 System Health Report:")
            print("="*80)
            print(f"   • Active Agents: {health_report.get('active_agents', 0)}")
            print(f"   • Healthy Agents: {health_report.get('healthy_agents', 0)}")
            print(f"   • Error Agents: {health_report.get('error_agents', 0)}")
            print(f"   • Total Messages: {health_report.get('total_messages', 0)}")
            print(f"   • Success Rate: {health_report.get('message_success_rate', 0):.1%}")
            
            # Cache statistics 
            cache_stats = health_report.get('cache_stats', {})
            print(f"   • Cache Items: {cache_stats.get('total_cached_items', 0)}")
            print(f"   • Cache Memory: {cache_stats.get('cache_memory_usage_mb', 0):.2f} MB")
            
            print(f"\n🎉 System execution completed successfully!")
            
        else:
            print("❌ Factor generation failed - no results generated")
            return 1
        
        # Cleanup system resources 
        mas_system.cleanup_system()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ User interrupted the analysis process")
        return 1
    except Exception as e:
        print(f"\n❌ System execution failed: {e}")
        traceback.print_exc()
        return 1


def setup_environment():
    """Setup environment and check dependencies"""
    print("🔧 Setting up environment...")
    
    # Check Python version 
    import sys
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check required packages 
    required_packages = [
        ('pandas', 'pandas'), 
        ('numpy', 'numpy'), 
        ('yfinance', 'yfinance'), 
        ('requests', 'requests'), 
        ('python-dotenv', 'dotenv'), 
        ('tqdm', 'tqdm')
    ]
    
    missing_packages = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("   Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ Environment setup complete")
    return True


def run_system_diagnostics():
    """Run comprehensive system diagnostics"""
    print("🔍 Running system diagnostics...")
    
    diagnostics = {
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'available_libraries': {
            'pandas': pd.__version__,
            'numpy': np.__version__,
            'yfinance': getattr(yf, '__version__', 'unknown'),
            'requests': requests.__version__,
        },
        'optional_libraries': {
            'openai': OPENAI_AVAILABLE,
            'anthropic': ANTHROPIC_AVAILABLE,
            'google': GOOGLE_AVAILABLE,
            'newsapi': NEWSAPI_AVAILABLE,
            'fred': FRED_AVAILABLE,
            'crewai': CREWAI_AVAILABLE,
            'autogen': AUTOGEN_AVAILABLE,
        },
        'environment_variables': {
            'OPENAI_API_KEY': bool(os.getenv("OPENAI_API_KEY")),
            'ANTHROPIC_API_KEY': bool(os.getenv("ANTHROPIC_API_KEY")),
            'GOOGLE_API_KEY': bool(os.getenv("GOOGLE_API_KEY")),
            'NEWS_API_KEY': bool(os.getenv("NEWS_API_KEY")),
            'FRED_API_KEY': bool(os.getenv("FRED_API_KEY")),
        }
    }
    
    print("\n📋 Diagnostics Results:")
    print(f"   Python Version: {diagnostics['python_version']}")
    print(f"   Required Libraries: All Available ✓")
    
    available_optional = sum(diagnostics['optional_libraries'].values())
    total_optional = len(diagnostics['optional_libraries'])
    print(f"   Optional Libraries: {available_optional}/{total_optional} available")
    
    available_keys = sum(diagnostics['environment_variables'].values())
    total_keys = len(diagnostics['environment_variables'])
    print(f"   API Keys: {available_keys}/{total_keys} configured")
    
    return diagnostics


if __name__ == "__main__":
    """Program entry point"""
    import sys
    
    print("="*80)
    print("🏦 Advanced MAS Financial Factor Scoring System")
    print("   Production-Ready Enhanced Edition")
    print("="*80)
    
    # Setup environment encoding (Windows compatibility) 
    if sys.platform.startswith('win'):
        try:
            import locale
            locale.setlocale(locale.LC_ALL, 'Chinese (Traditional)_Taiwan.950')
        except Exception:
            pass  # Ignore if setup fails 
    
    # Check environment setup 
    if not setup_environment():
        sys.exit(1)
    
    # Run system diagnostics 
    run_system_diagnostics()
    
    # Execute main program 
    exit_code = main()
    sys.exit(exit_code)