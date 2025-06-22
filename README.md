# LLM-MAS-DRL Framework Installation Guide for Google Colab

> **File Name**: `LLM-MAS-DRL_Installation_Guide.md`  
> **Created**: 2025-06-21  
> **Framework Version**: Enhanced Edition (2025-05-22)  
> **Author**: Edward Cheng

## Overview

The **LLM-MAS-DRL Framework** is an AI-powered quantitative trading strategy development system that combines reinforcement learning, multi-factor analysis, and advanced risk management. This guide provides step-by-step instructions for setting up the framework in Google Colab environment.

## System Requirements

- **Environment**: Google Colab (recommended) or Python 3.7+
- **Memory**: At least 4GB RAM (Colab provides 12GB)
- **Storage**: Minimum 2GB free space
- **Internet**: Required for data downloading and package installation

## Features

- **Market Data Integration**: Automatic OHLCV data download via yfinance
- **Technical Analysis**: 25+ technical indicators using TA-Lib
- **Reinforcement Learning**: PPO algorithm with Stable Baselines3
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Advanced Visualization**: Interactive charts with Plotly
- **Risk Management**: Stop-loss, take-profit, and position sizing
- **Comprehensive Reporting**: Automated performance analysis

## Installation Steps

### Step 1: Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook or open an existing one
3. Make sure you're using Python 3 runtime

### Step 2: Install Required Packages

Copy and paste the following installation commands into separate Colab cells:

#### Cell 1: Install TA-Lib (Technical Analysis Library)
```bash
# Install TA-Lib dependencies
!apt-get update
!apt-get install -y build-essential
!wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
!tar -xzf ta-lib-0.4.0-src.tar.gz
!cd ta-lib && ./configure --prefix=/usr && make && make install
!pip install TA-Lib
```

#### Cell 2: Install Core ML and Trading Libraries
```bash
# Install machine learning and trading packages
!pip install stable-baselines3[extra]
!pip install optuna
!pip install yfinance
!pip install plotly
!pip install gym==0.21.0  # Specific version for compatibility
```

#### Cell 3: Install Additional Dependencies
```bash
# Install additional required packages
!pip install pandas numpy scikit-learn
!pip install kaleido  # For plotly static image export
```

### Step 3: Verify Installation

Run this verification cell to ensure all packages are installed correctly:

```python
# Verification script
try:
    import yfinance as yf
    import talib
    import pandas as pd
    import numpy as np
    import gym
    from stable_baselines3 import PPO
    import optuna
    import plotly.graph_objects as go
    from sklearn.preprocessing import StandardScaler
    print("All packages installed successfully!")
    print(f"TA-Lib version: {talib.__version__}")
    print(f"Stable Baselines3 version: {stable_baselines3.__version__}")
    print(f"Optuna version: {optuna.__version__}")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please re-run the installation cells above.")
```

### Step 4: Upload the Framework File

1. **Method A: Direct Upload**
   - Click on the folder icon in the left sidebar
   - Click "Upload" and select your `llmmas.py` file

2. **Method B: Google Drive Integration**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # Then copy your file from Drive to Colab
   !cp '/content/drive/MyDrive/path/to/llmmas.py' '/content/'
   ```

3. **Method C: Download from URL**
   ```python
   !wget https://your-url.com/llmmas.py
   ```

### Step 5: Prepare Factor Data

The framework requires pre-calculated factor data. Create a sample factor data file:

```python
# Create sample factor data
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample dates
start_date = '2000-01-01'
end_date = '2024-12-31'
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Remove weekends (assuming stock market days only)
dates = dates[dates.weekday < 5]

# Create sample factor data
np.random.seed(101)
factor_data = {
    'Date': dates,
    'fundamental_score': np.random.normal(0, 1, len(dates)),
    'sentiment_score': np.random.normal(0, 1, len(dates)),
    'industry_trend_score': np.random.normal(0, 1, len(dates)),
    'market_risk_factor': np.random.normal(0, 1, len(dates)),
    'black_swan_risk': np.random.exponential(0.5, len(dates))
}

factor_df = pd.DataFrame(factor_data)
factor_df.set_index('Date', inplace=True)

# Create data directory and save
import os
os.makedirs('quantitative_trading_data', exist_ok=True)
factor_df.to_csv('quantitative_trading_data/AAPL_score_data_ok.csv')
print("Sample factor data created successfully!")
print(f"Saved to: quantitative_trading_data/AAPL_score_data_ok.csv")
print(f"Data shape: {factor_df.shape}")
```

## Running the Framework

### Option 1: Run the Complete Pipeline

```python
# Import and run the main function
exec(open('llmmas.py').read())
```

### Option 2: Step-by-Step Execution

```python
# Import the framework
import sys
sys.path.append('/content')

# Import all classes and functions
from llmmas import *

# Initialize logger
logger = setup_logger()

# Create data manager
data_manager = DataManager(logger)

# Load OHLCV data
ohlcv_df = data_manager.load_raw_stock_data(
    TradingConfig.STOCK_TICKER,
    TradingConfig.TRAIN_START_DATE,
    TradingConfig.TEST_END_DATE,
    TradingConfig.get_raw_data_path()
)

print(f"Data loaded: {len(ohlcv_df)} records")
```

## Configuration Options

You can modify the trading configuration by editing the `TradingConfig` class:

```python
# Example: Change stock ticker and parameters
TradingConfig.STOCK_TICKER = "TSLA"  # Tesla instead of Apple
TradingConfig.INITIAL_BALANCE = 100_000  # $100k instead of $1M
TradingConfig.N_OPTUNA_TRIALS = 10  # Fewer trials for faster testing
```

## Expected Output Files

After successful execution, the framework will generate:

```
quantitative_trading_data/
├── AAPL_raw_data.csv                   # Raw OHLCV data
├── AAPL_score_data_ok.csv              # Factor data (provided)
├── AAPL_final_processed_data.csv       # Merged data with indicators
├── AAPL_rl_model_best.zip              # Trained RL model
├── AAPL_trades_best.csv                # Trading history
├── AAPL_trading_chart_best.html        # Interactive chart
├── AAPL_report_best.md                 # Performance report
├── trading_strategy.log                # Detailed logs
├── optuna_study.db                     # Optimization database
└── sb3_logs/                           # Training logs
```

## Troubleshooting

### Common Issues and Solutions

#### 1. TA-Lib Installation Failed
```bash
# If TA-Lib installation fails, try alternative method:
!pip install --upgrade pip setuptools wheel
!pip install --no-binary :all: TA-Lib
```

#### 2. Memory Issues
```python
# Reduce dataset size or model complexity
TradingConfig.TRAIN_START_DATE = "2020-01-01"  # Shorter period
TradingConfig.N_OPTUNA_TRIALS = 5  # Fewer trials
TradingConfig.TOTAL_TRAINING_TIMESTEPS = 50_000  # Fewer steps
```

#### 3. Gym Version Compatibility
```bash
# If you encounter gym-related errors:
!pip uninstall gym -y
!pip install gym==0.21.0
```

#### 4. CUDA/GPU Issues
```python
# Check if GPU is available and working
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

# If needed, restart runtime and enable GPU:
# Runtime -> Change runtime type -> Hardware accelerator: GPU
```

#### 5. File Permission Issues
```bash
# Fix permission issues
!chmod +x /content/llmmas.py
!chmod -R 755 quantitative_trading_data/
```

## Performance Tips

### 1. Enable GPU Acceleration
- Go to Runtime → Change runtime type → Hardware accelerator: GPU
- This will significantly speed up RL training

### 2. Optimize Memory Usage
```python
# Clear memory periodically
import gc
gc.collect()

# Monitor memory usage
!free -h
```

### 3. Use Shorter Training Periods for Testing
```python
# For quick testing, use recent data only
TradingConfig.TRAIN_START_DATE = "2022-01-01"
TradingConfig.TEST_START_DATE = "2023-01-01"
TradingConfig.TEST_END_DATE = "2023-12-31"
```

## Advanced Usage

### Custom Factor Data Integration

If you have your own factor data, ensure it follows this format:

```python
# Required factor columns
required_columns = [
    'fundamental_score',      # Company fundamentals
    'sentiment_score',        # News/social sentiment
    'industry_trend_score',   # Industry trends
    'market_risk_factor',     # Market risk indicators
    'black_swan_risk'         # Extreme event indicators
]

# Data should be indexed by Date with daily frequency
# Values should be normalized (mean=0, std=1) for best results
```

### Multiple Stock Analysis

```python
# Run analysis for multiple stocks
stocks = ['AAPL', 'AMZN', 'MSFT', 'TSLA', 'GOOGL']

for stock in stocks:
    TradingConfig.STOCK_TICKER = stock
    # Create corresponding factor data file
    # Run main() function
    print(f"Completed analysis for {stock}")
```

## Support and Documentation

- **Framework Author**: Edward Cheng
- **Version**: Enhanced Edition (2025-05-22)
- **Issues**: Check the detailed logs in `trading_strategy.log`
- **Performance**: Monitor results in the generated HTML chart

## Next Steps

1. **Analyze Results**: Open the generated HTML chart to visualize performance
2. **Review Logs**: Check `trading_strategy.log` for detailed execution information
3. **Optimize Parameters**: Experiment with different hyperparameters
4. **Improve Factors**: Enhance factor quality for better performance
5. **Scale Up**: Apply to multiple stocks or longer time periods

## License and Disclaimer

This framework is for educational and research purposes. 
Always validate strategies thoroughly before applying to real trading. 
Past performance does not guarantee future results.

