# LLM-MAS-DRL Framework - 三階段訓練重構版
# Three-Stage Training Refactored Version

## 專案概述 / Project Overview

本專案將原有的 **TRAIN → TEST** 兩階段訓練模式重構為業界標準的 **TRAIN → VALIDATION → TEST** 三階段訓練模式，提供更可靠的模型性能評估和更好的泛化能力。

This project refactors the original **TRAIN → TEST** two-stage training into the industry-standard **TRAIN → VALIDATION → TEST** three-stage approach, providing more reliable model performance evaluation and better generalization.

---

## 核心變更 / Core Changes

### 1. 數據分割 / Data Split

#### 原架構 / Original
```
訓練集: 2000-01-01 ~ 2024-06-30 (24.5年)
測試集: 2024-07-01 ~ 2025-06-30 (1年)
```

#### 新架構 / New
```
訓練集 Training:    2000-01-01 ~ 2022-12-31 (23年)
驗證集 Validation:  2023-01-01 ~ 2024-06-30 (1.5年)
測試集 Test:        2024-07-01 ~ 2025-06-30 (1年)
```

### 2. 訓練流程 / Training Process

```
步驟 1: 在訓練集上訓練模型 / Train model on training set
步驟 2: 在驗證集上優化超參數 / Optimize hyperparameters on validation set
步驟 3: 使用最佳參數訓練最終模型 / Train final model with best parameters
步驟 4: 在驗證集上評估 / Evaluate on validation set
步驟 5: 在測試集上最終評估 / Final evaluation on test set
```

---

## 檔案結構 / File Structure

```
.
├── llmmas_train_val_test.py          # 完整重構程式碼 / Complete refactored code
├── ARCHITECTURE_REFACTORING_GUIDE.md  # 架構變更詳細說明 / Detailed architecture guide
└── README.md                          # 本文件 / This file
```

---

## 快速開始 / Quick Start

### 1. 環境要求 / Requirements

```bash
Python >= 3.8
pandas >= 1.3.0
numpy >= 1.20.0
yfinance >= 0.1.70
TA-Lib >= 0.4.24
gym >= 0.21.0
stable-baselines3 >= 1.5.0
optuna >= 2.10.0
plotly >= 5.3.0
scikit-learn >= 1.0.0
```

### 2. 安裝依賴 / Install Dependencies

```bash
pip install pandas numpy yfinance TA-Lib gym stable-baselines3 optuna plotly scikit-learn
```

### 3. 準備數據 / Prepare Data

確保以下檔案存在 / Ensure the following files exist:

```
quantitative_trading_data/
├── MSFT_score_data_ok.csv  # 因子數據 / Factor data
└── (MSFT_raw_data.csv)     # 將自動下載 / Will be downloaded automatically
```

### 4. 運行程式 / Run Program

```bash
python llmmas_train_val_test.py
```

---

## 配置說明 / Configuration

### 關鍵配置項 / Key Configuration Items

```python
class TradingConfig:
    # 股票代碼 / Stock Ticker
    STOCK_TICKER = "MSFT"
    
    # 數據期間 / Data Periods
    TRAIN_START_DATE = "2000-01-01"
    TRAIN_END_DATE = "2022-12-31"
    VALIDATION_START_DATE = "2023-01-01"
    VALIDATION_END_DATE = "2024-06-30"
    TEST_START_DATE = "2024-07-01"
    TEST_END_DATE = "2025-06-30"
    
    # 初始資金 / Initial Capital
    INITIAL_BALANCE = 1_000_000  # USD
    
    # 訓練步數 / Training Steps
    TOTAL_TRAINING_TIMESTEPS = 100_000
    TRIAL_TRAINING_TIMESTEPS = 10_000
    
    # Optuna 優化 / Optuna Optimization
    N_OPTUNA_TRIALS = 20
```

### 修改配置 / Modify Configuration

根據需求修改 `TradingConfig` 類中的參數：

Modify parameters in the `TradingConfig` class according to your needs:

- **股票代碼 / Stock Ticker**: 更改 `STOCK_TICKER`
- **數據期間 / Date Ranges**: 調整各階段日期
- **風險參數 / Risk Parameters**: 修改止損/止盈比例
- **訓練參數 / Training Parameters**: 調整訓練步數和優化試驗次數

---

## 輸出結果 / Output Results

程式執行完成後會生成以下檔案 / After execution, the following files will be generated:

```
quantitative_trading_data/
├── MSFT_raw_data.csv                 # 原始價格數據 / Raw price data
├── MSFT_final_processed_data.csv    # 處理後的數據 / Processed data
├── MSFT_rl_model_best.zip           # 訓練好的模型 / Trained model
├── MSFT_trades_best.csv             # 測試集交易記錄 / Test set trades
├── MSFT_trading_chart_best.html     # 互動式交易圖表 / Interactive chart
├── MSFT_report_best.md              # 綜合報告 / Comprehensive report
├── trading_strategy.log             # 詳細日誌 / Detailed logs
└── MSFT_optuna_study.db             # Optuna 優化記錄 / Optuna study
```

### 關鍵輸出 / Key Outputs

1. **綜合報告 / Comprehensive Report** (`MSFT_report_best.md`)
   - 包含訓練、驗證、測試三階段結果 / Contains train, validation, test results
   - 策略 vs 買入持有對比 / Strategy vs Buy & Hold comparison
   - 性能指標分析 / Performance metrics analysis

2. **互動式圖表 / Interactive Chart** (`MSFT_trading_chart_best.html`)
   - 價格走勢與交易點 / Price trends and trading points
   - 策略淨值曲線 / Strategy net worth curve
   - 買入持有基線 / Buy & Hold baseline

3. **交易記錄 / Trade Log** (`MSFT_trades_best.csv`)
   - 詳細的買賣記錄 / Detailed buy/sell records
   - 每筆交易的價格、數量、成本 / Price, quantity, cost per trade

---

## 性能指標 / Performance Metrics

報告中包含的關鍵指標 / Key metrics in the report:

### 驗證集 / Validation Set
- 年化收益率 / Annualized Return
- 夏普比率 / Sharpe Ratio
- 最大回撤 / Maximum Drawdown
- 條件風險價值 (CVaR) / Conditional VaR
- 年化波動率 / Annualized Volatility
- 勝率 / Win Rate
- 總交易數 / Total Trades

### 測試集 / Test Set
- 所有驗證集指標 / All validation set metrics
- 最終投資組合淨值 / Final Portfolio Net Worth
- 與買入持有策略對比 / Comparison with Buy & Hold

---

## 架構優勢 / Architecture Advantages

### 1. 防止過擬合 / Prevent Overfitting
✅ 測試集在訓練和優化過程中完全未見  
✅ Test set completely unseen during training and optimization

### 2. 可靠的性能估計 / Reliable Performance Estimation
✅ 驗證集用於模型選擇，測試集用於真實性能報告  
✅ Validation for model selection, test for true performance

### 3. 符合學術標準 / Academic Standards
✅ 遵循機器學習和金融工程最佳實踐  
✅ Follows ML and financial engineering best practices

### 4. 商業應用保證 / Business Application
✅ 測試集性能更接近實際部署表現  
✅ Test performance closer to real deployment

---

## 使用場景 / Use Cases

### 研究場景 / Research Scenario
```python
# 修改股票代碼進行不同股票研究
# Modify stock ticker for different stocks
STOCK_TICKER = "AAPL"  # 或 "AMZN", "NVDA", etc.

# 調整數據期間進行不同時期研究
# Adjust periods for different time ranges
TRAIN_END_DATE = "2020-12-31"
VALIDATION_START_DATE = "2021-01-01"
```

### 生產部署 / Production Deployment
```python
# 使用最新數據進行模型訓練
# Train with latest data
TEST_END_DATE = datetime.now().strftime("%Y-%m-%d")

# 增加訓練步數提高性能
# Increase training steps for better performance
TOTAL_TRAINING_TIMESTEPS = 200_000
N_OPTUNA_TRIALS = 50
```

---

## 故障排除 / Troubleshooting

### 常見問題 / Common Issues

#### 1. 數據下載失敗 / Data Download Failure
```
解決方案 / Solution:
- 檢查網路連接 / Check network connection
- 確認股票代碼正確 / Verify stock ticker
- 嘗試手動下載數據 / Try manual data download
```

#### 2. 因子數據缺失 / Missing Factor Data
```
錯誤 / Error: "Factor data file not found"

解決方案 / Solution:
- 確保 MSFT_score_data_ok.csv 存在 / Ensure factor file exists
- 檢查檔案路徑 / Check file path
- 驗證因子欄位完整性 / Verify factor columns
```

#### 3. 記憶體不足 / Out of Memory
```
解決方案 / Solution:
- 減少窗口大小 / Reduce WINDOW_SIZE
- 降低訓練步數 / Decrease training timesteps
- 使用較短的數據期間 / Use shorter date ranges
```

#### 4. 訓練時間過長 / Training Too Slow
```
解決方案 / Solution:
- 減少 Optuna 試驗次數 / Reduce N_OPTUNA_TRIALS
- 降低試驗訓練步數 / Decrease TRIAL_TRAINING_TIMESTEPS
- 使用 GPU 加速 / Use GPU acceleration (if available)
```

---

## 進階使用 / Advanced Usage

### 1. 自定義因子 / Custom Factors

```python
# 添加新的因子欄位
# Add new factor columns
EXPECTED_FACTOR_COLUMNS = [
    'fundamental_score',
    'sentiment_score',
    'industry_trend_score',
    'market_risk_factor',
    'black_swan_risk',
    'custom_factor_1',  # 新增 / New
    'custom_factor_2',  # 新增 / New
]
```

### 2. 調整風險參數 / Adjust Risk Parameters

```python
# 修改止損/止盈
# Modify stop-loss/take-profit
STOP_LOSS_PCT = 0.03      # 3% 止損
TAKE_PROFIT_PCT = 0.3     # 30% 止盈

# 調整倉位風險
# Adjust position risk
MAX_POSITION_RISK_PCT = 0.05  # 5% 最大風險
```

### 3. 超參數搜索空間 / Hyperparameter Search Space

```python
# 在 OptunaOptimizer.objective 中修改
# Modify in OptunaOptimizer.objective
hyperparams = {
    'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True),
    'n_steps': trial.suggest_categorical('n_steps', [256, 512, 1024, 2048, 4096]),
    'gamma': trial.suggest_float('gamma', 0.85, 0.9999),
    # 添加更多超參數 / Add more hyperparameters
}
```

---

## 性能優化建議 / Performance Optimization Tips

### 1. 數據層面 / Data Level
- 使用更多歷史數據增強訓練 / Use more historical data
- 提高數據質量（去噪、異常值處理）/ Improve data quality
- 添加更多有效因子 / Add more effective factors

### 2. 模型層面 / Model Level
- 增加訓練步數 / Increase training steps
- 擴大超參數搜索範圍 / Expand hyperparameter search space
- 嘗試不同的 RL 演算法 / Try different RL algorithms

### 3. 風險管理 / Risk Management
- 動態調整止損/止盈閾值 / Dynamically adjust stop-loss/take-profit
- 實施更複雜的倉位管理 / Implement more sophisticated position sizing
- 添加多維度風險控制 / Add multi-dimensional risk controls

---

## 版本歷史 / Version History

### v3.0 (2025-10-13) - 三階段重構版
- ✅ 實現 TRAIN-VALIDATION-TEST 三階段訓練
- ✅ Optuna 優化使用驗證集
- ✅ 報告包含驗證和測試結果
- ✅ 完整的雙語註解（中英文）
- ✅ 詳細的架構文檔

### v2.0 (2025-09-12) - 原版本
- ✓ TRAIN-TEST 兩階段訓練
- ✓ 基本 Optuna 優化
- ✓ 性能指標和報告

---

## 貢獻指南 / Contributing

歡迎貢獻改進！請遵循以下步驟：

Contributions are welcome! Please follow these steps:

1. Fork 本專案 / Fork the project
2. 創建特性分支 / Create feature branch
3. 提交變更 / Commit changes
4. 推送到分支 / Push to branch
5. 創建 Pull Request / Create Pull Request

---

## 授權 / License

MIT License

---

## 聯絡方式 / Contact

- **作者 / Author**: Edward Cheng
- **郵箱 / Email**: llm4edward@gmail.com
- **專案 / Project**: LLM-MAS-DRL Framework

---

## 致謝 / Acknowledgments

感謝以下開源專案：

Thanks to the following open-source projects:

- **Stable Baselines 3**: Reinforcement learning library
- **Optuna**: Hyperparameter optimization framework
- **TA-Lib**: Technical analysis library
- **Plotly**: Interactive visualization library

---

## 附錄 / Appendix

### A. 完整配置示例 / Complete Configuration Example

```python
class TradingConfig:
    # 日誌配置 / Logging
    LOGGING_LEVEL = logging.INFO
    DATA_DIR = "quantitative_trading_data"
    LOG_FILE_NAME = "trading_strategy.log"
    
    # 股票配置 / Stock
    STOCK_TICKER = "MSFT"
    
    # 日期配置（三階段）/ Dates (Three-stage)
    TRAIN_START_DATE = "2000-01-01"
    TRAIN_END_DATE = "2022-12-31"
    VALIDATION_START_DATE = "2023-01-01"
    VALIDATION_END_DATE = "2024-06-30"
    TEST_START_DATE = "2024-07-01"
    TEST_END_DATE = "2025-06-30"
    
    # 財務配置 / Financial
    INITIAL_BALANCE = 1_000_000
    WINDOW_SIZE = 40
    
    # RL 配置 / RL Configuration
    TOTAL_TRAINING_TIMESTEPS = 100_000
    TRIAL_TRAINING_TIMESTEPS = 10_000
    RL_ALGORITHM = PPO
    
    # 交易配置 / Trading
    COMMISSION_RATE = 0.002
    SLIPPAGE = 0.001
    MIN_TRADE_SHARES = 100
    
    # 風險管理 / Risk Management
    PROFIT_REWARD_FACTOR = 0.3
    VOLATILITY_PENALTY_FACTOR = 0.035
    STOP_LOSS_PCT = 0.05
    TAKE_PROFIT_PCT = 0.5
    MAX_POSITION_RISK_PCT = 0.1
    ATR_PERIOD_FOR_SIZING = 14
    
    # Optuna 配置 / Optuna
    N_OPTUNA_TRIALS = 20
    
    # 因子配置 / Factors
    EXPECTED_FACTOR_COLUMNS = [
        'fundamental_score',
        'sentiment_score',
        'industry_trend_score',
        'market_risk_factor',
        'black_swan_risk'
    ]
```

### B. 執行流程圖 / Execution Flow Chart

```
開始 / Start
    ↓
載入數據 / Load Data
    ↓
合併處理 / Merge & Process
    ↓
三段分割 / Three-way Split
    ├─ 訓練集 / Training Set
    ├─ 驗證集 / Validation Set
    └─ 測試集 / Test Set
    ↓
Optuna 優化（在驗證集上）/ Optuna Optimization (on Validation)
    ↓
訓練最終模型 / Train Final Model
    ↓
驗證集評估 / Validation Evaluation
    ↓
測試集評估 / Test Evaluation
    ↓
生成報告和圖表 / Generate Reports & Charts
    ↓
完成 / Complete
```

---

**最後更新 / Last Updated**: 2025-09-13  
**文檔版本 / Document Version**: 3.0  
**狀態 / Status**: ✅ Production Ready