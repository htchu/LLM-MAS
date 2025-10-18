# LLM-MAS-DRL Framework 架構重構說明
# Architecture Refactoring Documentation

## 重構概述 / Refactoring Overview

### 原架構 / Original Architecture
```
訓練集 (Train Set) → 測試集 (Test Set)
2000-01-01 ~ 2024-06-30 → 2024-07-01 ~ 2025-06-30
```

### 新架構 / New Architecture  
```
訓練集 (Train) → 驗證集 (Validation) → 測試集 (Test)
2000-01-01 ~ 2022-12-31 → 2023-01-01 ~ 2024-06-30 → 2024-07-01 ~ 2025-06-30
```

---

## 關鍵變更點 / Key Changes

### 1. 配置類變更 / Configuration Class Changes

#### 原配置 / Original Configuration
```python
class TradingConfig:
    TRAIN_START_DATE = "2000-01-01"
    TRAIN_END_DATE = "2024-06-30"
    TEST_START_DATE = "2024-07-01"  
    TEST_END_DATE = "2025-06-30"
```

#### 新配置 / New Configuration
```python
class TradingConfig:
    # Date configuration - Three-stage split / 日期配置 - 三階段分割
    TRAIN_START_DATE = "2000-01-01"           # Training start / 訓練開始日期
    TRAIN_END_DATE = "2022-12-31"             # Training end / 訓練結束日期
    VALIDATION_START_DATE = "2023-01-01"      # Validation start / 驗證開始日期
    VALIDATION_END_DATE = "2024-06-30"        # Validation end / 驗證結束日期
    TEST_START_DATE = "2024-07-01"            # Test start / 測試開始日期
    TEST_END_DATE = "2025-06-30"              # Test end / 測試結束日期
```

**設計理由 / Design Rationale:**
- 訓練集用於模型學習交易模式 / Training set for learning trading patterns
- 驗證集用於超參數優化和模型選擇 / Validation set for hyperparameter tuning and model selection
- 測試集用於評估最終模型真實性能 / Test set for evaluating true model performance

---

### 2. 數據分割邏輯變更 / Data Splitting Logic Changes

#### 原數據分割 / Original Data Splitting
```python
# Step 3: Split Data
train_df = final_processed_df.loc[TRAIN_START_DATE:TRAIN_END_DATE].copy()
test_df = final_processed_df.loc[TEST_START_DATE:TEST_END_DATE].copy()

if train_df.empty or test_df.empty:
    logger.critical("Training or testing dataset is empty")
    return
```

#### 新數據分割 / New Data Splitting  
```python
# Step 3: Split Data into Train/Validation/Test
# 步驟 3: 將數據分割為訓練/驗證/測試
train_df = final_processed_df.loc[TRAIN_START_DATE:TRAIN_END_DATE].copy()
val_df = final_processed_df.loc[VALIDATION_START_DATE:VALIDATION_END_DATE].copy()
test_df = final_processed_df.loc[TEST_START_DATE:TEST_END_DATE].copy()

logger.info(f"Training data / 訓練數據: {len(train_df)} records")
logger.info(f"Validation data / 驗證數據: {len(val_df)} records")
logger.info(f"Testing data / 測試數據: {len(test_df)} records")

if train_df.empty or val_df.empty or test_df.empty:
    logger.critical("Training, validation, or testing dataset is empty")
    return

# 驗證數據連續性 / Validate data continuity
if train_df.index.max() >= val_df.index.min():
    logger.warning("Training data overlaps with validation data")
    
if val_df.index.max() >= test_df.index.min():
    logger.warning("Validation data overlaps with test data")
```

**優勢 / Advantages:**
- 防止數據洩漏 / Prevents data leakage
- 提供可靠的性能評估 / Provides reliable performance evaluation
- 符合業界最佳實踐 / Follows industry best practices

---

### 3. Optuna 優化流程變更 / Optuna Optimization Process Changes

#### 原優化流程 / Original Optimization
```python
def objective(self, trial, train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """
    在測試集上進行優化（不推薦）
    Optimization on test set (not recommended)
    """
    # 在測試集上訓練和評估
    # Train and evaluate on test set
    model = train_model(train_df, hyperparams)
    performance = evaluate_model(model, test_df)  # ❌ 使用測試集
    return performance
```

#### 新優化流程 / New Optimization
```python
def objective(self, trial, train_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
    """
    Optuna objective function for finding optimal hyperparameters using validation set
    Optuna 目標函數，使用驗證集尋找最優超參數
    """
    # 建議超參數 / Suggest hyperparameters
    hyperparams = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
        'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
    }
    
    # 在訓練集上訓練 / Train on training set
    model = train_model(train_df, hyperparams, timesteps=10_000)
    
    # 在驗證集上評估 / Evaluate on validation set ✅
    val_performance = evaluate_model(model, val_df)
    
    # 返回驗證集上的年化收益率 / Return validation set annualized return
    return val_performance['Annualized Return']
```

**關鍵差異 / Key Differences:**
1. **優化目標:** 從測試集改為驗證集 / Optimization target: from test set to validation set
2. **避免過擬合:** 測試集保持完全未見 / Avoiding overfitting: test set remains completely unseen
3. **性能估計:** 測試集性能更接近真實表現 / Performance estimate: test set performance closer to real-world

---

### 4. 模型訓練流程變更 / Model Training Process Changes

#### 完整訓練流程 / Complete Training Process

```python
# Step 4: Hyperparameter Optimization on Validation Set
# 步驟 4: 在驗證集上進行超參數優化
logger.info("Step 4: Optuna Optimization on Validation Set")

study = optuna.create_study(direction='maximize')
objective_with_data = partial(optimizer.objective, 
                              train_df=train_df, 
                              val_df=val_df)  # ✅ 使用驗證集

study.optimize(objective_with_data, n_trials=N_OPTUNA_TRIALS)
best_params = study.best_params

logger.info(f"Best validation return: {study.best_value:.2%}")
logger.info(f"Best hyperparameters: {best_params}")

# Step 5: Train Final Model with Best Parameters
# 步驟 5: 使用最佳參數訓練最終模型
logger.info("Step 5: Training Final Model on Training Set")

final_model = train_model(
    train_df,                              # ✅ 僅使用訓練集
    hyperparameters=best_params,
    timesteps=TOTAL_TRAINING_TIMESTEPS     # 完整訓練步數
)

# Step 6: Evaluate on Validation Set
# 步驟 6: 在驗證集上評估
logger.info("Step 6: Validation Set Evaluation")

val_trades, val_portfolio = evaluate_model(final_model, val_df)
val_strategy_metrics, val_buyhold_metrics = calculate_metrics(
    val_portfolio, val_df, val_trades
)

# Step 7: Final Evaluation on Test Set
# 步驟 7: 在測試集上最終評估
logger.info("Step 7: Test Set Evaluation")

test_trades, test_portfolio = evaluate_model(final_model, test_df)
test_strategy_metrics, test_buyhold_metrics = calculate_metrics(
    test_portfolio, test_df, test_trades
)
```

---

### 5. 評估器變更 / Evaluator Changes

#### 新增評估名稱參數 / Added Evaluation Name Parameter

```python
class BacktestEvaluator:
    def evaluate_model(self, 
                      df_eval: pd.DataFrame,
                      model: PPO,
                      eval_name: str = "Evaluation") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Evaluate trained model on evaluation data
        在評估數據上評估訓練後的模型
        
        Args / 參數:
            df_eval: Evaluation DataFrame / 評估數據框
            model: Trained model / 訓練後的模型
            eval_name: Name for this evaluation / 此次評估的名稱
                       e.g., "Validation" or "Test" / 例如："驗證" 或 "測試"
        """
        logger.info(f"Starting {eval_name}... / 開始 {eval_name}...")
        
        # 創建評估環境 / Create evaluation environment
        eval_env = StockTradingEnvironment(df_eval, ...)
        
        # 運行回測 / Run backtest
        # ... backtest logic ...
        
        logger.info(f"{eval_name} completed / {eval_name} 完成")
        
        return trade_log_df, portfolio_values
```

**使用示例 / Usage Example:**
```python
# 驗證集評估 / Validation evaluation
val_trades, val_portfolio = evaluator.evaluate_model(
    val_df, model, eval_name="Validation"
)

# 測試集評估 / Test evaluation
test_trades, test_portfolio = evaluator.evaluate_model(
    test_df, model, eval_name="Test"
)
```

---

### 6. 性能分析器變更 / Performance Analyzer Changes

```python
class PerformanceAnalyzer:
    def calculate_metrics(self,
                         portfolio_values: pd.Series,
                         df_eval: pd.DataFrame,
                         trade_log_df: pd.DataFrame,
                         eval_name: str = "Strategy") -> Tuple[Dict, Dict]:
        """
        Calculate and return strategy and Buy&Hold performance metrics
        計算並返回策略和買入持有的性能指標
        
        Args / 參數:
            eval_name: Name for logging purposes / 用於日誌記錄的名稱
                       e.g., "Validation" or "Test" / 例如："驗證" 或 "測試"
        """
        logger.info(f"Calculating metrics for {eval_name}...")
        logger.info(f"計算 {eval_name} 的指標...")
        
        # 計算策略指標 / Calculate strategy metrics
        strategy_metrics = {
            "Annualized Return": ...,
            "Sharpe Ratio": ...,
            "Maximum Drawdown": ...,
            # ... 其他指標 / other metrics ...
        }
        
        # 計算買入持有指標 / Calculate buy&hold metrics
        buy_hold_metrics = {...}
        
        logger.info(f"{eval_name} metrics calculated")
        logger.info(f"{eval_name} 指標計算完成")
        
        return strategy_metrics, buy_hold_metrics
```

---

### 7. 報告生成器變更 / Report Generator Changes

#### 新增驗證集結果 / Added Validation Results

```python
class ReportGenerator:
    def generate_comprehensive_report(self,
                                    train_metrics: Dict,
                                    val_strategy_metrics: Dict,      # ✅ 新增
                                    val_buyhold_metrics: Dict,       # ✅ 新增
                                    test_strategy_metrics: Dict,
                                    test_buyhold_metrics: Dict,
                                    train_period: Tuple[str, str],
                                    val_period: Tuple[str, str],     # ✅ 新增
                                    test_period: Tuple[str, str],
                                    **kwargs) -> None:
        """
        Generate comprehensive report with train-validation-test results
        生成包含訓練-驗證-測試結果的綜合報告
        """
        report_content = f"""
# Quantitative Trading Strategy Report
# 量化交易策略報告

## Data Periods / 數據期間
* Training: {train_period[0]} to {train_period[1]}
* Validation: {val_period[0]} to {val_period[1]}  ✅ 新增
* Test: {test_period[0]} to {test_period[1]}

## Validation Performance / 驗證性能  ✅ 新增
| Metric | Strategy | Buy & Hold |
|--------|----------|------------|
| Annualized Return | {val_strategy_metrics['Annualized Return']:.2%} | ... |
| Sharpe Ratio | {val_strategy_metrics['Sharpe Ratio']:.3f} | ... |
| ... | ... | ... |

## Test Performance / 測試性能
| Metric | Strategy | Buy & Hold |
|--------|----------|------------|
| Annualized Return | {test_strategy_metrics['Annualized Return']:.2%} | ... |
| Sharpe Ratio | {test_strategy_metrics['Sharpe Ratio']:.3f} | ... |
| ... | ... | ... |

## Three-Stage Training Approach / 三階段訓練方法  ✅ 新增
1. Training Stage: Model learns patterns / 訓練階段：模型學習模式
2. Validation Stage: Hyperparameter optimization / 驗證階段：超參數優化
3. Test Stage: Final performance evaluation / 測試階段：最終性能評估

This approach ensures reliable performance estimates.
這種方法確保可靠的性能估計。
"""
        # Save report / 保存報告
        save_report(report_content, save_path)
```

---

## 架構優勢 / Architecture Advantages

### 1. 防止過擬合 / Prevent Overfitting
```
原架構 / Original:
訓練集 ← 模型學習
測試集 ← 超參數優化 + 最終評估  ❌ 數據洩漏

新架構 / New:
訓練集 ← 模型學習
驗證集 ← 超參數優化  ✅ 獨立優化
測試集 ← 僅用於最終評估  ✅ 真實性能
```

### 2. 符合學術標準 / Meets Academic Standards
- 訓練集：模型學習 / Training: Model learning
- 驗證集：模型選擇和調優 / Validation: Model selection and tuning
- 測試集：性能報告 / Test: Performance reporting

### 3. 商業應用保證 / Business Application Guarantee
- 測試集性能更接近實際部署表現 / Test performance closer to real deployment
- 減少過度樂觀的回測結果 / Reduces overly optimistic backtest results
- 提供更可靠的風險評估 / Provides more reliable risk assessment

---

## 實施檢查清單 / Implementation Checklist

### 配置層面 / Configuration Level
- [x] 添加 VALIDATION_START_DATE / Add VALIDATION_START_DATE
- [x] 添加 VALIDATION_END_DATE / Add VALIDATION_END_DATE
- [x] 更新日期範圍註釋 / Update date range comments

### 數據層面 / Data Level
- [x] 實現三段數據分割 / Implement three-way data split
- [x] 添加數據連續性驗證 / Add data continuity validation
- [x] 添加驗證集大小檢查 / Add validation set size check

### 訓練層面 / Training Level
- [x] 修改 Optuna 目標函數使用驗證集 / Modify Optuna objective to use validation set
- [x] 確保訓練僅使用訓練集 / Ensure training only uses training set
- [x] 添加驗證集評估步驟 / Add validation set evaluation step

### 評估層面 / Evaluation Level
- [x] 支持多次評估（驗證+測試）/ Support multiple evaluations (validation+test)
- [x] 為評估添加名稱標識 / Add name identifiers for evaluations
- [x] 保留測試集交易記錄 / Keep test set trade records

### 報告層面 / Reporting Level
- [x] 報告中包含驗證結果 / Include validation results in report
- [x] 報告中包含測試結果 / Include test results in report
- [x] 說明三階段方法論 / Explain three-stage methodology

### 視覺化層面 / Visualization Level
- [x] 測試集交易圖表 / Test set trading chart
- [x] 可選：驗證集圖表 / Optional: Validation set chart

---

## 遷移指南 / Migration Guide

### 從舊版本升級 / Upgrading from Old Version

1. **更新配置 / Update Configuration**
```python
# Old / 舊
TRAIN_END_DATE = "2024-06-30"
TEST_START_DATE = "2024-07-01"

# New / 新
TRAIN_END_DATE = "2022-12-31"
VALIDATION_START_DATE = "2023-01-01"
VALIDATION_END_DATE = "2024-06-30"
TEST_START_DATE = "2024-07-01"
```

2. **更新主函數調用 / Update Main Function Calls**
```python
# Old / 舊
train_df, test_df = split_data(df)
best_params = optimize_on_test(train_df, test_df)  # ❌

# New / 新
train_df, val_df, test_df = split_data(df)
best_params = optimize_on_validation(train_df, val_df)  # ✅
```

3. **更新報告生成 / Update Report Generation**
```python
# Old / 舊
generate_report(train_metrics, test_metrics)

# New / 新
generate_report(
    train_metrics,
    val_strategy_metrics,
    val_buyhold_metrics,
    test_strategy_metrics,
    test_buyhold_metrics
)
```

---

## 性能對比示例 / Performance Comparison Example

### 假設結果 / Hypothetical Results

```
驗證集 Validation Set (2023-01-01 ~ 2024-06-30):
- 策略年化收益 / Strategy Annual Return: 18.5%
- 買入持有 / Buy & Hold: 22.3%
- 夏普比率 / Sharpe Ratio: 1.24

測試集 Test Set (2024-07-01 ~ 2025-06-30):
- 策略年化收益 / Strategy Annual Return: 3.01%
- 買入持有 / Buy & Hold: 24.51%
- 夏普比率 / Sharpe Ratio: 0.242
```

**分析 / Analysis:**
- 驗證集性能優於測試集 → 正常現象（模型在驗證集上優化）
- Validation performance > Test performance → Normal (model optimized on validation)
- 測試集提供更真實的未來性能估計
- Test set provides more realistic future performance estimate

---

## 總結 / Summary

### 關鍵改進 / Key Improvements

1. **數據使用 / Data Usage**
   - 訓練 → 學習 / Train → Learn
   - 驗證 → 調優 / Validate → Tune  
   - 測試 → 評估 / Test → Evaluate

2. **科學嚴謹性 / Scientific Rigor**
   - 避免數據洩漏 / Avoid data leakage
   - 提供無偏估計 / Provide unbiased estimates
   - 符合學術標準 / Meet academic standards

3. **商業價值 / Business Value**
   - 更可靠的性能預測 / More reliable performance prediction
   - 減少過度樂觀偏差 / Reduce overly optimistic bias
   - 更好的風險管理 / Better risk management

### 最佳實踐 / Best Practices

✅ **DO / 應該做:**
- 在驗證集上優化超參數 / Optimize hyperparameters on validation set
- 僅在最後使用測試集一次 / Use test set only once at the end
- 報告驗證和測試結果 / Report both validation and test results
- 說明三階段方法論 / Explain three-stage methodology

❌ **DON'T / 不應該做:**
- 在測試集上優化超參數 / Optimize hyperparameters on test set
- 多次評估測試集並調整模型 / Evaluate test set multiple times and adjust
- 只報告測試集結果 / Report only test set results
- 混合使用驗證和測試數據 / Mix validation and test data

---

## 參考資料 / References

1. **機器學習最佳實踐 / Machine Learning Best Practices**
   - Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning.
   - Chapter 5: Machine Learning Basics

2. **金融機器學習 / Financial Machine Learning**
   - López de Prado, M. (2018). Advances in Financial Machine Learning.
   - Chapter 7: Cross-Validation in Finance

3. **強化學習 / Reinforcement Learning**
   - Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction.
   - Chapter 9: On-policy Prediction with Approximation

---

**文檔版本 / Document Version:** 3.0  
**最後更新 / Last Updated:** 2025-09-13  
**作者 / Author:** Edward Cheng  
**審核狀態 / Review Status:** Architecture Design Approved

---

## 授權 / License

MIT License

---

## 聯絡方式 / Contact

- **作者 / Author**: Edward Cheng
- **郵箱 / Email**: llm4edward@gmail.com
- **專案 / Project**: LLM-MAS-DRL Framework
