
# Quantitative Trading Strategy Backtest Report

**Stock Ticker:** AAPL
**Model:** Reinforcement Learning (PPO) + Multi-Factor Analysis (with pre-calculated factors)
**Model File:** `AAPL_rl_model_best.zip`

---

## Data Periods
* **Training Data:** 2000-10-16 to 2021-12-31
* **Validation Data:** 2022-01-03 to 2024-06-28
* **Test Data:** 2024-07-01 to 2025-06-27

---

## Three-Stage Training Approach

This framework implements the industry-standard **Train-Validation-Test** split:

1. **Training Stage**: Model learns trading patterns from historical data
2. **Validation Stage**: Hyperparameter optimization using Optuna
3. **Test Stage**: Final unbiased performance evaluation

**Key Advantage**: The test set remains completely unseen during training and optimization, providing reliable real-world performance estimates.

---

## Validation Set Performance

**Period:** 2022-01-03 to 2024-06-28

### Performance Metrics Comparison

| Metric | Strategy (RL + Multi-Factor) | Buy & Hold |
|--------|------------------------------|------------|
| Annualized Return | 9.87% | 11.19% |
| Sharpe Ratio | 0.845 | 0.517 |
| Maximum Drawdown | -16.66% | -30.14% |
| Conditional VaR (95%) | -1.62% | -3.95% |
| Annualized Volatility | 11.97% | 28.16% |
| Win Rate | N/A | N/A |
| Total Trades | 2 | 1 |
| Final Portfolio Net Worth | 1,246,557.62 USD | 1,281,767.20 USD |
| Initial Capital | 1,000,000.00 USD | 1,000,000.00 USD |

**Validation Set Purpose**: Hyperparameter optimization and model selection

**Trade Records**: See `AAPL_trades_validation.csv` (strategy only)
**Interactive Chart**: See `AAPL_trading_chart_validation.html` (open with browser, includes Buy&Hold comparison)

---

## Test Set Performance

**Period:** 2024-07-01 to 2025-06-27

### Performance Metrics Comparison

| Metric | Strategy (RL + Multi-Factor) | Buy & Hold |
|--------|------------------------------|------------|
| Annualized Return | -4.67% | -12.80% |
| Sharpe Ratio | -0.706 | -0.250 |
| Maximum Drawdown | -9.76% | -33.43% |
| Conditional VaR (95%) | -1.25% | -4.79% |
| Annualized Volatility | 6.47% | 33.14% |
| Win Rate | 0.00% | N/A |
| Total Trades | 4 | 1 |
| Final Portfolio Net Worth | 960,231.99 USD | 890,167.79 USD |
| Initial Capital | 1,000,000.00 USD | 1,000,000.00 USD |

**Test Set Purpose**: Final unbiased performance evaluation

**Trade Records**: See `AAPL_trades_test.csv` (strategy only)
**Interactive Chart**: See `AAPL_trading_chart_test.html` (open with browser, includes Buy&Hold comparison)

---

## Strategy Description

This strategy uses a reinforcement learning agent (PPO) to make trading decisions (buy/sell/hold) based on:
- Historical stock prices (OHLCV)
- Technical indicators (SMA, EMA, RSI, MACD, etc.)
- **Pre-calculated** fundamental, news sentiment, industry trend, and market risk factors

Trading and evaluation are primarily based on **closing prices ('Close')**. The report provides performance comparison with **Buy & Hold** baseline strategy.

**Important Note**: Factor data quality and predictive capability directly impact strategy performance.

---

## Risk Management Mechanisms

* **Stop Loss:** 3% (force sell when price drops from entry
* **Take Profit:** 30% (force sell when price rises from entry
* **Dynamic Position Management:** Based on ATR (14 days) and risk ratio (10.00%) to calculate maximum position size
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

* **Detailed Execution Log:** See `trading_strategy.log`
* **Architecture Documentation:** See `ARCHITECTURE_REFACTORING_GUIDE.md`
* **User Guide:** See `README_THREE_STAGE.md`

---

*Report generated on 2025-10-18 03:15:55*
*Framework Version: 3.0 (Three-Stage Training)*
