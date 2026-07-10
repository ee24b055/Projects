# Joint-Constraint Multi-Factor Risk Attribution & Optimization Engine

A production-grade quantitative investment framework that abstracts and neutralizes multi-variate risk parameters. This engine constructs an institutional-grade **Long/Short Statistical Arbitrage Portfolio** by estimating daily time-varying risk exposure profiles via the **Fama-French 3-Factor Framework** and isolating alpha through non-linear constrained convex optimization.

## Investment Thesis & Mathematical Framework
Most systematic portfolios suffer from hidden risk-factor loading (unintentional exposure to Value, Size, or Market directional betas). This framework decouples idiosyncratic outperformance (Alpha) from structural market risks by estimating the asset matrix daily via Ordinary Least Squares:

$$R_i(t) - R_f(t) = \alpha_i + \beta_{1,i}(Mkt - R_f) + \beta_{2,i}(SMB) + \beta_{3,i}(HML) + \epsilon_i$$

Instead of employing simplistic heuristic sorting or ranking, the execution engine uses a Sequential Least Squares Programming (SLSQP) optimizer to solve a matrix vector problem monthly:

$$\text{Maximize } \sum_{i=1}^{N} w_i \alpha_i$$

Subject to strict orthogonal factor boundaries:
$$\sum_{i=1}^{N} w_i \beta_{1,i} = 0 \quad \text{(Market Neutrality)}$$
$$\sum_{i=1}^{N} w_i \beta_{2,i} = 0 \quad \text{(Size Factor Neutrality)}$$
$$\sum_{i=1}^{N} w_i \beta_{3,i} = 0 \quad \text{(Value Factor Neutrality)}$$
$$\sum_{i=1}^{N} w_i = 0 \quad \text{(Net Dollar Neutrality)}$$
$$\sum_{i=1}^{N} |w_i| = 2.0 \quad \text{(Gross Leveraged Capacity constraint)}$$

## Operational Architecture & Code Pipeline
1. **Data Acquisition Pipeline (`core/data_fetcher.py`):** Automatically connects to and scrapes daily data streams from Ken French’s data repository archive, aligning factor models with pricing matrices.
2. **Dynamic Risk Extraction Engine (`core/risk_model.py`):** Runs rolling multivariate parameter regressions over an un-adjusted chronological index window to avoid look-ahead bias.
3. **Convex Portfolio Optimizer (`core/optimizer.py`):** Utilizes bounding arrays to solve for risk factor exposures simultaneously while strictly limiting asset concentration thresholds.
4. **Friction Simulation Matrix (`main.py`):** Implements localized turnover tracking calculations, imposing a fixed 10 bps transaction cost penalty on all portfolio weight updates.
