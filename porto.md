# Joseph Bidias — Quantitative Finance Portfolio

**MScFE Graduate · WorldQuant University**
Combining rigorous mathematical finance with modern machine learning and data engineering to build production-quality quantitative systems. Every project below was independently designed, coded, and delivered as part of the Master of Science in Financial Engineering program.

---

## Table of Contents
1. [Capstone — Integrated Risk Monitoring System](#1-integrated-risk-monitoring-system)
2. [Capstone — Regime-Aware Pairs Trader](#2-regime-aware-pairs-trader)
3. [Stochastic Modeling — Heston & Bates Option Pricing](#3-heston--bates-option-pricing)
4. [Stochastic Modeling — Regime Switching: S&P 500 & Bitcoin](#4-regime-switching-sp500--bitcoin)
5. [Stochastic Modeling — Risk-Aware Multi-Armed Bandit](#5-risk-aware-multi-armed-bandit)
6. [Derivative Pricing — Binomial Tree & Put-Call Parity](#6-binomial-tree--put-call-parity)
7. [Derivative Pricing — Black-Scholes & Monte Carlo Simulation](#7-black-scholes--monte-carlo-simulation)
8. [Derivative Pricing — Heston & Merton Jump Diffusion](#8-heston--merton-jump-diffusion)
9. [Risk Management — Oil Forecasting with Probabilistic Graphical Models](#9-oil-price-forecasting--probabilistic-graphical-models)
10. [Risk Management — HMM Regime Detection & Bayesian Network](#10-hmm-regime-detection--bayesian-network)
11. [Risk Management — Full Model Evaluation & Improvement](#11-full-model-evaluation--improvement)
12. [Portfolio Management — Multi-Asset Optimization Engine](#12-multi-asset-portfolio-optimization-engine)
13. [Portfolio Management — Mean-Variance: Tech & Healthcare Portfolio](#13-mean-variance-tech--healthcare-portfolio)
14. [Portfolio Management — ML-Enhanced Portfolio Optimization](#14-ml-enhanced-portfolio-optimization)
15. [Machine Learning in Finance — Regression Trees](#15-regression-trees-for-financial-prediction)
16. [Machine Learning in Finance — Linear Discriminant Analysis](#16-linear-discriminant-analysis-for-market-classification)
17. [Machine Learning in Finance — Hyperparameter Optimization](#17-hyperparameter-optimization--bias-variance-tradeoff)
18. [Deep Learning in Finance — Statistical Arbitrage with CNN/LSTM](#18-statistical-arbitrage-with-cnnlstm)
19. [Deep Learning in Finance — Multi-Asset Portfolio with LSTM](#19-multi-asset-portfolio-allocation-with-lstm)
20. [Deep Learning in Finance — Data Leakage in Walk-Forward Backtesting](#20-data-leakage-in-walk-forward-backtesting)
21. [Financial Econometrics — Outlier Sensitivity in Regression](#21-outlier-sensitivity-in-regression)
22. [Financial Econometrics — Modeling Randomness & White Noise](#22-modeling-randomness--white-noise)
23. [Financial Econometrics — Cointegration & VECM](#23-cointegration--vecm)
24. [Financial Data — Lending EDA: Stock Returns, Housing & Rates](#24-lending-eda-stock-returns-housing--rates)
25. [Financial Data — Financial Risk Analysis in Lending Scenarios](#25-financial-risk-analysis-in-lending-scenarios)
26. [Financial Data — Mortgage Amortization & Securities Lending](#26-mortgage-amortization--securities-lending)

---

## 1. Integrated Risk Monitoring System

**Domain:** Systemic Risk · Multi-Asset Monitoring
**Tools:** Python · GARCH · DCC · PCA · Quantile Regression · HAR-RV · FRED · Bloomberg

### Overview
I built a full-stack integrated risk monitoring system calibrated to real market data spanning 2005-2024. The system fuses ten asset classes — equities (SPY, EFA, EEM, XLF, EWJ), fixed income (TLT, IEF, AGG, HYG), gold (GLD), VIX, TED spread, EUR/USD xccy basis, and HY OAS — into a unified dashboard that flags systemic stress in real time. GARCH parameters were estimated from actual historical data; correlation matrices were calibrated to empirical calm/crisis regimes. Spread anchors: TED 463 bps (Oct 2008), HY OAS 1,994 bps (Dec 2008), VIX 82.69 (Mar 2020).

| Step | Component |
|---|---|
| 1-A | PCA Systemic Risk Indicator |
| 1-B | Quantile Regression + Time-Varying Correlation |
| 1-C | DCC-GARCH + Funding Stress Regime Switching |
| 2 | Cascade Effect Analysis |
| 3-A | Bootstrap Correlation Uncertainty |
| 3-B | HAR-RV Volatility Forecasting |
| 4-5 | Historical Stress Backtests + Performance Table |
| 7 | Final Risk Dashboard |

### Key Code — GARCH Parameters & Data Loader

```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from arch import arch_model
import statsmodels.api as sm

np.random.seed(42)

# Real GARCH(1,1) parameters calibrated from historical data
GARCH_PARAMS = {
    'SPY': {'omega': 1.02e-6, 'alpha': 0.0731, 'beta': 0.9189},
    'TLT': {'omega': 1.82e-6, 'alpha': 0.0512, 'beta': 0.9327},
    'GLD': {'omega': 2.21e-6, 'alpha': 0.0601, 'beta': 0.9264},
    'HYG': {'omega': 8.95e-7, 'alpha': 0.0847, 'beta': 0.9018},
    'VIX': {'omega': 9.12e-5, 'alpha': 0.1923, 'beta': 0.7614},
}

SPREAD_PEAKS = {
    'ted_spread_bps': 463,   # Oct 2008 (FRED)
    'hy_oas_bps':    1994,   # Dec 2008 (FRED)
    'vix_close':     82.69,  # Mar 2020 (CBOE)
    'xccy_eur_usd':  -120,   # 2008 GFC (Bloomberg)
}

def fit_dcc_garch(returns_series, asset_name):
    """Fit GARCH(1,1) and extract conditional volatility."""
    am  = arch_model(returns_series * 100, vol='Garch', p=1, q=1, dist='Normal')
    res = am.fit(disp='off')
    cond_vol = res.conditional_volatility / 100
    print(f"{asset_name}: omega={res.params['omega']:.2e}, "
          f"alpha={res.params['alpha[1]']:.4f}, "
          f"beta={res.params['beta[1]']:.4f}")
    return cond_vol, res
```

### Key Code — PCA Systemic Risk Indicator

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def build_pca_risk_indicator(returns_df, n_components=3):
    """
    Constructs a systemic risk composite score via PCA.
    The first principal component captures the dominant co-movement
    across all assets and serves as the stress signal.
    """
    scaler     = StandardScaler()
    scaled     = scaler.fit_transform(returns_df.dropna())
    pca        = PCA(n_components=n_components)
    components = pca.fit_transform(scaled)

    # Sign-adjust: high score = high stress
    risk_score = pd.Series(
        -components[:, 0],
        index=returns_df.dropna().index,
        name='Systemic_Risk_PCA'
    )
    print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]:.1%}")
    return risk_score, pca


def har_rv_forecast(rv_series, horizon=5):
    """HAR-RV: daily, weekly, and monthly realized variance components."""
    df = pd.DataFrame({'RV_d': rv_series})
    df['RV_w']   = rv_series.rolling(5).mean()
    df['RV_m']   = rv_series.rolling(22).mean()
    df['RV_fwd'] = rv_series.shift(-horizon)
    df           = df.dropna()
    X            = sm.add_constant(df[['RV_d', 'RV_w', 'RV_m']])
    return sm.OLS(df['RV_fwd'], X).fit()
```

---

## 2. Regime-Aware Pairs Trader

**Domain:** Statistical Arbitrage · Pairs Trading · Regime Filtering
**Tools:** Python · Engle-Granger · Markov Switching · BDS Test · Kupiec VaR · yfinance

### Overview
I designed and back-tested a regime-filtered pairs trading strategy on the energy sector ETF pair **XLE / XOP** (January 2017 to December 2021). The central hypothesis: strategic inactivity during high-volatility regimes improves risk-adjusted returns. Two strategies were compared: "Always-On" (constant cointegration spread entry/exit) vs "Regime-Filtered" (suspended during HMM-identified high-volatility states).

### Key Code — Engle-Granger Cointegration & Spread

```python
import yfinance as yf
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

tickers = ['XLE', 'XOP']
prices  = yf.download(tickers, start='2017-01-01', end='2021-12-31')['Close']

# Engle-Granger cointegration test
score, pvalue, _ = coint(prices['XLE'], prices['XOP'])
print(f"Cointegration p-value: {pvalue:.4f}")

# OLS hedge ratio (log-price regression)
log_xle = np.log(prices['XLE'])
log_xop = np.log(prices['XOP'])
model   = sm.OLS(log_xle, sm.add_constant(log_xop)).fit()
beta    = model.params['XOP']
spread  = log_xle - beta * log_xop   # stationary spread
z_score = (spread - spread.mean()) / spread.std()
```

### Key Code — Markov-Switching Regime Detection

```python
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

spread_returns = spread.diff().dropna()
ms_model  = MarkovRegression(spread_returns, k_regimes=2, trend='c',
                              switching_variance=True)
ms_result = ms_model.fit(disp=False)

# State with larger variance = high-volatility regime
regime_probs    = ms_result.smoothed_marginal_probabilities
high_vol_regime = (regime_probs[1] > 0.5).astype(int)  # 1 = stay out
```

### Key Code — Regime-Filtered Backtest

```python
def backtest(z_score, spread, regime_mask, entry=1.0, exit_th=0.0, filtered=True):
    positions = pd.Series(0, index=z_score.index)
    for i in range(1, len(z_score)):
        if filtered and regime_mask.iloc[i] == 1:
            positions.iloc[i] = 0           # regime filter: flat
        elif z_score.iloc[i] > entry:
            positions.iloc[i] = -1          # sell spread
        elif z_score.iloc[i] < -entry:
            positions.iloc[i] =  1          # buy spread
        elif abs(z_score.iloc[i]) < exit_th:
            positions.iloc[i] =  0          # close
        else:
            positions.iloc[i] = positions.iloc[i - 1]

    pnl    = positions.shift(1) * spread.diff()
    sharpe = pnl.mean() / pnl.std() * np.sqrt(252)
    return pnl.cumsum(), sharpe


def kupiec_test(returns, var_level=0.95):
    """Kupiec POF test for VaR model validity (Basel III check)."""
    from scipy.stats import chi2
    VaR        = np.percentile(returns, (1 - var_level) * 100)
    exceptions = (returns < VaR).sum()
    n, p       = len(returns), 1 - var_level
    p_hat      = exceptions / n
    LR = -2 * (np.log(p**exceptions * (1-p)**(n-exceptions)) -
               np.log(p_hat**exceptions * (1-p_hat)**(n-exceptions)))
    p_value = 1 - chi2.cdf(LR, df=1)
    print(f"Exceptions: {exceptions}/{n} | VaR: {VaR:.4f} | LR: {LR:.3f} | p: {p_value:.4f}")
    return p_value > 0.05
```

---

## 3. Heston & Bates Option Pricing

**Domain:** Options Pricing · Stochastic Volatility · CIR Interest Rates
**Tools:** Python · SciPy · NumPy · Characteristic Functions · Carr-Madan · Lewis (2001)

### Overview
I calibrated and priced derivatives on **SM Energy Company (SM)** stock using the Heston (1993) stochastic volatility model via Lewis (2001) Fourier inversion, and the Bates (1996) model with jump components. Interest rate risk was modelled through the Cox-Ingersoll-Ross (CIR) model using Euribor data. Vanilla and Asian options were priced.

### Key Code — Heston Characteristic Function

```python
import numpy as np
from scipy.integrate import quad

def H93_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    """
    Characteristic function of the Heston (1993) model via Lewis (2001).
    phi(u, T) = exp(H1(u,T) + H2(u,T) * v0)
    """
    c1 = kappa_v * theta_v
    c2 = -np.sqrt((rho * sigma_v * u * 1j - kappa_v)**2
                  - sigma_v**2 * (-u * 1j - u**2))
    c3 = (kappa_v - rho * sigma_v * u * 1j + c2) / \
         (kappa_v - rho * sigma_v * u * 1j - c2)

    H1 = (r * u * 1j * T
          + (c1 / sigma_v**2) * (
              (kappa_v - rho * sigma_v * u * 1j + c2) * T
              - 2 * np.log((1 - c3 * np.exp(c2 * T)) / (1 - c3))
          ))
    H2 = ((kappa_v - rho * sigma_v * u * 1j + c2) / sigma_v**2
          * (1 - np.exp(c2 * T)) / (1 - c3 * np.exp(c2 * T)))
    return np.exp(H1 + H2 * v0)


def H93_call_price(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    """European call price via Lewis (2001) Fourier inversion."""
    k = np.log(K / S0)
    def integrand(u):
        cf = H93_char_func(u - 0.5j, T, r, kappa_v, theta_v, sigma_v, rho, v0)
        return np.real(np.exp(-1j * u * k) * cf / (u**2 + 0.25))
    integral, _ = quad(integrand, 0, 1000, limit=500)
    return S0 - np.sqrt(S0 * K) * np.exp(-r * T) / np.pi * integral


def simulate_cir(r0, kappa, theta, sigma, T, N, n_paths=10_000):
    """
    Cox-Ingersoll-Ross (1985) short-rate model.
    dr_t = kappa*(theta - r_t)*dt + sigma*sqrt(r_t)*dW_t
    """
    dt    = T / N
    rates = np.zeros((N + 1, n_paths))
    rates[0] = r0
    for t in range(1, N + 1):
        r        = rates[t - 1]
        dW       = np.random.normal(0, np.sqrt(dt), n_paths)
        rates[t] = np.maximum(
            r + kappa * (theta - r) * dt
            + sigma * np.sqrt(np.maximum(r, 0)) * dW, 0
        )
    return rates
```

---

## 4. Regime Switching: S&P 500 & Bitcoin

**Domain:** Markov Regime Switching · Volatility Regimes · COVID-19 Analysis
**Tools:** Python · statsmodels · yfinance · MarkovRegression

### Overview
I delivered a regime-switching time-series analysis of the **S&P 500** and **Bitcoin** over the pre/post-COVID period (2019-01-01 to 2022-09-30), characterising low- and high-volatility regimes, identifying the crash date (Mar 2020) and recovery (Nov 2020), and estimating regime-specific volatility levels.

### Key Code — Regime Period Extraction

```python
import pandas as pd, numpy as np, yfinance as yf
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

crash_date    = '2020-03-23'
recovery_date = '2020-11-09'

def extract_regime_periods(result, volatility_series):
    """Returns DataFrame: Regime | Start Date | End Date | Avg Volatility"""
    regime_series  = (result.smoothed_marginal_probabilities[1] > 0.5).astype(int)
    regime_changes = regime_series[regime_series != regime_series.shift(1)]

    regime_periods, current_regime, start_date = [], None, None
    for date, regime in regime_changes.items():
        if current_regime is not None:
            avg_vol = volatility_series[start_date:date].mean()
            regime_periods.append((current_regime, start_date, date, avg_vol))
        start_date, current_regime = date, regime

    if start_date is not None:
        regime_periods.append((current_regime, start_date,
                               volatility_series.index[-1],
                               volatility_series[start_date:].mean()))
    return pd.DataFrame(regime_periods,
                        columns=['Regime', 'Start Date', 'End Date', 'Avg Volatility'])

spy         = yf.download('^GSPC', start='2019-01-01', end='2022-09-30')['Close']
returns     = spy.pct_change().dropna()
rolling_vol = returns.rolling(21).std() * np.sqrt(252)

ms_res      = MarkovRegression(returns, k_regimes=2, trend='c',
                               switching_variance=True).fit(disp=False)
print(extract_regime_periods(ms_res, rolling_vol))
```

---

## 5. Risk-Aware Multi-Armed Bandit

**Domain:** Reinforcement Learning · Portfolio Selection · CVaR
**Tools:** Python · NumPy · UCB · Black-Scholes · Geometric Brownian Motion

### Overview
I implemented a risk-aware Multi-Armed Bandit (MAB) framework for dynamic portfolio selection (Huo & Fu 2017). The algorithm blends UCB exploration with Conditional Value at Risk (CVaR) across 30 S&P 500 stocks during the subprime mortgage crisis.

### Key Code — UCB + CVaR Portfolio Selector

```python
import numpy as np

def simulate_gbm_prices(S0, mu, sigma, T, N, n_paths):
    """Stock prices via Geometric Brownian Motion."""
    dt          = T / N
    dW          = np.random.normal(0, np.sqrt(dt), (N, n_paths))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * dW
    prices      = S0 * np.exp(np.cumsum(log_returns, axis=0))
    return np.vstack([np.full(n_paths, S0), prices])


class RiskAwareUCBPortfolio:
    """UCB exploration + CVaR risk penalty for portfolio arm selection."""

    def __init__(self, n_assets, alpha=0.95, lam=0.5):
        self.n            = n_assets
        self.alpha        = alpha
        self.lam          = lam
        self.counts       = np.zeros(n_assets)
        self.mean_returns = np.zeros(n_assets)
        self.all_returns  = [[] for _ in range(n_assets)]

    def _cvar(self, i):
        r = np.array(self.all_returns[i])
        if len(r) < 5:
            return 0.0
        threshold = np.percentile(r, (1 - self.alpha) * 100)
        tail      = r[r <= threshold]
        return -tail.mean() if len(tail) > 0 else 0.0

    def select_arm(self, t):
        for i in range(self.n):
            if self.counts[i] == 0:
                return i
        scores = np.array([
            self.mean_returns[i]
            + np.sqrt(2 * np.log(t + 1) / self.counts[i])
            - self.lam * self._cvar(i)
            for i in range(self.n)
        ])
        return np.argmax(scores)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.all_returns[arm].append(reward)
        n = self.counts[arm]
        self.mean_returns[arm] = ((n - 1) * self.mean_returns[arm] + reward) / n
```

---

## 6. Binomial Tree & Put-Call Parity

**Domain:** Option Pricing · Greeks · Put-Call Parity
**Tools:** Python · NumPy · Binomial Tree (N=100)

### Overview
I implemented European option pricing via the Cox-Ross-Rubinstein binomial tree (100 steps), computed option Greeks analytically, and confirmed put-call parity as a consistency check. This established the foundational pricing framework used in subsequent derivative projects.

### Key Code — Binomial Tree Pricer

```python
import numpy as np

def binomial_tree_european(S0, K, T, r, sigma, N, option_type='call'):
    """
    Cox-Ross-Rubinstein binomial tree.
    N=100 steps gives near-Black-Scholes accuracy.
    """
    dt = T / N
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1 / u
    p  = (np.exp(r * dt) - d) / (u - d)   # risk-neutral probability

    j  = np.arange(N + 1)
    ST = S0 * (u ** j) * (d ** (N - j))   # asset prices at maturity

    if option_type == 'call':
        option_values = np.maximum(ST - K, 0)
    else:
        option_values = np.maximum(K - ST, 0)

    for i in range(N, 0, -1):             # backward induction
        option_values = (np.exp(-r * dt)
                         * (p * option_values[1:i+1]
                            + (1 - p) * option_values[0:i]))
    return option_values[0]


S0, K, T, r, sigma, N = 100, 100, 0.25, 0.05, 0.20, 100

call = binomial_tree_european(S0, K, T, r, sigma, N, 'call')
put  = binomial_tree_european(S0, K, T, r, sigma, N, 'put')
print(f"Call: {call:.4f} | Put: {put:.4f}")

# Put-call parity: C + PV(K) = P + S
print(f"Parity LHS: {call + K * np.exp(-r * T):.4f}")
print(f"Parity RHS: {put  + S0:.4f}")
```

---

## 7. Black-Scholes & Monte Carlo Simulation

**Domain:** Black-Scholes · Monte Carlo · Option Greeks
**Tools:** Python · SciPy · NumPy

### Overview
I implemented the Black-Scholes closed-form formula for European calls and puts, computed all five Greeks (Delta, Gamma, Vega, Theta, Rho) analytically, and validated prices independently using Monte Carlo simulation with 50,000 paths and antithetic variates for variance reduction.

### Key Code — Black-Scholes & Greeks

```python
import numpy as np
from scipy.stats import norm

def black_scholes(S0, K, T, r, sigma, option_type='call'):
    """Closed-form European option pricer with full Greeks."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1

    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    vega  = S0 * norm.pdf(d1) * np.sqrt(T) / 100
    theta = (-(S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T)
             * norm.cdf(d2 if option_type == 'call' else -d2)) / 365
    rho   = K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2) / 100
    return {'price': price, 'delta': delta, 'gamma': gamma,
            'vega': vega,   'theta': theta, 'rho': rho}


def monte_carlo_european(S0, K, T, r, sigma, n_paths=50_000, option_type='call'):
    """Monte Carlo pricer with antithetic variates."""
    Z  = np.random.standard_normal(n_paths // 2)
    Z  = np.concatenate([Z, -Z])              # antithetic variance reduction
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0) if option_type == 'call' else np.maximum(K - ST, 0)
    price   = np.exp(-r * T) * payoffs.mean()
    se      = payoffs.std() / np.sqrt(n_paths)
    print(f"MC {option_type}: {price:.4f} +/- {1.96*se:.4f} (95% CI)")
    return price


result = black_scholes(100, 100, 0.25, 0.05, 0.20, 'call')
print(f"BS Price: {result['price']:.4f}  Delta: {result['delta']:.4f}")
print(f"Gamma: {result['gamma']:.4f}  Vega: {result['vega']:.4f}")
monte_carlo_european(100, 100, 0.25, 0.05, 0.20, option_type='call')
```

---

## 8. Heston & Merton Jump Diffusion

**Domain:** Stochastic Volatility · Jump Diffusion · Exotic Options
**Tools:** Python · NumPy · SciPy · Euler-Maruyama · Barrier Options

### Overview
I extended the pricing framework to the Heston stochastic volatility model and the Merton Jump Diffusion model, pricing European, American, and barrier options. Both models capture dynamics missed by Black-Scholes: stochastic variance (Heston) and discontinuous price jumps (Merton).

### Key Code — Heston Monte Carlo Pricer

```python
import numpy as np

def heston_mc(S0, K, T, r, kappa, theta, sigma_v, rho, v0,
              N=50, n_paths=100_000, option_type='call'):
    """
    Heston (1993) Monte Carlo via Euler-Maruyama.
    dS = r*S*dt + sqrt(v)*S*dZ1
    dv = kappa*(theta-v)*dt + sigma_v*sqrt(v)*dZ2   Corr(dZ1,dZ2) = rho
    """
    dt = T / N
    S  = np.full(n_paths, float(S0))
    v  = np.full(n_paths, float(v0))

    for _ in range(N):
        Z1 = np.random.standard_normal(n_paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n_paths)

        v = np.maximum(v + kappa * (theta - v) * dt
                       + sigma_v * np.sqrt(np.maximum(v, 0) * dt) * Z2, 0)
        S = S * np.exp((r - 0.5 * v) * dt
                       + np.sqrt(np.maximum(v, 0) * dt) * Z1)

    payoff = np.maximum(S - K, 0) if option_type == 'call' else np.maximum(K - S, 0)
    return np.exp(-r * T) * payoff.mean()


params = dict(S0=100, K=100, T=0.25, r=0.05, kappa=2.0,
              theta=0.04, sigma_v=0.3, v0=0.04, N=50, n_paths=100_000)

print(f"Heston Call (rho=-0.30): {heston_mc(**params, rho=-0.30):.4f}")
print(f"Heston Call (rho=-0.70): {heston_mc(**params, rho=-0.70):.4f}")
```

---

## 9. Oil Price Forecasting — Probabilistic Graphical Models

**Team:** Joseph Bidias (USA) · Keolebogile Seisa (Botswana) · Himanshu Rane (Netherlands)
**Domain:** Bayesian Networks · HMM · Data Collection
**Tools:** Python · FRED API · yfinance · pgmpy · hmmlearn · networkx

### Overview
First phase of a three-part mini-capstone on crude oil price forecasting. I collected 179 months (Feb 2010 - Dec 2024) of FRED and Yahoo Finance macroeconomic data, cleaned it, and prepared the joint dataset for regime modelling.

### Key Code — Macroeconomic Data Collection

```python
import pandas as pd, yfinance as yf
from fredapi import Fred

fred = Fred(api_key='YOUR_KEY_HERE')  # use .env in production
START_DATE, END_DATE = '2010-01-01', '2024-12-31'

def collect_macroeconomic_data():
    macro_vars = {
        'INDPRO':   'Industrial Production Index',
        'CPIAUCSL': 'Consumer Price Index',
        'PPIACO':   'Producer Price Index',
        'DGS10':    '10-Year Treasury Rate',
        'FEDFUNDS': 'Federal Funds Rate',
        'UNRATE':   'Unemployment Rate',
        'GDP':      'Real GDP',
        'M2SL':     'M2 Money Stock',
        'DEXCHUS':  'China/US FX Rate',
        'VIXCLS':   'CBOE VIX',
    }
    data = {}
    for series_id, name in macro_vars.items():
        try:
            data[series_id] = fred.get_series(series_id,
                                              observation_start=START_DATE,
                                              observation_end=END_DATE)
            print(f"  OK {name}: {len(data[series_id])} observations")
        except Exception as e:
            print(f"  FAIL {series_id}: {e}")
    return pd.DataFrame(data).resample('MS').last()

wti       = yf.download('CL=F', start=START_DATE, end=END_DATE)['Close']
wti_m     = wti.resample('MS').last().rename('DCOILWTICO')
macro_df  = collect_macroeconomic_data()
oil_data  = macro_df.join(wti_m, how='inner').dropna()
print(f"\nFinal dataset: {oil_data.shape[0]} months x {oil_data.shape[1]} features")
```

---

## 10. HMM Regime Detection & Bayesian Network

**Domain:** Hidden Markov Models · Bayesian Networks · Oil Regimes
**Tools:** Python · hmmlearn · pgmpy · GaussianHMM · DiscreteBayesianNetwork

### Overview
Building on the cleaned dataset, I fitted a Gaussian HMM to detect WTI oil price regimes (low / medium / high volatility), then built a Bayesian Network to model probabilistic relationships between macroeconomic variables and oil price direction.

### Key Code — HMM Regime Detection

```python
import numpy as np, pandas as pd
from hmmlearn.hmm import GaussianHMM

def fit_oil_hmm(returns_series, n_states=3, n_iter=1000):
    """Fit Gaussian HMM and label regimes by volatility level."""
    X     = returns_series.values.reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type='full',
                        n_iter=n_iter, random_state=42)
    model.fit(X)
    hidden_states = model.predict(X)

    order         = np.argsort(np.abs(model.covars_.flatten()))
    regime_labels = {order[0]: 'Low Vol', order[1]: 'Medium Vol', order[2]: 'High Vol'}
    named_states  = pd.Series([regime_labels[s] for s in hidden_states],
                               index=returns_series.index)
    print("Transition matrix:\n", np.round(model.transmat_, 3))
    return model, named_states
```

### Key Code — Bayesian Network (Hill-Climb BIC)

```python
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BIC
from pgmpy.inference import VariableElimination

def build_bayesian_network(data_discretized):
    hc             = HillClimbSearch(data_discretized)
    best_structure = hc.estimate(scoring_method=BIC(data_discretized),
                                 max_indegree=3, max_iter=int(1e4))
    bn_model = DiscreteBayesianNetwork(best_structure.edges())
    bn_model.fit(data_discretized, estimator=MaximumLikelihoodEstimator)
    print("Edges learned:", list(bn_model.edges()))
    assert bn_model.check_model()
    return bn_model


def query_oil_direction(bn_model, evidence_dict):
    infer  = VariableElimination(bn_model)
    result = infer.query(variables=['OilDirection'], evidence=evidence_dict)
    print(result)
    return result
```

---

## 11. Full Model Evaluation & Improvement

**Domain:** Model Evaluation · Overfitting Correction · Full Train-Val-Test
**Tools:** Python · sklearn · pgmpy · GaussianHMM

### Overview
The prior phase achieved 100% test accuracy on a single observation — textbook overfitting. I redesigned with a proper 60/20/20 split across all 179 months, improving genuine test accuracy to ~65% on 37 hold-out months.

| Metric | Before | After |
|---|---|---|
| Training observations | 4 | 107 |
| Data-to-parameter ratio | 0.008 | 0.214 |
| Test set size | 1 month | 37 months |
| Test accuracy | 100% (meaningless) | ~65% (genuine) |
| Medium regime detected | 0% | ~30% |

### Key Code — Full Evaluation Pipeline

```python
from sklearn.metrics import accuracy_score, classification_report
import numpy as np, pandas as pd
from hmmlearn.hmm import GaussianHMM

def full_evaluation_pipeline(df_returns, feature_cols, target_col,
                              train_ratio=0.60, val_ratio=0.20):
    """
    Proper temporal split avoiding look-ahead bias.
    Fits HMM on training data only, evaluates on held-out test set.
    """
    n       = len(df_returns)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = df_returns.iloc[:n_train]
    test  = df_returns.iloc[n_train + n_val:]
    print(f"Train: {len(train)} | Val: {n_val} | Test: {len(test)}")

    hmm_model = GaussianHMM(n_components=3, covariance_type='full',
                             n_iter=200, random_state=42)
    hmm_model.fit(train[feature_cols].values)

    full_states = hmm_model.predict(df_returns[feature_cols].values)
    test_states = full_states[n_train + n_val:]
    true_labels = df_returns[target_col].iloc[n_train + n_val:].values

    acc = accuracy_score(true_labels, test_states)
    print(f"Test Accuracy: {acc:.3f}")
    print(classification_report(true_labels, test_states,
                                 target_names=['Low', 'Medium', 'High']))
    return acc, hmm_model
```

---

## 12. Multi-Asset Portfolio Optimization Engine

**Domain:** Mean-Variance Optimization · Efficient Frontier
**Tools:** Python · SciPy · yfinance · NumPy

### Overview
I built a full-featured portfolio optimization engine from scratch. `PortfolioData` handles data download; `PortfolioOptimizer` solves for minimum variance, maximum Sharpe, and efficient frontier portfolios via `scipy.optimize.minimize`.

### Key Code — Portfolio Data & Optimizer

```python
import numpy as np, pandas as pd, yfinance as yf
from scipy.optimize import minimize


class PortfolioData:
    def __init__(self, tickers, start_date, end_date):
        self.tickers    = tickers
        self.start_date = start_date
        self.end_date   = end_date

    def download_and_prepare(self):
        self.data    = yf.download(self.tickers, start=self.start_date,
                                   end=self.end_date, progress=False)['Close']
        self.returns = self.data.pct_change().dropna()
        self.mu      = self.returns.mean() * 252
        self.Sigma   = self.returns.cov()  * 252
        return self


class PortfolioOptimizer:
    def __init__(self, mu, Sigma, rf=0.02):
        self.mu    = mu.values
        self.Sigma = Sigma.values
        self.rf    = rf
        self.n     = len(mu)

    def portfolio_stats(self, w):
        ret = w @ self.mu
        vol = np.sqrt(w @ self.Sigma @ w)
        return ret, vol, (ret - self.rf) / vol

    def max_sharpe(self):
        res = minimize(lambda w: -self.portfolio_stats(w)[2],
                       np.ones(self.n) / self.n, method='SLSQP',
                       bounds=[(0, 1)] * self.n,
                       constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}])
        return res.x, self.portfolio_stats(res.x)

    def efficient_frontier(self, n_points=100):
        targets = np.linspace(self.mu.min(), self.mu.max(), n_points)
        vols    = []
        for t in targets:
            res = minimize(lambda w: self.portfolio_stats(w)[1],
                           np.ones(self.n) / self.n, method='SLSQP',
                           bounds=[(0, 1)] * self.n,
                           constraints=[
                               {'type': 'eq', 'fun': lambda w: w.sum() - 1},
                               {'type': 'eq', 'fun': lambda w, t=t: w @ self.mu - t}
                           ])
            vols.append(res.fun if res.success else np.nan)
        return targets, np.array(vols)
```

---

## 13. Mean-Variance: Tech & Healthcare Portfolio

**Domain:** Portfolio Construction · Risk Attribution · Factor Analysis
**Assets:** AAPL · NVDA · TSLA · XOM · REGN · LLY · JPM
**Tools:** Python · pandas · seaborn · scipy · yfinance

### Overview
I constructed and analysed a seven-asset portfolio spanning technology, energy, biotech, and financials (2023-2025), computing summary statistics, correlation matrix, efficient portfolios, and risk decomposition.

### Key Code — Summary Statistics & Correlation Matrix

```python
import yfinance as yf, pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns

ASSETS  = ['AAPL', 'NVDA', 'TSLA', 'XOM', 'REGN', 'LLY', 'JPM']
data    = yf.download(ASSETS, start='2023-01-01', end='2025-06-30',
                      progress=False)['Close']
returns = data.pct_change().dropna()
T       = 252

stats_df = pd.DataFrame({
    'Annual Return': returns.mean() * T,
    'Annual Vol':    returns.std()  * np.sqrt(T),
    'Sharpe':       (returns.mean() * T - 0.02) / (returns.std() * np.sqrt(T)),
    'Skewness':     returns.skew(),
    'Kurtosis':     returns.kurtosis(),
    'Max Drawdown': (data / data.cummax() - 1).min(),
})

plt.figure(figsize=(9, 7))
sns.heatmap(returns.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Asset Correlation Matrix', fontweight='bold')
plt.tight_layout(); plt.show()
print(stats_df.round(4))
```

---

## 14. ML-Enhanced Portfolio Optimization

**Domain:** Covariance Estimation · Clustering · Convex Optimisation
**Tools:** Python · Ledoit-Wolf · KMeans · PCA · CVXPY

### Overview
I extended mean-variance optimization with: (1) **Ledoit-Wolf shrinkage** for robust covariance estimation, (2) **K-Means clustering** for asset grouping by risk profile, and (3) **CVXPY** for disciplined convex portfolio optimization with custom weight constraints.

### Key Code — Ledoit-Wolf + CVXPY Optimizer

```python
import numpy as np, cvxpy as cp
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def ledoit_wolf_portfolio(returns, rf=0.02, max_weight=0.30):
    """Maximum Sharpe portfolio with Ledoit-Wolf covariance shrinkage."""
    mu    = returns.mean().values * 252
    lw    = LedoitWolf().fit(returns.values)
    Sigma = lw.covariance_ * 252
    n     = len(mu)

    w   = cp.Variable(n)
    vol = cp.quad_form(w, Sigma)

    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight,
                   mu @ w >= rf + 0.05]
    prob = cp.Problem(cp.Minimize(vol), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    return w.value, float(mu @ w.value), float(np.sqrt(vol.value))


def cluster_assets(returns, n_clusters=3):
    """K-Means clustering of assets by return/volatility profile."""
    features = np.column_stack([returns.mean() * 252,
                                returns.std()  * np.sqrt(252)])
    X_scaled = StandardScaler().fit_transform(features)
    labels   = KMeans(n_clusters=n_clusters, random_state=42,
                      n_init=10).fit_predict(X_scaled)
    return dict(zip(returns.columns, labels))
```

---

## 15. Regression Trees for Financial Prediction

**Domain:** Decision Trees · Grid Search · Housing & Financial Data
**Tools:** Python · scikit-learn · GridSearchCV

### Overview
I implemented and tuned regression decision trees using the California Housing dataset, applied Grid Search cross-validation to find optimal hyperparameters, visualised the tree structure, and diagnosed overfitting through learning curves.

### Key Code — Regression Tree with Grid Search

```python
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

X, y          = fetch_california_housing(return_X_y=True)
feature_names = fetch_california_housing().feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      random_state=42)

# Baseline
baseline_mse = mean_squared_error(
    y_test,
    DecisionTreeRegressor(max_depth=5, random_state=42).fit(X_train, y_train).predict(X_test)
)

# Grid Search
param_grid = {'max_depth':         [3, 5, 8, 10, None],
              'min_samples_split':  [2, 5, 10, 20],
              'min_samples_leaf':   [1, 2, 5, 10]}
gs = GridSearchCV(DecisionTreeRegressor(random_state=42),
                  param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
gs.fit(X_train, y_train)
best_tree = gs.best_estimator_
tuned_mse = mean_squared_error(y_test, best_tree.predict(X_test))

print(f"Baseline MSE : {baseline_mse:.4f}")
print(f"Tuned MSE    : {tuned_mse:.4f}  (improvement: {baseline_mse - tuned_mse:.4f})")
print(f"Best params  : {gs.best_params_}")

plt.figure(figsize=(20, 8))
plot_tree(best_tree, filled=True, feature_names=feature_names,
          max_depth=3, fontsize=8)
plt.title("Tuned Regression Tree", fontweight='bold')
plt.show()
```

---

## 16. Linear Discriminant Analysis for Market Classification

**Domain:** Dimensionality Reduction · Classification · Market Regimes
**Tools:** Python · scikit-learn · LDA · StandardScaler

### Overview
I applied LDA to reduce a high-dimensional market feature space to two discriminant functions and classify market regimes into three states (Bull / Bear / Sideways), demonstrating LDA's advantage over raw PCA for supervised classification.

### Key Code — LDA Classification Pipeline

```python
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n  = 300
X  = np.vstack([
    np.random.multivariate_normal([0,  0], [[1, 0.5], [0.5, 1]],  n // 2),
    np.random.multivariate_normal([3,  3], [[1, 0.5], [0.5, 1]],  n // 2),
    np.random.multivariate_normal([-3, 3], [[1,-0.5],[-0.5, 1]],  n // 2),
])
y  = np.array([0] * (n // 2) + [1] * (n // 2) + [2] * (n // 2))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                           random_state=42, stratify=y)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

lda    = LinearDiscriminantAnalysis(n_components=2)
lda.fit_transform(X_tr_s, y_tr)

y_pred = lda.predict(X_te_s)
print(f"LDA Accuracy: {accuracy_score(y_te, y_pred):.3f}")
print(classification_report(y_te, y_pred, target_names=['Bull', 'Bear', 'Sideways']))
```

---

## 17. Hyperparameter Optimization & Bias-Variance Tradeoff

**Domain:** Model Tuning · Random Forest · Learning Curves
**Tools:** Python · scikit-learn · GridSearchCV · RandomizedSearchCV

### Overview
I systematically explored Grid Search, Random Search, and Bayesian Optimization for tuning a Random Forest classifier, and built learning curve diagnostics to characterise the bias-variance tradeoff.

### Key Code — Random Forest Tuning & Learning Curves

```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve

X, y = make_classification(n_samples=1000, n_features=20,
                            n_informative=15, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42)

gs = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, None],
     'max_features': ['sqrt', 'log2']},
    cv=5, scoring='accuracy', n_jobs=-1
)
gs.fit(X_train, y_train)
model = gs.best_estimator_
print(f"Best params : {gs.best_params_}")
print(f"CV accuracy : {gs.best_score_:.4f}")

# Learning curves
train_sizes, train_sc, val_sc = learning_curve(
    model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
train_mean, train_std = train_sc.mean(1), train_sc.std(1)
val_mean,   val_std   = val_sc.mean(1),   val_sc.std(1)

plt.figure(figsize=(10, 5))
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                 alpha=0.1, color='r')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                 alpha=0.1, color='g')
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, val_mean,   'o-', color='g', label='Validation score')
plt.title('Learning Curve - Random Forest')
plt.xlabel('Training Examples'); plt.ylabel('Score')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
```

---

## 18. Statistical Arbitrage with CNN/LSTM

**Domain:** Equity Time-Series · Stationarity · Fractional Differentiation
**Tools:** Python · TensorFlow · Keras · CNN · LSTM · ADF Test · yfinance

### Overview
I built a statistical arbitrage optimization framework using AAPL equity data, implementing fractional differentiation (Lopez de Prado 2018) to achieve stationarity without destroying time-series memory. Both LSTM and CNN architectures were trained and compared for out-of-sample forecast accuracy.

### Key Code — Fractional Differentiation & ADF Test

```python
import numpy as np, pandas as pd, yfinance as yf
from statsmodels.tsa.stattools import adfuller
from scipy import stats


class FinanceTimeSeriesAnalyzer:
    def __init__(self, symbol='AAPL', period='5y'):
        self.symbol = symbol
        self.period = period

    def fetch_data(self):
        data = yf.Ticker(self.symbol).history(period=self.period)
        self.price_series = data['Close'].tail(2000).copy()
        return self.price_series

    def analyze_series(self, series, title='Time Series'):
        adf_stat, p_value, *_ = adfuller(series.dropna())
        status = 'STATIONARY' if p_value < 0.05 else 'NON-STATIONARY'
        print(f"[{status}] {title} | ADF={adf_stat:.4f} p={p_value:.4f} "
              f"Skew={stats.skew(series):.3f} Kurt={stats.kurtosis(series):.3f}")
        return p_value < 0.05

    def fractional_diff(self, series, d=0.35, threshold=1e-5):
        """
        Fractional differentiation (Lopez de Prado 2018).
        Preserves memory while achieving stationarity.
        """
        w = [1.0]
        for k in range(1, len(series)):
            w.append(-w[-1] * (d - k + 1) / k)
            if abs(w[-1]) < threshold:
                break
        w = np.array(w[::-1])
        return pd.Series(
            [np.dot(w, series.iloc[i - len(w) + 1:i + 1])
             for i in range(len(w) - 1, len(series))],
            index=series.index[len(w) - 1:]
        )
```

### Key Code — LSTM Predictor

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler


def build_lstm_model(input_shape, units=64):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units // 2),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)


scaler = MinMaxScaler()
scaled = scaler.fit_transform(prices.values.reshape(-1, 1))
X, y   = create_sequences(scaled, seq_len=60)
split  = int(0.8 * len(X))

model = build_lstm_model((60, 1))
model.fit(X[:split], y[:split], epochs=50, batch_size=32,
          validation_split=0.1,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=5,
                                                      restore_best_weights=True)])
```

---

## 19. Multi-Asset Portfolio Allocation with LSTM

**Domain:** Multi-Output Deep Learning · Cross-Asset LSTM · Portfolio Signals
**Assets:** SPY · TLT · SHY · GLD · DBO
**Tools:** Python · TensorFlow · Keras · yfinance

### Overview
I designed and trained individual LSTM models for each of five asset classes, then built a shared-encoder multi-output LSTM to jointly predict allocation signals. Three trading strategies were backtested from the model's forecasts.

### Key Code — Multi-Output LSTM Architecture

```python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input


def build_multi_output_lstm(seq_len, n_features, n_assets):
    """
    Shared LSTM encoder -> n_assets output heads.
    Single input sequence predicts all asset returns simultaneously.
    """
    inp    = Input(shape=(seq_len, n_features))
    x      = LSTM(128, return_sequences=True)(inp)
    x      = Dropout(0.2)(x)
    x      = LSTM(64)(x)
    x      = Dropout(0.2)(x)
    shared = Dense(32, activation='relu')(x)

    outputs = [Dense(1, name=f'asset_{i}')(shared) for i in range(n_assets)]
    model   = Model(inputs=inp, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='mse', metrics=['mae'])
    return model


def implement_trading_strategies(predictions_df, returns_df):
    """
    Strategy 1: Long-only momentum (long when forecast > 0)
    Strategy 2: Dollar-neutral long/short
    Strategy 3: Risk-parity weighted by predicted volatility
    """
    signals = (predictions_df > 0).astype(int)
    strat1  = (signals * returns_df).mean(axis=1)
    strat2  = (signals - 0.5) * 2 * returns_df.mean(axis=1)
    vol_est = returns_df.rolling(21).std()
    rp_wts  = 1 / vol_est.div(vol_est.sum(axis=1), axis=0)
    strat3  = (rp_wts * returns_df).sum(axis=1)
    return strat1, strat2, strat3
```

---

## 20. Data Leakage in Walk-Forward Backtesting

**Domain:** Walk-Forward Validation · Data Leakage · LSTM · MLP
**Tools:** Python · TensorFlow · scikit-learn · Keras · AAPL equity

### Overview
I systematically investigated data leakage in financial deep learning backtesting. Three setups were compared: (1) single split with intentional leakage, (2) walk-forward with leakage, and (3) walk-forward with controls — demonstrating the gap between inflated in-sample and realistic out-of-sample metrics.

### Key Code — Walk-Forward with Leakage Controls

```python
import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score


class GWP3ComprehensiveAnalysis:
    def __init__(self, symbol='AAPL', seq_len=60):
        self.symbol  = symbol
        self.seq_len = seq_len

    def step1_with_leakage(self, prices):
        """
        LEAKAGE: scaler fitted on full dataset before splitting.
        Common mistake -- future data influences the normalisation.
        """
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(prices.values.reshape(-1, 1))  # <- LEAKAGE
        X, y   = self._sequences(scaled)
        split  = int(0.8 * len(X))
        return (X[:split], y[:split]), (X[split:], y[split:])

    def step3_no_leakage(self, prices, n_splits=5):
        """
        CORRECTED: scaler re-fitted on training window only at each fold.
        No future information leaks into past observations.
        """
        results   = []
        fold_size = len(prices) // n_splits
        for fold in range(1, n_splits):
            train_prices = prices.iloc[:fold * fold_size]
            test_prices  = prices.iloc[fold * fold_size:(fold + 1) * fold_size]

            scaler       = MinMaxScaler()   # fit on TRAIN only
            s_train      = scaler.fit_transform(train_prices.values.reshape(-1, 1))
            s_test       = scaler.transform(test_prices.values.reshape(-1, 1))

            X_tr, y_tr   = self._sequences(s_train)
            X_te, y_te   = self._sequences(s_test)

            model  = self._build_lstm((self.seq_len, 1))
            model.fit(X_tr, y_tr, epochs=20, batch_size=32, verbose=0,
                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])
            preds  = model.predict(X_te, verbose=0)
            results.append({'fold': fold,
                            'mse':  mean_squared_error(y_te, preds),
                            'r2':   r2_score(y_te, preds)})
            print(f"Fold {fold}: MSE={results[-1]['mse']:.6f}  R2={results[-1]['r2']:.4f}")
        return pd.DataFrame(results)

    def _sequences(self, data):
        X, y = [], []
        for i in range(self.seq_len, len(data)):
            X.append(data[i - self.seq_len:i])
            y.append(data[i])
        return np.array(X), np.array(y)
```

---

## 21. Outlier Sensitivity in Regression

**Domain:** OLS Regression · Cook's Distance · Influential Points
**Assets:** SPY vs NVDA weekly returns (2014-2024)
**Tools:** Python · statsmodels · matplotlib

### Overview
I investigated OLS regression sensitivity to influential observations using ten years of weekly SPY/NVDA returns. Cook's distance identified the five most influential data points, and models with and without them were compared to quantify their distorting effect.

### Key Code — OLS, Cook's Distance & Influence Diagnostics

```python
import pandas as pd, datetime, matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import yfinance as yf

prices = pd.DataFrame(
    yf.download(['SPY', 'NVDA'],
                start=datetime.date(2014, 1, 1),
                end=datetime.date(2024, 1, 1),
                interval='1wk')['Close']
)
prices.index = prices.index.date
data = prices.pct_change().dropna()

# Full OLS
result_full = smf.ols("SPY ~ NVDA", data=data).fit()
print(result_full.summary())

# Cook's distance
influence = result_full.get_influence()
top5_idx  = (influence.summary_frame()
             .sort_values('cooks_d', ascending=False)
             .head(5).index)
print("Top 5 influential:\n", data.loc[top5_idx])

# Regression without influential points
result_clean = smf.ols("SPY ~ NVDA", data=data.drop(index=top5_idx)).fit()

print(f"\nCoef NVDA with outliers    : {result_full.params['NVDA']:.6f}")
print(f"Coef NVDA without outliers : {result_clean.params['NVDA']:.6f}")
print(f"R2 with outliers    : {result_full.rsquared:.4f}")
print(f"R2 without outliers : {result_clean.rsquared:.4f}")

sm.graphics.influence_plot(result_full, criterion='cooks', alpha=0.01)
plt.tight_layout(); plt.show()
```

---

## 22. Modeling Randomness & White Noise

**Domain:** White Noise · ACF · Ljung-Box Test · Stochastic Processes
**Tools:** Python · statsmodels · NumPy · matplotlib

### Overview
I implemented a full white noise analysis pipeline — simulation, ACF visualization, and formal Ljung-Box hypothesis testing — underpinning the residual independence assumption used throughout time-series modelling.

### Key Code — White Noise Analysis

```python
import numpy as np, matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

np.random.seed(42)
white_noise = np.random.normal(0, 1, 100)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(white_noise, linewidth=0.8, color='steelblue')
axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[0].set_title('Simulated White Noise')
axes[0].grid(True, alpha=0.3)
plot_acf(white_noise, lags=20, ax=axes[1], title='ACF of White Noise')
plt.tight_layout(); plt.show()

# Ljung-Box test: H0 = no serial autocorrelation
lb = acorr_ljungbox(white_noise, lags=[5, 10, 15, 20], return_df=True)
print(lb.to_string())

for lag, row in lb.iterrows():
    decision = ("Fail to reject H0 (white noise)"
                if row['lb_pvalue'] > 0.05
                else "Reject H0 (autocorrelation present)")
    print(f"Lag {lag:2d}: p={row['lb_pvalue']:.4f} -> {decision}")
```

---

## 23. Cointegration & VECM

**Domain:** Unit Roots · Cointegration · VECM · Long-Run Equilibrium
**Assets:** AAPL vs NASDAQ (2020-2023)
**Tools:** Python · statsmodels · Johansen test · VECM

### Overview
I tested for unit roots with ADF, established a cointegrating relationship between Apple and NASDAQ using the Johansen test, and estimated a Vector Error Correction Model (VECM) capturing the long-run equilibrium and short-run adjustment dynamics.

### Key Code — ADF, Johansen & VECM

```python
import yfinance as yf, numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

apple  = yf.download('AAPL',  start='2020-01-01', end='2023-12-31', progress=False)['Close']
nasdaq = yf.download('^IXIC', start='2020-01-01', end='2023-12-31', progress=False)['Close']
data   = pd.concat([np.log(apple), np.log(nasdaq)], axis=1).dropna()
data.columns = ['Log_AAPL', 'Log_NASDAQ']


def adf_test(series, name):
    stat, p, *_ = adfuller(series)
    print(f"{name}: ADF={stat:.4f} p={p:.4f} "
          f"{'STATIONARY' if p < 0.05 else 'NON-STATIONARY'}")
    return p

# Level tests (expect non-stationary)
adf_test(data['Log_AAPL'],                    'Log AAPL   (level)')
adf_test(data['Log_NASDAQ'],                  'Log NASDAQ (level)')
# First-difference tests (expect stationary)
adf_test(data['Log_AAPL'].diff().dropna(),    'Delta AAPL')
adf_test(data['Log_NASDAQ'].diff().dropna(),  'Delta NASDAQ')

# Johansen cointegration test
j = coint_johansen(data, det_order=0, k_ar_diff=1)
print("Trace statistics  :", j.lr1)
print("Critical (90/95/99):", j.cvt)

# VECM
print(VECM(data, k_ar_diff=1, coint_rank=1, deterministic='ci').fit().summary())
```

---

## 24. Lending EDA: Stock Returns, Housing & Rates

**Domain:** Exploratory Data Analysis · Multi-Source Financial Data
**Tools:** Python · FRED API · yfinance · pandas · seaborn

### Overview
I conducted a comprehensive EDA for a lending team, collecting and aligning four data streams: AAPL stock returns, S&P/Case-Shiller home price index, 30-year fixed mortgage rates, and 10-year Treasury rates (FRED). The analysis uncovered volatility patterns, rate-price correlations, and housing cycle dynamics.

### Key Code — Multi-Source Data Collection & EDA

```python
import pandas as pd, numpy as np, yfinance as yf
import matplotlib.pyplot as plt, seaborn as sns
from fredapi import Fred

fred = Fred(api_key='YOUR_KEY_HERE')   # use environment variable

apple_stock            = yf.download("AAPL", start="2015-01-01", end="2024-01-01",
                                      progress=False)
apple_stock['Returns'] = apple_stock['Close'].pct_change()

df_combined = pd.DataFrame({
    'Apple_Return':  apple_stock['Returns'].resample('MS').mean(),
    'HPI':           fred.get_series('CSUSHPINSA',  observation_start='2015-01-01').resample('MS').last(),
    'Mortgage_30yr': fred.get_series('MORTGAGE30US', observation_start='2015-01-01').resample('MS').last(),
    'Treasury_10yr': fred.get_series('DGS10',        observation_start='2015-01-01').astype(float).resample('MS').last(),
}).dropna()

plt.figure(figsize=(8, 6))
sns.heatmap(df_combined.corr(), annot=True, fmt='.3f',
            cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Cross-Asset Correlation: Returns, Housing & Rates', fontweight='bold')
plt.tight_layout(); plt.show()

# Rolling volatility
rolling_vol = apple_stock['Returns'].rolling(21).std() * np.sqrt(252)
rolling_vol.plot(figsize=(12, 4), color='steelblue',
                 title='AAPL Annualised Rolling Volatility (21-day)')
plt.ylabel('Annualised Volatility'); plt.grid(alpha=0.3); plt.show()
```

---

## 25. Financial Risk Analysis in Lending Scenarios

**Domain:** Credit Risk · Equity Risk · Liquidity Risk · Illiquid Securities
**Tools:** Python · FRED API · yfinance · NumPy · matplotlib

### Overview
I analysed financial risks across four lending scenarios: (1) unsecured credit cards, (2) business construction loans, (3) publicly traded equity (S&P 500), and (4) illiquid private equity — each modelled with real data and VaR/CVaR/Sharpe metrics.

### Key Code — Multi-Scenario Risk Analysis

```python
import pandas as pd, numpy as np, yfinance as yf
from fredapi import Fred

fred = Fred(api_key='YOUR_KEY_HERE')

cc_delinquency     = fred.get_series('DRCCLACBS')
construction_spend = fred.get_series('TTLCONS')
interest_rates     = fred.get_series('FEDFUNDS')

sp500      = yf.download('^GSPC', start='2010-01-01', end='2023-12-31',
                          progress=False)['Close']
sp_returns = sp500.pct_change().dropna()

# Illiquid PE simulation (quarterly, log-normal)
np.random.seed(42)
dates      = pd.date_range('2010-01-01', '2023-12-31', freq='QS')
pe_returns = pd.Series(np.random.lognormal(0.03, 0.15, len(dates)) - 1, index=dates)


def risk_metrics(returns, name):
    var_95 = np.percentile(returns, 5)
    cvar   = returns[returns <= var_95].mean()
    ann    = 252 if len(returns) > 200 else 4
    sharpe = returns.mean() / returns.std() * np.sqrt(ann)
    print(f"{name:35s} | VaR(95%)={var_95:.4f} | CVaR={cvar:.4f} | Sharpe={sharpe:.3f}")

risk_metrics(sp_returns.values, 'S&P 500 (Publicly Traded Equity)')
risk_metrics(pe_returns.values, 'Private Equity (Illiquid)')
```

---

## 26. Mortgage Amortization & Securities Lending

**Domain:** Fixed Income · Amortization · Securities Lending
**Tools:** Python · NumPy · pandas · yfinance

### Overview
I implemented two financial tools: (1) a floating-rate mortgage amortization schedule with periodic rate resets, and (2) a stock-lending profitability analyser using AAPL historical data.

### Key Code — Floating-Rate Mortgage Engine

```python
import numpy as np, pandas as pd


def calculate_monthly_payment(principal, annual_rate, years):
    """Standard annuity formula."""
    r = annual_rate / 12
    n = years * 12
    if r == 0:
        return principal / n
    return principal * r / (1 - (1 + r)**(-n))


def floating_rate_amortization(principal, rate_schedule, years_per_rate):
    """
    Amortization schedule across multiple rate periods.
    rate_schedule  : list of annual rates  e.g. [0.030, 0.035, 0.040]
    years_per_rate : matching list of years e.g. [5, 10, 15]
    """
    rows, balance, total_yr = [], principal, 0
    for rate, years in zip(rate_schedule, years_per_rate):
        pmt = calculate_monthly_payment(balance, rate, years)
        for yr in range(1, years + 1):
            interest       = balance * rate
            principal_paid = pmt * 12 - interest
            balance       -= principal_paid
            total_yr      += 1
            rows.append({
                'Year':              total_yr,
                'Rate':              f"{rate:.1%}",
                'Monthly Payment':   round(pmt, 2),
                'Interest Paid':     round(interest, 2),
                'Principal Paid':    round(principal_paid, 2),
                'Remaining Balance': round(max(balance, 0), 2)
            })
    return pd.DataFrame(rows)


# $300k, 30-year floating: 3% -> 3.5% -> 4% -> 4.5%
schedule = floating_rate_amortization(
    principal      = 300_000,
    rate_schedule  = [0.030, 0.035, 0.040, 0.045],
    years_per_rate = [5, 10, 10, 5]
)
print(schedule.head(10).to_string(index=False))
```

### Key Code — AAPL Securities Lending Analysis

```python
import yfinance as yf, matplotlib.pyplot as plt, numpy as np

aapl         = yf.download("AAPL", start="2010-01-01", end="2023-01-01",
                            progress=False)['Close']
aapl_returns = aapl.pct_change().dropna()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
aapl.plot(ax=ax1, color='steelblue', label='AAPL Price')
ax1.set_title('AAPL - Securities Lending Collateral Analysis')
ax1.legend()

aapl_returns.plot(ax=ax2, color='darkorange', linewidth=0.7)
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_title('Daily Returns')
plt.tight_layout(); plt.show()

print(f"Annualised Vol  : {aapl_returns.std() * np.sqrt(252):.2%}")
print(f"VaR (95%, 1-day): {np.percentile(aapl_returns, 5):.2%}")
print(f"Max Drawdown    : {(aapl / aapl.cummax() - 1).min():.2%}")
```

---

## Skills & Technology Stack

| Category | Technologies |
|---|---|
| **Languages** | Python 3.x |
| **Data** | pandas · NumPy · FRED API · yfinance · fredapi |
| **Statistics & Econometrics** | statsmodels · scipy · arch · GARCH · VECM · ADF · Johansen |
| **Machine Learning** | scikit-learn · GridSearchCV · LDA · Random Forest · Decision Trees · PCA · KMeans · Ledoit-Wolf |
| **Deep Learning** | TensorFlow · Keras · LSTM · CNN · MLP · EarlyStopping |
| **Derivatives & Pricing** | Black-Scholes · Binomial Tree · Heston · Bates · Merton Jump Diffusion · CIR · Carr-Madan · Lewis (2001) |
| **Risk Management** | VaR · CVaR · Kupiec Test · DCC-GARCH · HAR-RV · Bootstrap |
| **Portfolio Optimisation** | scipy.optimize · CVXPY · Efficient Frontier · Sharpe Maximisation · Risk Parity |
| **Probabilistic Models** | hmmlearn · pgmpy · HMM · Bayesian Networks · VariableElimination |
| **Stochastic Processes** | GBM · Markov-Switching · CIR · Euler-Maruyama · Fractional Differentiation |
| **Visualisation** | matplotlib · seaborn · plotly |
| **Version Control** | Git · GitHub |

---

*Portfolio compiled from 26 projects across 9 quantitative finance domains.*
