# Joseph Bidias â€” Quantitative Finance Portfolio



**MScFE Graduate Â· WorldQuant University**

Combining rigorous mathematical finance with modern machine learning and data engineering to build production-quality quantitative systems. Every project below was independently designed, coded, and delivered as part of the Master of Science in Financial Engineering program.



---



## Table of Contents

1. [Capstone â€” Integrated Risk Monitoring System](#1-integrated-risk-monitoring-system)

2. [Capstone â€” Regime-Aware Pairs Trader](#2-regime-aware-pairs-trader)

3. [Stochastic Modeling â€” Heston & Bates Option Pricing](#3-heston--bates-option-pricing)

4. [Stochastic Modeling â€” Regime Switching: S&P 500 & Bitcoin](#4-regime-switching-sp500--bitcoin)

5. [Stochastic Modeling â€” Risk-Aware Multi-Armed Bandit](#5-risk-aware-multi-armed-bandit)

6. [Derivative Pricing â€” Binomial Tree & Put-Call Parity](#6-binomial-tree--put-call-parity)

7. [Derivative Pricing â€” Black-Scholes & Monte Carlo Simulation](#7-black-scholes--monte-carlo-simulation)

8. [Derivative Pricing â€” Heston & Merton Jump Diffusion](#8-heston--merton-jump-diffusion)

9. [Risk Management â€” Oil Forecasting with Probabilistic Graphical Models](#9-oil-price-forecasting--probabilistic-graphical-models)

10. [Risk Management â€” HMM Regime Detection & Bayesian Network](#10-hmm-regime-detection--bayesian-network)

11. [Risk Management â€” Full Model Evaluation & Improvement](#11-full-model-evaluation--improvement)

12. [Portfolio Management â€” Multi-Asset Optimization Engine](#12-multi-asset-portfolio-optimization-engine)

13. [Portfolio Management â€” Mean-Variance: Tech & Healthcare Portfolio](#13-mean-variance-tech--healthcare-portfolio)

14. [Portfolio Management â€” ML-Enhanced Portfolio Optimization](#14-ml-enhanced-portfolio-optimization)

15. [Machine Learning in Finance â€” Regression Trees](#15-regression-trees-for-financial-prediction)

16. [Machine Learning in Finance â€” Linear Discriminant Analysis](#16-linear-discriminant-analysis-for-market-classification)

17. [Machine Learning in Finance â€” Hyperparameter Optimization](#17-hyperparameter-optimization--bias-variance-tradeoff)

18. [Deep Learning in Finance â€” Statistical Arbitrage with CNN/LSTM](#18-statistical-arbitrage-with-cnnlstm)

19. [Deep Learning in Finance â€” Multi-Asset Portfolio with LSTM](#19-multi-asset-portfolio-allocation-with-lstm)

20. [Deep Learning in Finance â€” Data Leakage in Walk-Forward Backtesting](#20-data-leakage-in-walk-forward-backtesting)

21. [Financial Econometrics â€” Outlier Sensitivity in Regression](#21-outlier-sensitivity-in-regression)

22. [Financial Econometrics â€” Modeling Randomness & White Noise](#22-modeling-randomness--white-noise)

23. [Financial Econometrics â€” Cointegration & VECM](#23-cointegration--vecm)

24. [Financial Data â€” Lending EDA: Stock Returns, Housing & Rates](#24-lending-eda-stock-returns-housing--rates)

25. [Financial Data â€” Financial Risk Analysis in Lending Scenarios](#25-financial-risk-analysis-in-lending-scenarios)

26. [Financial Data â€” Mortgage Amortization & Securities Lending](#26-mortgage-amortization--securities-lending)



---



## 1. Integrated Risk Monitoring System



**Domain:** Systemic Risk Â· Multi-Asset Monitoring  

**Tools:** Python Â· GARCH Â· DCC Â· PCA Â· Quantile Regression Â· HAR-RV Â· FRED Â· Bloomberg



### Overview

I built a full-stack integrated risk monitoring system calibrated to real market data spanning 2005â€“2024. The system fuses ten asset classes â€” equities (SPY, EFA, EEM, XLF, EWJ), fixed income (TLT, IEF, AGG, HYG), gold (GLD), VIX, TED spread, EUR/USD xccy basis, and HY OAS â€” into a unified dashboard that flags systemic stress in real time.



| Data Series | Instrument | Coverage |

|---|---|---|

| Equity ETFs | SPY, EFA, EEM, XLF, EWJ | 2005â€“2024 |

| Fixed Income | TLT, IEF, AGG, HYG | 2005â€“2024 |

| Commodity | GLD | 2005â€“2024 |

| Volatility | VIX (CBOE) | 2005â€“2024 |

| Funding stress | TED Spread (FRED) | 2005â€“2024 |

| Credit risk | HY OAS â€” BAMLH0A0HYM2 (FRED) | 2005â€“2024 |



GARCH parameters were estimated from actual historical data; correlation matrices were calibrated to empirical calm/crisis regimes (Engle 2002, Kritzman 2011). Spread anchors match real peaks: TED 463 bps (Oct 2008), HY OAS 1,994 bps (Dec 2008), VIX 82.69 (Mar 2020).



### Pipeline



| Step | Component |

|---|---|

| 1-A | PCA Systemic Risk Indicator |

| 1-B | Quantile Regression + Time-Varying Correlation |

| 1-C | DCC-GARCH + Funding Stress Regime Switching |

| 2 | Cascade Effect Analysis |

| 3-A | Bootstrap Correlation Uncertainty |

| 3-B | HAR-RV Volatility Forecasting |

| 4â€“5 | Historical Stress Backtests + Performance Table |

| 6 | Eight Improvements + Radar Chart |

| 7 | Final Risk Dashboard |



### Key Code â€” Environment & Data Loader



```python

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

from scipy import stats

from statsmodels.regression.quantile_regression import QuantReg

from arch import arch_model



np.random.seed(42)



# Real GARCH(1,1) parameters calibrated from historical data

GARCH_PARAMS = {

    'SPY':  {'omega': 1.02e-6, 'alpha': 0.0731, 'beta': 0.9189},

    'TLT':  {'omega': 1.82e-6, 'alpha': 0.0512, 'beta': 0.9327},

    'GLD':  {'omega': 2.21e-6, 'alpha': 0.0601, 'beta': 0.9264},

    'HYG':  {'omega': 8.95e-7, 'alpha': 0.0847, 'beta': 0.9018},

    'VIX':  {'omega': 9.12e-5, 'alpha': 0.1923, 'beta': 0.7614},

}



# Spread anchors to real historical peaks

SPREAD_PEAKS = {

    'ted_spread_bps': 463,      # Oct 2008 (FRED)

    'hy_oas_bps':    1994,      # Dec 2008 (FRED)

    'vix_close':     82.69,     # Mar 2020 (CBOE)

    'xccy_eur_usd':  -120,      # 2008 GFC (Bloomberg)

}

```



### Key Code â€” PCA Systemic Risk Indicator



```python

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



def build_pca_risk_indicator(returns_df, n_components=3):

    """

    Constructs a systemic risk composite score via PCA.

    First PC explains the dominant co-movement direction â€” the stress signal.

    """

    scaler = StandardScaler()

    scaled = scaler.fit_transform(returns_df.dropna())



    pca = PCA(n_components=n_components)

    pca.fit(scaled)

    components = pca.transform(scaled)



    # First PC as systemic risk indicator (sign-adjusted: high = stress)

    risk_score = pd.Series(-components[:, 0],

                           index=returns_df.dropna().index,

                           name='Systemic_Risk_PCA')

    explained = pca.explained_variance_ratio_

    print(f"Variance explained by PC1: {explained[0]:.1%}")

    return risk_score, pca

```



### Key Code â€” DCC-GARCH Regime Switch



```python

from arch import arch_model



def fit_dcc_garch(returns_series, asset_name):

    """Fit GARCH(1,1) and extract conditional volatility."""

    am = arch_model(returns_series * 100, vol='Garch', p=1, q=1, dist='Normal')

    res = am.fit(disp='off')

    cond_vol = res.conditional_volatility / 100

    print(f"{asset_name}: omega={res.params['omega']:.2e}, "

          f"alpha={res.params['alpha[1]']:.4f}, "

          f"beta={res.params['beta[1]']:.4f}")

    return cond_vol, res



# HAR-RV (Heterogeneous AutoRegressive Realized Volatility)

def har_rv_forecast(rv_series, horizon=5):

    """HAR-RV model: daily, weekly, monthly components."""

    df = pd.DataFrame({'RV_d': rv_series})

    df['RV_w'] = rv_series.rolling(5).mean()

    df['RV_m'] = rv_series.rolling(22).mean()

    df['RV_fwd'] = rv_series.shift(-horizon)

    df = df.dropna()



    X = sm.add_constant(df[['RV_d', 'RV_w', 'RV_m']])

    model = sm.OLS(df['RV_fwd'], X).fit()

    return model

```



---



## 2. Regime-Aware Pairs Trader



**Domain:** Statistical Arbitrage Â· Pairs Trading Â· Regime Filtering  

**Tools:** Python Â· Engle-Granger Â· Markov Switching Â· BDS Test Â· Kupiec VaR Â· yfinance



### Overview

I designed and back-tested a regime-filtered pairs trading strategy on the energy sector ETF pair **XLE / XOP** over January 2017 â€“ December 2021. The central hypothesis: *strategic inactivity during high-volatility regimes improves risk-adjusted returns.*



Two strategies were compared head-to-head:



| Strategy | Description |

|---|---|

| Always-On | Classic cointegration spread with constant entry/exit |

| Regime-Filtered | Suspends trading during HMM-identified high-volatility states |



### Pipeline



| Step | Content | Module |

|---|---|---|

| Data + EDA | XLE, XOP daily prices via yfinance | M5 + M6 |

| Cointegration | Engle-Granger test, spread construction | M5 Lessons 3â€“4 |

| Regime Detection | Markov-Switching (2-state) | M5 Lesson 2 |

| BDS Test | Nonlinearity test on residuals | M5 Lesson 1 |

| Market Impact | Liquidity-adjusted transaction costs | M6 Lesson 1 |

| Strategy Backtest | Full-period + 4 sub-periods | M5 + M6 |

| VaR Validation | Kupiec backtest, Basel III check | M6 Lesson 3 |

| Regulation | Regulatory impact on market liquidity | M6 Lesson 5 |



### Key Code â€” Engle-Granger Cointegration & Spread



```python

import yfinance as yf

import numpy as np, pandas as pd

from statsmodels.tsa.stattools import coint, adfuller

import statsmodels.api as sm



# Download XLE and XOP

tickers = ['XLE', 'XOP']

prices  = yf.download(tickers, start='2017-01-01', end='2021-12-31')['Close']



# Engle-Granger cointegration test

score, pvalue, _ = coint(prices['XLE'], prices['XOP'])

print(f"Cointegration p-value: {pvalue:.4f}")



# OLS hedge ratio (log-price regression)

log_xle = np.log(prices['XLE'])

log_xop = np.log(prices['XOP'])

model    = sm.OLS(log_xle, sm.add_constant(log_xop)).fit()

beta     = model.params['XOP']

spread   = log_xle - beta * log_xop        # stationary spread

z_score  = (spread - spread.mean()) / spread.std()

```



### Key Code â€” Markov-Switching Regime Detection



```python

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression



# Fit 2-state Markov-Switching model on spread returns

spread_returns = spread.diff().dropna()

ms_model  = MarkovRegression(spread_returns, k_regimes=2, trend='c', switching_variance=True)

ms_result = ms_model.fit(disp=False)



# High-volatility regime: state with larger variance

regime_probs     = ms_result.smoothed_marginal_probabilities

high_vol_regime  = (regime_probs[1] > 0.5).astype(int)   # 1 = high vol â†’ stay out

print(ms_result.summary())

```



### Key Code â€” Regime-Filtered Strategy Backtest



```python

def backtest(z_score, prices, regime_mask, entry=1.0, exit=0.0, filtered=True):

    """

    Pairs backtest with optional regime filter.

    filtered=True: skip all signals when high_vol_regime == 1

    """

    positions = pd.Series(0, index=z_score.index)

    for i in range(1, len(z_score)):

        if filtered and regime_mask.iloc[i] == 1:

            positions.iloc[i] = 0      # regime filter: flat

        elif z_score.iloc[i] > entry:

            positions.iloc[i] = -1     # sell spread

        elif z_score.iloc[i] < -entry:

            positions.iloc[i] =  1     # buy spread

        elif abs(z_score.iloc[i]) < exit:

            positions.iloc[i] =  0     # close

        else:

            positions.iloc[i] = positions.iloc[i-1]



    spread_returns = spread.diff()

    pnl = positions.shift(1) * spread_returns

    cum_pnl = pnl.cumsum()

    sharpe  = pnl.mean() / pnl.std() * np.sqrt(252)

    return cum_pnl, sharpe



cum_always_on,  sr_always  = backtest(z_score, prices, high_vol_regime, filtered=False)

cum_filtered,   sr_filtered = backtest(z_score, prices, high_vol_regime, filtered=True)

print(f"Always-On Sharpe:   {sr_always:.3f}")

print(f"Regime-Filtered:    {sr_filtered:.3f}")

```



### Key Code â€” Kupiec VaR Backtest



```python

from scipy.stats import chi2



def kupiec_test(returns, var_level=0.95, confidence=0.99):

    """Kupiec POF test for VaR model validity (Basel III check)."""

    VaR = np.percentile(returns, (1 - var_level) * 100)

    exceptions = (returns < VaR).sum()

    n  = len(returns)

    p  = 1 - var_level

    p_hat = exceptions / n



    # Likelihood ratio statistic

    LR = -2 * (np.log(p**exceptions * (1-p)**(n-exceptions)) -

               np.log(p_hat**exceptions * (1-p_hat)**(n-exceptions)))

    p_value = 1 - chi2.cdf(LR, df=1)

    print(f"Exceptions: {exceptions}/{n} | VaR: {VaR:.4f} | LR: {LR:.3f} | p: {p_value:.4f}")

    return p_value > 0.05   # True = model passes

```



---



## 3. Heston & Bates Option Pricing



**Domain:** Options Pricing Â· Stochastic Volatility Â· CIR Interest Rates  

**Tools:** Python Â· SciPy Â· NumPy Â· Characteristic Functions Â· Carr-Madan Â· Lewis (2001)



### Overview

I calibrated and priced derivatives on **SM Energy Company (SM)** stock using two industry-standard stochastic volatility models. The Heston (1993) model was implemented via the Lewis (2001) approach; the Bates (1996) model added jump components. Interest rate risk was approximated through the Cox-Ingersoll-Ross (CIR) model using Euribor data. Vanilla and Asian options were priced.



### Key Code â€” Heston Characteristic Function (Lewis 2001)



```python

import numpy as np

from scipy.integrate import quad



def H93_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0):

    """

    Characteristic function of the Heston (1993) model via Lewis (2001).



    Ï†^H(u, T) = exp(H1(u,T) + H2(u,T)Â·v0)

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

    call = S0 - np.sqrt(S0 * K) * np.exp(-r * T) / np.pi * integral

    return call

```



### Key Code â€” CIR Interest Rate Model



```python

def simulate_cir(r0, kappa, theta, sigma, T, N, n_paths=10_000):

    """

    Cox-Ingersoll-Ross (1985) interest rate model.

    dr_t = Îº(Î¸ âˆ’ r_t)dt + Ïƒâˆšr_t dW_t

    Euler-Maruyama discretisation.

    """

    dt   = T / N

    rates = np.zeros((N + 1, n_paths))

    rates[0] = r0

    for t in range(1, N + 1):

        r  = rates[t - 1]

        dW = np.random.normal(0, np.sqrt(dt), n_paths)

        rates[t] = np.maximum(r + kappa * (theta - r) * dt

                              + sigma * np.sqrt(np.maximum(r, 0)) * dW, 0)

    return rates

```



---



## 4. Regime Switching: S&P 500 & Bitcoin



**Domain:** Markov Regime Switching Â· Volatility Regimes Â· COVID-19 Analysis  

**Tools:** Python Â· statsmodels Â· yfinance Â· MarkovRegression



### Overview

I delivered a regime-switching time-series analysis of the **S&P 500** and **Bitcoin** over the pre/post-COVID period (2019-01-01 to 2022-09-30). The study was directed at portfolio and derivative risk management teams, characterising low- and high-volatility regimes, identifying the crash (Mar 2020) and recovery dates, and estimating regime-specific volatility levels.



### Key Code â€” Regime Period Extraction



```python

import pandas as pd

import numpy as np

import yfinance as yf

import statsmodels.api as sm

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression



crash_date    = '2020-03-23'

recovery_date = '2020-11-09'



def extract_regime_periods(result, volatility_series):

    """

    Extracts regime periods with start/end dates and average volatility.

    Returns a DataFrame: Regime | Start Date | End Date | Avg Volatility

    """

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



# Download data and fit model

spy = yf.download('^GSPC', start='2019-01-01', end='2022-09-30')['Close']

returns = spy.pct_change().dropna()

rolling_vol = returns.rolling(21).std() * np.sqrt(252)



ms = MarkovRegression(returns, k_regimes=2, trend='c', switching_variance=True)

ms_res = ms.fit(disp=False)



regime_table = extract_regime_periods(ms_res, rolling_vol)

print(regime_table)

```



---



## 5. Risk-Aware Multi-Armed Bandit



**Domain:** Reinforcement Learning Â· Portfolio Selection Â· CVaR  

**Tools:** Python Â· NumPy Â· UCB Â· Black-Scholes Â· Geometric Brownian Motion



### Overview

I implemented a risk-aware extension of the Multi-Armed Bandit (MAB) framework for dynamic portfolio selection, replicating and extending the work of Huo & Fu (*"Risk-aware Multi-Armed Bandit Problem with Application to Portfolio Selection"*). The algorithm blends the Upper Confidence Bound (UCB) strategy with Conditional Value at Risk (CVaR) for risk-sensitive arm selection across 30 S&P 500 stocks during the subprime mortgage crisis.



### Key Code â€” UCB + CVaR Portfolio Selector



```python

import numpy as np

import pandas as pd



def simulate_gbm_prices(S0, mu, sigma, T, N, n_paths):

    """Simulate stock prices via Geometric Brownian Motion."""

    dt   = T / N

    dW   = np.random.normal(0, np.sqrt(dt), (N, n_paths))

    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * dW

    prices = S0 * np.exp(np.cumsum(log_returns, axis=0))

    return np.vstack([np.full(n_paths, S0), prices])



class RiskAwareUCBPortfolio:

    """

    MAB portfolio selector: UCB exploration + CVaR risk penalty.

    Each stock is an arm; reward = log-return; risk = CVaR at alpha.

    """

    def __init__(self, n_assets, alpha=0.95, lam=0.5):

        self.n     = n_assets

        self.alpha = alpha       # CVaR confidence level

        self.lam   = lam         # risk-return trade-off weight

        self.counts     = np.zeros(n_assets)

        self.mean_returns = np.zeros(n_assets)

        self.all_returns  = [[] for _ in range(n_assets)]



    def _cvar(self, i):

        r = np.array(self.all_returns[i])

        if len(r) < 5:

            return 0.0

        threshold = np.percentile(r, (1 - self.alpha) * 100)

        tail = r[r <= threshold]

        return -tail.mean() if len(tail) > 0 else 0.0



    def select_arm(self, t):

        """UCB selection adjusted for CVaR risk."""

        ucb_scores = np.zeros(self.n)

        for i in range(self.n):

            if self.counts[i] == 0:

                return i

            exploit  = self.mean_returns[i]

            explore  = np.sqrt(2 * np.log(t + 1) / self.counts[i])

            risk     = self._cvar(i)

            ucb_scores[i] = exploit + explore - self.lam * risk

        return np.argmax(ucb_scores)



    def update(self, arm, reward):

        self.counts[arm] += 1

        self.all_returns[arm].append(reward)

        n = self.counts[arm]

        self.mean_returns[arm] = ((n - 1) * self.mean_returns[arm] + reward) / n

```



---



## 6. Binomial Tree & Put-Call Parity



**Domain:** Option Pricing Â· Greeks Â· Put-Call Parity  

**Tools:** Python Â· NumPy Â· Binomial Tree (N=100)



### Overview

I implemented and verified European option pricing via the binomial tree model (100 steps), computed option Greeks analytically, and confirmed put-call parity. The project established the foundational pricing framework used in subsequent derivative pricing projects.



### Key Code â€” Binomial Tree Pricer



```python

import numpy as np



def binomial_tree_european(S0, K, T, r, sigma, N, option_type='call'):

    """

    European option pricing via Cox-Ross-Rubinstein binomial tree.

    N=100 steps for near-Black-Scholes accuracy.

    """

    dt  = T / N

    u   = np.exp(sigma * np.sqrt(dt))

    d   = 1 / u

    p   = (np.exp(r * dt) - d) / (u - d)     # risk-neutral probability



    # Asset prices at maturity

    j   = np.arange(N + 1)

    ST  = S0 * (u ** j) * (d ** (N - j))



    # Payoffs

    if option_type == 'call':

        option_values = np.maximum(ST - K, 0)

    else:

        option_values = np.maximum(K - ST, 0)



    # Backward induction

    for i in range(N, 0, -1):

        option_values = (np.exp(-r * dt)

                         * (p * option_values[1:i+1]

                            + (1 - p) * option_values[0:i]))

    return option_values[0]



# Parameters: ATM, 3-month, 5% rate, 20% vol

S0, K, T, r, sigma, N = 100, 100, 0.25, 0.05, 0.20, 100



call = binomial_tree_european(S0, K, T, r, sigma, N, 'call')

put  = binomial_tree_european(S0, K, T, r, sigma, N, 'put')

print(f"Call: {call:.4f} | Put: {put:.4f}")



# Verify put-call parity: C + PV(K) = P + S

parity_lhs = call + K * np.exp(-r * T)

parity_rhs = put  + S0

print(f"Put-Call Parity check: {parity_lhs:.4f} â‰ˆ {parity_rhs:.4f}")

```



---



## 7. Black-Scholes & Monte Carlo Simulation



**Domain:** Black-Scholes Â· Monte Carlo Â· Option Greeks  

**Tools:** Python Â· SciPy Â· NumPy



### Overview

I implemented the Black-Scholes closed-form formula for European calls and puts, computed Delta, Gamma, Vega, Theta, and Rho analytically, then validated prices independently using Monte Carlo simulation (50,000 paths).



### Key Code â€” Black-Scholes & Greeks



```python

import numpy as np

from scipy.stats import norm



def black_scholes(S0, K, T, r, sigma, option_type='call'):

    """

    Black-Scholes closed-form European option pricer.

    Returns price and all five Greeks.

    """

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)



    if option_type == 'call':

        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        delta = norm.cdf(d1)

    else:

        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

        delta = norm.cdf(d1) - 1



    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))

    vega  = S0 * norm.pdf(d1) * np.sqrt(T) / 100      # per 1% vol move

    theta = (-(S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

             - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)) / 365

    rho   = K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2) / 100



    return {'price': price, 'delta': delta, 'gamma': gamma,

            'vega': vega, 'theta': theta, 'rho': rho}



# ATM example

result = black_scholes(100, 100, 0.25, 0.05, 0.20, 'call')

print(f"Call Price: {result['price']:.4f}")

print(f"Delta: {result['delta']:.4f}  Gamma: {result['gamma']:.4f}")

print(f"Vega:  {result['vega']:.4f}  Theta: {result['theta']:.6f}")

```



### Key Code â€” Monte Carlo Validation



```python

def monte_carlo_european(S0, K, T, r, sigma, n_paths=50_000, option_type='call'):

    """Monte Carlo European option price with antithetic variates."""

    Z   = np.random.standard_normal(n_paths // 2)

    Z   = np.concatenate([Z, -Z])              # antithetic variance reduction

    ST  = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type == 'call':

        payoffs = np.maximum(ST - K, 0)

    else:

        payoffs = np.maximum(K - ST, 0)

    price = np.exp(-r * T) * payoffs.mean()

    se    = payoffs.std() / np.sqrt(n_paths)

    print(f"MC {option_type}: {price:.4f} Â± {1.96*se:.4f} (95% CI)")

    return price

```



---



## 8. Heston & Merton Jump Diffusion



**Domain:** Stochastic Volatility Â· Jump Diffusion Â· Exotic Options  

**Tools:** Python Â· NumPy Â· SciPy Â· Euler-Maruyama Â· Barrier Options



### Overview

I extended the derivative pricing framework to the **Heston stochastic volatility model** and the **Merton Jump Diffusion model**, pricing European, American, and barrier options. Both models capture dynamics missed by Black-Scholes: stochastic variance (Heston) and discontinuous asset price jumps (Merton).



### Key Code â€” Heston Monte Carlo Pricer



```python

import numpy as np



def heston_mc(S0, K, T, r, kappa, theta, sigma_v, rho, v0,

              N=50, n_paths=100_000, option_type='call'):

    """

    Heston (1993) Monte Carlo via Euler-Maruyama discretization.

    dS = rÂ·SÂ·dt + âˆšvÂ·SÂ·dZ1

    dv = Îº(Î¸âˆ’v)dt + Ïƒ_vÂ·âˆšvÂ·dZ2    Corr(dZ1, dZ2) = Ï

    """

    dt  = T / N

    S   = np.full(n_paths, float(S0))

    v   = np.full(n_paths, float(v0))



    for _ in range(N):

        Z1 = np.random.standard_normal(n_paths)

        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n_paths)



        v  = np.maximum(v + kappa * (theta - v) * dt

                        + sigma_v * np.sqrt(np.maximum(v, 0) * dt) * Z2, 0)

        S  = S * np.exp((r - 0.5 * v) * dt

                        + np.sqrt(np.maximum(v, 0) * dt) * Z1)



    payoff = np.maximum(S - K, 0) if option_type == 'call' else np.maximum(K - S, 0)

    price  = np.exp(-r * T) * payoff.mean()

    return price



# Heston prices with different correlations

params = dict(S0=100, K=100, T=0.25, r=0.05, kappa=2.0,

              theta=0.04, sigma_v=0.3, v0=0.04, N=50, n_paths=100_000)



call_rho_030 = heston_mc(**params, rho=-0.30, option_type='call')

call_rho_070 = heston_mc(**params, rho=-0.70, option_type='call')

print(f"Heston Call (Ï=âˆ’0.30): {call_rho_030:.4f}")

print(f"Heston Call (Ï=âˆ’0.70): {call_rho_070:.4f}")

```



---



## 9. Oil Price Forecasting â€” Probabilistic Graphical Models



**Team:** Joseph Bidias (USA) Â· Keolebogile Seisa (Botswana) Â· Himanshu Rane (Netherlands)  

**Domain:** Bayesian Networks Â· HMM Â· Data Collection  

**Tools:** Python Â· FRED API Â· yfinance Â· pgmpy Â· hmmlearn Â· networkx



### Overview

This is the first phase of a three-part mini-capstone on crude oil price forecasting using Probabilistic Graphical Models. I was responsible for the **macroeconomic/geopolitical data collection** role (Student A). I collected 179 months (Feb 2010 â€“ Dec 2024) of FRED and Yahoo Finance data, cleaned it, and prepared the joint dataset for regime modelling.



### Key Code â€” Macroeconomic Data Collection



```python

import pandas as pd

import numpy as np

import yfinance as yf

from fredapi import Fred



FRED_API_KEY = 'YOUR_KEY_HERE'   # store in .env in production

START_DATE   = '2010-01-01'

END_DATE     = '2024-12-31'

fred = Fred(api_key=FRED_API_KEY)



def collect_macroeconomic_data():

    """Student A: Macroeconomic and geopolitical variables from FRED."""

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

    macro_data = pd.DataFrame()

    for series_id, name in macro_vars.items():

        try:

            series = fred.get_series(series_id,

                                     observation_start=START_DATE,

                                     observation_end=END_DATE)

            macro_data[series_id] = series

            print(f"  âœ“ {name}: {len(series)} observations")

        except Exception as e:

            print(f"  âœ— {series_id}: {e}")

    return macro_data.resample('MS').last()    # monthly frequency



# WTI crude oil price

wti = yf.download('CL=F', start=START_DATE, end=END_DATE)['Close']

wti_monthly = wti.resample('MS').last().rename('DCOILWTICO')

macro_df = collect_macroeconomic_data()

oil_dataset = macro_df.join(wti_monthly, how='inner').dropna()

print(f"\nFinal dataset: {oil_dataset.shape[0]} months Ã— {oil_dataset.shape[1]} features")

```



---



## 10. HMM Regime Detection & Bayesian Network



**Domain:** Hidden Markov Models Â· Bayesian Networks Â· Oil Regimes  

**Tools:** Python Â· hmmlearn Â· pgmpy Â· GaussianHMM Â· DiscreteBayesianNetwork



### Overview

Building on the cleaned dataset, I designed and fitted a **Hidden Markov Model** to detect WTI oil price regimes (low / medium / high volatility), then built a **Bayesian Network** to model the probabilistic relationships between macroeconomic variables and oil price direction.



### Key Code â€” HMM Regime Detection



```python

import numpy as np

from hmmlearn.hmm import GaussianHMM

import pandas as pd



def fit_oil_hmm(returns_series, n_states=3, n_iter=1000):

    """

    Fit a Gaussian HMM with n_states to oil return series.

    Returns fitted model and decoded state sequence.

    """

    X = returns_series.values.reshape(-1, 1)

    model = GaussianHMM(n_components=n_states,

                        covariance_type='full',

                        n_iter=n_iter,

                        random_state=42)

    model.fit(X)

    hidden_states = model.predict(X)



    # Label regimes by mean return (low â†’ medium â†’ high vol)

    means = model.means_.flatten()

    order = np.argsort(np.abs(model.covars_.flatten()))

    regime_labels = {order[0]: 'Low Vol', order[1]: 'Medium Vol', order[2]: 'High Vol'}

    named_states = pd.Series([regime_labels[s] for s in hidden_states],

                              index=returns_series.index)

    print("Transition matrix:\n", np.round(model.transmat_, 3))

    return model, named_states

```



### Key Code â€” Bayesian Network Construction



```python

from pgmpy.models import DiscreteBayesianNetwork

from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BIC

from pgmpy.inference import VariableElimination



def build_bayesian_network(data_discretized):

    """

    Learn Bayesian Network structure from discretized macro data using

    Hill-Climb search with BIC scoring.

    """

    hc = HillClimbSearch(data_discretized)

    best_structure = hc.estimate(scoring_method=BIC(data_discretized),

                                 max_indegree=3,

                                 max_iter=int(1e4))

    bn_model = DiscreteBayesianNetwork(best_structure.edges())

    bn_model.fit(data_discretized, estimator=MaximumLikelihoodEstimator)



    print("Edges learned:", list(bn_model.edges()))

    assert bn_model.check_model(), "Model structure invalid"

    return bn_model



def query_oil_direction(bn_model, evidence_dict):

    """Query P(OilDirection | evidence) via Variable Elimination."""

    infer  = VariableElimination(bn_model)

    result = infer.query(variables=['OilDirection'], evidence=evidence_dict)

    print(result)

    return result

```



---



## 11. Full Model Evaluation & Improvement



**Domain:** Model Evaluation Â· Overfitting Correction Â· Full Train-Val-Test  

**Tools:** Python Â· sklearn metrics Â· pgmpy Â· GaussianHMM



### Overview

The prior phase achieved 100% test accuracy on a single observation â€” textbook overfitting. I redesigned the experiment with a proper 60/20/20 train-validation-test split across all 179 months, resolved the data-to-parameter ratio issue, and improved genuine test accuracy from meaningless (100% on 1 sample) to ~65% on 37 hold-out months.



| Metric | Before | After |

|---|---|---|

| Training observations | 4 | 107 |

| Data-to-parameter ratio | 0.008 | 0.214 |

| Test set size | 1 month | 37 months |

| Test accuracy | 100% (meaningless) | ~65% (genuine) |

| Medium regime detected | 0% | ~30% |



### Key Code â€” Full Evaluation Pipeline



```python

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,

                              confusion_matrix, classification_report)

import numpy as np, pandas as pd

from hmmlearn.hmm import GaussianHMM

from pgmpy.models import DiscreteBayesianNetwork

from pgmpy.estimators import BayesianEstimator

from pgmpy.inference import VariableElimination



def full_evaluation_pipeline(df_returns, feature_cols, target_col,

                              train_ratio=0.60, val_ratio=0.20):

    """

    Proper temporal train-val-test split with HMM + Bayesian Network.

    Avoids look-ahead bias by strictly using past data for fitting.

    """

    n = len(df_returns)

    n_train = int(n * train_ratio)

    n_val   = int(n * val_ratio)



    train = df_returns.iloc[:n_train]

    val   = df_returns.iloc[n_train:n_train + n_val]

    test  = df_returns.iloc[n_train + n_val:]



    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")



    # Fit HMM on train only

    X_train = train[feature_cols].values

    hmm_model = GaussianHMM(n_components=3, covariance_type='full',

                            n_iter=200, random_state=42)

    hmm_model.fit(X_train)



    # Decode regimes on full dataset (train to label, test to evaluate)

    full_states  = hmm_model.predict(df_returns[feature_cols].values)

    test_states  = full_states[n_train + n_val:]

    true_labels  = df_returns[target_col].iloc[n_train + n_val:].values



    acc = accuracy_score(true_labels, test_states)

    print(f"\nTest Accuracy: {acc:.3f}")

    print(classification_report(true_labels, test_states,

                                 target_names=['Low', 'Medium', 'High']))

    return acc, hmm_model

```



---



## 12. Multi-Asset Portfolio Optimization Engine



**Domain:** Mean-Variance Optimization Â· Efficient Frontier  

**Tools:** Python Â· SciPy Â· yfinance Â· NumPy



### Overview

I built a full-featured portfolio optimization engine from scratch. The `PortfolioData` class handles data download and return computation; `PortfolioOptimizer` solves for minimum variance, maximum Sharpe, and efficient frontier portfolios using `scipy.optimize`.



### Key Code â€” Portfolio Data & Optimizer



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

        self.Sigma   = self.returns.cov() * 252

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

        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]

        bounds      = [(0, 1)] * self.n

        x0          = np.ones(self.n) / self.n

        result = minimize(lambda w: -self.portfolio_stats(w)[2],

                          x0, method='SLSQP',

                          bounds=bounds, constraints=constraints)

        return result.x, self.portfolio_stats(result.x)



    def min_variance(self):

        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]

        bounds      = [(0, 1)] * self.n

        result = minimize(lambda w: self.portfolio_stats(w)[1],

                          np.ones(self.n)/self.n, method='SLSQP',

                          bounds=bounds, constraints=constraints)

        return result.x, self.portfolio_stats(result.x)



    def efficient_frontier(self, n_points=100):

        target_returns = np.linspace(self.mu.min(), self.mu.max(), n_points)

        vols = []

        for target in target_returns:

            constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1},

                           {'type': 'eq', 'fun': lambda w: w @ self.mu - target}]

            res = minimize(lambda w: self.portfolio_stats(w)[1],

                           np.ones(self.n)/self.n, method='SLSQP',

                           bounds=[(0,1)]*self.n, constraints=constraints)

            vols.append(res.fun if res.success else np.nan)

        return target_returns, np.array(vols)

```



---



## 13. Mean-Variance: Tech & Healthcare Portfolio



**Domain:** Portfolio Construction Â· Risk Attribution Â· Factor Analysis  

**Assets:** AAPL Â· NVDA Â· TSLA Â· XOM Â· REGN Â· LLY Â· JPM  

**Tools:** Python Â· pandas Â· seaborn Â· scipy Â· yfinance



### Overview

I constructed and analysed a seven-asset portfolio spanning technology, energy, biotech, and financials (2023â€“2025). I computed full summary statistics, built the correlation matrix, identified efficient portfolios, and decomposed risk via variance attribution.



### Key Code â€” Summary Statistics & Correlation



```python

import yfinance as yf, pandas as pd, numpy as np

import matplotlib.pyplot as plt, seaborn as sns



ASSETS = ['AAPL', 'NVDA', 'TSLA', 'XOM', 'REGN', 'LLY', 'JPM']

data   = yf.download(ASSETS, start='2023-01-01', end='2025-06-30',

                     progress=False)['Close']

returns = data.pct_change().dropna()

TRADING_DAYS = 252



stats_df = pd.DataFrame({

    'Annual Return':    returns.mean() * TRADING_DAYS,

    'Annual Vol':       returns.std() * np.sqrt(TRADING_DAYS),

    'Sharpe':           (returns.mean() * TRADING_DAYS - 0.02)

                         / (returns.std() * np.sqrt(TRADING_DAYS)),

    'Skewness':         returns.skew(),

    'Kurtosis':         returns.kurtosis(),

    'Max Drawdown':     (data / data.cummax() - 1).min(),

})



# Correlation heatmap

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(returns.corr(), annot=True, fmt='.2f', cmap='coolwarm',

            center=0, ax=axes[0])

axes[0].set_title('Asset Correlation Matrix', fontweight='bold')

print(stats_df.round(4))

```



---



## 14. ML-Enhanced Portfolio Optimization



**Domain:** Covariance Estimation Â· Clustering Â· Convex Optimisation  

**Tools:** Python Â· Ledoit-Wolf Â· KMeans Â· PCA Â· CVXPY



### Overview

I extended classical mean-variance optimization with three machine learning enhancements: (1) **Ledoit-Wolf shrinkage** for robust covariance estimation, (2) **K-Means clustering** for asset grouping by risk profile, and (3) **CVXPY** for disciplined convex portfolio optimization with custom constraints.



### Key Code â€” Ledoit-Wolf + CVXPY Optimizer



```python

import numpy as np, pandas as pd, cvxpy as cp

from sklearn.covariance import LedoitWolf

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA



def ledoit_wolf_portfolio(returns, rf=0.02, max_weight=0.30):

    """

    Maximum Sharpe portfolio with Ledoit-Wolf covariance shrinkage.

    Solved as a convex QP via CVXPY.

    """

    mu    = returns.mean().values * 252

    lw    = LedoitWolf().fit(returns.values)

    Sigma = lw.covariance_ * 252

    n     = len(mu)



    w = cp.Variable(n)

    ret = mu @ w

    vol = cp.quad_form(w, Sigma)



    # Maximise Sharpe â‰¡ minimise variance for fixed excess return

    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]

    prob = cp.Problem(cp.Minimize(vol), constraints + [ret >= rf + 0.05])

    prob.solve(solver=cp.OSQP, verbose=False)

    return w.value, ret.value, np.sqrt(vol.value)



def cluster_assets(returns, n_clusters=3):

    """K-Means clustering of assets by return/volatility profile."""

    features = np.column_stack([returns.mean() * 252,

                                returns.std() * np.sqrt(252)])

    scaler   = StandardScaler()

    X_scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    labels = km.fit_predict(X_scaled)

    return dict(zip(returns.columns, labels))

```



---



## 15. Regression Trees for Financial Prediction



**Domain:** Decision Trees Â· Grid Search Â· Housing & Financial Data  

**Tools:** Python Â· scikit-learn Â· GridSearchCV



### Overview

I implemented and tuned regression decision trees using the California Housing dataset as a financial prediction analogue. I applied Grid Search cross-validation to find optimal depth and split parameters, visualised the resulting tree structure, and diagnosed overfitting through learning curves.



### Key Code â€” Regression Tree with Grid Search



```python

from sklearn.datasets import fetch_california_housing

from sklearn.tree import DecisionTreeRegressor, plot_tree

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt



# Load data

X, y = fetch_california_housing(return_X_y=True)

feature_names = fetch_california_housing().feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Baseline tree

tree = DecisionTreeRegressor(max_depth=5, random_state=42)

tree.fit(X_train, y_train)

baseline_mse = mean_squared_error(y_test, tree.predict(X_test))

print(f"Baseline Test MSE: {baseline_mse:.4f}")



# Grid Search hyperparameter tuning

param_grid = {

    'max_depth':        [3, 5, 8, 10, None],

    'min_samples_split':[2, 5, 10, 20],

    'min_samples_leaf': [1, 2, 5, 10],

}

gs = GridSearchCV(DecisionTreeRegressor(random_state=42),

                  param_grid, cv=5, scoring='neg_mean_squared_error',

                  n_jobs=-1)

gs.fit(X_train, y_train)

best_tree = gs.best_estimator_

tuned_mse = mean_squared_error(y_test, best_tree.predict(X_test))



print(f"Best params:  {gs.best_params_}")

print(f"Tuned MSE:    {tuned_mse:.4f}  (improvement: {baseline_mse - tuned_mse:.4f})")



# Visualise tree

plt.figure(figsize=(20, 8))

plot_tree(best_tree, filled=True, feature_names=feature_names,

          max_depth=3, fontsize=8)

plt.title("Tuned Regression Tree (max_depth=3 shown)", fontweight='bold')

plt.show()

```



---



## 16. Linear Discriminant Analysis for Market Classification



**Domain:** Dimensionality Reduction Â· Classification Â· Market Regimes  

**Tools:** Python Â· scikit-learn Â· LDA Â· StandardScaler



### Overview

I applied Linear Discriminant Analysis to reduce high-dimensional market feature space to two discriminant functions and classify market regimes into three states. The project demonstrates LDA's superiority over raw PCA for supervised classification tasks.



### Key Code â€” LDA Classification Pipeline



```python

import numpy as np

import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler



# Synthetic three-class market regime dataset

np.random.seed(42)

n = 300

X0 = np.random.multivariate_normal([0, 0],   [[1, 0.5], [0.5, 1]], n//2)

X1 = np.random.multivariate_normal([3, 3],   [[1, 0.5], [0.5, 1]], n//2)

X2 = np.random.multivariate_normal([-3, 3],  [[1,-0.5],[-0.5, 1]], n//2)

X  = np.vstack([X0, X1, X2])

y  = np.array([0]*(n//2) + [1]*(n//2) + [2]*(n//2))



# Train-test split + standardise

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,

                                           random_state=42, stratify=y)

scaler = StandardScaler()

X_tr_s = scaler.fit_transform(X_tr)

X_te_s  = scaler.transform(X_te)



# LDA: reduces to K-1 = 2 discriminant functions

lda   = LinearDiscriminantAnalysis(n_components=2)

X_lda = lda.fit_transform(X_tr_s, y_tr)



# Evaluate

y_pred = lda.predict(X_te_s)

print(f"LDA Accuracy: {accuracy_score(y_te, y_pred):.3f}")

print(classification_report(y_te, y_pred,

      target_names=['Bull', 'Bear', 'Sideways']))

```



---



## 17. Hyperparameter Optimization & Bias-Variance Tradeoff



**Domain:** Model Tuning Â· Random Forest Â· Learning Curves  

**Tools:** Python Â· scikit-learn Â· GridSearchCV Â· RandomizedSearchCV



### Overview

I systematically explored Grid Search, Random Search, and Bayesian Optimization for hyperparameter tuning of a Random Forest classifier. I built learning curve diagnostics to characterise bias-variance tradeoff and cross-validation score distributions.



### Key Code â€” Random Forest Tuning & Learning Curves



```python

import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (train_test_split, GridSearchCV,

                                     learning_curve)



X, y = make_classification(n_samples=1000, n_features=20,

                            n_informative=15, n_redundant=5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,

                                                     random_state=42)



# Grid Search

param_grid = {'n_estimators': [100, 200, 300],

              'max_depth':    [5, 10, None],

              'max_features': ['sqrt', 'log2']}

gs = GridSearchCV(RandomForestClassifier(random_state=42),

                  param_grid, cv=5, scoring='accuracy', n_jobs=-1)

gs.fit(X_train, y_train)

model = gs.best_estimator_

print(f"Best params: {gs.best_params_}")

print(f"CV accuracy: {gs.best_score_:.4f}")



# Learning curves â€” diagnose bias-variance tradeoff

train_sizes, train_sc, val_sc = learning_curve(

    model, X, y, cv=5, n_jobs=-1,

    train_sizes=np.linspace(0.1, 1.0, 10))



train_mean, train_std = train_sc.mean(1), train_sc.std(1)

val_mean,   val_std   = val_sc.mean(1),   val_sc.std(1)



plt.figure(figsize=(10, 5))

plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.1, color='r')

plt.fill_between(train_sizes, val_mean-val_std,   val_mean+val_std,   alpha=0.1, color='g')

plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')

plt.plot(train_sizes, val_mean,   'o-', color='g', label='Validation score')

plt.title('Learning Curve â€” Random Forest'); plt.legend(); plt.grid(True)

plt.xlabel('Training Examples'); plt.ylabel('Score')

plt.tight_layout(); plt.show()

```



---



## 18. Statistical Arbitrage with CNN/LSTM



**Domain:** Equity Time-Series Â· Stationarity Â· Fractional Differentiation  

**Tools:** Python Â· TensorFlow Â· Keras Â· CNN Â· LSTM Â· ADF Test Â· yfinance



### Overview

I built a statistical arbitrage optimization framework using AAPL equity data, implementing fractional differentiation to achieve stationarity without destroying time-series memory. I trained both LSTM and CNN architectures as predictors and compared out-of-sample forecast accuracy.



### Key Code â€” Stationarity & ADF Testing



```python

import numpy as np, pandas as pd, yfinance as yf

from statsmodels.tsa.stattools import adfuller

from scipy import stats



class FinanceTimeSeriesAnalyzer:

    def __init__(self, symbol='AAPL', period='5y'):

        self.symbol = symbol

        self.period = period



    def fetch_data(self):

        ticker = yf.Ticker(self.symbol)

        data   = ticker.history(period=self.period)

        self.price_series = data['Close'].tail(2000).copy()

        return self.price_series



    def analyze_series(self, series, title='Time Series'):

        adf_stat, p_value, _, _, critical, _ = adfuller(series.dropna())

        is_stationary = p_value < 0.05

        print(f"{'âœ“' if is_stationary else 'âœ—'} {title}")

        print(f"  ADF: {adf_stat:.4f}  p={p_value:.4f}  "

              f"Skew={stats.skew(series):.3f}  Kurt={stats.kurtosis(series):.3f}")

        return is_stationary



    def fractional_diff(self, series, d=0.35, threshold=1e-5):

        """

        Fractional differentiation: preserves memory while achieving stationarity.

        LÃ³pez de Prado (2018) method â€” minimal memory loss.

        """

        w   = [1.0]

        for k in range(1, len(series)):

            w.append(-w[-1] * (d - k + 1) / k)

            if abs(w[-1]) < threshold:

                break

        w = np.array(w[::-1])

        return pd.Series(

            [np.dot(w, series.iloc[i-len(w)+1:i+1]) for i in range(len(w)-1, len(series))],

            index=series.index[len(w)-1:]

        )

```



### Key Code â€” LSTM Predictor



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

        X.append(data[i-seq_len:i])

        y.append(data[i])

    return np.array(X), np.array(y)



scaler = MinMaxScaler()

scaled = scaler.fit_transform(prices.values.reshape(-1, 1))

X, y   = create_sequences(scaled, seq_len=60)

split  = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]

y_train, y_test = y[:split], y[split:]



model = build_lstm_model((60, 1))

model.fit(X_train, y_train, epochs=50, batch_size=32,

          validation_split=0.1,

          callbacks=[tf.keras.callbacks.EarlyStopping(patience=5,

                                                      restore_best_weights=True)])

```



---



## 19. Multi-Asset Portfolio Allocation with LSTM



**Domain:** Multi-Output Deep Learning Â· Cross-Asset LSTM Â· Portfolio Signals  

**Assets:** SPY Â· TLT Â· SHY Â· GLD Â· DBO  

**Tools:** Python Â· TensorFlow Â· Keras Â· yfinance



### Overview

I designed and trained individual LSTM models for each of five asset classes (equity, bonds, cash-equivalent, gold, oil), then built a **multi-output LSTM architecture** to jointly predict allocation signals. Backtested three trading strategies derived from the model's forecasts.



### Key Code â€” Multi-Output LSTM Architecture



```python

import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate



def build_multi_output_lstm(seq_len, n_features, n_assets):

    """

    Shared-encoder multi-output LSTM.

    Single input sequence â†’ n_assets return forecasts.

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



etfs = {'SPY': 'Equity', 'TLT': 'Fixed Income',

        'SHY': 'Cash-like', 'GLD': 'Gold', 'DBO': 'Oil'}



class ExecutableMultiAssetAnalyzer:

    def __init__(self):

        self.etfs    = etfs

        self.results = {}



    def fetch_and_analyze_data(self, start='2015-01-01', end='2024-01-01'):

        import yfinance as yf

        data    = yf.download(list(self.etfs.keys()), start=start, end=end,

                              progress=False)['Close']

        returns = data.pct_change().dropna()

        self.data    = data

        self.returns = returns

        return returns



    def implement_trading_strategies(self, predictions_df, returns_df):

        """

        Strategy 1: Long-only momentum signal from LSTM

        Strategy 2: Dollar-neutral long/short

        Strategy 3: Risk-parity weighted by predicted vol

        """

        signals = (predictions_df > 0).astype(int)    # long when forecast > 0

        strat1  = (signals * returns_df).mean(axis=1)

        strat2  = (signals - 0.5) * 2 * returns_df.mean(axis=1)

        vol_est = returns_df.rolling(21).std()

        rp_wts  = 1 / vol_est.div(vol_est.sum(axis=1), axis=0)

        strat3  = (rp_wts * returns_df).sum(axis=1)

        return strat1, strat2, strat3

```



---



## 20. Data Leakage in Walk-Forward Backtesting



**Domain:** Walk-Forward Validation Â· Data Leakage Â· LSTM Â· MLP  

**Tools:** Python Â· TensorFlow Â· scikit-learn Â· Keras Â· AAPL equity



### Overview

I systematically investigated the impact of **data leakage** on deep learning model performance in financial backtesting. Three progressively rigorous experimental setups were compared: (1) single train/test split with intentional leakage, (2) walk-forward with leakage, and (3) walk-forward with leakage controls â€” demonstrating the gap between inflated in-sample metrics and realistic out-of-sample performance.



### Key Code â€” Walk-Forward with Leakage Detection



```python

import numpy as np, pandas as pd

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error, r2_score



class GWP3ComprehensiveAnalysis:

    def __init__(self, symbol='AAPL', seq_len=60, max_obs=2000):

        self.symbol  = symbol

        self.seq_len = seq_len

        self.max_obs = max_obs



    # â”€â”€ STEP 1: Single split with intentional leakage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def step1_single_split_with_leakage(self, prices):

        """

        Leakage source: scaler fitted on FULL dataset before split.

        Represents a common but incorrect practice.

        """

        scaler = MinMaxScaler()

        scaled = scaler.fit_transform(prices.values.reshape(-1, 1))  # â† LEAKAGE



        X, y = self._make_sequences(scaled, self.seq_len)

        split = int(0.8 * len(X))

        return (X[:split], y[:split]), (X[split:], y[split:]), scaler



    # â”€â”€ STEP 3: Walk-forward with leakage controls â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def step3_walk_forward_no_leakage(self, prices, n_splits=5):

        """

        Corrected approach: scaler re-fitted on TRAINING window only at each fold.

        No information from the future leaks into past observations.

        """

        results = []

        fold_size = len(prices) // n_splits



        for fold in range(1, n_splits):

            train_end   = fold * fold_size

            train_prices = prices.iloc[:train_end]

            test_prices  = prices.iloc[train_end: train_end + fold_size]



            scaler = MinMaxScaler()                      # â† fit on train only

            scaled_train = scaler.fit_transform(train_prices.values.reshape(-1, 1))

            scaled_test  = scaler.transform(test_prices.values.reshape(-1, 1))



            X_tr, y_tr = self._make_sequences(scaled_train, self.seq_len)

            X_te, y_te = self._make_sequences(scaled_test,  self.seq_len)



            model = self._build_lstm((self.seq_len, 1))

            model.fit(X_tr, y_tr, epochs=20, batch_size=32, verbose=0,

                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])



            preds   = model.predict(X_te, verbose=0)

            mse     = mean_squared_error(y_te, preds)

            r2      = r2_score(y_te, preds)

            results.append({'fold': fold, 'mse': mse, 'r2': r2,

                            'train_size': len(X_tr), 'test_size': len(X_te)})

            print(f"Fold {fold}: MSE={mse:.6f}  RÂ²={r2:.4f}")

        return pd.DataFrame(results)



    def _make_sequences(self, data, seq_len):

        X, y = [], []

        for i in range(seq_len, len(data)):

            X.append(data[i-seq_len:i])

            y.append(data[i])

        return np.array(X), np.array(y)

```



---



## 21. Outlier Sensitivity in Regression



**Domain:** OLS Regression Â· Cook's Distance Â· Influential Points  

**Assets:** SPY vs NVDA weekly returns (2014â€“2024)  

**Tools:** Python Â· statsmodels Â· matplotlib



### Overview

I investigated the sensitivity of OLS regression to influential observations using ten years of weekly SPY and NVDA returns. Cook's distance was used to identify the five most influential data points, and I compared regression models with and without those points to quantify their distorting effect.



### Key Code â€” OLS, Cook's Distance & Influence Diagnostics



```python

import pandas as pd, datetime, matplotlib.pyplot as plt

import statsmodels.api as sm

import statsmodels.formula.api as smf

from statsmodels.graphics.regressionplots import abline_plot

import yfinance as yf



# Download weekly returns 2014-2024

end   = datetime.date(2024, 1, 1)

start = datetime.date(2014, 1, 1)

prices = pd.DataFrame(

    yf.download(['SPY', 'NVDA'], start=start, end=end, interval='1wk')['Close']

)

prices.index = prices.index.date

data = prices.pct_change().dropna()



# Full OLS regression

result_full = smf.ols("SPY ~ NVDA", data=data).fit()

print(result_full.summary())



# Cook's distance â€” detect influential points

influence = result_full.get_influence()

inf_frame = influence.summary_frame().sort_values('cooks_d', ascending=False)

top5_idx  = inf_frame.head(5).index

print("\nTop 5 most influential observations:")

print(data.loc[top5_idx])



# Regression without influential points

data_clean   = data.drop(index=top5_idx)

result_clean = smf.ols("SPY ~ NVDA", data=data_clean).fit()



print(f"\nCoefficient (NVDA) with outliers:    {result_full.params['NVDA']:.6f}")

print(f"Coefficient (NVDA) without outliers: {result_clean.params['NVDA']:.6f}")

print(f"RÂ² with outliers:    {result_full.rsquared:.4f}")

print(f"RÂ² without outliers: {result_clean.rsquared:.4f}")



# Influence plot

sm.graphics.influence_plot(result_full, criterion='cooks', alpha=0.01)

plt.tight_layout(); plt.show()

```



---



## 22. Modeling Randomness & White Noise



**Domain:** White Noise Â· ACF Â· Ljung-Box Test Â· Stochastic Processes  

**Tools:** Python Â· statsmodels Â· NumPy Â· matplotlib



### Overview

I implemented a full white noise analysis pipeline: simulation, ACF visualization, and formal Ljung-Box hypothesis testing. The project underpins the fundamental assumption of residual independence in time-series models used throughout the curriculum.



### Key Code â€” White Noise Analysis



```python

import numpy as np, matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.stats.diagnostic import acorr_ljungbox



np.random.seed(42)

white_noise = np.random.normal(0, 1, 100)



# Plot time series

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(white_noise, linewidth=0.8, color='steelblue')

axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)

axes[0].set_title('Simulated White Noise'); axes[0].set_xlabel('Time')

axes[0].grid(True, alpha=0.3)



# ACF

plot_acf(white_noise, lags=20, ax=axes[1], title='ACF of White Noise')

plt.tight_layout(); plt.show()



# Ljung-Box test: H0 = no autocorrelation up to lag L

lb_result = acorr_ljungbox(white_noise, lags=[5, 10, 15, 20], return_df=True)

print(lb_result.to_string())



for lag, row in lb_result.iterrows():

    decision = "Fail to reject H0 (white noise)" if row['lb_pvalue'] > 0.05 \

               else "Reject H0 (autocorrelation present)"

    print(f"Lag {lag:2d}: p={row['lb_pvalue']:.4f}  â†’ {decision}")

```



---



## 23. Cointegration & VECM



**Domain:** Unit Roots Â· Cointegration Â· VECM Â· Long-Run Equilibrium  

**Assets:** AAPL vs NASDAQ (2020â€“2023)  

**Tools:** Python Â· statsmodels Â· Johansen test Â· VECM



### Overview

I tested for unit roots with the ADF test, established a cointegrating relationship between Apple and the NASDAQ index using the Johansen test, and estimated a Vector Error Correction Model (VECM) capturing the long-run equilibrium and short-run dynamics.



### Key Code â€” ADF, Johansen & VECM



```python

import yfinance as yf, numpy as np, pandas as pd

import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM



# Download data

apple  = yf.download('AAPL',  start='2020-01-01', end='2023-12-31', progress=False)['Close']

nasdaq = yf.download('^IXIC', start='2020-01-01', end='2023-12-31', progress=False)['Close']

data   = pd.concat([np.log(apple), np.log(nasdaq)], axis=1).dropna()

data.columns = ['Log_AAPL', 'Log_NASDAQ']



def adf_test(series, name):

    stat, p, _, _, crit, _ = adfuller(series)

    print(f"{name}: ADF={stat:.4f}  p={p:.4f}  "

          f"{'Stationary' if p < 0.05 else 'Non-Stationary'}")

    return p



# Level tests: expect non-stationary

adf_test(data['Log_AAPL'],   'Log AAPL  (level)')

adf_test(data['Log_NASDAQ'], 'Log NASDAQ (level)')



# First-difference tests: expect stationary

adf_test(data['Log_AAPL'].diff().dropna(),   'Î”AAPL')

adf_test(data['Log_NASDAQ'].diff().dropna(), 'Î”NASDAQ')



# Johansen cointegration test

johansen = coint_johansen(data, det_order=0, k_ar_diff=1)

print("\nJohansen Trace Statistics:")

print(johansen.lr1)          # trace statistics

print("Critical Values (90%, 95%, 99%):")

print(johansen.cvt)          # critical values



# VECM

vecm    = VECM(data, k_ar_diff=1, coint_rank=1, deterministic='ci')

vecm_res = vecm.fit()

print(vecm_res.summary())

```



---



## 24. Lending EDA: Stock Returns, Housing & Rates



**Domain:** Exploratory Data Analysis Â· Multi-Source Financial Data  

**Tools:** Python Â· FRED API Â· yfinance Â· pandas Â· seaborn



### Overview

I conducted a comprehensive EDA in support of a lending team's analysis, collecting and aligning four distinct financial data streams: AAPL stock returns (Yahoo Finance), S&P/Case-Shiller home price index, 30-year fixed mortgage rates, and 10-year Treasury constant maturity rates (all via FRED). The analysis uncovered volatility patterns, rate-price correlations, and housing cycle dynamics.



### Key Code â€” Multi-Source Data Collection & EDA



```python

import pandas as pd, yfinance as yf

import matplotlib.pyplot as plt, seaborn as sns

from fredapi import Fred



FRED_KEY = 'YOUR_KEY_HERE'    # use environment variable in production

fred     = Fred(api_key=FRED_KEY)



# Collect all four data streams

apple_stock    = yf.download("AAPL", start="2015-01-01", end="2024-01-01",

                              progress=False)

apple_stock['Returns'] = apple_stock['Close'].pct_change()



case_shiller   = fred.get_series('CSUSHPINSA', observation_start="2015-01-01")

mortgage_rates = fred.get_series('MORTGAGE30US', observation_start="2015-01-01")

treasury_rates = fred.get_series('DGS10',        observation_start="2015-01-01")



# Align to monthly frequency

monthly_apple  = apple_stock['Returns'].resample('MS').mean()

df_combined    = pd.DataFrame({

    'Apple_Return':  monthly_apple,

    'HPI':           case_shiller.resample('MS').last(),

    'Mortgage_30yr': mortgage_rates.resample('MS').last(),

    'Treasury_10yr': treasury_rates.astype(float).resample('MS').last()

}).dropna()



# Correlation heatmap

plt.figure(figsize=(8, 6))

sns.heatmap(df_combined.corr(), annot=True, fmt='.3f',

            cmap='coolwarm', center=0, linewidths=0.5)

plt.title('Cross-Asset Correlation: Returns, Housing & Rates', fontweight='bold')

plt.tight_layout(); plt.show()



# Rolling volatility of Apple returns

rolling_vol = apple_stock['Returns'].rolling(21).std() * np.sqrt(252)

rolling_vol.plot(figsize=(12, 4), title='AAPL Annualised Rolling Volatility (21-day)',

                 color='steelblue')

plt.ylabel('Annualised Volatility'); plt.grid(alpha=0.3); plt.show()

```



---



## 25. Financial Risk Analysis in Lending Scenarios



**Domain:** Credit Risk Â· Equity Risk Â· Liquidity Risk Â· Illiquid Securities  

**Tools:** Python Â· FRED API Â· yfinance Â· NumPy Â· matplotlib



### Overview

I analysed financial risks across four lending scenarios: (1) unsecured credit cards, (2) business construction loans, (3) publicly traded equity (S&P 500), and (4) illiquid private equity. Each scenario was modelled with real data and appropriate risk metrics.



### Key Code â€” Multi-Scenario Risk Analysis



```python

import pandas as pd, numpy as np

import matplotlib.pyplot as plt

import yfinance as yf

from fredapi import Fred



fred = Fred(api_key='YOUR_KEY_HERE')



# 1. Unsecured credit: credit card delinquency rate

cc_delinquency     = fred.get_series('DRCCLACBS')



# 2. Business construction: interest rate vs spending

construction_spend = fred.get_series('TTLCONS')

interest_rates     = fred.get_series('FEDFUNDS')



# 3. Publicly traded equity: S&P 500 returns

sp500      = yf.download('^GSPC', start='2010-01-01', end='2023-12-31',

                          progress=False)['Close']

sp_returns = sp500.pct_change().dropna()



# 4. Illiquid private equity simulation (quarterly, log-normal returns)

np.random.seed(42)

dates      = pd.date_range('2010-01-01', '2023-12-31', freq='QS')

pe_returns = pd.Series(np.random.lognormal(0.03, 0.15, len(dates)) - 1,

                        index=dates)



# Risk metrics comparison

def risk_metrics(returns, name):

    var_95 = np.percentile(returns, 5)

    cvar   = returns[returns <= var_95].mean()

    sharpe = returns.mean() / returns.std() * np.sqrt(252 if len(returns) > 200 else 4)

    print(f"{name:30s} | VaR(95%)={var_95:.4f} | CVaR={cvar:.4f} | Sharpe={sharpe:.3f}")



risk_metrics(sp_returns.values,     'S&P 500 (Publicly Traded Equity)')

risk_metrics(pe_returns.values,     'Private Equity (Illiquid)')

```



---



## 26. Mortgage Amortization & Securities Lending



**Domain:** Fixed Income Â· Amortization Â· Securities Lending  

**Tools:** Python Â· NumPy Â· pandas Â· yfinance



### Overview

I implemented two financial tools for a lending team: (1) a floating-rate mortgage amortization schedule with quarterly rate resets, and (2) a stock-lending profitability analyser using AAPL historical data. Both tools simulate cashflows and remaining balances over multi-year horizons.



### Key Code â€” Floating-Rate Mortgage Engine



```python

import numpy as np, pandas as pd



def calculate_monthly_payment(principal, annual_rate, years):

    """Standard annuity formula for fixed-rate period payment."""

    r = annual_rate / 12

    n = years * 12

    if r == 0:

        return principal / n

    return principal * r / (1 - (1 + r)**(-n))



def floating_rate_amortization(principal, rate_schedule, years_per_rate):

    """

    Floating-rate mortgage: amortization across multiple rate periods.

    rate_schedule  : list of annual rates e.g. [0.03, 0.035, 0.04]

    years_per_rate : matching list of years per period

    Returns a DataFrame with Year, Rate, Payment, Interest, Principal, Balance

    """

    rows      = []

    balance   = principal

    total_yr  = 0



    for rate, years in zip(rate_schedule, years_per_rate):

        monthly_pmt = calculate_monthly_payment(balance, rate, years)

        for yr in range(1, years + 1):

            interest       = balance * rate

            principal_paid = monthly_pmt * 12 - interest

            balance       -= principal_paid

            total_yr      += 1

            rows.append({'Year':             total_yr,

                          'Interest Rate':   f"{rate:.1%}",

                          'Monthly Payment': round(monthly_pmt, 2),

                          'Interest Paid':   round(interest, 2),

                          'Principal Paid':  round(principal_paid, 2),

                          'Remaining Balance': round(max(balance, 0), 2)})

    return pd.DataFrame(rows)



# $300k, 30-year floating: 3% for 5yr â†’ 3.5% for 10yr â†’ 4% for 10yr â†’ 4.5% for 5yr

schedule = floating_rate_amortization(

    principal      = 300_000,

    rate_schedule  = [0.030, 0.035, 0.040, 0.045],

    years_per_rate = [5, 10, 10, 5]

)

print(schedule.head(10).to_string(index=False))



# Securities lending: AAPL stock collateral analysis

import yfinance as yf, matplotlib.pyplot as plt



aapl = yf.download("AAPL", start="2010-01-01", end="2023-01-01",

                   progress=False)['Close']

aapl_returns = aapl.pct_change().dropna()



plt.figure(figsize=(12, 5))

plt.subplot(2, 1, 1); aapl.plot(label='AAPL Price', color='steelblue')

plt.title('AAPL â€” Securities Lending Collateral Analysis'); plt.legend()

plt.subplot(2, 1, 2); aapl_returns.plot(color='darkorange', linewidth=0.7)

plt.axhline(0, color='black', linewidth=0.5)

plt.title('Daily Returns'); plt.tight_layout(); plt.show()

```



---



## Skills & Technology Stack



| Category | Technologies |

|---|---|

| **Languages** | Python 3.x |

| **Data** | pandas Â· NumPy Â· FRED API Â· yfinance Â· fredapi |

| **Statistics & Econometrics** | statsmodels Â· scipy Â· arch Â· GARCH Â· VECM Â· ADF Â· Johansen |

| **Machine Learning** | scikit-learn Â· GridSearchCV Â· LDA Â· Random Forest Â· Decision Trees Â· PCA Â· KMeans Â· Ledoit-Wolf |

| **Deep Learning** | TensorFlow Â· Keras Â· LSTM Â· CNN Â· MLP Â· EarlyStopping |

| **Derivatives & Pricing** | Black-Scholes Â· Binomial Tree Â· Heston Â· Bates Â· Merton Jump Diffusion Â· CIR Â· Carr-Madan Â· Lewis (2001) |

| **Risk Management** | VaR Â· CVaR Â· Kupiec Test Â· DCC-GARCH Â· HAR-RV Â· Bootstrap |

| **Portfolio Optimisation** | scipy.optimize Â· CVXPY Â· Efficient Frontier Â· Sharpe Maximisation Â· Risk Parity |

| **Probabilistic Models** | hmmlearn Â· pgmpy Â· HMM Â· Bayesian Networks Â· VariableElimination |

| **Stochastic Processes** | GBM Â· Markov-Switching Â· CIR Â· Euler-Maruyama Â· Fractional Differentiation |

| **Visualisation** | matplotlib Â· seaborn Â· plotly |

| **Version Control** | Git Â· GitHub |



---



*Portfolio compiled from 26 projects across 9 quantitative finance domains.*

