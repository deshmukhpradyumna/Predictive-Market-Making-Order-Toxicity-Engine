Predictive Market Making & Order Toxicity Engine

This repository implements a full-stack quantitative market-making algorithm designed to mitigate adverse selection in high-frequency trading environments.

By combining market microstructure analysis with a machine learning-driven order toxicity classifier, this engine dynamically adjusts bid-ask spreads in real-time to protect inventory from toxic order flow while maximizing capture on benign trades.

📂 Dataset Description

The pipeline relies on high-frequency, tick-level trade data. The expected input file is trade_data.csv, which must contain the following columns:

Date & time: The exact timestamp of the execution.

Name: Client identifier (e.g., Client A, Client B). Crucial for profiling flow toxicity.

Side: The direction of the client's trade. 1 indicates the client bought (Market Maker sold), -1 indicates the client sold (Market Maker bought).

Volume: The size of the executed order.

Trade Price: The actual price at which the transaction occurred.

M0: The prevailing mid-price of the order book at the exact time of execution ($t=0$).

Spread: The bid-ask spread at execution.

M5 to M30: The future mid-prices at micro-horizons $\tau \in \{5, 10, 15, 20, 25, 30\}$ seconds post-trade. These are used strictly as target variables for training and evaluation, never as predictive features.

📊 Core Architecture & Mathematical Breakdown

The project is structured as a chronological algorithmic trading pipeline. Here is the mathematical functionality executed by each script:

Phase 1: Microstructure Profiling (q1.py, q2.py)

These scripts establish the empirical baseline of how toxic each client's flow is before applying machine learning.

q1.py (Adversity Profile): Calculates the historical probability that a trade will move against the market maker. A trade is defined as mathematically "adverse" if the mark-to-market PnL at time $\tau$ is negative:

P(Adverse) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I} \Big(Side_i \times (M_{i,\tau} - P_{trade, i}) < 0 \Big)


q2.py (Profitability & Spread): Calculates the Expected PnL per client and computes the minimum theoretical half-spread ($\delta^*$) required to break even against their specific informed flow. It calculates the volume-weighted required delta:

\delta^*_c = \max \left( 0, \frac{\sum_{i \in c} \Big( -Side_i \times (\mathbb{E}[M_\tau]_i - M_{0,i}) \times Volume_i \Big)}{\sum_{i \in c} Volume_i} \right)


Phase 2: Predictive Toxicity Modeling (q3.py)

This script trains a microsecond-resolution HistGradientBoostingClassifier to forecast short-term directional price adversity.

The Math: It models the probability $\alpha = P(Y=1 | X)$, where $Y=1$ represents an adverse trade.

Features ($X$): Engineered from $t \le 0$ data, including Volume, Spread, Hour, Minute, Trade_Aggressiveness (defined as $Side \times (P_{trade} - M_0)$), and Name.

Leakage Prevention: It utilizes native categorical handling for client names and strict chronological time-series cross-validation (60/20/20 train-val-test split) to completely prevent look-ahead bias and target leakage.

Phase 3: Threshold Optimization (q4.py)

Standard ML metrics (like a 0.50 probability threshold) are suboptimal for trading. This script transitions the ML model into actionable trading logic.

The Math: It dynamically calculates a client-specific externalization threshold ($\theta^*$) on the validation fold to maximize simulated out-of-sample PnL. If the model's predicted toxicity $\alpha$ exceeds $\theta^*$, the algorithm will defensively reject or widen quotes for that client.

\theta^*_c = \arg\max_{\theta} \sum_{i \in c, \ \alpha_i \le \theta} \Big( Side_i \times Volume_i \times (M_{i,\tau} - P_{trade, i}) \Big)


Phase 4: Dynamic Execution Engine (q5.py)

A modified Avellaneda-Stoikov market-making framework. It ingests the live alpha signal ($\alpha$) from Phase 2 and dynamically calculates optimal bid/ask quotes based on three mathematical forces:

Base Volatility: Proportional to recent realized volatility $\sigma$.

Adversity Premium: Widens the spread defensively when the ML model predicts a high probability of toxicity ($\alpha$).

Inventory Skew: Shifts the midpoint to dump excess inventory $I$ before the end of the day, scaled by an urgency parameter $\eta$.

The Quoting Equations:

\delta_{bid} = (W_{vol} \cdot \sigma) + (W_{adv} \cdot \alpha \cdot \sigma) + Skew


\delta_{ask} = (W_{vol} \cdot \sigma) + (W_{adv} \cdot \alpha \cdot \sigma) - Skew


Where:

Skew = $W_{inv} \cdot K_{base} \cdot I \cdot \sigma \cdot (1 + K_{\eta}\eta^2)$

Hard limits are enforced ensuring $c_{min} \cdot \sigma \le \delta \le \delta_{max}$.

🚀 Installation & Usage

Prerequisites:

Python 3.9+

Required libraries: pandas, numpy, scikit-learn, matplotlib, joblib

Setup:

Clone the repository:

git clone [https://github.com/YourUsername/your-repo-name.git](https://github.com/YourUsername/your-repo-name.git)
cd your-repo-name


Install dependencies:

pip install -r requirements.txt


Place your raw trade_data.csv file in the root directory of the project.

Step-by-Step Execution:

You can run the pipeline sequentially to see the data evolve, or jump straight to the backtest.

Generate Baselines:

python q1.py  # Generates task1_results.csv (Adversity baselines)
python q2.py  # Generates task2_results.csv (Expected PnL & required spreads)


Train the Toxicity Model:

python q3.py  


Trains the gradient boosting classifiers across all time horizons, evaluates log-loss/precision, and saves .pkl model weights to disk.

Optimize Trading Thresholds:

python q4.py  


Generates task4_results.csv and a pnl_vs_theta.png chart showing the exact inflection point where rejecting toxic trades maximizes portfolio PnL.

Run the Live Market-Making Backtest:

python q5.py


This is the capstone script. It runs a synthetic regime validation (hidden parameters) and a historical CSV backtest, feeding live tick data into the ML model and applying the dynamic quoting math to output a final Sharpe Score.

📈 Key Results

Signal Generation: Extracted a ~7% predictive edge over baseline market toxicity at the $\tau=30s$ horizon using strictly point-in-time features.

Risk Management: Successfully shifts the portfolio Sharpe ratio positive during hidden-regime synthetic simulations by appropriately penalizing latency-arbitrage flow while tightening spreads for retail-like flow.

🛠️ Tech Stack

Language: Python

Data Manipulation: Pandas, NumPy

Machine Learning: Scikit-Learn (HistGradientBoosting)

Evaluation: Strictly chronological Time-Series Split, Custom Economic Cost Functions.
