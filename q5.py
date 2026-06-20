"""
Task 5: Dynamic Quoting Under Inventory Pressure
=================================================
Strategy: Signal-adjusted Avellaneda-Stoikov market-making framework.
"""

import numpy as np
import pandas as pd
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Tuple

import q3  # Bridges Task 3 predictions for live alpha

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Constants & hyper-parameters (Strictly adhering to Section 3.6.1 constraints)
# ─────────────────────────────────────────────────────────────────────────────
C_MIN       = 0.5      # c_min = 0.5 (from prompt)
DELTA_MAX   = 0.0050   # 50 bps maximum half-spread (from prompt)
N_VOL       = 20       # Look-back window for realized vol (Equation 10)

# Avellaneda-Stoikov style parameters
K_BASE      = 1.5      # inventory risk-aversion 
K_ETA       = 3.0      # urgency ramp power at end-of-day
SIGMA_FLOOR = 5e-5     # vol floor to avoid division-by-zero

# Spread composition weights
W_VOL       = 1.2      # Base vol width
W_INV       = 0.8      # Inventory skew coefficient
W_ADV       = 0.6      # Adversity premium

# ─────────────────────────────────────────────────────────────────────────────
# Core Quoting Function (Required Signature)
# ─────────────────────────────────────────────────────────────────────────────
def quote(inventory: float, sigma: float, alpha: float, eta: float) -> tuple[float, float]:
    """Returns (delta_bid, delta_ask) dynamically adjusted by state."""
    sigma = max(sigma, SIGMA_FLOOR)
    
    # 1. Base Volatility Spread
    delta_base = W_VOL * sigma
    
    # 2. Adversity Premium (Protects against toxic flow)
    adv_premium = W_ADV * alpha * sigma
    
    # 3. Inventory Skew (Shifts prices to dump excess inventory)
    inv_skew = W_INV * K_BASE * inventory * sigma
    
    # 4. End-of-Day Urgency (Quadratic scaling to avoid Penalty_D)
    urgency = 1.0 + K_ETA * (eta ** 2)
    inv_skew_scaled = inv_skew * urgency

    delta_bid = delta_base + adv_premium + inv_skew_scaled
    delta_ask = delta_base + adv_premium - inv_skew_scaled

    # Enforce Bounds (Equation 14: c_min * sigma <= delta <= delta_max)
    sigma_min = C_MIN * sigma
    delta_bid = float(np.clip(delta_bid, sigma_min, DELTA_MAX))
    delta_ask = float(np.clip(delta_ask, sigma_min, DELTA_MAX))

    return delta_bid, delta_ask


# ─────────────────────────────────────────────────────────────────────────────
# Validation Engine (Handles Hidden Regimes and CSV Backtesting)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MarketParams:
    lam: float   
    gamma: float 
    phi: float   
    adv_mu: float   
    adv_sig: float  

def realized_vol(returns: deque) -> float:
    if len(returns) < 2: return SIGMA_FLOOR
    return float(np.sqrt(np.mean(np.array(returns) ** 2)))

def _synthetic_regime_simulation(regimes: List[MarketParams], n_days=10, n_trades=300) -> dict:
    """Tests the quote() function against hidden, shifting parameters."""
    rng = np.random.default_rng(42)
    all_daily_pnl = []

    for params in regimes:
        for d in range(n_days):
            mid = 100.0 + rng.normal(0, 0.5)
            returns_q: deque = deque(maxlen=N_VOL)
            inventory = 0.0          
            day_pnl = 0.0
            vol_acc = []

            for t in range(n_trades):
                ret = rng.normal(0, 2e-4)
                mid = max(mid * (1 + ret), 0.01)
                returns_q.append(ret)

                sigma = realized_vol(returns_q)
                vol_acc.append(sigma)
                
                # Synthetic Adversity
                alpha = float(np.clip(rng.normal(params.adv_mu, params.adv_sig), 0, 1))
                eta = t / n_trades

                db, da = quote(inventory / mid, sigma, alpha, eta)
                
                # Fill simulation (Equation 15)
                side = rng.choice([-1, 1])   
                delta_side = da if side == 1 else db
                p_fill = params.lam * np.exp(-params.gamma * delta_side / max(sigma, SIGMA_FLOOR))
                
                if rng.random() < p_fill:
                    day_pnl += (mid * delta_side)
                    inventory -= side  

            # Penalty Calculation (Equation 16)
            avg_vol = np.mean(vol_acc) if vol_acc else SIGMA_FLOOR
            penalty = params.phi * (inventory ** 2) * avg_vol
            day_pnl -= penalty
            all_daily_pnl.append(day_pnl)

    total_pnl = float(np.sum(all_daily_pnl))
    vol_portfolio = float(np.std(all_daily_pnl))
    score = total_pnl / max(vol_portfolio, 1.0)
    return {"total_pnl": total_pnl, "sharpe_score": score}

def _historical_csv_backtest(filepath='trade_data.csv', tau=30):
    """Feeds actual historical data and live Task 3 predictions into quote()."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"{filepath} not found. Skipping historical test.")
        return

    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['time'], format='%d-%m-%Y %H:%M:%S')
    df = df.sort_values('datetime').reset_index(drop=True)
    unique_days = df['Date'].unique()
    
    total_pnl = 0.0
    daily_pnls = []
    
    for day in unique_days:
        day_df = df[df['Date'] == day].reset_index(drop=True)
        n_trades = len(day_df)
        inventory = 0.0
        day_pnl = 0.0
        returns_q = deque(maxlen=N_VOL)
        vol_acc = []
        
        for idx, row in day_df.iterrows():
            mid = row['M0']
            if idx > 0:
                returns_q.append((mid - day_df.iloc[idx-1]['M0']) / day_df.iloc[idx-1]['M0'])
            sigma = realized_vol(returns_q)
            vol_acc.append(sigma)
            
            # LIVE HANDOFF: Pass actual row to Task 3 to get Alpha
            alpha = q3.predict_adversity(
                tau=tau, Name=row['Name'], Side=row['Side'], Volume=row['Volume'],
                Trade_Price=row['Trade Price'], M0=row['M0'], Spread=row['Spread'],
                Date=row['Date'], time=row['time']
            )
            
            eta = min(idx / n_trades, 1.0)
            db, da = quote(inventory, sigma, alpha, eta)
            
            # Simulated Fill using proxy parameters for backtesting
            historical_side = row['Side']
            delta_side = da if historical_side == 1 else db
            p_fill = 0.8 * np.exp(-1.5 * delta_side / max(sigma, SIGMA_FLOOR))
            
            if np.random.random() < p_fill:
                day_pnl += (mid * delta_side)
                inventory -= historical_side
                
        avg_vol = np.mean(vol_acc) if vol_acc else SIGMA_FLOOR
        penalty = 0.05 * (inventory ** 2) * avg_vol
        day_pnl -= penalty
        daily_pnls.append(day_pnl)
        total_pnl += day_pnl
        print(f"Date: {day} | End Inv: {inventory:>5.1f} | Day PnL: ${day_pnl:>8.2f}")
        
    vol_portfolio = float(np.std(daily_pnls))
    sharpe = total_pnl / max(vol_portfolio, 1.0)
    print(f"-> CSV Historical Sharpe Score: {sharpe:.3f}")

def validate_quote() -> None:
    """
    Required Function: Runs validation of the quoting strategy.
    Tests against both hidden synthetic regimes and actual CSV data.
    """
    print("=" * 65)
    print("PART 1: SYNTHETIC REGIME VALIDATION (HIDDEN PARAMETERS)")
    print("=" * 65)
    regimes = [
        MarketParams(lam=0.7, gamma=1.5, phi=0.05, adv_mu=0.3, adv_sig=0.15),
        MarketParams(lam=0.5, gamma=2.5, phi=0.10, adv_mu=0.6, adv_sig=0.20),
        MarketParams(lam=0.9, gamma=0.8, phi=0.02, adv_mu=0.2, adv_sig=0.10),
    ]
    res = _synthetic_regime_simulation(regimes)
    print(f"Total PnL across shifts : ${res['total_pnl']:.2f}")
    print(f"Sharpe Score (Eq 18)    : {res['sharpe_score']:.3f}")
    
    print("\n" + "=" * 65)
    print("PART 2: HISTORICAL CSV BACKTEST (INTEGRATING TASK 3)")
    print("=" * 65)
    _historical_csv_backtest('trade_data.csv')


if __name__ == "__main__":
    validate_quote()