import pandas as pd
from typing import List

# Assuming 'df' is your loaded pandas DataFrame containing the trade data.
df = pd.read_csv('trade_data.csv')

def expected_pnl(client: str, tau: List[int]) -> dict:
    """
    Parameters:
    client: Client identifier
    tau: List of horizons e.g. [5, 10, 15, 20, 25, 30]
    
    Returns:
    Dictionary with keys: 'per_horizon' and 'aggregate'
    """
    client_df = df[df['Name'] == client]
    
    if client_df.empty:
        return {'per_horizon': [0.0] * len(tau), 'aggregate': 0.0}

    per_horizon_pnl = []
    
    for t in tau:
        m_col = f'M{t}'
        # Equation 5: PnL at tau = side * Volume * (M_tau - Trade Price)
        pnl_series = client_df['Side'] * client_df['Volume'] * (client_df[m_col] - client_df['Trade Price'])
        
        # Expected PnL is the sample average across all trades
        expected_pnl_t = pnl_series.mean()
        per_horizon_pnl.append(expected_pnl_t)

    # Equation 6: Aggregate Expected PnL is the average of the 6 horizons
    aggregate_pnl = sum(per_horizon_pnl) / len(tau)

    return {
        'per_horizon': per_horizon_pnl,
        'aggregate': aggregate_pnl
    }

def classify_client(client: str) -> str:
    """
    Parameters:
    client: Client identifier
    
    Returns:
    'profitable' or 'costly'
    """
    tau = [5, 10, 15, 20, 25, 30]
    pnl_data = expected_pnl(client, tau)
    
    if pnl_data['aggregate'] >= 0:
        return 'profitable'
    else:
        return 'costly'

def min_half_spread(client: str) -> float:
    """
    Parameters:
    client: Client identifier
    
    Returns:
    Minimum half-spread delta* such that expected aggregate PnL >= 0
    """
    client_df = df[df['Name'] == client]
    
    if client_df.empty or client_df['Volume'].sum() == 0:
        return 0.0

    tau_cols = ['M5', 'M10', 'M15', 'M20', 'M25', 'M30']
    
    # Calculate the average future mid-price across all 6 horizons
    avg_future_mid = client_df[tau_cols].mean(axis=1)
    
    # Calculate the total adverse movement from M0 weighted by side
    adverse_movement = -client_df['Side'] * (avg_future_mid - client_df['M0'])
    
    # Volume-weighted sum
    required_total_delta = (adverse_movement * client_df['Volume']).sum()
    total_volume = client_df['Volume'].sum()
    
    delta_star = required_total_delta / total_volume
    
    # A true half-spread must be >= 0 in the orderbook
    return max(0.0, float(delta_star))


if __name__ == "__main__":
    horizons = [5, 10, 15, 20, 25, 30]
    
    # Get unique clients, sorted alphabetically
    unique_clients = sorted(df['Name'].unique())
    
    results = []
    for client in unique_clients:
        # Format name for output specifically to look like "Client A"
        client_name = f"Client {client}" if not str(client).startswith("Client") else client
        
        # Calculate required data
        pnl_dict = expected_pnl(client, horizons)
        agg_pnl = pnl_dict['aggregate']
        classification = classify_client(client)
        delta_star = min_half_spread(client)
        
        # Construct the row matching the CSV schema
        row = [client_name] + pnl_dict['per_horizon'] + [agg_pnl, delta_star]
        results.append(row)
        
    # Generate the DataFrame and export to CSV
    columns = ['client', 'τ=5', 'τ=10', 'τ=15', 'τ=20', 'τ=25', 'τ=30', 'agg_pnl', 'δ*']
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv('task2_results.csv', index=False)