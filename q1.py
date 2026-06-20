import pandas as pd
from typing import List

df = pd.read_csv('trade_data.csv')

def adversity_profile(client: str, tau: List[int]) -> List[float]:
    """
    Parameters:
    client: Client identifier (single character or string)
    tau: List of horizons e.g. [5, 10, 15, 20, 25, 30]
    
    Returns:
    List of floats representing adversity percentage at each horizon
    """
    # 1. Filter data for the specific client
    client_df = df[df['Name'] == client]
    
    # Handle edge case where client has no trades
    total_trades = len(client_df)
    if total_trades == 0:
        return [0.0] * len(tau)
        
    adversity_percentages = []
    
    # 2. Iterate through each time horizon
    for t in tau:
        m_col = f'M{t}'  # Construct column name dynamically (e.g., 'M5', 'M10')
        
        # 3. Evaluate the adversity condition: side * (M_t - Trade Price) < 0
        # Volume is omitted as it does not affect the sign.
        is_adverse = (client_df['Side'] * (client_df[m_col] - client_df['Trade Price'])) < 0
        
        # 4. Aggregate and calculate the percentage
        adverse_count = is_adverse.sum()
        adversity_pct = (adverse_count / total_trades) * 100.0
        
        adversity_percentages.append(adversity_pct)
        
    return adversity_percentages

if __name__ == "__main__":
    horizons = [5, 10, 15, 20, 25, 30]
    
    # CHANGE 1: Sort the clients alphabetically
    unique_clients = sorted(df['Name'].unique())
    
    results = []
    for client in unique_clients:
        profile = adversity_profile(client, horizons)
        
        # CHANGE 2: Format the name to "Client <Name>"
        results.append([f"Client {client}"] + profile)
        
    columns = ['client', 'τ=5', 'τ=10', 'τ=15', 'τ=20', 'τ=25', 'τ=30']
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv('task1_results.csv', index=False)