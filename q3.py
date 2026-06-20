import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

# Global state to cache our trained models per horizon
_trained_models = {}

def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Parses raw data and engineers point-in-time features."""
    # Convert to datetime and sort to maintain strictly chronological time-series splitting
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['time'], format='%d-%m-%Y %H:%M:%S')
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Engineer Time Features
    df['Hour'] = df['datetime'].dt.hour
    df['Minute'] = df['datetime'].dt.minute
    
    # Engineer Aggressiveness: Side * (Execution Price - Mid Price)
    # Measures how much premium the client was willing to pay crossing the spread.
    df['Trade_Aggressiveness'] = df['Side'] * (df['Trade Price'] - df['M0'])
    
    return df

def train_pipeline(filepath: str = 'trade_data.csv') -> pd.DataFrame:
    """Trains the models and evaluates the metrics without data leakage."""
    global _trained_models
    df = pd.read_csv(filepath)
    df = _prepare_data(df)
    
    taus = [5, 10, 15, 20, 25, 30]
    n = len(df)
    train_idx = int(n * 0.6)
    val_idx = int(n * 0.8)
    
    # 60/20/20 Chronological Split
    train_df = df.iloc[:train_idx].copy()
    val_df = df.iloc[train_idx:val_idx].copy()
    test_df = df.iloc[val_idx:].copy()
    
    # THE FEATURE VECTOR ORDERING:
    # 0: Volume               (Float/Int)
    # 1: Spread               (Float)
    # 2: Hour                 (Int)
    # 3: Minute               (Int)
    # 4: Trade_Aggressiveness (Float)
    # 5: Side                 (Int: 1 or -1)
    # 6: Name_TE              (Float: Target Encoded Client Profile)
    features = ['Volume', 'Spread', 'Hour', 'Minute', 'Trade_Aggressiveness', 'Side', 'Name_TE']
    
    overall_metrics = {s: {'acc':0, 'prec':0, 'rec':0, 'll':0} for s in ['train', 'validation', 'test']}
    
    for t in taus:
        m_col = f'M{t}'
        # Create Target Variables
        for d in [train_df, val_df, test_df]:
            d['target'] = (d['Side'] * (d[m_col] - d['Trade Price']) < 0).astype(int)
            
        # Target Encoding (Performed STRICTLY on Train split to prevent leakage)
        te_map = train_df.groupby('Name')['target'].mean().to_dict()
        global_mean = train_df['target'].mean()
        
        for d in [train_df, val_df, test_df]:
            d['Name_TE'] = d['Name'].map(te_map).fillna(global_mean)
            
        X_train, y_train = train_df[features], train_df['target']
        X_val, y_val = val_df[features], val_df['target']
        X_test, y_test = test_df[features], test_df['target']
        
        # Train Gradient Boosting Classifier
        clf = HistGradientBoostingClassifier(random_state=42, max_iter=100, learning_rate=0.05, l2_regularization=0.1)
        clf.fit(X_train, y_train)

        joblib.dump(clf, f'task3_model_tau{t}.pkl')
        joblib.dump(te_map, f'task3_temap_tau{t}.pkl')
        
        _trained_models[t] = {
            'model': clf,
            'te_map': te_map,
            'global_mean': global_mean
        }
        
        _trained_models[t] = {
            'model': clf,
            'te_map': te_map,
            'global_mean': global_mean
        }
        
        # Accumulate metrics for averaging
        for split_name, X, y in [('train', X_train, y_train), ('validation', X_val, y_val), ('test', X_test, y_test)]:
            preds = clf.predict(X)
            probs = clf.predict_proba(X)[:, 1]
            overall_metrics[split_name]['acc'] += accuracy_score(y, preds) / 6.0
            overall_metrics[split_name]['prec'] += precision_score(y, preds, zero_division=0) / 6.0
            overall_metrics[split_name]['rec'] += recall_score(y, preds, zero_division=0) / 6.0
            overall_metrics[split_name]['ll'] += log_loss(y, probs) / 6.0
            
    # Format and save output
    rows = []
    for split in ['train', 'validation', 'test']:
        rows.append({
            'split': split,
            'accuracy': overall_metrics[split]['acc'],
            'precision': overall_metrics[split]['prec'],
            'recall': overall_metrics[split]['rec'],
            'log_loss': overall_metrics[split]['ll']
        })
    res_df = pd.DataFrame(rows)
    res_df.to_csv('task3_results.csv', index=False)
    return res_df

def compute_metrics(*args, **kwargs) -> pd.DataFrame:
    """
    Returns: DataFrame with [accuracy, precision, recall, log_loss] 
    and rows [train, validation, test].
    """
    filepath = kwargs.get('filepath', 'trade_data.csv')
    return train_pipeline(filepath)

def predict_adversity(*args, **kwargs) -> float:
    """
    Returns: Probability in [0, 1] that the trade is adverse.
    Expected kwargs: tau, Name, Side, Volume, Trade_Price, M0, Spread, Date, time
    """
    tau = kwargs.get('tau')
    if not _trained_models or tau not in _trained_models:
        # Auto-train if models aren't instantiated yet
        train_pipeline()
        
    model_data = _trained_models[tau]
    
    # Process inputs matching the feature engineering step
    dt = pd.to_datetime(f"{kwargs['Date']} {kwargs['time']}", format='%d-%m-%Y %H:%M:%S')
    hour = dt.hour
    minute = dt.minute
    aggressiveness = kwargs['Side'] * (kwargs['Trade_Price'] - kwargs['M0'])
    name_te = model_data['te_map'].get(kwargs['Name'], model_data['global_mean'])
    
    # Assemble Feature Vector (Must match training order perfectly)
    x_input = pd.DataFrame([{
        'Volume': kwargs['Volume'],
        'Spread': kwargs['Spread'],
        'Hour': hour,
        'Minute': minute,
        'Trade_Aggressiveness': aggressiveness,
        'Side': kwargs['Side'],
        'Name_TE': name_te
    }])
    
    clf = model_data['model']
    prob = clf.predict_proba(x_input)[0, 1]  # Index 1 is the probability of class 1 (Adverse)
    return float(prob)

if __name__ == "__main__":
    # Example execution
    results = compute_metrics()
    print("Metrics Evaluated:")
    print(results.to_string(index=False))