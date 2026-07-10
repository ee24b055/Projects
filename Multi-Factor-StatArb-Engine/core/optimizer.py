import numpy as np
from scipy.optimize import minimize

def optimize_factor_neutral_weights(alphas, mkt_betas, smb_betas, hml_betas):
    """
    Applies Convex Optimization (Quadratic Programming) to solve for weights that maximize 
    idiosyncratic alpha while enforcing simultaneous neutrality across all three factor channels.
    """
    n_assets = len(alphas)
    
    def objective(w):
        return -np.dot(w, alphas)
    
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.dot(w, mkt_betas)},
        {'type': 'eq', 'fun': lambda w: np.dot(w, smb_betas)},
        {'type': 'eq', 'fun': lambda w: np.dot(w, hml_betas)},
        {'type': 'eq', 'fun': lambda w: np.sum(w)},
        {'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 2.0}
    ]
    
    bounds = [(-0.75, 0.75) for _ in range(n_assets)]
    initial_w = np.zeros(n_assets)
    
    res = minimize(objective, initial_w, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x if res.success else np.zeros(n_assets)
