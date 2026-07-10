import pandas as pd
import statsmodels.api as sm

def calculate_rolling_exposure(stock_returns, ff_factors, window=60):
    """
    Runs multi-factor OLS regressions inside a moving window to capture time-varying factor exposures.
    Formula: R_i - Rf = Alpha + Beta_Mkt*(Mkt-RF) + Beta_SMB*SMB + Beta_HML*HML
    """
    print("Estimating dynamic rolling Fama-French betas...")
    dates = stock_returns.index
    alphas = pd.DataFrame(index=dates, columns=stock_returns.columns, dtype=float)
    mkt_betas = pd.DataFrame(index=dates, columns=stock_returns.columns, dtype=float)
    smb_betas = pd.DataFrame(index=dates, columns=stock_returns.columns, dtype=float)
    hml_betas = pd.DataFrame(index=dates, columns=stock_returns.columns, dtype=float)
    
    X_factors = ff_factors[['Mkt-RF', 'SMB', 'HML']]
    
    for i in range(window, len(dates)):
        window_slice = slice(i - window, i)
        X = sm.add_constant(X_factors.iloc[window_slice].values)
        
        for stock in stock_returns.columns:
            Y = (stock_returns[stock].iloc[window_slice] - ff_factors['RF'].iloc[window_slice]).values
            model = sm.OLS(Y, X).fit()
            
            alphas.loc[dates[i], stock] = model.params[0]
            mkt_betas.loc[dates[i], stock] = model.params[1]
            smb_betas.loc[dates[i], stock] = model.params[2]
            hml_betas.loc[dates[i], stock] = model.params[3]
            
    return alphas.dropna(), mkt_betas.dropna(), smb_betas.dropna(), hml_betas.dropna()
