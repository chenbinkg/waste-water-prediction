"""
Add additional lag features based on PACF analysis
"""
import pandas as pd
import numpy as np
import math

def add_lag_features(df, site_name=None):
    """Add new lag features based on PACF recommendations
    
    Args:
        df: DataFrame with original features
        site_name: Site name for site-specific features
        
    Returns:
        DataFrame with additional lag features
    """
    df = df.copy()

    # K=0.2
    # df['storage'] = np.zeros(len(df))
    # #print(df)
    # C1=math.exp(-K)
    # for i in range(1,len(df)):
    #     df.at[i,'storage']=C1*df.at[i-1,'storage']+((1-C1)*df.at[i,'Rainfall']/K)
    
    # API lags (1, 2)
    if 'API' in df.columns:
        df['lag1API'] = df['API'].shift(1).ffill().bfill()
    
    # SoilM lags (1, 2, 3)
    if 'SoilMoisture' in df.columns:
        df['SoilMoisturelag1'] = df['SoilMoisture'].shift(1).ffill().bfill()
    
    # Air temp lags (1, 2, 3, 24) and rolling means (6hr, 12hr, 24hr)
    if 'Dry bulb degC' in df.columns:
        df['Dry bulb degC lag1'] = df['Dry bulb degC'].shift(1).ffill().bfill()
        df['Dry bulb degC lag2'] = df['Dry bulb degC'].shift(2).ffill().bfill()
        df['Dry bulb degC lag3'] = df['Dry bulb degC'].shift(3).ffill().bfill()
        df['Dry bulb degC lag24'] = df['Dry bulb degC'].shift(24).ffill().bfill()
        df['Temp_6HR_mean'] = df['Dry bulb degC'].rolling(window=6).mean().bfill()
        df['Temp_12HR_mean'] = df['Dry bulb degC'].rolling(window=12).mean().bfill()
        df['Temp_24HR_mean'] = df['Dry bulb degC'].rolling(window=24).mean().bfill()
    
    # Rainfall lags (3)
    if 'Rainfall' in df.columns:
        df['Rainlag3'] = df['Rainfall'].shift(3).ffill().bfill()
    
    # Stat lags (1, 2, 3) - site-specific
    if site_name:
        stat_col = f'{site_name} stat'
        if stat_col in df.columns:
            for lag in [1, 2, 3]:
                df[f'{site_name} statlag{lag}'] = df[stat_col].shift(lag).ffill().bfill()
    else:
        if 'stat' in df.columns:
            for lag in [1, 2, 3]:
                df[f'statlag{lag}'] = df['stat'].shift(lag).ffill().bfill()
    
    return df
