"""
Noise XGBoost model for AE index forecasting, with Optuna hyperparameter tuning and rolling mean features.
Adds noise parameters on top of base features to compare information gain.

Author: Cara Waters
10/02/2026
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_squared_error
import pyspedas
import optuna
import pathlib
import datetime as dt
from datetime import timezone
import json
import gc

pyspedas.projects.omni.data(trange=['2009-12-31', '2025-01-01'], time_clip=True) # loads all omni data products

t, ae = pyspedas.get_data('AE_INDEX') # 1 minute resolution
_, bx = pyspedas.get_data('BX_GSE')
_, by = pyspedas.get_data('BY_GSM')
_, bz = pyspedas.get_data('BZ_GSM')
_, vx = pyspedas.get_data('Vx')
_, vy = pyspedas.get_data('Vy')
_, vz = pyspedas.get_data('Vz')
_, n = pyspedas.get_data('proton_density')

df_omni = pd.DataFrame({
    'AE_INDEX': ae,
    'BX_GSM': bx,
    'BY_GSM': by,
    'BZ_GSM': bz,
    'n': n,
    'Vx': vx,
    'Vy': vy,
    'Vz': vz,
}, index=pd.to_datetime(t, unit='s')) # into DataFrame with DateTimeIndex

del t, ae, bx, by, bz, vx, vy, vz, n

def interpolate_short_gaps(
    df: pd.DataFrame,
    freq: str | None = None,
    max_gap: str = "5min",
    method: str = "time",
    limit_direction: str = "both"
) -> pd.DataFrame:
    """
    Interpolates short gaps within a DataFrame given measurement frequency and maximum data gap to interpolate.

    Args:
        df (pd.DataFrame): DataFrame with a DateTimeIndex to be interpolated.
        freq (str | None, optional): Measurement frequency e.g. "1min". Defaults to None (then infers cadence).
        max_gap (str, optional): Maximum time duration of gaps to interpolate over. Defaults to "5min".
        method (str, optional): Interpolation method; "time" recommended for DateTimeIndex. Defaults to "time".
        limit_direction (str, optional): Direction to fill within the gap ("forward"|"backward"|"both"). Defaults to "both".

    Raises:
        TypeError: DataFrame index is not a DateTimeIndex.
        ValueError: Cannot infer a valid frequency from the index or max_gap is invalid.

    Returns:
        pd.DataFrame: Interpolated DataFrame.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a pandas DatetimeIndex.")

    df = df.sort_index()

    # Decide target frequency
    if freq is None:
        freq = pd.infer_freq(df.index)
        if freq is None:
            # fall back to median spacing
            med_dt = (df.index.to_series().diff().dropna().median())
            if pd.isna(med_dt) or med_dt <= pd.Timedelta(0):
                raise ValueError("Cannot infer a valid frequency from the index.")
            # round med_dt to nearest second for a clean freq string
            med_sec = int(round(med_dt.total_seconds()))
            freq = f"{med_sec}S"

    # Reindex to a regular grid
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df_reg = df.reindex(full_index)

    # Compute the max number of consecutive NaNs to fill ("limit")
    freq_td = pd.to_timedelta(pd.tseries.frequencies.to_offset(freq))
    max_gap_td = pd.to_timedelta(max_gap)
    if max_gap_td <= pd.Timedelta(0):
        # nothing to fill
        return df_reg
    limit = int(np.floor(max_gap_td / freq_td))
    if limit < 1:
        # user set max_gap smaller than one step; do not fill
        return df_reg

    # Interpolate numeric columns only; leave non-numeric as-is
    num_cols = df_reg.select_dtypes(include=[np.number]).columns
    out = df_reg.copy()

    # Uses limit_area="inside" so doesn't invent values at the very start/end
    out[num_cols] = out[num_cols].interpolate(
        method=method,
        limit=limit,
        limit_direction=limit_direction,
        limit_area="inside"
    )

    return out

df_clean = interpolate_short_gaps(df_omni, freq="1min", max_gap="3min", method="time", limit_direction="forward")

def make_multiwindow_features(
    df: pd.DataFrame,
    cols: list[str],
    windows: list[int] = [5, 10, 20, 30, 60],
    min_fraction: float = 0.5,
    prefix_mean: str = "_mean",
    prefix_std: str = "_std",
) -> pd.DataFrame:
    """
    Calculates rolling mean and std features for specified columns and window sizes, with a minimum data fraction requirement.

    Args:
        df (pd.DataFrame): Input DataFrame with a DateTimeIndex.
        cols (list[str]): List of column names to calculate rolling features for.
        windows (list[int], optional): List of window sizes in minutes (or samples if 1-min cadence). Defaults to [5, 10, 20, 30, 60].
        min_fraction (float, optional): Minimum fraction of samples required in the window for a valid result. Defaults to 0.5.
        prefix_mean (str, optional): Suffix for new rolling mean feature names. Defaults to "_mean".
        prefix_std (str, optional): Suffix for new rolling std feature names. Defaults to "_std".

    Returns:
        pd.DataFrame: DataFrame with new rolling mean and std features added.
    """
    df_out = df.copy()
    for W in windows:
        min_periods = int(W * min_fraction)
        for c in cols:
            df_out[f"{c}{prefix_mean}_{W}"] = (
                df_out[c].rolling(window=W, min_periods=min_periods).mean()
            )
            # Uncomment to calculate standard deviations - functionality not used in final work
            #df_out[f"{c}{prefix_std}_{W}"] = (
            #    df_out[c].rolling(window=W, min_periods=min_periods).std()/df_out[c].rolling(window=W, min_periods=min_periods).mean()
            #)
    return df_out

# Add target columns for each forecast horizon, and also add Bmag as a feature
df_add = pd.DataFrame({
    'AE_INDEX_next_30min': df_clean['AE_INDEX'].shift(-30),
    'AE_INDEX_next_45min': df_clean['AE_INDEX'].shift(-45),
    'AE_INDEX_next_1h': df_clean['AE_INDEX'].shift(-60),
    'AE_INDEX_next_75min': df_clean['AE_INDEX'].shift(-75),
    'AE_INDEX_next_90min': df_clean['AE_INDEX'].shift(-90),
    'AE_INDEX_next_2h': df_clean['AE_INDEX'].shift(-120),
    'AE_INDEX_next_3h': df_clean['AE_INDEX'].shift(-180),
    'Bmag': np.sqrt(df_clean['BX_GSM']**2 + df_clean['BY_GSM']**2 + df_clean['BZ_GSM']**2)},
    index=df_clean.index
    )

# Combine target features and Bmag with interpolated features (left join to keep NaNs where targets are missing at the end)
df_clean = df_clean.join(df_add, how="left")

features_to_roll = ["BX_GSM", "BY_GSM", "BZ_GSM", "Vx", "Vy", "Vz", "n", "Bmag", "AE_INDEX"]

df_roll = make_multiwindow_features(df_clean, cols=features_to_roll, windows=[5, 10, 30, 60, 120]) # Features for rolling means

df_new = df_roll.loc['2010-01-01T00:00:00':'2024-12-31T23:59:59']

del df_omni, df_add, df_clean, df_roll # Free memory

df_new['Vmag'] = np.sqrt(df_new['Vx']**2 + df_new['Vy']**2 + df_new['Vz']**2)

V_gse = df_new[['Vx', 'Vy', 'Vz']].values
V_gsm = np.empty_like(V_gse)

pyspedas.store_data('V_GSE', data={'x': df_new.index.astype(np.int64) // 10**9,
                                 'y': V_gse})

# Coordinate transform from GSE to GSM using pySPEDAS (handles the time-varying transformation)
# Required for calculating turbulence features in the correct coordinate system (e.g., Alfven speed components)
pyspedas.cotrans(name_in='V_GSE', coord_in='GSE', coord_out='GSM', name_out='V_GSM')

df_new['Vx_GSM'] = pyspedas.get_data('V_GSM')[1][:,0]
df_new['Vy_GSM'] = pyspedas.get_data('V_GSM')[1][:,1]
df_new['Vz_GSM'] = pyspedas.get_data('V_GSM')[1][:,2]

# Calculate Alfvén speed components in GSM coordinates (v_A = B / sqrt(mu_0 * n * m_p))
# B in Tesla, n in m^-3, m_p in kg, mu_0 in H/m. Convert to km/s by dividing by 1e3.
df_new['bx_alfven'] = df_new['BX_GSM'] * 1e-9 / np.sqrt(4 * np.pi * 1e-7 * df_new['n'] * 1e6 * 1.6726219e-27) / 1e3
df_new['by_alfven'] = df_new['BY_GSM'] * 1e-9 / np.sqrt(4 * np.pi * 1e-7 * df_new['n'] * 1e6 * 1.6726219e-27) / 1e3
df_new['bz_alfven'] = df_new['BZ_GSM'] * 1e-9 / np.sqrt(4 * np.pi * 1e-7 * df_new['n'] * 1e6 * 1.6726219e-27) / 1e3

# Windows to calculate turbulence features using
windows = ['5', '15', '30', '45', '60']

for window in windows:
    # Rolling averages to take fluctuations relevant to
    av_B = df_new['Bmag'].rolling(window=int(window), min_periods=int(int(window) * 0.5)).mean()
    av_V = df_new['Vmag'].rolling(window=int(window), min_periods=int(int(window) * 0.5)).mean()
    
    av_Bx = df_new['BX_GSM'].rolling(window=int(window), min_periods=int(int(window) * 0.5)).mean()
    av_By = df_new['BY_GSM'].rolling(window=int(window), min_periods=int(int(window) * 0.5)).mean()
    av_Bz = df_new['BZ_GSM'].rolling(window=int(window), min_periods=int(int(window) * 0.5)).mean()
    
    av_Vx = df_new['Vx_GSM'].rolling(window=int(window), min_periods=int(int(window) * 0.5)).mean()
    av_Vy = df_new['Vy_GSM'].rolling(window=int(window), min_periods=int(int(window) * 0.5)).mean()
    av_Vz = df_new['Vz_GSM'].rolling(window=int(window), min_periods=int(int(window) * 0.5)).mean()
    
    av_n = df_new['n'].rolling(window=int(window), min_periods=int(int(window) * 0.5)).mean()
    
    # Using a rolling window
    fluc_Bx = df_new['BX_GSM'] - av_Bx
    fluc_By = df_new['BY_GSM'] - av_By
    fluc_Bz = df_new['BZ_GSM'] - av_Bz
    fluc_Bmag = df_new['Bmag'] - av_B
    fluc_Vx = df_new['Vx_GSM'] - av_Vx
    fluc_Vy = df_new['Vy_GSM'] - av_Vy
    fluc_Vz = df_new['Vz_GSM'] - av_Vz
    fluc_n = df_new['n'] - av_n
    
    df_new[f'Bx_rms_{window}'] = np.sqrt((fluc_Bx**2).rolling(int(window), min_periods=int(int(window) * 0.5)).mean())
    df_new[f'By_rms_{window}'] = np.sqrt((fluc_By**2).rolling(int(window), min_periods=int(int(window) * 0.5)).mean())
    df_new[f'Bz_rms_{window}'] = np.sqrt((fluc_Bz**2).rolling(int(window), min_periods=int(int(window) * 0.5)).mean())
    df_new[f'Vx_rms_{window}'] = np.sqrt((fluc_Vx**2).rolling(int(window), min_periods=int(int(window) * 0.5)).mean())
    df_new[f'Vy_rms_{window}'] = np.sqrt((fluc_Vy**2).rolling(int(window), min_periods=int(int(window) * 0.5)).mean())
    df_new[f'Vz_rms_{window}'] = np.sqrt((fluc_Vz**2).rolling(int(window), min_periods=int(int(window) * 0.5)).mean())
    
    df_new[f'B_rms_{window}'] = np.sqrt((fluc_Bx**2 + fluc_By**2 + fluc_Bz**2).rolling(int(window), min_periods=int(int(window) * 0.5)).mean())
    df_new[f'n_rms_{window}'] = np.sqrt((fluc_n**2).rolling(int(window), min_periods=int(int(window) * 0.5)).mean()) # not used
    
    df_new[f'Bx_skew_{window}'] = fluc_Bx.rolling(window=int(window), min_periods=int(int(window) * 0.5)).skew()
    df_new[f'By_skew_{window}'] = fluc_By.rolling(window=int(window), min_periods=int(int(window) * 0.5)).skew()
    df_new[f'Bz_skew_{window}'] = fluc_Bz.rolling(window=int(window), min_periods=int(int(window) * 0.5)).skew()
    
    df_new[f'Bx_kurt_{window}'] = fluc_Bx.rolling(window=int(window), min_periods=int(int(window) * 0.5)).kurt()
    df_new[f'By_kurt_{window}'] = fluc_By.rolling(window=int(window), min_periods=int(int(window) * 0.5)).kurt()
    df_new[f'Bz_kurt_{window}'] = fluc_Bz.rolling(window=int(window), min_periods=int(int(window) * 0.5)).kurt()
    
    # Find fluctuations in Alfven units
    av_bxalfven = df_new['bx_alfven'].rolling(window=int(window), min_periods=int(int(window) * 0.5)).mean()
    av_byalfven = df_new['by_alfven'].rolling(window=int(window), min_periods=int(int(window) * 0.5)).mean()
    av_bzalfven = df_new['bz_alfven'].rolling(window=int(window), min_periods=int(int(window) * 0.5)).mean()
    
    fluc_bxalfven = df_new['bx_alfven'] - av_bxalfven
    fluc_byalfven = df_new['by_alfven'] - av_byalfven
    fluc_bzalfven = df_new['bz_alfven'] - av_bzalfven
    
    vfluc_mag = np.sqrt(fluc_Vx**2 + fluc_Vy**2 + fluc_Vz**2)
    bfluc_mag = np.sqrt(fluc_bxalfven**2 + fluc_byalfven**2 + fluc_bzalfven**2)
    
    vfluc_mag_sq = vfluc_mag ** 2
    bfluc_mag_sq = bfluc_mag ** 2
    
    bfluxdotvfluc = (fluc_bxalfven * fluc_Vx + fluc_byalfven * fluc_Vy + fluc_bzalfven * fluc_Vz)
    
    bfluxdotvfluc_avg = bfluxdotvfluc.rolling(window=int(window), min_periods=int(int(window) * 0.5)).mean()
    vfluc_mag_sq_avg = vfluc_mag_sq.rolling(window=int(window), min_periods=int(int(window) * 0.5)).mean()
    bfluc_mag_sq_avg = bfluc_mag_sq.rolling(window=int(window), min_periods=int(int(window) * 0.5)).mean()
    
    # Normalised cross helicity 2 * (b' . v') / (|b'|^2 + |v'|^2)
    df_new[f'sigma_c_{window}'] = (2 * bfluxdotvfluc_avg) / (bfluc_mag_sq_avg + vfluc_mag_sq_avg)
    
    # Normalised residual energy (|v'|^2 - |b'|^2) / (|v'|^2 + |b'|^2)
    df_new[f'sigma_r_{window}'] = (vfluc_mag_sq_avg - bfluc_mag_sq_avg) / (vfluc_mag_sq_avg + bfluc_mag_sq_avg)
    
    # Compressibility calculations, using field strength rather than a component
    comp_top = fluc_Bmag**2
    comp_top = comp_top.rolling(window=int(window), min_periods=int(int(window) * 0.5)).mean()
    comp_bot = (fluc_Bx**2 + fluc_By**2 + fluc_Bz**2)
    comp_bot = comp_bot.rolling(window=int(window), min_periods=int(int(window) * 0.5)).mean()
    
    comp_ratio = comp_top / comp_bot
    
    df_new[f'compressibility_{window}'] = comp_ratio
    
    # PVI calculations - not used (not beneficial without additional window combinations)
    dBx = df_new['BX_GSM'] - df_new['BX_GSM'].shift(int(window))
    dBy = df_new['BY_GSM'] - df_new['BY_GSM'].shift(int(window))
    dBz = df_new['BZ_GSM'] - df_new['BZ_GSM'].shift(int(window))
    
    dB2 = dBx**2 + dBy**2 + dBz**2
    dB = np.sqrt(dB2)
    
    dB2_window = dB2.rolling(window=int(window), min_periods=int(int(window) * 0.5)).mean()
    PVI = dB / np.sqrt(dB2_window)
    
    df_new[f'PVI_{window}'] = PVI.rolling(window=int(window), min_periods=int(int(window) * 0.5)).mean()
    
    # Free memory
    del av_B, av_V, av_Bx, av_By, av_Bz, av_Vx, av_Vy, av_Vz, av_n
    del fluc_Bx, fluc_By, fluc_Bz, fluc_Bmag, fluc_Vx, fluc_Vy, fluc_Vz, fluc_n
    del av_bxalfven, av_byalfven, av_bzalfven, fluc_bxalfven, fluc_byalfven, fluc_bzalfven
    del vfluc_mag, bfluc_mag, vfluc_mag_sq, bfluc_mag_sq, bfluxdotvfluc
    del bfluxdotvfluc_avg, vfluc_mag_sq_avg, bfluc_mag_sq_avg
    del comp_top, comp_bot, comp_ratio, dBx, dBy, dBz, dB2, dB, dB2_window, PVI
    gc.collect()

# Features to replace with noise
FEATURES_noise =  ["Bx_rms_5", "Bx_rms_10", "Bx_rms_20", "Bx_rms_30", "Bx_rms_60",
                    "By_rms_5", "By_rms_10", "By_rms_20", "By_rms_30", "By_rms_60",
                    "Bz_rms_5", "Bz_rms_10", "Bz_rms_20", "Bz_rms_30", "Bz_rms_60",
                    "Vx_rms_5", "Vx_rms_10", "Vx_rms_20", "Vx_rms_30", "Vx_rms_60",
                    "Vy_rms_5", "Vy_rms_10", "Vy_rms_20", "Vy_rms_30", "Vy_rms_60",
                    "Vz_rms_5", "Vz_rms_10", "Vz_rms_20", "Vz_rms_30", "Vz_rms_60",
                    "B_rms_5", "B_rms_10", "B_rms_20", "B_rms_30", "B_rms_60",
                    "Bx_skew_5", "Bx_skew_10", "Bx_skew_20", "Bx_skew_30", "Bx_skew_60",
                    "By_skew_5", "By_skew_10", "By_skew_20", "By_skew_30", "By_skew_60",
                    "Bz_skew_5", "Bz_skew_10", "Bz_skew_20", "Bz_skew_30", "Bz_skew_60",
                    "Bx_kurt_5", "Bx_kurt_10", "Bx_kurt_20", "Bx_kurt_30", "Bx_kurt_60",
                    "By_kurt_5", "By_kurt_10", "By_kurt_20", "By_kurt_30", "By_kurt_60",
                    "Bz_kurt_5", "Bz_kurt_10", "Bz_kurt_20", "Bz_kurt_30", "Bz_kurt_60",
                    "sigma_c_5", "sigma_c_10", "sigma_c_20", "sigma_c_30", "sigma_c_60",
                    "sigma_r_5", "sigma_r_10", "sigma_r_20", "sigma_r_30", "sigma_r_60",
                    "compressibility_5", "compressibility_10", "compressibility_20", "compressibility_30", "compressibility_60",
                  ]

for col in FEATURES_noise:
    # Preserves first and second moments of the distribution, but removes temporal correlations and any relationship with the target
    df_new[col] = np.random.normal(loc=df_new[col].mean(), scale=df_new[col].std(), size=len(df_new))
    mask = df_new[col].isna()
    df_new.loc[mask, col] = np.nan
    
# ===== Target and Feature definitions for models =====
# Each model targets one of these forecast horizons
TARGETS = {
    "30min": "AE_INDEX_next_30min",
    "45min": "AE_INDEX_next_45min",
    "1h":    "AE_INDEX_next_1h",
    "75min": "AE_INDEX_next_75min",
    "90min": "AE_INDEX_next_90min",
    "2h":    "AE_INDEX_next_2h",
    "3h":    "AE_INDEX_next_3h",
}

# Each model makes use of all of these features
FEATURES = [
    "BX_GSM_mean_5", "BX_GSM_mean_10", "BX_GSM_mean_30", "BX_GSM_mean_60", "BX_GSM_mean_120",
    "BY_GSM_mean_5", "BY_GSM_mean_10", "BY_GSM_mean_30", "BY_GSM_mean_60", "BY_GSM_mean_120",
    "BZ_GSM_mean_5", "BZ_GSM_mean_10", "BZ_GSM_mean_30", "BZ_GSM_mean_60", "BZ_GSM_mean_120",
    "Vx_mean_5", "Vx_mean_10", "Vx_mean_30", "Vx_mean_60", "Vx_mean_120",
    "Vy_mean_5", "Vy_mean_10", "Vy_mean_30", "Vy_mean_60", "Vy_mean_120",
    "Vz_mean_5", "Vz_mean_10", "Vz_mean_30", "Vz_mean_60", "Vz_mean_120",
    "n_mean_5", "n_mean_10", "n_mean_30", "n_mean_60", "n_mean_120",
    "Bmag_mean_5", "Bmag_mean_10", "Bmag_mean_30", "Bmag_mean_60", "Bmag_mean_120",
    "AE_INDEX_mean_5", "AE_INDEX_mean_10", "AE_INDEX_mean_30", "AE_INDEX_mean_60", "AE_INDEX_mean_120",
    "Bx_rms_5", "Bx_rms_15", "Bx_rms_30", "Bx_rms_45", "Bx_rms_60",
    "By_rms_5", "By_rms_15", "By_rms_30", "By_rms_45", "By_rms_60",
    "Bz_rms_5", "Bz_rms_15", "Bz_rms_30", "Bz_rms_45", "Bz_rms_60",
    "Vx_rms_5", "Vx_rms_15", "Vx_rms_30", "Vx_rms_45", "Vx_rms_60",
    "Vy_rms_5", "Vy_rms_15", "Vy_rms_30", "Vy_rms_45", "Vy_rms_60",
    "Vz_rms_5", "Vz_rms_15", "Vz_rms_30", "Vz_rms_45", "Vz_rms_60",
    "B_rms_5", "B_rms_15", "B_rms_30", "B_rms_45", "B_rms_60",
    "Bx_skew_5", "Bx_skew_15", "Bx_skew_30", "Bx_skew_45", "Bx_skew_60",
    "By_skew_5", "By_skew_15", "By_skew_30", "By_skew_45", "By_skew_60",
    "Bz_skew_5", "Bz_skew_15", "Bz_skew_30", "Bz_skew_45", "Bz_skew_60",
    "Bx_kurt_5", "Bx_kurt_15", "Bx_kurt_30", "Bx_kurt_45", "Bx_kurt_60",
    "By_kurt_5", "By_kurt_15", "By_kurt_30", "By_kurt_45", "By_kurt_60",
    "Bz_kurt_5", "Bz_kurt_15", "Bz_kurt_30", "Bz_kurt_45", "Bz_kurt_60",
    "sigma_c_5", "sigma_c_15", "sigma_c_30", "sigma_c_45", "sigma_c_60",
    "sigma_r_5", "sigma_r_15", "sigma_r_30", "sigma_r_45", "sigma_r_60",
    "compressibility_5", "compressibility_15", "compressibility_30", "compressibility_45", "compressibility_60",
]

# Train/test split time
split_time = pd.Timestamp("2023-01-01")

# Where to save model
MODELS_DIR = pathlib.Path("models_ae_xgb_noise_new")
MODELS_DIR.mkdir(exist_ok=True) # Allows creation if directory doesn't already exist

def time_split(df_model, feature_cols, target_col, split_time):
    """
    Splits training and test data at specified time.

    Args:
        df_model (pd.DataFrame): Complete DataFrame of features and targets.
        feature_cols (list[str]): List of feature column names.
        target_col (str): Name of the target column.
        split_time (pd.Timestamp): Timestamp to split training and test data.

    Returns:
        tuple: Tuple containing training features, training targets, test features, test targets, and persistence baseline targets.
    """
    dfm = df_model[feature_cols + [target_col, "AE_INDEX"]].dropna().sort_index()
    train_idx = dfm.index < split_time
    test_idx  = ~train_idx

    X_train = dfm.loc[train_idx, feature_cols]
    y_train = dfm.loc[train_idx, target_col]
    X_test  = dfm.loc[test_idx,  feature_cols]
    y_test  = dfm.loc[test_idx,  target_col]

    # persistence baseline uses AE(t) on same test rows (not a feature)
    y_persist = dfm.loc[test_idx, "AE_INDEX"]
    return X_train, y_train, X_test, y_test, y_persist


# -----------------------------
# Train/tune/eval
# -----------------------------
def train_eval(X_train, y_train, X_test, y_test, y_persist,
               n_trials=50, seed=42,
               study_db_path="optuna_studies.db", # only necessary if using SQL database
               study_name_prefix="xgb"):

    # inner validation split (time-aware)
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_va = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_va = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

    # Convert to float32 numpy (reduces copies + RAM)
    feat_names = list(X_train.columns)

    X_tr_np = X_tr.to_numpy(dtype=np.float32, copy=False)
    y_tr_np = y_tr.to_numpy(dtype=np.float32, copy=False)
    X_va_np = X_va.to_numpy(dtype=np.float32, copy=False)
    y_va_np = y_va.to_numpy(dtype=np.float32, copy=False)

    # Prebuild DMatrices once per horizon
    dtr = xgb.DMatrix(X_tr_np, label=y_tr_np, feature_names=feat_names)
    dva = xgb.DMatrix(X_va_np, label=y_va_np, feature_names=feat_names)

    # Optuna storage: SQLite on local disk (prevents RAM creep)
    # Put this DB in a non-cloud-synced location for speed
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage("/tmp/optuna_journal_noise.log")
    )

    # Keep study_name stable per horizon/seed (avoid creating many studies)
    study_name = f"{study_name_prefix}_seed{seed}"

    # Optuna objective
    def objective(trial: optuna.Trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "seed": seed,

            # Stability + memory
            "tree_method": "hist",
            "max_bin": 256,
            "n_jobs": 8,  # keep low during tuning to avoid memory spikes

            # Search space (bounded to avoid huge trees / questionable subsampling)
            "eta": trial.suggest_float("eta", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.3, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "lambda": trial.suggest_float("lambda", 0.1, 10.0, log=True),
        }

        bst = xgb.train(
            params=params,
            dtrain=dtr,
            num_boost_round=2500,
            evals=[(dva, "valid")],
            callbacks=[
                xgb.callback.EarlyStopping(rounds=120, save_best=False),
            ],
            verbose_eval=False,
        )

        # Extract scalars only
        best_iter = int(getattr(bst, "best_iteration", 0) or 0)
        best_score = float(getattr(bst, "best_score", np.inf))

        # Report only ONE scalar (prevents lots of intermediate values in Optuna/SQLite)
        trial.report(best_score, step=best_iter)
        if trial.should_prune():
            del bst
            gc.collect()
            raise optuna.TrialPruned()

        # Store one small int
        trial.set_user_attr("best_iteration", best_iter)

        # Free booster ASAP (please stop using so much RAM Optuna...)
        del bst
        gc.collect()

        return best_score

    # Create/load study in SQLite
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    # gc_after_trial important for long runs on large datasets (prevents RAM creep from old boosters)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, gc_after_trial=True, n_jobs=1)

    best_params = dict(study.best_params)
    best_params.update({
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": seed,
        "tree_method": "hist",
        "max_bin": 256,
        "n_jobs": 8,
    })

    # Max 1200 trees to prevent overfitting and long runtimes; can be adjusted based on dataset size and early stopping behavior
    best_iter = int(study.best_trial.user_attrs.get("best_iteration", 1200))

    # clean up study object early
    del study
    gc.collect()

    # ----- Refit with best params on full training slice -----
    X_train_np = X_train.to_numpy(dtype=np.float32, copy=False)
    y_train_np = y_train.to_numpy(dtype=np.float32, copy=False)
    dfull = xgb.DMatrix(X_train_np, label=y_train_np, feature_names=feat_names)

    bst = xgb.train(
        params=best_params,
        dtrain=dfull,
        num_boost_round=max(50, best_iter),
        verbose_eval=False,
    )

    # ----- Test evaluation -----
    X_test_np = X_test.to_numpy(dtype=np.float32, copy=False)
    y_test_np = y_test.to_numpy(dtype=np.float32, copy=False)
    dtest = xgb.DMatrix(X_test_np, label=y_test_np, feature_names=feat_names)

    y_pred = bst.predict(dtest)

    rmse  = root_mean_squared_error(y_test, y_pred)
    mse   = mean_squared_error(y_test, y_pred)
    mae   = mean_absolute_error(y_test, y_pred)
    r2    = r2_score(y_test, y_pred)

    # Corr can be NaN if constant; guard it
    corr = np.corrcoef(y_test.values, y_pred)[0, 1] if len(y_test) > 1 else np.nan

    # Persistence skill
    y_persist_aligned = y_persist.reindex(y_test.index)
    mask = y_persist_aligned.notna()
    mse_p = mean_squared_error(y_test[mask], y_persist_aligned[mask])
    skill = 1 - mse / mse_p if mse_p > 0 else np.nan

    # Feature importances (gain)
    fmap = {f"f{i}": name for i, name in enumerate(feat_names)}
    gains = bst.get_score(importance_type="gain")
    importances = sorted([(fmap.get(k, k), v) for k, v in gains.items()],
                         key=lambda x: x[1], reverse=True)

    print("  Best params:", best_params, "| best_iteration:", best_iter)

    y_pred_series = pd.Series(y_pred, index=y_test.index, name="AE_pred")
    return bst, {"RMSE": rmse, "MAE": mae, "R2": r2, "Correlation": corr, "Skill_vs_Persist": skill}, importances, best_params, best_iter, y_pred_series

# -----------------------------
# Save artifacts
# -----------------------------
def save_artifacts(horizon_name, bst, best_params, best_iter,
                   feat_cols, target_col, split_time,
                   Xtr, Xte, yte, yper, metrics, y_pred_series,
                   MODELS_DIR):

    base = f"{horizon_name}"
    stamp = dt.datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    model_path = MODELS_DIR / f"{base}.json"
    meta_path  = MODELS_DIR / f"{base}.meta.json"
    preds_path = MODELS_DIR / f"{base}.predictions.csv"

    bst.save_model(str(model_path))

    preds_df = pd.DataFrame({
        "AE_true": yte,
        "AE_pred": y_pred_series,
        "AE_persist": yper.reindex(yte.index),
    }, index=yte.index)

    preds_df.to_csv(preds_path)

    meta = {
        "horizon": horizon_name,
        "features": feat_cols,
        "target": target_col,
        "split_time": str(split_time),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "params": best_params,
        "best_iteration": int(best_iter),
        "metrics_test": metrics,
        "saved_utc": stamp,
        "xgboost_version": xgb.__version__,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Saved model: {model_path.name} | meta: {meta_path.name} | predictions: {preds_path.name}")


# -----------------------------
# Run loop (with cleanup)
# -----------------------------
def run_all(df_cut, FEATURES, TARGETS, split_time, MODELS_DIR, n_trials=60, seed=42):
    results = []

    for horizon_name, target_col in TARGETS.items():
        print(f"\n=== {horizon_name} ===")

        Xtr, ytr, Xte, yte, yper = time_split(df_cut, FEATURES, target_col, split_time)

        model, metrics, importances, best_params, best_iter, y_pred_series = train_eval(
            Xtr, ytr, Xte, yte, yper,
            n_trials=n_trials,
            seed=seed,
            study_db_path="optuna_studies.db",
            study_name_prefix=f"{horizon_name}"
        )

        save_artifacts(
            horizon_name, model, best_params, best_iter,
            FEATURES, target_col, split_time,
            Xtr, Xte, yte, yper, metrics, y_pred_series,
            MODELS_DIR=MODELS_DIR
        )

        results.append({"Horizon": horizon_name, **metrics})

        print(f"[{horizon_name:>6}] | "
              f"RMSE={metrics['RMSE']:.1f}  R²={metrics['R2']:.3f}  "
              f"Skill={metrics['Skill_vs_Persist']:.2%}")

        # BIG cleanup between horizons (prevents creeping RAM)
        del model, y_pred_series, importances, best_params, Xtr, ytr, Xte, yte, yper
        gc.collect()

    res_df = pd.DataFrame(results)
    print("\nSummary:")
    print(res_df.to_string(index=False))
    res_df.to_csv("xgb_optuna_results_noise_new.csv", index=False)
    return res_df

df_new = df_new.astype(np.float32)

res_df_noise_new = run_all(df_new, FEATURES, TARGETS, split_time, MODELS_DIR, n_trials=60, seed=42)