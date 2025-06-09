import pandas as pd
import numpy as np
import re
import requests
from io import StringIO
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from bs4 import BeautifulSoup


def load_and_merge_region_data(DATA_DIR: Path, region: str):
    """Gets the per region energy and weather data and combines with feature engineering"""
    eia_df = pd.read_csv(DATA_DIR / f"eia_hourly_{region}.csv", parse_dates=["period"])
    weather_df = pd.read_csv(DATA_DIR / f"weather_power_{region}.csv", parse_dates=["datetime"])

    # Merge on datetime
    merged_df = pd.merge(eia_df, weather_df, left_on="period", right_on="datetime", how="inner")

    # Add features for calendar, percentage contribution of wind/solar, and lag some generation items
    df = add_calendar_features(merged_df)
    df = add_pct_features(df)
    df = add_lag_features(df)

    # Remove rows without lagged terms
    df.dropna(inplace=True)

    return df

def add_calendar_features(df):
    """Create new calendar features to account for seasonailty"""
    # Create date metrics
    df['day_of_year'] = df['period'].dt.dayofyear.astype(int)
    df['day_of_week'] = df['period'].dt.dayofweek.astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    return df

def add_pct_features(df):
    """Create new percentage features for the renewables"""
    # Fill missing values in Solar and Wind
    if 'Solar' not in df.columns:
        df['Solar'] = 0
    if 'Wind' not in df.columns:
        df['Wind'] = 0
    df['Solar'] = df['Solar'].fillna(0)
    df['Wind'] = df['Wind'].fillna(0)

    # Create percentage columns
    df['Pct_Solar'] = (df['Solar'] / df['Net generation'].replace(0, np.nan)) * 100
    df['Pct_Wind'] = (df['Wind'] / df['Net generation'].replace(0, np.nan)) * 100

    # Fill NaN/infinite from bad division with zero
    df[['Pct_Solar', 'Pct_Wind']] = df[['Pct_Solar', 'Pct_Wind']].fillna(0)

    return df

def add_lag_features(df):
    """Adding lagged features to enhance manchine learning at later steps"""
    # Create lagged forecast from 24 hours ago
    df['DA_forecast_yesterday'] = df['Day-ahead demand forecast'].shift(24)
    df['unexpect_dem_diff'] = df['Demand'] - df['DA_forecast_yesterday']

    # Create lags for net gen and interchange
    df['lag_interchange_1h'] = df['Total interchange'].shift(1)
    df['lag_interchange_24h'] = df['Total interchange'].shift(24)
    df['interchange_roll_mean_3h'] = df['Total interchange'].shift(1).rolling(3).mean()
    df['Net_generation_lag_1'] = df['Net generation'].shift(1)

    return df

def add_rolling_means(df, cols, window=720):
    """Create rolling average features for window size based on hours"""
    for col in cols:
        df[f'{col}_30d_avg'] = df[col].rolling(window=window, min_periods=1).mean()

    return df

def drop_cols(df):
    """Drop extra columns that are not needed or may cause leakage to target"""
    df = df.drop(['Demand', 'Net generation', 'datetime'], axis=1)
    df.dropna(inplace=True)
    return df

def create_correlation_matrix(FIG_DIR: Path, region: str, df: pd.DataFrame):
    """Create correlation matrix based on selected features"""
    cols = [
        'ALLSKY_SFC_SW_DWN', 'T2M', 'WSC',  # Weather features
        'Solar', 'Wind', 'Pct_Solar', 'Pct_Wind',  # Generation + features
        'Demand', 'Net generation', 'Total interchange',  # EIA key indicators
        'Day-ahead demand forecast', 'unexpect_dem_diff'  # EIA forecast
    ]

    # Correlation matrix
    corr_matrix = df[cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f'{region} Correlation Matrix')
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'{region}_correl_matrix.png')
    plt.close()

    return corr_matrix

def create_EDA_plots(FIG_DIR: Path, region: str, df: pd.DataFrame):
    """Create EDA plots of various interactions including heatmaps"""
    # Rolling demand over time
    plt.figure(figsize=(14, 5))
    plt.plot(df['datetime'], df['Demand'], label='Raw Demand', alpha=0.3)
    plt.plot(df['datetime'], df['Demand_30d_avg'], label='30-Day Avg')
    plt.legend()
    plt.title(f'{region} Demand Over Time')
    plt.savefig(FIG_DIR / f'{region}_demand_over_time.png')
    plt.close()

    # Solar gen vs irradiance
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='ALLSKY_SFC_SW_DWN', y='Solar', data=df, alpha=0.4)
    plt.title(f'{region} Solar Generation vs Solar Irradiance')
    plt.savefig(FIG_DIR / f'{region}_sol_gen_vs_sol_irr.png')
    plt.close()

    # Wind gen vs wind Speed
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='WSC', y='Wind', data=df, alpha=0.4)
    plt.title(f'{region} Wind Generation vs Wind Speed')
    plt.savefig(FIG_DIR / f'{region}_wnd_gen_vs_wnd_speed.png')
    plt.close()

    # Temp vs demand
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='T2M', y='Demand', data=df, alpha=0.4)
    plt.title(f'{region} Temperature vs Energy Demand')
    plt.savefig(FIG_DIR / f'{region}_t2m_vs_energy_demand.png')
    plt.close()

    # Heatmap demand by hr and month
    plt.figure(figsize=(8, 6))
    pivot = df.pivot_table(index='HR', columns='MO', values='Demand', aggfunc='mean')
    sns.heatmap(pivot, cmap='coolwarm')
    plt.title(f'{region} Average Demand by Hour and Month')
    plt.savefig(FIG_DIR / f'{region}_avg_hrly_demand_by_month.png')
    plt.close()

    # Heatmap avg wind gen by hr and month
    plt.figure(figsize=(8, 6))
    pivot = df.pivot_table(index='HR', columns='MO', values='Wind', aggfunc='mean')
    sns.heatmap(pivot, cmap='coolwarm')
    plt.title(f'{region} Average Wind Generation by Hour and Month')
    plt.savefig(FIG_DIR / f'{region}_avg_hrly_wind_gen_by_month.png')
    plt.close()

    # Heatmap avg solar gen by hr and month
    plt.figure(figsize=(8, 6))
    pivot = df.pivot_table(index='HR', columns='MO', values='Solar', aggfunc='mean')
    sns.heatmap(pivot, cmap='coolwarm')
    plt.title(f'{region} Average Solar Generation by Hour and Month')
    plt.savefig(FIG_DIR / f'{region}_avg_hrly_solar_gen_by_month.png')
    plt.close()

def run_EDA(DATA_DIR: Path, FIG_DIR: Path, region: str):
    """Run the whole EDA process from loading data, to feature engineering and plotting. Saves final csv for model"""
    # Load and merge region energy and weather data and engineer new features
    df = load_and_merge_region_data(DATA_DIR, region)

    # Create correlation matrix heatmap
    create_correlation_matrix(FIG_DIR, region, df)

    # Create rolling 30 day averages
    features_to_roll = [
        'Demand',
        'Day-ahead demand forecast',
        'Solar',
        'Wind',
        'Net generation',
        'Total interchange',
        'Pct_Solar',
        'Pct_Wind',
        'ALLSKY_SFC_SW_DWN',
        'T2M',
        'WSC',
        'unexpect_dem_diff'
    ]

    # Add rolling means for above columns for 720 hours(30 days)
    df = add_rolling_means(df, features_to_roll, window=720)

    # Create plots for rolling demand over time, solar generation, wind generation, temp vs energy demand and heatmaps
    create_EDA_plots(FIG_DIR, region, df)

    # Save final engineered df for model building including cleanup of lagged terms and unneeded source features
    df = drop_cols(df)
    out_path = DATA_DIR / f"final_data_{region}.csv"

    if out_path.exists():
        print(f"Skipping {out_path.name} (already exists)")

    else:
        df.to_csv(out_path, index=False)
        print(f"Saved: final_data_{region}.csv")


def main():

    # Base directory: go up one level from current script (i.e., from 'src/' to project root)
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Path to the data directory at the same level as 'src'
    DATA_DIR = BASE_DIR / "data"
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Data Directory: {DATA_DIR}")

    # Path to the figs directory at the same level as 'src'
    FIG_DIR = BASE_DIR / "figs"
    FIG_DIR.mkdir(exist_ok=True)
    print(f"Figures Directory: {FIG_DIR}")

    # 13 EIA region codes
    regions = ['MIDW', 'SE', 'NE', 'MIDA', 'NW', 'CENT', 'SW', 'CAR', 'CAL', 'FLA', 'NY', 'TEN', 'TEX']

    # Time window backward 5 years from 7/31/2024
    end = datetime(2024, 7, 31)
    start = datetime(2019, 8, 2)
    print(f"Start date: {start}")
    print(f"End date: {end}")

    for region in regions:
        print(f"Running EDA for {region}")
        run_EDA(DATA_DIR, FIG_DIR, region)

if __name__ == "__main__":
    main()