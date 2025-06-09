import pandas as pd
import numpy as np
import re
import requests
from io import StringIO
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import os
from tqdm import tqdm
from pathlib import Path
import statsmodels.api as sm
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from bs4 import BeautifulSoup

def load_final_data(DATA_DIR: Path, region: str):
    """Gets the finalized per region energy and weather data"""
    df = pd.read_csv(DATA_DIR / f"final_data_{region}.csv", parse_dates=["period"])

    return df

def XG_train_test_time_split(df: pd.DataFrame, split_date:str):
    """Creates the time based train/test split using supplied dates and drops non-numerical features"""
    XG_train = df[df['period'] < split_date]
    XG_train = XG_train.drop(['period', 'respondent', 'respondent-name', 'Region'], axis=1)
    XG_test = df[df['period'] >= split_date]
    XG_test = XG_test.drop(['period', 'respondent', 'respondent-name', 'Region'], axis=1)
    y_train = XG_train[['Total interchange']]
    X_train = XG_train.drop(['Total interchange'], axis=1)
    y_test = XG_test[['Total interchange']]
    X_test = XG_test.drop(['Total interchange'], axis=1)
    print(f'Train shapes(y, X): {y_train.shape, X_train.shape}, Test shapes(y, X): {y_test.shape, X_test.shape}')

    return y_train, X_train, y_test, X_test

def fit_best_XG(param_grid: dict, X_train: pd.DataFrame, y_train: pd.DataFrame, n_splits=4):
    """Use Time Series Split to perform GridSearchCV on given parameter grid"""

    tscv = TimeSeriesSplit(n_splits=n_splits)  # default is 4 since we have 4 full years of data in train set

    xgb = XGBRegressor(random_state=42)

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    start_time = time.time()
    grid.fit(X_train, y_train)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f'Time taken to form all fits: {elapsed} seconds')

    print("Best parameters:", grid.best_params_)

    return grid

def display_XG_analytics(region: str, best_model, X_test: pd.DataFrame, y_test: pd.DataFrame, FIG_DIR: Path):
    """Calculate performance analytics for the best model from GirdSearchCV for train/test data"""
    y_pred = best_model.predict(X_test)
    y_true = y_test.values.ravel()

    # Avoid division by zero in MAPE
    mask = y_true != 0
    percent_error = np.zeros_like(y_true, dtype=float)
    percent_error[mask] = (y_pred[mask] - y_true[mask]) / y_true[mask]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs(percent_error[mask])) * 100

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")

    plot_importance(best_model, max_num_features=15)
    plt.title(f'{region} XGBoost Feature Importance')
    plt.savefig(FIG_DIR / f'{region}_XGBoost_feature_importance.png')

    return rmse, mae, mape


def train_test_result_XGBoost(region: str, df: pd.DataFrame, split_date: str, param_grid: dict, FIG_DIR: Path):
    """Run the XGBoost pipeline for the given region and split dates"""
    y_train, X_train, y_test, X_test = XG_train_test_time_split(df, split_date)
    grid = fit_best_XG(param_grid, X_train, y_train, n_splits=4)
    best_model = grid.best_estimator_
    rmse, mae, mape = display_XG_analytics(region, best_model, X_train, y_train, FIG_DIR)

    return best_model, rmse, mae, mape, grid.best_params_

def run_regional_XGBoost(DATA_DIR: Path, FIG_DIR: Path, regions: list, split_date: str, param_grid: dict):
    """Parse through provided regions and run XGBoost for each while logging results in data frame for comparison"""
    train_start = time.time()

    results = []
    for region in regions:
        print(f"Running XGBoost for {region}")
        df = load_final_data(DATA_DIR, region)
        best_model, rmse, mae, mape, best_params = train_test_result_XGBoost(region, df, split_date, param_grid, FIG_DIR)

        results.append({
            'Region': region,
            'Best_Model': best_model,
            'Best_Params': best_params,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        })

    results_df = pd.DataFrame(results)
    out_path = DATA_DIR / f"XGBoost_train_results.csv"

    if out_path.exists():
        print(f"Skipping {out_path.name} (already exists)")

    else:
        results_df.to_csv(out_path, index=False)
        print(f"Saved: XGBoost_train_results.csv")

    train_end = time.time()

    print(f"Total training time for all selected regions: {train_end - train_start} seconds")

    return results_df

def create_feature_importance_heatmap_by_region(FIG_DIR: Path, results_df: pd.DataFrame):
    """Creates normalized feature importance heat map by region"""
    feature_df_list = []

    for i, row in results_df.iterrows():
        model = row['Best_Model']
        region = row['Region']
        features = model.get_booster().feature_names

        importances = model.feature_importances_

        feature_df = pd.DataFrame({
            'Region': region,
            'Feature': features,
            'Importance': importances
        })
        feature_df_list.append(feature_df)

    feature_importance_df = pd.concat(feature_df_list, ignore_index=True)
    pivot_df = feature_importance_df.pivot(index='Feature', columns='Region', values='Importance')
    pivot_df = pivot_df.fillna(0)

    pivot_df_normalized = pivot_df.div(pivot_df.max(axis=1), axis=0)
    top_features = pivot_df_normalized.mean(axis=1).nlargest(10).index
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df_normalized.loc[top_features], cmap="coolwarm", linewidths=0.5)
    plt.title(f'XGBoost Feature Importance by Region')
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'XGBoost_feature_importance_region.png')
    plt.close()



def plot_XGBoost_analytics(FIG_DIR: Path, results_df: pd.DataFrame):
    """Creates comparison plots for each region result evaluation metrics from best model as well as feature import"""
    results_df_sorted = results_df.sort_values(by="Region")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    sns.barplot(data=results_df_sorted, x='Region', y='RMSE', ax=axes[0], palette='Blues_r', hue='Region', legend=False)
    axes[0].set_title("RMSE by Region")
    axes[0].set_ylabel("RMSE")

    sns.barplot(data=results_df_sorted, x='Region', y='MAE', ax=axes[1], palette='Greens_r', hue='Region', legend=False)
    axes[1].set_title("MAE by Region")
    axes[1].set_ylabel("MAE")

    sns.barplot(data=results_df_sorted, x='Region', y='MAPE', ax=axes[2], palette='Reds_r', hue='Region', legend=False)
    axes[2].set_title("MAPE by Region")
    axes[2].set_ylabel("MAPE")
    axes[2].set_xlabel("Region")

    plt.tight_layout()
    plt.savefig(FIG_DIR / f'XGBoost_train_results_by_region.png')
    plt.close()

    create_feature_importance_heatmap_by_region(FIG_DIR, results_df)



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

    # regions = ['MIDW']

    # Time window backward 5 years from 7/31/2024
    end = datetime(2024, 7, 31)
    start = datetime(2019, 8, 2)
    print(f"Start date: {start}")
    print(f"End date: {end}")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1]

    }

    split_date = '2023-08-01'  # Choosing to split after 4 years to give approx 80/20

    # Run XGBoost for selected regions, training split date, grid search parameters
    results_df = run_regional_XGBoost(DATA_DIR, FIG_DIR, regions, split_date, param_grid)

    # Plot results to compare across regions
    plot_XGBoost_analytics(FIG_DIR, results_df)



if __name__ == "__main__":
    main()