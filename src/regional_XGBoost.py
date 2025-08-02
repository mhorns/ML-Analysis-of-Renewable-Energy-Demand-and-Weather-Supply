import pandas as pd
import numpy as np
import random
import time
from pathlib import Path
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def load_final_data(DATA_DIR: Path, region: str):
    """
    Load finalized train, validation, and test datasets for a specific region.

    :param DATA_DIR: Path to the directory containing the final CSVs.
    :param region: Region code (e.g., 'NY', 'CAL') to load data for.
    :return: train, validation, and test DataFrames.
    """
    train_df = pd.read_csv(DATA_DIR / f"final_data_train_{region}.csv", parse_dates=["period"])
    val_df = pd.read_csv(DATA_DIR / f"final_data_val_{region}.csv", parse_dates=["period"])
    test_df = pd.read_csv(DATA_DIR / f"final_data_test_{region}.csv", parse_dates=["period"])

    return train_df, val_df, test_df

def XG_train_test_time_split(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Creates the time based train/test split using supplied dates and drops non-numerical and cyclical features

    :param train_df: Training DataFrame including target column.
    :param val_df: Validation DataFrame including target column.
    :return: y_train, X_train, y_test, X_test as separate DataFrames.
    """
    drop_cols = ['period', 'respondent', 'respondent-name', 'Region', 'MO', 'DY', 'HR', 'day_of_year', 'day_of_week']
    targ_col = ['Total interchange']
    XG_train = train_df.copy()
    XG_train = XG_train.drop(drop_cols, axis=1)
    XG_test = val_df.copy()
    XG_test = XG_test.drop(drop_cols, axis=1)
    y_train = XG_train[targ_col]
    X_train = XG_train.drop(targ_col, axis=1)
    y_test = XG_test[targ_col]
    X_test = XG_test.drop(targ_col, axis=1)
    print(f'Train shapes(y, X): {y_train.shape, X_train.shape}, Test shapes(y, X): {y_test.shape, X_test.shape}')

    return y_train, X_train, y_test, X_test

def fit_best_XG(param_grid: dict, X_train: pd.DataFrame, y_train: pd.DataFrame, n_splits: int = 7) -> GridSearchCV:
    """
    Perform grid search with time series split cross-validation for XGBoost.

    :param param_grid: Dictionary of XGBoost hyperparameters to search.
    :param X_train: Feature matrix for training.
    :param y_train: Target vector for training.
    :param n_splits: Number of time-based splits for cross-validation.
    :return: Fitted GridSearchCV object with best estimator.
    """

    tscv = TimeSeriesSplit(n_splits=n_splits)

    xgb = XGBRegressor(random_state=42)

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
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

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Safely calculate Mean Absolute Percentage Error (MAPE), avoiding divide-by-zero.

    :param y_true: True target values.
    :param y_pred: Predicted target values.
    :return: MAPE score as a percentage.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid divide-by-zero
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan

    mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100
    return mape

def display_XG_analytics(region: str, best_model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.DataFrame, FIG_DIR: Path):
    """
    Display performance metrics and save XGBoost feature importance plot.

    :param region: Region name for labeling.
    :param best_model: Fitted XGBoost model.
    :param X_test: Test features.
    :param y_test: Test targets.
    :param FIG_DIR: Directory path to save the plot.
    :return: RMSE, MAE, MAPE, and R2 scores.
    """
    y_pred = best_model.predict(X_test)
    y_true = y_test.values.ravel()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"R2: {r2}")

    plot_importance(best_model, max_num_features=15)
    plt.title(f'{region} XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'{region}_XGBoost_feature_importance.png')

    return rmse, mae, mape, r2


def train_test_result_XGBoost(region: str, train_df: pd.DataFrame, val_df: pd.DataFrame, param_grid: dict,
                              DATA_DIR: Path, FIG_DIR: Path):
    """
    Train XGBoost with grid search and evaluate on validation data.

    :param region: Region identifier.
    :param train_df: Training DataFrame.
    :param val_df: Validation DataFrame.
    :param param_grid: Grid of hyperparameters for XGBoost.
    :param DATA_DIR: Path to store model output.
    :param FIG_DIR: Path to store visual output.
    :return: Trained model, RMSE, MAE, MAPE, R2, and best params.
    """
    y_train, X_train, y_val, X_val = XG_train_test_time_split(train_df, val_df)
    grid = fit_best_XG(param_grid, X_train, y_train, n_splits=7)
    best_model = grid.best_estimator_
    best_model.save_model(DATA_DIR / f"{region}_xgboost_model.json")
    rmse, mae, mape, r2 = display_XG_analytics(region, best_model, X_val, y_val, FIG_DIR)

    return best_model, rmse, mae, mape, r2, grid.best_params_

def run_regional_XGBoost(DATA_DIR: Path, FIG_DIR: Path, regions: list, param_grid: dict):
    """
    Parse through provided regions and run XGBoost for each while logging results in data frame for comparison

    :param DATA_DIR: Path to dataset directory.
    :param FIG_DIR: Path to save figures.
    :param regions: List of region codes.
    :param param_grid: Hyperparameter grid for model tuning.
    :return: Summary DataFrame with results for all regions.
    """
    train_start = time.time()

    results = []
    for region in regions:
        print(f"Running XGBoost for {region}")
        train_df, val_df, test_df = load_final_data(DATA_DIR, region)
        best_model, rmse, mae, mape, r2, best_params = train_test_result_XGBoost(region, train_df, val_df, param_grid,
                                                                                 DATA_DIR, FIG_DIR)

        results.append({
            'Region': region,
            'Best_Model': best_model,
            'Model': (f"XGBoost | {best_params}"),
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        })

    # Save full model results per region
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
    """
    Creates normalized feature importance heat map by region

    :param FIG_DIR: Path to save the heatmap image.
    :param results_df: DataFrame with models and importance data.
    """
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
    """
    Creates comparison plots for each region result evaluation metrics from best model as well as feature import

    :param FIG_DIR: Path to save visualizations.
    :param results_df: DataFrame containing model metrics per region.
    """
    results_df_sorted = results_df.sort_values(by="Region")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    sns.barplot(data=results_df_sorted, x='Region', y='RMSE', ax=axes[0], palette='Blues_r', hue='Region', legend=False)
    axes[0].set_title("RMSE by Region")
    axes[0].set_ylabel("RMSE")

    sns.barplot(data=results_df_sorted, x='Region', y='MAE', ax=axes[1], palette='Greens_r', hue='Region', legend=False)
    axes[1].set_title("MAE by Region")
    axes[1].set_ylabel("MAE")

    sns.barplot(data=results_df_sorted, x='Region', y='MAPE', ax=axes[2], palette='Reds_r', hue='Region', legend=False)
    axes[2].set_title("MAPE by Region")
    axes[2].set_ylabel("MAPE")
    # axes[2].set_xlabel("Region")

    sns.barplot(data=results_df_sorted, x='Region', y='R2', ax=axes[3], palette='Purples_r', hue='Region', legend=False)
    axes[3].set_title("R2 by Region")
    axes[3].set_ylabel("R2")
    axes[3].set_xlabel("Region")

    plt.tight_layout()
    plt.savefig(FIG_DIR / f'XGBoost_train_results_by_region.png')
    plt.close()

    create_feature_importance_heatmap_by_region(FIG_DIR, results_df)



def main():
    # Setting seeds for the RNN reproducibility
    np.random.seed(42)
    random.seed(42)

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

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1]
    }

    # Run XGBoost for selected regions, training split date, grid search parameters
    results_df = run_regional_XGBoost(DATA_DIR, FIG_DIR, regions, param_grid)

    # Plot results to compare across regions
    plot_XGBoost_analytics(FIG_DIR, results_df)


if __name__ == "__main__":
    main()