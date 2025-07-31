import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from pathlib import Path

def load_data_and_model(DATA_DIR: Path, region: str):
    """
    Load the trained XGBoost model and final train/val/test datasets for a given region.

    :param DATA_DIR: Path to the data directory containing model and CSV files.
    :param region: Region identifier string.
    :return: Tuple of (model, train_df, val_df, test_df).
    """
    model = XGBRegressor()
    model.load_model(DATA_DIR / f"{region}_xgboost_model.json")
    print(f"Model loaded: {region}_xgboost_model.json")

    test_df = pd.read_csv(DATA_DIR / f"final_data_test_{region}.csv", parse_dates=["period"])
    train_df = pd.read_csv(DATA_DIR / f"final_data_train_{region}.csv", parse_dates=["period"])
    val_df = pd.read_csv(DATA_DIR / f"final_data_val_{region}.csv", parse_dates=["period"])

    return model, train_df, val_df, test_df


def prepare_features(df: pd.DataFrame):
    """
    Drop non-feature and target columns from the dataframe to prepare features and target.

    :param df: Full DataFrame with all columns including and target.
    :return: Tuple of (features, target) arrays.
    """
    drop_cols = ['period', 'respondent', 'respondent-name', 'Region', 'MO', 'DY', 'HR', 'day_of_year', 'day_of_week']
    targ_col = ['Total interchange']
    features = df.drop(columns=drop_cols + targ_col)
    target = df[targ_col].values.ravel()
    return features, target

def identify_scenarios(df: pd.DataFrame) -> dict:
    """
    Identify four key scenario rows from the data:
    Worst deficit, median, winter peak, and summer peak cases.

    :param df: Combined validation and test dataframe with 'Total interchange' column.
    :return: Dictionary of scenario name: row.
    """
    df = df.copy()
    df['period'] = pd.to_datetime(df['period'], errors='coerce')
    # print(df['period'].apply(type).value_counts())
    df = df.sort_values("period")
    df['month'] = df['period'].dt.month
    scenarios = {}

    # Full target column
    targ_col = df['Total interchange']

    # Worst case scenario
    scenarios['Worst Deficit'] = df.loc[targ_col.idxmin()]

    # Median scenario
    median_val = targ_col.median()
    scenarios['Median Deficit'] = df.iloc[(targ_col - median_val).abs().argsort()[:1]].iloc[0]

    # Winter Peak (Dec, Jan, Feb)
    winter_df = df[df['month'].isin([12, 1, 2])]
    scenarios['Winter Peak'] = winter_df.loc[winter_df['Total interchange'].idxmin()]

    # Summer Peak (Jun, Jul, Aug)
    summer_df = df[df['month'].isin([6, 7, 8])]
    scenarios['Summer Peak'] = summer_df.loc[summer_df['Total interchange'].idxmin()]

    # Drop the 'month' field from all returned scenario rows
    for key in scenarios:
        if 'month' in scenarios[key]:
            scenarios[key] = scenarios[key].drop('month')

    return scenarios


def generate_counterfactuals(original_row: pd.Series, multipliers=(1.1, 1.25, 1.5, 3.0, 5.0, 10.0)):
    """
    Generate counterfactual input rows by multiplying solar and/or wind values by given multipliers.

    :param original_row: Original input row (with features).
    :param multipliers: List of multipliers to apply to solar/wind features.
    :return: List of (label, modified_row) tuples.
    """
    rows = []

    has_wind = 'Wind' in original_row.index
    has_solar = 'Solar' in original_row.index

    for m in multipliers:
        if has_solar:
            row_solar = original_row.copy()
            row_solar['Solar'] *= m
            rows.append((f"Solar +{int((m-1)*100)}%", row_solar))

        if has_wind:
            row_wind = original_row.copy()
            row_wind['Wind'] *= m
            rows.append((f"Wind +{int((m-1)*100)}%", row_wind))

    if has_solar and has_wind:
        for m in multipliers:
            row_both = original_row.copy()
            row_both['Solar'] *= m
            row_both['Wind'] *= m
            rows.append((f"Solar+Wind +{int((m-1)*100)}%", row_both))

    return rows


def recompute_pct_features(row: pd.Series, original_row: pd.Series = None) -> pd.Series:
    """
    Recalculate percentage solar/wind features and optionally update moving averages
    based on changes from original_row.

    :param row: Modified row with changed solar/wind values.
    :param original_row: Original row used to calculate deltas for moving averages.
    :return: Updated row with recomputed percent features.
    """
    # Apply percent scaling (×100) to match original preprocessing
    if 'Solar' in row and 'Net_generation_lag_1' in row:
        row['Pct_Solar'] = (row['Solar'] / row['Net_generation_lag_1']) * 100 if row['Net_generation_lag_1'] != 0 else 0

    if 'Wind' in row and 'Net_generation_lag_1' in row:
        row['Pct_Wind'] = (row['Wind'] / row['Net_generation_lag_1']) * 100 if row['Net_generation_lag_1'] != 0 else 0

    if original_row is not None:
        if 'Solar' in row and 'Solar_30d_avg' in row and 'Pct_Solar_30d_avg' in original_row:
            orig_ratio = original_row['Pct_Solar_30d_avg']
            if orig_ratio != 0:
                row['Solar_30d_avg'] = original_row['Solar_30d_avg'] + ((row['Solar'] - original_row['Solar'])/30)
                row['Pct_Solar_30d_avg'] = original_row['Pct_Solar_30d_avg'] + ((row['Pct_Solar'] - original_row['Pct_Solar'])/30)

        if 'Wind' in row and 'Wind_30d_avg' in row and 'Pct_Wind_30d_avg' in original_row:
            orig_ratio = original_row['Pct_Wind_30d_avg']
            if orig_ratio != 0:
                row['Wind_30d_avg'] = original_row['Wind_30d_avg'] + ((row['Wind'] - original_row['Wind'])/30)
                row['Pct_Wind_30d_avg'] = original_row['Pct_Wind_30d_avg'] + ((row['Pct_Wind'] - original_row['Pct_Wind'])/30)

    return row


def safe_percent_change(new: float, old: float):
    """Avoid divide-by-zero when calculating percent change"""
    if old == 0:
        return np.nan
    return (new - old) / abs(old) * 100


def run_scenario_analysis(DATA_DIR: Path, region: str):
    """
    Run scenario analysis for a given region using the trained model and val/test data.
    Predicts the impact of increasing solar/wind on power interchange and logs results.

    :param DATA_DIR: Path to the data directory containing model and data files.
    :param region: Region identifier string.
    :return: None. Saves comparison CSVs and summary results to disk.
    """
    model, train_df, val_df, test_df = load_data_and_model(DATA_DIR, region)

    # Combine val and test with all metadata intact
    combined_df = pd.concat([val_df, test_df], ignore_index=True)
    print("Data loaded and combined")

    # Identify rows of interest
    scenario_rows = identify_scenarios(combined_df)
    print("Scenarios identified")

    results = []

    for scenario_name, full_row in scenario_rows.items():
        # Save original solar/wind
        original_solar = full_row.get("Solar", np.nan)
        original_sol_pct = full_row.get("Pct_Solar", np.nan)
        original_wind = full_row.get("Wind", np.nan)
        original_wind_pct = full_row.get("Pct_Wind", np.nan)

        # Drop non-feature columns before prediction
        feature_row = full_row.drop([
            'period', 'respondent', 'respondent-name', 'Region',
            'MO', 'DY', 'HR', 'day_of_year', 'day_of_week', 'Total interchange'
        ])
        feature_row = feature_row.astype(float)  # coerce row back to float
        base_pred = model.predict(feature_row.to_frame().T)[0]

        for label, modified_row in generate_counterfactuals(feature_row):
            modified_row = modified_row.astype(float)  # coerce row back to float
            modified_row = recompute_pct_features(modified_row, original_row=feature_row)

            if (modified_row != feature_row).any():
                # print((modified_row - feature_row)[(modified_row - feature_row) != 0])

                # Full compare csv
                full_compare = pd.DataFrame({
                    'Original': feature_row,
                    'Modified': modified_row,
                    'Difference': modified_row - feature_row
                }).dropna()
                df_compare = full_compare.copy()
                df_compare['Scenario Label'] = label  # <-- this adds the label
                df_compare['Scenario Type'] = scenario_name  # Optional: also add scenario type
                df_compare['Feature'] = df_compare.index  # Keep feature name before reset

                # Reset index to preserve feature names in the CSV
                df_compare = df_compare.reset_index(drop=True)

                # Save
                df_compare.to_csv(DATA_DIR / f"{region}_{scenario_name}_{label}_comp.csv", index=False)

            mod_solar = modified_row.get("Solar", np.nan)
            mod_sol_pct = modified_row.get("Pct_Solar", np.nan)
            mod_wind = modified_row.get("Wind", np.nan)
            mod_wind_pct = modified_row.get("Pct_Wind", np.nan)

            pred = model.predict(modified_row.to_frame().T)[0]
            delta = pred - base_pred
            delta_pct = safe_percent_change(pred, base_pred)

            results.append({
                "Region": region,
                "Scenario Type": scenario_name,
                "Scenario Label": label,
                "Original Time": full_row['period'],
                "Original Interchange": full_row['Total interchange'],
                "Original Solar": original_solar,
                "Modified Solar": mod_solar,
                "Original Pct Solar": original_sol_pct,
                "Modified Solar Pct": mod_sol_pct,
                "Original Wind": original_wind,
                "Modified Wind": mod_wind,
                "Original Pct Wind": original_wind_pct,
                "Modified Wind Pct": mod_wind_pct,
                "Baseline Prediction": base_pred,
                "Modified Prediction": pred,
                "Delta Prediction": delta,
                "Delta Prediction (%)": delta_pct
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv(DATA_DIR / f"{region}_scenario_analysis_results.csv", index=False)
    print(f"Scenario analysis complete for {region} (val + test set)")

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

    regions = ['MIDW', 'SE', 'NE', 'MIDA', 'NW', 'CENT', 'SW', 'CAR', 'CAL', 'FLA', 'NY', 'TEN', 'TEX']
    # regions = ['MIDA', 'TEN']

    for region in regions:
        try:
            run_scenario_analysis(DATA_DIR, region)
        except Exception as e:
            print(f"[ERROR] {region}: {e}")


if __name__ == "__main__":
    main()
