import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pathlib import Path


def load_combined_scenario_data(DATA_DIR: Path, COORDS_FILE: Path) -> pd.DataFrame:
    """
    Combines all regional scenario CSVs into a single DataFrame with lat/lon metadata.

    :param DATA_DIR: Directory containing individual region scenario analysis CSVs.
    :param COORDS_FILE: CSV file mapping each region to city, state, latitude, and longitude.
    :return: Combined DataFrame with scenario data and geographic coordinates.
    """
    # Load airport/city coordinates
    df_coords = pd.read_csv(COORDS_FILE)
    df_coords = df_coords[["Region", "City", "State", "latitude_deg", "longitude_deg"]]

    # Loop through all available region files
    combined_dfs = []
    for region in df_coords["Region"].unique():
        file_path = DATA_DIR / f"{region}_scenario_analysis_results.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            coord_row = df_coords[df_coords["Region"] == region].iloc[0]

            df["Region"] = region
            df["City"] = coord_row["City"]
            df["State"] = coord_row["State"]
            df["Latitude"] = coord_row["latitude_deg"]
            df["Longitude"] = coord_row["longitude_deg"]

            combined_dfs.append(df)
        else:
            print(f"[WARNING] File not found for {region}")

    if combined_dfs:
        df_results = pd.concat(combined_dfs, ignore_index=True)
        df_results.to_csv(DATA_DIR / f"combined_scenario_analysis_results.csv", index=False)
        print(f"Total region scenario results files successfully combined")
        return df_results
    else:
        raise FileNotFoundError("No regional scenario result files were found in the specified directory.")


def get_scenario_map_trace(df: pd.DataFrame, scenario_type: str, scenario_label: str,
                               visible: bool = False) -> go.Scattergeo:
    """
    Creates a Plotly Scattergeo trace for a given scenario type and label.

    :param df: Combined scenario result DataFrame including geographic coordinates.
    :param scenario_type: Scenario Type (e.g., 'Worst Deficit', 'Winter Peak').
    :param scenario_label: Scenario Label (e.g., 'Wind +100%').
    :param visible: Whether this trace should be initially visible in the figure.
    :return: Plotly Scattergeo trace.
    """
    # Filter down to the specified scenario type and label
    scenario_df = df[
        (df["Scenario Type"] == scenario_type) &
        (df["Scenario Label"] == scenario_label)
        ]

    if scenario_df.empty:
        return go.Scattergeo()  # invisible placeholder for consistency

    color_values = scenario_df["Delta Prediction (%)"]
    max_abs = np.nanmax(np.abs(df["Delta Prediction (%)"]))

    trace = go.Scattergeo(
        lon=scenario_df["Longitude"],
        lat=scenario_df["Latitude"],
        text=(
            scenario_df["Region"]
            + "<br>Change Interchange: "
            + scenario_df["Delta Prediction (%)"].round(2).astype(str)
            + " %"
        ),
        marker=dict(
            size=16,
            color=color_values,
            coloraxis="coloraxis",  # this links to a global color scale
            line=dict(width=0.5, color="black")
        ),
        name=f"{scenario_type} | {scenario_label}",
        visible=visible
    )

    return trace

def make_visibility(selected_type: str, selected_label: str, trace_lookup: dict) -> list[bool]:
    """
    Creates a visibility mask for Plotly traces based on selected scenario type and label.

    :param selected_type: Scenario type currently selected by the user.
    :param selected_label: Scenario label currently selected by the user.
    :param trace_lookup: Mapping of (scenario_type, scenario_label) to trace index.
    :return: List of booleans to toggle visibility of each trace.
    """
    visibilities = [
        (stype == selected_type and slabel == selected_label)
        for (stype, slabel) in trace_lookup
    ]
    return visibilities

def build_scenario_figure(df_all: pd.DataFrame, FIG_DIR: Path) -> go.Figure:
    """
    Builds a Plotly map figure with dropdowns for scenario type and label.

    :param df_all: Combined DataFrame with all scenario and geographic data.
    :param FIG_DIR: Directory where the resulting HTML map should be saved.
    :return: Final Plotly figure object.
    """
    scenario_types = sorted(df_all["Scenario Type"].unique())
    scenario_labels = sorted(df_all["Scenario Label"].unique())

    traces = []
    trace_lookup = {}

    # Build all traces for each scenario combination
    for i, stype in enumerate(scenario_types):
        for j, slabel in enumerate(scenario_labels):
            is_visible = (i == 0 and j == 0)
            trace = get_scenario_map_trace(df_all, stype, slabel, visible=is_visible)
            traces.append(trace)
            trace_lookup[(stype, slabel)] = len(traces) - 1

    # Dropdowns for scenario type
    type_buttons = [
        dict(
            method="update",
            label=stype,
            args=[{"visible": make_visibility(stype, scenario_labels[0], trace_lookup)},
                  {"title": f"{stype} | {scenario_labels[0]}"}]
        )
        for stype in scenario_types
    ]

    # Dropdowns for scenario label
    label_buttons = [
        dict(
            method="update",
            label=slabel,
            args=[{"visible": make_visibility(scenario_types[0], slabel, trace_lookup)},
                  {"title": f"{scenario_types[0]} | {slabel}"}]
        )
        for slabel in scenario_labels
    ]

    lower = np.nanpercentile(df_all["Delta Prediction (%)"], 5)
    upper = np.nanpercentile(df_all["Delta Prediction (%)"], 95)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{scenario_types[0]} | {scenario_labels[0]}",
        geo=dict(scope="usa"),
        coloraxis=dict(
            colorscale="RdYlGn",
            cmin=lower,
            cmax=upper,
            colorbar=dict(title="% Chg Interchange MW")
        ),
        updatemenus=[
            dict(
                buttons=type_buttons,
                direction="down",
                showactive=True,
                x=0.05, xanchor="left",
                y=1.1, yanchor="top"
            ),
            dict(
                buttons=label_buttons,
                direction="down",
                showactive=True,
                x=0.35, xanchor="left",
                y=1.1, yanchor="top"
            )
        ]
    )

    fig.write_html(FIG_DIR, include_plotlyjs="cdn")
    print(f"Interactive map saved to: {FIG_DIR}")
    return fig


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

    COORDS_FILE = DATA_DIR / "regional_airport_locale.csv"
    print(f"Coordinates File: {COORDS_FILE}")

    # Load the combined dataframe
    df_all = load_combined_scenario_data(DATA_DIR, COORDS_FILE)
    print(f"Loaded {len(df_all)} rows across {df_all['Region'].nunique()} regions.")

    output_file = FIG_DIR / "scenario_interchange_map.html"
    fig = build_scenario_figure(df_all, output_file)


if __name__ == "__main__":
    main()
