import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pathlib import Path


def load_combined_scenario_data(DATA_DIR: Path, COORDS_FILE: Path):
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


def get_scenario_map_trace(df: pd.DataFrame, scenario_type: str, scenario_label: str, visible: bool = False):
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

    # Compose multi-line hover text
    hover_text = (
        "Region: " + scenario_df["Region"] + "<br>" +
        "Time: " + scenario_df["Original Time"].astype(str) + "<br>" +
        "Scenario: " + scenario_type + " | " + scenario_label + "<br>" +
        "Delta Interchange (%): " + scenario_df["Delta Prediction (%)"].round(2).astype(str) + " %<br>" +
        "Orig Solar: " + scenario_df["Original Solar"].round(1).astype(str) + " MW<br>" +
        "Mod Solar: " + scenario_df["Modified Solar"].round(1).astype(str) + " MW<br>" +
        "Orig Wind: " + scenario_df["Original Wind"].round(1).astype(str) + " MW<br>" +
        "Mod Wind: " + scenario_df["Modified Wind"].round(1).astype(str) + " MW"
    )

    trace = go.Scattergeo(
        lon=scenario_df["Longitude"],
        lat=scenario_df["Latitude"],
        text=hover_text,
        hoverinfo="text",
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


def make_visibility(selected_type: str, selected_label: str, trace_lookup: dict):
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


def build_scenario_figure(df_all: pd.DataFrame, output_file: Path):
    """
    Builds a Plotly map figure with a single dropdown for all scenario combinations,
    and background choropleth for U.S. time zones.

    :param df_all: Combined DataFrame with all scenario and geographic data.
    :param output_file: Directory where the resulting HTML map should be saved.
    :return: None: saves figure as html file
    """
    # Create time zone base layer map
    tz_df = build_timezone_choropleth()

    scenario_pairs = sorted([
        (stype, slabel)
        for stype in df_all["Scenario Type"].unique()
        for slabel in df_all["Scenario Label"].unique()
    ])

    # Track each unique scenario trace
    traces = []
    trace_lookup = {}

    for i, (stype, slabel) in enumerate(scenario_pairs):
        visible = (i == 0)
        trace = get_scenario_map_trace(df_all, stype, slabel, visible=visible)
        traces.append(trace)
        trace_lookup[f"{stype} | {slabel}"] = len(traces) - 1

    lower = np.nanpercentile(df_all["Delta Prediction (%)"], 5)
    upper = np.nanpercentile(df_all["Delta Prediction (%)"], 95)

    # Start figure and add timezone choropleth
    fig = go.Figure()

    fig.add_trace(go.Choropleth(
        locations=tz_df["state_code"],
        locationmode="USA-states",
        z=tz_df["tz_code"],
        zmin=5,
        zmax=8,
        colorscale=["#f0f8ff", "#dceefc", "#c5e1f7", "#add8e6"],  # pastel blue hues
        showscale=False,
        marker_line_color="white",
        name="Time Zones",
        hoverinfo="none"
    ))

    # Add each scatter trace for scenario combinations
    for trace in traces:
        fig.add_trace(trace)

    # Single dropdown button
    dropdown_buttons = [
        dict(
            method="update",
            label=label,
            args=[{"visible": [True] + [i == trace_lookup[label] for i in range(len(traces))]},
                  {"title": label}]
        )
        for label in trace_lookup
    ]

    # Final layout configuration including dropdown menu
    fig.update_layout(
        title=list(trace_lookup.keys())[0],
        geo=dict(scope="usa"),
        coloraxis=dict(
            colorscale="RdYlGn",
            cmin=lower,
            cmax=upper,
            colorbar=dict(title="% Chg Interchange MW")
        ),
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.01,
                xanchor="left",
                y=0.98,
                yanchor="top"
            )
        ]
    )

    # Export to HTML
    fig.write_html(output_file, include_plotlyjs="cdn")
    print(f"Interactive map saved to: {output_file}")


def build_timezone_choropleth():
    """
    Creates a DataFrame mapping U.S. state codes to time zones with numeric codes for plotting.

    :return: DataFrame with columns ['state_code', 'timezone', 'tz_code'].
    """
    # Define time zones by US states
    timezone_map = {
        'Eastern UTC + 5:00': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA', 'OH', 'MI', 'IN', 'KY', 'GA', 'FL', 'SC', 'NC', 'VA', 'WV', 'MD', 'DE'],
        'Central UTC + 6:00': ['WI', 'IL', 'MN', 'IA', 'MO', 'AR', 'LA', 'MS', 'AL', 'TN', 'TX', 'OK', 'KS', 'NE', 'SD', 'ND'],
        'Mountain UTC + 7:00': ['MT', 'WY', 'CO', 'NM', 'AZ', 'UT', 'ID'],
        'Pacific UTC + 8:00': ['WA', 'OR', 'CA', 'NV']
    }

    # Assign time zone numeric code based on difference to UTC
    timezone_color_map = {
        'Eastern UTC + 5:00': 5,
        'Central UTC + 6:00': 6,
        'Mountain UTC + 7:00': 7,
        'Pacific UTC + 8:00': 8
    }

    # Build data frame with related code mappings
    tz_rows = []
    for tz, states in timezone_map.items():
        for st in states:
            tz_rows.append({
                "state_code": st,
                "timezone": tz,
                "tz_code": timezone_color_map[tz]
            })

    return pd.DataFrame(tz_rows)


def plot_static_weather_map(df_all: pd.DataFrame, metric_col: str, output_dir: Path):
    """
    Generates one static HTML map per scenario type for the selected weather metric.

    :param df_all: Combined scenario dataframe with lat/lon, weather, and region metadata.
    :param metric_col: Column name for the weather metric to visualize (e.g., 'Solar_Irradiance').
    :param output_dir: Path to save the generated HTML maps.
    :return: None
    """
    # Create time zone base layer map
    tz_df = build_timezone_choropleth()

    # Set range values for weather dot colors
    vmin = df_all[metric_col].min()
    vmax = df_all[metric_col].max()

    # Define red - yellow - green gradient for intensity
    color_scale = [
        [0.0, "red"],
        [0.5, "yellow"],
        [1.0, "green"]
    ]

    for scenario in df_all["Scenario Type"].unique():
        subset = df_all[df_all["Scenario Type"] == scenario]

        # Build base USA figure with pastel time zone shading
        fig = go.Figure()

        # Add pastel shading for US time zones
        fig.add_trace(go.Choropleth(
            locations=tz_df["state_code"],
            locationmode="USA-states",
            z=tz_df["tz_code"],
            zmin=5,
            zmax=8,
            colorscale=["#f0f8ff", "#e6f2ff", "#d9eaf7", "#cce5ff"],
            showscale=False,
            marker_line_color="white",
            name="Time Zones"
        ))

        # Hover info for weather data
        hover_text = [
            f"{row['Region']}<br>{metric_col.replace('_', ' ')}: {row[metric_col]:.1f}"
            + (f"<br>City: {row['City']}" if "City" in row else "")
            + (f"<br>Time: {row['Original Time']}" if "Original Time" in row else "")
            for _, row in subset.iterrows()
        ]

        # Weather data dots
        fig.add_trace(go.Scattergeo(
            lon=subset["Longitude"],
            lat=subset["Latitude"],
            text=hover_text,
            hoverinfo="text",
            marker=dict(
                size=30,
                color=subset[metric_col],
                colorscale=color_scale,
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(title=metric_col.replace("_", " ")),
                line=dict(width=0.5, color='gray'),
                sizemode='diameter'
            ),
            name=""
        ))

        fig.update_geos(showcountries=True)
        fig.update_layout(
            title=f"{metric_col.replace('_', ' ')} - {scenario}",
            geo=dict(scope="usa"),
            margin=dict(l=0, r=0, t=40, b=0))

        output_file = output_dir / f"static_map_{metric_col.replace(' ', '_')}_{scenario.replace(' ', '_')}.html"
        fig.write_html(output_file, include_plotlyjs="cdn")
        print(f"Saved: {output_file}")

def build_interactive_weather_map(df_all: pd.DataFrame, metric_cols: list, output_file: Path):
    """
    Builds a Plotly map figure with a single dropdown for all weather metric scenario combinations,
    and background choropleth for U.S. time zones.

    :param df_all: Combined DataFrame with all scenario and geographic data.
    :param FIG_DIR: Directory where the resulting HTML map should be saved.
    :return: None: saves figure as HTML file
    """
    # Create time zone base layer map
    tz_df = build_timezone_choropleth()
    fig = go.Figure()

    # Add pastel time zone choropleth
    fig.add_trace(go.Choropleth(
        locations=tz_df["state_code"],
        locationmode="USA-states",
        z=tz_df["tz_code"],
        zmin=5,
        zmax=8,
        colorscale=["#f0f8ff", "#e6f2ff", "#d9eaf7", "#cce5ff"],
        showscale=False,
        marker_line_color="white",
        name="Time Zones",
    ))

    # track each unique metric, scenario trace
    trace_lookup = {}
    buttons = []
    traces = []

    # Predefine color scale
    color_scale = [[0.0, "red"], [0.5, "yellow"], [1.0, "green"]]

    # Create all combinations of metric, scenario
    for i, metric in enumerate(metric_cols):
        vmin = df_all[metric].min()
        vmax = df_all[metric].max()

        for j, scenario in enumerate(df_all["Scenario Type"].unique()):
            subset = df_all[df_all["Scenario Type"] == scenario]

            # Construct hover text for the map
            hover_text = [
                f"{row['Region']}<br>{metric.replace('_', ' ')}: {row[metric]:.1f}"
                + (f"<br>City: {row['City']}" if "City" in row else "")
                + (f"<br>Time: {row['Original Time']}" if "Original Time" in row else "")
                for _, row in subset.iterrows()
            ]

            trace = go.Scattergeo(
                lon=subset["Longitude"],
                lat=subset["Latitude"],
                text=hover_text,
                hoverinfo="text",
                marker=dict(
                    size=16,
                    color=subset[metric],
                    colorscale=color_scale,
                    cmin=vmin,
                    cmax=vmax,
                    colorbar=dict(title=metric.replace("_", " ")) if (i == 0 and j == 0) else None,
                    line=dict(width=0.5, color='gray')
                ),
                name=f"{metric} | {scenario}",
                visible=(i == 0 and j == 0)
            )

            trace_index = len(fig.data)
            fig.add_trace(trace)
            trace_lookup[f"{metric} | {scenario}"] = trace_index

    # Create dropdown buttons for all combinations
    for key, idx in trace_lookup.items():
        visibility = [False] * len(fig.data)
        visibility[0] = True  # Keep time zone choropleth always visible
        visibility[idx] = True

        buttons.append(dict(
            method="update",
            label=key,
            args=[{"visible": visibility},
                  {"title": key}]
        ))

    # Final layout configuration including dropdown menu
    fig.update_layout(
        title=next(iter(trace_lookup)),
        geo=dict(scope="usa"),
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.01, xanchor="left",
            y=0.98, yanchor="top"
        )],
        margin=dict(l=0, r=0, t=40, b=0)
    )

    # Export to HTML
    fig.write_html(output_file, include_plotlyjs="cdn")
    print(f"Saved interactive weather map to {output_file}")

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

    # Plot weather scenarios as static figures
    plot_static_weather_map(df_all, "Solar Irradiance", FIG_DIR)
    plot_static_weather_map(df_all, "Wind Speed", FIG_DIR)

    # Plot interactive weather scenario map
    weather_scenario_file = FIG_DIR / "weather_interactive_map.html"
    metric_cols = ["Solar Irradiance", "Wind Speed"]
    build_interactive_weather_map(df_all, metric_cols, weather_scenario_file)

    # Plot interactive interchange scenario map
    interchange_scenario_file = FIG_DIR / "scenario_interchange_map.html"
    build_scenario_figure(df_all, interchange_scenario_file)


if __name__ == "__main__":
    main()
