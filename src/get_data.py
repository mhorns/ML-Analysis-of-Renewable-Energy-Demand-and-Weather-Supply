import pandas as pd
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

def get_and_set_region_coordinates(iso_map:dict):
    """Use Wikipedia to find the related largest airports for each EIA region by number of enplanements"""
    # Load tables into pandas
    url = "https://en.wikipedia.org/wiki/List_of_airports_in_the_United_States"
    tables = pd.read_html(url)

    # Concatenate all tables with city and airport columns
    airport_tables = [t for t in tables if 'City' in t.columns and 'Airport' in t.columns]
    airport_df = pd.concat(airport_tables, ignore_index=True)

    # Find and forward-fill state rows
    state_mask = airport_df['City'].str.isupper() & airport_df['Airport'].isna()
    airport_df.loc[state_mask, 'State'] = airport_df.loc[state_mask, 'City']
    airport_df['State'] = airport_df['State'].ffill()
    airport_df = airport_df[~state_mask].reset_index(drop=True)

    # Keep and drop cols
    cols_to_keep = ['State', 'City', 'Airport', 'FAA', 'IATA', 'ICAO', 'Enplanements']
    airport_df = airport_df[[col for col in cols_to_keep if col in airport_df.columns]]

    # Clean city and airport names
    airport_df['City'] = airport_df['City'].str.replace(r'\[.*?\]', '', regex=True).str.strip()
    airport_df['Airport'] = airport_df['Airport'].str.replace(r'\[.*?\]', '', regex=True).str.strip()

    # Clean enplanement text to get as numbers
    airport_df['Enplanements'] = (
        airport_df['Enplanements']
        .astype(str)
        .str.replace(r'\D', '', regex=True)
        .replace('', pd.NA)
        .astype(float)
    )

    airport_df['State'] = airport_df['State'].str.title()  # Convert 'ALABAMA' → 'Alabama'

    url = "https://ourairports.com/data/airports.csv"
    our_airports_df = pd.read_csv(url)

    our_airports_df['iata_code'] = our_airports_df['iata_code'].str.upper()

    merged_airports = pd.merge(
        airport_df,
        our_airports_df[['iata_code', 'latitude_deg', 'longitude_deg']],
        left_on='IATA',
        right_on='iata_code',
        how='left'
    )

    # Drop columns and round coordinates
    merged_airports = merged_airports.drop(columns=['FAA', 'ICAO', 'iata_code'], errors='ignore')
    merged_airports['latitude_deg'] = merged_airports['latitude_deg'].round(4)
    merged_airports['longitude_deg'] = merged_airports['longitude_deg'].round(4)

    merged_airports['Region'] = merged_airports['State'].map(iso_map)

    # Removing Newark as candidate for Mid Atlantic region as too close to NY
    merged_airports = merged_airports[merged_airports['IATA'] != 'EWR']

    busiest_by_region = (
        merged_airports.sort_values('Enplanements', ascending=False)
        .groupby('Region')
        .first()
        .reset_index()
    )

    return busiest_by_region

def fetch_eia_data(base_url:str, API_KEY:str, region:list, start:datetime, end:datetime, extra_facets=None):
    """Fetch paginated data from EIA API.  The EIA will only allow 5000 rows at a time"""
    offset = 0
    length = 5000
    all_data = []

    while True:  # Loop through paginated EIA results
        params = {
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": region,
            "start": start.strftime("%Y-%m-%dT%H"),
            "end": end.strftime("%Y-%m-%dT%H"),
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "offset": offset,
            "length": length,
            "api_key": API_KEY
        }

        if extra_facets:  # Allows for getting solar and wind data as extra facets
            for facet, values in extra_facets.items():
                key = f"facets[{facet}][]"
                if key not in params:
                    params[key] = []
                params[key].extend(values)

        response = requests.get(base_url, params=params)
        response.raise_for_status()
        result = response.json()
        batch = result["response"]["data"]
        if not batch:
            break  # Exit if no more data needed
        all_data.extend(batch)
        offset += length
        total = int(result["response"]["total"])
        if offset >= total:
            break  # Al data fetched

    return pd.DataFrame(all_data)


def save_combined_EIA_incl_solar_wind(DATA_DIR:Path, regions:list, API_KEY:str, start:datetime, end:datetime):
    """This function will run the fetch_eia_data() function for both the base info and the solar/wind production info"""
    # Keep track of how long download takes
    total_start = time.time()
    i = 0

    for region in regions:
        start_time = time.time()
        print(f"Fetching region: {region}")

        # Get regions data
        base_energy_url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
        df_region = fetch_eia_data(base_energy_url, API_KEY, region, start, end)
        df_region['period'] = pd.to_datetime(df_region['period'])
        df_region['value'] = pd.to_numeric(df_region['value'], errors='coerce')

        # Make wider to move observations to cols with one col per type
        df_main_wide = df_region.pivot_table(
            index=['period', 'respondent', 'respondent-name'],
            columns='type-name',
            values='value',
            aggfunc='first'
        ).reset_index()
        df_main_wide.columns.name = None

        print(f"Base Energy data fetched")

        # Get fuel type data
        base_type_url = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
        df_fuel = fetch_eia_data(base_type_url, API_KEY, region, start, end, extra_facets={"fueltype": ["SUN", "WND"]})

        df_fuel['period'] = pd.to_datetime(df_fuel['period'])
        df_fuel['value'] = pd.to_numeric(df_fuel['value'], errors='coerce')

        # Make wider to move observations to cols with one col per type
        df_fuel_wide = df_fuel.pivot_table(
            index=['period', 'respondent', 'respondent-name'],
            columns='type-name',
            values='value',
            aggfunc='first'
        ).reset_index()
        df_fuel_wide.columns.name = None

        print(f"Energy type data fetched")

        # Merge datasets on period + respondent
        merged = pd.merge(df_main_wide, df_fuel_wide, on=['period', 'respondent', 'respondent-name'], how='outer')

        # Save to csv
        out_path = DATA_DIR / f"eia_hourly_{region}.csv"
        if out_path.exists():
            print(f"Skipping {out_path.name} (already exists)")
            continue
        else:
            merged.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")

        elapsed = time.time() - start_time
        print(f"[{region}] Time taken: {elapsed:.2f} seconds")

        # Estimate remaining time
        avg_time_so_far = (time.time() - total_start) / (i + 1)
        i += 1
        est_total = avg_time_so_far * len(regions)
        remaining = est_total - (time.time() - total_start)
        print(f"Estimated total time: {est_total / 60:.2f} min | Remaining: {remaining / 60:.2f} min")



def fetch_with_retry(url, params, retries=8, delay=15):
    """This function inserts a try/except mechanism into getting the NASA POWER data as the API has timeout issues"""
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise


def save_NASA_weather_data(DATA_DIR: Path, busiest_by_region: pd.DataFrame, start: datetime, end: datetime):
    """Go out to NASA POWER Project and get the weather data related to the regional coordinates"""
    # Keep track of how long download takes
    total_start = time.time()
    i = 0

    # Parameters for NASA POWER
    base_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    base_params = {
        "community": "re",
        "parameters": "ALLSKY_SFC_SW_DWN,T2M,WSC",
        "wind-elevation": 50,
        "wind-surface": "airportgrass",
        "format": "csv",
        "header": "true",
        "time-standard": "utc"
    }

    # Loop through each region
    for _, row in busiest_by_region.iterrows():
        start_time = time.time()

        region = row['Region']
        lat = row['latitude_deg']
        lon = row['longitude_deg']

        print(f"Fetching NASA POWER weather data for region: {region} ({lat}, {lon})")

        region_dfs = []

        # NASA POWER API can timeout on long-range queries - break into 6-month chunks
        current = start
        while current < end:
            block_start = current.strftime("%Y%m%d")
            block_end = (current + relativedelta(months=6) - pd.Timedelta(days=1)).strftime("%Y%m%d")
            print(f"Getting data starting {block_start} through {block_end}...")

            params = base_params.copy()
            params.update({
                "latitude": lat,
                "longitude": lon,
                "start": block_start,
                "end": block_end
            })

            try:
                response = fetch_with_retry(base_url, params)
                csv_data = StringIO(response.text)
                df = pd.read_csv(csv_data, skiprows=14)

                df['Region'] = region
                df['datetime'] = pd.to_datetime(df[['YEAR', 'MO', 'DY', 'HR']].rename(
                    columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day', 'HR': 'hour'}))

                region_dfs.append(df)
                time.sleep(10)

            except Exception as e:
                print(f"Failed {block_start} to {block_end} for {region}: {e}")

            current += relativedelta(months=6)

        if region_dfs:
            full_df = pd.concat(region_dfs).sort_values('datetime')
            out_path = DATA_DIR / f"weather_power_{region}.csv"
            if out_path.exists():
                print(f"Skipping {out_path.name} (already exists)")
                continue
            else:
                full_df.to_csv(out_path, index=False)
                print(f"Saved: weather_power_{region}.csv")

        elapsed = time.time() - start_time
        print(f"[{region}] Time taken: {elapsed:.2f} seconds")

        # Estimate remaining time
        avg_time_so_far = (time.time() - total_start) / (i + 1)
        i += 1
        est_total = avg_time_so_far * len(busiest_by_region)
        remaining = est_total - (time.time() - total_start)
        print(f"Estimated total time: {est_total / 60:.2f} min | Remaining: {remaining / 60:.2f} min")


def main():

    # Base directory: go up one level from current script (i.e., from 'src/' to project root)
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Path to the data directory at the same level as 'src'
    DATA_DIR = BASE_DIR / "data"
    DATA_DIR.mkdir(exist_ok=True)
    print(DATA_DIR)

    iso_map = {
        'Alabama': 'SE',
        'Arizona': 'SW',
        'Arkansas': 'SE',
        'California': 'CAL',
        'Colorado': 'CENT',
        'Connecticut': 'NE',
        'Delaware': 'MIDA',
        'Florida': 'FLA',
        'Georgia': 'SE',
        'Idaho': 'NW',
        'Illinois': 'MIDW',
        'Indiana': 'MIDW',
        'Iowa': 'MIDW',
        'Kansas': 'CENT',
        'Kentucky': 'SE',
        'Louisiana': 'SE',
        'Maine': 'NE',
        'Maryland': 'MIDA',
        'Massachusetts': 'NE',
        'Michigan': 'MIDW',
        'Minnesota': 'MIDW',
        'Mississippi': 'SE',
        'Missouri': 'CENT',
        'Montana': 'NW',
        'Nebraska': 'CENT',
        'Nevada': 'SW',
        'New Hampshire': 'NE',
        'New Jersey': 'MIDA',
        'New Mexico': 'SW',
        'New York': 'NY',
        'North Carolina': 'CAR',
        'North Dakota': 'MIDW',
        'Ohio': 'MIDA',
        'Oklahoma': 'CENT',
        'Oregon': 'NW',
        'Pennsylvania': 'MIDA',
        'Rhode Island': 'NE',
        'South Carolina': 'CAR',
        'South Dakota': 'MIDW',
        'Tennessee': 'TEN',
        'Texas': 'TEX',
        'Utah': 'NW',
        'Vermont': 'NE',
        'Virginia': 'MIDA',
        'Washington': 'NW',
        'West Virginia': 'MIDW',
        'Wisconsin': 'MIDW',
        'Wyoming': 'NW'
    }

    # Get coordinates for largest airports in each ISO region
    busiest_by_region = get_and_set_region_coordinates(iso_map)

    # Get regional EIA data from API
    API_KEY = "pk8B1LVqwNuQqjMxZlYAWlDzIv9Zoq6GMDVUKTOz"  # Your API key

    # 13 EIA region codes
    regions = ['MIDW', 'SE', 'NE', 'MIDA', 'NW', 'CENT', 'SW', 'CAR', 'CAL', 'FLA', 'NY', 'TEN', 'TEX']

    # Time window backward 5 years from 7/31/2024
    end = datetime(2024, 7, 31)
    start = datetime(2019, 8, 2)
    print(f"Start date: {start}")
    print(f"End date: {end}")

    # Each function to get the data for all 13 regions can take 40 minutes(!) due to batching
    save_combined_EIA_incl_solar_wind(DATA_DIR, regions, API_KEY, start, end)
    save_NASA_weather_data(DATA_DIR, busiest_by_region, start, end)


if __name__ == "__main__":
    main()