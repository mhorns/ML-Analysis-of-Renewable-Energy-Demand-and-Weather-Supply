# ML Analysis of Renewable Energy Demand and Weather Supply

This project analyzes U.S. energy demand and renewable energy supply (solar and wind) using machine learning. It aligns weather and demand data to identify underserved regions with investment potential.

## Goal
- Merge and align hourly weather, solar/wind generation, and demand data
- Identify temporal and geographic mismatches
- Use clustering and predictive models to support planning for solar/wind investment

## Setup
- TODO

## Data

[US Energy Information Administration](https://www.eia.gov/opendata/)
[NASA - The POWER Project](https://power.larc.nasa.gov/)

Two API are used for the above data in addition to scraping Wikipedia tables to gather coordinates to pass to NASA.  Examples for New York are below.

```python
# Read all tables from the page
url = "https://en.wikipedia.org/wiki/List_of_largest_cities_of_U.S._states_and_territories_by_population"
tables = pd.read_html(url)

# Get list of largest cities
cities_df = tables[1]
```

We access the EIA API and data the electricity data per below.

```python
url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
params = {
    "frequency": "hourly",
    "data[0]": "value",
    "facets[respondent][]": "NY",
    "start": "2024-01-01T00",
    "end": "2024-07-31T00",
    "sort[0][column]": "period",
    "sort[0][direction]": "desc",
    "offset": 0,
    "length": 5000,
    "api_key": "YOUR_KEY"
}

response = requests.get(url, params=params)
json_data = response.json()

eia1_df = pd.DataFrame(json_data['response']['data'])
```

```text
period respondent respondent-name type type-name value value-units 2024-07-31 00:00:00 NY New York D Demand 26508 megawatthours 2024-07-31 00:00:00 NY New York DF Day-ahead demand forecast 25158 megawatthours 2024-07-31 00:00:00 NY New York NG Net generation 24555 megawatthours 2024-07-31 00:00:00 NY New York TI Total interchange -1953 megawatthours 2024-07-30 23:00:00 NY New York D Demand 27352 megawatthours 
```

And NASA POWER data using the Wikipedia obtained coordinates for NYC.

```python
url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
params = {
    "start": "20240101",
    "end": "20240731",
    "latitude": 40.78,
    "longitude": -73.87,
    "community": "re",
    "parameters": "ALLSKY_SFC_SW_DWN,T2M,WSC",
    "wind-elevation": 100,
    "wind-surface": "airportgrass",
    "format": "csv",
    "header": "true",
    "time-standard": "utc"
}

response = requests.get(url, params=params)
response.raise_for_status()
csv_data = StringIO(response.text)
environ_df = pd.read_csv(csv_data, skiprows=14)  # Skip metadata rows
```

```text
YEAR  MO  DY  HR  ALLSKY_SFC_SW_DWN   T2M   WSC
2024   1   1   0                0.0  1.76  3.72
2024   1   1   1                0.0  1.27  3.48
2024   1   1   2                0.0  1.31  3.03
2024   1   1   3                0.0  1.52  2.55
2024   1   1   4                0.0  1.71  2.26
```



## Running The Code

## Methods

## Status
In progress.  Data generation, preprocessing and EDA underway.

## Analysis

## Future Considerations

### Contributors

[Mitchell Hornsby](https://github.com/mhorns)