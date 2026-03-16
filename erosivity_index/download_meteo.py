"""
Download 10min precipitation data from MeteoSwiss stations: https://www.meteoswiss.admin.ch/services-and-publications/service/open-data.html
Includes data from automatic, precipitation and tower stations from 2000 onwards
"""


import os
import pandas as pd
from pystac_client import Client
import matplotlib.pyplot as plt


def sanitize_column_names(df):
    """
    Removes single quotes from column names in a given Pandas DataFrame.
    Also rename "REFERENCE_TS" to "reference_timestamp".

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with cleaned column names.
    """
    new_column_names = {col: col.replace("'", "") for col in df.columns}
    df_cleaned = df.rename(columns=new_column_names)
   
    if "reference_timestamp" in df_cleaned.columns:
        df_cleaned = df_cleaned.rename(columns={"reference_timestamp": "time"})

    if not 'rre150z0' in df_cleaned.columns:
        print("Error: No valid precipitaiton column found!")
        return None


    return df_cleaned


if not os.path.exists('raw_station_data.pkl'):

    catalog = Client.open("https://data.geo.admin.ch/api/stac/v1")

    collections = {
    'automatic': 'ch.meteoschweiz.ogd-smn', #https://data.geo.admin.ch/browser/index.html#/collections/ch.meteoschweiz.ogd-smn?.language=en
    'precip': 'ch.meteoschweiz.ogd-smn-precip', #https://data.geo.admin.ch/browser/index.html#/collections/ch.meteoschweiz.ogd-smn-precip?.language=en
    #'tower': 'ch.meteoschweiz.ogd-smn-tower', #https://data.geo.admin.ch/browser/index.html#/collections/ch.meteoschweiz.ogd-smn-tower?.language=en
    }

    t_assets = [] # 10min data
    for c in collections:
        print(f"Collecting data from {c} stations")
        collection = collections[c]

        items = list(catalog.search(
            collections=[collection], 
            limit=200
        ).items())

        print("Stations:", len(items))

        for item in items:   # loop over stations
            for key, asset in item.assets.items():
                if "_t_" in key:   # 10-minute resolution
                    t_assets.append({
                        "station": item.id,
                        "name": key,
                        "url": asset.href
                    })


    print('Files:', len(t_assets))

    dfs = []
    for asset in t_assets:
        
        print("Loading", asset["station"], asset["name"])
        
        df = pd.read_csv(asset["url"], delimiter=';')
        df = sanitize_column_names(df)    
        df["station"] = asset["station"]    
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data.to_pickle('raw_station_data.pkl')

else:
    pass #data = pd.read_pickle('raw_station_data.pkl')


data["time"] = pd.to_datetime(
    data["time"],
    format="%d.%m.%Y %H:%M"
)
data = data[data["time"] < "2026-01-01"]

"""
# Some stats on the raw data:
# find min-max date for each site
# number of data points for each site

site_stats = data.groupby("station")["time"].agg(
    start_date="min",
    end_date="max",
    n_points="count"
).reset_index()
site_stats = site_stats.sort_values("start_date").reset_index(drop=True)

plt.figure(figsize=(8,5))
plt.hist(site_stats["n_points"], bins=40)
plt.xlabel("Number of observations per site")
plt.ylabel("Number of sites")
plt.title("Distribution of data points per site")
plt.savefig('sites_distribution.png')

plt.figure(figsize=(10,8))
for i, row in site_stats.iterrows():
    plt.plot([row["start_date"], row["end_date"]], [i, i])

plt.xlabel("Time")
plt.ylabel("Site index")
plt.title("Temporal coverage of each site")
plt.savefig('site_temporal.png')
"""


output_dir = "stations_data/stations_csv"
os.makedirs(output_dir, exist_ok=True)
for station, df_station in data.groupby("station"):
    df_station = df_station[["time", "rre150z0"]]
    df_station.to_csv(f"stations_csv/{station}.csv", index=False)
    print(f'Saved for {station}')
