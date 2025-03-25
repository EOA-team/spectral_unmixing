import pystac_client
import geopandas as gpd
import rasterio
import xarray as xr
import rioxarray
import requests
from io import BytesIO
import os
import numpy as np
import geopandas as gpd
import time
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import contextily as ctx
import pickle
import glob
import seaborn as sns


def download_SRC_from_STAC(bbox, assets, download_dir, STAC_URL='https://geoservice.dlr.de/eoc/ogc/stac/v1', collection_id="S2-soilsuite-europe-2018-2022-P5Y"):
  """ 
  Get all Soil Reflectance Composite data from the STAC API for Switzerland
  More info: https://download.geoservice.dlr.de/SOILSUITE/files/EUROPE_5Y/000_Data_Overview/SoilSuite_Data_Description_Europe_V1.pdf

  :param STAC_URL: str, STAC API endpoint (defualt https://geoservice.dlr.de/eoc/ogc/stac/v1)
  :param collection_id: str, collection ID in the STAC ("S2-soilsuite-europe-2018-2022-P5Y")
  :param bbox: list, bounding box coordinates [minlon, minlat, maxlon, maxlat] in EPSG:4326
  :param assets: list of str, choose among ['metadata', 'MREF', 'MREF-STD', 'SRC', 'SRC-STD', 'SRC-CI95', 'SFREQ', 'MASK', 'thumbnail']
  :param download_dir: where zarr files should be stored
  """
 
  # Connect to STAC catalog
  catalog = pystac_client.Client.open(STAC_URL)

  # List available collections
  #collections = [col.id for col in catalog.get_all_collections()]
  #print("Available Collections:", collections)

  # Search for items in the collection
  search = catalog.search(
      collections=[collection_id],
      bbox=bbox
  )

  items = search.item_collection()

  if not items:
    print("No matching items found.")
    return

  # Collect URLs for requested assets
  asset_urls = {asset: [] for asset in assets}  # Dictionary to store asset URLs
  for item in items:
      for asset in assets:
          if asset in item.assets:
              asset_urls[asset].append(item.assets[asset].href)

  # Download TIF files (SRC is multiband so cannot with stackstac directly)
  # Store tile by tile, creating one dataset with all different assets
  for item in items:
    print(f"Processing item: {item.id}")

    # Create a list to collect datasets for this item
    item_ds_list = []

    for asset in assets:  # Loop through each asset you're interested in
        if asset in item.assets:
            url = item.assets[asset].href  # Get the URL for the asset
            print(f"Downloading {asset} from {url}")

            # Stream file into memory
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with rasterio.open(BytesIO(response.content)) as src:
                    ds = rioxarray.open_rasterio(BytesIO(response.content))

                    if src.count == 1:  
                      # Single-band asset
                      ds = ds.squeeze().rename(asset)
                    else:  
                      long_names = ds.attrs.get('long_name', []) # the band names are available in the attributes
                      ds = ds.to_dataset("band")
                      if long_names:
                          band_names = {i+1: f"{asset}_{name.split(' ')[0]}" for i, name in enumerate(long_names)}
                          ds = ds.rename(band_names)

                    item_ds_list.append(ds)
            else:
              print(f"Failed to download {url}")

    # After processing all assets for this item, combine them into a single dataset
    if item_ds_list:
        item_ds = xr.merge(item_ds_list)  # Merge all assets into one dataset
        print(item_ds)

        # Save the dataset to disk
        tile_id = item.id.split('_')[2]
        file_path = os.path.join(download_dir, f"soilsuite_{tile_id}.zarr")
        item_ds.to_zarr(file_path, mode="w")
        print(f'Saved to {file_path}')


        



######################
# 1. Download baresoil composite from DLR

# Will keep data in its native projection (EPSG:3035), grid and tile format provided by DLR
# There are 4 tiles covering CH: 0040-0024, 0040-0026, 0042-0024, 0042-0026

STAC_URL = "https://geoservice.dlr.de/eoc/ogc/stac/v1/"
collection_id = "S2-soilsuite-europe-2018-2022-P5Y"
bbox = [5.96, 45.82, 10.49, 47.81]  # Switzerland
assets = ['SRC', 'SRC-STD', 'SRC-CI95', 'MASK']
download_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/DLR_soilsuite/')

#download_SRC_from_STAC(STAC_URL=STAC_URL, collection_id=collection_id, bbox=bbox, assets=assets, download_dir=download_dir)


######################
# 2. Analyse downloaded data
""" 
data_files = os.listdir(download_dir)

# Count bare soil occurences (nbr of 20m pixels) --> MASK is 1
for f in data_files:
  ds = xr.open_zarr(os.path.join(download_dir, f))
  print(f"Number of bare soil pixels in {f}: {np.sum(ds['MASK'].values == 1)}")

# Filter for CH
swiss_borders_shp = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET.shp')
swiss_borders = gpd.read_file(swiss_borders_shp).to_crs(3035).dissolve()

total_CH_pixels = 0
for f in data_files:
  ds = xr.open_zarr(os.path.join(download_dir, f)).rio.write_crs(3035)
  ds = ds.rio.clip(swiss_borders.geometry)
  bs_pixs = np.sum(ds['MASK'].values == 1)
  print(f"Number of bare soil pixels in {f}: {bs_pixs}") # 3879457, 5809039, 1674128, 5660281
  total_CH_pixels += bs_pixs

print(f"Total number of bare soil pixels in CH: {total_CH_pixels}") # 17022905
"""

######################
# 3. Sample 250k points 
""" 
data_files = os.listdir(download_dir)
ds = xr.open_mfdataset([os.path.join(download_dir, f) for f in data_files])
ds = ds.rio.write_crs(3035)

# Resample to 10m resolution
# See if need to use regular grid inteprolator instead to minimise artifacts
scale_factor = 2  # 20m to 10m (double the resolution)
x_new = np.linspace(ds.x.values[0], ds.x.values[-1], len(ds.x.values) * scale_factor)
y_new = np.linspace(ds.y.values[-1], ds.y.values[0], len(ds.y.values) * scale_factor)
ds = ds.interp(x=x_new, y=y_new, method="nearest")

# Filter for CH
swiss_borders_shp = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET.shp')
swiss_borders = gpd.read_file(swiss_borders_shp).to_crs(3035).dissolve()
ds = ds.rio.clip(swiss_borders.geometry)

# Find baresoil locations (where MASK == 1)
valid_points = ds['MASK'] == 1
stacked = valid_points.stack(z=('y', 'x'))  # Create 1D index
valid_indices = stacked.values.nonzero()[0]  # Get valid indices

# Sample points
n_samples = min(270000, len(valid_indices)) # sample more, incase some are filled with missing data
np.random.seed(42)
sample_indices = np.random.choice(valid_indices, n_samples, replace=False)
sampled_locs = stacked.isel(z=sample_indices).coords
x_coords = sampled_locs['x'].values
y_coords = sampled_locs['y'].values

# Efficiently select sampled locations
ds = ds.drop_vars(['band', 'spatial_ref'])
samples = ds.sel(x=xr.DataArray(x_coords), y=xr.DataArray(y_coords), method='nearest')

# Drop if missing data (-10000)
df = samples.to_dataframe().reset_index()
df = df[(df != -10000).all(axis=1)]

# Subsample 250k
df = df.sample(n=250000, random_state=42)

# Save samples for future use
df.to_csv("sampled_data.csv", index=False)
"""

######################
# 4. K-means clustering - testing number of clusters
"""
# Sampled points 
samples = pd.read_csv('sampled_data.csv') # contains x and y in EPSG:3035, band reflectances
 
gdf = gpd.GeoDataFrame(samples, geometry=gpd.points_from_xy(samples['x'], samples['y']), crs="EPSG:3035").to_crs(epsg=3857)
fig, ax = plt.subplots(figsize=(8, 6))
gdf.plot(ax=ax, color='red', markersize=0.3, label="Sampled Points")
ctx.add_basemap(ax, source=ctx.providers.SwissFederalGeoportal.NationalMapColor, crs=gdf.crs)
plt.title('Sampled reflectances from SRC')
plt.savefig('sampled_pts.png')

# Prepare data
X = samples[['SRC_B2', 'SRC_B3', 'SRC_B4', 'SRC_B5', 'SRC_B6', 'SRC_B7', 'SRC_B8', 'SRC_B8A', 'SRC_B11', 'SRC_B12']].values
# No data is set to -10000, drop rows
count_missing = np.sum(X == -10000)
print(f"Number of missing values: {count_missing}")
X[X == -10000] = np.nan
X = X[~np.isnan(X).any(axis=1)]

# Convert to refl
X = X/10000

# Apply Min-Max Scaling per band
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal number of clusters (Silhouette score, Elbow method)
sil_scores = []
wcss = []

for n_clusters in range(2,11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    wcss.append(kmeans.inertia_)  # WCSS (sum of squared distances to cluster centers)

    # Save the model
    with open(f"kmeans_{n_clusters}_clusters_norm.pkl", "wb") as f:
        pickle.dump(kmeans, f)


plt.figure(figsize=(8, 4))
plt.plot(range(2, 11), sil_scores)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.savefig('sil_score_norm.png')
plt.clf()

plt.figure(figsize=(6, 4))  
plt.plot(range(2, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.title('Elbow Method for Optimal k')
plt.savefig('elbow_method_norm.png')
"""

######################
# 5. Fitting final model
"""
# Sampled points 
samples = pd.read_csv('sampled_data.csv') # contains x and y in EPSG:3035, band reflectances
# Prepare data
X = samples[['SRC_B2', 'SRC_B3', 'SRC_B4', 'SRC_B5', 'SRC_B6', 'SRC_B7', 'SRC_B8', 'SRC_B8A', 'SRC_B11', 'SRC_B12']].values
# No data is set to -10000, convert to nan
X[X == -10000] = np.nan
X = X[~np.isnan(X).any(axis=1)]
# Convert to refl
X = X/10000

# Apply Min-Max Scaling per band
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

with open(f"kmeans_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


# Fit final KMeans
for n_optim in range(3,7): 

    # Load the model
    with open(f"kmeans_{n_optim}_clusters_norm.pkl", "rb") as f:
        kmeans = pickle.load(f)

    # Predict
    labels = kmeans.predict(X_scaled)
    samples['cluster'] = labels
    samples.to_csv(f'sampled_data_{n_optim}_clusters_norm.csv')

    # Plot the samples according to cluster
    gdf = gpd.GeoDataFrame(samples, geometry=gpd.points_from_xy(samples['x'], samples['y']), crs="EPSG:3035").to_crs(epsg=3857)
    gdf['cluster'] = gdf['cluster'].astype('category')

    fig, ax = plt.subplots(figsize=(8, 6))
    colors =['teal', 'darkorange', 'purple', 'deeppink', 'limegreen', 'dodgerblue'][:n_optim] # adapt cmap in function of nbr of clusters
    custom_cmap = ListedColormap(colors)

    gdf.plot(ax=ax, column='cluster', cmap=custom_cmap, markersize=0.1, legend=True, categorical=True)
    ctx.add_basemap(ax, source=ctx.providers.SwissFederalGeoportal.NationalMapColor, crs=gdf.crs)
    plt.title(f'Sampled Reflectances from SRC ({n_optim} clusters)')
    plt.savefig(f'sampled_pts_{n_optim}_clusters_norm.png')
"""

######################
# 6. Analyse clusters
"""
# Check 25th, 50th, 75th percentiles of each cluster
bands = ['SRC_B2', 'SRC_B3','SRC_B4','SRC_B5','SRC_B6','SRC_B7','SRC_B8', 'SRC_B8A', 'SRC_B11', 'SRC_B12']
wvl = [490,560,665,705,740,783,842,865,1610,2190]


for n_optim in range(3,7): 
  df = pd.read_csv(f'sampled_data_{n_optim}_clusters.csv')
  summary = df.groupby('cluster')[bands].quantile([0.25, 0.50, 0.75])

  summary_melted = summary.rename_axis(index=['cluster', 'percentile']).reset_index().melt(id_vars=['cluster', 'percentile'], var_name='band', value_name='reflectance')
  band_to_wvl = dict(zip(bands, wvl))
  summary_melted['wavelength'] = summary_melted['band'].map(band_to_wvl)
  colors =['teal', 'darkorange', 'purple', 'deeppink', 'limegreen', 'dodgerblue'][:n_optim] # adapt cmap in function of nbr of clusters
  custom_cmap = ListedColormap(colors)
  
  # Create the plot
  plt.figure(figsize=(8, 5))
  sns.lineplot(
      data=summary_melted, 
      x='wavelength', 
      y='reflectance', 
      hue='cluster',  # Different colors per cluster
      style='percentile',  # Different line styles for quantiles
      markers=True,  # Adds markers for better readability
      dashes=True,  # Uses dashed lines for differentiation
      palette=custom_cmap
  )
  plt.legend(loc='upper left')

  # Formatting
  plt.xlabel('Wavelength (nm)')
  plt.ylabel('Reflectance')
  plt.ylim(0,5000)
  plt.title('Soil Reflectance Spectra')
  plt.savefig(f'soil_endmembers_{n_optim}_clusters_norm.png')
"""


# Plot clusters in different subplots
for n_optim in range(3,7): 
    df = pd.read_csv(f'sampled_data_{n_optim}_clusters_norm.csv')
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['x'], df['y']), crs="EPSG:3035").to_crs(epsg=3857)
    gdf['cluster'] = gdf['cluster'].astype('category')

    # Create n_optim subplots, plotting the different clusters
    fig, axs = plt.subplots(nrows=int(np.ceil(n_optim/3)), ncols=3, figsize=(15, 10))
    axs = axs.flatten()
    colors =['teal', 'orange', 'purple', 'deeppink', 'limegreen', 'dodgerblue'][:n_optim] # adapt cmap in function of nbr of clusters
    custom_cmap = ListedColormap(colors)

    for i in range(n_optim):
        ax = axs[i]
        gdf[gdf['cluster'] == i].plot(ax=ax, color=colors[i], markersize=0.1)
        ctx.add_basemap(ax, source=ctx.providers.SwissFederalGeoportal.NationalMapColor, crs=gdf.crs)
        ax.set_title(f'Cluster {i}')

    plt.savefig(f'sampled_pts_{n_optim}_clusters_norm_seperated.png')


"""
# Plot clusters added one at a time
for n_optim in range(3,7): 
    df = pd.read_csv(f'sampled_data_{n_optim}_clusters_norm.csv')
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['x'], df['y']), crs="EPSG:3035").to_crs(epsg=3857)

    # Create n_optim subplots, plotting the different clusters
    fig, axs = plt.subplots(nrows=int(np.ceil(n_optim/3)), ncols=3, figsize=(15, 10))
    axs = axs.flatten()
    colors =['teal', 'darkorange', 'purple', 'deeppink', 'limegreen', 'dodgerblue'][:n_optim] # adapt cmap in function of nbr of clusters
    custom_cmap = ListedColormap(colors)

    for i in range(n_optim):
        ax = axs[i]
        gdf_clusters = gdf[gdf['cluster'] <= i]
        gdf_clusters['cluster'] = gdf_clusters['cluster'].astype('category')
        gdf_clusters.plot(ax=ax, column='cluster', cmap=ListedColormap(colors[:i+1]), markersize=0.01, legend=True)
        ctx.add_basemap(ax, source=ctx.providers.SwissFederalGeoportal.NationalMapColor, crs=gdf_clusters.crs)
        ax.set_title(f'Cluster 0-{i}')

    plt.savefig(f'sampled_pts_{n_optim}_clusters_norm_add.png')
"""

"""
# Plot difference maps: wehere points changed clustering
for n_optim in range(3,6):

  df_base = pd.read_csv(f'sampled_data_{n_optim}_clusters_norm.csv')
  fig, axs = plt.subplots(1, 7-n_optim-1, figsize=(15, 5))

  for n_compare in range(n_optim+1, 7):

    df_compare = pd.read_csv(f'sampled_data_{n_compare}_clusters_norm.csv')
    df_diff = df_base.merge(df_compare, on=['x', 'y'], suffixes=('_3', '_5'))
    # Create a difference column (1 if cluster changed, 0 if unchanged)
    df_diff['changed'] = df_diff['cluster_3'] != df_diff['cluster_5']
    gdf_diff = gpd.GeoDataFrame(df_diff, 
                                geometry=gpd.points_from_xy(df_diff['x'], df_diff['y']), 
                                crs="EPSG:3035").to_crs(epsg=3857)
    try:
      ax = axs[n_compare-n_optim-1] 
    except:
      ax = axs
    gdf_diff.plot(ax=ax, column='changed', cmap='bwr', markersize=0.5, legend=True)
    ctx.add_basemap(ax=ax, source=ctx.providers.SwissFederalGeoportal.NationalMapColor, crs=gdf_diff.crs)
    ax.set_title(f"K={n_optim} vs K={n_compare}")
  
  plt.suptitle('Changes in Cluster Assignment')
  plt.savefig(f'cluster_diff_{n_optim}_norm.png')
"""



######################
# 7. Save endmembers
"""
bands = ['SRC_B2', 'SRC_B3','SRC_B4','SRC_B5','SRC_B6','SRC_B7','SRC_B8', 'SRC_B8A', 'SRC_B11', 'SRC_B12']

n_optim = 5
df = pd.read_csv(f'sampled_data_{n_optim}_clusters.csv')
summary = df.groupby('cluster')[bands].quantile([0.25, 0.50, 0.75]).reset_index().rename({'level_1':'percentile'}, axis=1)

summary.to_pickle('summarised_soil_samples.pkl')
"""



######################
# 8. Inference and soil map: apply K-means model to all pixels in CH

# Load the model
""" 
n_optim = 2
with open(f"kmeans_{n_optim}_clusters.pkl", "rb") as f:
    kmeans_loaded = pickle.load(f)


# Process one file at a time
data_files = os.listdir(download_dir)
for i, f in enumerate(data_files):
    print(f'Predicting for file {i}/{(len(data_files))}')
    ds = xr.open_zarr(os.path.join(download_dir, f))
    ds = ds.rio.write_crs(3035)

    # Resample to 10m resolution
    scale_factor = 2
    x_new = np.linspace(ds.x.values[0], ds.x.values[-1], len(ds.x.values) * scale_factor)
    y_new = np.linspace(ds.y.values[-1], ds.y.values[0], len(ds.y.values) * scale_factor)
    ds = ds.interp(x=x_new, y=y_new, method="nearest")
    print('Resampled')

    # Filter for CH
    swiss_borders_shp = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET.shp')
    swiss_borders = gpd.read_file(swiss_borders_shp).to_crs(3035).dissolve()
    ds = ds.rio.clip(swiss_borders.geometry)
    print('Clipped', ds.sizes)

    # Prepare data
    mask = (ds['MASK'] == 1).compute()
    print('Nbr of baresoil pixels', mask.sum().item())
    X = ds[['SRC_B2', 'SRC_B3', 'SRC_B4', 'SRC_B5', 'SRC_B6', 'SRC_B7', 
            'SRC_B8', 'SRC_B8A', 'SRC_B11', 'SRC_B12']].where(mask, drop=True)

    X_flat = X.to_array().stack(z=('x', 'y')).transpose('z', ...)  # flatten to 2D (n_samples, n_bands)
    X_flat = X_flat.fillna(-10000) / 10000  # Replace missing values and normalize
    X_values = X_flat.values  # Convert to NumPy array
    print(X_values.shape)

    # Split into batches to avoid memory overload
    batch_size = 100000  # Adjust based on memory availability
    n_samples = X_values.shape[0]

    predictions = np.empty(n_samples, dtype=int)  # Preallocate array for efficiency

    for j in range(0, n_samples, batch_size):
        batch = X_values[j:j+batch_size]
        predictions[j:j+batch_size] = kmeans_loaded.predict(batch)

    # Assign cluster labels back to the dataset
    spectra = pd.DataFrame(X_values, columns=X.data_vars) 
    coords = X_flat['z'].values 
    spectra['x'], spectra['y'] = [c[0] for c in coords], [c[1] for c in coords]  # Assign coordinates
    spectra['cluster'] = predictions
    
    spectra.to_csv(f'SRC_clustered_{n_optim}_{i}.csv')
"""

"""
######################## careful: --> TAKES TOO LONG/TOO MUCH MEMORY
# Plot soil map with cluster colors 
pattern = f'SRC_clustered_{n_optim}_*.csv'
SRC_preds_files = glob.glob(pattern)

preds = []
for f in SRC_preds_files:
    df = pd.read_csv(f)
    print(len(df))
    preds.append(df[['x', 'y', 'cluster']])

preds = pd.concat(preds, ignore_index=True)
gdf = gpd.GeoDataFrame(preds, geometry=gpd.points_from_xy(preds['x'], preds['y']), crs="EPSG:3035").to_crs(epsg=3857)
gdf['cluster'] = gdf['cluster'].astype('category')

fig, ax = plt.subplots(figsize=(8, 6))
custom_cmap = ListedColormap(['teal', 'orange'])  #'purple', 'limegreen'
gdf.plot(ax=ax, column='cluster', cmap=custom_cmap, markersize=0.3, legend=True, categorical=True)
ctx.add_basemap(ax, source=ctx.providers.SwissFederalGeoportal.NationalMapColor, crs=gdf.crs)
plt.title(f'Sampled Reflectances from SRC ({n_optim} clusters)')
plt.savefig(f'SRC_cluster_map_{n_optim}.png')

"""
