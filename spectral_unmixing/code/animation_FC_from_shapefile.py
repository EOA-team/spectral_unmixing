import geopandas as gpd
import pandas as pd
import os
from shapely.geometry import box
import xarray as xr
import joblib
import numpy as np
import rioxarray
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
from pathlib import Path
import sys
sys.path.insert(0, str(Path(os.path.dirname(os.path.realpath("__file__"))).parent))
from models import MODELS
import torch
import math
import warnings
warnings.filterwarnings("ignore")


def extract_bounds_year(file):
    """ 
    Get extent and year from S2 zarr file name
    """
    parts = file.split('_')
    minx = int(parts[1])
    maxx = minx + 1280
    maxy = int(parts[2])
    miny = maxy - 1280
    yr = int(parts[3][:4])

    return minx, miny, maxx, maxy, yr


def intersects_with_bbox(row, bbox_aoi):
  """ 
  Function to check intersection 
  """
  file_polygon = box(row['minx'], row['miny'], row['maxx'], row['maxy'])
  return bbox_aoi.intersects(file_polygon)


def find_cubes(shp, data_folder, years):
  """
  Find S2 cubes that fall within the shapefile extent

  :param shp: gpd.GeoDataFrame
  :param data_folder: where S2 data is stored
  :param years: list of integer yeears to filter the data by
  :return : list of zarr stores of S2 data
  """
  # Get shp bounds
  minx, miny, maxx, maxy = shp.total_bounds

  # Extract location info from cubes
  cubes = [f for f in os.listdir(data_folder) if f.endswith('zarr')]
  df_cubes = pd.DataFrame(cubes, columns=['file'])
  df_cubes[['minx', 'miny', 'maxx', 'maxy', 'yr']] = df_cubes['file'].apply(lambda x: pd.Series(extract_bounds_year(x)))
  df_cubes = df_cubes[df_cubes.yr.isin(years)]

  # Filter files by bbox
  bbox_aoi = box(minx, miny, maxx, maxy)

  filtered_files = df_cubes[df_cubes.apply(intersects_with_bbox, axis=1, bbox_aoi=bbox_aoi)].file.tolist()

  return filtered_files


def extract_time(product_uri_array):
  """
  Extract time from product_uri string
  """
  time_array = []
  for product_uri in product_uri_array:
    parts = product_uri.split('_')
    time_str = f"{parts[2][:4]}-{parts[2][4:6]}-{parts[2][6:8]}" 
    time = pd.to_datetime(time_str).to_numpy()
    time_array.append(time)
  return time_array


def has_all_65535(ds):
    return ((ds == 65535).all(dim=['lat', 'lon'])).to_array().sum()


def has_clouds(ds, cloud_thresh=0.1):
    cloud_condition = (ds.s2_mask == 1) & (ds.s2_SCL.isin([8, 9, 10]))
    return cloud_condition.sum(dim=['lat', 'lon'])/(len(ds.lat)*len(ds.lon)) > cloud_thresh


def has_shadows(ds, shadow_thresh=0.1):
    shadow_condition = (ds.s2_mask == 2) & (ds.s2_SCL == 3)
    return shadow_condition.sum(dim=['lat', 'lon'])/(len(ds.lat)*len(ds.lon)) > shadow_thresh


def has_snow(ds, snow_thresh=0.1):
    snow_condition = (ds.s2_mask == 3) & (ds.s2_SCL == 11)
    return snow_condition.sum(dim=['lat', 'lon'])/(len(ds.lat)*len(ds.lon)) > snow_thresh


def has_cirrus(ds, cirrus_thresh=1000):
    cirrus_mask = ds.s2_SCL == 10
    cirrus_b02_mean = ds.s2_B02.where(cirrus_mask).mean(dim=['lat', 'lon'])
    return cirrus_b02_mean > cirrus_thresh


def clean_dataset(ds, cloud_thresh=0.1, shadow_thresh=0.1, snow_thresh=0.1, cirrus_thresh=1000):
  """
  Drop dates with no data and clouds/snow/shadows
  """
  n_times = len(ds.time)

  # Remove cloudy or missing dates: any of the bands is all 65535
  dates_to_drop = [i for i, date in enumerate(ds.time.values) if has_all_65535(ds.isel(time=i))]
  mask_dates = np.ones(len(ds.time), dtype=bool)
  mask_dates[dates_to_drop] = False
  ds = ds.isel(time=mask_dates)

  # Remove too many clouds (mask=1), shadows (mask=2) or snow (mask=3)
  dates_to_drop = [i for i, date in enumerate(ds.time.values) if has_clouds(ds.isel(time=i), cloud_thresh)] + \
                [i for i, date in enumerate(ds.time.values) if has_shadows(ds.isel(time=i), shadow_thresh)] + \
                [i for i, date in enumerate(ds.time.values) if has_snow(ds.isel(time=i), snow_thresh)] +\
                [i for i, date in enumerate(ds.time.values) if has_cirrus(ds.isel(time=i), cirrus_thresh)]
  mask_dates = np.ones(len(ds.time), dtype=bool)
  print(f'Dropping {len(set(dates_to_drop))}/{n_times} dates') # flag if too many dates dropped?
  mask_dates[dates_to_drop] = False
  ds = ds.isel(time=mask_dates)

  return ds


def predict_nn_in_batches(model, X_tensor, device, batch_size=10000):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X_tensor.shape[0], batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            out = model(batch).cpu()
            preds.append(out)
    return torch.cat(preds).numpy()



# === Load shapefile and file list ===
area_shp_path = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/spectral_unmixing/data/Reckenholz.shp')
area_shp = gpd.read_file(area_shp_path).to_crs(32632)
data_folder = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH')
s2_files = find_cubes(area_shp,data_folder, [2019])


# === Prediction config ===
soil_group = 4
model_type = 'NN'
input_features = ['s2_B02','s2_B03','s2_B04','s2_B05','s2_B06','s2_B07','s2_B08','s2_B8A','s2_B11','s2_B12']
chunk_size = 10


# === Load S2 data ===
ds = xr.open_mfdataset([os.path.join(data_folder, f) for f in s2_files]).compute()
print('Loaded data...')


# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Prepare empty list for results ===
df_preds_list = []


# === Loop over time in chunks ===
time_coords = ds['time'].values
num_chunks = math.ceil(len(time_coords) / chunk_size)

for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, ds.sizes['time'])  # Avoid going out of bounds
    ds_chunk = ds.isel(time=slice(start, end))

    df = ds_chunk.to_dataframe().reset_index()
    df[df == 65535] = np.nan
    df_valid = df.dropna().copy()

    if df_valid.empty:
        continue

    X = (df_valid[input_features] / 10000).values

    # Prepare storage
    predictions_pv, predictions_npv, predictions_soil = [], [], []

    for iteration in range(1, 6):
        # Load models
        model_pv = joblib.load(f'../models/{model_type}_CLASS2_SOIL{soil_group}_ITER{iteration}.pkl')
        model_npv = joblib.load(f'../models/{model_type}_CLASS1_SOIL{soil_group}_ITER{iteration}.pkl')
        model_soil = joblib.load(f'../models/{model_type}_CLASS3_SOIL{soil_group}_ITER{iteration}.pkl')

        if model_type == 'NN':
            X_tensor = torch.FloatTensor(X)
            pv_pred = predict_nn_in_batches(model_pv, X_tensor, device)
            npv_pred = predict_nn_in_batches(model_npv, X_tensor, device)
            soil_pred = predict_nn_in_batches(model_soil, X_tensor, device)
            del X_tensor, model_pv, model_npv, model_soil
            torch.cuda.empty_cache()
        else:
            pv_pred = model_pv.predict(X)
            npv_pred = model_npv.predict(X)
            soil_pred = model_soil.predict(X)

        predictions_pv.append(pv_pred)
        predictions_npv.append(npv_pred)
        predictions_soil.append(soil_pred)

    # Average and store
    df_valid.loc[:, 'PV'] = np.mean(predictions_pv, axis=0)
    df_valid.loc[:, 'NPV'] = np.mean(predictions_npv, axis=0)
    df_valid.loc[:, 'Soil'] = np.mean(predictions_soil, axis=0)

    df_preds_list.append(df_valid)


# === Combine all chunks ===
df_all = pd.concat(df_preds_list, ignore_index=True)

# === Merge back with original data ===
df = ds.to_dataframe().reset_index()
df_merged = df_all.merge(df, on=['lat', 'lon', 'product_uri'], how='left')
df_merged = df_merged.drop(columns=[col for col in df_merged.columns if col.endswith('_x')])
df_merged = df_merged.rename(columns=lambda x: x.rstrip('_y'))

# === Convert back to xarray ===
df_merged.set_index(['lat', 'lon', 'product_uri'], inplace=True)
preds = df_merged.to_xarray()

# === Add time coords and save ===
time_coords = xr.apply_ufunc(extract_time, preds['product_uri'])
preds = preds.assign_coords(time=("product_uri", time_coords.data))
preds = preds.swap_dims({"product_uri": "time"}).reset_coords("product_uri", drop=False)

output_path = f'../data/Reckenholz_FC_{model_type}_SOIL{soil_group}.zarr'
preds.compute().to_zarr(output_path, consolidated=True, mode='w')

print('Saved')


 
##############
# Cleaning data, pre-processing

ds = xr.open_zarr(f'../data/Reckenholz_FC_{model_type}_SOIL{soil_group}.zarr').compute()

# Crop for area of interest
nonspatial_vars = ['mean_sensor_azimuth', 'mean_sensor_zenith', 'mean_solar_azimuth', 'mean_solar_zenith', 'product_uri']
ds_area = ds.drop_vars(nonspatial_vars).astype(np.float32).rio.write_crs("EPSG:32632").rename({'lon':'x', 'lat':'y'}).rio.clip(area_shp.geometry)

# Drop clouds, snow, missing data
ds_area = clean_dataset(ds_area.rename({'x':'lon', 'y':'lat'}), cloud_thresh=0.08, snow_thresh=0.1, shadow_thresh=0.1, cirrus_thresh=800)
ds_area = ds_area.rename({'lon':'x', 'lat':'y'})

# Crop for single field only
field_shp_path = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/spectral_unmixing/data/Reckenholz_field3.shp')
field_shp = gpd.read_file(field_shp_path).to_crs(32632)
ds_field = ds_area.rio.clip(field_shp.geometry)

# Normalise the PV, NPV, and soil fractions by the sum of the 3
ds_field['PV_norm'] = ds_field['PV'].clip(min=0)
ds_field['NPV_norm'] = ds_field['NPV'].clip(min=0)
ds_field['Soil_norm'] = ds_field['Soil'].clip(min=0)
ds_field['PV_norm'] = ds_field['PV_norm'] / (ds_field['PV_norm'] + ds_field['NPV_norm'] + ds_field['Soil_norm'])
ds_field['NPV_norm'] = ds_field['NPV_norm'] / (ds_field['PV_norm'] + ds_field['NPV_norm'] + ds_field['Soil_norm'])
ds_field['Soil_norm'] = ds_field['Soil_norm'] / (ds_field['PV_norm'] + ds_field['NPV_norm'] + ds_field['Soil_norm'])



##############
# Analyse

# Order ds by time
ds_field = ds_field.sortby('time')
ds_area = ds_area.sortby('time')

# compute the mean PV, NPV, and soil fractions for the field
mean_PV = ds_field.PV_norm.mean(dim=['x', 'y'])
mean_NPV = ds_field.NPV_norm.mean(dim=['x', 'y'])
mean_Soil = ds_field.Soil_norm.mean(dim=['x', 'y'])

# Plot those across time on the same plot
fig, ax = plt.subplots(figsize=(12, 5))
mean_PV.plot(ax=ax, label='PV', color='g')
mean_NPV.plot(ax=ax, label='NPV', color='goldenrod')
mean_Soil.plot(ax=ax, label='Soil', color='saddlebrown')
plt.xlabel('Date')
plt.ylabel('Fraction')
plt.title('Soil cover type')
plt.legend()
plt.savefig(f'fractions_{model_type}_SOIL{soil_group}.png')


##############
# Animation


# Setup the figure and axes
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(12, 9))
gs = GridSpec(2,2, figure=fig, width_ratios=[2,1], height_ratios=[1,1])

# Initialize the time series plot
time_series_ax = fig.add_subplot(gs[1, :])

mean_PV_scatter = time_series_ax.scatter([], [], color='g', s=10)
mean_NPV_scatter = time_series_ax.scatter([], [], color='goldenrod', s=10)
mean_Soil_scatter = time_series_ax.scatter([], [], color='saddlebrown', s=10)

mean_PV_line, = time_series_ax.plot([], [], color='g')
mean_NPV_line, = time_series_ax.plot([], [], color='goldenrod')
mean_Soil_line, = time_series_ax.plot([], [], color='saddlebrown')

time_series_ax.set_xlim(ds_area.time[0], ds_area.time[-1])
time_series_ax.set_ylim(0, 1)
time_series_ax.set_ylabel("Mean Fraction", fontsize=18)

# Initialize RGB subplot
ax_rgb = fig.add_subplot(gs[:, 0])

scale_factor = 1.0 / 10000.0 
r = ds_area['s2_B04']* scale_factor
g = ds_area['s2_B03']* scale_factor
b = ds_area['s2_B02']* scale_factor
rgb = xr.concat([r, g, b], dim='band').transpose('time', 'y', 'x', 'band')
rgb = rgb.where(~np.isnan(rgb), other=1.0)
brightness = 5


# Function to update the frame
def update(frame):
    ax_rgb.clear()
    ax_rgb.axis('off')
    
    # Plot RGB image
    x_coords = rgb[frame].coords['x'].values
    y_coords = rgb[frame].coords['y'].values
    ax_rgb.imshow(rgb[frame, :, :] * brightness, origin='lower', extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()])
    ax_rgb.set_title(f"RGB {str(ds_area.time.values[frame]).split('T')[0]}")

    # Add outline of field
    field_shp.plot(ax=ax_rgb, alpha=0.5, edgecolor='r', facecolor='None', linewidth=2)

    # Convert time to numeric for scatter plot
    time_numeric = mdates.date2num(ds_area.time.values[:frame+1])
    
    # Update scatter plots
    mean_PV_scatter.set_offsets(np.c_[time_numeric, mean_PV.values[:frame+1]])
    mean_NPV_scatter.set_offsets(np.c_[time_numeric, mean_NPV.values[:frame+1]])
    mean_Soil_scatter.set_offsets(np.c_[time_numeric, mean_Soil.values[:frame+1]])
    
    # Update line plots
    mean_PV_line.set_data(ds_area.time.values[:frame+1], mean_PV.values[:frame+1])
    mean_NPV_line.set_data(ds_area.time.values[:frame+1], mean_NPV.values[:frame+1])
    mean_Soil_line.set_data(ds_area.time.values[:frame+1], mean_Soil.values[:frame+1])

# Create the animation
anim = FuncAnimation(fig, update, frames=len(ds_area.time), interval=500, repeat=False)
FFwriter = animation.FFMpegWriter(fps=6)
anim.save(filename=f"Reckenholz_unmixing_{model_type}_SOIL{soil_group}.mp4", writer=FFwriter)

plt.tight_layout()
