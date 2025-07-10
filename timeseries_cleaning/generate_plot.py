import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from shapely.geometry import box
import xarray as xr
import numpy as np
import rioxarray
from pathlib import Path
import sys
sys.path.insert(0, os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/spectral_unmixing'))
from models import MODELS
import torch
import math
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
import matplotlib.animation as animation
import joblib
from collections import defaultdict
import json
import matplotlib.lines as mlines
from scipy.stats import zscore
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import trange
import datetime
from statsmodels.tsa.statespace.structural import UnobservedComponents
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split




def extract_ids(product_uri):
  """
  Extract tile_id and granule_id from product_uri string
  """
  parts = product_uri.split('_')
  tile_id = parts[5]  # e.g., T31TGM
  granule_id = parts[-1].split('.')[0]  # e.g., 20210529T141943
  return tile_id, granule_id


def open_cubes_conflicting(cubes):
  """
  Open and combine zarr files where there are conflicts along the itme dimensions due to multiple tiles/multiple acquisitions at a same timestamps. 
  Use product_uri to merge along time correctly

  :param cubes: list of zarr files
  :returns ds: combined zarr files in xr Dataset
  """
  grouped_datasets = defaultdict(list)

  # Load each Zarr file and organize it by (timestamp, tile_id, granule_id)
  for zarr_path in cubes:
      ds = xr.open_dataset(zarr_path, engine="zarr").compute()
      for i in range(ds.sizes['time']):
          ds_time_slice = ds.isel(time=i)
          timestamp = ds_time_slice['time'].values
          product_uri = ds_time_slice.product_uri.item()  # Assuming product_uri is scalar for each time slice
          tile_id, granule_id = extract_ids(product_uri)
          
          # Use a combination of timestamp, tile_id, and granule_id as the key
          key = (pd.Timestamp(timestamp), tile_id, granule_id)
          grouped_datasets[key].append(ds_time_slice)
  
  # Combine datasets with matching (timestamp, tile_id, granule_id)
  combined_datasets = []
  for i, ((timestamp, tile_id, granule_id), datasets) in enumerate(grouped_datasets.items()):
      if len(datasets) > 1:
          #datasets[5][['s2_B04', 's2_B03', 's2_B02']].rename({'lat':'y', 'lon':'x'}).rio.to_raster(f'ds5_{tile_id}_{granule_id}.tif')
          combined_ds = xr.combine_by_coords(datasets, combine_attrs='override')
          #combined_ds[['s2_B04', 's2_B03', 's2_B02']].rename({'lat':'y', 'lon':'x'}).rio.to_raster(f'{tile_id}_{granule_id}.tif')
          combined_ds[['mean_sensor_azimuth', 'mean_sensor_zenith','mean_solar_azimuth', 'mean_solar_zenith', 'product_uri']] = \
            datasets[0][['mean_sensor_azimuth', 'mean_sensor_zenith','mean_solar_azimuth', 'mean_solar_zenith', 'product_uri']]
      else:
        combined_ds = datasets[0]
      
      combined_datasets.append(combined_ds)
  
  
  ds = xr.concat(combined_datasets, dim="time", fill_value=65535)

  if ds.lat.values[0]<ds.lat.values[1]:
    ds = ds.sel(lat=slice(None, None, -1))

  return ds


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


def has_all_65535(ds):
    return ((ds == 65535).all(dim=['lat', 'lon'])).to_array().sum()


def has_clouds(ds, cloud_thresh=0.1):
    cloud_condition = (ds.s2_mask == 1) | (ds.s2_SCL.isin([8, 9, 10])) #
    return cloud_condition.sum(dim=['lat', 'lon'])/(len(ds.lat)*len(ds.lon)) > cloud_thresh


def has_shadows(ds, shadow_thresh=0.1):
    shadow_condition = (ds.s2_mask == 2) | (ds.s2_SCL == 3) # 
    return shadow_condition.sum(dim=['lat', 'lon'])/(len(ds.lat)*len(ds.lon)) > shadow_thresh


def has_snow(ds, snow_thresh=0.1):
    snow_condition = (ds.s2_mask == 3) | (ds.s2_SCL == 11) # 
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


def predict_nn_in_batches(model, X_tensor, device, batch_size=10000):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X_tensor.shape[0], batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            out = model(batch).cpu()
            preds.append(out)
    return torch.cat(preds).numpy()


def predict_fc(parcel_shp, s2_data_dir, yr, soil_group, model_type, chunk_size=10):
    
    s2_files = find_cubes(parcel_shp, s2_data_dir, [yr, yr-1])

    # Determine dominant soil group of parcel
    if soil_group is None:
        soil_cubes = []
        for f in s2_files:
            coords = f'{f.split("_")[1]}_{f.split("_")[2]}'
            soil = xr.open_zarr(os.path.join(soil_dir, f'SRC_{coords}.zarr')).compute()
            soil_cubes.append(soil)
        soil = xr.merge(soil_cubes, combine_attrs='override') # Ignore time dimesion, keep only one file per loc
        # Get most common soil in area
        valid_soil = soil['soil_group'].fillna(0).where(soil['soil_group'] != -10000, 0).values.flatten()
        if len(valid_soil[valid_soil>0]):
            soil_group = np.bincount(valid_soil[valid_soil>0].astype(int)).argmax()
            soil['soil_group'] = soil['soil_group'].astype('float32') # othewise clipping has porblems introducing nan when var is int
            soil = soil.rio.write_crs(32632).rio.clip(parcel_shp.geometry)
            # Try to get soil count in parcels (if data is available)
            valid_soil = soil['soil_group'].fillna(0).where(soil['soil_group'] != -10000, 0).values.flatten()
            if len(valid_soil[valid_soil>0]):
                soil_group = np.bincount(valid_soil[valid_soil>0].astype(int)).argmax() # ye dont want to count the class 0
            
            print('SOIL GROUP', soil_group)

    if soil_group is None:
        # could not determine soil group, will not bother to predict FC with a specific soil model
        return None, None, None

    # Open S2 data
    try: # No conflicting timestamps
        ds = xr.open_mfdataset([os.path.join(s2_data_dir, f) for f in s2_files], combine='by_coords').compute()
    except: # Use product_uri to merge
        ds = open_cubes_conflicting([os.path.join(s2_data_dir, f) for f in s2_files])

    ds = ds.drop_duplicates(dim='time', keep='first') # TO DO : prioritise the tiles in UTM 32

    #  Drop non-spatial variables (for clipping)
    nonspatial_vars = ['mean_sensor_azimuth', 'mean_sensor_zenith', 'mean_solar_azimuth', 'mean_solar_zenith', 'product_uri']
    product_uri = ds['product_uri']
    ds = ds.drop_vars(nonspatial_vars)

    # Clip to field 
    ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    ds_field = ds.rio.write_crs(32632).rio.clip(parcel_shp.geometry) 
    ds_field = ds_field.assign(product_uri=product_uri)

    input_features = ['s2_B02','s2_B03','s2_B04','s2_B05','s2_B06','s2_B07','s2_B08','s2_B8A','s2_B11','s2_B12']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_preds_list = []

    # === Loop over time in chunks ===
    time_coords = ds['time'].values
    num_chunks = math.ceil(len(time_coords) / chunk_size)

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, ds.sizes['time'])  # Avoid going out of bounds
        ds_chunk = ds_field.isel(time=slice(start, end))

        df = ds_chunk.to_dataframe().reset_index()
        df[df == 65535] = np.nan
        df = df[~(df[input_features] == 0).all(axis=1)] # areas that are outside of geom (after clip) get put to 0
        df_valid = df.dropna().copy()

        if df_valid.empty:
            continue

        X = (df_valid[input_features] / 10000).values

        # Prepare storage
        predictions_pv, predictions_npv, predictions_soil = [], [], []

        for iteration in range(1, 6):
            # Load models
            model_pv = joblib.load(f'../spectral_unmixing/models/{model_type}_CLASS2_SOIL{soil_group}_ITER{iteration}.pkl')
            model_npv = joblib.load(f'../spectral_unmixing/models/{model_type}_CLASS1_SOIL{soil_group}_ITER{iteration}.pkl')
            model_soil = joblib.load(f'../spectral_unmixing/models/{model_type}_CLASS3_SOIL{soil_group}_ITER{iteration}.pkl')

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
        
        df_valid.loc[:, 'PV_std'] = np.std(predictions_pv, axis=0)
        df_valid.loc[:, 'NPV_std'] = np.std(predictions_npv, axis=0)
        df_valid.loc[:, 'Soil_std'] = np.std(predictions_soil, axis=0)

        # Initialize prediction columns with NaNs
        df['PV'] = np.nan
        df['NPV'] = np.nan
        df['Soil'] = np.nan

        df['PV_std'] = np.nan
        df['NPV_std'] = np.nan
        df['Soil_std'] = np.nan

        # Fill prediction columns with predicted values at valid indices
        df.loc[df_valid.index, 'PV'] = df_valid['PV']
        df.loc[df_valid.index, 'NPV'] = df_valid['NPV']
        df.loc[df_valid.index, 'Soil'] = df_valid['Soil']

        df.loc[df_valid.index, 'PV_std'] = df_valid['PV_std']
        df.loc[df_valid.index, 'NPV_std'] = df_valid['NPV_std']
        df.loc[df_valid.index, 'Soil_std'] = df_valid['Soil_std']


        df_preds_list.append(df)#(df_valid)


    if not len(df_preds_list):
        return None, None, soil_group
    
    # === Combine all chunks ===
    df_all = pd.concat(df_preds_list, ignore_index=True)

    # === Merge back with original data ===
    df = ds_field.to_dataframe().reset_index()
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

    # Normalise the PV, NPV, and soil fractions by the sum of the 3
    preds['PV_norm'] = preds['PV'].clip(min=0)
    preds['NPV_norm'] = preds['NPV'].clip(min=0)
    preds['Soil_norm'] = preds['Soil'].clip(min=0)
    
    denom = preds['PV_norm'] + preds['NPV_norm'] + preds['Soil_norm']
    preds['PV_norm'] = preds['PV_norm'] / denom
    preds['NPV_norm'] = preds['NPV_norm'] / denom
    preds['Soil_norm'] = preds['Soil_norm'] / denom
    

    # Order ds by time
    preds = preds.sortby('time')
    ds = ds.sortby('time')

    return ds, preds, soil_group


def plot_timeseries(df_parcel, dates_raw, mean_PV, mean_NPV, mean_Soil, dates_clean, mean_PV_clean, mean_NPV_clean, mean_Soil_clean, save_path, smoothing=None):
    
    removed_PV = removed_NPV = removed_Soil = []

    if smoothing is not None:
        algo = smoothing.get('algorithm')
        kwargs = smoothing.get('kwargs', {})

        # Apply algorithm and track removed points
        mean_PV_clean, removed_PV = algo(mean_PV_clean, **kwargs)
        mean_NPV_clean, removed_NPV = algo(mean_NPV_clean, **kwargs)
        mean_Soil_clean, removed_Soil = algo(mean_Soil_clean, **kwargs)
        

    # Plot timeseries with activities as vertical lines
    fig, ax = plt.subplots(figsize=(20, 4))

    # Plot vertical dashed lines for each activity
    df_parcel = df_parcel.dropna(subset=['datum'])
    for _, row in df_parcel.iterrows():
        remove_residues = row['ernteresteeingearbeitet']
        color = 'red' if remove_residues else 'gray'
        # Mark the activity and its date
        ax.axvline(x=row['datum'], color=color, linestyle='--', alpha=0.7)
        ax.text(row['datum'], 0.6, row['activity'], rotation=90, verticalalignment='bottom', fontsize=10)

    # Scatter points for all original dates (raw)
    ax.scatter(dates_raw, mean_PV.values, color='g', label='Raw PV', alpha=0.5, s=15)
    ax.scatter(dates_raw, mean_NPV.values, color='goldenrod', label='Raw NPV', alpha=0.5, s=15)
    ax.scatter(dates_raw, mean_Soil.values, color='saddlebrown', label='Raw Soil', alpha=0.5, s=15)

    # Plot removed spikes
    if removed_PV:
        times_PV, vals_PV = zip(*removed_PV)
        ax.scatter(times_PV, vals_PV, color='g', marker='x', label='Removed PV', s=40)
    if removed_NPV:
        times_NPV, vals_NPV = zip(*removed_NPV)
        ax.scatter(times_NPV, vals_NPV, color='goldenrod', marker='x', label='Removed NPV', s=40)
    if removed_Soil:
        times_Soil, vals_Soil = zip(*removed_Soil)
        ax.scatter(times_Soil, vals_Soil, color='saddlebrown', marker='x', label='Removed Soil', s=40)

    # Lineplot for clean dates
    if len(dates_clean):
        ax.plot(dates_clean, mean_PV_clean.values, label='Cleaned PV', color='g', linewidth=2)
        ax.plot(dates_clean, mean_NPV_clean.values, label='Cleaned NPV', color='goldenrod', linewidth=2)
        ax.plot(dates_clean, mean_Soil_clean.values, label='Cleaned Soil', color='saddlebrown', linewidth=2)

    # Set plot formatting
    ax.set_ylim(0, 1) 
    plt.xlabel('Date')
    plt.ylabel('Fraction')
    plt.title('Predicted soil cover')
    ax.set_xlim(df_parcel['datum'].min()-pd.Timedelta(days=20), df_parcel['datum'].max()+pd.Timedelta(days=20))
    
    # Create a custom for "residues" line
    residue_line = mlines.Line2D([], [], color='red', linestyle='--', label='Remove residues')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(residue_line)
    labels.append("Remove residues")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout() 
    plt.savefig(save_path)

    return


def plot_timeseries_images(df_parcel, ds, dates_clean, dates_raw, preds, parcel_shp, buffer_plot, save_path, smoothing):
    
    # Shorten S2 timeseries to a single year
    last_year = pd.to_datetime(ds.time.values[-1]).year
    ds = ds.sel(time=ds.time.dt.year == last_year)
    preds = preds.sel(time=preds.time.dt.year == last_year)
    dates_clean = np.array([d for d in ds.time.values if d in dates_clean])
    dates_raw = np.array([d for d in ds.time.values if d in dates_raw])
    #df_parcel = df_parcel[pd.to_datetime(df_parcel['datum']).dt.year == last_year]
    df_parcel = df_parcel[(df_parcel['datum'] >= dates_clean.min()) & (df_parcel['datum'] <= dates_clean.max())]
    """
    # Shorten S2 timeseries to field calendar
    start_date = df_parcel['datum'].min()-pd.Timedelta(days=20)
    end_date = df_parcel['datum'].max()+pd.Timedelta(days=20)
    start_s2 = ds.sel(time=start_date, method='nearest').time.values
    end_s2 = ds.sel(time=end_date, method='nearest').time.values
    ds = ds.sel(time=slice(start_s2, end_s2))
    preds = preds.sel(time=slice(start_s2, end_s2))
    dates_clean = np.array([d for d in ds.time.values if d in dates_clean])
    dates_raw = np.array([d for d in ds.time.values if d in dates_raw])
    """
    # Get timeseries/RGB for shorten timeseries and cleaned dates
    mean_PV = preds.PV_norm.mean(dim=['lon', 'lat'])
    mean_NPV = preds.NPV_norm.mean(dim=['lon', 'lat'])
    mean_Soil = preds.Soil_norm.mean(dim=['lon', 'lat'])
    
    if len(dates_clean):
        mean_PV_clean = mean_PV.sel(time=dates_clean)
        mean_NPV_clean = mean_NPV.sel(time=dates_clean)
        mean_Soil_clean = mean_Soil.sel(time=dates_clean)

        # Spike detection
        removed_PV = removed_NPV = removed_Soil = []
        if smoothing is not None:
            algo = smoothing.get('algorithm')
            kwargs = smoothing.get('kwargs', {})

            # Apply algorithm and track removed points
            mean_PV_clean, removed_PV = algo(mean_PV_clean, **kwargs)
            mean_NPV_clean, removed_NPV = algo(mean_NPV_clean, **kwargs)
            mean_Soil_clean, removed_Soil = algo(mean_Soil_clean, **kwargs)
        

    # For plotting, keep a 100m padded around field bbox
    scale_factor = 1.0 / 10000.0 
    r = ds['s2_B04']* scale_factor
    g = ds['s2_B03']* scale_factor
    b = ds['s2_B02']* scale_factor
    rgb = xr.concat([r, g, b], dim='band').transpose('time', 'lat', 'lon', 'band')
    rgb = rgb.where(~np.isnan(rgb), other=1.0)
    brightness = 3

    rgb = rgb.sel(time=dates_clean)
    minx, miny, maxx, maxy = parcel_shp.total_bounds
    rgb = rgb.sel(lon=slice(minx-buffer_plot, maxx+buffer_plot), lat=slice(maxy+buffer_plot, miny-buffer_plot))


    # Prepare plot
    n_imgs = len(dates_clean)
    n_cols = 4
    n_rows = math.ceil(n_imgs / n_cols)

    fig = plt.figure(figsize=(5 * n_cols, 3 * (n_rows + 1)))
    gs = GridSpec(n_rows + 1, n_cols, figure=fig)

    # Plot RGB fields as subplots
    for i, date in enumerate(dates_clean):
        row = i // n_cols
        col = i % n_cols
        ax_rgb = fig.add_subplot(gs[row, col])

        # Plot RGB image
        x_coords = rgb[i].coords['lon'].values
        y_coords = rgb[i].coords['lat'].values
        ax_rgb.imshow(rgb[i, :, :] * brightness, origin='upper', extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()])
        ax_rgb.set_title(f"RGB {str(rgb.time.values[i]).split('T')[0]}")
        ax_rgb.axis('off')

        # Add outline of field
        parcel_shp.plot(ax=ax_rgb, alpha=0.5, edgecolor='r', facecolor='None', linewidth=2)


    # Add time series subplot spanning all columns on the last row
    ax_ts = fig.add_subplot(gs[n_rows, :])  # last row, all columns

    # Plot vertical dashed lines for each activity
    df_parcel = df_parcel.dropna(subset=['datum'])
    df_parcel = df_parcel[df_parcel.activity != 'Mulching']
    for _, row in df_parcel.iterrows():
        remove_residues = 0 #row['ernteresteeingearbeitet']
        color = 'red' if remove_residues else 'gray'
        # Mark the activity and its date
        ax_ts.axvline(x=row['datum'], color=color, linestyle='--', alpha=0.7)
        ax_ts.text(row['datum'], 0.5, row['activity'], rotation=90, verticalalignment='bottom', fontsize=10)


    # Scatter points for all original dates (raw)
    ax_ts.scatter(dates_raw, mean_PV.values, color='g', label='Raw PV', alpha=0.5, s=15)
    ax_ts.scatter(dates_raw, mean_NPV.values, color='goldenrod', label='Raw NPV', alpha=0.5, s=15)
    ax_ts.scatter(dates_raw, mean_Soil.values, color='saddlebrown', label='Raw Soil', alpha=0.5, s=15)


    # Lineplot for clean dates
    if len(dates_clean):
        ax_ts.plot(dates_clean, mean_PV_clean.values, label='Cleaned PV', color='g', marker='s', markersize=5)
        ax_ts.plot(dates_clean, mean_NPV_clean.values, label='Cleaned NPV', color='goldenrod', marker='s', markersize=5)
        ax_ts.plot(dates_clean, mean_Soil_clean.values, label='Cleaned Soil', color='saddlebrown', marker='s', markersize=5)

        # Plot removed spikes
        if removed_PV:
            times_PV, vals_PV = zip(*removed_PV)
            ax_ts.scatter(times_PV, vals_PV, color='g', marker='x', label='Removed PV', s=40)
        if removed_NPV:
            times_NPV, vals_NPV = zip(*removed_NPV)
            ax_ts.scatter(times_NPV, vals_NPV, color='goldenrod', marker='x', label='Removed NPV', s=40)
        if removed_Soil:
            times_Soil, vals_Soil = zip(*removed_Soil)
            ax_ts.scatter(times_Soil, vals_Soil, color='saddlebrown', marker='x', label='Removed Soil', s=40)


    # Create a custom for "residues" line
    residue_line = mlines.Line2D([], [], color='red', linestyle='--', label='Remove residues')
    handles, labels = ax_ts.get_legend_handles_labels()
    handles.append(residue_line)
    labels.append("Remove residues")
    ax_ts.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Set plot format
    ax_ts.set_ylim(0, 1) 
    ax_ts.set_xlabel('Date')
    ax_ts.set_ylabel('Fraction')
    ax_ts.set_title('Predicted soil cover', y=1.1)
    ax_ts.set_xlim(dates_clean[0]-pd.Timedelta(days=5), dates_clean[-1]+pd.Timedelta(days=5))

    plt.tight_layout() #
    plt.savefig(save_path)

    return


def plot_timeseries_images_gapfill(df_parcel, ds, dates_clean, dates_raw, preds, parcel_shp, buffer_plot, save_path, gapfilling_list):
  
    # Shorten S2 timeseries to a single year
    last_year = pd.to_datetime(ds.time.values[-1]).year
    ds = ds.sel(time=ds.time.dt.year == last_year)
    preds = preds.sel(time=preds.time.dt.year == last_year)
    dates_clean = np.array([d for d in ds.time.values if d in dates_clean])
    dates_raw = np.array([d for d in ds.time.values if d in dates_raw])
    df_parcel = df_parcel[pd.to_datetime(df_parcel['datum']).dt.year == last_year]
   
    # Get timeseries/RGB for shorten timeseries and cleaned dates
    mean_PV = preds.PV_norm.mean(dim=['lon', 'lat'])
    mean_NPV = preds.NPV_norm.mean(dim=['lon', 'lat'])
    mean_Soil = preds.Soil_norm.mean(dim=['lon', 'lat'])

    std_PV = preds.PV_std.mean(dim=['lon', 'lat'])
    std_NPV = preds.NPV_std.mean(dim=['lon', 'lat'])
    std_Soil = preds.Soil_std.mean(dim=['lon', 'lat'])
    
    if len(dates_clean):
        mean_PV_clean = mean_PV.sel(time=dates_clean)
        mean_NPV_clean = mean_NPV.sel(time=dates_clean)
        mean_Soil_clean = mean_Soil.sel(time=dates_clean)

        std_PV_clean = std_PV.sel(time=dates_clean)
        std_NPV_clean = std_NPV.sel(time=dates_clean)
        std_Soil_clean = std_Soil.sel(time=dates_clean)
    

    # For plotting, keep a 100m padded around field bbox
    scale_factor = 1.0 / 10000.0 
    r = ds['s2_B04']* scale_factor
    g = ds['s2_B03']* scale_factor
    b = ds['s2_B02']* scale_factor
    rgb = xr.concat([r, g, b], dim='band').transpose('time', 'lat', 'lon', 'band')
    rgb = rgb.where(~np.isnan(rgb), other=1.0)
    brightness = 3

    rgb = rgb.sel(time=dates_clean)
    minx, miny, maxx, maxy = parcel_shp.total_bounds
    rgb = rgb.sel(lon=slice(minx-buffer_plot, maxx+buffer_plot), lat=slice(maxy+buffer_plot, miny-buffer_plot))

    # Prepare plot
    n_imgs = len(dates_clean)
    n_cols = 4
    n_rows = math.ceil(n_imgs / n_cols)
    extra_rows = len(gapfilling_list)

    fig = plt.figure(figsize=(5 * n_cols, 3 * (n_rows + extra_rows + 1)))
    gs = GridSpec(n_rows + extra_rows + 1, n_cols, figure=fig)

    # Plot RGB fields as subplots
    for i, date in enumerate(dates_clean):
        row = i // n_cols
        col = i % n_cols
        ax_rgb = fig.add_subplot(gs[row, col])

        # Plot RGB image
        x_coords = rgb[i].coords['lon'].values
        y_coords = rgb[i].coords['lat'].values
        ax_rgb.imshow(rgb[i, :, :] * brightness, origin='upper', extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()])
        ax_rgb.set_title(f"RGB {str(rgb.time.values[i]).split('T')[0]}")
        ax_rgb.axis('off')

        # Add outline of field
        parcel_shp.plot(ax=ax_rgb, alpha=0.5, edgecolor='r', facecolor='None', linewidth=2)


    # Add time series subplot spanning all columns on the last row
    ax_ts = fig.add_subplot(gs[n_rows, :])  # last row, all columns

    # Plot vertical dashed lines for each activity
    df_parcel = df_parcel.dropna(subset=['datum'])
    for _, row in df_parcel.iterrows():
        remove_residues = row['ernteresteeingearbeitet']
        color = 'red' if remove_residues else 'gray'
        # Mark the activity and its date
        ax_ts.axvline(x=row['datum'], color=color, linestyle='--', alpha=0.7)
        ax_ts.text(row['datum'], 0.6, row['activity'], rotation=90, verticalalignment='bottom', fontsize=10)


    # Scatter points for all original dates (raw)
    ax_ts.scatter(dates_raw, mean_PV.values, color='g', label='Raw PV', alpha=0.5, s=15)
    ax_ts.scatter(dates_raw, mean_NPV.values, color='goldenrod', label='Raw NPV', alpha=0.5, s=15)
    ax_ts.scatter(dates_raw, mean_Soil.values, color='saddlebrown', label='Raw Soil', alpha=0.5, s=15)

    # Lineplot for clean dates
    if len(dates_clean):
        ax_ts.plot(dates_clean, mean_PV_clean.values, label='Cleaned PV', color='g', marker='s', markersize=5)
        ax_ts.plot(dates_clean, mean_NPV_clean.values, label='Cleaned NPV', color='goldenrod', marker='s', markersize=5)
        ax_ts.plot(dates_clean, mean_Soil_clean.values, label='Cleaned Soil', color='saddlebrown', marker='s', markersize=5)


    # Add gapfilled plots
    for i, gapfiller in enumerate(gapfilling_list):
        ax_gap = fig.add_subplot(gs[n_rows + 1 + i, :])

        algo = gapfiller.get('algorithm')
        kwargs = gapfiller.get('kwargs', {})
        name = algo.__name__

        if any(k in algo.__name__ for k in ('loess', 'spline', 'savgol', 'gpr_simple')):
            result = algo(
                dates=pd.to_datetime(dates_clean),
                pv_vals=mean_PV_clean.values,
                npv_vals=mean_NPV_clean.values,
                soil_vals=mean_Soil_clean.values,
                pred_dates=dates_raw,
                **kwargs
            )
            dates_smooth = result['dates']
            pv_smooth, pv_q05, pv_q95 = result['PV']
            npv_smooth, npv_q05, npv_q95 = result['NPV']
            soil_smooth, soil_q05, soil_q95 = result['Soil']
        elif any(k in algo.__name__ for k in ('kalman', 'gpr_combine_uncertainty')):
            result = algo(
                dates=pd.to_datetime(dates_clean),
                pv_vals=mean_PV_clean.values,
                npv_vals=mean_NPV_clean.values,
                soil_vals=mean_Soil_clean.values,
                pv_stds=std_PV_clean.values,
                npv_stds=std_NPV_clean.values,
                soil_stds=std_Soil_clean.values,
                pred_dates=dates_raw
            )
            dates_smooth = result['dates']
            pv_smooth, pv_q05, pv_q95 = result['PV'][0], result['PV'][0]-1.96*result['PV'][1], result['PV'][0]+1.96*result['PV'][1]
            npv_smooth, npv_q05, npv_q95 = result['NPV'][0], result['NPV'][0]-1.96*result['NPV'][1], result['NPV'][0]+1.96*result['NPV'][1]
            soil_smooth, soil_q05, soil_q95 = result['Soil'][0], result['Soil'][0]-1.96*result['Soil'][1], result['Soil'][0]+1.96*result['Soil'][1]
        else:
            continue  # skip unknown

        # Scatter points for all original dates (raw)
        ax_gap.scatter(dates_raw, mean_PV.values, color='g', label='Raw PV', alpha=0.5, s=15)
        ax_gap.scatter(dates_raw, mean_NPV.values, color='goldenrod', label='Raw NPV', alpha=0.5, s=15)
        ax_gap.scatter(dates_raw, mean_Soil.values, color='saddlebrown', label='Raw Soil', alpha=0.5, s=15)

        # Lineplot for clean dates
        if len(dates_clean):
            ax_gap.scatter(dates_clean, mean_PV_clean.values, label='Cleaned PV', color='g', marker='s', s=15)
            ax_gap.scatter(dates_clean, mean_NPV_clean.values, label='Cleaned NPV', color='goldenrod', marker='s', s=15)
            ax_gap.scatter(dates_clean, mean_Soil_clean.values, label='Cleaned Soil', color='saddlebrown', marker='s', s=15)

        # Plot Gapfilled
        ax_gap.set_title(f"{name} Gapfilling")
        ax_gap.plot(dates_smooth, pv_smooth, color='g', label='Gapfilled PV', linewidth=2)
        ax_gap.plot(dates_smooth, npv_smooth, color='goldenrod', label='Gapfilled NPV', linewidth=2)
        ax_gap.plot(dates_smooth, soil_smooth, color='saddlebrown', label='Gapfilled Soil', linewidth=2)

        ax_gap.fill_between(dates_smooth, pv_q05, pv_q95, color='g', alpha=0.2)
        ax_gap.fill_between(dates_smooth, npv_q05, npv_q95, color='goldenrod', alpha=0.2)
        ax_gap.fill_between(dates_smooth, soil_q05, soil_q95, color='saddlebrown', alpha=0.2)

        ax_gap.set_ylim(0, 1)
        #ax_gap.set_xlim(df_parcel['datum'].min()-pd.Timedelta(days=20), df_parcel['datum'].max()+pd.Timedelta(days=20))
        ax_gap.set_ylabel('Fraction')
        ax_gap.legend(loc='upper left')


    # Create a custom legend for "residues" line
    residue_line = mlines.Line2D([], [], color='red', linestyle='--', label='Remove residues')
    handles, labels = ax_ts.get_legend_handles_labels()
    handles.append(residue_line)
    labels.append("Remove residues")
    ax_ts.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout() #
    plt.savefig(save_path)

    return





###########
# GAP FILLING FUNCTIONS

def spline(dates, pv_vals, npv_vals, soil_vals, pred_dates=None, s=0.001):
    """
    Fit spline to PV, NPV, and Soil time series, normalize the outputs, and return the smoothed means.
    Quantiles are filled with 0 since no bootstrapping is done.
    
    Args:
        dates: list or array of datetime.date
        pv_vals, npv_vals, soil_vals: arrays of same length as dates
        pred_dates: list of datetime.date to predict on. If None, use evenly spaced points between min and max date.
        s: smoothing factor for the spline
        
    Returns:
        Dictionary with dates and (mean, q05, q95) tuples for PV, NPV, and Soil (quantiles are zero).
    """
    # Convert dates to ordinal
    dates_ord = np.array([d.toordinal() for d in dates])
    
    if pred_dates is None:
        pred_times = np.linspace(dates_ord.min(), dates_ord.max(), len(dates))
        pred_dates_out = np.array([datetime.date.fromordinal(int(p)) for p in pred_times])
    else:
        pred_times = np.array([d.toordinal() for d in pd.to_datetime(pred_dates)])
        pred_dates_out = np.array(pred_dates)
    
    # Fit splines
    pv_spline = UnivariateSpline(dates_ord, pv_vals, s=s)
    npv_spline = UnivariateSpline(dates_ord, npv_vals, s=s)
    soil_spline = UnivariateSpline(dates_ord, soil_vals, s=s)
    
    # Evaluate splines on prediction grid
    pv_interp = pv_spline(pred_times)
    npv_interp = npv_spline(pred_times)
    soil_interp = soil_spline(pred_times)
    
    # Normalize
    total = pv_interp + npv_interp + soil_interp
    pv_norm = pv_interp / total
    npv_norm = npv_interp / total
    soil_norm = soil_interp / total

    # Return mean with 0s for quantiles
    return {
        'dates': pred_dates_out,
        'PV': (pv_norm, np.zeros_like(pv_norm), np.zeros_like(pv_norm)),
        'NPV': (npv_norm, np.zeros_like(npv_norm), np.zeros_like(npv_norm)),
        'Soil': (soil_norm, np.zeros_like(soil_norm), np.zeros_like(soil_norm))
    }


def loess(dates, pv_vals, npv_vals, soil_vals, pred_dates=None, frac=0.3):
    """
    Apply LOESS smoothing to PV, NPV, and Soil time series, normalize the outputs, and return smoothed means.
    Quantiles are filled with 0 since no bootstrapping is done.

    Args:
        dates: list or array of datetime.date
        pv_vals, npv_vals, soil_vals: arrays of same length as dates
        pred_dates: list of datetime.date to predict on. If None, use evenly spaced points between min and max date.
        frac: LOESS span parameter (proportion of data used in local regression)

    Returns:
        Dictionary with dates and (mean, q05, q95) tuples for PV, NPV, and Soil (quantiles are zero).
    """
    dates_ord = np.array([d.toordinal() for d in dates])

    if pred_dates is None:
        pred_times = np.linspace(dates_ord.min(), dates_ord.max(), len(dates))
        pred_dates_out = np.array([datetime.date.fromordinal(int(p)) for p in pred_times])
    else:
        pred_times = np.array([d.toordinal() for d in pd.to_datetime(pred_dates)])
        pred_dates_out = np.array(pred_dates)

    # LOESS smoothing (fit to original dates)
    pv_loess = lowess(pv_vals, dates_ord, frac=frac, return_sorted=False)
    npv_loess = lowess(npv_vals, dates_ord, frac=frac, return_sorted=False)
    soil_loess = lowess(soil_vals, dates_ord, frac=frac, return_sorted=False)

    # Interpolate LOESS-smoothed results to prediction dates
    pv_interp = np.interp(pred_times, dates_ord, pv_loess)
    npv_interp = np.interp(pred_times, dates_ord, npv_loess)
    soil_interp = np.interp(pred_times, dates_ord, soil_loess)

    # Normalize
    total = pv_interp + npv_interp + soil_interp
    pv_norm = pv_interp / total
    npv_norm = npv_interp / total
    soil_norm = soil_interp / total

    return {
        'dates': pred_dates_out,
        'PV': (pv_norm, np.zeros_like(pv_norm), np.zeros_like(pv_norm)),
        'NPV': (npv_norm, np.zeros_like(npv_norm), np.zeros_like(npv_norm)),
        'Soil': (soil_norm, np.zeros_like(soil_norm), np.zeros_like(soil_norm))
    }


def savgol(dates, pv_vals, npv_vals, soil_vals, pred_dates=None, window_length=7, polyorder=2):
    """
    Apply Savitzky-Golay filter to PV, NPV, and Soil time series, normalize the outputs, and return smoothed means.
    Quantiles are filled with 0 since no bootstrapping is done.

    Args:
        dates: list or array of datetime.date
        pv_vals, npv_vals, soil_vals: arrays of same length as dates
        pred_dates: list of datetime.date to predict on. If None, use original dates.
        window_length: length of the filter window (must be odd and <= len(dates))
        polyorder: order of the polynomial to fit within each window

    Returns:
        Dictionary with dates and (mean, q05, q95) tuples for PV, NPV, and Soil (quantiles are zero).
    """
    # Convert dates to ordinal
    dates_ord = np.array([d.toordinal() for d in dates])

    if pred_dates is None:
        pred_times = dates_ord
        pred_dates_out = np.array(dates)
    else:
        pred_times = np.array([d.toordinal() for d in pd.to_datetime(pred_dates)])
        pred_dates_out = np.array(pred_dates)

    # Ensure window length is valid
    if window_length > len(dates):
        window_length = len(dates) if len(dates) % 2 == 1 else len(dates) - 1
    if window_length < polyorder + 2:
        window_length = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3

    # Apply Savitzky-Golay filter
    pv_smooth = savgol_filter(pv_vals, window_length=window_length, polyorder=polyorder, mode='interp')
    npv_smooth = savgol_filter(npv_vals, window_length=window_length, polyorder=polyorder, mode='interp')
    soil_smooth = savgol_filter(soil_vals, window_length=window_length, polyorder=polyorder, mode='interp')

    # Interpolate to prediction grid if needed
    pv_interp = np.interp(pred_times, dates_ord, pv_smooth)
    npv_interp = np.interp(pred_times, dates_ord, npv_smooth)
    soil_interp = np.interp(pred_times, dates_ord, soil_smooth)

    # Normalize
    total = pv_interp + npv_interp + soil_interp
    pv_norm = pv_interp / total
    npv_norm = npv_interp / total
    soil_norm = soil_interp / total

    return {
        'dates': pred_dates_out,
        'PV': (pv_norm, np.zeros_like(pv_norm), np.zeros_like(pv_norm)),
        'NPV': (npv_norm, np.zeros_like(npv_norm), np.zeros_like(npv_norm)),
        'Soil': (soil_norm, np.zeros_like(soil_norm), np.zeros_like(soil_norm))
    }


def gpr_simple(dates, pv_vals, npv_vals, soil_vals, pred_dates=None, kernel=None, alpha=1e-6,  random_state=42, validate=True):
    """
    Apply Gaussian Process Regression to PV, NPV, and Soil time series, normalize outputs, and return mean + 5–95% quantiles.

    Args:
        dates: list or array of datetime.date
        pv_vals, npv_vals, soil_vals: arrays of same length as dates
        pred_dates: list of datetime.date to predict on. If None, use evenly spaced points between min and max date
        kernel: scikit-learn kernel (optional)
        alpha: noise level

    Returns:
        Dictionary with dates and (mean, q05, q95) tuples for PV, NPV, and Soil
    """
    dates_ord = np.array([d.toordinal() for d in dates]).reshape(-1, 1)
    doy = np.array([d.timetuple().tm_yday for d in dates]).reshape(-1, 1)

    if pred_dates is None:
        pred_times = np.linspace(dates_ord.min(), dates_ord.max(), len(dates))
        pred_dates_out = [datetime.date.fromordinal(int(p)) for p in pred_times]
    else:
        pred_times = np.array([d.toordinal() for d in pd.to_datetime(pred_dates)])
        pred_dates_out = list(pred_dates)
    pred_times = pred_times.reshape(-1, 1)
    pred_doy = np.array([pd.Timestamp(d).timetuple().tm_yday for d in pred_dates_out]).reshape(-1, 1)

    # Default kernel if none provided
    if kernel is None:
        kernel = RBF(length_scale=20.0, length_scale_bounds=(1.0, 100.0)) + WhiteKernel(noise_level=alpha)


    def fit_predict(y_vals, name='Component'):
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=10, random_state=random_state)
        gp.fit(doy, y_vals)
        mean, std = gp.predict(pred_doy, return_std=True)
        
        # Optional validation
        if validate:
            X_train, X_test, y_train, y_test = train_test_split(doy, y_vals, test_size=0.2, random_state=42)
            gp_val = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=10, random_state=random_state)
            gp_val.fit(X_train, y_train)
            y_pred = gp_val.predict(X_test)
            rmse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"{name} validation RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return mean, std

    # Fit GPR for each class
    pv_mean, pv_std = fit_predict(pv_vals, 'PV')
    npv_mean, npv_std = fit_predict(npv_vals, 'NPV')
    soil_mean, soil_std = fit_predict(soil_vals, 'Soil')

    # Normalize
    total = pv_mean + npv_mean + soil_mean
    pv_norm = pv_mean / total
    npv_norm = npv_mean / total
    soil_norm = soil_mean / total

    # Quantiles (mean ± 2 std ≈ 95% CI for normal distribution)
    def get_bounds(mean, std):
        return mean - 1.96 * std, mean + 1.96 * std

    pv_q05, pv_q95 = get_bounds(pv_norm, pv_std / total)
    npv_q05, npv_q95 = get_bounds(npv_norm, npv_std / total)
    soil_q05, soil_q95 = get_bounds(soil_norm, soil_std / total)

    return {
        'dates': pred_dates_out,
        'PV': (pv_norm, pv_q05, pv_q95),
        'NPV': (npv_norm, npv_q05, npv_q95),
        'Soil': (soil_norm, soil_q05, soil_q95)
    }


def gpr_combine_uncertainty(dates, pv_vals, npv_vals, soil_vals, pv_stds=None, npv_stds=None, soil_stds=None, pred_dates=None, kernel=None, alpha=1e-6, random_state=42):
    """
    Apply Gaussian Process Regression to PV, NPV, and Soil time series, normalize outputs, and return mean + 5–95% quantiles.

    Args:
        dates: list or array of datetime.date
        pv_vals, npv_vals, soil_vals: arrays of same length as dates
        pred_dates: list of datetime.date to predict on. If None, use evenly spaced points between min and max date
        kernel: scikit-learn kernel (optional)
        alpha: noise level

    Returns:
        Dictionary with dates and (mean, q05, q95) tuples for PV, NPV, and Soil
    """
    dates_ord = np.array([d.toordinal() for d in dates]).reshape(-1, 1)
    doy = np.array([d.timetuple().tm_yday for d in dates]).reshape(-1, 1)

    if pred_dates is None:
        pred_times = np.linspace(dates_ord.min(), dates_ord.max(), len(dates))
        pred_dates_out = [datetime.date.fromordinal(int(p)) for p in pred_times]
    else:
        pred_times = np.array([d.toordinal() for d in pd.to_datetime(pred_dates)])
        pred_dates_out = list(pred_dates)
    pred_times = pred_times.reshape(-1, 1)
    pred_doy = np.array([pd.Timestamp(d).timetuple().tm_yday for d in pred_dates_out]).reshape(-1, 1)

    # Default kernel if none provided
    if kernel is None:
        kernel = RBF(length_scale=20.0) + WhiteKernel(noise_level=alpha)

    def fit_predict(y_vals, stds):
        if stds is not None:
            alpha_vals = stds**2  # convert std to variance
        else:
            alpha_vals = alpha_default
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha_vals, normalize_y=True, n_restarts_optimizer=10, random_state=42)
        gp.fit(doy, y_vals)
        mean, std = gp.predict(pred_doy, return_std=True)
        return mean, std

    # Fit GPR for each class
    pv_mean, pv_std = fit_predict(pv_vals, pv_stds)
    npv_mean, npv_std = fit_predict(npv_vals, npv_stds)
    soil_mean, soil_std = fit_predict(soil_vals, soil_stds)

    # Normalize
    total = pv_mean + npv_mean + soil_mean
    pv_norm = pv_mean / total
    npv_norm = npv_mean / total
    soil_norm = soil_mean / total

    # Quantiles (mean ± 2 std ≈ 95% CI for normal distribution)
    def get_bounds(mean, std):
        return mean - 1.96 * std, mean + 1.96 * std

    pv_q05, pv_q95 = get_bounds(pv_norm, pv_std / total)
    npv_q05, npv_q95 = get_bounds(npv_norm, npv_std / total)
    soil_q05, soil_q95 = get_bounds(soil_norm, soil_std / total)

    return {
        'dates': pred_dates_out,
        'PV': (pv_norm, pv_q05, pv_q95),
        'NPV': (npv_norm, npv_q05, npv_q95),
        'Soil': (soil_norm, soil_q05, soil_q95)
    }


def loess_bootstrap_normalized(dates, pv_vals, npv_vals, soil_vals, pred_dates=None, frac=0.3, n_bootstrap=200):
    dates_ord = np.array([d.toordinal() for d in dates])

    if pred_dates is None:
        pred_times = np.linspace(dates_ord.min(), dates_ord.max(), len(dates))
        pred_dates_out = np.array([datetime.date.fromordinal(int(p)) for p in pred_times])
    else:
        pred_times = np.array([d.toordinal() for d in pd.to_datetime(pred_dates)])
        pred_dates_out = np.array(pred_dates)

    pv_preds = []
    npv_preds = []
    soil_preds = []

    for _ in trange(n_bootstrap):
        # Bootstrap sampling
        idx = np.random.choice(len(dates), len(dates), replace=True)

        # Apply LOESS smoothing to bootstrapped sample
        pv_fit = lowess(pv_vals[idx], dates_ord[idx], frac=frac, return_sorted=False)
        npv_fit = lowess(npv_vals[idx], dates_ord[idx], frac=frac, return_sorted=False)
        soil_fit = lowess(soil_vals[idx], dates_ord[idx], frac=frac, return_sorted=False)

        # Interpolate all to common prediction grid
        pv_interp = np.interp(pred_times, dates_ord, pv_fit)
        npv_interp = np.interp(pred_times, dates_ord, npv_fit)
        soil_interp = np.interp(pred_times, dates_ord, soil_fit)
    
        # Normalize so PV + NPV + Soil = 1
        total = pv_interp + npv_interp + soil_interp
        pv_preds.append(pv_interp / total)
        npv_preds.append(npv_interp / total)
        soil_preds.append(soil_interp / total)
     

    # Convert to arrays
    pv_preds = np.array(pv_preds)
    npv_preds = np.array(npv_preds)
    soil_preds = np.array(soil_preds)

    # Compute mean + percentiles
    pv_mean = pv_preds.mean(axis=0)
    npv_mean = npv_preds.mean(axis=0)
    soil_mean = soil_preds.mean(axis=0)

    pv_q05 = np.percentile(pv_preds, 5, axis=0)
    pv_q95 = np.percentile(pv_preds, 95, axis=0)
    npv_q05 = np.percentile(npv_preds, 5, axis=0)
    npv_q95 = np.percentile(npv_preds, 95, axis=0)
    soil_q05 = np.percentile(soil_preds, 5, axis=0)
    soil_q95 = np.percentile(soil_preds, 95, axis=0)

    return {
        'dates': pred_dates_out,
        'PV': (pv_mean, pv_q05, pv_q95),
        'NPV': (npv_mean, npv_q05, npv_q95),
        'Soil': (soil_mean, soil_q05, soil_q95)
    }


def kalman_combine_uncertainty(dates, pv_vals, npv_vals, soil_vals, pv_stds, npv_stds, soil_stds, pred_dates):
    """ 
    After smoothing we add the uncertainty from the enseble models post hoc. The ensemble uncertainty does not affect the smoothing
    Obtain combined uncertainty (Kalman + ensemble) at every prediction date
    """
    # Regular time index (daily)
    full_time = pd.date_range(start=pred_dates.min(), end=pred_dates.max(), freq='D')

    pv_series = pd.Series(pv_vals, index=dates)
    npv_series = pd.Series(npv_vals, index=dates)
    soil_series = pd.Series(soil_vals, index=dates)

    pv_regular = pv_series.reindex(full_time)
    npv_regular = npv_series.reindex(full_time)
    soil_regular = soil_series.reindex(full_time)

    # local level model as simple example
    model_pv = UnobservedComponents(pv_regular, level='lltrend').fit(disp=False)
    model_npv = UnobservedComponents(npv_regular, level='lltrend').fit(disp=False)
    model_soil = UnobservedComponents(soil_regular, level='lltrend').fit(disp=False)

    # Get smoothed (gap-filled) estimates
    pv_filled = model_pv.smoothed_state[0]
    npv_filled = model_npv.smoothed_state[0]
    soil_filled = model_soil.smoothed_state[0]

    # Kalman Uncertainty
    std_pv_kalman = np.sqrt(model_pv.smoothed_state_cov[0, 0, :])
    std_npv_kalman = np.sqrt(model_npv.smoothed_state_cov[0, 0, :])
    std_soil_kalman = np.sqrt(model_soil.smoothed_state_cov[0, 0, :])

    pv_filled = xr.DataArray(pv_filled, coords=[full_time], dims=['time'])
    npv_filled = xr.DataArray(npv_filled, coords=[full_time], dims=['time'])
    soil_filled = xr.DataArray(soil_filled, coords=[full_time], dims=['time'])
    std_pv_kalman = xr.DataArray(std_pv_kalman, coords=[full_time], dims=['time'])
    std_npv_kalman = xr.DataArray(std_npv_kalman, coords=[full_time], dims=['time'])
    std_soil_kalman = xr.DataArray(std_soil_kalman, coords=[full_time], dims=['time'])

    # Ensemble (FC models) uncertainty
    pv_ens_std = xr.DataArray(pv_stds, coords=[dates], dims=['time']).reindex(time=full_time)
    npv_ens_std = xr.DataArray(npv_stds, coords=[dates], dims=['time']).reindex(time=full_time)
    soil_ens_std = xr.DataArray(soil_stds, coords=[dates], dims=['time']).reindex(time=full_time)
    pv_ens_std = pv_ens_std.fillna(0) #no ensemble info on gap-filled days
    npv_ens_std = npv_ens_std.fillna(0)
    soil_ens_std = soil_ens_std.fillna(0)

    # Combine variances
    pv_std_total = np.sqrt(std_pv_kalman**2 + pv_ens_std**2)
    npv_std_total = np.sqrt(std_npv_kalman**2 + npv_ens_std**2)
    soil_std_total = np.sqrt(std_soil_kalman**2 + soil_ens_std**2)

    # Stack and normalize
    total = pv_filled + npv_filled + soil_filled
    pv_filled /= total
    npv_filled /= total
    soil_filled /= total

    return {
        'dates': pred_dates,
        'PV': (pv_filled.sel(time=pred_dates), pv_std_total.sel(time=pred_dates)),
        'NPV': (npv_filled.sel(time=pred_dates), npv_std_total.sel(time=pred_dates)),
        'Soil': (soil_filled.sel(time=pred_dates), soil_std_total.sel(time=pred_dates))
    }


def kalman_with_time_varying_obs_error_lltrend(values, stds, dates, pred_dates):
    full_time = pd.date_range(start=pred_dates.min(), end=pred_dates.max(), freq='D')
    
    # Reindex to daily regular series
    value_series = pd.Series(values, index=dates).reindex(full_time)
    std_series = pd.Series(stds, index=dates).reindex(full_time).fillna(np.inf)  # np.inf -> missing obs

    y = value_series.values
    R_t = std_series.values ** 2  # Observation variance
    T = len(full_time)

    # Kalman filter setup for Local Linear Trend model (2D state: level and trend)
    x_filtered = np.zeros((T, 2))  # Store state vectors
    P_filtered = np.zeros((T, 2, 2))  # Store covariance matrices

    # Initial state estimate
    init_obs = y[~np.isnan(y)]
    x_est = np.array([init_obs[0] if len(init_obs) > 0 else 0.0, 0.0])  # [level, trend]
    P_est = np.eye(2) * 1.0

    # Model matrices
    F = np.array([[1, 1],
                  [0, 1]])  # state transition
    H = np.array([[1, 0]])  # observation model

    Q = np.array([[1e-6, 0],
                  [0, 1e-6]])  # process noise covariance (tune if needed)


    for t in range(T):
        # Predict
        x_pred = F @ x_est
        P_pred = F @ P_est @ F.T + Q

        if np.isfinite(R_t[t]) and np.isfinite(y[t]):
            # Observation available
            S = H @ P_pred @ H.T + R_t[t]
            K = P_pred @ H.T / S  # Kalman gain (2x1)
            residual = y[t] - (H @ x_pred)

            x_est = x_pred + K.flatten() * residual
            P_est = (np.eye(2) - K @ H) @ P_pred
        else:
            # Missing observation
            x_est = x_pred
            P_est = P_pred

        x_filtered[t] = x_est
        P_filtered[t] = P_est

    smoothed_state = x_filtered[:, 0]  # level component
    smoothed_std = np.sqrt(P_filtered[:, 0, 0])  # variance of level

    # Wrap in DataArrays
    da_smoothed = xr.DataArray(smoothed_state, coords=[full_time], dims=['time'])
    da_std = xr.DataArray(smoothed_std, coords=[full_time], dims=['time'])

    return da_smoothed.sel(time=pred_dates), da_std.sel(time=pred_dates)


def kalman_with_time_varying_obs_randomwalk(values, stds, dates, pred_dates):
    full_time = pd.date_range(start=pred_dates.min(), end=pred_dates.max(), freq='D')
    
    # Reindex to daily regular series
    value_series = pd.Series(values, index=dates).reindex(full_time)
    std_series = pd.Series(stds, index=dates).reindex(full_time).fillna(np.inf)  # np.inf -> missing obs

    observations = value_series.values
    R_t = std_series.values ** 2  # Observation variance
    T = len(full_time)

    # Kalman filter setup for 1D state
    x_filtered = np.zeros(T)
    P_filtered = np.zeros(T)

    # Initial state
    x_est = observations[~np.isnan(observations)][0] if np.any(~np.isnan(observations)) else 0.0
    P_est = 1.0  # Initial state uncertainty

    Q = 1e-5  # Process noise variance (tune this if needed)

    for t in range(T):
        # Prediction
        x_pred = x_est
        P_pred = P_est + Q

        if np.isfinite(R_t[t]) and np.isfinite(observations[t]):
            # Kalman Gain
            K = P_pred / (P_pred + R_t[t])

            # Update
            x_est = x_pred + K * (observations[t] - x_pred)
            P_est = (1 - K) * P_pred
        else:
            # No observation available: prediction becomes estimate
            x_est = x_pred
            P_est = P_pred

        x_filtered[t] = x_est
        P_filtered[t] = P_est

    smoothed_state = x_filtered
    smoothed_std = np.sqrt(P_filtered)

    # Wrap in DataArrays
    da_smoothed = xr.DataArray(smoothed_state, coords=[full_time], dims=['time'])
    da_std = xr.DataArray(smoothed_std, coords=[full_time], dims=['time'])

    return da_smoothed.sel(time=pred_dates), da_std.sel(time=pred_dates)


def kalman_with_obs_uncertainty(dates, pv_vals, npv_vals, soil_vals, pv_stds, npv_stds, soil_stds, pred_dates):

    pv_filled, pv_std = kalman_with_time_varying_obs_error_lltrend(pv_vals, pv_stds, dates, pred_dates)
    npv_filled, npv_std = kalman_with_time_varying_obs_error_lltrend(npv_vals, npv_stds, dates, pred_dates)
    soil_filled, soil_std = kalman_with_time_varying_obs_error_lltrend(soil_vals, soil_stds, dates, pred_dates)

    # Clip
    pv_filled = pv_filled.clip(min=0, max=1)
    npv_filled = npv_filled.clip(min=0, max=1)
    soil_filled = soil_filled.clip(min=0, max=1)

    # Normalize the components
    total = pv_filled + npv_filled + soil_filled
    pv_filled /= total
    npv_filled /= total
    soil_filled /= total

    return {
        'dates': pred_dates,
        'PV': (pv_filled, pv_std),
        'NPV': (npv_filled, npv_std),
        'Soil': (soil_filled, soil_std),
    }




########################
# PATHS TO DATA

# Farm data
hauptkultur_csv = os.path.expanduser('~/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.2_AUI_Monitoring_protected/Georeferenzierung/Hauptkulturen_Georeferenzierung_ZA-AUI.csv')
hauptkultur_df = pd.read_csv(hauptkultur_csv, encoding='latin1', sep=';')

farm_gpkg = os.path.expanduser('~/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.2_AUI_Monitoring_protected/Georeferenzierung/Georeferenzierung_AUI.gpkg')
farms_polys = gpd.read_file(farm_gpkg) #, crs=4326)

calendar_dir = os.path.expanduser('~/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.4_WAUM_protected/Daten/Feldkalender_AUI')

# Satellite data
s2_data_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH')
soil_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/DLR_soilsuite_preds/')

# Model data
soil_group = 0 # set to None if want to use soil-specific models
model_type = 'NN'

# Results storage
results_dir = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/timeseries_cleaning')



########################
# PARCEL AND YEAR SELECTION

# General plotting and data params
buffer_field = -10
buffer_plot = 200

yr = 2021
crop = 'Winterweizen'
betrID = 20424608001
parcel_name = '01.Lehmacker'

yr = 2018
crop = 'Sommerhafer'
betrID = 20400014298
parcel_name = '14Pt.Ependes'

yr = 2021
crop = 'Dauerweide intensiv'
betrID = 50023000173
parcel_name = '07LaPâtureM18'
         
"""
yr = 2020
crop = 'Futterweizen'
betrID = 20424608001
parcel_name = '04.O.Neuhaus'

yr = 2020
crop = 'Dinkel (Winterkorn)'
betrID = 20424608001
parcel_name = '05.M.Neuhaus'

yr = 2018
crop = 'Wintertriticale'
betrID = 20600006785
parcel_name = 'Tennis'

yr = 2018
crop = 'Sommerhafer'
betrID = 20400014298
parcel_name = '14Pt.Ependes'

yr = 2016
crop = 'Winterweizen'
betrID = 50023000105
parcel_name = 'LesEssapeux'

yr = 2019
crop = 'Winterweizen'
betrID = 50023000105
parcel_name = 'CombeVatio'

yr = 2017
crop = 'Winterraps für Speiseöl'
betrID = 50023000123
parcel_name = 'Sermuz'

yr = 2019
crop = 'Wintertriticale'
betrID = 20600006785
parcel_name = 'Bosse'

yr = 2017
crop = 'Wintertriticale'
betrID = 20600006785
parcel_name = 'Couliay'
"""


farm_crop_df = hauptkultur_df[(hauptkultur_df[f'hauptkultur_{yr}']==crop) & (hauptkultur_df.betr_ID==betrID)][['betr_ID', 'name', 'farmname']]
farm_name = farm_crop_df.farmname.unique()[0]
farms_polys['name'] = farms_polys['name'].apply(lambda x: x.replace(" ", "") if isinstance(x, str) else x)
parcel_shp = farms_polys[(farms_polys.betr_ID==betrID) & (farms_polys.farmname==farm_name) & (farms_polys.name==parcel_name)].to_crs(32632)



############################
# Generate FC timeseries

# Add inward 10m buffer
parcel_shp['geometry'] = parcel_shp['geometry'].to_crs(32632).buffer(buffer_field)
if parcel_shp.is_empty.any() or parcel_shp.geometry.area.any()<150:
    #If becomes empty geom after buffering, remove buffer
    parcel_shp = farms_polys[(farms_polys.betr_ID==betrID) & (farms_polys.farmname==farm_name) & (farms_polys.name==parcel_name)].to_crs(32632)          

# Get field calendar for that parcel
field_calendar_path = os.path.join(calendar_dir, f'Feldkalender_{yr}.txt')
parcel_id = parcel_shp[f'parzellen_id_{yr}'].values[0]

df = pd.read_csv(field_calendar_path, encoding='latin1', delimiter='\t')
df_parcel = df[df['parzellen_id']==parcel_id]
                
# Load activity dictionary
acitivity_dict = json.load(open(os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/spectral_unmixing/code/activities_dict.json'), 'r'))
df_parcel['activity'] = df_parcel['massnahme'].map(acitivity_dict)

# Sort dates
df_parcel['datum'] = pd.to_datetime(df_parcel['datum'], dayfirst=True)
df_parcel = df_parcel.sort_values('datum').dropna(subset=['datum'])

if len(df_parcel)==0:
    print('No farm management info')
  
                
                                                  

########################
# PREDICT FC ON S2 DATA  
ds, preds, soil_group_parcel = predict_fc(parcel_shp, s2_data_dir, yr, soil_group, model_type, chunk_size=10)
#soil_group_parcel is either defined as 0, or the None should be changed to a nbr if found


if ds is not None and soil_group_parcel is not None: 

    # Compute the mean PV, NPV, and soil fractions for the field
    mean_PV = preds.PV_norm.mean(dim=['lon', 'lat'])
    mean_NPV = preds.NPV_norm.mean(dim=['lon', 'lat'])
    mean_Soil = preds.Soil_norm.mean(dim=['lon', 'lat'])

    
    #########################
    # DATA CLEANING WITH SCL/MASK 
    # for whole image at each timestamp

    ds_clean = clean_dataset(ds, cloud_thresh=0.05, snow_thresh=0.1, shadow_thresh=0.1, cirrus_thresh=800)
    dates_clean = ds_clean.time.values
    dates_raw = ds.time.values
    
    print(f'Before cleaning: {len(dates_raw)}, After cleaning: {len(dates_clean)}')

    mean_PV_clean = mean_PV.sel(time=dates_clean) if len(dates_clean)>0 else []
    mean_NPV_clean = mean_NPV.sel(time=dates_clean) if len(dates_clean)>0 else []
    mean_Soil_clean = mean_Soil.sel(time=dates_clean) if len(dates_clean)>0 else []

    
   
    ####################################################################################################
    # GAP FILLING ON CLOUDY DATES (NO SPIKE REMOVAL)

    gapfilling_list = [{'algorithm': spline, 'kwargs': {'s': 0.02}},
                        {'algorithm': loess, 'kwargs': {'frac': 0.25}},
                        {'algorithm': savgol, 'kwargs': {'window_length': 5, 'polyorder':2}},
                        {'algorithm': kalman_combine_uncertainty},
                        {'algorithm': gpr_simple, 'kwargs': {'kernel': None, 'alpha':1e-6, 'random_state':42, 'validate':True}},
                        {'algorithm': gpr_combine_uncertainty, 'kwargs': {'kernel': None, 'alpha':1e-6, 'random_state':42}}]

    

    #########################
    # PLOT TIMESERIES W/ IMAGES - STATIC
    if soil_group == 0:
        save_path = os.path.join(results_dir, f'fulltime/{parcel_name.replace(" ", "").replace("/", "")}_FC_timeseries_imgs_cleanplot.png')
    else:
        save_path = os.path.join(results_dir, f'fulltime/{parcel_name.replace(" ", "").replace("/", "")}_FC_timeseries_imgs_soil{soil_group_parcel}_cleanplot.png')
    print(save_path)

    # Generate just a cloud cleaned timeseries plot (with images)
    plot_timeseries_images(df_parcel, ds, dates_clean, dates_raw, preds, parcel_shp, buffer_plot, save_path, smoothing=None)