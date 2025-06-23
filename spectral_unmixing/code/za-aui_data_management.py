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
sys.path.insert(0, str(Path(os.path.dirname(os.path.realpath("__file__"))).parent))
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
    cloud_condition = (ds.s2_mask == 1) | (ds.s2_SCL.isin([8, 9, 10]))
    return cloud_condition.sum(dim=['lat', 'lon'])/(len(ds.lat)*len(ds.lon)) > cloud_thresh


def has_shadows(ds, shadow_thresh=0.1):
    shadow_condition = (ds.s2_mask == 2) | (ds.s2_SCL == 3)
    return shadow_condition.sum(dim=['lat', 'lon'])/(len(ds.lat)*len(ds.lon)) > shadow_thresh


def has_snow(ds, snow_thresh=0.1):
    snow_condition = (ds.s2_mask == 3) | (ds.s2_SCL == 11)
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

    s2_files = find_cubes(parcel_shp,s2_data_dir, [yr, yr-1])

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

        # Initialize prediction columns with NaNs
        df['PV'] = np.nan
        df['NPV'] = np.nan
        df['Soil'] = np.nan

        # Fill prediction columns with predicted values at valid indices
        df.loc[df_valid.index, 'PV'] = df_valid['PV']
        df.loc[df_valid.index, 'NPV'] = df_valid['NPV']
        df.loc[df_valid.index, 'Soil'] = df_valid['Soil']

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


def plot_timeseries(df_parcel, dates_raw, mean_PV, mean_NPV, mean_Soil, dates_clean, mean_PV_clean, mean_NPV_clean, mean_Soil_clean, save_path):
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


def plot_timeseries_images(df_parcel, ds, dates_clean, dates_raw, preds, parcel_shp, buffer_plot, save_path):
    
    # Shorten S2 timeseries to field calendar
    start_date = df_parcel['datum'].min()-pd.Timedelta(days=20)
    end_date = df_parcel['datum'].max()+pd.Timedelta(days=20)
    start_s2 = ds.sel(time=start_date, method='nearest').time.values
    end_s2 = ds.sel(time=end_date, method='nearest').time.values
    ds = ds.sel(time=slice(start_s2, end_s2))
    preds = preds.sel(time=slice(start_s2, end_s2))
    dates_clean = np.array([d for d in ds.time.values if d in dates_clean])
    dates_raw = np.array([d for d in ds.time.values if d in dates_raw])
   
    # Get timeseries/RGB for shorten timeseries and cleaned dates
    mean_PV = preds.PV_norm.mean(dim=['lon', 'lat'])
    mean_NPV = preds.NPV_norm.mean(dim=['lon', 'lat'])
    mean_Soil = preds.Soil_norm.mean(dim=['lon', 'lat'])
    
    if len(dates_clean):
        mean_PV_clean = mean_PV.sel(time=dates_clean)
        mean_NPV_clean = mean_NPV.sel(time=dates_clean)
        mean_Soil_clean = mean_Soil.sel(time=dates_clean)
    
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
        ax_ts.plot(dates_clean, mean_PV_clean.values, label='Cleaned PV', color='g', linewidth=2)
        ax_ts.plot(dates_clean, mean_NPV_clean.values, label='Cleaned NPV', color='goldenrod', linewidth=2)
        ax_ts.plot(dates_clean, mean_Soil_clean.values, label='Cleaned Soil', color='saddlebrown', linewidth=2)

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
    ax_ts.set_title('Predicted soil cover')
    ax_ts.set_xlim(df_parcel['datum'].min()-pd.Timedelta(days=20), df_parcel['datum'].max()+pd.Timedelta(days=20))

    plt.tight_layout() #
    plt.savefig(save_path)

    return


def anim_s2_raw(df_parcel, ds, preds, dates_clean, parcel_shp, buffer_plot, save_path):
   
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2,2, figure=fig, width_ratios=[2,1], height_ratios=[2,1])


    # Initialize the time series plot
    time_series_ax = fig.add_subplot(gs[1, :])

    mean_PV_line, = time_series_ax.plot([], [], color='g', label='Cleaned PV')
    mean_NPV_line, = time_series_ax.plot([], [], color='goldenrod', label='Cleaned NPV')
    mean_Soil_line, = time_series_ax.plot([], [], color='saddlebrown', label='Cleaned soil')

    mean_PV_scatter = time_series_ax.scatter([], [], color='g', alpha=0.5, label='Raw PV', s=15)
    mean_NPV_scatter = time_series_ax.scatter([], [], color='goldenrod', alpha=0.5, label='Raw NPV', s=15)
    mean_Soil_scatter = time_series_ax.scatter([], [], color='saddlebrown', alpha=0.5, label='Raw soil', s=15)

    time_series_ax.set_ylim(0, 1)
    time_series_ax.set_ylabel("Mean field cover", fontsize=18)
    time_series_ax.set_xlim(df_parcel['datum'].min()-pd.Timedelta(days=20), df_parcel['datum'].max()+pd.Timedelta(days=20))


    # Shorten S2 timeseries to field calendar
    start_date = df_parcel['datum'].min()-pd.Timedelta(days=20)
    end_date = df_parcel['datum'].max()+pd.Timedelta(days=20)
    start_s2 = ds.sel(time=start_date, method='nearest').time.values
    end_s2 = ds.sel(time=end_date, method='nearest').time.values
    ds = ds.sel(time=slice(start_s2, end_s2))
    preds = preds.sel(time=slice(start_s2, end_s2))

    # Compute the mean PV, NPV, and soil fractions for the field and new time range
    mean_PV = preds.PV_norm.mean(dim=['lon', 'lat'])
    mean_NPV = preds.NPV_norm.mean(dim=['lon', 'lat'])
    mean_Soil = preds.Soil_norm.mean(dim=['lon', 'lat'])

    dates_clean = np.array([d for d in preds.time.values if d in dates_clean])
    mean_PV_clean = mean_PV.sel(time=dates_clean)
    mean_NPV_clean = mean_NPV.sel(time=dates_clean)
    mean_Soil_clean = mean_Soil.sel(time=dates_clean)

    # Initialize RGB subplot
    ax_rgb = fig.add_subplot(gs[0, :])

    scale_factor = 1.0 / 10000.0 
    r = ds['s2_B04']* scale_factor
    g = ds['s2_B03']* scale_factor
    b = ds['s2_B02']* scale_factor
    rgb = xr.concat([r, g, b], dim='band').transpose('time', 'lat', 'lon', 'band')
    rgb = rgb.where(~np.isnan(rgb), other=1.0)
    brightness = 3

    minx, miny, maxx, maxy = parcel_shp.total_bounds
    rgb = rgb.sel(lon=slice(minx-buffer_plot, maxx+buffer_plot), lat=slice(maxy+buffer_plot, miny-buffer_plot))

    """ 
    # Remove activity if same has happened within 5 days
    filtered_rows = []
    for activity, group in df_parcel.groupby('activity'):
        last_date = None
        for _, row in group.iterrows():
            if last_date is None or (row['datum'] - last_date).days >= 5:
                filtered_rows.append(row)
                last_date = row['datum']
    df_parcel = pd.DataFrame(filtered_rows)
    """

    # Plot vertical dashed lines for each activity
    df_parcel = df_parcel.dropna(subset=['datum'])
    for _, row in df_parcel.iterrows():
        remove_residues = row['ernteresteeingearbeitet']
        color = 'red' if remove_residues else 'gray'
        time_series_ax.axvline(x=row['datum'], color=color, linestyle='--', alpha=0.7)
        time_series_ax.text(row['datum'], 0.6, row['activity'], rotation=90, verticalalignment='bottom', fontsize=10)

    # Function to update the frame
    def update(frame):
        ax_rgb.clear()
        ax_rgb.axis('off')

        # Plot RGB image
        x_coords = rgb[frame].coords['lon'].values
        y_coords = rgb[frame].coords['lat'].values  
        ax_rgb.imshow(rgb[frame, :, :] * brightness, origin='upper', extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()])
        ax_rgb.set_title(f"RGB {str(preds.time.values[frame]).split('T')[0]}")

        # Add outline of field
        parcel_shp.plot(ax=ax_rgb, alpha=0.5, edgecolor='r', facecolor='None', linewidth=2)

        # Raw dates (for scatter): all time steps up to this frame
        time_raw_numeric = mdates.date2num(preds.time.values[:frame+1])
        
        mean_PV_scatter.set_offsets(np.c_[time_raw_numeric, mean_PV.values[:frame+1]])
        mean_NPV_scatter.set_offsets(np.c_[time_raw_numeric, mean_NPV.values[:frame+1]])
        mean_Soil_scatter.set_offsets(np.c_[time_raw_numeric, mean_Soil.values[:frame+1]])
        
        # Clean dates: subset to those within current frame
        current_time = preds.time.values[frame]
        if current_time in dates_clean:
            clean_mask = dates_clean <= current_time
            
            time_clean_numeric = mdates.date2num(dates_clean[clean_mask])
            mean_PV_line.set_data(time_clean_numeric, mean_PV_clean.values[clean_mask])
            mean_NPV_line.set_data(time_clean_numeric, mean_NPV_clean.values[clean_mask])
            mean_Soil_line.set_data(time_clean_numeric, mean_Soil_clean.values[clean_mask])

        # Create a custom for "residues" line
        residue_line = mlines.Line2D([], [], color='red', linestyle='--', label='Remove residues')
        handles, labels = time_series_ax.get_legend_handles_labels()
        handles.append(residue_line)
        labels.append("Remove residues")
        time_series_ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(ds.time.values), interval=500, repeat=False)
    FFwriter = animation.FFMpegWriter(fps=1)
    anim.save(filename=save_path, writer=FFwriter)

    return


def anim_s2_clean(df_parcel, parcel_shp, buffer_plot, ds, preds, dates_clean, save_path):

    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2,2, figure=fig, width_ratios=[2,1], height_ratios=[2,1])


    # Initialize the time series plot
    time_series_ax = fig.add_subplot(gs[1, :])

    mean_PV_line, = time_series_ax.plot([], [], color='g', label='Cleaned PV')
    mean_NPV_line, = time_series_ax.plot([], [], color='goldenrod', label='Cleaned NPV')
    mean_Soil_line, = time_series_ax.plot([], [], color='saddlebrown', label='Cleaned soil')

    mean_PV_scatter = time_series_ax.scatter([], [], color='g', alpha=0.5, label='Raw PV', s=15)
    mean_NPV_scatter = time_series_ax.scatter([], [], color='goldenrod', alpha=0.5, label='Raw NPV', s=15)
    mean_Soil_scatter = time_series_ax.scatter([], [], color='saddlebrown', alpha=0.5, label='Raw soil', s=15)

    time_series_ax.set_ylim(0, 1)
    time_series_ax.set_ylabel("Mean field cover", fontsize=18)
    time_series_ax.set_xlim(df_parcel['datum'].min()-pd.Timedelta(days=20), df_parcel['datum'].max()+pd.Timedelta(days=20))


    # Shorten S2 timeseries to field calendar
    start_date = df_parcel['datum'].min()-pd.Timedelta(days=20)
    end_date = df_parcel['datum'].max()+pd.Timedelta(days=20)
    start_s2 = ds.sel(time=start_date, method='nearest').time.values
    end_s2 = ds.sel(time=end_date, method='nearest').time.values
    ds = ds.sel(time=slice(start_s2, end_s2))
    preds = preds.sel(time=slice(start_s2, end_s2))

    # Compute the mean PV, NPV, and soil fractions for the field and new time range
    dates_clean = np.array([d for d in preds.time.values if d in dates_clean])
    mean_PV = preds.PV_norm.mean(dim=['lon', 'lat']).sel(time=dates_clean)
    mean_NPV = preds.NPV_norm.mean(dim=['lon', 'lat']).sel(time=dates_clean)
    mean_Soil = preds.Soil_norm.mean(dim=['lon', 'lat']).sel(time=dates_clean)

    # Initialize RGB subplot
    ax_rgb = fig.add_subplot(gs[0, :])

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

    """
    # Remove activity if same has happened within 5 days
    filtered_rows = []
    for activity, group in df_parcel.groupby('activity'):
        last_date = None
        for _, row in group.iterrows():
            if last_date is None or (row['datum'] - last_date).days >= 5:
                filtered_rows.append(row)
                last_date = row['datum']
    df_parcel = pd.DataFrame(filtered_rows)
    """

    # Plot vertical dashed lines for each activity
    df_parcel = df_parcel.dropna(subset=['datum'])
    for _, row in df_parcel.iterrows():
        remove_residues = row['ernteresteeingearbeitet']
        color = 'red' if remove_residues else 'gray'
        time_series_ax.axvline(x=row['datum'], color=color, linestyle='--', alpha=0.7)
        time_series_ax.text(row['datum'], 0.6, row['activity'], rotation=90, verticalalignment='bottom', fontsize=10)


    # Function to update the frame
    def update(frame):
        ax_rgb.clear()
        ax_rgb.axis('off')

        # Plot RGB image
        x_coords = rgb[frame].coords['lon'].values
        y_coords = rgb[frame].coords['lat'].values
        ax_rgb.imshow(rgb[frame, :, :] * brightness, origin='upper', extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()])
        ax_rgb.set_title(f"RGB {str(rgb.time.values[frame]).split('T')[0]}")

        # Add outline of field
        parcel_shp.plot(ax=ax_rgb, alpha=0.5, edgecolor='r', facecolor='None', linewidth=2)

        time_numeric = mdates.date2num(dates_clean[:frame+1])
        
        mean_PV_scatter.set_offsets(np.c_[time_numeric, mean_PV.values[:frame+1]])
        mean_NPV_scatter.set_offsets(np.c_[time_numeric, mean_NPV.values[:frame+1]])
        mean_Soil_scatter.set_offsets(np.c_[time_numeric, mean_Soil.values[:frame+1]])
        
        mean_PV_line.set_data(time_numeric,  mean_PV.values[:frame+1])
        mean_NPV_line.set_data(time_numeric, mean_NPV.values[:frame+1])
        mean_Soil_line.set_data(time_numeric, mean_Soil.values[:frame+1])

        # Create a custom for "residues" line
        residue_line = mlines.Line2D([], [], color='red', linestyle='--', label='Remove residues')
        handles, labels = time_series_ax.get_legend_handles_labels()
        handles.append(residue_line)
        labels.append("Remove residues")
        time_series_ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(dates_clean), interval=500, repeat=False)
    FFwriter = animation.FFMpegWriter(fps=1)
    anim.save(filename=save_path, writer=FFwriter)

    return





########################
# PATHS TO DATA

# Farm data
hauptkultur_csv = os.path.expanduser('~/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.2_AUI_Monitoring_protected/Georeferenzierung/Hauptkulturen_Georeferenzierung_ZA-AUI.csv')
hauptkultur_df = pd.read_csv(hauptkultur_csv, encoding='latin1', sep=';')

farm_gpkg = os.path.expanduser('~/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.2_AUI_Monitoring_protected/Georeferenzierung/Georeferenzierung_AUI.gpkg')
farms_polys = gpd.read_file(farm_gpkg, crs=4326)

calendar_dir = os.path.expanduser('~/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.4_WAUM_protected/Daten/Feldkalender_AUI')

# Satellite data
s2_data_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH')
soil_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/DLR_soilsuite_preds/')

# Model data
soil_group = 0 # set to None if want to use soil-specific models
model_type = 'NN'

# Results storage
results_dir = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/spectral_unmixing/results/ZA-AUI/NPV_check')



########################
# LOOP OVER DATA AND GENERATE PLOTS FOR EACH FIELD

# General plotting and data params
buffer_field = -10
buffer_plot = 200

management = 'Mowing weeds'
file_suffix = 'mowweeds'

# Loop through years
years = [int(col.split('_')[1]) for col in hauptkultur_df.columns if 'hauptkultur' in col]
years = [y for y in years if y>2015] # Available S2 data

for yr in years:
    os.makedirs(results_dir, exist_ok=True)

    # Get field calendar for that year
    field_calendar_path = os.path.join(calendar_dir, f'Feldkalender_{yr}.txt')
    df = pd.read_csv(field_calendar_path, encoding='latin1', delimiter='\t')
    acitivity_dict = json.load(open('activities_dict.json', 'r'))
    df['activity'] = df['massnahme'].map(acitivity_dict)

    # Sample a parcel that has given management
    farm_polys_with_calendar = pd.merge(farms_polys, df, left_on=f'parzellen_id_{yr}', right_on='parzellen_id')
    try:
        #parcel_with_mgmt = farm_polys_with_calendar[(farm_polys_with_calendar.activity==management) & (farm_polys_with_calendar.name=='Rte Bétonnée')]
        parcel_with_mgmt = farm_polys_with_calendar[farm_polys_with_calendar.activity==management].sample(1)
        parcel_id = parcel_with_mgmt.parzellen_id.values[0]
        betr_ID = parcel_with_mgmt.betr_ID.values[0]
        farm_name = parcel_with_mgmt.farmname.values[0]
        parcel_name = parcel_with_mgmt.name.values[0]
        
        farm_data = hauptkultur_df[(hauptkultur_df.betr_ID==betr_ID) & (hauptkultur_df.farmname==farm_name) & (hauptkultur_df.name==parcel_name)]
        crop = farm_data[f'hauptkultur_{yr}']

        # Prepare parcel geom
        parcel_with_mgmt = parcel_with_mgmt[~parcel_with_mgmt.geometry.is_empty]
        original_geom = parcel_with_mgmt.to_crs(32632)
        parcel_with_mgmt['geometry'] = parcel_with_mgmt['geometry'].to_crs(32632).buffer(buffer_field)
        if parcel_with_mgmt.is_empty.any() or parcel_with_mgmt.geometry.area.any()<150:
            #If becomes empty geom after buffering, remove buffer
            parcel_with_mgmt = original_geom #farm_polys_with_calendar[farm_polys_with_calendar.activity==management].sample(1).to_crs(32632)
               

        # Get full management calendar
        df_parcel = df[df['parzellen_id']==parcel_id]
        df_parcel['datum'] = pd.to_datetime(df_parcel['datum'], dayfirst=True)
        df_parcel = df_parcel.sort_values('datum').dropna(subset=['datum'])

        #########################
        # PREDICT FC ON S2 DATA  
        ds, preds, soil_group_parcel = predict_fc(parcel_with_mgmt, s2_data_dir, yr, soil_group, model_type, chunk_size=10)
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


            #########################
            # PLOT TIMESERIES W/ IMAGES - STATIC
            if soil_group == 0:
                save_path = os.path.join(results_dir, f'{str(yr)}_{crop.values[0]}_{parcel_name.replace(" ", "").replace("/", "")}_FC_timeseries_imgs_{file_suffix}.png')
            else:
                save_path = os.path.join(results_dir, f'{str(yr)}_{crop.values[0]}_{parcel_name.replace(" ", "").replace("/", "")}_FC_timeseries_imgs_{file_suffix}_soil{soil_group_parcel}.png')
            
            plot_timeseries_images(df_parcel, ds, dates_clean, dates_raw, preds, parcel_with_mgmt, buffer_plot, save_path)
            print(save_path)
        

    except:
        # That management not present that year
        continue
    
