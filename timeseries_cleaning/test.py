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
    print(preds.PV_norm.isel(time=0, lat=0, lon=0).values, preds.NPV_norm.isel(time=0, lat=0, lon=0).values, preds.Soil_norm.isel(time=0, lat=0, lon=0).values)
    denom = preds['PV_norm'] + preds['NPV_norm'] + preds['Soil_norm']
    preds['PV_norm'] = preds['PV_norm'] / denom
    preds['NPV_norm'] = preds['NPV_norm'] / denom
    preds['Soil_norm'] = preds['Soil_norm'] / denom
    print(preds.PV_norm.isel(time=0, lat=0, lon=0).values, preds.NPV_norm.isel(time=0, lat=0, lon=0).values, preds.Soil_norm.isel(time=0, lat=0, lon=0).values)


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





###########
# TIMESERIES CLEANING FUNCTIONS

def remove_spikes(data, max_days=10, threshold=0.1):
    """
    Remove spikes that quickly revert within max_days.
    This looks at each point (except first and last).
    If the current point jumps more than 15% from previous (threshold=0.15) but next point returns close to previous (within 15%), it replaces the spike with the average of neighbors.
    The max_days ensures the reversal happens within 10 days.

    
    Parameters:
    - data: xarray.DataArray with 'time' dimension
    - max_days: max time window to consider a spike
    - threshold: minimum relative change to consider a spike
    
    Returns:
    - corrected_data: DataArray with spikes removed
    """
    data = data.copy()
    times = data['time'].values
    values = data.values
    
    for i in range(1, len(values)-1):
        prev_val = values[i-1]
        cur_val = values[i]
        next_val = values[i+1]
        
        # Check if current value is a spike:
        # It deviates strongly from prev_val, then next_val returns close to prev_val
        rel_change_up = (cur_val - prev_val) / prev_val if prev_val != 0 else 0
        rel_change_down = (prev_val - cur_val) / prev_val if prev_val != 0 else 0
        
        # Time differences
        dt_next = (times[i+1] - times[i]).astype('timedelta64[D]').astype(int)
        
        if dt_next <= max_days:
            # Spike up then back down
            if rel_change_up > threshold and abs(next_val - prev_val) / prev_val < threshold:
                # Replace spike with average of neighbors
                data[i] = (prev_val + next_val) / 2
            # Spike down then back up
            elif rel_change_down > threshold and abs(next_val - prev_val) / prev_val < threshold:
                data[i] = (prev_val + next_val) / 2

    return data





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
yr = 2020
crop = 'Dinkel (Winterkorn)'
betrID = 20424608001
parcel_name = '05.M.Neuhaus'


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

    ds_clean = clean_dataset(ds, cloud_thresh=0.1, snow_thresh=0.1, shadow_thresh=0.1, cirrus_thresh=800)
    dates_clean = ds_clean.time.values
    dates_raw = ds.time.values
    
    print(f'Before cleaning: {len(dates_raw)}, After cleaning: {len(dates_clean)}')

    mean_PV_clean = mean_PV.sel(time=dates_clean) if len(dates_clean)>0 else []
    mean_NPV_clean = mean_NPV.sel(time=dates_clean) if len(dates_clean)>0 else []
    mean_Soil_clean = mean_Soil.sel(time=dates_clean) if len(dates_clean)>0 else []


    #########################
    # FURTHER TIMESERIES CLEANING

    # to try
    # - remove abrupt spikes (spike/drop that goes back to its previous result in too short time)
    # could be done using Zscore anomyla detection on a difference threshold between two cosecutive points
    # but need to be carefult of what is possible due to management
    # - temporal smoothing (rolling mean, loess, SG filter)
    # - interpolate missing/dropped values 
    # to get uncertainty esures: trained GPR (but not good for scaling), kalman filter, loess with bootstrap uncertainty, bayesian curve fitting (could be slow)

    # Spike/drop in 10 day interval
    mean_PV_clean = remove_spikes(mean_PV_clean, max_days=20, threshold=0.2)
    mean_NPV_clean = remove_spikes(mean_NPV_clean, max_days=20, threshold=0.2)
    mean_Soil_clean = remove_spikes(mean_Soil_clean, max_days=20, threshold=0.2)



    #########################
    # PLOT TIMESERIES
    if soil_group == 0:
        save_path = os.path.join(results_dir, f'{parcel_name.replace(" ", "").replace("/", "")}_FC_timeseries_spike.png')
    else:
        save_path = os.path.join(results_dir, f'{parcel_name.replace(" ", "").replace("/", "")}_FC_timeseries_soil{soil_group_parcel}.png')
    print(save_path)
    plot_timeseries(df_parcel, dates_raw, mean_PV, mean_NPV, mean_Soil, dates_clean, mean_PV_clean, mean_NPV_clean, mean_Soil_clean, save_path)
