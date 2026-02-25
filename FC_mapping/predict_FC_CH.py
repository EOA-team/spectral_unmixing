"""
Predict FC of PV, NPV and Soil of S2 data using trainedd spectral unmixing models across Switzerland for a year.
Predict using global model, as well as soil-specific models
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA version:", torch.version.cuda)
import math
import joblib
from pathlib import Path
import sys
sys.path.insert(0, '/mnt/eo-nas1/eoa-share/projects/028_Erosion/Erosion/spectral_unmixing')
from models import MODELS
import time
from shapely.geometry import box
from shapely import unary_union
from scipy.ndimage import distance_transform_edt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import shutil



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


def create_file_gdf(files, crs='EPSG:32632'):    
    records = []
    for f in files:
        try:
            minx, miny, maxx, maxy, yr = extract_bounds_year(os.path.basename(f))
            records.append({"filename": f, "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy, "yr": yr})
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not records:
        return []

    df = pd.DataFrame(records)
    df["geometry"] = df.apply(lambda row: box(row["minx"], row["miny"], row["maxx"], row["maxy"]), axis=1)
    gdf_files = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)

    return gdf_files


def create_square(row):
    x_max = row['E_COORD']
    y_max = row['N_COORD']
    x_min = x_max - 100
    y_min = y_max - 100
    return box(x_min, y_min, x_max, y_max)


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


def fill_nearest(data_array):
    data = data_array.values
    mask = np.isnan(data)
    
    if np.all(~mask):
        return data_array  # No fill needed

    distances, idx = distance_transform_edt(mask, return_indices=True)
    idx = tuple(i.astype(int) for i in idx)  # Cast indices to integers to avoid IndexError
    filled = data[tuple(idx)]
    return xr.DataArray(filled, coords=data_array.coords, dims=data_array.dims)


def find_soil_group(coords, soil_dir):
  # Open 3x3 cubes around central cubes (or as many as possible)
  x, y = coords.split('_')[0], coords.split('_')[1] # minx, maxy of central dataset
  surrounding_coords = []
  for i in range(-1, 1):
    for j in range(-1, 1):
      surrounding_coords.append(f'{int(x)+i*1280}_{int(y)+j*1280}')
  
  soil_ds = []
  for c in surrounding_coords:
    path = os.path.join(soil_dir, f'SRC_{c}.zarr')
    if os.path.exists(path):
      soil = xr.open_zarr(path).compute()
      soil_ds.append(soil)
    else:
      continue
  soil_ds = xr.combine_by_coords(soil_ds, combine_attrs='override')

  # FIll NaNs with nearest neigbor
  soil_ds['soil_group'] = soil_ds['soil_group'].where(soil_ds['soil_group'] != -10000, np.nan)
  soil_group = fill_nearest(soil_ds['soil_group'])

  # Extract central cube
  soil_group = soil_group.sel(x=slice(int(x), int(x)+1280), y=slice(int(y), int(y)-1270))

  return soil_group


def load_all_models(model_dir, model_type, soil_groups):
    models_dict = {}

    for soil_group in soil_groups:
        soil_group = int(soil_group)
        models_dict[soil_group] = {}

        for iteration in range(1, 6):  # 5 iterations
            models_dict[soil_group][iteration] = {
                'PV': joblib.load(os.path.join(model_dir, f'{model_type}_CLASS2_SOIL{soil_group}_ITER{iteration}.pkl')),
                'NPV': joblib.load(os.path.join(model_dir, f'{model_type}_CLASS1_SOIL{soil_group}_ITER{iteration}.pkl')),
                'Soil': joblib.load(os.path.join(model_dir, f'{model_type}_CLASS3_SOIL{soil_group}_ITER{iteration}.pkl'))
            }

    return models_dict


def predict_nn_in_batches(model, X_tensor, device, batch_size=10000):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X_tensor.shape[0], batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            out = model(batch).cpu()
            preds.append(out)
    return torch.cat(preds).numpy()


def predict_fc(f, models_dict, soil_dir, chunk_size=10, forced_soil_group=None): #model_dir, model_type,

    input_features = ['s2_B02','s2_B03','s2_B04','s2_B05','s2_B06','s2_B07','s2_B08','s2_B8A','s2_B11','s2_B12']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_preds_list = []  

    st = time.time()
    # === Load soil group data ===  
    if forced_soil_group is None: 
        # Determine soil group of each pixel parcel (fillna with nearest neighbor)
        coords = f'{f.split("_")[1]}_{f.split("_")[2]}'
        soil_group = find_soil_group(coords, soil_dir).rename({'y':'lat', 'x':'lon'})
        soil_ids = np.unique(soil_group.values)
        soil_ids = soil_ids[~np.isnan(soil_ids)]
        if not len(soil_ids):
            return None, None
    else:
        # Create dummy soil group 0
        ds_sample = xr.open_zarr(f, chunks={}).isel(time=0) #.reset_coords('time')
        soil_group = xr.DataArray(np.zeros((128, 128)), coords=ds_sample.coords, dims=ds_sample.dims)
        soil_ids = [0]
    print('Find soil group', time.time()-st)

    # === Load full dataset ===
    st = time.time()
    ds = xr.open_zarr(f)
    ds['soil_group'] = soil_group
    ds = ds.swap_dims({'time': 'product_uri'})
    ds = ds[input_features + ['soil_group', 'product_uri']].load() 
    print('Load data', time.time()-st)

    st = time.time()
    df_full = ds.to_dataframe().reset_index()
    df_full[input_features] = df_full[input_features].replace(65535, np.nan)
    print('To df', time.time()-st)
    
    for soil_group in soil_ids:
        soil_group = int(soil_group)
        models = models_dict[soil_group]
        """
        # Pre-load models
        models = {}
        for iteration in range(1, 6):
            models[iteration] = {
              'PV': joblib.load(os.path.join(model_dir, f'{model_type}_CLASS2_SOIL{soil_group}_ITER{iteration}.pkl')),
              'NPV': joblib.load(os.path.join(model_dir, f'{model_type}_CLASS1_SOIL{soil_group}_ITER{iteration}.pkl')),
              'Soil': joblib.load(os.path.join(model_dir, f'{model_type}_CLASS3_SOIL{soil_group}_ITER{iteration}.pkl'))
            }
        """
        # Filter for soil group
        df_soil = df_full[df_full['soil_group'] == soil_group]

        # === Loop over time in chunks ===
        uri_coords = df_soil['product_uri'].unique()
        num_chunks = math.ceil(len(uri_coords) / chunk_size)
        st = time.time()
        for i in range(num_chunks):
            uris_chunk = uri_coords[i * chunk_size: (i + 1) * chunk_size]
            df_chunk = df_soil[df_soil['product_uri'].isin(uris_chunk)].copy()

            df_valid = df_chunk.dropna(subset=input_features).copy()
            if df_valid.empty:
                continue
            
            X = (df_valid[input_features] / 10000).values

            # Predict across 5 iterations
            predictions_pv, predictions_npv, predictions_soil = [], [], []
            
            for iteration in range(1, 6):
                # Get pre-loaded models
                model_pv = models[iteration]["PV"]
                model_npv = models[iteration]["NPV"]
                model_soil = models[iteration]["Soil"]
               
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
           
            # Fill results back into chunk
            for col in ['PV', 'NPV', 'Soil']:
                df_chunk[col] = np.nan
                df_chunk.loc[df_valid.index, col] = df_valid[col]

            df_preds_list.append(df_chunk)
            
        if not len(df_preds_list):
            return None, None
        print('Predict', time.time()-st)

    st = time.time()
    # === Combine all predictions ===
    df_all = pd.concat(df_preds_list, ignore_index=True, copy=False)
    print('Concat', time.time()-st)
    st = time.time()
    #df_all.set_index(['lat', 'lon', 'product_uri'], inplace=True)
    preds = df_all.set_index(['lat', 'lon', 'product_uri'])[['PV', 'NPV', 'Soil', 'soil_group']].to_xarray()
    print('To xarray', time.time()-st)
    st = time.time()
    # Add time coords and restore dims
    time_coords = xr.apply_ufunc(extract_time, preds['product_uri'])
    preds = preds.assign_coords(time=("product_uri", time_coords.data))
    preds = preds.swap_dims({"product_uri": "time"}).reset_coords("product_uri", drop=False)
    print('Restore dims', time.time()-st)
    st = time.time()
    # Clip and normalize
    for key in ['PV', 'NPV', 'Soil']:
        preds[f'{key}_norm'] = preds[key].clip(min=0)

    denom = preds['PV_norm'] + preds['NPV_norm'] + preds['Soil_norm']
    for key in ['PV', 'NPV', 'Soil']:
        preds[f'{key}_norm'] /= denom
    print('Clip and norm', time.time()-st)
    st = time.time()
    # Sort by time
    preds = preds.sortby('time')
    ds = ds.sortby('time').swap_dims({"product_uri": "time"}).reset_coords("product_uri", drop=False)
    print('Sortby time', time.time()-st)
   

    return ds, preds


def predict_fc_optim(f, models_dict, soil_dir, chunk_size=10, forced_soil_group=None): #model_dir, model_type,

    input_features = ['s2_B02','s2_B03','s2_B04','s2_B05','s2_B06','s2_B07','s2_B08','s2_B8A','s2_B11','s2_B12']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_preds_list = []  

    # === Load soil group data ===  
    if forced_soil_group is None: 
        # Determine soil group of each pixel parcel (fillna with nearest neighbor)
        coords = f'{f.split("_")[1]}_{f.split("_")[2]}'
        soil_group = find_soil_group(coords, soil_dir).rename({'y':'lat', 'x':'lon'})
        soil_ids = np.unique(soil_group.values)
        soil_ids = soil_ids[~np.isnan(soil_ids)]
        if not len(soil_ids):
            return None, None
    else:
        # Create dummy soil group 0
        ds_sample = xr.open_zarr(f, chunks={}).isel(time=0) #.reset_coords('time')
        soil_group = xr.DataArray(np.zeros((128, 128)), coords=ds_sample.coords, dims=ds_sample.dims)
        soil_ids = [0]


    # === Load full dataset ===
    ds = xr.open_zarr(f)
    ds['soil_group'] = soil_group
    ds = ds.swap_dims({'time': 'product_uri'})
    ds = ds[input_features + ['soil_group', 'product_uri']].load() 
    ds = ds.where(ds != 65535)

    preds_list = []
    
    for soil_group in soil_ids:
        soil_group = int(soil_group)
        models = models_dict[soil_group]
        
        # Filter for soil group
        ds_sub = ds.where(ds['soil_group'] == soil_group, drop=True)
        if ds_sub.sizes == {} or any(v == 0 for v in ds_sub.sizes.values()):
            continue

        # Extract as array (shape: [n_uri, n_lat, n_lon, n_features])
        X = ds_sub[input_features].to_array().transpose("product_uri", "lat", "lon", "variable").values
        X = X.reshape(-1, len(input_features)).astype("float32") / 10000

        # Filter valid samples
        valid_mask = np.isfinite(X).all(axis=1)
        if not np.any(valid_mask):
            continue
        X = X[valid_mask]

        # Create coordinate arrays for reconstruction
        uris = ds_sub['product_uri'].values
        lat = ds_sub['lat'].values
        lon = ds_sub['lon'].values
        uri_grid, lat_grid, lon_grid = np.meshgrid(uris, lat, lon, indexing='ij')
        uri_valid = uri_grid.ravel()[valid_mask]
        lat_valid = lat_grid.ravel()[valid_mask]
        lon_valid = lon_grid.ravel()[valid_mask]

        # === Loop over time in chunks ===
        n = len(X)
        num_chunks = math.ceil(n / chunk_size)
        preds = np.empty((n, 3), dtype=np.float32)

        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n)
            X_batch = X[start:end]

            # Predict across 5 iterations
            predictions_pv, predictions_npv, predictions_soil = [], [], []
            
            for iteration in range(1, 6):
                # Get pre-loaded models
                model_pv = models[iteration]["PV"]
                model_npv = models[iteration]["NPV"]
                model_soil = models[iteration]["Soil"]
               
                if model_type == 'NN':
                    X_tensor = torch.FloatTensor(X_batch)
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
            preds[start:end, 0] = np.mean(predictions_pv, axis=0).squeeze()
            preds[start:end, 1] = np.mean(predictions_npv, axis=0).squeeze()
            preds[start:end, 2] = np.mean(predictions_soil, axis=0).squeeze()
           
        # === Convert predictions to xarray ===
        preds_group = xr.Dataset(
            {
                "PV": (("point",), preds_arr[:, 0]),
                "NPV": (("point",), preds_arr[:, 1]),
                "Soil": (("point",), preds_arr[:, 2]),
                "soil_group": (("point",), np.full(n, soil_group)),
            },
            coords={
                "product_uri": ("point", uri_valid),
                "lat": ("point", lat_valid),
                "lon": ("point", lon_valid),
            },
        )

        preds_list.append(preds_group)

    # === Combine all soil groups ===
    if not preds_list:
        return None, None

    preds = xr.concat(preds_list, dim="point")

    # === Add time coordinates and normalize ===
    time_coords = xr.apply_ufunc(extract_time, preds['product_uri'])
    preds = preds.assign_coords(time=("point", time_coords.data))

    # Normalize (avoid division by 0)
    for key in ['PV', 'NPV', 'Soil']:
        preds[f'{key}_norm'] = preds[key].clip(min=0)

    denom = preds['PV_norm'] + preds['NPV_norm'] + preds['Soil_norm']
    denom = denom.where(denom != 0, np.nan)

    for key in ['PV', 'NPV', 'Soil']:
        preds[f'{key}_norm'] = preds[f'{key}_norm'] / denom

    # === Sort by time for consistency ===
    preds = preds.sortby('time')

    return ds, preds


def process_file_old(f, model_dir, model_type, soil_dir, out_dir):
    try:
        print(f'Processing file {f}...')
        start = time.time()

        # Predict with global and soil-specific models
        ds_global, preds_global = predict_fc(f, model_dir, model_type, soil_dir, chunk_size=10, forced_soil_group=0)
        ds_soil, preds_soil = predict_fc(f, model_dir, model_type, soil_dir, chunk_size=10)

        # Combine results
        preds_combined = preds_global.drop_vars('soil_group').copy()

        for var in ['PV_norm', 'NPV_norm', 'Soil_norm']:
            if preds_global is not None:
                preds_combined[f'{var}_global'] = preds_global[var]
            if preds_soil is not None:
                preds_combined[f'{var}_soil'] = preds_soil[var]
                preds_combined['soil_group'] = preds_soil['soil_group']

        preds_combined = preds_combined.drop_vars(['PV_norm', 'NPV_norm', 'Soil_norm'])

        # Save predictions
        save_path = os.path.join(out_dir, os.path.basename(f))
        preds_combined.to_zarr(save_path, consolidated=True, mode='w')

        end = time.time()
        print(f'...saved to {save_path} (took {end-start:.2f}s)')

    except Exception as e:
        print(f'Error processing file {f}: {e}')


def process_file(i, f, model_dir, model_type, soil_dir, out_dir):
    
    save_path = os.path.join(out_dir, os.path.basename(f))
    if os.path.exists(save_path):
        return i, f"Skipped (already exists) {f}", 0

    try:
        start = time.time()
        # Predict with global and soil-specific models
        ds_global, preds_global = predict_fc(f, model_dir, model_type, soil_dir, chunk_size=10, forced_soil_group=0)
        ds_soil, preds_soil = predict_fc(f, model_dir, model_type, soil_dir, chunk_size=10)
    
        # Combine results
        preds_combined = preds_global.drop_vars('soil_group').copy()

        for var in ['PV_norm', 'NPV_norm', 'Soil_norm']:
            if preds_global is not None:
                preds_combined[f'{var}_global'] = preds_global[var]
            if preds_soil is not None:
                preds_combined[f'{var}_soil'] = preds_soil[var]
                preds_combined['soil_group'] = preds_soil['soil_group']

        preds_combined = preds_combined.drop_vars(['PV_norm', 'NPV_norm', 'Soil_norm'])
        preds_combined.to_zarr(save_path, consolidated=True, mode='w')
        elapsed = time.time() - start
        return i, f"Done {f}", elapsed

    except Exception as e:
        return i, f"Error: {e}", 0


if __name__ == '__main__':


  # PATHS
  s2_dir = '/srv/data/sentinel2' #os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH')
  out_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/FC')
  soil_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/DLR_soilsuite_preds/')
  model_dir = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/028_Erosion/Erosion/spectral_unmixing/models')
  landuse_data = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/EVI_CH/ag-b-00.03-37-area-current-csv.csv')

  # PARAMS
  year = 2022
  model_type = 'NN'

  
  ###############################
  # RUN PREDICTIONS

  gdf_s2_files_path = f'gdf_s2_files_{year}_cropland.pkl'
  
  if not os.path.exists(gdf_s2_files_path):
    if year is None:
      s2_files = [f for f in os.listdir(s2_dir)]
    else:
      if isinstance(year, int):
          year = [str(year)]
      else:
          year = [str(y) for y in year]
      s2_files = [os.path.join(s2_dir, f) for f in os.listdir(s2_dir) if f.endswith('.zarr') and any(f.split('_')[3].startswith(y) for y in year)]
    

    # Prefilter for where landuse is agricultural (AS_17 codes 6-9)
    lu_csv = pd.read_csv(landuse_data, sep=';')
    lu_csv['geometry'] = lu_csv.apply(create_square, axis=1)
    gdf_lu = gpd.GeoDataFrame(lu_csv, geometry='geometry', crs='EPSG:2056').to_crs(32632)
    gdf_lu = gdf_lu[gdf_lu.AS_17.isin([6,7,8,9])] # 6 obst, 7 ackerland, 8 and 9 grassland

    gdf_s2_files = create_file_gdf(s2_files)
    intersecting = gdf_s2_files.geometry.intersects(unary_union(gdf_lu.geometry))
    gdf_s2_files = gdf_s2_files[intersecting].copy()
    gdf_s2_files.to_pickle(gdf_s2_files_path)
      
  else:
    gdf_s2_files = pd.read_pickle(gdf_s2_files_path)

  s2_files = gdf_s2_files.filename.tolist()
  tot_files = len(s2_files)

  # === Load models once ===
  soil_groups = [0, 1, 2, 3, 4, 5]
  models_dict = load_all_models(model_dir, model_type, soil_groups)

  # === Predict on all Sentinel-2 files ===
  for i,f in enumerate(s2_files):
    
    save_path = os.path.join(out_dir, f.split('/')[-1])
    if not os.path.exists(save_path):
    
        print(f'Processing file {i}/{tot_files}...')
        start = time.time()
        # Predict with global and soil-specific models
        #ds_global, preds_global = predict_fc(f, model_dir, model_type, soil_dir, chunk_size=10, forced_soil_group=0)
        #ds_soil, preds_soil = predict_fc(f, model_dir, model_type, soil_dir, chunk_size=10)
        ds_global, preds_global = predict_fc(f, models_dict, soil_dir, chunk_size=50, forced_soil_group=0)
        ds_soil, preds_soil = predict_fc(f, models_dict, soil_dir, chunk_size=50)

        st = time.time()
        # Combine results
        preds_combined = preds_global.drop_vars('soil_group').copy()

        for var in ['PV_norm', 'NPV_norm', 'Soil_norm']:
            if preds_global is not None:
                preds_combined[f'{var}_global'] = preds_global[var]
            if preds_soil is not None:
                preds_combined[f'{var}_soil'] = preds_soil[var]
                preds_combined['soil_group'] = preds_soil['soil_group']
        preds_combined = preds_combined.drop_vars(['PV_norm', 'NPV_norm', 'Soil_norm'])
        print('Combine soil and global', time.time()-st)

        # Save predictions (faster: write to scratch first and then move)
        st = time.time()
        #local_path = os.path.expanduser(f"~/scratch/{os.path.basename(save_path)}")
        #preds_combined.to_zarr(local_path, consolidated=True, mode='w')
        #shutil.move(local_path, save_path)
        preds_combined.to_zarr(save_path, consolidated=True, mode='w')
        print('Saving', time.time()-st)
        print(f'...saved to {save_path}')

        end = time.time()
        print('Took', end-start)
  """

  mp.set_start_method('spawn', force=True)
  with ProcessPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(process_file, i, f, model_dir, model_type, soil_dir, out_dir) 
    for i, f in enumerate(s2_files) if i>1699
    ]

    for j, future in enumerate(as_completed(futures)):
        i, msg, elapsed = future.result()
        print(f"{i}: {msg}. Took {elapsed}")

"""