"""
Perform inference an dused trained K-means clustering model to classify all soil pixels obtained from DLR soil suite

Sélène Ledain, 16 May 2025
"""

import os
import xarray as xr
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import zarr
import shutil
import warnings 
warnings.filterwarnings("ignore")
import zarr



##############
# PATHS AND PARAMETERS

soil_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/DLR_soilsuite/')
n_clusters = 5
output_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/DLR_soilsuite_preds/')
input_bands = ['SRC_B2', 'SRC_B3', 'SRC_B4', 'SRC_B5', 'SRC_B6', 'SRC_B7', 'SRC_B8', 'SRC_B8A', 'SRC_B11', 'SRC_B12']
label_map = {0: 5, 1: 2, 2: 4, 3: 1, 4: 3}


##############
# LOAD MODEL

with open(f"models/kmeans_scaler_agri_v2.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(f"models/kmeans_{n_clusters}_clusters_agri_v2.pkl", "rb") as f:
    kmeans = pickle.load(f)



##############
# PREDICT SOIL GROUP

soil_files = os.listdir(soil_dir)
 
for i, f in enumerate(soil_files):

  if not os.path.exists(os.path.join(output_dir, f)):
      try:
        ds = xr.open_zarr(os.path.join(soil_dir, f)).compute()
        
        # Select bands, scale 
        X = ds[input_bands].to_dataframe().values
        X = X/10000
        X_scaled = scaler.transform(X)

        # Predict
        labels = kmeans.predict(X_scaled)
        new_labels = [label_map[label] for label in labels]

        # Reshape back to original xarray dataset with additional variable
        ds['soil_group'] = (('y', 'x'), np.array(new_labels).reshape(ds[input_bands[0]].shape))
        ds['soil_group'].attrs['long_name'] = 'Soil group'
        ds['soil_group'].attrs['description'] = 'Soil group predicted by KMeans clustering. Fill value is -10000'

        #ds['soil_group'].attrs['_FillValue'] = -10000
        #ds['soil_group'].transpose('y', 'x').expand_dims(band=[1]).rio.to_raster(f'SRC_preds_{i}.tif')
        
        # If MASK !=1 (it is 1 if htere is bare soil occurence), set soil group to -10000
        ds['soil_group'] = ds['soil_group'].where(ds['MASK'] == 1, -10000)
        # If any input band was -10000, set soil group to -10000 
        valid_data = (ds[input_bands] != -10000).to_array().all(dim='variable')  # True where all bands are valid
        ds['soil_group'] = ds['soil_group'].where(valid_data & (ds['MASK'] == 1), -10000)

        #ds['soil_group'].transpose('y', 'x').expand_dims(band=[1]).rio.to_raster(f'SRC_predsmask_{i}.tif')

        # Save data
        save_path = os.path.join(output_dir, f)
        ds.to_zarr(save_path, mode='w', consolidated=True)
        print(f"Saved prediction {i}/{len(soil_files)} to {save_path}")
   
      except Exception as e:
        print(f"Error processing {f}. Skipping file.")
        print(e)
        continue
