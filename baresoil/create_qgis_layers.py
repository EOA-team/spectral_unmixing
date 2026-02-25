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



######################
# 1. Saving different clusters to shp

# Rename clusters based on final definition
label_map = {0: 5, 1: 2, 2: 4, 3: 1, 4: 3}


for n_optim in range(5,6): 

    # Load the saved predictions
    file_name = f"sampled_data_{n_optim}_clusters_agri_v2.csv"
    samples = pd.read_csv(file_name)

    # Rename
    new_labels = [label_map[label] for label in samples['cluster'].values]
    samples['cluster'] = new_labels

    # Save each cluster as shp
    gdf = gpd.GeoDataFrame(samples, geometry=gpd.points_from_xy(samples['x'], samples['y']), crs="EPSG:3035").to_crs(epsg=2056)
    
    for cluster in range(1,n_optim+1): #range(n_optim) if not renamed
        gdf_cluster = gdf[gdf['cluster'] == cluster][['x', 'y', 'cluster', 'geometry']]
        save_name = f"qgis_soilclusters/k{n_optim}_cluster{cluster}_agri_v2_renamed.shp" 
        gdf_cluster.to_file(save_name)
