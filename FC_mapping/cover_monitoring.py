"""
Extract periods of bare soil vs covered

7 July 2025
Sélène Ledain selene.ledain@agroscope.admin.ch

- For each pixel, check thresholds of PV, NPV, Soil to determine bare and residual periods
- Append data and save
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import xarray as xr
import rioxarray
import pickle
import shutil
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import contextily as cx
import warnings
warnings.filterwarnings("ignore") #, message="invalid value encountered in divide")
import time
import pyproj
from shapely.ops import transform
from concurrent.futures import ProcessPoolExecutor, as_completed



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
  mask_dates[dates_to_drop] = False
  ds = ds.isel(time=mask_dates)

  return ds


def clean_dataset_optimized(ds, cloud_thresh=0.1, shadow_thresh=0.1, snow_thresh=0.1, cirrus_thresh=1000):
    """
    Drop dates with all 65535, and high cloud/snow/shadow or cirrus contamination.
    """
    """ 
    # Shape info
    nlat = ds.sizes['lat']
    nlon = ds.sizes['lon']
    npixels = nlat * nlon
    """

    # 1. Drop dates where all values in the dataset are 65535
    is_65535 = (ds == 65535).to_array().all(dim=['lat', 'lon'])
    all_65535 = is_65535.any(dim='variable')
    ds = ds.sel(time=~all_65535)

    # 2. Vectorized cloud, snow, shadow and cirrus masking
    """
    mask_clouds  = ((ds.s2_mask == 1) | ds.s2_SCL.isin([8, 9, 10])).sum(dim=['lat', 'lon']) / npixels > cloud_thresh
    mask_shadows = ((ds.s2_mask == 2) | (ds.s2_SCL == 3)).sum(dim=['lat', 'lon']) / npixels > shadow_thresh
    #mask_snow = ((ds.s2_mask == 3) | (ds.s2_SCL == 11)).sum(dim=['lat', 'lon']) / npixels > snow_thresh
    mask_snow = (((ds.s2_B03/10000 - ds.s2_B11/10000)/(ds.s2_B03/10000 + ds.s2_B11/10000)  > 0.4)).sum(dim=['lat', 'lon']) / npixels > snow_thresh #& (ds.s2_B08 > 0.11)).sum(dim=['lat', 'lon']) 
    """
    # Compute valid pixel count per timestamp (exclude pixels where s2_mask == 4)
    valid_pixels = (ds.s2_mask != 4).sum(dim=['lat', 'lon'])
    mask_clouds  = (((ds.s2_mask == 1) | ds.s2_SCL.isin([8, 9, 10])).sum(dim=['lat', 'lon']) / valid_pixels.data) > cloud_thresh
    mask_shadows = (((ds.s2_mask == 2) | (ds.s2_SCL == 3)).sum(dim=['lat', 'lon']) / valid_pixels.data) > shadow_thresh
    mask_snow = ((((ds.s2_B03 / 10000 - ds.s2_B11 / 10000) / (ds.s2_B03 / 10000 + ds.s2_B11 / 10000)) > 0.4) & (ds.s2_mask != 4)).sum(dim=['lat', 'lon']) / (ds.s2_mask != 4).sum(dim=['lat', 'lon']) > snow_thresh

    cirrus_mask = (ds.s2_SCL == 10)
    cirrus_b02_mean = ds.s2_B02.where(cirrus_mask).mean(dim=['lat', 'lon'])
    mask_cirrus = cirrus_b02_mean > cirrus_thresh

    # Combine masks
    drop_mask = mask_clouds | mask_shadows | mask_snow | mask_cirrus
    ds = ds.sel(time=~drop_mask)

    return ds


def extract_dominant_periods(ds, suffix, soil_thresh, npv_thresh):

  if f'PV_norm_{suffix}' not in ds.data_vars:
    return ds

  if npv_thresh is None:
    npv_dominant = (ds[f'NPV_norm_{suffix}'] > ds[f'PV_norm_{suffix}']) & (ds[f'NPV_norm_{suffix}'] > ds[f'Soil_norm_{suffix}'])
    # Preserve NaNs
    npv_dominant = npv_dominant.where(ds[f'NPV_norm_{suffix}'].notnull() &
                                       ds[f'PV_norm_{suffix}'].notnull() &
                                       ds[f'Soil_norm_{suffix}'].notnull())
  else:
    npv_dominant = (ds[f'NPV_norm_{suffix}'] > npv_thresh) 
    npv_dominant = npv_dominant.where(ds[f'NPV_norm_{suffix}'].notnull())

  if soil_thresh is None:
    soil_dominant = (ds[f'Soil_norm_{suffix}'] > ds[f'PV_norm_{suffix}']) & (ds[f'Soil_norm_{suffix}'] > ds[f'NPV_norm_{suffix}'])
    soil_dominant = soil_dominant.where(ds[f'Soil_norm_{suffix}'].notnull() &
                                         ds[f'PV_norm_{suffix}'].notnull() &
                                         ds[f'NPV_norm_{suffix}'].notnull())
  else:
    soil_dominant = (ds[f'Soil_norm_{suffix}'] > soil_thresh)
    soil_dominant = soil_dominant.where(ds[f'Soil_norm_{suffix}'].notnull())

  # store as 3D DataArrays: time x lat x lon
  npv_da = xr.DataArray(
      npv_dominant,
      dims=('lat', 'lon', 'time'),
      coords=ds.coords,
      name=f'npv_dominant_{suffix}'
  )
  soil_da = xr.DataArray(
      soil_dominant,
      dims=('lat', 'lon', 'time'),
      coords=ds.coords,
      name=f'soil_dominant_{suffix}'
  )

  return ds.assign({
      f'npv_dominant_{suffix}': npv_da,
      f'soil_dominant_{suffix}': soil_da
  })


def monitor_pixel(FC_dir, s2_dir, year, canton_poly, canton_name, fields, out_dir):
  
    if canton_name is not None:
        # Cantonal boundaries
        gdf = gpd.read_file(canton_poly, layer='tlm_kantonsgebiet') 
        gdf = gdf[gdf.name==canton_name]

        # Open fields in that area (canton)
        fields = gpd.read_file(fields, bbox=tuple(gdf.total_bounds))
        fields = gpd.clip(fields, gdf)
    else:
        # Open all fields
        fields = gpd.read_file(fields)

    fields = fields[~fields.geometry.is_empty & fields.geometry.notnull()]
    fields = fields[fields.is_valid]
    fields = fields.to_crs(32632)
    # Apply a 0-width buffer to fix subtle issues
    fields['geometry'] = fields.buffer(0)   
    #fields = fields.geometry[fields.geometry.is_valid & ~fields.geometry.is_empty]

    fc_files = [f for f in os.listdir(FC_dir) if f.endswith('.zarr') and f.split('_')[3].startswith(str(year))]
    gdf_FC_files = create_file_gdf(fc_files)
    gdf_FC_files = gdf_FC_files[gdf_FC_files.intersects(fields.union_all())]

    tot_files = len(gdf_FC_files)
    done_files = [os.path.join(out_dir, f) for f in os.listdir(out_dir)]

    for i, f in enumerate(gdf_FC_files.filename.tolist()):
        save_path = os.path.join(out_dir, f)
        if 1: #save_path not in done_files:

            print(f'Monitoring pixels in file {i}/{tot_files}')

            ds = xr.open_zarr(os.path.join(FC_dir, f)).compute()
            
            # Perform data cleaning
            s2 = xr.open_zarr(os.path.join(s2_dir, os.path.basename(f)))[['s2_mask', 's2_SCL', 's2_B02']].load()
            ds['s2_mask'] = s2['s2_mask'].transpose(*ds.dims)
            ds['s2_SCL']  = s2['s2_SCL'].transpose(*ds.dims)
            ds['s2_B02']  = s2['s2_B02'].transpose(*ds.dims)
            ds = clean_dataset(ds, cloud_thresh=0.05, snow_thresh=0.1, shadow_thresh=0.1, cirrus_thresh=800)

            try:
                ds = ds.rio.write_crs(32632).rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=False).rio.clip(fields.geometry, all_touched=False)
            except:
                continue
            
            # Extract periods where NPV dominant, Soil dominant OR provide thresholds to look for
            ds = extract_dominant_periods(ds, 'global', soil_thresh=None, npv_thresh=None)
            ds = extract_dominant_periods(ds, 'soil', soil_thresh=None, npv_thresh=None)
            
            # Save 
            ds["product_uri"] = (("time",), ds["product_uri"].values.astype(str))
            
            save_path = os.path.join(out_dir, f)
            ds.to_zarr(save_path, consolidated=True, mode='w')
            print(f"Saved to store {save_path}") 

        break           

    return


def create_hexplot_binary(data_dir, canton_name, canton_poly, suffix, week_start, week_end, save_path):
    
    year = week_start.split('-')[0]  

    fc_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.zarr') and f.split('_')[3].startswith(year)]
    gdf_FC_files = create_file_gdf(fc_files)

    if canton_name is not None:
        # Cantonal boundaries
        gdf = gpd.read_file(canton_poly, layer='tlm_kantonsgebiet') 
        gdf = gdf[gdf.name==canton_name].to_crs('EPSG:32632')
        gdf_FC_files = gdf_FC_files[gdf_FC_files.intersects(gdf.geometry.union_all())]

    all_points = []

    for path in gdf_FC_files.filename.tolist():
      try:
          ds = xr.open_zarr(path)
          ds_week = ds.sel(time=slice(week_start, week_end))
          if ds_week.time.size == 0:
              continue
        
          mask = (ds_week[f"npv_dominant_{suffix}"] == 0) & (ds_week[f"soil_dominant_{suffix}"] == 0)
          valid = ds_week[f"npv_dominant_{suffix}"].notnull() & ds_week[f"soil_dominant_{suffix}"].notnull()
          ds_week[f"pv_dominant_{suffix}"] = xr.where(valid, xr.where(mask, 1, 0), np.nan)

          # Mean, std, and count
          week_mean = ds_week[[f"soil_dominant_{suffix}", f"npv_dominant_{suffix}", f"pv_dominant_{suffix}"]].mean(dim="time")
          week_std = ds_week[[f"soil_dominant_{suffix}", f"npv_dominant_{suffix}", f"pv_dominant_{suffix}"]].std(dim="time")
          valid_counts = ds_week[[f"soil_dominant_{suffix}", f"npv_dominant_{suffix}", f"pv_dominant_{suffix}"]].notnull().sum(dim="time")

          # Convert to DataFrames
          df_mean = week_mean.to_dataframe().reset_index().rename(columns={
              f"soil_dominant_{suffix}": "soil_frac",
              f"npv_dominant_{suffix}": "npv_frac",
              f"pv_dominant_{suffix}": "pv_frac"
          })

          df_std = week_std.to_dataframe().reset_index().rename(columns={
              f"soil_dominant_{suffix}": "soil_std",
              f"npv_dominant_{suffix}": "npv_std",
              f"pv_dominant_{suffix}": "pv_std"
          })

          df_count = valid_counts.to_dataframe().reset_index().rename(columns={
              f"soil_dominant_{suffix}": "soil_count",
              f"npv_dominant_{suffix}": "npv_count",
              f"pv_dominant_{suffix}": "pv_count"
          })
          
          # Drop points outside of field geometries
          df_mean = df_mean[~(df_mean[['soil_frac', 'npv_frac', 'pv_frac']] == 0).all(axis=1)] # areas that are outside of geom (after clip) get put to 0
          df_std = df_std[~(df_std[['soil_std', 'npv_std', 'pv_std']] == 0).all(axis=1)] # areas that are outside of geom (after clip) get put to 0
          df_count = df_count[~(df_count[['soil_count', 'npv_count', 'npv_count']] == 0).all(axis=1)] # areas that are outside of geom (after clip) get put to 0
          

          df = df_mean.merge(df_std, on=['lat', 'lon']).merge(df_count, on=['lat', 'lon'])
          all_points.append(df)

      except Exception as e:
          print(f"Error processing {path}: {e}")

    
    
    if not len(all_points):
        return 

    df_all = pd.concat(all_points, ignore_index=True)

 
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(30, 24), constrained_layout=True)

    # Common parameters
    vmin, vmax = 0, 100
    gridsize = 20
    cmap_soil = LinearSegmentedColormap.from_list("white_brown", ["white", "saddlebrown"])
    cmap_npv = LinearSegmentedColormap.from_list("white_brown", ["white", "goldenrod"])
    cmap_pv = 'Greens'
    crs = 'EPSG:32632'
    basemap = cx.providers.SwissFederalGeoportal.NationalMapColor
    minx, miny, maxx, maxy = gdf.total_bounds

    # -------------------------
    # Row 0: Mean Fractions
    # -------------------------

    # 1. Soil Mean
    hb1 = axes[0, 0].hexbin(df_all["lon"], df_all["lat"], C=df_all["soil_frac"]*100,
                            gridsize=gridsize, cmap=cmap_soil,
                            reduce_C_function=np.nanmean, vmin=vmin, vmax=vmax)
    axes[0,0].set_xlim(minx, maxx)
    axes[0,0].set_ylim(miny, maxy)
    cx.add_basemap(axes[0, 0], source=basemap, crs=crs)
    axes[0, 0].set_title(f"Amount of fields with bare soil \n{week_start} – {week_end}")
    axes[0, 0].set_xlabel("Longitude")
    axes[0, 0].set_ylabel("Latitude")
    fig.colorbar(hb1, ax=axes[0, 0], label="%")

    # 2. NPV Mean
    hb2 = axes[0, 1].hexbin(df_all["lon"], df_all["lat"], C=df_all["npv_frac"]*100,
                            gridsize=gridsize, cmap=cmap_npv,
                            reduce_C_function=np.nanmean, vmin=vmin, vmax=vmax)
    axes[0,1].set_xlim(minx, maxx)
    axes[0,1].set_ylim(miny, maxy)
    cx.add_basemap(axes[0, 1], source=basemap, crs=crs)
    axes[0, 1].set_title(f"Amount of fields covered by residue \n{week_start} – {week_end}")
    axes[0, 1].set_xlabel("Longitude")
    axes[0, 1].set_ylabel("Latitude")
    fig.colorbar(hb2, ax=axes[0, 1], label="%")

    # 3. PV Mean
    hb3 = axes[0, 2].hexbin(df_all["lon"], df_all["lat"], C=df_all["pv_frac"]*100,
                            gridsize=gridsize, cmap=cmap_pv,
                            reduce_C_function=np.nanmean, vmin=vmin, vmax=vmax)
    axes[0,2].set_xlim(minx, maxx)
    axes[0,2].set_ylim(miny, maxy)
    cx.add_basemap(axes[0, 2], source=basemap, crs=crs)
    axes[0, 2].set_title(f"Amount of fields fully covered\n{week_start} – {week_end}")
    axes[0, 2].set_xlabel("Longitude")
    axes[0, 2].set_ylabel("Latitude")
    fig.colorbar(hb3, ax=axes[0, 2], label="%")

    # -------------------------
    # Row 1: Std Dev
    # -------------------------

    # You can adjust std vmax if it’s noisy or narrow:
    std_vmax = df_all[['soil_std', 'npv_std', 'pv_std']].max().max()

    # 1. Soil Std
    hb4 = axes[1, 0].hexbin(df_all["lon"], df_all["lat"], C=df_all["soil_std"],
                            gridsize=gridsize, cmap='Reds',
                            reduce_C_function=np.nanmean, vmin=0, vmax=std_vmax)
    axes[1,0].set_xlim(minx, maxx)
    axes[1,0].set_ylim(miny, maxy)
    cx.add_basemap(axes[1, 0], source=basemap, crs=crs)
    axes[1, 0].set_title(f"Standard deviation of bare soil fraction\n{week_start} – {week_end}")
    axes[1, 0].set_xlabel("Longitude")
    axes[1, 0].set_ylabel("Latitude")
    fig.colorbar(hb4, ax=axes[1, 0], label="Std Dev")

    # 2. NPV Std
    hb5 = axes[1, 1].hexbin(df_all["lon"], df_all["lat"], C=df_all["npv_std"],
                            gridsize=gridsize, cmap='Reds',
                            reduce_C_function=np.nanmean, vmin=0, vmax=std_vmax)
    axes[1,1].set_xlim(minx, maxx)
    axes[1,1].set_ylim(miny, maxy)
    cx.add_basemap(axes[1, 1], source=basemap, crs=crs)
    axes[1, 1].set_title(f"Standard deviation of NPV fraction\n{week_start} – {week_end}")
    axes[1, 1].set_xlabel("Longitude")
    axes[1, 1].set_ylabel("Latitude")
    fig.colorbar(hb5, ax=axes[1, 1], label="Std Dev")

    # 3. PV Std
    hb6 = axes[1, 2].hexbin(df_all["lon"], df_all["lat"], C=df_all["pv_std"],
                            gridsize=gridsize, cmap='Reds',
                            reduce_C_function=np.nanmean, vmin=0, vmax=std_vmax)
    axes[1,2].set_xlim(minx, maxx)
    axes[1,2].set_ylim(miny, maxy)
    cx.add_basemap(axes[1, 2], source=basemap, crs=crs)
    axes[1, 2].set_title(f"Standard deviation of PV fraction\n{week_start} – {week_end}")
    axes[1, 2].set_xlabel("Longitude")
    axes[1, 2].set_ylabel("Latitude")
    fig.colorbar(hb6, ax=axes[1, 2], label="Std Dev")

    """ 
    # Set consistent axis limits
    xlim = (df_all["lon"].min() - 1000, df_all["lon"].max() + 1000)
    ylim = (df_all["lat"].min() - 1000, df_all["lat"].max() + 1000)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
    """


    # -------------------------
    # Row 3: Valid pixel counts
    # -------------------------

    # 1. Soil
    hb7 = axes[2, 0].hexbin(df_all["lon"], df_all["lat"], C=df_all["soil_count"],
                            gridsize=gridsize, cmap='Reds',
                            reduce_C_function=np.sum)
    axes[2,0].set_xlim(minx, maxx)
    axes[2,0].set_ylim(miny, maxy)
    cx.add_basemap(axes[2, 0], source=basemap, crs=crs)
    axes[2, 0].set_title(f"alid pixel count bare soil\n{week_start} – {week_end}")
    axes[2, 0].set_xlabel("Longitude")
    axes[2, 0].set_ylabel("Latitude")
    fig.colorbar(hb7, ax=axes[2, 0], label="Mean Fraction")

    # 2. NPV
    hb8 = axes[2, 1].hexbin(df_all["lon"], df_all["lat"], C=df_all["npv_count"],
                            gridsize=gridsize, cmap='Reds',
                            reduce_C_function=np.sum)
    axes[2,1].set_xlim(minx, maxx)
    axes[2,1].set_ylim(miny, maxy)
    cx.add_basemap(axes[2, 1], source=basemap, crs=crs)
    axes[2, 1].set_title(f"alid pixel count NPV\n{week_start} – {week_end}")
    axes[2, 1].set_xlabel("Longitude")
    axes[2, 1].set_ylabel("Latitude")
    fig.colorbar(hb8, ax=axes[2, 1], label="Mean Fraction")

    # 3. PV
    hb9 = axes[2, 2].hexbin(df_all["lon"], df_all["lat"], C=df_all["pv_count"],
                            gridsize=gridsize, cmap='Reds',
                            reduce_C_function=np.sum)
    axes[2,2].set_xlim(minx, maxx)
    axes[2,2].set_ylim(miny, maxy)
    cx.add_basemap(axes[2, 2], source=basemap, crs=crs)
    axes[2, 2].set_title(f"Valid pixel count PV\n{week_start} – {week_end}")
    axes[2, 2].set_xlabel("Longitude")
    axes[2, 2].set_ylabel("Latitude")
    fig.colorbar(hb9, ax=axes[2, 2], label="Mean Fraction")


    # Save
    plt.savefig(save_path)


    return


def create_square(row):
    x_max = row['E_COORD']
    y_max = row['N_COORD']
    x_min = x_max - 100
    y_min = y_max - 100
    return box(x_min, y_min, x_max, y_max)


def process_file(path, suffix, week_start, week_end, s2_dir, fields, lu_csv=None, lnf_codes=None):
    try:
        ds = xr.open_zarr(path)
        if f'PV_norm_{suffix}' not in ds.data_vars:
            return None

        ds_week = ds.sel(time=slice(week_start, week_end))
        if ds_week.time.size == 0:
            return None

        # Load and clean S2
        s2 = xr.open_zarr(os.path.join(s2_dir, os.path.basename(path))).load().sel(time=slice(week_start, week_end)) #[['s2_mask', 's2_SCL', 's2_B02']]
        s2 = clean_dataset_optimized(s2, cloud_thresh=0.05, snow_thresh=0.1, shadow_thresh=0.1, cirrus_thresh=800)
        s2 = s2.drop_duplicates(dim='time', keep='first')
        mask = ds_week['product_uri'].isin(s2['product_uri'].compute().values).compute()
        ds_week = ds_week.where(mask, drop=True)
        
        if lu_csv is not None:
            # Clip fields
            f = os.path.basename(path)
            bbox_32632 = box(int(f.split('_')[1]), int(f.split('_')[2]), int(f.split('_')[1]) + 1280, int(f.split('_')[2]) + 1280)
            project = pyproj.Transformer.from_crs('EPSG:32632', 'EPSG:2056', always_xy=True).transform
            bbox_2056 = transform(project, bbox_32632)
            minx, miny, maxx, maxy = bbox_2056.bounds
            lu_csv = lu_csv[(lu_csv['E_COORD'] >= minx) & (lu_csv['E_COORD'] -100 <= maxx) & 
                            (lu_csv['N_COORD'] >= miny) & (lu_csv['N_COORD'] -100 <= maxy)]
    
            if len(lu_csv):
                lu_csv['geometry'] = lu_csv.apply(create_square, axis=1)
                gdf_lu = gpd.GeoDataFrame(lu_csv, geometry='geometry', crs='EPSG:2056').to_crs(32632)
            else:
                return None
            try:
                ds_week = ds_week.rio.write_crs(32632).rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=False).rio.clip(gdf_lu.geometry, all_touched=False)
                #if ds_week.isel(time=0)['NPV_norm_global'].mean().compute()>0.5:
                #    print(f, ds_week.time.values)
            except:
                return None # There is no data intersceting the field type 

        else:
            # Clip fields
            f = os.path.basename(path)
            bbox_32632 = box(int(f.split('_')[1]), int(f.split('_')[2]), int(f.split('_')[1]) + 1280, int(f.split('_')[2]) + 1280)
            project = pyproj.Transformer.from_crs('EPSG:32632', 'EPSG:2056', always_xy=True).transform
            bbox_2056 = transform(project, bbox_32632)
            fields_cube = gpd.read_file(fields, bbox=bbox_2056).to_crs(32632)
            fields_cube = fields_cube[fields_cube['geometry'].is_valid]

            if lnf_codes is not None:
                fields_cube = fields_cube[fields_cube.lnf_code.isin(lnf_codes)]
        
            try:
                ds_week = ds_week.rio.write_crs(32632).rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=False).rio.clip(fields_cube.geometry, all_touched=False)
            except:
                return None # There is no data intersceting the field type 

        # Compute stats
        week_mean = ds_week[[f"PV_norm_{suffix}", f"NPV_norm_{suffix}", f"Soil_norm_{suffix}"]].mean(dim="time")
        week_std = ds_week[[f"PV_norm_{suffix}", f"NPV_norm_{suffix}", f"Soil_norm_{suffix}"]].std(dim="time")
        valid_counts = ds_week[[f"PV_norm_{suffix}", f"NPV_norm_{suffix}", f"Soil_norm_{suffix}"]].notnull().sum(dim="time")

        # To dataframe
        df_mean = week_mean.to_dataframe().reset_index()
        df_std = week_std.to_dataframe().reset_index().rename(columns={
            f"PV_norm_{suffix}": "soil_std",
            f"NPV_norm_{suffix}": "npv_std",
            f"Soil_norm_{suffix}": "pv_std"
        })
        df_count = valid_counts.to_dataframe().reset_index().rename(columns={
            f"PV_norm_{suffix}": "soil_count",
            f"NPV_norm_{suffix}": "npv_count",
            f"Soil_norm_{suffix}": "pv_count"
        })

        # Filter 0 areas
        df_mean = df_mean[~(df_mean[[f'PV_norm_{suffix}', f'NPV_norm_{suffix}', f'Soil_norm_{suffix}']] == 0).all(axis=1)]
        df_std = df_std[~(df_std[['soil_std', 'npv_std', 'pv_std']] == 0).all(axis=1)]
        df_count = df_count[~(df_count[['soil_count', 'npv_count', 'npv_count']] == 0).all(axis=1)]

        df = df_mean.merge(df_std, on=['lat', 'lon']).merge(df_count, on=['lat', 'lon'])
        return df

    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None


def create_hexplot_fraction(data_dir, canton_name, canton_poly, fields, suffix, week_start, week_end, save_path, swisstopo_landuse=None, lnf_codes=None):
    
    year = week_start.split('-')[0]  

    fc_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.zarr') and f.split('_')[3].startswith(year)]
    gdf_FC_files = create_file_gdf(fc_files)

    if canton_name is not None:
        # Cantonal boundaries
        gdf = gpd.read_file(canton_poly, layer='tlm_kantonsgebiet') 
        gdf = gdf[gdf.name==canton_name].to_crs('EPSG:32632')
        gdf_FC_files = gdf_FC_files[gdf_FC_files.intersects(gdf.geometry.union_all())]

    
    if swisstopo_landuse is not None:
        # Further filter grassland based on other landuse data layer
        lu_csv = pd.read_csv(swisstopo_landuse,
            sep=';',
            usecols=['AS_17', 'E_COORD', 'N_COORD'],  # adjust columns to those needed in create_square
            dtype={'AS_17': 'int32', 'E_COORD': 'float64', 'N_COORD': 'float64'}  # faster parsing
        )
        lu_csv = lu_csv[lu_csv['AS_17'].isin([8, 9])].copy()
        
    all_points = []
    """
    for i, path in enumerate(gdf_FC_files.filename.tolist()):
        
        try:
            ds = xr.open_zarr(path)
            
            if f'PV_norm_{suffix}' not in ds.data_vars:
                continue

            ds_week = ds.sel(time=slice(week_start, week_end))
            if ds_week.time.size == 0:
                continue

         
            # Do cleaning: clouds/snow/shadow and duplicate timestamps
            s2 = xr.open_zarr(os.path.join(s2_dir, os.path.basename(path))).sel(time=slice(week_start, week_end)) #[['s2_mask', 's2_SCL', 's2_B02']].load()
            s2 = clean_dataset_optimized(s2, cloud_thresh=0.05, snow_thresh=0.1, shadow_thresh=0.1, cirrus_thresh=800)
            s2 = s2.drop_duplicates(dim='time', keep='first')
            mask = ds_week['product_uri'].isin(s2['product_uri'].compute().values).compute()
            ds_week = ds_week.where(mask, drop=True)
            #ds_week['s2_mask'] = s2['s2_mask'].transpose(*ds_week.dims)
            #ds_week['s2_SCL']  = s2['s2_SCL'].transpose(*ds_week.dims)
            #ds_week['s2_B02']  = s2['s2_B02'].transpose(*ds_week.dims)
            #ds_week = clean_dataset_optimized(ds_week, cloud_thresh=0.05, snow_thresh=0.1, shadow_thresh=0.1, cirrus_thresh=800)
            #ds_week = ds_week.drop_duplicates(dim='time', keep='first')
            
            # Clip for fields (open the fields file just in the extent of the datacube)
            f = os.path.basename(path)
            bbox_32632 = box(int(f.split('_')[1]), int(f.split('_')[2]), int(f.split('_')[1]) + 1280, int(f.split('_')[2]) + 1280)
            project = pyproj.Transformer.from_crs('EPSG:32632', 'EPSG:2056', always_xy=True).transform
            bbox_2056 = transform(project, bbox_32632)
            fields_cube = gpd.read_file(fields, bbox=bbox_2056).to_crs(32632) 
            fields_cube = fields_cube[fields_cube['geometry'].is_valid]

        
            if lnf_codes is not None:
                fields_cube = fields_cube[fields_cube.lnf_code.isin(lnf_codes)]
                
            try:
                ds_week = ds_week.rio.write_crs(32632).rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=False).rio.clip(fields_cube.geometry, all_touched=False)
            except:
                continue
            
            # Mean, std, and count
            week_mean = ds_week[[f"PV_norm_{suffix}", f"NPV_norm_{suffix}", f"Soil_norm_{suffix}"]].mean(dim="time")
            week_std = ds_week[[f"PV_norm_{suffix}", f"NPV_norm_{suffix}", f"Soil_norm_{suffix}"]].std(dim="time")
            valid_counts = ds_week[[f"PV_norm_{suffix}", f"NPV_norm_{suffix}", f"Soil_norm_{suffix}"]].notnull().sum(dim="time")

            # Convert to DataFrames
            df_mean = week_mean.to_dataframe().reset_index()
            df_std = week_std.to_dataframe().reset_index().rename(columns={
                f"PV_norm_{suffix}": "soil_std",
                f"NPV_norm_{suffix}": "npv_std",
                f"Soil_norm_{suffix}": "pv_std"
            })
            df_count = valid_counts.to_dataframe().reset_index().rename(columns={
                f"PV_norm_{suffix}": "soil_count",
                f"NPV_norm_{suffix}": "npv_count",
                f"Soil_norm_{suffix}": "pv_count"
            })

            # Drop points outside of field geometries
            df_mean = df_mean[~(df_mean[[f'PV_norm_{suffix}', f'NPV_norm_{suffix}', f'Soil_norm_{suffix}']] == 0).all(axis=1)] # areas that are outside of geom (after clip) get put to 0
            df_std = df_std[~(df_std[['soil_std', 'npv_std', 'pv_std']] == 0).all(axis=1)] # areas that are outside of geom (after clip) get put to 0
            df_count = df_count[~(df_count[['soil_count', 'npv_count', 'npv_count']] == 0).all(axis=1)] # areas that are outside of geom (after clip) get put to 0

            df = df_mean.merge(df_std, on=['lat', 'lon']).merge(df_count, on=['lat', 'lon'])
            all_points.append(df)
            print(f'Processed file {i}/{len(gdf_FC_files.filename.tolist())}')

        except Exception as e:
            print(f"Error processing {path}: {e}")
    """

    paths = gdf_FC_files.filename.tolist()
    with ProcessPoolExecutor(max_workers=10) as executor:  
        futures = {
            executor.submit(process_file, path, suffix, week_start, week_end, s2_dir, fields, lu_csv, lnf_codes): path
            for path in paths
        }

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result is not None:
                all_points.append(result)
            print(f"Processed file {i + 1}/{len(paths)}")
   

    if not len(all_points):
        return 

    df_all = pd.concat(all_points, ignore_index=True)
    if not len(df_all):
        return
    
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(40, 24), constrained_layout=True)

    # Common parameters
    vmin, vmax = 0, 100
    gridsize = 20
    cmap_soil = LinearSegmentedColormap.from_list("white_brown", ["white", "saddlebrown"])
    cmap_npv = LinearSegmentedColormap.from_list("white_brown", ["white", "goldenrod"])
    cmap_pv = 'Greens'
    crs = 'EPSG:32632'
    basemap = cx.providers.SwissFederalGeoportal.NationalMapColor

    # -------------------------
    # Row 0: Mean Fractions
    # -------------------------

    # 1. Soil Mean
    hb1 = axes[0, 0].hexbin(df_all["lon"], df_all["lat"], C=df_all[f"Soil_norm_{suffix}"]*100,
                            gridsize=gridsize, cmap=cmap_soil,
                            reduce_C_function=np.nanmean, vmin=vmin, vmax=vmax)
    cx.add_basemap(axes[0, 0], source=basemap, crs=crs)
    axes[0, 0].set_title(f"Average fraction of bare soil \n{week_start} – {week_end}")
    axes[0, 0].set_xlabel("Longitude")
    axes[0, 0].set_ylabel("Latitude")
    fig.colorbar(hb1, ax=axes[0, 0], label="%")

    # 2. NPV Mean
    hb2 = axes[0, 1].hexbin(df_all["lon"], df_all["lat"], C=df_all[f"NPV_norm_{suffix}"]*100,
                            gridsize=gridsize, cmap=cmap_npv,
                            reduce_C_function=np.nanmean, vmin=vmin, vmax=vmax)
    cx.add_basemap(axes[0, 1], source=basemap, crs=crs)
    axes[0, 1].set_title(f"Average fraction of residue \n{week_start} – {week_end}")
    axes[0, 1].set_xlabel("Longitude")
    axes[0, 1].set_ylabel("Latitude")
    fig.colorbar(hb2, ax=axes[0, 1], label="%")

    # 3. PV Mean
    hb3 = axes[0, 2].hexbin(df_all["lon"], df_all["lat"], C=df_all[f"PV_norm_{suffix}"]*100,
                            gridsize=gridsize, cmap=cmap_pv,
                            reduce_C_function=np.nanmean, vmin=vmin, vmax=vmax)
    cx.add_basemap(axes[0, 2], source=basemap, crs=crs)
    axes[0, 2].set_title(f"Average fraction of vegetation \n{week_start} – {week_end}")
    axes[0, 2].set_xlabel("Longitude")
    axes[0, 2].set_ylabel("Latitude")
    fig.colorbar(hb3, ax=axes[0, 2], label="%")

    # -------------------------
    # Row 1: Std Dev
    # -------------------------

    # You can adjust std vmax if it’s noisy or narrow:
    std_vmax = df_all[['soil_std', 'npv_std', 'pv_std']].max().max()

    # 1. Soil Std
    hb4 = axes[1, 0].hexbin(df_all["lon"], df_all["lat"], C=df_all["soil_std"],
                            gridsize=gridsize, cmap='Reds',
                            reduce_C_function=np.nanmean, vmin=0, vmax=std_vmax)
    cx.add_basemap(axes[1, 0], source=basemap, crs=crs)
    axes[1, 0].set_title(f"Standard deviation of bare soil fraction\n{week_start} – {week_end}")
    axes[1, 0].set_xlabel("Longitude")
    axes[1, 0].set_ylabel("Latitude")
    fig.colorbar(hb4, ax=axes[1, 0], label="Std Dev")

    # 2. NPV Std
    hb5 = axes[1, 1].hexbin(df_all["lon"], df_all["lat"], C=df_all["npv_std"],
                            gridsize=gridsize, cmap='Reds',
                            reduce_C_function=np.nanmean, vmin=0, vmax=std_vmax)
    cx.add_basemap(axes[1, 1], source=basemap, crs=crs)
    axes[1, 1].set_title(f"Standard deviation of NPV fraction\n{week_start} – {week_end}")
    axes[1, 1].set_xlabel("Longitude")
    axes[1, 1].set_ylabel("Latitude")
    fig.colorbar(hb5, ax=axes[1, 1], label="Std Dev")

    # 3. PV Std
    hb6 = axes[1, 2].hexbin(df_all["lon"], df_all["lat"], C=df_all["pv_std"],
                            gridsize=gridsize, cmap='Reds',
                            reduce_C_function=np.nanmean, vmin=0, vmax=std_vmax)
    cx.add_basemap(axes[1, 2], source=basemap, crs=crs)
    axes[1, 2].set_title(f"Standard deviation of PV fraction\n{week_start} – {week_end}")
    axes[1, 2].set_xlabel("Longitude")
    axes[1, 2].set_ylabel("Latitude")
    fig.colorbar(hb6, ax=axes[1, 2], label="Std Dev")


    # -------------------------
    # Row 3: Valid pixel counts
    # -------------------------

    # 1. Soil
    hb7 = axes[2, 0].hexbin(df_all["lon"], df_all["lat"], C=df_all["soil_count"],
                            gridsize=gridsize, cmap='Reds',
                            reduce_C_function=np.sum)
    cx.add_basemap(axes[2, 0], source=basemap, crs=crs)
    axes[2, 0].set_title(f"alid pixel count bare soil\n{week_start} – {week_end}")
    axes[2, 0].set_xlabel("Longitude")
    axes[2, 0].set_ylabel("Latitude")
    fig.colorbar(hb7, ax=axes[2, 0], label="Mean Fraction")

    # 2. NPV
    hb8 = axes[2, 1].hexbin(df_all["lon"], df_all["lat"], C=df_all["npv_count"],
                            gridsize=gridsize, cmap='Reds',
                            reduce_C_function=np.sum)
    cx.add_basemap(axes[2, 1], source=basemap, crs=crs)
    axes[2, 1].set_title(f"alid pixel count NPV\n{week_start} – {week_end}")
    axes[2, 1].set_xlabel("Longitude")
    axes[2, 1].set_ylabel("Latitude")
    fig.colorbar(hb8, ax=axes[2, 1], label="Mean Fraction")

    # 3. PV
    hb9 = axes[2, 2].hexbin(df_all["lon"], df_all["lat"], C=df_all["pv_count"],
                            gridsize=gridsize, cmap='Reds',
                            reduce_C_function=np.sum)
    cx.add_basemap(axes[2, 2], source=basemap, crs=crs)
    axes[2, 2].set_title(f"Valid pixel count PV\n{week_start} – {week_end}")
    axes[2, 2].set_xlabel("Longitude")
    axes[2, 2].set_ylabel("Latitude")
    fig.colorbar(hb9, ax=axes[2, 2], label="Mean Fraction")

    # Set consistent axis limits
    xlim = (260000, 620000) #(df_all["lon"].min() - 1000, df_all["lon"].max() + 1000)
    ylim = (5050000, 5300000) #(df_all["lat"].min() - 1000, df_all["lat"].max() + 1000)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)


    # Save
    plt.savefig(save_path)


    return




if __name__ == '__main__':

    FC_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/FC')
    s2_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH')
    year = 2021
    canton_poly = 'swissBOUNDARIES3D_1_5_LV95_LN02.gpkg'
    canton_name = None #'Valais'
    fields = os.path.expanduser(f'~/mnt/eo-nas1/data/landuse/raw/lnf{year}.gpkg')
    #field_raster = os.path.expanduser(f'~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/pv_npv_members/crop_maps/lnf_code_{year}.tif')
    out_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/FC_binary')
    suffix = 'global'
    

    ###########################
    # AVG FRACTION: avg PV/NPV/Soil fraction per hex --> filter first for some lnf code


    grass = True # if True, creates map for grasslands. If False, will create for arable land
    swisstopo_landuse = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/EVI_CH/ag-b-00.03-37-area-current-csv.csv') # Classification on areal imagey of landuse, classes 8 and 9 are grassland


    ### Open data on LNF codes
    crop_labels = os.path.expanduser('~/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20250217.xlsx')
    crop_names = pd.read_excel(crop_labels, sheet_name=1) # important cols: LNF_code, categories2024
    # Drop some categories
    to_drop = ['Apples', 'Asparagus', 'Berries', 'Biodiversity promotion area', 'Chestnut', 'Fallow', 'Fallow extensive', 'Field margin',\
            'Flowerstrips', 'Forest', 'Forest Pasture', 'Gardens', 'Greenhouses', 'Greenhouse', 'Hedge', 'Non agriculture', 'Nurseries',\
            'Orchards', 'Orchards other', 'Other', 'Other indoor', 'Paths natural', 'Pears', 'Permaculture', 'Special cultures', 'Tree Crop',\
            'Vines'] 
    crop_names = crop_names[~crop_names.categories2024.isin(to_drop)]
    # Group up some labels
    crop_names.loc[crop_names.categories2024.isin(['Meadow', 'Meadow extensiv', 'Meadow other', 'Meadow permanent', 'Meadow sown', 'Summering Meadow', 'Summering Meadow Extensiv']), 'categories2024'] = 'Grassland'
    crop_names.loc[crop_names.categories2024.isin(['Pasture', 'Pasture extensiv', 'Summering Pasture']), 'categories2024'] = 'Grassland'

    # Create mapping from LNF code to crop category
    #lnf_mapping = dict(zip(crop_names["LNF_code"], crop_names["categories2024"]))

    

    if grass: 
        # Get codes of grassland
        lnf_codes = crop_names[crop_names.categories2024=='Grassland'].LNF_code.tolist()

        for m in range(1,13):
            if m < 10:
                month_start = f'0{m}'
                month_end = f'0{m+1}' if m<9 else str(m+1)
            else:
                month_start = str(m)
                month_end = str(m+1) if m!=12 else '01'
            create_hexplot_fraction(data_dir=FC_dir, canton_name=canton_name, canton_poly=canton_poly, fields=fields, suffix=suffix, week_start=f'2021-{month_start}-01', week_end=f'2021-{month_end}-01', swisstopo_landuse=swisstopo_landuse, lnf_codes=lnf_codes, save_path=f'CH_fraction_monthly/CH_fraction_{m}_{suffix}_grass_lu_ndsi2.png')
    

    else:
        # Get code of arable land
        lnf_codes = crop_names[crop_names.categories2024!='Grassland'].LNF_code.tolist()

        for m in range(1,13):
            if m < 10:
                month_start = f'0{m}'
                month_end = f'0{m+1}' if m<9 else str(m+1)
            else:
                month_start = str(m)
                month_end = str(m+1) if m!=12 else '01'
            create_hexplot_fraction(data_dir=FC_dir, canton_name=canton_name, canton_poly=canton_poly, fields=fields, suffix=suffix, week_start=f'2021-{month_start}-01', week_end=f'2021-{month_end}-01', lnf_codes=lnf_codes, save_path=f'CH_fraction_monthly/CH_fraction_{m}_{suffix}_arable_debug.png')
    






