import os
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import Counter
import xarray as xr
import rioxarray
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import warnings
warnings.simplefilter("ignore")
import sys
sys.path.insert(0, os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/SALI_models'))
from src.model_utils import compute_FC, compute_FC_grassland


def sample_locations(crop_labels, lnf_dir, tot_samples, save_path, seed=42):
    """Sample locations with a certain crop type, uniformly across crop types and available years (yearly crop maps) (EPSG:2056)"""

    lnf_files = sorted([f for f in os.listdir(lnf_dir) if f.endswith('.gpkg')])
    n_years = len(lnf_files)
    n_crops = len(top_lnfs_arable)
    samples_per_year = int(np.floor(tot_samples/n_years))
    samples_per_crop = int(np.floor(samples_per_year/n_crops))

    print(f'Sampling {tot_samples}: {samples_per_year} per year, {samples_per_crop} per crop in a year')
    all_samples = []
    for lnf_yr in lnf_files:
        yr = lnf_yr.split('lnf')[-1].split('.gpkg')[0]
        lnf = gpd.read_file(os.path.join(lnf_dir, lnf_yr))
        # Set all grasslands to same class
        lnf.loc[lnf['lnf_code'].isin(lnfs_grassland), 'lnf_code'] = lnfs_grassland[0]
        # Filter to keep only relevant polys
        lnf = lnf[lnf.lnf_code.isin(top_lnfs_arable)]
        lnf["poly_id"] = lnf.index
        if not len(lnf):
            continue
        seed += 1 # so that among differetn years it varies
        for crop in top_lnfs_arable:
            crop_polys = lnf[lnf.lnf_code == crop]

            sampled_polys = crop_polys.sample(
                n=samples_per_crop,
                weights=crop_polys.area,
                replace=True,
                random_state=seed
            )

            buffered = sampled_polys.copy()
            buffered["geometry"] = buffered.geometry.buffer(-10)
            pts = buffered.sample_points(1, random_state=seed)

            pts = pts.explode(index_parts=False).reset_index()
            pts = gpd.GeoDataFrame(pts, geometry='sampled_points', crs=crop_polys.crs)\
                .rename(columns={'sampled_points': "point_geom"})
            pts = pts.merge(
                crop_polys[["poly_id", "geometry", "lnf_code"]],
                left_on="index",
                right_index=True
            )
            pts["x"] = pts.point_geom.x
            pts["y"] = pts.point_geom.y
            pts["yr"] = yr

            all_samples.append(pts)

    samples = gpd.GeoDataFrame(pd.concat(all_samples, ignore_index=True), geometry="point_geom", crs=lnf.crs)
    samples.to_pickle(save_path)

    return


def sample_locations_with_field(crop_labels, lnf_dir, tot_samples, save_path, target_crs=32632, seed=42):
    """Sample locations with a certain crop type, uniformly across crop types and available years (yearly crop maps)"""

    lnf_files = sorted([f for f in os.listdir(lnf_dir) if f.endswith('.gpkg')])
    n_years = len(lnf_files)
    n_crops = len(top_lnfs_arable)
    samples_per_year = int(np.floor(tot_samples/n_years))
    samples_per_crop = int(np.floor(samples_per_year/n_crops))

    print(f'Sampling {tot_samples}: {samples_per_year} per year, {samples_per_crop} per crop in a year')
    all_samples = []
    for lnf_yr in lnf_files:
        yr = lnf_yr.split('lnf')[-1].split('.gpkg')[0]
        lnf = gpd.read_file(os.path.join(lnf_dir, lnf_yr))
        # Set all grasslands to same class
        lnf.loc[lnf['lnf_code'].isin(lnfs_grassland), 'lnf_code'] = lnfs_grassland[0]
        # Filter to keep only relevant polys
        lnf = lnf[lnf.lnf_code.isin(top_lnfs_arable)]
        lnf["poly_id"] = lnf.index
        if not len(lnf):
            continue
        seed += 1 # so that among differetn years it varies
        for crop in top_lnfs_arable:
            crop_polys = lnf[lnf.lnf_code == crop].to_crs(target_crs)

            sampled_polys = crop_polys.sample(
                n=samples_per_crop,
                weights=crop_polys.area,
                replace=True,
                random_state=seed
            )

            buffered = sampled_polys.copy()
            buffered["geometry"] = buffered.geometry.buffer(-10)
            pts = buffered.sample_points(1, random_state=seed)

            pts = pts.explode(index_parts=False).reset_index()
            pts = gpd.GeoDataFrame(pts, geometry='sampled_points', crs=target_crs)\
                .rename(columns={'sampled_points': "point_geom"})
            pts = pts.merge(
                crop_polys[["poly_id", "geometry", "lnf_code"]],
                left_on="index",
                right_index=True
            )

            # Reproject to target CRS
            #pts = pts.set_geometry("geometry").to_crs(epsg=target_crs)
            pts = pts.rename(columns={"geometry": "polygon_geom"})
            
            pts["x"] = pts.point_geom.x
            pts["y"] = pts.point_geom.y
            pts["yr"] = yr

            all_samples.append(pts)

    samples = gpd.GeoDataFrame(pd.concat(all_samples, ignore_index=True), geometry="point_geom", crs=target_crs)
    samples.to_pickle(save_path)

    print(samples)

    return


def snap_to_grid(x, y, ds):

    res = abs(ds.rename({'lat':'y', 'lon':'x'}).rio.resolution()[0])

    x0 = ds.lon.values[0]
    y0 = ds.lat.values[0]

    snapped_x = x0 + np.round((x - x0) / res) * res
    snapped_y = y0 + np.round((y - y0) / res) * res

    return snapped_x, snapped_y


def extract_s2_data(save_path_sampledloc, s2_grid_path, s2_dir, soil_dir, save_path_sampledloc_S2):
    print('Extract S2 (30 Jun previous yr to 1 Jul current yr) and Soil data')
    samples = pd.read_pickle(save_path_sampledloc).to_crs(32632)

    # Intersect S2 grid with locations of samples to get list of files and coords concerned
    grid = gpd.read_file(s2_grid_path)
    grid_samples = gpd.overlay(samples, grid, how='intersection') # keeps sample point geometry and S2 name
    s2_files = [f for f in os.listdir(s2_dir)]
    
    soil_cache = {}
    df_samples = []
    for (left, top, yr), pts_df in grid_samples.groupby(['left', 'top', 'yr']):
        #print(f'Extracting for {yr}: {left}-{top}')
        # Extract S2 data
        file_str1 = f"S2_{int(left)}_{int(top)}_{yr}"
        file_str2 = f"S2_{int(left)}_{int(top)}_{int(yr)-1}"
        f = [os.path.join(s2_dir, f) for f in s2_files if f.startswith((file_str1, file_str2))]
        ds_s2 = xr.open_mfdataset(f).sel(time=slice(f'{int(yr)-1}-06-30', f'{yr}-07-01'))
        ds_s2_sampled = ds_s2.sel(
            lon=xr.DataArray(pts_df["x"].values, dims="points"),
            lat=xr.DataArray(pts_df["y"].values, dims="points"),
            method="nearest"
        )
        ds_s2_sampled = ds_s2_sampled.assign_coords(
            sampled_x=("points", pts_df["x"].values),
            sampled_y=("points", pts_df["y"].values),
            lnf_code=("points", pts_df["lnf_code"].values)
        )
        df_s2_sampled = ds_s2_sampled.to_dataframe().reset_index()
        df_s2_sampled["yr"] = yr

        # Extract soil data: if sampled coord is not labeled, look at mode in 10x10 pixels around
        key = (left, top)
        if key not in soil_cache:
            soil_file = os.path.join(soil_dir, f'SRC_{int(left)}_{int(top)}.zarr')
            if not os.path.exists(soil_file):
                soil_group = 0
            else:
                ds_soil = xr.open_zarr(os.path.join(soil_dir, f'SRC_{int(left)}_{int(top)}.zarr'))
                ds_soil_sampled = ds_soil.sel(
                    x=xr.DataArray(pts_df["x"].values, dims="points"),
                    y=xr.DataArray(pts_df["y"].values, dims="points"),
                    method="nearest"
                )
                soil_group = ds_soil_sampled.soil_group.values[0]
                
                if int(soil_group) == -10000:
                    offsets = np.arange(-5, 5) * 10
                    dx, dy = np.meshgrid(offsets, offsets)
                    dx = dx.ravel()
                    dy = dy.ravel()
                    expanded = pts_df.loc[pts_df.index.repeat(len(dx))].copy()
                    expanded["x"] = expanded["x"].values + np.tile(dx, len(pts_df))
                    expanded["y"] = expanded["y"].values + np.tile(dy, len(pts_df))
                    
                    ds_soil_around = ds_soil.sel(
                        x=xr.DataArray(expanded["x"].values, dims="points"),
                        y=xr.DataArray(expanded["y"].values, dims="points"),
                        method="nearest"
                    )
                    soil_around = ds_soil_around.soil_group.values
                    soil_around = [s for s in soil_around if int(s) != -10000]
                    if len(soil_around):
                        soil_group = Counter(soil_around).most_common(1)[0][0]
                    else:
                        soil_group = 0
                
                # Save for later
                soil_cache[key] = soil_group 
        else:
            soil_group = soil_cache[key]

        df_s2_sampled['soil_group'] = soil_group
        df_samples.append(df_s2_sampled)

    df_samples = pd.concat(df_samples, ignore_index=True)
    df_samples.to_pickle(save_path_sampledloc_S2)

    return


def extract_s2_data_field(save_path_sampledloc, s2_grid_path, s2_dir, soil_dir, save_path_sampledloc_S2):
    print('Extract S2 (30 Jun previous yr to 1 Jul current yr) and Soil data')
    samples = pd.read_pickle(save_path_sampledloc)

    # Intersect S2 grid with locations of samples to get list of files and coords concerned
    grid = gpd.read_file(s2_grid_path)
    grid_samples = gpd.overlay(samples, grid, how='intersection') # keeps sample point geometry and S2 name
    s2_files = [f for f in os.listdir(s2_dir)]
    
    soil_cache = {}
    df_samples = []
    for (left, top, yr), pts_df in grid_samples.groupby(['left', 'top', 'yr']):
        #print(f'Extracting for {yr}: {left}-{top}')
        # Extract S2 data
        file_str1 = f"S2_{int(left)}_{int(top)}_{yr}"
        file_str2 = f"S2_{int(left)}_{int(top)}_{int(yr)-1}"
        f = [os.path.join(s2_dir, f) for f in s2_files if f.startswith((file_str1, file_str2))]
        ds_s2 = xr.open_mfdataset(f).sel(time=slice(f'{int(yr)-1}-06-30', f'{yr}-07-01'))
        
        # Extract data for each polygon
        for _, row in pts_df.iterrows():
            polygon = row["polygon_geom"]  # Use the polygon geometry
            mask = ds_s2.rename({'lat':'y', 'lon':'x'}).rio.write_crs(32632).rio.clip([polygon], ds_s2.rio.crs, drop=True)  # Clip S2 data to the polygon
            df_s2_sampled = mask.to_dataframe().reset_index()
            df_s2_sampled["lnf_code"] = row["lnf_code"]
            df_s2_sampled["yr"] = yr
            df_s2_sampled["poly_id"] = row["poly_id"]

            # Extract soil data for the polygon
            key = (left, top)
            if key not in soil_cache:
                soil_file = os.path.join(soil_dir, f'SRC_{int(left)}_{int(top)}.zarr')
                if not os.path.exists(soil_file):
                    soil_group = 0
                else:
                    ds_soil = xr.open_zarr(soil_file).astype("float32")
                    ds_soil_clipped = ds_soil.rio.write_crs(32632).rio.clip([polygon], 32632, drop=True)
                    soil_values = ds_soil_clipped["soil_group"].values.flatten()
                    soil_values = soil_values[(soil_values != -10000) & (~np.isnan(soil_values))] # Remove invalid values
                    if len(soil_values):
                        soil_group = Counter(soil_values).most_common(1)[0][0]
                    else:
                        soil_group = 0
                soil_cache[key] = soil_group
            else:
                soil_group = soil_cache[key]

            # Find the row corresponding to sampled point in poly
            sampled_x = row["x"]
            sampled_y = row["y"]
            df_s2_sampled['x'] = df_s2_sampled['x'].apply(lambda x: x+5)
            df_s2_sampled['y'] = df_s2_sampled['y'].apply(lambda y: y-5)
            snap_x, snap_y = snap_to_grid(sampled_x, sampled_y, ds_s2)
            df_s2_sampled["is_sample_pixel"] = (
                (df_s2_sampled["x"] == snap_x) &
                (df_s2_sampled["y"] == snap_y)
            )
            df_s2_sampled["sampled_x"] = sampled_x
            df_s2_sampled["sampled_y"] = sampled_y
            df_s2_sampled['x'] = df_s2_sampled['x'].apply(lambda x: x-5)
            df_s2_sampled['y'] = df_s2_sampled['y'].apply(lambda y: y+5)
                
            df_s2_sampled['soil_group'] = soil_group
            df_samples.append(df_s2_sampled)

    df_samples = pd.concat(df_samples, ignore_index=True)
    df_samples.to_pickle(save_path_sampledloc_S2)

    return


def predict_FC(save_path_sampledloc_S2, save_path_FCpreds):
    print('Predict FC')

    bands = ['s2_B02','s2_B03','s2_B04','s2_B05','s2_B06','s2_B07','s2_B08','s2_B8A','s2_B11','s2_B12']
    df_samples = pd.read_pickle(save_path_sampledloc_S2)

    # Prepare data
    df_samples[df_samples==65535] = np.nan
    df_samples = df_samples.dropna()
    df_samples[bands] /= 10000

    # Predict using correct model
    df_global = df_samples[df_samples.soil_group==0].copy()
    df_soil   = df_samples[df_samples.soil_group!=0].copy()

    data_global = df_samples[df_samples.soil_group==0][bands].to_numpy()
    data_soil = df_samples[df_samples.soil_group!=0][bands].to_numpy()

    all_preds_global = []
    all_preds_soil =  []
    chunk_size = 10000
    for i in range(0, len(data_global), chunk_size):
        chunk = data_global[i:i+chunk_size]
        preds = compute_FC_grassland(chunk)   # pv, npv, soil
        all_preds_global.append(preds)
    for i in range(0, len(data_soil), chunk_size):
        chunk = data_soil[i:i+chunk_size]
        preds = compute_FC(chunk)   # pv, npv, soil
        all_preds_soil.append(preds)

    # Combine results
    preds_global = np.vstack(all_preds_global)
    preds_soil = np.vstack(all_preds_soil)

    # Assing predictions
    df_global[['pv', 'npv', 'soil']] = preds_global
    
    df_soil = df_soil.copy()
    df_soil['row_idx'] = np.arange(len(df_soil))
    preds_soil = preds_soil.copy()
    preds_soil['row_idx'] = np.tile(np.arange(len(df_soil)), 5)
    df_soil = pd.merge(df_soil, preds_soil, left_on=['row_idx', 'soil_group'], right_on=['row_idx', 'soil_id'])

    df_samples = pd.concat([df_global, df_soil]).sort_index().drop(columns=['row_idx', 'soil_id','points'])
    df_samples.to_pickle(save_path_FCpreds)

    return


def clean_timeseries_df(group, cirrus_thresh=1000):

    band_cols = [c for c in group.columns if c.startswith('s2_B')]

    # Missing data
    missing_mask = (group[band_cols].isna()).all(axis=1)

    # Clouds
    cloud_mask = (
        (group['s2_mask'] == 1) |
        (group['s2_SCL'].isin([8, 9, 10]))
    )

    # Shadows
    shadow_mask = (
        (group['s2_mask'] == 2) |
        (group['s2_SCL'] == 3)
    )

    # Snow
    snow_mask = (
        (group['s2_mask'] == 3) |
        (group['s2_SCL'] == 11)
    )

    # Cirrus
    cirrus_mask = (
        (group['s2_SCL'] == 10) &
        (group['s2_B02'] > cirrus_thresh)
    )

    drop_mask = (
        missing_mask |
        cloud_mask |
        shadow_mask |
        snow_mask |
        cirrus_mask
    )

    return group.loc[~drop_mask]


def clean_timeseries_field(field_df, cirrus_thresh=1000, max_missing_frac=0.2):
    """
    field_df: all pixels in one field for a given year (or grouping)
    max_missing_frac: drop dates where more than this fraction of pixels are masked
    """
    band_cols = [c for c in field_df.columns if c.startswith('s2_B')]

    # Create masks for all pixels
    missing_mask = (field_df[band_cols].isna()).all(axis=1)
    cloud_mask = (field_df['s2_mask'] == 1) | (field_df['s2_SCL'].isin([8, 9, 10]))
    shadow_mask = (field_df['s2_mask'] == 2) | (field_df['s2_SCL'] == 3)
    snow_mask = (field_df['s2_mask'] == 3) | (field_df['s2_SCL'] == 11)
    cirrus_mask = (field_df['s2_SCL'] == 10) & (field_df['s2_B02'] > cirrus_thresh)

    # Combine masks
    drop_mask = missing_mask | cloud_mask | shadow_mask | snow_mask | cirrus_mask
    field_df['masked'] = drop_mask.astype(int)

    # Compute fraction of masked pixels per date
    masked_frac_per_date = field_df.groupby('date')['masked'].mean()

    # Keep only dates where fraction of masked pixels <= threshold
    valid_dates = masked_frac_per_date[masked_frac_per_date <= max_missing_frac].index

    return field_df[field_df['date'].isin(valid_dates)].drop(columns='masked')


def gpr_fit_predict(doy_train, y_train, doy_pred, kernel, alpha):
    
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        optimizer=None,
        normalize_y=True
    )

    gp.fit(doy_train, y_train)

    mean, std = gp.predict(doy_pred, return_std=True)

    return mean, std


def train_gpr_kernel(df, ts_cols, value_col, n_samples=200, random_state=42,
                     kernel=None, alpha=1e-4):

    if kernel is None:
        kernel = RBF(length_scale=10) + WhiteKernel(noise_level=alpha)

    # select subset of timeseries
    unique_ts = df[ts_cols].drop_duplicates()
    train_ts = unique_ts.sample(min(n_samples, len(unique_ts)),
                                random_state=random_state)

    train_df = df.merge(train_ts, on=ts_cols)

    dates = pd.to_datetime(train_df['time'])
    doy = dates.dt.dayofyear.values.reshape(-1,1)

    y = train_df[value_col].values

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=10
    )

    gp.fit(doy, y)

    print("Optimized kernel:", gp.kernel_)

    return gp.kernel_


def apply_gpr_timeseries(group_clean, group_raw, kernel, ts_cols, alpha=1e-6):

    dates_train = pd.to_datetime(group_clean["time"])
    doy_train = dates_train.dt.dayofyear.values.reshape(-1,1)

    dates_pred = pd.to_datetime(group_raw["time"])
    doy_pred = dates_pred.dt.dayofyear.values.reshape(-1,1)

    pv_mean, pv_std = gpr_fit_predict(
        doy_train, group_clean["pv"].values, doy_pred, kernel, alpha
    )
    npv_mean, npv_std = gpr_fit_predict(
        doy_train, group_clean["npv"].values, doy_pred, kernel, alpha
    )
    soil_mean, soil_std = gpr_fit_predict(
        doy_train, group_clean["soil"].values, doy_pred, kernel, alpha
    )

    total = pv_mean + npv_mean + soil_mean
    pv = pv_mean / total
    npv = npv_mean / total
    soil = soil_mean / total

    result = group_raw[ts_cols + ["time"]].copy()
    result["pv"] = pv
    result["npv"] = npv
    result["soil"] = soil

    return result


def gapfill_dataframe_gpr(df_raw, df_clean, kernel, ts_cols):

    results = []
    for keys, group_raw in df_raw.groupby(ts_cols):

        group_clean = df_clean[
            (df_clean[ts_cols] == pd.Series(keys, index=ts_cols)).all(axis=1)
        ]

        if len(group_clean) < 3:
            continue

        res = apply_gpr_timeseries(
            group_clean,
            group_raw,
            kernel,
            ts_cols
        )

        results.append(res)

    return pd.concat(results, ignore_index=True)


def gapfill_dataframe_gpr_per_ts(df_raw, df_clean, ts_cols, value_cols, alpha=1e-2):
    results = []
    for keys, group_raw in df_raw.groupby(ts_cols):
        group_clean = df_clean[
            (df_clean[ts_cols] == pd.Series(keys, index=ts_cols)).all(axis=1)
        ]

        if len(group_clean) < 3:
            continue

        res = group_raw[ts_cols + ["time"]].copy()

        for value_col in value_cols:
            # Train GPR for this value_col and timeseries
            dates_train = pd.to_datetime(group_clean["time"])
            doy_train = dates_train.dt.dayofyear.values.reshape(-1, 1)
            y_train = group_clean[value_col].values

            # Estimate variability of the timeseries
            variability = group_clean[value_col].std()
            if variability > 0.2:
                length_scale = 20  # Shorter length scale for high variability
            elif variability <= 0.2:
                length_scale = 80  # Longer length scale for low variability
            else:
                length_scale = 50  # Default length scale
            alpha = max(1e-2, 1 / len(group_clean))

            kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=alpha)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True)
            gp.fit(doy_train, y_train)

            # Predict for the raw timeseries
            dates_pred = pd.to_datetime(group_raw["time"])
            doy_pred = dates_pred.dt.dayofyear.values.reshape(-1, 1)
            mean, _ = gp.predict(doy_pred, return_std=True)

            res[value_col] = mean

        results.append(res)

    return pd.concat(results, ignore_index=True)


# =====================================
# Find top crops in Switzerland
#top_crops = None
top_crops = ['Winter wheat (excluding Swiss Granum fodder wheat)', 'Silage and green maize', 'Winter rapeseed for edible oil', 'Winter barley', 'Sugar beets',\
    'Grain maize', 'Annual open-field vegetables, excluding preserved vegetables', 'Potatoes', 'Fodder wheat according to Swiss Granum list',\
    'Sunflower for edible oil', 'Spelt', 'Triticale', 'Spring wheat (excluding Swiss Granum fodder wheat)', 'Soybean', 'Peas for grain production (e.g., protein peas)',\
    'Oats', 'Rye', 'Open-field vegetables for preservation', 'Seed potatoes (contract farming)', 'Spring wheat (excluding Swiss Granum fodder wheat)',\
    'Beans and vetches for grain production (e.g., field beans)', 'Mixtures of beans, vetch, peas, chickpeas, and lupins with cereals or camelina, minimum 30% legumes at harvest (for grain)',\
    'Spring barley', 'Annual berries (e.g., strawberries)']

if top_crops is None:
    top_crops = []
    lnf_labels = os.path.expanduser('~/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20260217.xlsx')
    df_labels = pd.read_excel(lnf_labels, sheet_name='label_sheet')
    df_labels = df_labels[df_labels['Crop_Label_lv3'].isin(['Arable Land'])]
    for c in df_labels.columns:
        print(c)
        if 'Area' not in c:
            continue
        top_yr = df_labels.sort_values(by=c, ascending=False)[:20]
        top_crops.extend(top_yr['Crop_EN'].tolist())
    top_crops = set(top_crops)
    top_lnfs_arable = df_labels[df_labels['Crop_EN'].isin(top_crops)]['LNF_code'].unique().tolist()
else:
    lnf_labels = os.path.expanduser('~/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20260217.xlsx')
    df_labels = pd.read_excel(lnf_labels, sheet_name='label_sheet')
    top_lnfs_arable = df_labels[df_labels['Crop_EN'].isin(top_crops)]['LNF_code'].unique().tolist()


lnfs_grassland = df_labels[df_labels['Crop_Label_lv3'].isin(['Grassland'])]['LNF_code'].tolist()
top_lnfs_arable += [lnfs_grassland[0]]
print("Will use folowwing LNF codes (and mixed grassland):\n", top_lnfs_arable)
print(top_crops)

# =====================================
# Sample main crops locations (per year, across space)
crop_labels = os.path.expanduser('~/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20250217.xlsx')
lnf_dir = os.path.expanduser('~/mnt/eo-nas1/data/landuse/raw')
tot_samples = 1000
save_path_sampledloc = 'samples.pkl'
if not os.path.exists(save_path_sampledloc):
    sample_locations_with_field(crop_labels, lnf_dir, tot_samples, save_path_sampledloc)


# =====================================
# Extract S2 and Soil data
save_path_sampledloc_S2 = 'samples_data.pkl'
if not os.path.exists(save_path_sampledloc_S2):

    s2_grid_path = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp')
    s2_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH')
    soil_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/DLR_soilsuite_preds/')

    extract_s2_data_field(save_path_sampledloc, s2_grid_path, s2_dir, soil_dir, save_path_sampledloc_S2)


# =====================================
# Predict FC using right soil model

save_path_FCpreds = 'samples_data_pred.pkl'
if not os.path.exists(save_path_FCpreds):
    predict_FC(save_path_sampledloc_S2, save_path_FCpreds)

df = pd.read_pickle(save_path_FCpreds)
print(df.head())
print(df.columns)

# =====================================
# Count how many available timeseries per crop and year

df_samples = pd.read_pickle(save_path_FCpreds)

counts = (
    df_samples
    .drop_duplicates(['lnf_code', 'yr', 'sampled_x', 'sampled_y'])
    .groupby(['lnf_code', 'yr'])
    .size()
    .rename('n_samples')
)
print("Number of samples per year and crop:\n", counts)
# TODO: if needed, cap the number per crop/year -> perhaps wait unitl cleaning to do that, to see how many avlid tiemseries there will be


# =====================================
# Timeseries cleaning
"""
df_clean = (
    df_samples.groupby(['lnf_code', 'yr', 'sampled_x', 'sampled_y'], group_keys=False) # TO DO: change to x and y for each pixel 
      .apply(clean_timeseries_df)
      .reset_index(drop=True)
)

# Cleaning per field
df_clean = (
    df_samples.groupby(['lnf_code', 'yr', 'polygon_geom'], group_keys=False)
              .apply(clean_timeseries_field)
              .reset_index(drop=True)
)

# Get stats on cleaning
ts_cols = ['lnf_code', 'yr', 'sampled_x', 'sampled_y'] # to do: check per field or just per "is_sampled"
n_before = (
    df_samples
    .groupby(ts_cols)
    .size()
    .rename("n_total")
)
n_after = (
    df_clean
    .groupby(ts_cols)
    .size()
    .rename("n_kept")
)
df_drop_stats = (
    pd.concat([n_before, n_after], axis=1)
    .fillna(0)
)
df_drop_stats["n_kept"] = df_drop_stats["n_kept"].astype(int)
df_drop_stats["n_dropped"] = df_drop_stats["n_total"] - df_drop_stats["n_kept"]
df_drop_stats["drop_fraction"] = df_drop_stats["n_dropped"] / df_drop_stats["n_total"]
df_drop_stats = df_drop_stats.reset_index()
"""
# =====================================
# Filter timeseries: only keep if enough data points
"""
print(df_drop_stats.drop_fraction.describe())

drop_fraction_threshold = 0.7 # max fraction dropped
filtered_stats = df_drop_stats[df_drop_stats["drop_fraction"] <= drop_fraction_threshold]
keys_to_keep = filtered_stats[ts_cols]
df_clean_filtered = df_clean.merge(keys_to_keep, on=ts_cols, how="inner")
df_samples_filtered = df_samples.merge(keys_to_keep, on=ts_cols, how="inner")
print(f"Number of timeseries retained: {len(filtered_stats)}/{len(df_drop_stats)}")

# TO DO: could cap the number of samples per crop/year
"""
# =====================================
# GPR gapfilling 

# TODO: train per field and then keep only the sample timeseries?
"""
ts_cols = ['lnf_code','yr','sampled_x','sampled_y']
fractions = ["PV", "NPV", "Soil"]
kernels = {}

# ---------------------------
# Train / load kernels
# ---------------------------
for frac in fractions:
    kernel_file = f"gpr_kernel_{frac}.joblib"
    if not os.path.exists(kernel_file):
        print('Training GPR for', frac)
        # Train kernel on cleaned data subset
        kernel = train_gpr_kernel(
            df_clean,
            ts_cols,
            value_col=frac.lower()  # match your column names
        )
        dump(kernel, kernel_file)
        print(f"Saved kernel for {frac} to {kernel_file}")
    
    # Load kernel
    kernels[frac] = load(kernel_file)

# ---------------------------
# Gapfill all timeseries
# ---------------------------
if not os.path.exists('sampled_data_gpr.pkl'):
    df_gpr_list = []

    for frac in fractions:
        print('Gapfilling for', frac)
        df_frac_gpr = gapfill_dataframe_gpr(
            df_samples,
            df_clean,
            kernels[frac],
            ts_cols,
            value_col=frac.lower()   # Make sure apply_gpr_timeseries uses this column
        )
        df_gpr_list.append(df_frac_gpr)

    # Merge PV/NPV/Soil back into single dataframe
    df_gpr = df_gpr_list[0][ts_cols + ["time"]].copy()
    for i, frac in enumerate(fractions):
        df_gpr[frac] = df_gpr_list[i][frac].values

    df_gpr.to_pickle('sampled_data_gpr.pkl')
else:
    df_gpr = pd.read_pickle('sampled_data_gpr.pkl')
"""

"""
# GPR per timeseries
ts_cols = ['lnf_code', 'yr', 'sampled_x', 'sampled_y']
value_cols = ["pv", "npv", "soil"]
df_gpr = gapfill_dataframe_gpr_per_ts(df_samples_filtered, df_clean_filtered, ts_cols, value_cols)

# =====================================
# Check some cleaned timeseries

ts_cols = ['lnf_code', 'yr', 'sampled_x', 'sampled_y']

#  Sample 5 random timeseries
unique_ts = df_samples_filtered[ts_cols].drop_duplicates()
sampled_ts = unique_ts.sample(5, random_state=42)

# Loop over each timeseries and plot
for i, row in sampled_ts.iterrows():
    # Masks for raw, cleaned, and GPR
    mask_raw = (df_samples_filtered[ts_cols] == row[ts_cols]).all(axis=1)
    mask_clean = (df_clean_filtered[ts_cols] == row[ts_cols]).all(axis=1)
    mask_gpr = (df_gpr[ts_cols] == row[ts_cols]).all(axis=1)
    
    # Select and sort
    df_raw_ts = df_samples_filtered[mask_raw].sort_values('time')
    df_clean_ts = df_clean_filtered[mask_clean].sort_values('time')
    df_gpr_ts = df_gpr[mask_gpr].sort_values('time')

    variability = df_clean_ts["pv"].std()  # Replace "pv" with the desired value_col if needed
    print(f"Timeseries {i}: Variability = {variability:.4f}")

    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(df_raw_ts['time'], df_raw_ts['pv'], 'o-', label='Raw PV', alpha=0.5)
    plt.plot(df_clean_ts['time'], df_clean_ts['pv'], 's-', label='Cleaned PV', alpha=0.7)
    plt.plot(df_gpr_ts['time'], df_gpr_ts['pv'], 'x--', label='GPR PV', alpha=0.8)
    
    plt.title(f"Timeseries {i}: {row['lnf_code']} - {row['yr']} ({row['sampled_x']},{row['sampled_y']})")
    plt.xlabel("Time")
    plt.ylabel("PV fraction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'cleaning_{i}_ts.png')
"""

# =====================================
# Compute SLR with a range of betas