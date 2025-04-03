
import os
import pandas as pd
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')
from shapely.geometry import Point
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation


def extract_minx_maxy(file):
    parts = file.split('_')
    minx = int(parts[1])
    maxy = int(parts[2])
    yr = int(parts[3][:4])
    return minx, maxy, yr


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
  n_dropped = len(set(dates_to_drop))
  mask_dates[dates_to_drop] = False
  ds = ds.isel(time=mask_dates)

  # Remove too many clouds (mask=1), shadows (mask=2) or snow (mask=3)
  dates_to_drop = [i for i, date in enumerate(ds.time.values) if has_clouds(ds.isel(time=i), cloud_thresh)] + \
                [i for i, date in enumerate(ds.time.values) if has_shadows(ds.isel(time=i), shadow_thresh)] + \
                [i for i, date in enumerate(ds.time.values) if has_snow(ds.isel(time=i), snow_thresh)] +\
                [i for i, date in enumerate(ds.time.values) if has_cirrus(ds.isel(time=i), cirrus_thresh)]
  mask_dates = np.ones(len(ds.time), dtype=bool)
  print(f'Dropping {n_dropped+len(set(dates_to_drop))}/{n_times} dates') # flag if too many dates dropped?
  mask_dates[dates_to_drop] = False
  ds = ds.isel(time=mask_dates)

  return ds


def sample_point_for_crop_yr(crop_name, crop_names, yr, poi=None):
    """ 
    Sample a location where there is crop_name in year yr
    Can also specify a poi (lon, lat)
    """

    if poi is not None:
        crop_code = crop_names[crop_names['categories2024'] == crop_name]['LNF_code'].values[0]
        sample = pd.DataFrame({'x': [poi[0]], 'y': [poi[1]], 'lnf_code': [crop_code]})

    if poi is None:
        crop_code = crop_names[crop_names['categories2024'] == crop_name]['LNF_code'].values
        coords_file = [os.path.join(coords_dir,f) for f in os.listdir(coords_dir) if str(yr) in f][0]
        coords_file = pd.read_csv(coords_file)
        if len(coords_file[coords_file['lnf_code'].isin(crop_code)]):
            sample = coords_file[coords_file['lnf_code'].isin(crop_code)].sample(1)
        else:
            sample = []
    
    return sample


def check_in_field(field_shp_path, sample_gdf, buffer):
    """
    Check if point is atleast 20m inside fields

    :param field_shp_path: path to shapefile with fields geometry for that year
    :param sample_gdf: gpd.GeoDataFrame with point geometry of location and crop lnf_code
    :param buffer: buffer around field to check if point is inside
    """
  
    field_shp = gpd.read_file(field_shp_path, crs=2056) # in EPSG 2056
    field_shp['geometry'] = field_shp.geometry.buffer(-buffer) # Add an inward 20m bbuffer
    field_shp = field_shp[field_shp.geometry.is_valid & ~field_shp.geometry.is_empty] # remove empty or invalid geometries

    field_shp = field_shp.to_crs(sample_gdf.crs)
    intersects = gpd.sjoin(sample_gdf, field_shp, how='inner', predicate='within') # contains only pt geom

    return intersects, field_shp


def find_s2_files_for_sample(s2_dir, sample_gdf):

    s2_files = [f for f in os.listdir(s2_dir) if f.endswith('zarr')]
    df_zarr = pd.DataFrame(s2_files, columns=['file'])
    df_zarr[['minx', 'maxy', 'yr']] = df_zarr['file'].apply(lambda x: pd.Series(extract_minx_maxy(x)))
    s2_grid_files = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp')
    s2_grid = gpd.read_file(s2_grid_files)
    gdf_zarr = gpd.GeoDataFrame(df_zarr.merge(s2_grid[['left', 'top', 'geometry']], left_on=['minx', 'maxy'], right_on=['left', 'top']))

    if sample_gdf.crs != gdf_zarr.crs:
        sample_gdf = sample_gdf.set_crs(gdf_zarr.crs, allow_override=True)

    gdf_zarr_yr = gdf_zarr[gdf_zarr.yr == int(yr)]

    tiles_to_keep = sample_gdf.sjoin(gdf_zarr_yr, how='inner')

    return tiles_to_keep


def extract_and_clean(s2_dir, f, sample):
    """
    Open s2 file, extract s2 timeseries for sample location and area of 200mx200m around it
    Apply cleaning
    """

    ds = xr.open_zarr(os.path.join(s2_dir, f))

    # Delete duplicate time (keep those in UTM 32 if possible)
    priority = ['T32' in p.split('_')[5] if len(p.split('_')) > 5 else False for p in ds['product_uri'].values]
    ds = ds.assign_coords(priority=priority)#.stack(sample=('time', 'product_uri'))
    ds = ds.sortby('priority', ascending=False)
    ds = ds.groupby('time').first()

    # Keep area around pt, and clean
    min_lon, max_lon = ds.lon.min().values, ds.lon.max().values
    min_lat, max_lat = ds.lat.min().values, ds.lat.max().values
    ds_area = ds.sel(
        lon=slice(max(sample.x.values[0]-200, min_lon), min(sample.x.values[0]+200, max_lon)),
        lat=slice(min(sample.y.values[0]+200, max_lat), max(sample.x.values[0]-200, min_lat))
    )
    ds_area = clean_dataset(ds_area, cloud_thresh=0.1, snow_thresh=0, shadow_thresh=0.1, cirrus_thresh=800) #drop where any band 65535, clouds, snow, ...
    ds_area = ds_area.isel(lat=slice(1,-2), lon=slice(1,-2)) # remove border 
    min_lon, max_lon = ds_area.lon.min().values, ds_area.lon.max().values
    min_lat, max_lat = ds_area.lat.min().values, ds_area.lat.max().values

    sample = sample[(sample.y >= min_lat) & (sample.y <= max_lat) & (sample.x >= min_lon) & (sample.x <= max_lon)]

    s2_pt = ds_area.sel(lon=xr.DataArray(sample.x.values), lat=xr.DataArray(sample.y.values))
    s2_pt = s2_pt.assign_coords(crop_type=("crop", sample["lnf_code"].values)) # Add crop type

    # For each sample, create feature space (NDVI and SWIR ratio)
    bands = ['s2_B02', 's2_B03', 's2_B04', 's2_B08', 's2_B11', 's2_B12']
    s2_pt[bands] = s2_pt[bands].where(s2_pt[bands] != 65535, np.nan)  # Set value 65535 to nan
    s2_pt['NDVI'] = (s2_pt['s2_B08'] - s2_pt['s2_B04'])/(s2_pt['s2_B04'] + s2_pt['s2_B08'])
    s2_pt['SWIR_ratio'] = (s2_pt['s2_B12']/s2_pt['s2_B11'])

    # Cleaning: drop when NDVI<=0.1
    valid_times = s2_pt.where(s2_pt['NDVI'] > 0.1)['NDVI'].notnull().any(dim=['dim_0']).values.nonzero()[0]
    s2_pt = s2_pt.isel(time=valid_times)
    ds_area = ds_area.isel(time=valid_times)

    return ds_area, s2_pt, sample


def find_pv_npv(s2_pt):

    if len(s2_pt.time.values) == 0:
        return None, None, None

    bands = ['s2_B02', 's2_B03', 's2_B04', 's2_B05', 's2_B06', 's2_B07', 's2_B08', 's2_B8A', 's2_B11', 's2_B12']
    
    try:
        # PV selection: Highest NDVI & Lowest SWIR_ratio --> use percentiles since the feature space might not be exactly triangular
        pv_candidates = s2_pt.where(s2_pt['NDVI'].compute() >= s2_pt['NDVI'].quantile(0.7, dim='time').compute(), drop=True)
        pv_best = pv_candidates.where(pv_candidates['SWIR_ratio'].compute() == pv_candidates['SWIR_ratio'].min().compute(), drop=True)
        pv_spectra_sampled = pv_best.to_dataframe().reset_index().dropna()[bands+['lat', 'lon', 'time', 'crop_type']]
    except:
        pv_spectra_sampled = None

    try:
        # NPV selection: Lowest NDVI & Lowest SWIR_ratio --> use percentiles since the feature space might not be exactly triangular
        npv_candidates = s2_pt.where(s2_pt['NDVI'].compute() <= s2_pt['NDVI'].quantile(0.3, dim='time').compute(), drop=True)
        yr = str(s2_pt.time.values[0].astype('datetime64[Y]'))
        npv_candidates = npv_candidates.where((npv_candidates['time'] >= np.datetime64(f'{yr}-06-01')) & (npv_candidates['time'] <= np.datetime64(f'{yr}-11-15')), drop=True) # between 1st June and 15 Nov
        npv_best = npv_candidates.where(npv_candidates['SWIR_ratio'].compute() == npv_candidates['SWIR_ratio'].min().compute(), drop=True)
        npv_spectra_sampled = npv_best.to_dataframe().reset_index().dropna()[bands+['lat', 'lon', 'time', 'crop_type']]
    except:
        npv_spectra_sampled = None

    try:
        soil_candidates = s2_pt.where(s2_pt['NDVI'].compute() <= s2_pt['NDVI'].quantile(0.3, dim='time').compute(), drop=True)
        soil_best = soil_candidates.where(soil_candidates['SWIR_ratio'].compute() == soil_candidates['SWIR_ratio'].max().compute(), drop=True)
        soil_spectra_sampled = soil_best.to_dataframe().reset_index().dropna()[bands+['lat', 'lon', 'time', 'crop_type']]
    except:
        soil_spectra_sampled = None

    return pv_spectra_sampled, npv_spectra_sampled, soil_spectra_sampled


def update(frame):
    # Function to update the frame
    ax_rgb.clear()
    ax_rgb.grid(False) #axis('off')
    
    # Plot RGB image
    x_coords = rgb[frame].coords['lon'].values
    y_coords = rgb[frame].coords['lat'].values
    ax_rgb.imshow(rgb.isel(time=frame).values * brightness, origin='upper', extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]) #origin upper due to np array
    ax_rgb.set_title(f"RGB {str(rgb.time.values[frame]).split('T')[0]}")

    # Add location of point and field 
    sample_gdf_update.plot(ax=ax_rgb, alpha=0.8, edgecolor='r', facecolor='r')
    intersects_update['geometry_y'].plot(ax=ax_rgb, alpha=0.8, edgecolor='r', facecolor='none')

    # Convert time to numeric for scatter plot
    time_numeric = mdates.date2num(pd.to_datetime(rgb.time.values[:frame+1]))
    
    # Update scatter plots
    NDVI_scatter.set_offsets(np.c_[time_numeric, NDVI_vals[:frame+1]])
    SWIR_scatter.set_offsets(np.c_[time_numeric, SWIR_vals[:frame+1]])
    
    # Update line plots
    NDVI_line.set_data(rgb.time.values[:frame+1], NDVI_vals[:frame+1])
    SWIR_line.set_data(rgb.time.values[:frame+1], SWIR_vals[:frame+1])


def create_animation(s2_pt, ds_area, pv_spectra_sampled, npv_spectra_sampled, soil_spectra_sampled, crop_name, yr, sample, sample_gdf, intersects):

    global fig, ax_rgb, time_series_ax, ax_featspace, NDVI_scatter, SWIR_scatter, NDVI_line, SWIR_line, rgb, brightness, NDVI_vals, SWIR_vals
    global sample_gdf_update, intersects_update

    sample_gdf_update = sample_gdf.copy()
    intersects_update = intersects.copy()

    NDVI_vals = s2_pt['NDVI'].values.squeeze()
    SWIR_vals = s2_pt['SWIR_ratio'].values.squeeze()

    # Setup the figure and axes
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2,2, figure=fig, width_ratios=[1,1], height_ratios=[2,1])

    # Initialize the time series plot
    time_series_ax = fig.add_subplot(gs[1, :])

    NDVI_scatter = time_series_ax.scatter([], [], color='g', s=10)
    SWIR_scatter = time_series_ax.scatter([], [], color='sandybrown', s=10)

    NDVI_line, = time_series_ax.plot([], [], color='g', label='NDVI')
    SWIR_line, = time_series_ax.plot([], [], color='sandybrown', label='SWIR2/SWIR1')

    time_series_ax.set_xlim(s2_pt.time.values[0], s2_pt.time.values[-1])
    time_series_ax.set_ylim(-0.5, 2.5)
    time_series_ax.set_ylabel("NDVI and SWIR ratio", fontsize=14)
    time_series_ax.legend(loc='upper left')

    # Only plot the vertical lines if the data exists
    if pv_spectra_sampled is not None:
        time_series_ax.axvline(x=pv_spectra_sampled.time.values[0], color='green', linestyle='--', linewidth=1)
        time_series_ax.text(pv_spectra_sampled.time.values[0], 1.7, 'PV spectra', fontsize=10, color='gray', 
                            rotation=90, verticalalignment='bottom', horizontalalignment='right')
    if npv_spectra_sampled is not None:
        time_series_ax.axvline(x=npv_spectra_sampled.time.values[0], color='sandybrown', linestyle='--', linewidth=1)
        time_series_ax.text(npv_spectra_sampled.time.values[0], 1.7, 'NPV spectra', fontsize=10, color='gray', 
                            rotation=90, verticalalignment='bottom', horizontalalignment='right')
    if soil_spectra_sampled is not None:
        time_series_ax.axvline(x=soil_spectra_sampled.time.values[0], color='saddlebrown', linestyle='--', linewidth=1)
        time_series_ax.text(soil_spectra_sampled.time.values[0], 1.7, 'Soil examples', fontsize=10, color='gray', 
                            rotation=90, verticalalignment='bottom', horizontalalignment='right')


    # Initialize RGB subplot
    ax_rgb = fig.add_subplot(gs[0, 0])

    scale_factor = 1.0 / 10000.0 
    r = ds_area['s2_B04']* scale_factor
    g = ds_area['s2_B03']* scale_factor
    b = ds_area['s2_B02']* scale_factor
    rgb = xr.concat([r, g, b], dim='band').transpose('time', 'lat', 'lon', 'band')
    rgb = rgb.where(~np.isnan(rgb), other=1.0)
    brightness = 3

    # Plot feature space (will not be updated)
    ax_featspace = fig.add_subplot(gs[0, 1])
    s2_pt.plot.scatter(x='NDVI', y='SWIR_ratio', ax=ax_featspace)
    # Change color of selected PV and NPV spectra
    if pv_spectra_sampled is not None:
        pv_spectra_sampled['NDVI'] = (pv_spectra_sampled['s2_B08']-pv_spectra_sampled['s2_B04'])/(pv_spectra_sampled['s2_B08']+pv_spectra_sampled['s2_B04'])
        pv_spectra_sampled['SWIR_ratio'] = pv_spectra_sampled['s2_B12']/pv_spectra_sampled['s2_B11']
        pv_spectra_sampled.plot.scatter(x='NDVI', y='SWIR_ratio', ax=ax_featspace, color='limegreen', s=10)
    if npv_spectra_sampled is not None:
        npv_spectra_sampled['NDVI'] = (npv_spectra_sampled['s2_B08']-npv_spectra_sampled['s2_B04'])/(npv_spectra_sampled['s2_B08']+npv_spectra_sampled['s2_B04'])
        npv_spectra_sampled['SWIR_ratio'] = npv_spectra_sampled['s2_B12']/npv_spectra_sampled['s2_B11']
        npv_spectra_sampled.plot.scatter(x='NDVI', y='SWIR_ratio', ax=ax_featspace, color='sandybrown', s=10)
    if soil_spectra_sampled is not None:
        soil_spectra_sampled['NDVI'] = (soil_spectra_sampled['s2_B08']-soil_spectra_sampled['s2_B04'])/(soil_spectra_sampled['s2_B08']+soil_spectra_sampled['s2_B04'])
        soil_spectra_sampled['SWIR_ratio'] = soil_spectra_sampled['s2_B12']/soil_spectra_sampled['s2_B11']
        soil_spectra_sampled.plot.scatter(x='NDVI', y='SWIR_ratio', ax=ax_featspace, color='saddlebrown', s=10)

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(s2_pt.time), interval=500, repeat=False)
    FFwriter = animation.FFMpegWriter(fps=6)
    anim.save(filename=f"animations/{crop_name}_{yr}_{int(sample['x'].values[0])}_{int(sample['y'].values[0])}.mp4", writer=FFwriter)

    plt.tight_layout()

    return


def run_analysis(crop_name, crop_names, yr, s2_dir, field_shp_path, poi=None):
    """
    Run the analysis for a given crop, year and location
    generates an mpv showing PV and NPV selection
    """

    ##### SAMPLE A LOCATION FOR CROP, YR

    sample = sample_point_for_crop_yr(crop_name, crop_names, yr, poi=poi)
    if not len(sample):
        print('No sample for that crop in yr')
        return 
    sample["geometry"] = sample.apply(lambda row: Point(row["x"], row["y"]), axis=1)
    sample_gdf = gpd.GeoDataFrame(sample, geometry="geometry", crs="EPSG:32632")
    print('Sampled a location', sample_gdf.x.values[0], sample_gdf.y.values[0])

    ##### CHECK IF IN FIELD
    intersects, field_shp = check_in_field(field_shp_path, sample_gdf, buffer=0)
    
    if len(intersects) == 0:
        print('Point was too close to field boundary')
        return

    #### FIND THE S2 FILE FOR SAMPLE LOCAITON
    if len(intersects):
        print('Point is within field')
        # Add field again
        intersects = intersects.merge(field_shp[['geometry']], left_on='index_right', right_index=True, how='left') # pt will be geometry_x and field geometry_y

        tiles_to_keep = find_s2_files_for_sample(s2_dir, sample_gdf)
        f = tiles_to_keep.file.values[0] # should be only one file normally

        #### EXTRACT S2 TIMESERIES, CLEAN
        ds_area, s2_pt, sample = extract_and_clean(s2_dir, f, sample)

        #### FIND PV AND NPV
        pv_spectra_sampled, npv_spectra_sampled, soil_spectra_sampled = find_pv_npv(s2_pt)

        if pv_spectra_sampled is None and npv_spectra_sampled is None and soil_spectra_sampled is None:
          print('No data left after cleaning')
          return

        create_animation(s2_pt, ds_area, pv_spectra_sampled, npv_spectra_sampled, soil_spectra_sampled, crop_name, yr, sample, sample_gdf, intersects)
        
    return
            



#### DEFINE SAMPLE LOCATION TO ANALYSE

# Sampled coordinates
coords_dir = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/pv_npv_members/sampled_coords')

# Mapping from LNF code to crop name
crop_labels = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/pv_npv_members/label_sheet_2025.csv')
crop_names = pd.read_csv(crop_labels) 

# Location, crop type, year of interest
#crop_name = 'Wheat'
yr = 2023
poi = None # x, y defined by user

# S2 data
s2_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH')

field_shp_path = os.path.expanduser(f'~/mnt/eo-nas1/data/landuse/raw/lnf{yr}.gpkg')
    
# Generate MP4
crops = ['SpecialCrops', 'Pasture', 'Potatoes', 'Wheat', 'Winter Rapeseed', 'Sugar_beets', 'Meadow', 'Maize grain', 'Maize silage',\
     'Winter Wheat', 'Winter Barley', 'Spelt', 'Peas', 'Linen','SpecialCrops', 'Oat', 'Rye', 'Sunflowers', 'Tobacco']

for crop_name in crops:
  print(crop_name)
  run_analysis(crop_name, crop_names, yr, s2_dir, field_shp_path, poi)