import os
import numpy as np
import xarray as xr
import rioxarray
import pandas as pd
import itertools
import warnings
warnings.filterwarnings('ignore')
from shapely.geometry import box, Point
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
import rasterio
import geopandas as gpd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def sample_locations(ds, year, lnf_mapping, n_loc=10000, n_crops=20):
  """
  Given ds, an xarray dataset in EPSG:32632 and 10m resolution, where the variable lnf_code indicates the crop type
  Identify the top n_crops by pixel count
  For each crop sample n_loc coordinates
  Save locations in a csv called sampled_lcoations_{year}.csv

  :param ds: xarray DataSet
  :param year: int
  :param n_loc: int, number of locations to sample per crop
  :param n_crops: int, number of crops to consider
  :param lnf_mapping: dict, convert lnf codes to category names
  """

  # Identfy the top n_crops by count 
  lnf_values = ds["lnf_code"].values
  unique_lnf, counts = np.unique(lnf_values, return_counts=True)
  #print('Following values not in LNF codes:', [lnf for lnf in unique_lnf if lnf not in lnf_mapping])
  unique_lnf = [lnf for lnf in unique_lnf if lnf in lnf_mapping] # drop if unique_lnf not in lnf_mapping keys
  counts = [count for lnf, count in zip(unique_lnf, counts) if lnf in lnf_mapping]
  top_crops = pd.DataFrame({'LNF':unique_lnf, 'Count':counts})
  top_crops['Category'] = top_crops['LNF'].apply(lambda x: lnf_mapping[int(x)]) # convert lnf to category
  topn_LNF = top_crops.sort_values(by='Count', ascending=False, ignore_index=True).head(n_crops).LNF.tolist()
  topn_cat = top_crops.sort_values(by='Count', ascending=False, ignore_index=True).head(n_crops)#.Category.tolist()
  print(topn_cat)
  
  # Sample 10k locations per croptype
  sampled_points = []
  crop_mask_df = ds["lnf_code"].to_dataframe().reset_index()

  for crop in topn_LNF:
    print(f"Sampling points for crop: {lnf_mapping[crop]}")

    valid_points = crop_mask_df[crop_mask_df["lnf_code"] == crop]
    
    if len(valid_points) > n_loc:
        sampled_crop_points = valid_points.sample(n_loc, random_state=42)
    else:
        sampled_crop_points = valid_points # If less than 10k, take all points
    
    sampled_points.append(sampled_crop_points[['x', 'y', 'lnf_code']])

  df = pd.concat(sampled_points, ignore_index=True)

  # Save as CSV for each year
  df.to_csv(f'sampled_coords/sampled_locations_{year}.csv', index=False)
  print(f'Saved to sampled_locations_{year}.csv')
 
  return


def reproj_dataset(ds, res=10):
  """
  Reproject data from EPSG 2056 to 32632 using regular grid inteprolator

  :param ds: xr Dataset
  :param res: final resolution in m
  :returns ds: ds with reprojected data and updated coords
  """
  # Take bounds of current tif, add some padding, convert to epsg 32632, round to nearest 10m
  lef, rig, top, bot = ds.x.values[0], ds.x.values[-1]+10, ds.y.values[0], ds.y.values[-1]

  bbox = box(lef-100, bot-100, rig+60, top+100)
  transformer = Transformer.from_crs(ds.rio.crs, "EPSG:32632", always_xy=True)
  minx, miny = transformer.transform(bbox.bounds[0], bbox.bounds[1])
  maxx, maxy = transformer.transform(bbox.bounds[2], bbox.bounds[3])

  lef_final, bot_final = np.floor(minx / 10) * 10, np.floor(miny / 10) * 10
  rig_final, top_final = np.ceil(maxx / 10) * 10, np.ceil(maxy / 10) * 10
  
  # Reproject data to EPSG:32632 (regular grid inteprolator)
  
  transformer = Transformer.from_crs(32632, 2056, always_xy=True)
  xt = np.arange(lef_final, rig_final, res) 
  yt = np.arange(top_final, bot_final, -res) 
  XT, YT = np.meshgrid(xt, yt, indexing='ij') 
  XT,YT = transformer.transform(XT,YT)
  
  # Interpolate merged_array to new coords in EPSG:2056
  x_old = ds.x.values
  y_old = ds.y.values
  f = RegularGridInterpolator((y_old,x_old), ds.lnf_code.values, method='nearest',bounds_error=False, fill_value=0)
  reproj_arr = f((YT, XT)).T.astype(np.float64)

  # Extract raster to see what it looks like
  ds = xr.Dataset(
    {"lnf_code": (["y", "x"], reproj_arr)}, 
    coords={"y": yt, "x": xt}
  )

  return ds
  

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




############
# 1. Sampling with crop type maps
""" 
data_dir = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/pv_npv_members/crop_maps') # lnf files, 10m resolution, EPSG:2056, 0 is nodata
crop_map_files = os.listdir(data_dir)
crop_labels = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/pv_npv_members/label_sheet_2025.csv')

crop_names = pd.read_csv(crop_labels) # important cols: LNF_code, categories2024
# Some categories are nan, and ikely not important crops so just drop them
crop_names = crop_names.dropna(subset="categories2024")
# Drop some categories
to_drop = ['Biodiversity promotion area', 'Fallow', 'Fallow extensive', 'Field margin', 'Flowerstrips', 'Waters', 'Walls', 'Paths natural', \
   'Forest Pasture', 'Riverside meadows', 'Permanent green area other', 'Tree Crop', 'Special cultures', 'Hedge', 'Forest', \
  'Greenhouses', 'Greenhause', 'Greenhouse',  'Non agriculture', 'Gardens', 'Vines', 'Apples', 'Pears', 'StoneFruit', 'Chestnut', 'Permaculture', \
    'Orchards other', 'Special cultrues', 'Asparagus', 'Berries'] 
#'Pasture', 'Pasture extensiv', 'Summering Pasture', 'Summering Meadow', 'Summering Meadow Extensive', 'Meadow', 'Meadow extensiv', 'Meadow permanent',
crop_names = crop_names[~crop_names.categories2024.isin(to_drop)]

lnf_mapping = dict(zip(crop_names["LNF_code"], crop_names["categories2024"]))

n_loc=10000
n_crops=15

for f in crop_map_files:
  year = f.split('_')[-1].split('.tif')[0]
  print(f'----Sampling locations for {year}----')

  ds = rioxarray.open_rasterio(os.path.join(data_dir, f)).to_dataset("band").rename({1:"lnf_code"}) # places the coords in the center of pixels

  # Shift to topleft
  dx = abs(ds.x[1] - ds.x[0])  
  dy = abs(ds.y[1] - ds.y[0]) 
  ds = ds.assign_coords(
      x=ds.x - dx / 2,  # Move left by half a pixel
      y=ds.y + dy / 2   # Move up by half a pixel
  )

  # Reproject data
  ds = reproj_dataset(ds, res=10)

  # Drop missing and nodata (0) values
  ds = ds.fillna(0)
  ds = ds.where(ds != 0, drop=True)

  # Clip with field boundaries
  field_shp_path = os.path.expanduser(f'~/mnt/eo-nas1/data/landuse/raw/lnf{year}.gpkg')
  field_shp = gpd.read_file(field_shp_path, crs=2056).to_crs(32632)
  field_shp['geometry'] = field_shp.geometry.buffer(-20) # Add an inward 20m buffer
  field_shp = field_shp[field_shp.geometry.is_valid & ~field_shp.geometry.is_empty] # remove empty or invalid geometries
  ds = ds.rio.write_crs("EPSG:32632").rio.clip(field_shp.geometry)

  sample_locations(ds, year, lnf_mapping, n_loc=n_loc, n_crops=n_crops)
"""

###########
# 1.B Analyse sampled coords
"""
coords_dir = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/pv_npv_members/sampled_coords')
coords_files = [os.path.join(coords_dir,f) for f in os.listdir(coords_dir)]

crop_labels = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/pv_npv_members/label_sheet_2025.csv')
crop_names = pd.read_csv(crop_labels) 
crop_names = crop_names.dropna(subset="categories2024")
lnf_mapping = dict(zip(crop_names["LNF_code"], crop_names["categories2024"]))

# A coord should only be sampled once per year
for f in coords_files:
  print(f'Year {f.split("_")[-1].split(".")[0]}')
  df = pd.read_csv(f)
  df['categ'] = df['lnf_code'].apply(lambda x: lnf_mapping[x])
  print('Top 15 Categories:')
  gb = df.groupby('categ').count().iloc[:, 0].sort_values(ascending=False).index.tolist()
  print(gb)
"""

"""
  print(f'Number of samples: {len(df)}')
  gb = df.groupby(['x', 'y']).count()
  if len(gb[gb.lnf_code>1]):
    print('There are duplicated sample locations:')
    print(gb[gb.lnf_code>1])
"""


############
# 2. Find S2 files
"""
crop_map_dir = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/pv_npv_members/crop_maps') # lnf files, 10m resolution, EPSG:2056, 0 is nodata
crop_map_files = os.listdir(crop_map_dir)
years = ['2021', '2022', '2023'] #[f.split('_')[-1].split('.tif')[0] for f in crop_map_files]

coords_dir = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/pv_npv_members/sampled_coords')
coords_files = [os.path.join(coords_dir,f) for f in os.listdir(coords_dir)]

s2_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH')
s2_files = [f for f in os.listdir(s2_dir) if f.endswith('zarr')]
df_zarr = pd.DataFrame(s2_files, columns=['file'])
df_zarr[['minx', 'maxy', 'yr']] = df_zarr['file'].apply(lambda x: pd.Series(extract_minx_maxy(x)))
s2_grid_files = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp')
s2_grid = gpd.read_file(s2_grid_files)
gdf_zarr = gpd.GeoDataFrame(df_zarr.merge(s2_grid[['left', 'top', 'geometry']], left_on=['minx', 'maxy'], right_on=['left', 'top']))

# Find S2 cubes that are needed
filtered_files = []
for yr in years:

  # Get sampled coords for that year
  coord_file = [f for f in coords_files if yr in f][0]
  coords = pd.read_csv(coord_file)

  coords["geometry"] = coords.apply(lambda row: Point(row["x"], row["y"]), axis=1)
  coords_gdf = gpd.GeoDataFrame(coords, geometry="geometry")

  if coords_gdf.crs != gdf_zarr.crs:
    coords_gdf = coords_gdf.set_crs(gdf_zarr.crs, allow_override=True)

  # Get S2 files for that year
  gdf_zarr_yr = gdf_zarr[gdf_zarr.yr == int(yr)]

  # Don't consider the border of the cube
  gdf_zarr_yr['geometry'] = gdf_zarr_yr['geometry'].buffer(-10)
  # Intersect: find files for the coords that yr
  tiles_to_keep = coords_gdf.sjoin(gdf_zarr_yr, how='inner')#

  # Special case: coord is on cube border (2 cubes will intersect), keep only one cube 
  #tiles_to_keep = tiles_to_keep[tiles_to_keep.x < tiles_to_keep.minx+1280]
  #tiles_to_keep = tiles_to_keep[tiles_to_keep.y > tiles_to_keep.maxy-1280]

  filtered_files.append(tiles_to_keep[['x', 'y', 'lnf_code', 'file', 'yr', 'geometry']])

# Save info on: files, coord, lnf code, year
filtered_files = pd.concat(filtered_files, ignore_index=True)
filtered_files.to_pickle('s2_files_to_sample.pkl')
""" 

################
# 3. Extract S2 data
# TO DO: maybe parallelise this part of sampling
""" 
s2_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH')
filtered_files = pd.read_pickle('s2_files_to_sample.pkl')
unique_files = pd.unique(filtered_files.file)

start_file = 0
unique_files =unique_files[start_file:]

# Loop through files
n_files = len(unique_files)
print(f'There are {len(unique_files)} files to sample data from')


manager = multiprocessing.Manager()  
lock = manager.Lock()  # Cross-process lock
all_pv = manager.list()  # Shared list
all_npv = manager.list()  # Shared list

def process_file(f):

    # Get coords to sample in that file
    pts = filtered_files[filtered_files.file==f]
    
    # Open cube
    ds = xr.open_zarr(os.path.join(s2_dir, f))
    yr = f.split('_')[-1][:4]

    # Filter clouds and baddata
    ds = clean_dataset(ds, cloud_thresh=0.1, snow_thresh=0, shadow_thresh=0.1, cirrus_thresh=800) #drop where any band 65535, clouds, snow, ...

    # For each sample, create feature space (NDVI and SWIR ratio)
    samples = ds.sel(lon=xr.DataArray(pts.x.values), lat=xr.DataArray(pts.y.values))
    #samples = samples.assign_coords(crop_type=("crop", pts["lnf_code"].values)) # Add crop type
    samples['crop_type'] = (("dim_0",), pts["lnf_code"].values)
    bands =  ['s2_B02', 's2_B03', 's2_B04', 's2_B05', 's2_B06', 's2_B07', 's2_B08', 's2_B8A', 's2_B11', 's2_B12']
    samples[bands] = samples[bands].where(samples[bands].compute() != 65535, np.nan) 
    samples['NDVI'] = (samples['s2_B08'] - samples['s2_B04'])/(samples['s2_B04'] + samples['s2_B08'])
    samples['SWIR_ratio'] = (samples['s2_B12']/samples['s2_B11'])

    
    valid_times = samples.where(samples['NDVI'] > 0.1)['NDVI'].notnull().any(dim=['dim_0']).values.nonzero()[0]
    if len(valid_times):
      samples = samples.isel(time=valid_times)

      # PV selection: Highest NDVI & Lowest SWIR_ratio --> use percentiles since the feature space might not be exactly triangular
      try:
        pv_candidates = samples.where(samples['NDVI'].compute() >= samples['NDVI'].quantile(0.7, dim='time').compute(), drop=True)
        pv_best = pv_candidates.where(pv_candidates['SWIR_ratio'].compute() == pv_candidates['SWIR_ratio'].min().compute(), drop=True)
        pv_spectra = pv_best.to_dataframe().reset_index().dropna()[bands+['lat', 'lon', 'time', 'crop_type']]
      except:
        pv_spectra = None

      # NPV selection: Lowest NDVI & Lowest SWIR_ratio --> use percentiles since the feature space might not be exactly triangular
      try:
        npv_candidates = samples.where(samples['NDVI'].compute() <= samples['NDVI'].quantile(0.3, dim='time').compute(), drop=True)
        yr = str(samples.time.values[0].astype('datetime64[Y]'))
        npv_candidates = npv_candidates.where((npv_candidates['time'] >= np.datetime64(f'{yr}-06-01')) & (npv_candidates['time'] <= np.datetime64(f'{yr}-11-15')), drop=True) # between 1st June and 15 Nov
        npv_best = npv_candidates.where(npv_candidates['SWIR_ratio'].compute()  == npv_candidates['SWIR_ratio'].min().compute(), drop=True)
        npv_spectra = npv_best.to_dataframe().reset_index().dropna()[bands+['lat', 'lon', 'time', 'crop_type']]
      except:
        npv_spectra = None

    return pv_spectra, npv_spectra


with ProcessPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(process_file, f): f for f in unique_files}

    for i, future in enumerate(as_completed(futures)):
        pv_spectra, npv_spectra = future.result()
        
        with lock:
          if pv_spectra is not None:
              all_pv.append(pv_spectra)
          if npv_spectra is not None:
              all_npv.append(npv_spectra)
          
       
          if len(all_pv):
            pd.concat(list(all_pv), ignore_index=True).to_pickle('pv_spectra.pkl')
          if len(all_npv):
            pd.concat(list(all_npv), ignore_index=True).to_pickle('npv_spectra.pkl')
          
          if pv_spectra is not None:
            print(f'--Saved PV spectra from file {i+1}/{n_files}--')
          if npv_spectra is not None:
            print(f'--Saved NPV spectra from file {i+1}/{n_files}--')

"""


################
# 4. Summarise and plot spectra --> PER LNF CODE

crop_labels = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/pv_npv_members/label_sheet_2025.csv')
crop_names = pd.read_csv(crop_labels) 
crop_names = crop_names.dropna(subset="categories2024")
lnf_mapping = dict(zip(crop_names["LNF_code"], crop_names["categories2024"]))

pv_spectra = pd.read_pickle('pv_spectra.pkl')
npv_spectra = pd.read_pickle('npv_spectra.pkl')

# Drop duplicates (sometimes there are 2 images for one timestamp and both were kept)
pv_spectra = pv_spectra.drop_duplicates(subset=['lat', 'lon', 'time'])
npv_spectra = npv_spectra.drop_duplicates(subset=['lat', 'lon', 'time'])

# Add year data
pv_spectra['yr'] = pv_spectra['time'].apply(lambda x: x.year)
npv_spectra['yr'] = npv_spectra['time'].apply(lambda x: x.year)

# For crop and year, compute 25/50/75 percentiles
bands = ['s2_B02','s2_B03','s2_B04', 's2_B05','s2_B06','s2_B07', 's2_B08', 's2_B8A', 's2_B11', 's2_B12']  
pv_crop_year = pv_spectra.groupby(['crop_type', 'yr'])[bands].quantile([0.25, 0.50, 0.75])
npv_crop_year = npv_spectra.groupby(['crop_type', 'yr'])[bands].quantile([0.25, 0.50, 0.75])

# Save summarised spectra
pv_crop_year.reset_index().to_pickle('summarised_pv_samples_perlnf.pkl')
npv_crop_year.reset_index().to_pickle('summarised_npv_samples_perlnf.pkl')

"""
# Plot per year
for yr in range(2021,2024):
    pv_crop_year = pv_crop_year.reset_index()
    pv_crop = pv_crop_year[pv_crop_year.yr ==yr].drop('yr', axis=1).rename({'level_2':'percentile'}, axis=1)
    npv_crop_year = npv_crop_year.reset_index()
    npv_crop = npv_crop_year[npv_crop_year.yr ==yr].drop('yr', axis=1).rename({'level_2':'percentile'}, axis=1)

    lnfs = pd.unique(pv_crop.crop_type).tolist()#[:3]
    pv_crop = pv_crop[pv_crop.crop_type.isin(lnfs)]
    lnfs = pd.unique(npv_crop.crop_type).tolist()#[:3]
    npv_crop = npv_crop[npv_crop.crop_type.isin(lnfs)]

    wvl = [490,560,665,705,740,783,842,865,1610,2190]
    band_to_wvl = dict(zip(bands, wvl))

    summary_pv = pv_crop.melt(id_vars=['crop_type', 'percentile'], var_name='band', value_name='reflectance')
    summary_pv['wavelength'] = summary_pv['band'].map(band_to_wvl)
    summary_npv = npv_crop.melt(id_vars=['crop_type', 'percentile'], var_name='band', value_name='reflectance')
    summary_npv['wavelength'] = summary_npv['band'].map(band_to_wvl)

    # All crops on one plot
    f, axs = plt.subplots(1,2,figsize=(16, 5))
    sns.lineplot(
        ax=axs[0],
        data=summary_pv, 
        x='wavelength', 
        y='reflectance', 
        hue='crop_type',  # Different colors per cluster
        style='percentile',  # Different line styles for quantiles
        markers=True,  # Adds markers for better readability
        dashes=True,  # Uses dashed lines for differentiation
        palette='Set1'
    )
    axs[0].set_title(f'PV endmembers for {yr}')
    axs[0].set_ylim(0,8000)
    axs[0].legend_.set_visible(False)

    sns.lineplot(
        ax=axs[1],
        data=summary_npv, 
        x='wavelength', 
        y='reflectance', 
        hue='crop_type',  # Different colors per cluster
        style='percentile',  # Different line styles for quantiles
        markers=True,  # Adds markers for better readability
        dashes=True,  # Uses dashed lines for differentiation
        palette='Set1'
    )
    axs[1].set_title(f'NPV endmembers for {yr}')
    axs[1].set_ylim(0,8000)
    sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 1))

    plt.savefig(f'plots/pv_npv_summary_{yr}.png')



    # PLOT EACH CROP SEPERATELY

    crop_types = pd.unique(pv_crop['crop_type'])

    # Create a 2-column grid with rows corresponding to each crop type
    f, axs = plt.subplots(len(crop_types), 2, figsize=(16, 3 * len(crop_types)))

    # Iterate through each crop type and plot PV and NPV spectra in each row
    for i, crop in enumerate(crop_types):
        # Filter data for the current crop type
        pv_crop_filtered = summary_pv[summary_pv['crop_type'] == crop]
        npv_crop_filtered = summary_npv[summary_npv['crop_type'] == crop]

        crop_name = lnf_mapping[crop]
        
        # Plot PV spectra (first column)
        sns.lineplot(
            ax=axs[i, 0],
            data=pv_crop_filtered,
            x='wavelength',
            y='reflectance',
            hue='percentile',  # Different line styles for quantiles
            markers=True,
            dashes=True,
            palette='Set2'
        )
        
        axs[i, 0].set_title(f'PV endmembers for LNF {crop} ({crop_name})')
        axs[i, 0].set_ylim(0, 6000)
        
        # Plot NPV spectra (second column)
        sns.lineplot(
            ax=axs[i, 1],
            data=npv_crop_filtered,
            x='wavelength',
            y='reflectance',
            hue='percentile',  # Different line styles for quantiles
            markers=True,
            dashes=True,
            palette='Set2'
        )
        axs[i, 1].set_title(f'NPV endmembers for {crop} ({crop_name})')
        axs[i, 1].set_ylim(0, 6000)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(f'plots/pv_npv_summary_per_lnf_{yr}.png')

"""


################
# 4b. Summarise and plot spectra --> PER CROP NAME

crop_labels = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/pv_npv_members/label_sheet_2025.csv')
crop_names = pd.read_csv(crop_labels) 
crop_names = crop_names.dropna(subset="categories2024")
lnf_mapping = dict(zip(crop_names["LNF_code"], crop_names["categories2024"]))

pv_spectra = pd.read_pickle('pv_spectra.pkl')
npv_spectra = pd.read_pickle('npv_spectra.pkl')

# Drop duplicates (sometimes there are 2 images for one timestamp and both were kept)
pv_spectra = pv_spectra.drop_duplicates(subset=['lat', 'lon', 'time'])
npv_spectra = npv_spectra.drop_duplicates(subset=['lat', 'lon', 'time'])

# Convert lnf codes to crop names
pv_spectra['crop_type'] = pv_spectra['crop_type'].map(lnf_mapping)
npv_spectra['crop_type'] = npv_spectra['crop_type'].map(lnf_mapping)

# Add year data
pv_spectra['yr'] = pv_spectra['time'].apply(lambda x: x.year)
npv_spectra['yr'] = npv_spectra['time'].apply(lambda x: x.year)

# For crop and year, compute 25/50/75 percentiles
bands = ['s2_B02','s2_B03','s2_B04', 's2_B05','s2_B06','s2_B07', 's2_B08', 's2_B8A', 's2_B11', 's2_B12']  
pv_crop_year = pv_spectra.groupby(['crop_type', 'yr'])[bands].quantile([0.25, 0.50, 0.75])
npv_crop_year = npv_spectra.groupby(['crop_type', 'yr'])[bands].quantile([0.25, 0.50, 0.75])


# Save summarised spectra
pv_crop_year.reset_index().to_pickle('summarised_pv_samples_pername.pkl')
npv_crop_year.reset_index().to_pickle('summarised_npv_samples_pername.pkl')

""" 
# Plot per year
for yr in range(2021,2024):
    pv_crop_year = pv_crop_year.reset_index()
    pv_crop = pv_crop_year[pv_crop_year.yr ==yr].drop('yr', axis=1).rename({'level_2':'percentile'}, axis=1)
    npv_crop_year = npv_crop_year.reset_index()
    npv_crop = npv_crop_year[npv_crop_year.yr ==yr].drop('yr', axis=1).rename({'level_2':'percentile'}, axis=1)

    lnfs = pd.unique(pv_crop.crop_type).tolist()#[:3]
    pv_crop = pv_crop[pv_crop.crop_type.isin(lnfs)]
    lnfs = pd.unique(npv_crop.crop_type).tolist()#[:3]
    npv_crop = npv_crop[npv_crop.crop_type.isin(lnfs)]

    wvl = [490,560,665,705,740,783,842,865,1610,2190]
    band_to_wvl = dict(zip(bands, wvl))

    summary_pv = pv_crop.melt(id_vars=['crop_type', 'percentile'], var_name='band', value_name='reflectance')
    summary_pv['wavelength'] = summary_pv['band'].map(band_to_wvl)
    summary_npv = npv_crop.melt(id_vars=['crop_type', 'percentile'], var_name='band', value_name='reflectance')
    summary_npv['wavelength'] = summary_npv['band'].map(band_to_wvl)

    # All crops on one plot
    f, axs = plt.subplots(1,2,figsize=(16, 5))
    sns.lineplot(
        ax=axs[0],
        data=summary_pv, 
        x='wavelength', 
        y='reflectance', 
        hue='crop_type',  # Different colors per cluster
        style='percentile',  # Different line styles for quantiles
        markers=True,  # Adds markers for better readability
        dashes=True,  # Uses dashed lines for differentiation
        palette='Set1'
    )
    axs[0].set_title(f'PV endmembers for {yr}')
    axs[0].set_ylim(0,8000)
    axs[0].legend_.set_visible(False)

    sns.lineplot(
        ax=axs[1],
        data=summary_npv, 
        x='wavelength', 
        y='reflectance', 
        hue='crop_type',  # Different colors per cluster
        style='percentile',  # Different line styles for quantiles
        markers=True,  # Adds markers for better readability
        dashes=True,  # Uses dashed lines for differentiation
        palette='Set1'
    )
    axs[1].set_title(f'NPV endmembers for {yr}')
    axs[1].set_ylim(0,8000)
    sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 1))

    plt.savefig(f'plots/pv_npv_summary_{yr}_cropnames.png')



    # PLOT EACH CROP SEPERATELY

    crop_types = pd.unique(pv_crop['crop_type'])

    # Create a 2-column grid with rows corresponding to each crop type
    f, axs = plt.subplots(len(crop_types), 2, figsize=(16, 3 * len(crop_types)))

    # Iterate through each crop type and plot PV and NPV spectra in each row
    for i, crop in enumerate(crop_types):
        # Filter data for the current crop type
        pv_crop_filtered = summary_pv[summary_pv['crop_type'] == crop]
        npv_crop_filtered = summary_npv[summary_npv['crop_type'] == crop]
        
        # Plot PV spectra (first column)
        sns.lineplot(
            ax=axs[i, 0],
            data=pv_crop_filtered,
            x='wavelength',
            y='reflectance',
            hue='percentile',  # Different line styles for quantiles
            markers=True,
            dashes=True,
            palette='Set2'
        )
        
        axs[i, 0].set_title(f'PV endmembers for {crop}')
        axs[i, 0].set_ylim(0, 6000)
        
        # Plot NPV spectra (second column)
        sns.lineplot(
            ax=axs[i, 1],
            data=npv_crop_filtered,
            x='wavelength',
            y='reflectance',
            hue='percentile',  # Different line styles for quantiles
            markers=True,
            dashes=True,
            palette='Set2'
        )
        axs[i, 1].set_title(f'NPV endmembers for {crop}')
        axs[i, 1].set_ylim(0, 6000)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(f'plots/pv_npv_summary_per_cropname_{yr}.png')

"""