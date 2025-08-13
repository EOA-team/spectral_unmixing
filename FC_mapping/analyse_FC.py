import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import xarray as xr
from shapely.geometry import box
import rioxarray
import pyproj
from shapely.ops import transform
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in divide")


def create_hex_grid(gdf, hex_size):
    xmin, ymin, xmax, ymax = gdf.total_bounds
    dx = 3/2 * hex_size
    dy = np.sqrt(3) * hex_size
    cols = int((xmax - xmin) / dx) + 2
    rows = int((ymax - ymin) / dy) + 2

    hexes = []
    for row in range(rows):
        for col in range(cols):
            x = xmin + col * dx
            y = ymin + row * dy
            if col % 2:
                y += dy / 2

            hex = Polygon([
                (x + hex_size * np.cos(np.pi / 3 * i),
                 y + hex_size * np.sin(np.pi / 3 * i)) for i in range(6)
            ])
            hexes.append(hex)

    return gpd.GeoDataFrame(geometry=hexes, crs=gdf.crs)


def compute_hex_area(gdf, hex_size, borders):
    gdf['area'] = gdf.geometry.area
    hex_grid = create_hex_grid(gdf, hex_size)
    hex_grid = hex_grid.reset_index().rename(columns={"index": "hex_id"})
    hex_grid = hex_grid[hex_grid.intersects(borders.geometry.union_all())]
    
    joined = gpd.overlay(gdf, hex_grid, how='intersection')
    joined["intersect_area"] = joined.geometry.area
    
    area_per_hex = joined.groupby("hex_id")["intersect_area"].sum()
    hex_grid["total_area"] = hex_grid["hex_id"].map(area_per_hex).fillna(0)
    
    return hex_grid


"""
year = 2021
canton_poly = 'swissBOUNDARIES3D_1_5_LV95_LN02.gpkg'
canton_name = None #'Aargau'
fields = os.path.expanduser(f'~/mnt/eo-nas1/data/landuse/raw/lnf{year}.gpkg')
canton_poly = 'swissBOUNDARIES3D_1_5_LV95_LN02.gpkg'

# Define LNF codes
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


# Codes for grassland
lnf_codes = crop_names[crop_names.categories2024!='Grassland'].LNF_code.tolist()


CRS = 32632
hex_size = 10000  # 10 km

fields = gpd.read_file(fields).to_crs(CRS)
borders = gpd.read_file(canton_poly, layer='tlm_landesgebiet').to_crs(CRS)
canton_outline = gpd.read_file(canton_poly, layer='tlm_kantonsgebiet').to_crs(CRS)

arable_codes = crop_names[crop_names.categories2024 != 'Grassland'].LNF_code.tolist()
grassland_codes = crop_names[crop_names.categories2024 == 'Grassland'].LNF_code.tolist()

gdf_arable = fields[fields.lnf_code.isin(arable_codes)].copy()
gdf_grass = fields[fields.lnf_code.isin(grassland_codes)].copy()

hex_arable = compute_hex_area(gdf_arable, hex_size, borders)
hex_grass = compute_hex_area(gdf_grass, hex_size, borders)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True)

vmin = 0
vmax = max(hex_arable['total_area'].max(), hex_grass['total_area'].max())
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.YlGn

canton_outline.plot(ax=axes[0], edgecolor='blue', facecolor='none')
hex_arable.plot(column='total_area', ax=axes[0], cmap=cmap, legend=False, edgecolor='gray', alpha=0.5, norm=norm)
axes[0].set_title("Arable area")
cx.add_basemap(ax=axes[0], source=cx.providers.SwissFederalGeoportal.NationalMapColor, crs=CRS)

canton_outline.plot(ax=axes[1], edgecolor='blue', facecolor='none')
hex_grass.plot(column='total_area', ax=axes[1], cmap=cmap, legend=False, edgecolor='gray', alpha=0.5, norm=norm)
axes[1].set_title("Grassland area")
cx.add_basemap(ax=axes[1], source=cx.providers.SwissFederalGeoportal.NationalMapColor, crs=CRS)

# Shared colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []  # empty array for the scalar mappable
cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.035, pad=0.04)
cbar.set_label('Area [m²]')

plt.savefig("hexplot_arable_vs_grassland.png", dpi=300)

"""

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


def clean_dataset_optimized(ds, cloud_thresh=0.1, shadow_thresh=0.1, snow_thresh=0.1, cirrus_thresh=1000):
    # Shape info
    """ 
    nlat = ds.sizes['lat']
    nlon = ds.sizes['lon']
    npixels = nlat * nlon
    """

    # 1. Drop dates where all values in the dataset are 65535
    is_65535 = (ds == 65535).to_array().all(dim=['lat', 'lon'])
    all_65535 = is_65535.any(dim='variable')
    #ds = ds.sel(time=~all_65535)
    keep_idx = np.where(~all_65535.values)[0]
    ds = ds.isel(time=keep_idx)

    # 2. Vectorized cloud, snow, shadow and cirrus masking
    """
    mask_clouds  = ((ds.s2_mask == 1) | ds.s2_SCL.isin([8, 9, 10])).sum(dim=['lat', 'lon']) / npixels > cloud_thresh
    mask_shadows = ((ds.s2_mask == 2) | (ds.s2_SCL == 3)).sum(dim=['lat', 'lon']) / npixels > shadow_thresh
    #mask_snow_scl  = ((ds.s2_mask == 3) | (ds.s2_SCL == 11)).sum(dim=['lat', 'lon']) / npixels > snow_thresh
    mask_snow_ndsi = (((ds.s2_B03/10000 - ds.s2_B11/10000)/(ds.s2_B03/10000 + ds.s2_B11/10000)  > 0.4)).sum(dim=['lat', 'lon']) / npixels > snow_thresh #& (ds.s2_B08 > 0.11)).sum(dim=['lat', 'lon']) 
    mask_snow = mask_snow_ndsi #| mask_snow_scl  
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
    #ds = ds.sel(time=~drop_mask)
    keep_idx = np.where(~drop_mask.values)[0]
    ds = ds.isel(time=keep_idx)

    return ds



"""
# Find cubes in a canton and in a month where there are cloud free images for grassland
FC_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/FC')
s2_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH')
canton_poly = 'swissBOUNDARIES3D_1_5_LV95_LN02.gpkg'
canton_name = 'Valais'
year = '2021'

# Prepare Landuse file
swisstopo_landuse = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/EVI_CH/ag-b-00.03-37-area-current-csv.csv')
lu_csv = pd.read_csv(swisstopo_landuse,
            sep=';',
            usecols=['AS_17', 'E_COORD', 'N_COORD'],  # adjust columns to those needed in create_square
            dtype={'AS_17': 'int32', 'E_COORD': 'float64', 'N_COORD': 'float64'}  # faster parsing
        )
lu_csv = lu_csv[lu_csv['AS_17'].isin([8, 9])]

# Filter for canton
fc_files = [f for f in os.listdir(FC_dir) if f.endswith('.zarr') and f.split('_')[3].startswith(year)]
gdf_FC_files = create_file_gdf(fc_files)
gdf = gpd.read_file(canton_poly, layer='tlm_kantonsgebiet') 
gdf = gdf[gdf.name==canton_name].to_crs('EPSG:32632')
gdf_FC_files = gdf_FC_files[gdf_FC_files.intersects(gdf.geometry.union_all())]
 

for f in gdf_FC_files.filename.tolist():
    # Open S2 and cloud clean
    s2 = xr.open_zarr(os.path.join(s2_dir, os.path.basename(f)))
    s2 = clean_dataset_optimized(s2, cloud_thresh=0.05, snow_thresh=0.1, shadow_thresh=0.1, cirrus_thresh=800)
    s2 = s2.drop_duplicates(dim='time', keep='first')
    # Select FC based on matching product_uri
    ds = xr.open_zarr(os.path.join(FC_dir, os.path.basename(f)))
    mask = ds['product_uri'].isin(s2['product_uri'].compute().values).compute()
    ds = ds.where(mask, drop=True)

    in_jan = (ds['time'].dt.month == 1).any().item() # True if any timestamp in January

    if in_jan:
        # Crop for grassland 
        bbox_32632 = box(int(f.split('_')[1]), int(f.split('_')[2]), int(f.split('_')[1]) + 1280, int(f.split('_')[2]) + 1280)
        project = pyproj.Transformer.from_crs('EPSG:32632', 'EPSG:2056', always_xy=True).transform
        bbox_2056 = transform(project, bbox_32632)
        minx, miny, maxx, maxy = bbox_2056.bounds
        lu_csv = lu_csv[(lu_csv['E_COORD'] >= minx) & (lu_csv['E_COORD'] -100 <= maxx) & 
                        (lu_csv['N_COORD'] >= miny) & (lu_csv['N_COORD'] -100 <= maxy)]

        if len(lu_csv):
            lu_csv.loc[:, 'geometry'] = lu_csv.apply(create_square, axis=1)
            gdf_lu = gpd.GeoDataFrame(lu_csv, geometry='geometry', crs='EPSG:2056').to_crs(32632)
        
        try:
            ds = ds.rio.write_crs(32632).rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=False).rio.clip(gdf_lu.geometry, all_touched=False)
        except:
            continue 

        # Check if any data left
        if ds:
            print(f)
         
"""
"""
# Check which cubes have data in a given month and canton

FC_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/FC')
s2_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH')
canton_poly = 'swissBOUNDARIES3D_1_5_LV95_LN02.gpkg'
canton_name = 'St. Gallen'
year = '2021'
swisstopo_landuse = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/EVI_CH/ag-b-00.03-37-area-current-csv.csv')


fc_files = [os.path.join(FC_dir, f) for f in os.listdir(FC_dir) if f.endswith('.zarr') and f.split('_')[3].startswith(year)]
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


for f in gdf_FC_files.filename.tolist():

        ds = xr.open_zarr(f)

        # Load and clean S2
        s2 = xr.open_zarr(os.path.join(s2_dir, os.path.basename(f)))
        s2 = clean_dataset_optimized(s2, cloud_thresh=0.05, snow_thresh=0.1, shadow_thresh=0.1, cirrus_thresh=800)
        s2 = s2.drop_duplicates(dim='time', keep='first')
        mask = ds['product_uri'].isin(s2['product_uri'].compute().values).compute()
        ds = ds.where(mask, drop=True)
       
        if lu_csv is not None:
            # Clip fields
            ff = os.path.basename(f)
            bbox_32632 = box(int(ff.split('_')[1]), int(ff.split('_')[2]), int(ff.split('_')[1]) + 1280, int(ff.split('_')[2]) + 1280)
            project = pyproj.Transformer.from_crs('EPSG:32632', 'EPSG:2056', always_xy=True).transform
            bbox_2056 = transform(project, bbox_32632)
            minx, miny, maxx, maxy = bbox_2056.bounds
            lu_csv = lu_csv[(lu_csv['E_COORD'] >= minx) & (lu_csv['E_COORD'] -100 <= maxx) & 
                            (lu_csv['N_COORD'] >= miny) & (lu_csv['N_COORD'] -100 <= maxy)]
    
            if len(lu_csv):
                lu_csv['geometry'] = lu_csv.apply(create_square, axis=1)
                gdf_lu = gpd.GeoDataFrame(lu_csv, geometry='geometry', crs='EPSG:2056').to_crs(32632)
            else:
                continue
            try:
                ds = ds.rio.write_crs(32632).rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=False).rio.clip(gdf_lu.geometry, all_touched=False)
               
                if len(ds.time) and (ds.time.dt.month == 1).any():
                    print(os.path.basename(os.path.basename(f)), ds.time[ds.time.dt.month == 1])
            except:
                continue # There is no data intersceting the field type 



"""









# Plot some FC NPV data in east vs west Switzerland

FC_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/FC')
s2_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH')

east = (391900, 5123500) #(463580, 5254060)
west = (546780, 5251500) #(375260, 5190060)

fc_west = [f for f in os.listdir(FC_dir) if f'S2_{west[0]}_{west[1]}' in f][0]
fc_east = [f for f in os.listdir(FC_dir) if f'S2_{east[0]}_{east[1]}' in f][0]

ds_west = xr.open_zarr(os.path.join(FC_dir, fc_west))
ds_east = xr.open_zarr(os.path.join(FC_dir, fc_east))


# Open S2 and cloud clean
s2_west = xr.open_zarr(os.path.join(s2_dir, os.path.basename(fc_west)))#[['s2_mask', 's2_SCL', 's2_B02']].load()
s2_west = clean_dataset_optimized(s2_west, cloud_thresh=0.05, snow_thresh=0.1, shadow_thresh=0.1, cirrus_thresh=800)
s2_west = s2_west.drop_duplicates(dim='time', keep='first')
# Select ds_west based on matching product_uri
mask = ds_west['product_uri'].isin(s2_west['product_uri'].compute().values).compute()
ds_west = ds_west.where(mask, drop=True)


s2_east = xr.open_zarr(os.path.join(s2_dir, os.path.basename(fc_east)))#[['s2_mask', 's2_SCL', 's2_B02']].load()
s2_east = clean_dataset_optimized(s2_east, cloud_thresh=0.05, snow_thresh=0.1, shadow_thresh=0.1, cirrus_thresh=800)
s2_east = s2_east.drop_duplicates(dim='time', keep='first')
# Select ds_west based on matching product_uri
mask = ds_east['product_uri'].isin(s2_east['product_uri'].compute().values).compute()
ds_east = ds_east.where(mask, drop=True)

print('East', ds_east.time.values[0], ds_east.time.values[-1])
print('West', ds_west.time.values[0], ds_west.time.values[-1])
# Plot RGB next to the different FCs, see if there is a difference
time_plot = '2021-01-01'

"""
# Clip fields
fields = os.path.expanduser('~/mnt/eo-nas1/data/landuse/raw/lnf2021.gpkg')
bbox_west_utm32 = box(west[0], west[1], west[0] + 1280, west[1] + 1280)
bbox_east_utm32 = box(east[0], east[1], east[0] + 1280, east[1] + 1280)
project = pyproj.Transformer.from_crs('EPSG:32632', 'EPSG:2056', always_xy=True).transform
bbox_west_2056 = transform(project, bbox_west_utm32)
bbox_east_2056 = transform(project, bbox_east_utm32)
fields_west = gpd.read_file(fields, bbox=bbox_west_2056).to_crs(32632)
fields_west = fields_west[fields_west['geometry'].is_valid]
fields_east = gpd.read_file(fields, bbox=bbox_east_2056).to_crs(32632)
fields_east = fields_east[fields_east['geometry'].is_valid]
"""

# Clip with landuse data
swisstopo_landuse = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/EVI_CH/ag-b-00.03-37-area-current-csv.csv') # Classification on areal imagey of landuse, classes 8 and 9 are grassland
lu_csv = pd.read_csv(swisstopo_landuse,
            sep=';',
            usecols=['AS_17', 'E_COORD', 'N_COORD'],  # adjust columns to those needed in create_square
            dtype={'AS_17': 'int32', 'E': 'float64', 'N': 'float64'}  # faster parsing
        )
lu_csv = lu_csv[lu_csv['AS_17'].isin([8, 9])]
bbox_west_utm32 = box(west[0], west[1], west[0] + 1280, west[1] + 1280)
bbox_east_utm32 = box(east[0], east[1], east[0] + 1280, east[1] + 1280)
project = pyproj.Transformer.from_crs('EPSG:32632', 'EPSG:2056', always_xy=True).transform
bbox_west_2056 = transform(project, bbox_west_utm32)
bbox_east_2056 = transform(project, bbox_east_utm32)
minxW, minyW, maxxW, maxyW = bbox_west_2056.bounds
minxE, minyE, maxxE, maxyE = bbox_east_2056.bounds
lu_csv_west = lu_csv[(lu_csv['E_COORD'] >= minxW) & (lu_csv['E_COORD'] -100 <= maxxW) & 
                (lu_csv['N_COORD'] >= minyW) & (lu_csv['N_COORD'] -100 <= maxyW)]
lu_csv_west.loc[:, 'geometry'] = lu_csv_west.apply(create_square, axis=1)
fields_west = gpd.GeoDataFrame(lu_csv_west, geometry='geometry', crs='EPSG:2056').to_crs(32632)
lu_csv_east = lu_csv[(lu_csv['E_COORD'] >= minxE) & (lu_csv['E_COORD'] -100 <= maxxE) & 
                (lu_csv['N_COORD'] >= minyE) & (lu_csv['N_COORD'] -100 <= maxyE)]
lu_csv_east.loc[:, 'geometry'] = lu_csv_east.apply(create_square, axis=1)
fields_east = gpd.GeoDataFrame(lu_csv_east, geometry='geometry', crs='EPSG:2056').to_crs(32632)


ds_west = ds_west.rio.write_crs(32632).rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=False).rio.clip(fields_west.geometry, all_touched=False)
ds_east = ds_east.rio.write_crs(32632).rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=False).rio.clip(fields_east.geometry, all_touched=False)
    

fig, axs = plt.subplots(2, 4, figsize=(15, 10))

# Top row: West
time_west = ds_west.sel(time=time_plot, method='nearest').time.values
print('Time west', time_west)
axs[0, 0].set_title(f'RGB {str(time_west).split("T")[0]} - West')
scale_factor = 1.0 / 10000.0 
r = s2_west['s2_B04']* scale_factor
g = s2_west['s2_B03']* scale_factor
b = s2_west['s2_B02']* scale_factor
rgb_west = xr.concat([r, g, b], dim='band').transpose('time', 'lat', 'lon', 'band')
rgb_west = rgb_west.where(~np.isnan(rgb_west), other=1.0)
x_rgb = rgb_west['lon'].values
y_rgb = rgb_west['lat'].values
extent = [x_rgb.min(), x_rgb.max(), y_rgb.min(), y_rgb.max()]
axs[0, 0].imshow(rgb_west.sel(time=time_west, method='nearest')*3, vmin=0, vmax=3000, extent=extent)
axs[0, 1].set_title(f'FC NPV')

fc_west_npv = ds_west.sel(time=time_west)['NPV_norm_soil'].squeeze()
x = fc_west_npv['lon'].values
y = fc_west_npv['lat'].values
extent = [x.min(), x.max(), y.min(), y.max()]
axs[0, 1].imshow(fc_west_npv, vmin=0, vmax=1, cmap='coolwarm', origin='lower', extent=extent)
axs[0, 1].set_xlim(x_rgb.min(), x_rgb.max())
axs[0, 1].set_ylim(y_rgb.min(), y_rgb.max())


axs[0, 2].set_title(f'FC PV')
fc_west_pv = ds_west.sel(time=time_west)['PV_norm_soil'].squeeze()
x = fc_west_pv['lon'].values
y = fc_west_pv['lat'].values
extent = [x.min(), x.max(), y.min(), y.max()]
axs[0, 2].imshow(fc_west_pv, vmin=0, vmax=1, cmap='coolwarm', origin='lower', extent=extent)
axs[0, 2].set_xlim(x_rgb.min(), x_rgb.max())
axs[0, 2].set_ylim(y_rgb.min(), y_rgb.max())

axs[0, 3].set_title(f'FC Soil')
fc_west_soil = ds_west.sel(time=time_west)['Soil_norm_soil'].squeeze()
x = fc_west_soil['lon'].values
y = fc_west_soil['lat'].values
extent = [x.min(), x.max(), y.min(), y.max()]
axs[0, 3].imshow(fc_west_soil, vmin=0, vmax=1, cmap='coolwarm', origin='lower', extent=extent)
axs[0, 3].set_xlim(x_rgb.min(), x_rgb.max())
axs[0, 3].set_ylim(y_rgb.min(), y_rgb.max())


# Top row: East
time_east = ds_east.sel(time=time_plot, method='nearest').time.values
print('Time east', time_east)
axs[1, 0].set_title(f'RGB {str(time_east).split("T")[0]} - East')
scale_factor = 1.0 / 10000.0 
r = s2_east['s2_B04']* scale_factor
g = s2_east['s2_B03']* scale_factor
b = s2_east['s2_B02']* scale_factor
rgb_east = xr.concat([r, g, b], dim='band').transpose('time', 'lat', 'lon', 'band')
#print(rgb_east.values)
rgb_east = rgb_east.where(~np.isnan(rgb_east), other=1.0)
x_rgb = rgb_east['lon'].values
y_rgb = rgb_east['lat'].values
extent = [x_rgb.min(), x_rgb.max(), y_rgb.min(), y_rgb.max()]
axs[1, 0].imshow(rgb_east.sel(time=time_east)*3, vmin=0, vmax=3000, extent=extent)
axs[1, 0].set_xlim(x_rgb.min(), x_rgb.max())
axs[1, 0].set_ylim(y_rgb.min(), y_rgb.max())

axs[1, 1].set_title(f'FC NPV')
fc_east_npv = ds_east.sel(time=time_east)['NPV_norm_soil'].squeeze()
x = fc_east_npv['lon'].values
y = fc_east_npv['lat'].values
extent = [x.min(), x.max(), y.min(), y.max()]
axs[1, 1].imshow(fc_east_npv, vmin=0, vmax=1, cmap='coolwarm', origin='lower', extent=extent)
axs[1, 1].set_xlim(x_rgb.min(), x_rgb.max())
axs[1, 1].set_ylim(y_rgb.min(), y_rgb.max())

axs[1, 2].set_title(f'FC PV')
fc_east_pv = ds_east.sel(time=time_east)['PV_norm_soil'].squeeze()
x = fc_east_pv['lon'].values
y = fc_east_pv['lat'].values
extent = [x.min(), x.max(), y.min(), y.max()]
axs[1, 2].imshow(fc_east_pv, vmin=0, vmax=1, cmap='coolwarm', origin='lower', extent=extent)
axs[1, 2].set_xlim(x_rgb.min(), x_rgb.max())
axs[1, 2].set_ylim(y_rgb.min(), y_rgb.max())

axs[1, 3].set_title(f'FC Soil')
fc_east_soil = ds_east.sel(time=time_east)['Soil_norm_soil'].squeeze()
x = fc_east_soil['lon'].values
y = fc_east_soil['lat'].values
extent = [x.min(), x.max(), y.min(), y.max()]
axs[1, 3].imshow(fc_east_soil, vmin=0, vmax=1, cmap='coolwarm', origin='lower', extent=extent)
axs[1, 3].set_xlim(x_rgb.min(), x_rgb.max())
axs[1, 3].set_ylim(y_rgb.min(), y_rgb.max())


plt.savefig("FC_west_vs_east_grassland.png")
