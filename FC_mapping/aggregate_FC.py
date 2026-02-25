import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import numpy as np



def create_square(row):
    x_max = row['E_COORD']
    y_max = row['N_COORD']
    x_min = x_max - 100
    y_min = y_max - 100
    return box(x_min, y_min, x_max, y_max)


def compute_max_area(gdf_bounds, fields, swisstopo_landuse, lnf_codes):
  """
  Adds the total arbale/grass land area per geometry in gdf_bounds
  """
  if swisstopo_landuse is not None:
    # Further filter grassland based on other landuse data layer
    lu_csv = pd.read_csv(swisstopo_landuse,
        sep=';',
        usecols=['AS_17', 'E_COORD', 'N_COORD'],  # adjust columns to those needed in create_square
        dtype={'AS_17': 'int32', 'E_COORD': 'float64', 'N_COORD': 'float64'}  # faster parsing
    )
    lu_csv = lu_csv[lu_csv['AS_17'].isin([8, 9])].copy()
    lu_csv['geometry'] = lu_csv.apply(create_square, axis=1)
    gdf_ref  = gpd.GeoDataFrame(lu_csv, geometry='geometry', crs='EPSG:2056').to_crs(32632)


  else:
    fields_gdf = gpd.read_file(fields).to_crs(32632)
    fields_gdf = fields_gdf[fields_gdf['geometry'].is_valid]
    # Filter for arable only
    gdf_ref  = fields_gdf[fields_gdf.lnf_code.isin(lnf_codes)]
   
  # Compute intersection area
  intersections = gpd.overlay(gdf_bounds, gdf_ref, how="intersection")
  intersections["total_valid_area"] = intersections["geometry"].apply(lambda x: x.area)
  intersection_per_admin = intersections.groupby("name")["total_valid_area"].sum().reset_index()
  
  gdf_bounds = gdf_bounds.merge(intersection_per_admin, on="name", how="left")

  return gdf_bounds


def plot_median_of_pixelmean(fc_data, gdf_bounds, suffix, level, area_thresh, grass, fc_dir, lnf_codes=None):
  
  # --- Load crop classification ---
  crop_labels = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/028_Erosion/Erosion/LNF_code_classification_20251031.xlsx')
  crop_names = pd.read_excel(crop_labels, sheet_name='arable_grassland')
  lnf_codes_arable = crop_names[crop_names['Type'] == 'Arable'].LNF_code.tolist()
  lnf_codes_grassland = crop_names[crop_names['Type'] == 'Grassland'].LNF_code.tolist()

  if grass:
    fields = None
    swisstopo_landuse = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/EVI_CH/ag-b-00.03-37-area-current-csv.csv') 
    lnf_codes = lnf_codes_grassland 
  else:
    fields = os.path.expanduser(f'~/mnt/eo-nas1/data/landuse/raw/lnf{year}.gpkg')
    swisstopo_landuse = None 
    if lnf_codes is None:
      lnf_codes = lnf_codes_arable

  # Compute max possible grassland/arable land area
  gdf_bounds = compute_max_area(gdf_bounds, fields, swisstopo_landuse, lnf_codes)

  # Already drop the areas that can't contain S2
  gdf_bounds_valid = gdf_bounds[gdf_bounds.total_valid_area>300] # 1 S2 pixel is 100m2

  for f in fc_data:

    i = f.split('_')[2]
    print(f)
    
    fc = gpd.read_file(os.path.join(fc_dir, f)) # epsg 32632
    # Remove borders left after clipping fields (NA)
    fc = fc.dropna(subset=[f'PV_norm_{suffix}', f'NPV_norm_{suffix}', f'Soil_norm_{suffix}'], how='all')

    # Filter for a specific crop
    if lnf_codes is not None:
      fc = fc[fc.lnf_code.isin(lnf_codes)]
    
    # Compute per-pixel mean
    fc = fc.groupby(['lat', 'lon']).agg({
        f'PV_norm_{suffix}': 'mean',
        f'NPV_norm_{suffix}': 'mean',
        f'Soil_norm_{suffix}': 'mean',
        'lnf_code': 'first',
        'geometry': 'first'
    }).reset_index()
    fc = gpd.GeoDataFrame(fc, geometry='geometry', crs='EPSG:32632')

    # Spatial join to link each fc geometry with the canton it falls into
    joined = gpd.sjoin(gdf_bounds_valid, fc, how="left", predicate="intersects")


    ####### Filter for areas with enough coverage by FC data

    # 1. Check if at least x % of administrative area is covered by FC data (ie. if there is enough data) per timestamp
    # Count number of fc rows per geometry of gdf_bounds
    counts = joined.groupby(['name']).size().reset_index(name='count')
    counts['fc_area'] = counts['count'] * 100 # convert to area in m2
    # For each admin geom, check if there are timestamps with too few coverage
    counts = counts.merge(gdf_bounds_valid[['name', 'total_valid_area']], on='name', how='left')
    counts['fraction_area'] = counts.fc_area / counts.total_valid_area
    covered_geoms = counts[counts.fraction_area > area_thresh]

    print('Cover stats', counts.fraction_area.mean(), counts.fraction_area.min(), counts.fraction_area.max())
    print(f'With enough cover: {len(covered_geoms.name.unique())}/{len(gdf_bounds_valid)}')
    joined = joined[joined.name.isin(covered_geoms.name)]

    ####### # Stats per canton/municipality
    canton_stats = (
        joined.groupby('name')[
            [f'PV_norm_{suffix}', f'NPV_norm_{suffix}', f'Soil_norm_{suffix}']
        ]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    canton_stats.columns = ['name'] + [
        f"{col}_{stat}" for col, stat in canton_stats.columns[1:]
    ]
  
    gdf_plot = gdf_bounds.merge(canton_stats, how='left', on='name')

    # Plot 
    fig, axs = plt.subplots(3, 3, figsize=(21, 18))

    metrics = ['mean', 'std', 'count']
    titles = {
        'mean': "Mean FC",
        'std': "Std FC",
        'count': "Count FC"
    }
    cmaps = {
        'mean': ['Greens', 'Oranges', 'pink_r'],
        'std': ['Reds', 'Reds', 'Reds'],  # could pick others
        'count': ['Reds', 'Reds', 'Reds']     # usually same colormap
    }
    variables = ['PV_norm', 'NPV_norm', 'Soil_norm']
    # set vmin/vmax per metric
    limits = {
        'mean': (0, 1),
        'std': (0, 0.5),
        'count': (0, None)  # None lets geopandas auto-pick the max
    }

    for row, metric in enumerate(metrics):
        for col, var in enumerate(variables):
            ax = axs[row, col]

            gdf_bounds.plot(ax=ax, facecolor='whitesmoke', edgecolor='black', linewidth=0.2)

            vmin, vmax = limits[metric]

            gdf_plot.plot(
                ax=ax,
                column=f"{var}_{suffix}_{metric}",
                cmap=cmaps[metric][col],
                legend=True,
                edgecolor='black',
                linewidth=0.2,
                vmin=vmin,
                vmax=vmax,
                missing_kwds={
                  "color": "lightgray",
                  "edgecolor":"gray",
                }
            )
            ax.set_title(f"{titles[metric]} {var.split('_')[0]} per {level}", fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{fc_dir}/CH_fraction_{i}_{'grass' if grass else 'arable'}_{suffix}_{level.upper()}.png')
    print(f'Saved {fc_dir}/CH_fraction_{i}_{'grass' if grass else 'arable'}_{suffix}_{level.upper()}.png')

    # Save as gpkg
    gdf_plot.to_file(f'{fc_dir}/CH_fraction_{i}_{'grass' if grass else 'arable'}_{suffix}_{level.upper()}.gpkg')
   
  return


def plot_median_of_pixelmean_bivariate(fc_data, gdf_bounds, suffix, level, area_thresh, grass, fc_dir, lnf_codes=None):
  
  # --- Load crop classification ---
  crop_labels = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/028_Erosion/Erosion/LNF_code_classification_20251031.xlsx')
  crop_names = pd.read_excel(crop_labels, sheet_name='arable_grassland')
  lnf_codes_arable = crop_names[crop_names['Type'] == 'Arable'].LNF_code.tolist()
  lnf_codes_grassland = crop_names[crop_names['Type'] == 'Grassland'].LNF_code.tolist()

  if grass:
    fields = None
    swisstopo_landuse = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/EVI_CH/ag-b-00.03-37-area-current-csv.csv') 
    lnf_codes = lnf_codes_grassland 
  else:
    fields = os.path.expanduser(f'~/mnt/eo-nas1/data/landuse/raw/lnf{year}.gpkg')
    swisstopo_landuse = None 
    if lnf_codes is None:
      lnf_codes = lnf_codes_arable

  # Compute max possible grassland/arable land area
  gdf_bounds = compute_max_area(gdf_bounds, fields, swisstopo_landuse, lnf_codes)

  # Already drop the areas that can't contain S2
  gdf_bounds_valid = gdf_bounds[gdf_bounds.total_valid_area>300] # 1 S2 pixel is 100m2

  for f in fc_data:

    i = f.split('_')[2]
    print(f)
    
    fc = gpd.read_file(os.path.join(fc_dir, f)) # epsg 32632
    # Remove borders left after clipping fields (NA)
    fc = fc.dropna(subset=[f'PV_norm_{suffix}', f'NPV_norm_{suffix}', f'Soil_norm_{suffix}'], how='all')

    # Filter for a specific crop
    if lnf_codes is not None:
      fc = fc[fc.lnf_code.isin(lnf_codes)]
    
    # Compute per-pixel mean
    fc = fc.groupby(['lat', 'lon']).agg({
        f'PV_norm_{suffix}': 'mean',
        f'NPV_norm_{suffix}': 'mean',
        f'Soil_norm_{suffix}': 'mean',
        'lnf_code': 'first',
        'geometry': 'first'
    }).reset_index()
    fc = gpd.GeoDataFrame(fc, geometry='geometry', crs='EPSG:32632')

    # Spatial join to link each fc geometry with the canton it falls into
    joined = gpd.sjoin(gdf_bounds, fc, how="left", predicate="intersects")


    ####### Filter for areas with enough coverage by FC data

    # 1. Check if at least x % of administrative area is covered by FC data (ie. if there is enough data) per timestamp
    # Count number of fc rows per geometry of gdf_bounds
    counts = joined.groupby(['name']).size().reset_index(name='count')
    counts['fc_area'] = counts['count'] * 100 # convert to area in m2
    # For each admin geom, check if there are timestamps with too few coverage
    counts = counts.merge(gdf_bounds_valid[['name', 'total_valid_area']], on='name', how='left')
    counts['fraction_area'] = counts.fc_area / counts.total_valid_area
    covered_geoms = counts[counts.fraction_area > area_thresh]

    print('Cover stats', counts.fraction_area.mean(), counts.fraction_area.min(), counts.fraction_area.max())
    print(f'With enough cover: {len(covered_geoms.name.unique())}/{len(gdf_bounds_valid)}')
    joined = joined[joined.name.isin(covered_geoms.name)]

    ####### # Stats per canton/municipality
    canton_stats = (
        joined.groupby('name')[
            [f'PV_norm_{suffix}', f'NPV_norm_{suffix}', f'Soil_norm_{suffix}']
        ]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    canton_stats.columns = ['name'] + [
        f"{col}_{stat}" for col, stat in canton_stats.columns[1:]
    ]
  
    gdf_plot = gdf_bounds.merge(canton_stats, how='left', on='name')
    

    # Normalize the count for transparency or size scaling
    gdf_plot['count_norm'] = gdf_plot[f"{variables[0]}_{suffix}_count"] / gdf_plot[f"{variables[0]}_{suffix}_count"].max()

    # Plot
    fig, axs = plt.subplots(3, 3, figsize=(21, 18))

    metrics = ['mean', 'std', 'count']
    titles = {
        'mean': "Mean FC",
        'std': "Std FC",
        'count': "Count FC"
    }
    cmaps = {
        'mean': ['Greens', 'Oranges', 'pink_r'],
        'std': ['Reds', 'Reds', 'Reds'],  # could pick others
        'count': ['Reds', 'Reds', 'Reds']     # usually same colormap
    }
    variables = ['PV_norm', 'NPV_norm', 'Soil_norm']
    # set vmin/vmax per metric
    limits = {
        'mean': (0, 1),
        'std': (0, 0.5),
        'count': (0, None)  # None lets geopandas auto-pick the max
    }

    for row, metric in enumerate(metrics):
        for col, var in enumerate(variables):
            ax = axs[row, col]

            gdf_bounds.plot(ax=ax, facecolor='whitesmoke', edgecolor='black', linewidth=0.2)

            vmin, vmax = limits[metric]

            # Use color for the variable and transparency for the count
            gdf_plot.plot(
                ax=ax,
                column=f"{var}_{suffix}_{metric}",
                cmap=cmaps[metric][col],
                legend=True,
                edgecolor='black',
                linewidth=0.2,
                vmin=vmin,
                vmax=vmax,
                alpha=gdf_plot['count_norm'],  # Transparency based on normalized count
                missing_kwds={
                  "color": "lightgray",
                  "edgecolor":"gray",
                }
            )
            ax.set_title(f"{titles[metric]} {var.split('_')[0]} per {level}", fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{fc_dir}/CH_bivariate_{i}_{'grass' if grass else 'arable'}_{suffix}_{level.upper()}.png')
    print(f'Saved {fc_dir}/CH_bivariate_{i}_{'grass' if grass else 'arable'}_{suffix}_{level.upper()}.png')

    
  return


def count_fraction_soil_above_50percent(fc_data, gdf_bounds, suffix, level, area_thresh, grass, fc_dir, lnf_codes=None):
  
  # --- Load crop classification ---
  crop_labels = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/028_Erosion/Erosion/LNF_code_classification_20251031.xlsx')
  crop_names = pd.read_excel(crop_labels, sheet_name='arable_grassland')
  lnf_codes_arable = crop_names[crop_names['Type'] == 'Arable'].LNF_code.tolist()
  lnf_codes_grassland = crop_names[crop_names['Type'] == 'Grassland'].LNF_code.tolist()

  if grass:
    fields = None
    swisstopo_landuse = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/EVI_CH/ag-b-00.03-37-area-current-csv.csv') 
    lnf_codes = lnf_codes_grassland 
  else:
    fields = os.path.expanduser(f'~/mnt/eo-nas1/data/landuse/raw/lnf{year}.gpkg')
    swisstopo_landuse = None 
    if lnf_codes is None:
      lnf_codes = lnf_codes_arable

  # Compute max possible grassland/arable land area
  gdf_bounds = compute_max_area(gdf_bounds, fields, swisstopo_landuse, lnf_codes)

  # Already drop the areas that can't contain S2
  gdf_bounds_valid = gdf_bounds[gdf_bounds.total_valid_area>300] # 1 S2 pixel is 100m2

  for f in fc_data:

    i = f.split('_')[2]
    print(f)
    
    fc = gpd.read_file(os.path.join(fc_dir, f)) # epsg 32632
    # Remove borders left after clipping fields (NA)
    fc = fc.dropna(subset=[f'PV_norm_{suffix}', f'NPV_norm_{suffix}', f'Soil_norm_{suffix}'], how='all')

    # Filter for a specific crop
    if lnf_codes is not None:
      fc = fc[fc.lnf_code.isin(lnf_codes)]
    
    # Compute per-pixel mean
    fc = fc.groupby(['lat', 'lon']).agg({
        f'PV_norm_{suffix}': 'mean',
        f'NPV_norm_{suffix}': 'mean',
        f'Soil_norm_{suffix}': 'mean',
        'lnf_code': 'first',
        'geometry': 'first'
    }).reset_index()
    fc = gpd.GeoDataFrame(fc, geometry='geometry', crs='EPSG:32632')

    # Spatial join to link each fc geometry with the canton it falls into
    joined = gpd.sjoin(gdf_bounds, fc, how="left", predicate="intersects")


    ####### Filter for areas with enough coverage by FC data

    # Check if at least x % of administrative area is covered by FC data (ie. if there is enough data) per timestamp
    # Count number of fc rows per geometry of gdf_bounds
    counts = joined.groupby(['name']).size().reset_index(name='count')
    counts['fc_area'] = counts['count'] * 100 # convert to area in m2
    # For each admin geom, check if there are timestamps with too few coverage
    counts = counts.merge(gdf_bounds_valid[['name', 'total_valid_area']], on='name', how='left')
    counts['fraction_area'] = counts.fc_area / counts.total_valid_area
    covered_geoms = counts[counts.fraction_area > area_thresh]

    print('Cover stats', counts.fraction_area.mean(), counts.fraction_area.min(), counts.fraction_area.max())
    print(f'With enough cover: {len(covered_geoms.name.unique())}/{len(gdf_bounds_valid)}')
    joined = joined[joined.name.isin(covered_geoms.name)]

    ####### Count number of pixels where FC soil > 0.5

    stats = (
      joined
        .groupby('name')[f'Soil_norm_{suffix}']
        .agg(
            total_pixels=lambda x: x.notna().sum(),
            above_threshold=lambda x: (x > 0.5).sum()
        )
        .reset_index()
    )

    stats['fraction_above_threshold'] = (
        stats['above_threshold'] / stats['total_pixels']
    )

    # Merge the stats back with the gdf_bounds
    gdf_plot = gdf_bounds.merge(stats, on='name', how='left')

  
    ####### Save gpkg
    gdf_plot.to_file(f'{fc_dir}/CH_fraction_soilthresh_{i}_{level.upper()}.gpkg', driver="GPKG")
    print(f"Saved GeoPackage: {fc_dir}/CH_fraction_soilthresh_{i}_{level.upper()}.gpkg")
    
  return


def avg_baresoil_days(fc_data, gdf_bounds, suffix, level, area_thresh, grass, fc_dir, lnf_codes=None):
  
  # --- Load crop classification ---
  crop_labels = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/028_Erosion/Erosion/LNF_code_classification_20251031.xlsx')
  crop_names = pd.read_excel(crop_labels, sheet_name='arable_grassland')
  lnf_codes_arable = crop_names[crop_names['Type'] == 'Arable'].LNF_code.tolist()
  lnf_codes_grassland = crop_names[crop_names['Type'] == 'Grassland'].LNF_code.tolist()

  if grass:
    fields = None
    swisstopo_landuse = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/EVI_CH/ag-b-00.03-37-area-current-csv.csv') 
    lnf_codes = lnf_codes_grassland 
  else:
    fields = os.path.expanduser(f'~/mnt/eo-nas1/data/landuse/raw/lnf{year}.gpkg')
    swisstopo_landuse = None 
    if lnf_codes is None:
      lnf_codes = lnf_codes_arable

  # Compute max possible grassland/arable land area
  gdf_bounds = compute_max_area(gdf_bounds, fields, swisstopo_landuse, lnf_codes)

  # Already drop the areas that can't contain S2
  gdf_bounds_valid = gdf_bounds[gdf_bounds.total_valid_area>300] # 1 S2 pixel is 100m2

  bare_days = None
  for f in fc_data:

      i = f.split('_')[2]
      print(f)
      
      fc = gpd.read_file(os.path.join(fc_dir, f)) # epsg 32632
      # Remove borders left after clipping fields (NA)
      fc = fc.dropna(subset=[f'Soil_norm_{suffix}'])

      # Filter for a specific crop
      if lnf_codes is not None:
        fc = fc[fc.lnf_code.isin(lnf_codes)]
      
      # Compute per-pixel mean
      fc = fc.groupby(['lat', 'lon']).agg({
          f'Soil_norm_{suffix}': 'mean',
          'geometry': 'first'
      }).reset_index()
      fc = gpd.GeoDataFrame(fc, geometry='geometry', crs='EPSG:32632')

      # Binary bare-soil state for this week
      thresh_bare = 0.5
      fc['is_bare'] = fc[f'Soil_norm_{suffix}'] > thresh_bare

      if bare_days is None:
          # Initialize accumulators using first week
          bare_days = fc[['lat', 'lon', 'geometry']].copy()
          bare_days['bare_days'] = 0
          bare_days['valid_weeks'] = 0
      
      # Align current week with accumulator
      merged = bare_days.merge(
          fc[['lat', 'lon', 'is_bare']],
          on=['lat', 'lon'],
          how='left'
      )

      # Update accumulators
      merged.loc[merged['is_bare'] == True, 'bare_days'] += 7
      merged.loc[merged['is_bare'].notna(), 'valid_weeks'] += 1

      # Drop helper column and write back
      bare_days = merged.drop(columns='is_bare')
    
  
  # Keep ony pixels where at least 50% year has data
  total_weeks = len(fc_data)
  bare_days['valid_fraction'] = bare_days['valid_weeks'] / total_weeks
  bare_days.loc[bare_days['valid_fraction'] < 0.5, 'bare_days'] = np.nan

  # Aggregate to municipality
  joined = gpd.sjoin(gdf_bounds, bare_days, how="left", predicate="intersects")

  municipal_stats = (
      joined.groupby('name')['bare_days']
      .agg(
          mean_bare_days='mean'
      )
      .reset_index()
  )

  municipal_gdf = gdf_bounds.merge(
      municipal_stats,
      on='name',
      how='left'
  )

  # Save
  municipal_gdf.to_file(f'{fc_dir}/CH_baredays_{level.upper()}.gpkg', driver="GPKG")
  print(f"Saved GeoPackage: {fc_dir}/CH_baredays_{level.upper()}.gpkg")
    
  return


swissbounds = 'swissBOUNDARIES3D_1_5_LV95_LN02.gpkg'
grass = False
level = 'commune'
year = 2023
fc_dir = f'CH_fraction_weekly_{year}'
area_thresh = 0.5

if level=='canton':
  gdf_bounds = gpd.read_file(swissbounds, layer='tlm_kantonsgebiet').to_crs(32632) #originally in epsg 2056
if level=='commune':
  gdf_bounds = gpd.read_file(swissbounds, layer='tlm_hoheitsgebiet').to_crs(32632) #originally in epsg 2056

if grass:
  suffix = 'global'
  fc_data = [f for f in os.listdir(fc_dir) if f.endswith('raw.gpkg') and 'grass' in f]
else:
  suffix = 'soil'
  fc_data = [f for f in os.listdir(fc_dir) if f.endswith('raw.gpkg') and 'grass' not in f]


lnf_codes = None #[513]  winterwheat [524] potatoes [522] sugar beet
plot_median_of_pixelmean(fc_data=fc_data, gdf_bounds=gdf_bounds, suffix=suffix, level=level, area_thresh=area_thresh, grass=grass, fc_dir=fc_dir, lnf_codes=lnf_codes)

#avg_baresoil_days(fc_data, gdf_bounds, suffix, level, area_thresh, grass, fc_dir, lnf_codes=None)
#count_fraction_soil_above_50percent(fc_data=fc_data, gdf_bounds=gdf_bounds, suffix=suffix, level=level, area_thresh=area_thresh, grass=grass, fc_dir=fc_dir, lnf_codes=lnf_codes)
