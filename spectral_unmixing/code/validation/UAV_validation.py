import os
import xarray as xr
import rioxarray
import geopandas as gpd
from shapely.geometry import box
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import contextily as cx
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys
sys.path.insert(0, str(Path(os.path.dirname(os.path.realpath("__file__"))).parent))
from models import MODELS
import torch
import joblib
from scipy.stats import linregress, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



def round_down_to_nearest_10(value):
    return value - (value % 10)


def round_up_to_nearest_10(value):
    return math.ceil(value / 10) * 10


def find_closest_date(ds, uav_date):

    # Find nearest dates to UAV flight
    uav_date = pd.to_datetime(uav_date)
    times = ds.time.values

    if uav_date in times:
      return [uav_date]

    else:
      before = times[times < uav_date]
      after = times[times > uav_date]
      closest_before = before[-1] if before.size > 0 else None
      closest_after = after[0] if after.size > 0 else None
      return [closest_before, closest_after]


def trim_raster(ds):
    # Trim the raster (there are 0s filling around the data)
    mask_ones = mask == 1
    y_any = mask_ones.any(dim='x') #find where 1s exist
    x_any = mask_ones.any(dim='y')
    y_vals = mask.y.values
    x_vals = mask.x.values
    min_y = y_vals[y_any.values][0]
    max_y = y_vals[y_any.values][-1]
    min_x = x_vals[x_any.values][0]
    max_x = x_vals[x_any.values][-1]
    trimmed_mask = mask.sel(
        y=slice(min_y, max_y),
        x=slice(min_x, max_x)
    )

    return trimmed_mask


def create_s2_pix_grid(xmin, xmax, ymin, ymax, row):
    # Create S2 pixel grid, omitting fields border
    start_x = xmin if row.left < xmin else row.left
    end_x = xmax if row.right > xmax else row.right
    start_y = ymax if row.top > ymax else row.top
    end_y = ymin if row.bottom < ymin else row.bottom
    s2_pixels_x = np.arange(start_x, end_x, 10)
    s2_pixels_y = np.arange(start_y, end_y, -10)

    return s2_pixels_x, s2_pixels_y


def generate_valdata_csv(uav_data_path, s2_data_path, s2_bands, grid_shp, val_data_path):
  """ 
  Upscale drone data to S2 pixels and compute FC
  Save S2 bands for corresponding flight dates (same date, nearest dates possible right before/after)
  """

  fc_results = []

  for location in os.listdir(uav_data_path):
      print(f'-------------------------\n--Location: {location}')
      for site in os.listdir(os.path.join(uav_data_path, location)):
          print(f'---Site: {site}')
          for mission in os.listdir(os.path.join(uav_data_path, location, site)):
              if os.path.isdir(os.path.join(uav_data_path, location, site, mission)) and mission.endswith('MappingLayer'):
                print(f'----Mission: {mission}')
                year = mission[:4]
                date = mission[:8]

                for layer in os.listdir(os.path.join(uav_data_path, location, site, mission)):
                    if layer.endswith('prediction_epsg32632.tif'):
                        print(layer)

                        # Tbd: are coords center or top left?
                        mask = rioxarray.open_rasterio(os.path.join(uav_data_path, location, site, mission, layer))
                        mask = mask.sel(band=1)
                        dx = mask.x.values[1]-mask.x.values[0]
                        dy = mask.y.values[0]-mask.y.values[1]

                        # Clip with field shp
                        mask_shp = gpd.read_file(os.path.join(uav_data_path, location, site, mission, layer.replace('.tif', '_outline.shp')), crs=32632)
                        mask = mask.rio.clip(mask_shp.geometry)

                        # Trim the raster (there are 0s filling around the data)
                        #mask = trim_raster(mask)
                        xmin, xmax = round_down_to_nearest_10(mask.x.values.min()), round_up_to_nearest_10(mask.x.values.max())
                        ymin, ymax = round_down_to_nearest_10(mask.y.values.min()), round_up_to_nearest_10(mask.y.values.max())
                        
                        # Intersect with S2 10m-grid to find cube of interest
                        s2_cubes = grid_shp.cx[xmin:xmax, ymin:ymax]

          
                        for i, row in s2_cubes.iterrows():

                          f = [f for f in os.listdir(s2_data_path) if f'{int(row.left)}_{int(row.top)}' in f and f.split('_')[3].startswith(year)][0]
                          ds = xr.open_zarr(os.path.join(s2_data_path, f)).compute()
                          ds = ds.drop_duplicates('time', keep='first')

                          # Get closest dates to UAV flight
                          closest_dates = find_closest_date(ds, date)
                          
                          # Create S2 pixel grid over field, omitting border 
                          s2_pixels_x, s2_pixels_y = create_s2_pix_grid(xmin+10, xmax-10, ymin+10, ymax-10, row)
                          s2area = gpd.GeoDataFrame({'geometry': [box(xmin, ymin, xmax, ymax)]}, crs=mask.rio.crs)

                          for x in s2_pixels_x:
                            for y in s2_pixels_y:
                              try:
                                # Get FC per S2 pix              
                                pix = box(x, y - 10, x + 10, y)  # (minx, miny, maxx, maxy)
                                pix_fc = mask.rio.clip(gpd.GeoDataFrame({'geometry': [pix]}, crs=mask.rio.crs).geometry)
                                fc = pix_fc.sum() / pix_fc.size

                                # Save coordinates, FC, dates, s2 bands, to table
                                for t in closest_dates:
                                  fc_results.append({
                                    'x': x,
                                    'y': y,
                                    'fc': fc.values,
                                    'mask_file': os.path.join(uav_data_path, location, site, mission, layer),
                                    'uav_date': date,
                                    's2_date': t,
                                    **{band: s.item() for band, s in ds.sel(time=t, lat=y, lon=x)[s2_bands].data_vars.items()}
                                  })
                              except Exception as e:
                                print(e)
                                # not all pixels overlap the data
                                continue
                    
                              
  df = pd.DataFrame(fc_results)
  df.to_csv(val_data_path, index=False)

  return


def plot_valdata(val_data_path, save_dir):

    df = pd.read_csv(val_data_path)

    missions = df.mask_file.unique()

    for mission in missions:

        df_mission = df[df.mask_file==mission]

        # There could be several S2 dates per UAV flight
        duplicates = df_mission[df_mission.duplicated(subset=['x', 'y', 'uav_date'], keep=False)]
        s2_dates = duplicates['s2_date'].unique()

        for d in s2_dates:
          df_val = df_mission[df_mission.s2_date==d]

          # Pivot the data to create 2D grids (assumes gridded data!)
          df_fc = df_val.pivot(index='y', columns='x', values='fc')
          df_r = df_val.pivot(index='y', columns='x', values='s2_B04')
          df_g = df_val.pivot(index='y', columns='x', values='s2_B03')
          df_b = df_val.pivot(index='y', columns='x', values='s2_B02')

          x_coords = df_fc.columns.values
          y_coords = df_fc.index.values
          extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]

          # Create 2D arrays
          fc_grid = df_fc.values
          r_grid = df_r.values / 10000
          g_grid = df_g.values / 10000
          b_grid = df_b.values / 10000

          # Stack into RGB image
          rgb_image = np.stack([r_grid, g_grid, b_grid], axis=-1)

          # Define figure and GridSpec layout
          fig = plt.figure(figsize=(14, 6))
          gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.2)

          # Subplots for images
          ax1 = fig.add_subplot(gs[0])
          ax2 = fig.add_subplot(gs[1])
          cax = fig.add_subplot(gs[2])  # colorbar axis

          # RGB image
          ax1.imshow(rgb_image*3, origin='lower', extent=extent)
          ax1.set_title('RGB S2')
          ax1.set_xlabel('x')
          ax1.set_ylabel('y')
          ax1.set_aspect('equal')

          # FC image
          im = ax2.imshow(fc_grid, origin='lower', extent=extent, cmap='Greens')
          ax2.set_title('FC from UAV')
          ax2.set_xlabel('x')
          ax2.set_ylabel('y')
          ax2.set_aspect('equal')

          # Colorbar in separate axis
          fig.colorbar(im, cax=cax, label='fc')

          plt.suptitle(mission.split('/')[-1])
          plt.savefig(os.path.join(save_dir, f'GT_{mission.split("/")[-1]}_{d.strip("-")}.png'))

    return


def plot_valdata_preds(pred_path, class_type, save_dir, model_nbr=0):

    df = pd.read_csv(pred_path)
    missions = df.mask_file.unique()

    class_col = f'{class_type.lower()}_pred_{model_nbr}'

    df[class_col] = df.apply(lambda x: x[class_col]/(x[f'pv_pred_{model_nbr}'] + x[f'npv_pred_{model_nbr}'] + x[f'soil_pred_{model_nbr}']), axis=1)

    for mission in missions:

        df_mission = df[df.mask_file==mission]

        # There could be several S2 dates per UAV flight
        duplicates = df_mission[df_mission.duplicated(subset=['x', 'y', 'uav_date'], keep=False)]
        s2_dates = duplicates['s2_date'].unique()

        for d in s2_dates:
          df_val = df_mission[df_mission.s2_date==d]

          # Pivot the data to create 2D grids (assumes gridded data!)
          df_fc = df_val.pivot(index='y', columns='x', values='fc')
          df_preds = df_val.pivot(index='y', columns='x', values=class_col)
          df_r = df_val.pivot(index='y', columns='x', values='s2_B04')
          df_g = df_val.pivot(index='y', columns='x', values='s2_B03')
          df_b = df_val.pivot(index='y', columns='x', values='s2_B02')

          x_coords = df_fc.columns.values
          y_coords = df_fc.index.values
          extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]

          # Create 2D arrays
          fc_grid = df_fc.values
          pred_grid = df_preds.values
          r_grid = df_r.values / 10000
          g_grid = df_g.values / 10000
          b_grid = df_b.values / 10000

          # Stack into RGB image
          rgb_image = np.stack([r_grid, g_grid, b_grid], axis=-1)

          # Define figure and GridSpec layout
          fig = plt.figure(figsize=(14, 6))
          gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.2)

          # Subplots for images
          ax1 = fig.add_subplot(gs[0])
          ax2 = fig.add_subplot(gs[1])
          ax3 = fig.add_subplot(gs[2])
          cax = fig.add_subplot(gs[3])  # colorbar axis

          # RGB image
          ax1.imshow(rgb_image*3, origin='lower', extent=extent)
          ax1.set_title('RGB S2')
          ax1.set_xlabel('x')
          ax1.set_ylabel('y')
          ax1.set_aspect('equal')

          # FC image
          im = ax2.imshow(fc_grid, origin='lower', extent=extent, cmap='Greens', vmin=0, vmax=1)
          ax2.set_title('FC from UAV')
          ax2.set_xlabel('x')
          ax2.set_ylabel('y')
          ax2.set_aspect('equal')

          # Preds
          im_pred = ax3.imshow(pred_grid, origin='lower', extent=extent, cmap='Greens', vmin=0, vmax=1)
          ax3.set_title('FC from NN global')
          ax3.set_xlabel('x')
          ax3.set_ylabel('y')
          ax3.set_aspect('equal')

          # Colorbar in separate axis
          fig.colorbar(im, cax=cax, label='FC')

          plt.suptitle(mission.split('/')[-1])
          plt.savefig(os.path.join(save_dir, f'preds_{mission.split("/")[-1]}_{d.strip("-")}.png'))

    return


def plot_valdata_preds_modelcompare(pred_path, class_type, save_dir):

    df = pd.read_csv(pred_path)
    missions = df.mask_file.unique()

    # Identify all relevant prediction groups
    pred_cols = [col for col in df.columns if col.startswith(f'{class_type.lower()}_pred')]
    pred_sets = [col.split('_')[-1] for col in pred_cols]  # extract suffixes (e.g. '0', '1')
    
    # Normalize each prediction set individually
    for suffix in pred_sets:
        pv_col = f'pv_pred_{suffix}'
        npv_col = f'npv_pred_{suffix}'
        soil_col = f'soil_pred_{suffix}'
        target_col = f'{class_type.lower()}_pred_{suffix}'

        df[target_col] = df.apply(
            lambda x: x[target_col] / (x[pv_col] + x[npv_col] + x[soil_col]) if (x[pv_col] + x[npv_col] + x[soil_col]) != 0 else 0,
            axis=1
        )

    for mission in missions:

        df_mission = df[df.mask_file==mission]

        # There could be several S2 dates per UAV flight
        duplicates = df_mission[df_mission.duplicated(subset=['x', 'y', 'uav_date'], keep=False)]
        s2_dates = duplicates['s2_date'].unique()

        for d in s2_dates:
          df_val = df_mission[df_mission.s2_date==d]

          # Pivot the data to create 2D grids (assumes gridded data!)
          df_fc = df_val.pivot(index='y', columns='x', values='fc')
          df_r = df_val.pivot(index='y', columns='x', values='s2_B04')
          df_g = df_val.pivot(index='y', columns='x', values='s2_B03')
          df_b = df_val.pivot(index='y', columns='x', values='s2_B02')

          x_coords = df_fc.columns.values
          y_coords = df_fc.index.values
          extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]

          # Create 2D arrays
          fc_grid = df_fc.values
          r_grid = df_r.values / 10000
          g_grid = df_g.values / 10000
          b_grid = df_b.values / 10000

          # Stack into RGB image
          rgb_image = np.stack([r_grid, g_grid, b_grid], axis=-1)

          # Define figure and GridSpec layout
          n_preds = len(pred_sets)
          fig = plt.figure(figsize=(5 * (2 + n_preds), 6))
          gs = gridspec.GridSpec(1, 2 + n_preds + 1, width_ratios=[1] * (2 + n_preds) + [0.05], wspace=0.2)

          # RGB image
          ax1 = fig.add_subplot(gs[0])
          ax1.imshow(rgb_image * 3, origin='lower', extent=extent)
          ax1.set_title('RGB S2')
          ax1.set_xlabel('x')
          ax1.set_ylabel('y')
          ax1.set_aspect('equal')

          # UAV FC image
          ax2 = fig.add_subplot(gs[1])
          im_fc = ax2.imshow(fc_grid, origin='lower', extent=extent, cmap='Greens', vmin=0, vmax=1)
          ax2.set_title('FC from UAV')
          ax2.set_xlabel('x')
          ax2.set_ylabel('y')
          ax2.set_aspect('equal')

          # NN Predictions
          for i, suffix in enumerate(pred_sets):
              pred_col = f'{class_type.lower()}_pred_{suffix}'
              pred_grid = df_val.pivot(index='y', columns='x', values=pred_col).values

              ax = fig.add_subplot(gs[2 + i])
              ax.imshow(pred_grid, origin='lower', extent=extent, cmap='Greens', vmin=0, vmax=1)
              ax.set_title(f'FC NN soil group {pred_col}')
              ax.set_xlabel('x')
              ax.set_ylabel('y')
              ax.set_aspect('equal')

          # Colorbar
          cax = fig.add_subplot(gs[-1])
          fig.colorbar(im_fc, cax=cax, label='FC')

          plt.suptitle(mission.split('/')[-1])
          plt.savefig(os.path.join(save_dir, f'compare_preds_{mission.split("/")[-1]}_{d.strip("-")}.png'))

    return


def plot_valdata_scatters(pred_path, class_type, save_dir, model_nbr=0):

    df = pd.read_csv(pred_path)
    missions = df.mask_file.unique()

    class_col = f'{class_type.lower()}_pred_{model_nbr}'

    df[class_col] = df.apply(lambda x: x[class_col]/(x[f'pv_pred_{model_nbr}'] + x[f'npv_pred_{model_nbr}'] + x[f'soil_pred_{model_nbr}']), axis=1)

    for mission in missions:

        df_mission = df[df.mask_file==mission]

        # There could be several S2 dates per UAV flight
        duplicates = df_mission[df_mission.duplicated(subset=['x', 'y', 'uav_date'], keep=False)]
        s2_dates = duplicates['s2_date'].unique()

        for d in s2_dates:
          df_val = df_mission[df_mission.s2_date==d]

          fig, ax = plt.subplots(figsize=(8, 8))
          ax.scatter(df_val.fc, df_val[class_col], alpha=0.5)
          ax.set_xlabel('FC from UAV')
          ax.set_ylabel('FC from NN')

          # Plot 1:1 line
          lims = [min(df_val.fc.min(), df_val[class_col].min()),
                  max(df_val.fc.max(), df_val[class_col].max())]
          ax.plot(lims, lims, 'k--', label='1:1 line')

          # Compute rmse, pearson r2, mae and add to plot
          rmse = np.sqrt(mean_squared_error(df_val.fc, df_val[class_col]))
          mae = mean_absolute_error(df_val.fc, df_val[class_col])
          r, p_value = pearsonr(df_val.fc, df_val[class_col])
          r2 = r**2
          textstr = f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\n$R^2$: {r2:.3f}'
          ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
          
          # Plot regresison line
          slope, intercept, r_value, p_value, std_err = linregress(df_val.fc, df_val[class_col])
          x_vals = np.linspace(*lims, 100)
          ax.plot(x_vals, slope * x_vals + intercept, 'r-', label='Regression line')

          ax.set_title(f'Validation on UAV PV FC data')

          ax.legend()
          plt.tight_layout()
          plt.savefig(os.path.join(save_dir, f'scatter_{mission.split("/")[-1]}_{d.strip("-")}.png'))

    return


def predict_FC(val_data_path, s2_bands, pred_path, soil_group, model_type):
    """
    Predict FC with global model
    """
    df = pd.read_csv(val_data_path)

    X = (df[s2_bands] / 10000).values

    # Prepare storage
    predictions_pv, predictions_npv, predictions_soil = [], [], []
    
    for iteration in range(1, 6):
        # Load models
        model_pv = joblib.load(f'../models/{model_type}_CLASS2_SOIL{soil_group}_ITER{iteration}.pkl')
        model_npv = joblib.load(f'../models/{model_type}_CLASS1_SOIL{soil_group}_ITER{iteration}.pkl')
        model_soil = joblib.load(f'../models/{model_type}_CLASS3_SOIL{soil_group}_ITER{iteration}.pkl')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tensor = torch.FloatTensor(X).to(device)

        pv_pred = model_pv(X_tensor)
        npv_pred = model_npv(X_tensor)
        soil_pred = model_soil(X_tensor)

        if device == torch.device("cuda"):
            pv_pred = pv_pred.cpu().detach().numpy()
            npv_pred = npv_pred.cpu().detach().numpy()
            soil_pred = soil_pred.cpu().detach().numpy()

        predictions_pv.append(pv_pred)
        predictions_npv.append(npv_pred)
        predictions_soil.append(soil_pred)

    # Average and store
    mean_pv = np.mean(predictions_pv, axis=0)
    mean_npv = np.mean(predictions_npv, axis=0)
    mean_soil = np.mean(predictions_soil, axis=0)

    # Save predictions
    df[f'pv_pred_{soil_group}'] = mean_pv
    df[f'npv_pred_{soil_group}'] = mean_npv
    df[f'soil_pred_{soil_group}'] = mean_soil

    df.to_csv(pred_path, index=False)

    return





if __name__ == '__main__':

  #####################
  # EXTRACT VAL DATA FROM UAV DATA

  uav_data_path = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/011_experimentEObserver/data/UAV/Layer/')
  s2_data_path = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/')
  s2_bands = ['s2_B02','s2_B03','s2_B04','s2_B05','s2_B06','s2_B07','s2_B08','s2_B8A','s2_B11','s2_B12']
  s2_grid = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp')
  grid_shp = gpd.read_file(s2_grid, crs=32632)
  val_data_path = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/spectral_unmixing/data/UAV_FC_clip.csv')

  generate_valdata_csv(uav_data_path, s2_data_path, s2_bands, grid_shp, val_data_path)

  
  #####################
  # PLOT VAL FC DATA AND RGB 

  # Will randomly select a site
  save_dir = '../results/UAV_validation/'
  os.makedirs(save_dir, exist_ok=True)
  plot_valdata(val_data_path, save_dir)


  #####################
  # PREDICT FC PV
 
  res_dir = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/spectral_unmixing/results/UAV_validation/')
  os.makedirs(res_dir, exist_ok=True)
  result_path = os.path.join(res_dir, 'UAV_FC_preds.csv')
 
  model_type = 'NN'

  for soil_group in range(0,6):
    if not os.path.exists(result_path):
      predict_FC(val_data_path, s2_bands, result_path, soil_group, model_type)
    else:
      # Append to existing predictions
      predict_FC(result_path, s2_bands, result_path, soil_group, model_type)

  #####################
  # PLOT PREDS vs GROUND TRUTH

  # For the global model
  plot_valdata_preds(result_path, 'PV', res_dir) # plot S2 RGB, UAV FV, soil group 0 FC
  plot_valdata_scatters(result_path, 'PV', res_dir) 
 
  plot_valdata_preds_modelcompare(result_path, 'PV', res_dir) # plot S2 RGB, UAV FV, each soil group FC



  # Report overall scores
  df = pd.read_csv(result_path)

  # Identify all relevant prediction groups
  pred_cols = [col for col in df.columns if col.startswith('pv_pred')]
  pred_sets = [col.split('_')[-1] for col in pred_cols]  # extract suffixes (e.g. '0', '1')
  
  # Normalize each prediction set individually
  for suffix in pred_sets:
      print('Scores model soil group', suffix)

      pv_col = f'pv_pred_{suffix}'
      npv_col = f'npv_pred_{suffix}'
      soil_col = f'soil_pred_{suffix}'

      df['FC'] = df.apply(lambda x: x[pv_col]/(x[pv_col] + x[npv_col] + x[soil_col]), axis=1)
      rmse = np.sqrt(mean_squared_error(df.FC, df.fc))
      mae = mean_absolute_error(df.FC, df.fc)
      r, p_value = pearsonr(df.FC, df.fc)
      r2 = r**2
      print(f'RMSE {rmse}, MAE {mae}, R2 {r2}')
