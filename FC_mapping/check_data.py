import os
import xarray as xr
import rioxarray

data_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/FC')
soil_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/DLR_soilsuite_preds/')
s2_dir = os.path.expanduser('~/mnt/eo-nas1/data/satellite/sentinel2/raw/CH')



fc_maps = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.zarr')]
sample_f = fc_maps[100]
#ds = xr.open_zarr(sample_f).compute()
#ds.isel(time=20)[['PV_norm_global', 'NPV_norm_global', 'Soil_norm_global', 'Soil_norm_soil']].rename({'lat':'y', 'lon':'x'}).rio.write_crs(32632).rio.to_raster('FC_global.tif')
#ds.isel(time=20)[['PV_norm_soil', 'NPV_norm_soil', 'Soil_norm_soil']].rename({'lat':'y', 'lon':'x'}).rio.write_crs(32632).rio.to_raster('FC_soil.tif')
#ds.isel(time=20)[['soil_group']].rename({'lat':'y', 'lon':'x'}).rio.write_crs(32632).rio.to_raster('soil_group_interp.tif')

ds_s2 = xr.open_zarr(os.path.join(s2_dir, sample_f.split('/')[-1])).compute()
ds_s2.isel(time=20)[['s2_B04', 's2_B03', 's2_B02']].rename({'lat':'y', 'lon':'x'}).rio.write_crs(32632).rio.to_raster('s2.tif')

soil_f = "_".join(sample_f.split("_")[1:3]) + ".zarr"
ds_soil = xr.open_zarr(os.path.join(soil_dir, 'SRC_'+soil_f.split('/')[-1])).compute()
#ds_soil[['soil_group']].rio.write_crs(32632).rio.to_raster('soil_group.tif')
