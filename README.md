# Spectral unmixing


## 1. Baresoil composite classification

Different soil groups (in terms of spectral properties) are determine using K-means clustering. The bare soil composite from the [DLR soilsuite](https://geoservice.dlr.de/web/datasets/soilsuite_eur_5y) is used. 

In `baresoil/soilsuite.py` the following steps are performed:
1. Download data
    - The bare soil composite is downloaded for Switzerland
2. Sample points
    - The data is resampled to 10m resolution from 20m
    - 250k points that fall within Swiss borders and where MASK is 1 (presence of baresoil) are sampled
3. Cluster the samples
    - The samples are then clustered using their bands
    - Different numbers of clusters are tested, using silhouette score and elbow method to look at optimal cluster size
4. The final Kmeans model is fitted and the soil groups are plotted
    - We determined 5 groups to be enough to seperate the reflectectances and also discussed with soil scientists
    - Plot of the clustered data: `baresoil/plots/sampled_pts_5_clusters_agri_v2.png`
5. For each group, the 25/50/75th percentiles are determined
    - Plot of the summarised spectra per soil group: `baresoil/plots/soil_endmembers_5_clusters_agri_v2.png`

The soil clusters were then renamed:
```
label_map = {0: 5, 1: 2, 2: 4, 3: 1, 4: 3}
```
and the updated plots can be found in the following plots: `baresoil/plots/sampled_pts_5_clusters_agri_v2_renamed.png`, `baresoil/plots/soil_endmembers_5_clusters_agri_v2_renamed.png`

### Description of soil groups
- Soil clusters 4 and 5 are bright soils, 1 and 2 are darker soils and cluster 3 is a medium brightness soil
- Cluster 1 are the darkest soils, primarily found in the plateau where SOC is high and otherwise found in alpine valleys
- Cluster 2 are medium dark soils found in the plateau, the jura and alpine valleys. They have moderate SOC in the plateau and low-moderate SOC in the jura and alpine valleys. 
- Cluster 3 are medium bright soils with low-moderate SOC and moderate-high silt contents
- Cluster 4 are bright soils with low SOC, mainly in the plateau and in the jura range. They have moderate clay contents
- Cluster 5 represents the brightest soils, mainly in the plateau and characterized by low SOC and low-moderate clay contents


All soils will be classified to these groups. 



## 2. PV and NPV endmembers

Different PV (photosynthetic vegetation) and NPV (non photosynthetic vegetation) are also sampled.

In `pv_npv_members/sample_endmembers.py`
- For each year between 2021 and 2023, we identified the top 15 crops types according to pixel count
- We know the classification of each pixel thanks to crop maps (`pv_npv_members/crop_maps`) and we use the classification determined in `~/mnt/eo-nas1/data/landuse/documentation/LNF_code_classification_20250217.xlsx` (column "categories2024"). The crop type is thus a crop name or crop group.
- 10k locations are sampled per crop type and year (saved in `pv_npv_members/sampled_coords`)
- For each location, the yearly S2 data is extracted
  - After data cleaning (missing data, clouds, snow...), we sample PV and NPV spectra if possible
  - For PV it's when NDVI is max and SWIR ratio is min
  - For NPV it's when NDVI and SWIR ratio are min. Additionally this must be between Jun. 1st and Nov. 15th
- The spectra at these two moments are saved as examples of PV and NPV spectra (`pv_npv_members/pv_spectra_v2.pkl`, `pv_npv_members/npv_spectra_v2.pkl`)
- For each crop type and year, the PV and NPV spectra are summarised according to the 25/50/75th percentiles (`pv_npv_members/summarised_npv_samples_pername.pkl`, `pv_npv_members/summarised_pv_samples_pername.pkl`)

The summarised spectra will serve as endmembers for generating training data for a spectral unmixing model.
These can be visualised in the plots `pv_npv_members/plots/pv_npv_summary_per_cropname_*.png` and `pv_npv_members/plots/pv_npv_samples_locations_per_crop.png`

## 3. Spectral mixing

Following the methodology in [Locher et al. (2025)](https://www.sciencedirect.com/science/article/pii/S0034425724006205?via%3Dihub#s0040):

We created 1000 synthetic mixtures for each cover fraction based on the spectral library. Each synthetic mixture comprised two to three endmembers from the three cover fractions randomly sampled from the library. These were then linearly combined with random fractions assigned to each endmember class, ensuring the fractions always sum up to 1. The resulting synthetic spectrum was then added to the training dataset, with the six spectral bands as input variables and the share of the target cover fraction as the label. Additionally, we included a shade spectrum in the synthetic mixtures to represent direct and structural shade components (Shimabukuro and Smith, 1991). The shade endmember, with a near-zero reflectance of 0.01 across all bands, was treated like any other endmember during the mixing step but was not considered as target fraction.

The synthetic mixtures (pairs of features/labels) are stored in the following way:
```
feat_filename = f'synthetic_samples/SYNTHMIX_SOIL-00{soil_group_nbr}_SHADOW-TRUE_FEATURES_CLASS-00{feature_class}_ITERATION-00{i}.txt'
resp_filename = f'synthetic_samples/SYNTHMIX_SOIL-00{soil_group_nbr}_SHADOW-TRUE_RESPONSE_CLASS-00{feature_class}_ITERATION-00{i}.txt'
```

where 0 means the global model, and the soil groups determined from the K-means clustering  are named 1 to n. There are 3 feature classes (1=NPV, 2=PV, 3=Soil). Each dataset contains 1000 mixtures, and these datasets are iterated 5 times.
