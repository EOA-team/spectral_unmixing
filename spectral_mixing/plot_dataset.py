""" 
Plot the synthetic datasets created
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


def load_data(data_folder, iteration, soil_group, class_type):
  """ 
  Function to load data for each iteration

  :param interation: int from 0 to 5
  :param class_type: int 1 (NPV), 2 (PV), 3 (soil)
  """

  features_file = os.path.join(data_folder, f'SYNTHMIX_SOIL-00{soil_group}_SHADOW-TRUE_FEATURES_CLASS-00{class_type}_ITERATION-00{iteration}.txt')
  response_file = os.path.join(data_folder, f'SYNTHMIX_SOIL-00{soil_group}_SHADOW-TRUE_RESPONSE_CLASS-00{class_type}_ITERATION-00{iteration}.txt')
  
  features = pd.read_csv(features_file, sep=" ")
  response = pd.read_csv(response_file, sep=" ").iloc[:, class_type-1] # select only the fraction of interest
  
  return features.values, response.values.ravel()


def load_data_composition(data_folder, iteration, soil_group, class_type):
  """ 
  Function to load data for each iteration

  :param interation: int from 0 to 5
  :param class_type: int 1 (NPV), 2 (PV), 3 (soil)
  """

  features_file = os.path.join(data_folder, f'SYNTHMIX_SOIL-00{soil_group}_SHADOW-TRUE_FEATURES_CLASS-00{class_type}_ITERATION-00{iteration}.txt')
  response_file = os.path.join(data_folder, f'SYNTHMIX_SOIL-00{soil_group}_SHADOW-TRUE_RESPONSE_CLASS-00{class_type}_ITERATION-00{iteration}.txt')
  
  features = pd.read_csv(features_file, sep=" ")
  response = pd.read_csv(response_file, sep=" ") # select only the fraction of interest
  
  return features.values, response.values




data_folder = 'synthetic_samples_composition'
n_soils = 5
n_iter = 5
plot_folder = 'plots'
os.makedirs(plot_folder, exist_ok=True)  # Ensure output folder exists

# Plot histograms of fractions for each group (incl global)

for soil_group in range(1, n_soils+1): # soil group 0 is not specific, 1-5 are specific clusters
  
  features_pv, response_pv = [], []
  features_npv, response_npv = [], []
  features_soil, response_soil = [], []
  
  for it in range(n_iter):
    f, r = load_data(data_folder, it+1, soil_group, 1)  # NPV
    features_npv.append(f)
    response_npv.append(r)

    f, r = load_data(data_folder, it+1, soil_group, 2)  # PV
    features_pv.append(f)
    response_pv.append(r)

    f, r = load_data(data_folder, it+1, soil_group, 3)  # Soil
    features_soil.append(f)
    response_soil.append(r)

  features_pv = np.concatenate(features_pv)
  features_npv = np.concatenate(features_npv)
  features_soil = np.concatenate(features_soil)

  response_pv = np.concatenate(response_pv)
  response_npv = np.concatenate(response_npv)
  response_soil = np.concatenate(response_soil)

  f, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

  if soil_group == 0:
      f.suptitle(f'Global dataset - Density plots of fractions')
  else:
      f.suptitle(f'Soil group {soil_group} - Density plots of fractions')


  sns.histplot(response_npv, ax=axs[0], color='red', bins=50) # fill=True, alpha=0.5)
  axs[0].set_title('NPV')
  axs[0].set_xlabel('Fraction')
  axs[0].set_xlim(0,1)

  sns.histplot(response_pv, ax=axs[1], color='green', bins=50) # fill=True, alpha=0.5)
  axs[1].set_title('PV')
  axs[1].set_xlabel('Fraction')
  axs[1].set_xlim(0,1)

  sns.histplot(response_soil, ax=axs[2], color='blue', bins=50) # fill=True, alpha=0.5)
  axs[2].set_title('Soil')
  axs[2].set_xlabel('Fraction')
  axs[2].set_xlim(0,1)

  plt.tight_layout()
  plt.subplots_adjust(top=0.85)  # adjust to make room for suptitle
  if soil_group == 0:
    plt.savefig(f'{plot_folder}/fraction_density_global.png')
  else:
    plt.savefig(f'{plot_folder}/fraction_density_soilgroup{soil_group}.png')



# Plot spectra for different fraction levels

for soil_group in range(1, n_soils+1): # soil group 0 is not specific, 1-5 are specific clusters
  
  features_pv, response_pv = [], []
  features_npv, response_npv = [], []
  features_soil, response_soil = [], []
  
  for it in range(n_iter):
    f, r = load_data(data_folder, it+1, soil_group, 1)  # NPV
    features_npv.append(f)
    response_npv.append(r)

    f, r = load_data(data_folder, it+1, soil_group, 2)  # PV
    features_pv.append(f)
    response_pv.append(r)

    f, r = load_data(data_folder, it+1, soil_group, 3)  # Soil
    features_soil.append(f)
    response_soil.append(r)

  features_pv = np.concatenate(features_pv)
  features_npv = np.concatenate(features_npv)
  features_soil = np.concatenate(features_soil)

  response_pv = np.concatenate(response_pv)
  response_npv = np.concatenate(response_npv)
  response_soil = np.concatenate(response_soil)

  # Plot mean, 25th percentile and 75th percentile spectra for each 0.1 interval of fraction in different subplots 
  fraction_levels = np.linspace(0, 1, 11)[:-1]
  x = [492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 1613.7, 2202.4]
  
  f, axs = plt.subplots(10, 3, figsize=(20, 4*len(fraction_levels)), sharey=True)

  for i, frac in enumerate(fraction_levels):
    frac = round(frac, 1)
    
    # NPV
    mask = (response_npv >= frac) & (response_npv < frac + 0.1)
    if np.sum(mask) > 0:
      mean_spectrum = np.mean(features_npv[mask], axis=0)
      p25 = np.percentile(features_npv[mask], 25, axis=0)
      p75 = np.percentile(features_npv[mask], 75, axis=0)
      axs[i, 0].plot(x, mean_spectrum, color='blue', label='Mean spectra')
      axs[i, 0].plot(x, p25, label='25th percentile', color='steelblue', linestyle='dashed')
      axs[i, 0].plot(x, p75, label='75th percentile', color='steelblue',linestyle='dotted')
      axs[i, 0].set_title(f'NPV fraction {frac:.1f} - {frac+0.1:.1f}')
      axs[i, 0].set_xlabel('wavelength')
      axs[i, 0].set_ylabel('Reflectance')
      axs[i, 0].set_ylim(0,10000)
      axs[i, 0].legend()

    # PV
    mask = (response_pv >= frac) & (response_pv < frac + 0.1)
    if np.sum(mask) > 0:
      mean_spectrum = np.mean(features_pv[mask], axis=0)
      p25 = np.percentile(features_pv[mask], 25, axis=0)
      p75 = np.percentile(features_pv[mask], 75, axis=0)
      axs[i, 1].plot(x, mean_spectrum, color='blue', label='Mean spectra')
      axs[i, 1].plot(x, p25, label='25th percentile', color='steelblue', linestyle='dashed')
      axs[i, 1].plot(x, p75, label='75th percentile', color='steelblue', linestyle='dotted')
      axs[i, 1].set_title(f'PV fraction {frac:.1f} - {frac+0.1:.1f}')
      axs[i, 1].set_ylabel('Reflectance')
      axs[i, 1].set_xlabel('wavelength')
      axs[i, 1].set_ylim(0,10000)
      axs[i, 1].legend()

    # Soil
    mask = (response_soil >= frac) & (response_soil < frac + 0.1)
    if np.sum(mask) > 0:
      mean_spectrum = np.mean(features_soil[mask], axis=0)
      p25 = np.percentile(features_soil[mask], 25, axis=0)
      p75 = np.percentile(features_soil[mask], 75, axis=0)
      axs[i, 2].plot(x, mean_spectrum, color='blue', label='Mean spectra')
      axs[i, 2].plot(x, p25, label='25th percentile', color='steelblue', linestyle='dashed')
      axs[i, 2].plot(x, p75, label='75th percentile', color='steelblue', linestyle='dotted')
      axs[i, 2].set_title(f'Soil fraction {frac:.1f} - {frac+0.1:.1f}')
      axs[i, 2].set_ylabel('Reflectance')
      axs[i, 2].set_xlabel('wavelength')
      axs[i, 2].set_ylim(0,10000)
      axs[i, 2].legend()

  plt.tight_layout()

  if soil_group == 0:
    f.suptitle(f'Global dataset spectra')
    plt.savefig(f'{plot_folder}/spectra_global.png')
  else:
    f.suptitle(f'Soil group {soil_group} spectra')
    plt.savefig(f'{plot_folder}/spectra_soilgroup{soil_group}.png')


# Plot RGB examples of pure endmembers

soil_em = '../baresoil/summarised_soil_samples_renamed.pkl'
pv_em = '../pv_npv_members/summarised_pv_samples_pername.pkl'
npv_em = '../pv_npv_members/summarised_npv_samples_pername.pkl'

soil_em = pd.read_pickle(soil_em)
pv_em = pd.read_pickle(pv_em)
npv_em = pd.read_pickle(npv_em)

soil_cols = ['SRC_B2', 'SRC_B3', 'SRC_B4', 'SRC_B5', 'SRC_B6', 'SRC_B7', 'SRC_B8', 'SRC_B8A', 'SRC_B11', 'SRC_B12']
pv_npv_cols = ['s2_B02', 's2_B03', 's2_B04', 's2_B05', 's2_B06', 's2_B07', 's2_B08', 's2_B8A', 's2_B11', 's2_B12']


for soil_group in range(1, n_soils+1): # 1-5 are specific clusters

  n_samples = 3

  # Soil endmembers
  soil_pixs = soil_em[soil_em.cluster == soil_group][soil_cols].values

  # PV endmembers
  pv_pixs = pv_em.sample(n_samples)[pv_npv_cols].values

  # NPV endmembers
  npv_pixs = npv_em.sample(n_samples)[pv_npv_cols].values

  datasets = [(soil_pixs, 'Soil'),
             (pv_pixs, 'PV'),
             (npv_pixs, 'NPV')]
             

  # Plot RGB pixels with fraction writen above
  f, axs = plt.subplots(len(datasets), n_samples, figsize=(len(datasets)*5, n_samples*5))
  brightness = 3
  for i, dataset in enumerate(datasets):
    spectra, label = dataset
    for j in range(n_samples):
      rgb_pixel = np.array([spectra[j][2], spectra[j][1], spectra[j][0]]).reshape((1, 1, 3)) / 10000
      axs[i, j].imshow(rgb_pixel*brightness)
      axs[i, j].set_title(f'Pure {label}', size=18)
      axs[i, j].axis('off')

  plt.suptitle(f'RGB examples - Soil group {soil_group}', size=20)
  plt.tight_layout()
  plt.savefig(f'plots/rgb_pure_endmembers_soilgroup{soil_group+1}.png')

# # Plot RGB examples of different mixtures
 
for soil_group in range(1, n_soils+1): # soil group 0 is not specific, 1-5 are specific clusters
  
  features, response = [], []
  
  for it in range(n_iter):
    f, r = load_data_composition(data_folder, it+1, soil_group, 1)  # NPV
    features.append(f)
    response.append(r)

  features = np.concatenate(features)
  response = np.concatenate(response)

  n_samples = 3

  # NPV dominating
  npv_spectra = features[response[:,0] > 0.7]
  npv_frac = response[response[:,0] > 0.7]
  idx = np.random.choice(npv_spectra.shape[0], size=n_samples, replace=False)
  npv_spectra = npv_spectra[idx]
  npv_frac = npv_frac[idx]

  # PV dominating
  pv_spectra = features[response[:,1] > 0.7]
  pv_frac = response[response[:,1] > 0.7]
  idx = np.random.choice(pv_spectra.shape[0], size=n_samples, replace=False)
  pv_spectra = pv_spectra[idx]
  pv_frac = pv_frac[idx]

  # Soil dominating
  soil_spectra = features[response[:,2] > 0.7]
  soil_frac = response[response[:,2] > 0.7]
  idx = np.random.choice(soil_spectra.shape[0], size=n_samples, replace=False)
  soil_spectra = soil_spectra[idx]
  soil_frac = soil_frac[idx]

  # Random mixutres
  idx = np.random.choice(features.shape[0], size=n_samples, replace=False)
  mix_spectra = features[idx]
  mix_frac = response[idx]

  datasets = [(npv_spectra, npv_frac, 'NPV'),
             (pv_spectra, pv_frac, 'PV'),
             (soil_spectra, soil_frac, 'Soil'),
             (mix_spectra, mix_frac, 'Mixture')]
             

  # Plot RGB pixels with fraction writen above
  f, axs = plt.subplots(len(datasets), n_samples, figsize=(len(datasets)*5, n_samples*5))
  brightness = 3
  for i, dataset in enumerate(datasets):
    spectra, fractions, label = dataset
    for j in range(n_samples):
      rgb_pixel = np.array([spectra[j][2], spectra[j][1], spectra[j][0]]).reshape((1, 1, 3)) / 10000
      axs[i, j].imshow(rgb_pixel*brightness)
      axs[i, j].set_title(f'{label} - {fractions[j][:-1]}', size=18)
      axs[i, j].axis('off')

  plt.suptitle(f'RGB examples - Soil group {soil_group} (frac NPV/PV/Soil)', size=20)
  plt.tight_layout()
  plt.savefig(f'plots/rgb_soilgroup{soil_group}.png')
