import pandas as pd
import os
import numpy as np

# Open files containing endmembers

pv_file = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/pv_npv_members/summarised_pv_samples_pername.pkl')
npv_file = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/pv_npv_members/summarised_npv_samples_pername.pkl')
soil_file = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/baresoil/summarised_soil_samples_renamed.pkl')

pv_endmembers = pd.read_pickle(pv_file)
npv_endmembers = pd.read_pickle(npv_file)
soil_endmembers = pd.read_pickle(soil_file)


# Format data
SRC_cols = ['SRC_B2','SRC_B3','SRC_B4','SRC_B5','SRC_B6','SRC_B7','SRC_B8','SRC_B8A','SRC_B11','SRC_B12']
S2_cols = ['s2_B02','s2_B03','s2_B04','s2_B05','s2_B06','s2_B07','s2_B08','s2_B8A','s2_B11','s2_B12']
clean_names = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
input_bands = clean_names

pv_endmembers = pv_endmembers[S2_cols].rename(columns=dict(zip(S2_cols, clean_names)))[input_bands]
npv_endmembers = npv_endmembers[S2_cols].rename(columns=dict(zip(S2_cols, clean_names)))[input_bands]
soil_endmembers = soil_endmembers[SRC_cols].rename(columns=dict(zip(SRC_cols, clean_names)))[input_bands]
shadow_endmember =  np.full(len(input_bands), 0.01)

spectral_libraries = [npv_endmembers, pv_endmembers, soil_endmembers]


# Generate synthetic mixtures -- GLOBAL MODEL 

iterations = 5 # Number of times to create a dataset
classes = 3 # (1=NPV, 2=PV, 3=Soil)
num_mixtures = 1000  # Number of synthetic mixtures per class

os.makedirs("synthetic_samples_composition", exist_ok=True)  # Ensure output folder exists

for feature in range(classes):
  print('Creating data for cover class', feature+1)
  for i in range(iterations):
    print('Iteration', i+1)
    features_list = []
    response_list = []

    for _ in range(num_mixtures):
      num_endmembers = np.random.choice([2, 3])  # Randomly choose 2 or 3 endmembers
      selected_classes = np.random.choice([0, 1, 2], size=num_endmembers, replace=False)

      endmembers = []
      for cl in selected_classes:
          sample = spectral_libraries[cl].sample(n=1).values.flatten()  # Sample one spectrum
          endmembers.append(sample)

      # Randomly add shade with 50% chance
      if np.random.rand() < 0.5:
          endmembers.append(shadow_endmember)
          selected_classes = np.append(selected_classes, 3) # Add shade class (index 3)

      # Mix spectra
      endmembers = np.array(endmembers)  # Convert to array
      fractions = np.random.rand(len(endmembers)) # Assign random weights (random uniform 0 to 1)
      fractions /= fractions.sum() # Normalize to sum to 1

      synthetic_spectrum = np.dot(fractions, endmembers)  # Compute weighted sum
      features_list.append(synthetic_spectrum)

      response = np.zeros(4)
      for cl, frac in zip(selected_classes, fractions):  
          response[cl] += frac
      response_list.append(response.tolist())  # Full compositional label


    # Add all pure endmembers 
    for j, sp in enumerate(spectral_libraries):
        label = [0, 0, 0, 0]
        label[j] = 1  # Pure spectrum = 100% of class i
        features_list.extend(sp[input_bands].values.tolist())
        response_list.extend([label] * len(sp))

    # Convert to DataFrame and save
    features_df = pd.DataFrame(features_list, columns=input_bands)
    response_df = pd.DataFrame(response_list, columns=['NPV', 'PV', 'Soil', 'Shade'])


    feat_filename = f'synthetic_samples_composition/SYNTHMIX_SOIL-000_SHADOW-TRUE_FEATURES_CLASS-00{feature+1}_ITERATION-00{i+1}.txt'
    resp_filename = f'synthetic_samples_composition/SYNTHMIX_SOIL-000_SHADOW-TRUE_RESPONSE_CLASS-00{feature+1}_ITERATION-00{i+1}.txt'

    features_df.to_csv(feat_filename, index=False, sep=" ")
    response_df.to_csv(resp_filename, index=False, sep=" ")



# Format data

pv_endmembers = pd.read_pickle(pv_file)
npv_endmembers = pd.read_pickle(npv_file)
soil_endmembers = pd.read_pickle(soil_file)

SRC_cols = ['SRC_B2','SRC_B3','SRC_B4','SRC_B5','SRC_B6','SRC_B7','SRC_B8','SRC_B8A','SRC_B11','SRC_B12']
S2_cols = ['s2_B02','s2_B03','s2_B04','s2_B05','s2_B06','s2_B07','s2_B08','s2_B8A','s2_B11','s2_B12']
clean_names = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
input_bands = clean_names

pv_endmembers = pv_endmembers[S2_cols].rename(columns=dict(zip(S2_cols, clean_names)))[input_bands]
npv_endmembers = npv_endmembers[S2_cols].rename(columns=dict(zip(S2_cols, clean_names)))[input_bands]
#soil_endmembers = soil_endmembers[SRC_cols+['cluster']].rename(columns=dict(zip(SRC_cols, clean_names+['cluster'])))
shadow_endmember =  np.full(len(input_bands), 0.01)

spectral_libraries = [npv_endmembers, pv_endmembers, soil_endmembers]


# Generate synthetic mixtures -- SOIL SPECIFIC MODELS

iterations = 5 # Number of times to create a dataset
classes = 3 # (1=NPV, 2=PV, 3=Soil)
num_mixtures = 1000  # Number of synthetic mixtures per class
soil_groups = 5 # Will be named 1-5 since 0 is global

os.makedirs("synthetic_samples_composition", exist_ok=True)  # Ensure output folder exists

for soil in range(1, soil_groups+1):
  print('Considering only soil group', soil+1)

  for feature in range(classes):
    print('Creating data for cover class', feature+1)

    for i in range(iterations):
      print('Iteration', i+1)
      features_list = []
      response_list = []

      for _ in range(num_mixtures):
        num_endmembers = np.random.choice([2, 3])  # Randomly choose 2 or 3 endmembers
        selected_classes = np.random.choice([0, 1, 2], size=num_endmembers, replace=False)

        endmembers = []
        for cl in selected_classes:
            # If soil is a selected_class, sample only from soil group
            if cl == 2:
              df = spectral_libraries[cl]
              df = df[df.cluster==soil][SRC_cols].rename(columns=dict(zip(SRC_cols, clean_names)))[input_bands]
              sample = df.sample(n=1).values.flatten()
            else:
              sample = spectral_libraries[cl].sample(n=1).values.flatten()  # Sample one spectrum
            endmembers.append(sample)

        # Randomly add shade with 50% chance
        if np.random.rand() < 0.5:
            endmembers.append(shadow_endmember)
            selected_classes = np.append(selected_classes, 3) # Add shade class (index 3)

        # Mix spectra
        endmembers = np.array(endmembers)  # Convert to array
        fractions = np.random.rand(len(endmembers)) # Assign random weights (random uniform 0 to 1)
        fractions /= fractions.sum() # Normalize to sum to 1

        synthetic_spectrum = np.dot(fractions, endmembers)  # Compute weighted sum
        features_list.append(synthetic_spectrum)

        response = np.zeros(4)
        for cl, frac in zip(selected_classes, fractions):  
            response[cl] += frac
        response_list.append(response.tolist())  # Full compositional label

      
      # Add all pure endmembers --> TO DO: add specific soil cluster
      for j, sp in enumerate(spectral_libraries):
          label = [0, 0, 0, 0]
          label[j] = 1  # Pure spectrum = 100% of class i
          if j == 2:
            # Select only specific soil cluster
            df = sp[sp.cluster == soil][SRC_cols].rename(columns=dict(zip(SRC_cols, clean_names)))[input_bands]
            features_list.extend(df.values.tolist())
            response_list.extend([label] * len(df))
          else:
            features_list.extend(sp[input_bands].values.tolist())
            response_list.extend([label] * len(sp))

      # Convert to DataFrame and save
      features_df = pd.DataFrame(features_list, columns=input_bands)
      response_df = pd.DataFrame(response_list, columns=['NPV', 'PV', 'Soil', 'Shade'])

      feat_filename = f'synthetic_samples_composition/SYNTHMIX_SOIL-00{soil}_SHADOW-TRUE_FEATURES_CLASS-00{feature+1}_ITERATION-00{i+1}.txt'
      resp_filename = f'synthetic_samples_composition/SYNTHMIX_SOIL-00{soil}_SHADOW-TRUE_RESPONSE_CLASS-00{feature+1}_ITERATION-00{i+1}.txt'

      features_df.to_csv(feat_filename, index=False, sep=" ")
      response_df.to_csv(resp_filename, index=False, sep=" ")
