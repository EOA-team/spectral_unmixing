import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats
import joblib
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import sys
sys.path.insert(0, str(Path(os.path.dirname(os.path.realpath("__file__"))).parent))
from models import MODELS
import warnings
warnings.filterwarnings("ignore")



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


def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = (stats.pearsonr(y_true, y_pred).statistic)**2
    return rmse, r2


def prepare_data(data_folder, iteration, soil_group, test_size, random_state):

    # Load data
    X_npv, y_npv = load_data(data_folder, iteration, soil_group, 1)
    X_pv, y_pv = load_data(data_folder, iteration, soil_group, 2)
    X_soil, y_soil = load_data(data_folder, iteration, soil_group, 3)

    X_npv = X_npv / 10000  # Convert to reflectances
    X_pv = X_pv / 10000  # Convert to reflectances
    X_soil = X_soil / 10000  # Convert to reflectances

    # Split into train/test
    X_npv_train, X_npv_test, y_npv_train, y_npv_test = train_test_split(X_npv, y_npv, test_size=0.2, random_state=42)
    X_pv_train, X_pv_test, y_pv_train, y_pv_test = train_test_split(X_pv, y_pv, test_size=0.2, random_state=42)
    X_soil_train, X_soil_test, y_soil_train, y_soil_test = train_test_split(X_soil, y_soil, test_size=0.2, random_state=42)

    return [X_npv_train, X_pv_train, X_soil_train], [X_npv_test, X_pv_test, X_soil_test],\
          [y_npv_train, y_pv_train, y_soil_train], [y_npv_test, y_pv_test, y_soil_test]


def train_model(model_type, model_params, soil_group, data_folder, test_size, random_state):

    predictions = {'PV': [], 'NPV': [], 'SOIL': []}
    scores = {'PV': [], 'NPV': [], 'SOIL': []}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for iteration in range(1, 6):
        # Prepare data

        [X_npv_train, X_pv_train, X_soil_train], [X_npv_test, X_pv_test, X_soil_test],\
        [y_npv_train, y_pv_train, y_soil_train], [y_npv_test, y_pv_test, y_soil_test] =\
        prepare_data(data_folder, iteration, soil_group, test_size, random_state)
        
        for target_name, X_train, X_test, y_train, y_test in zip(
            ['NPV', 'PV', 'SOIL'], 
            [X_npv_train, X_pv_train, X_soil_train], [X_npv_test, X_pv_test, X_soil_test],
            [y_npv_train, y_pv_train, y_soil_train], [y_npv_test, y_pv_test, y_soil_test]
        ):  
            if device == torch.device('cuda') and model_type == 'NN':
                X_train, X_test, y_train, y_test = (
                    torch.FloatTensor(X_train).to(device),
                    torch.FloatTensor(X_test).to(device),
                    torch.FloatTensor(y_train).view(-1, 1).to(device),
                    torch.FloatTensor(y_test).view(-1, 1).to(device),
                ) 

            model = MODELS[model_type](**model_params)
            if model_type == 'NN':
                model.fit(X_train, y_train, X_test, y_test)
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # Save model
            model_path = f"../models/{model_type.lower()}_{target_name.lower()}_soilgroup{soil_group}_iter{iteration}.pkl"
            #joblib.dump(model, model_path)

            # Save prediction and scores
            if device == torch.device('cuda') and model_type == 'NN':
                y_test = y_test.cpu().numpy().flatten()
            
            predictions[target_name].append(y_pred)
            rmse, r2 = evaluate_model(y_test, y_pred)
            scores[target_name].append({'iteration': iteration, 'RMSE': rmse, 'R2': r2})

    
    return predictions, scores



# Input model data
model_type = 'NN' # 'RF, 'NN' 'SVR' 
#model_params = {'C': 1, 'epsilon': 0.01, 'gamma': 1}
#model_params = {'n_estimators': 100, 'random_state': 42}
model_params = {'input_dim': 10, 'num_layers': 3, 'hidden_dim': 64, 'batch_size': 32, 'epochs': 100, 'lr': 0.001, 'optimizer': 'Adam', 'scheduler': 'ReduceLROnPlateau', 'scheduler_params': {'factor': 0.1, 'patience': 5, 'verbose': True}}


# Input paths
data_folder = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/spectral_mixing/synthetic_samples_composition')
results_folder = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/spectral_unmixing/results')

# Other setup params
test_size=0.2
random_state=42

for soil_group in range(6):
    print('Training for soil group', soil_group)
    # TRAIN 
    predictions, scores = train_model(model_type=model_type, 
                                      model_params=model_params,
                                      soil_group=soil_group,
                                      data_folder=data_folder,
                                      test_size=test_size,
                                      random_state=random_state)

    # AVG SCORES
    rows = []
    for target, entries in scores.items():  # target = 'PV', 'NPV', 'SOIL'
        for entry in entries:  # each entry = {'iteration': ..., 'RMSE': ..., 'R2': ...}
            row = {
                'model': target,
                'iteration': entry['iteration'],
                'RMSE': entry['RMSE'],
                'R2': entry['R2']
            }
            rows.append(row)
    scores_df = pd.DataFrame(rows)
    print(scores_df.groupby(['model']).mean()[['RMSE', 'R2']])

    break


