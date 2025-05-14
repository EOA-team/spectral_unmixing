import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(os.path.dirname(os.path.realpath("__file__"))).parent))
from models import MODELS
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

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
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, r2, mae


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


def train_model_specific(model_type, model_params, soil_group, class_type, data_folder, test_size, random_state):

    class_name = ['NPV', 'PV', 'SOIL'][class_type-1]
    predictions = []
    scores = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for iteration in range(1, 6):
        # Prepare data

        [X_npv_train, X_pv_train, X_soil_train], [X_npv_test, X_pv_test, X_soil_test],\
        [y_npv_train, y_pv_train, y_soil_train], [y_npv_test, y_pv_test, y_soil_test] =\
        prepare_data(data_folder, iteration, soil_group, test_size, random_state)
        
        X_train = [X_npv_train, X_pv_train, X_soil_train][class_type-1]
        X_test = [X_npv_test, X_pv_test, X_soil_test][class_type-1]
        y_train = [y_npv_train, y_pv_train, y_soil_train][class_type-1]
        y_test = [y_npv_test, y_pv_test, y_soil_test][class_type-1]

          
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
        model_path = f"../models/{model_type}_CLASS{class_type}_SOIL{soil_group}_ITER{iteration}.pkl"
        joblib.dump(model, model_path)

        # Save prediction and scores
        if device == torch.device('cuda') and model_type == 'NN':
            y_test = y_test.cpu().numpy().flatten()
        
        predictions.append(y_pred)
        rmse, r2, mae = evaluate_model(y_test, y_pred)
        scores.append({'iteration': iteration, 'RMSE': rmse, 'MAE': mae, 'R2': r2})

    
    return predictions, scores, y_test


# Input paths
results_folder = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/spectral_unmixing/results')
data_folder = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/spectral_mixing/synthetic_samples_composition_10k')

# Other setup params
test_size=0.2
random_state=42
set_seed(random_state)

# Input model hyperparams
models = ['NN']  #'SVR', 'RF', 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare to save results
score_columns = ['class_type', 'soil_group', 'iteration', 'test_soil', 'rmse', 'r2', 'mae']
prediction_columns = ['class_type', 'soil_group', 'iteration', 'test_soil', 'y_test', 'y_pred']


for model_type in models:
    score_list = []
    prediction_list = []

    for class_nbr, class_type in enumerate(['NPV', 'PV', 'SOIL']):  
        for soil_group in range(0,6): 
            print(f'Testing {model_type},  soil group {soil_group}, target class {class_type}')

            for iteration in range(1,6):
                # Load model
                model_path = f"../models/{model_type}_CLASS{class_nbr+1}_SOIL{soil_group}_ITER{iteration}.pkl"
                model = joblib.load(model_path)
                print(model_path)

                # Test on all soil groups for that class (cross soil group testing)
                # TO DO: loop over soil groups
                for soil_test in range(0,6):
                    _, [X_npv_test, X_pv_test, X_soil_test],\
                    _, [y_npv_test, y_pv_test, y_soil_test] =\
                    prepare_data(data_folder, iteration, soil_test, test_size, random_state)

                    print('Testing on soil group', soil_test)
                    X_test = [X_npv_test, X_pv_test, X_soil_test][class_nbr]
                    y_test = [y_npv_test, y_pv_test, y_soil_test][class_nbr]

                    if device == torch.device('cuda') and model_type == 'NN':
                        X_test, y_test = (
                            torch.FloatTensor(X_test).to(device),
                            torch.FloatTensor(y_test).view(-1, 1).to(device)
                        ) 
                    y_pred = model.predict(X_test)

                    if device == torch.device('cuda') and model_type == 'NN':
                        y_test = y_test.cpu().numpy().flatten()
                    
                    # Compute metrics
                    rmse, r2, mae = evaluate_model(y_test, y_pred)
                    # Save scores
                    score_list.append([class_type, soil_group, iteration, soil_test, rmse, r2, mae])

                    # Save predictions
                    prediction_list.append([class_type, soil_group, iteration, soil_test, y_test, y_pred])


    
    # Save scores and predictions for that model
    scores_df = pd.DataFrame(score_list, columns=score_columns)
    scores_df.to_pickle(f"../results/{model_type}_full_test_scores.pkl") #, index=False)

    predictions_df = pd.DataFrame(prediction_list, columns=prediction_columns)
    predictions_df.to_pickle(f"../results/{model_type}_full_test_predictions.pkl") #, index=False)



