import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from pathlib import Path
import os
import sys
sys.path.insert(0, str(Path(os.path.dirname(os.path.realpath("__file__"))).parent))
from models import MODELS
import pandas as pd
from scipy import stats
import torch


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


def suggest_params(trial, model_type, input_dim):
    if model_type == "NN":
        optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD", "AdamW"])
        params = {
            'input_dim': input_dim,
            'num_layers': trial.suggest_int("num_layers", 1, 5),
            'hidden_dim': trial.suggest_int("hidden_dim", 32, 128),
            'batch_size': trial.suggest_categorical("batch_size", [16, 32, 64]),
            'epochs': 100,
            'lr': trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            'optimizer': optimizer,
            'scheduler': trial.suggest_categorical("scheduler", [None, "CosineAnnealingLR", "ReduceLROnPlateau"])
        }

        if optimizer == "SGD":
            params['momentum'] = trial.suggest_float("momentum", 0.5, 0.99)
        
        return params

    elif model_type == "SVR":
        return {
            'C': trial.suggest_float("C", 0.1, 10.0, log=True),
            'epsilon': trial.suggest_float("epsilon", 0.001, 0.1, log=True),
            'gamma': trial.suggest_categorical("gamma", ['scale', 'auto'])
        }

    elif model_type == "RF":
        return {
            'n_estimators': trial.suggest_int("n_estimators", 50, 200),
            'max_depth': trial.suggest_int("max_depth", 3, 20),
            'min_samples_split': trial.suggest_int("min_samples_split", 2, 10),
            'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 5)
        }

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def objective(trial, model_type, target_name, soil_group, data_folder, test_size, random_state, results_file):
    all_scores = []
    all_r2 = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for iteration in range(1, 6):
        # Prepare data
        [X_npv_train, X_pv_train, X_soil_train], [X_npv_test, X_pv_test, X_soil_test],\
        [y_npv_train, y_pv_train, y_soil_train], [y_npv_test, y_pv_test, y_soil_test] =\
        prepare_data(data_folder, iteration, soil_group, test_size, random_state)

        X_train, X_test, y_train, y_test = {
            'NPV': (X_npv_train, X_npv_test, y_npv_train, y_npv_test),
            'PV': (X_pv_train, X_pv_test, y_pv_train, y_pv_test),
            'SOIL': (X_soil_train, X_soil_test, y_soil_train, y_soil_test)
        }[target_name]

        if device == torch.device('cuda') and model_type == 'NN':
                X_train, X_test, y_train, y_test = (
                    torch.FloatTensor(X_train).to(device),
                    torch.FloatTensor(X_test).to(device),
                    torch.FloatTensor(y_train).view(-1, 1).to(device),
                    torch.FloatTensor(y_test).view(-1, 1).to(device),
                ) 

        input_dim = X_train.shape[1]
        params = suggest_params(trial, model_type, input_dim)
        model = MODELS[model_type](**params)

        if model_type == 'NN':
            model.fit(X_train, y_train, X_test, y_test)
        else:
            model.fit(X_train, y_train)

        if device == torch.device('cuda') and model_type == 'NN':
            y_test = y_test.cpu().numpy().flatten()
            
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        all_scores.append(rmse)
        r2 = (stats.pearsonr(y_test, y_pred).statistic)**2
        all_r2.append(r2)


    # Save hyperparams and results
    mean_rmse = np.mean(all_scores)
    mean_r2 = np.mean(all_r2)
    result_row = params.copy()
    result_row['RMSE'] = mean_rmse
    result_row['R2'] = mean_r2
    result_row['Trial'] = trial.number
    df_row = pd.DataFrame([result_row])

    if os.path.exists(results_file):
        existing_df = pd.read_excel(results_file)
        updated_df = pd.concat([existing_df, df_row], ignore_index=True)
    else:
        updated_df = df_row
    updated_df.to_excel(results_file, index=False)

    return mean_rmse



def tune_model(model_type, target_name, soil_group, data_folder, test_size, random_state, n_trials, result_file):
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, model_type, target_name, soil_group, data_folder, test_size, random_state, result_file), n_trials=n_trials)

    return study.best_params, study.best_value



# TO DO: loop over soil groups

data_folder = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/spectral_mixing/synthetic_samples_composition')
results_folder = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/spectral_unmixing/results')
n_trials = 1

for soil_group in range(6):
    print('Training for soil group', soil_group)

    model_type = "SVR"  # or "SVR", "RF"

    result_file = f"{results_folder}/{model_type}_tuning_PV_soilgroup{soil_group}.xlsx"
    best_params_pv, score_pv = tune_model(model_type, "PV", soil_group, data_folder, test_size=0.2, random_state=42, n_trials=n_trials, result_file=result_file)
    print("Best PV Params:", best_params_pv)

    result_file = f"{results_folder}/{model_type}_tuning_NPV_soilgroup{soil_group}.xlsx"
    best_params_npv, score_npv = tune_model(model_type, "NPV", soil_group, data_folder, test_size=0.2, random_state=42, n_trials=n_trials, result_file=result_file)
    print("Best NPV Params:", best_params_npv)

    result_file = f"{results_folder}/{model_type}_tuning_SOIL_soilgroup{soil_group}.xlsx"
    best_params_soil, score_soil = tune_model(model_type, "SOIL", soil_group, data_folder, test_size=0.2, random_state=42, n_trials=n_trials, result_file=result_file)
    print("Best SOIL Params:", best_params_soil)

    break
