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
from pathlib import Path
import sys
sys.path.insert(0, str(Path(os.path.dirname(os.path.realpath("__file__"))).parent))
from models import MODELS
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


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
data_folder = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/spectral_mixing/synthetic_samples_composition')
results_folder = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/010_CropCovEO/Erosion/spectral_unmixing/results')

# Other setup params
test_size=0.2
random_state=42

# Input model hyperparams
hyperparams_file = 'tuned_hyperparams.xlsx'
hyperparams_xl = pd.ExcelFile(hyperparams_file)

for model_type in hyperparams_xl.sheet_names:
    df = hyperparams_xl.parse(model_type)
    print(f"Processing params for model: {model_type}")
    
    fig, axs = plt.subplots(len(df['Soil group'].unique()), 3, figsize=(5*len(df['Class'].unique()), 5*len(df['Soil group'].unique())))
    
    # Loop over soil groups
    for soil in df['Soil group'].unique():
        for class_type in df['Class'].unique():
            
            df_model = df[(df['Soil group']==soil) & (df['Class']==class_type)]
            model_params = {
                k: df_model[k].values[0] if not pd.isna(df_model[k].values[0]) else None
                for k in df_model.drop(['Soil group', 'Class', 'RMSE'], axis=1).columns
            }
            
            print(f'Training for soil group {soil}, target class {class_type}')
            # TRAIN 
            predictions, scores, y_test = train_model_specific(model_type=model_type, 
                                            model_params=model_params,
                                            soil_group=soil,
                                            class_type=class_type,
                                            data_folder=data_folder,
                                            test_size=test_size,
                                            random_state=random_state)

         
            # AVG SCORES
            rows = []
            for i in range(len(scores)):  
                row = {
                    'iteration': scores[i]['iteration'],
                    'RMSE': scores[i]['RMSE'],
                    'MAE': scores[i]['MAE'],
                    'R2': scores[i]['R2']
                }
                rows.append(row)
            scores_df = pd.DataFrame(rows)
            print(scores_df.mean()[['RMSE', 'MAE', 'R2']])

            scores_df.to_csv(f'../results/{model_type}_SOIL{soil}_CLASS{class_type}.csv', index=False)

            # TO DO: generate plots.
            mean_preds = np.mean(predictions, axis=0)  # mean of all iterations
            target_name = ['NPV', 'PV', 'SOIL'][class_type-1]
            color = ['darkgoldenrod', 'green', 'saddlebrown'][class_type-1]
            axs[soil, class_type-1].scatter(y_test, mean_preds, c=color, s=15, alpha=0.5)
            axs[soil, class_type-1].set_title(f'Soil group {soil}, {target_name}')
            axs[soil, class_type-1].set_xlabel('Reference FC')
            axs[soil, class_type-1].set_ylabel('Predicted FC')
            axs[soil, class_type-1].set_xlim(-0.2,1.2)
            axs[soil, class_type-1].set_ylim(-0.2,1.2)
            axs[soil, class_type-1].plot([-0.2, 1.2], [-0.2, 1.2], color='gray', linestyle='--', label='1:1 Line')
            # Add regression line
            slope, intercept = np.polyfit(y_test, mean_preds, 1)
            x_vals = np.array([-0.2, 1.2])
            regression_line = slope * x_vals + intercept
            axs[soil, class_type-1].plot(x_vals, regression_line, 'k-')
            # Add scores on plot
            mean_rmse = scores_df['RMSE'].mean()
            mean_r2 = scores_df['R2'].mean()
            mean_mae = scores_df['MAE'].mean()
            axs[soil, class_type-1].text(
                0.05, 0.95, f'RMSE: {mean_rmse:.3f}\nMAE: {mean_mae:.3f}\nR²: {mean_r2:.3f}',
                transform=axs[soil, class_type-1].transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
                    
    fig.suptitle('Pred vs Test Labels', y=1.1) 
    plt.tight_layout()
    plt.savefig(f'../results/test_preds_{model_type}.png')
    


