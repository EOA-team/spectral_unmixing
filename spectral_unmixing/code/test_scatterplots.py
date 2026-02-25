import numpy as np
import pandas as pd
import os
import joblib
import torch
from scipy.stats import t
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
from pathlib import Path
import sys
sys.path.insert(0, str(Path(os.path.dirname(os.path.realpath("__file__"))).parent))
from models import MODELS
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = (stats.pearsonr(y_true, y_pred).statistic)**2
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, r2, mae

def load_data(data_folder, iteration, soil_group, class_type):
    """ 
    Function to load data for each iteration
    """
    features_file = os.path.join(data_folder, f'SYNTHMIX_SOIL-00{soil_group}_SHADOW-TRUE_FEATURES_CLASS-00{class_type}_ITERATION-00{iteration}.txt')
    response_file = os.path.join(data_folder, f'SYNTHMIX_SOIL-00{soil_group}_SHADOW-TRUE_RESPONSE_CLASS-00{class_type}_ITERATION-00{iteration}.txt')
    
    features = pd.read_csv(features_file, sep=" ")
    response = pd.read_csv(response_file, sep=" ").iloc[:, class_type-1]  # select only the fraction of interest
    
    return features.values, response.values.ravel()

def prepare_test_data(data_folder, iteration, soil_group):
    # Load data
    X_npv, y_npv = load_data(data_folder, iteration, soil_group, 1)
    X_pv, y_pv = load_data(data_folder, iteration, soil_group, 2)
    X_soil, y_soil = load_data(data_folder, iteration, soil_group, 3)

    X_npv = X_npv / 10000  # Convert to reflectances
    X_pv = X_pv / 10000  # Convert to reflectances
    X_soil = X_soil / 10000  # Convert to reflectances

    return [X_npv, X_pv, X_soil], [y_npv, y_pv, y_soil]

# Input paths
data_folder = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/028_Erosion/Erosion/spectral_mixing/synthetic_samples_composition_10k')
results_folder = os.path.expanduser('~/mnt/eo-nas1/eoa-share/projects/028_Erosion/Erosion/spectral_unmixing/results')

# Other setup params
test_size = 0.2
random_state = 42
model_type = 'NN'


fig, axs = plt.subplots(6, 3, figsize=(15, 30), sharex=True, sharey=True)  # 6 rows for 5 soil groups + 1 global model
class_order = [2, 1, 3]  # Green vegetation (2), Brown vegetation (1), Soil (3)
for soil_group in range(0, 6):  # Include soil group 0 for the global model
    for class_type in class_order: #range(1, 4):  # Assuming class types are 1 (NPV), 2 (PV), 3 (Soil)
        predictions = []
        labels = []
        scores = []

        for iteration in range(1, 6):  # Assuming 5 iterations
            if soil_group == 0:
                # Combine data from all soil groups for the global model
                X_all, y_all = [], []
                for sg in range(0, 6):  # Soil groups 1 to 5
                    X, y = prepare_test_data(data_folder, iteration, sg)
                    X_all.append(X[class_type-1])
                    y_all.append(y[class_type-1])
                X_test = np.vstack(X_all)  # Combine features
                y_test = np.hstack(y_all)  # Combine labels
            else:
                # Load data for the specific soil group
                X, y = prepare_test_data(data_folder, iteration, soil_group)
                X_test = X[class_type-1]
                y_test = y[class_type-1]

            #print(len(y_test[(y_test>0.8) & (y_test<0.9)]), y_test[(y_test>0.8) & (y_test<0.9)].mean())
           
            # Load saved model
            model_path = f"../models/{model_type}_CLASS{class_type}_SOIL{soil_group}_ITER{iteration}.pkl"
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                continue
            
            model = joblib.load(model_path)

            # Convert X_test to a PyTorch tensor if the model is NN
            if model_type == 'NN':
                X_test = torch.FloatTensor(X_test).to(model.device)

            # Predict
            y_pred = model.predict(X_test)
            predictions.append(y_pred)
            labels.append(y_test)

            # Evaluate
            rmse, r2, mae = evaluate_model(y_test, y_pred)
            scores.append({'iteration': iteration, 'RMSE': rmse, 'MAE': mae, 'R2': r2})

        # Plot
        column_index = class_order.index(class_type)
        target_name = ['Brown vegetation', 'Green vegetation', 'Soil'][class_type-1]
        color = ['mediumslateblue', 'green', 'saddlebrown'][class_type-1]
        
        for iteration_labels, iteration_preds in zip(labels, predictions):
            axs[soil_group, column_index].scatter(
                iteration_labels, iteration_preds, c=color, s=15, alpha=0.5
            )
        axs[soil_group, column_index].set_xlim(-0.1, 1.1)
        axs[soil_group, column_index].set_ylim(-0.1, 1.1)
        axs[soil_group, column_index].plot([-0.1, 1.1], [-0.1, 1.1], color='gray', linestyle='--', label='1:1 Line')

        # Add scores
        all_labels = np.concatenate(labels)
        all_predictions = np.concatenate(predictions)
        rmse_all = np.sqrt(mean_squared_error(all_labels, all_predictions))
        r2_all = (stats.pearsonr(all_labels, all_predictions).statistic) ** 2
        mae_all = mean_squared_error(all_labels, all_predictions)
        axs[soil_group, column_index].text(
            0.05, 0.95, f'RMSE: {rmse_all:.3f}\nMAE: {mae_all:.3f}\nR²: {r2_all:.3f}',
            transform=axs[soil_group, column_index].transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )

        # Add regression line
        slope, intercept = np.polyfit(all_labels, all_predictions, 1)
        x_vals = np.array([0, 1])
        regression_line = slope * x_vals + intercept
        #axs[soil_group, class_type-1].plot(x_vals, regression_line, 'k-',  label='Regression Line')
        axs[soil_group, column_index].plot(x_vals, regression_line, color='black', linestyle='-', label='Regression Line')
        
        
        # Set tick label size
        axs[soil_group, column_index].tick_params(axis='both', labelsize=14)

        # Add soil group label and "Predicted fractional cover" on the left of each row
        if class_type == 1:  # Only add the label once per row
            soil_label = "Global model" if soil_group == 0 else f"Soil group {soil_group}"
            
            # Add the soil group label (larger font size)
            axs[soil_group, class_type-1].annotate(
                soil_label,
                xy=(0, 0),
                xytext=(-axs[soil_group, class_type-1].yaxis.labelpad - 50, 10),
                xycoords=axs[soil_group, class_type-1].yaxis.label,
                textcoords='offset points',
                size=24,  # Larger font size for soil group
                ha='center',
                va='center',
                rotation=90
            )
            
            # Add the "Predicted fractional cover" label (smaller font size)
            axs[soil_group, class_type-1].annotate(
                "Predicted fractional cover [-]",
                xy=(0, 0),
                xytext=(-axs[soil_group, class_type-1].yaxis.labelpad - 10, -5),
                xycoords=axs[soil_group, class_type-1].yaxis.label,
                textcoords='offset points',
                size=14,  # Smaller font size for "Predicted fractional cover"
                ha='center',
                va='center',
                rotation=90
            )

        # Add "Reference fractional cover" at the bottom of each column
        if soil_group == 5:  # Only add the label once per column (last row)
            axs[soil_group, class_type-1].set_xlabel(
                "Reference fractional cover [-]",
                fontsize=14,
                #labelpad=15
            )
        # Add class label at the top of each column
        if soil_group == 0:  # Only add the class label once per column
            axs[soil_group, column_index].set_title(
                target_name,
                fontsize=24,
                pad=20
            )
      
plt.tight_layout()
plt.savefig(f'NN_test_scatters.png')
