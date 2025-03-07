import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats
import joblib
import pandas as pd


def load_data(iteration, class_type):
  """ 
  Function to load data for each iteration

  :param interation: int from 0 to 5
  :param class_type: int 1 (NPV), 2 (PV), 3 (soil)
  """

  features_file = f'../data/synthetic_samples/SYNTHMIX_SOIL-000_SHADOW-TRUE_FEATURES_CLASS-00{class_type}_ITERATION-00{iteration}.txt'
  response_file = f'../data/synthetic_samples/SYNTHMIX_SOIL-000_SHADOW-TRUE_RESPONSE_CLASS-00{class_type}_ITERATION-00{iteration}.txt'
  
  features = pd.read_csv(features_file, sep=" ", header=None)
  response = pd.read_csv(response_file, sep=" ", header=None)
  
  return features.values, response.values.ravel()




# Set SVR hyperparameters
svr_params = {'C': 1, 'epsilon': 0.01, 'gamma': 1}

# Initialize lists to store the predictions
predictions_pv, predictions_npv, predictions_soil = [], [], []

# Initialize lists to store the scores for each iteration
scores_pv, scores_npv, scores_soil = [], [], []

# Iterate over each of the 5 iterations
for iteration in range(1, 6):
    # Load data for this iteration (same X for all, but different label)
    X, y_npv = load_data(iteration, 1)  # NPV class
    _, y_pv = load_data(iteration, 2)  # PV class
    _, y_soil = load_data(iteration, 3)  # Soil fraction class

    # Conver to reflectances
    X = X / 10000    

    # Split the data into train and test sets (80/20 split)
    X_train, X_test, y_pv_train, y_pv_test = train_test_split(X, y_pv, test_size=0.2, random_state=42)
    _, _, y_npv_train, y_npv_test = train_test_split(X, y_npv, test_size=0.2, random_state=42)
    _, _, y_soil_train, y_soil_test = train_test_split(X, y_soil, test_size=0.2, random_state=42)

    # Train the SVR model with fixed hyperparameters for each class
    svr_pv = SVR(C=svr_params['C'], epsilon=svr_params['epsilon'], gamma=svr_params['gamma'])
    svr_npv = SVR(C=svr_params['C'], epsilon=svr_params['epsilon'], gamma=svr_params['gamma'])
    svr_soil = SVR(C=svr_params['C'], epsilon=svr_params['epsilon'], gamma=svr_params['gamma'])

    # Train the models
    svr_pv.fit(X_train, y_pv_train)
    svr_npv.fit(X_train, y_npv_train)
    svr_soil.fit(X_train, y_soil_train)

    # Predict on the test set
    y_pv_pred = svr_pv.predict(X_test)
    y_npv_pred = svr_npv.predict(X_test)
    y_soil_pred = svr_soil.predict(X_test)

    # Store predictions for each iteration
    predictions_pv.append(y_pv_pred)
    predictions_npv.append(y_npv_pred)
    predictions_soil.append(y_soil_pred)

    # Calculate performance metrics
    rmse_pv = np.sqrt(mean_squared_error(y_pv_test, y_pv_pred))
    r2_pv = (stats.pearsonr(y_pv_test, y_pv_pred).statistic)**2
    rmse_npv = np.sqrt(mean_squared_error(y_npv_test, y_npv_pred))
    r2_npv = (stats.pearsonr(y_npv_test, y_npv_pred).statistic)**2
    rmse_soil = np.sqrt(mean_squared_error(y_soil_test, y_soil_pred))
    r2_soil = (stats.pearsonr(y_soil_test, y_soil_pred).statistic)**2 

    # Save iteration scores
    scores_pv.append({'iteration': iteration, 'RMSE': rmse_pv, 'R2': r2_pv})
    scores_npv.append({'iteration': iteration, 'RMSE': rmse_npv, 'R2': r2_npv})
    scores_soil.append({'iteration': iteration, 'RMSE': rmse_soil, 'R2': r2_soil})

    # Save the trained models for each iteration
    joblib.dump(svr_pv, f"../models/globalDE/svr_pv_iteration_{iteration}.pkl")
    joblib.dump(svr_npv, f"../models/globalDE/svr_npv_iteration_{iteration}.pkl")
    joblib.dump(svr_soil, f"../models/globalDE/svr_soil_iteration_{iteration}.pkl")

# Now, average the predictions over all 5 iterations for each class
final_pv_pred = np.mean(predictions_pv, axis=0)
final_npv_pred = np.mean(predictions_npv, axis=0)
final_soil_pred = np.mean(predictions_soil, axis=0)

# Calculate RMSE and R² for the final averaged predictions
final_rmse_pv = np.sqrt(mean_squared_error(y_pv_test, final_pv_pred))
final_r2_pv = (stats.pearsonr(y_pv_test, final_pv_pred).statistic)**2

final_rmse_npv = np.sqrt(mean_squared_error(y_npv_test, final_npv_pred))
final_r2_npv = (stats.pearsonr(y_npv_test, final_npv_pred).statistic)**2

final_rmse_soil = np.sqrt(mean_squared_error(y_soil_test, final_soil_pred))
final_r2_soil = (stats.pearsonr(y_soil_test, final_soil_pred).statistic)**2

# Output final averaged results
print("PV Model - RMSE:", final_rmse_pv, "R2:", final_r2_pv)
print("NPV Model - RMSE:", final_rmse_npv, "R2:", final_r2_npv)
print("Soil Model - RMSE:", final_rmse_soil, "R2:", final_r2_soil)

# Save scores
scores_pv.append({'iteration': 'mean pred', 'RMSE': final_rmse_pv, 'R2': final_r2_pv})
scores_npv.append({'iteration': 'mean pred', 'RMSE': final_rmse_npv, 'R2': final_r2_npv})
scores_soil.append({'iteration': 'mean pred', 'RMSE': final_rmse_soil, 'R2': final_r2_soil})

scores_pv_df = pd.DataFrame(scores_pv)
scores_npv_df = pd.DataFrame(scores_npv)
scores_soil_df = pd.DataFrame(scores_soil)

scores_pv_df.to_excel('../results/globalDE_scores_pv.xlsx', index=False)
scores_npv_df.to_excel('../results/globalDE_scores_npv.xlsx', index=False)
scores_soil_df.to_excel('../results/globalDE_scores_soil.xlsx', index=False)
