import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import os

# Required: column name for response variable
rv = 'wab_aq'
# Optional: remove specific features
columns_to_drop = []
# Optional: only preserve specific features (make sure to keep rv)
columns_to_keep = []
# Required: name of spreadsheet to analyze
file_path = 'merged_artery_participants.tsv'  # Replace with the path to your data file

# 1. Import data from Excel
data = pd.read_csv(file_path, sep='\t', index_col=0)
# Remove rows where the rv has NaN values
data = data.dropna(subset=[rv])

if columns_to_keep:
    data = data[columns_to_keep]
if columns_to_drop:
    data = data.drop(columns_to_drop, axis=1)

# 3. Replace NaN with zeros and remove sparse columns
data = data.fillna(0)
data = data.loc[:, (data != 0).sum() > 0.1 * data.shape[0]]

# 4. Prepare data for leave-one-out
X = data.drop([rv], axis=1)
y = data[rv]
# Optional normalize rv in range of 0..1, but based on whole sample leave-one-out
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Store lesion volumes for dot sizes
lesion_volumes = data['lesion_volume']  # Ensure 'lesion_volume' is the correct column name

# Leave-One-Out cross-validation
loo = LeaveOneOut()
y_true = []
nn_predicted_values = []
svr_predicted_values = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Neural Network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, verbose=0)
    y_pred_nn = model.predict(X_test)

    # Support Vector Regression model
    svr = SVR(kernel='linear')
    svr.fit(X_train, y_train)
    y_pred_svr = svr.predict(X_test)

    # Store actual and predicted values
    y_true.append(y_test.values[0])
    nn_predicted_values.append(y_pred_nn[0][0])
    svr_predicted_values.append(y_pred_svr[0])

# Calculate evaluation metrics
mse_nn = mean_squared_error(y_true, nn_predicted_values)
rmse_nn = np.sqrt(mse_nn)
r2_nn = r2_score(y_true, nn_predicted_values)

mse_svr = mean_squared_error(y_true, svr_predicted_values)
rmse_svr = np.sqrt(mse_svr)
r2_svr = r2_score(y_true, svr_predicted_values)

# Print the evaluation metrics
print(f"NN Model - MSE: {mse_nn}, RMSE: {rmse_nn}, R2: {r2_nn}")
print(f"SVR Model - MSE: {mse_svr}, RMSE: {rmse_svr}, R2: {r2_svr}")

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Actual Values': y_true,
    'NN Predicted Values': nn_predicted_values,
    'SVR Predicted Values': svr_predicted_values,
    'Lesion Volume': lesion_volumes  # Add the lesion_volume column
})

# Clamp values such that they are forced to make sense (no WAB > 100 and no WAB < 0 as they wouldn't make sense here)
results_df = results_df[(results_df['NN Predicted Values'] <= 100) & (results_df['SVR Predicted Values'] <= 100) & (results_df['NN Predicted Values'] >= 0) & (results_df['SVR Predicted Values'] >= 0)]

# 6. Calculate the correlation (R) and p-value for both models
correlation_nn, p_value_nn = pearsonr(y_true, nn_predicted_values)
correlation_svr, p_value_svr = pearsonr(y_true, svr_predicted_values)

print(f'Neural Network - Correlation (R): {correlation_nn}, p-value: {p_value_nn}')
print(f'SVR - Correlation (R): {correlation_svr}, p-value: {p_value_svr}')

# Normalize 'lesion_volume' to scale dot sizes. Adjust as needed.
min_size = 0.5  # Minimum size of the dot
max_size = 2.0  # Maximum size of the dot
sizes = (results_df['Lesion Volume'] / results_df['Lesion Volume'].max() * (max_size - min_size) + min_size) * 100  # Scaling factor for visibility

# Plot the results
legend_size = 14

plt.figure(figsize=(12, 10))  # Set the figure size

# Plotting NN Predicted vs. Actual Values in green
plt.scatter(results_df['NN Predicted Values'], results_df['Actual Values'], color='green', s=sizes, label='NN Predicted', alpha=0.6)

# Plotting SVR Predicted vs. Actual Values in red
plt.scatter(results_df['SVR Predicted Values'], results_df['Actual Values'], color='red', s=sizes, label='SVR Predicted', alpha=0.6)

# Adding diagonal line for perfect agreement
plt.plot([results_df['Actual Values'].min(), results_df['Actual Values'].max()], [results_df['Actual Values'].min(), results_df['Actual Values'].max()], 'k--', linewidth=2, label='Perfect Agreement')

# Adding titles and labels
plt.title('Aphasia Quotient Prediction', fontsize=18, fontweight='bold')
plt.xlabel('Predicted Values', fontsize=16, fontweight='bold')
plt.ylabel('True Values', fontsize=16, fontweight='bold')

# Significantly increase font size of the legend
plt.legend(loc='upper left', fontsize=legend_size, title='Model Predictions', title_fontsize=legend_size, prop={'size': legend_size}, frameon=False)

# Set tick labels larger and bold
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')

plt.grid(False)  # Optional: Adds a grid
plt.show()  # Display the plot
