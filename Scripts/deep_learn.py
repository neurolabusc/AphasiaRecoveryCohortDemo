import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import os

#This script takes merged_artery_participants.tsv in as input
#By default, it attempts to use all columns of data to predict the final column of data (wab-aq)
#As a first pass it trains a model on these data to identify outliers
#Any extreme outliers identified during model training are automatically removed before training the final model.
#In the case of this repository, one such participant was removed at this stage of processing.
#As a second step, it trains the model using the remaining dataframe
#*Note, any extreme outliers with impossible scores (i.e. wab-aq actual or wab-aq predicted) are removed before plotting the figure.
#In the case of is repository data, one participant was removed at this stage of processing.
#As a final step, the data is plotted.

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
# Optional normalize rv in range of 0..1, but based on whole sample leakage for leave-one-out
# y = (y - y.min()) / (y.max() - y.min())
scaler = StandardScaler()

# 5. Initialize for leave-one-out cross-validation
loo = LeaveOneOut()
y_true, y_pred_nn, y_pred_svr = [], [], []

for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Store m/std dev of training data and apply normalization to train and test data separately
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Neural Network
    model_nn = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model_nn.compile(optimizer='adam', loss='mean_squared_error')
    model_nn.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
    pred_nn = model_nn.predict(X_test).flatten()

    # SVR
    model_svr = SVR(kernel='rbf')
    model_svr.fit(X_train, y_train)
    pred_svr = model_svr.predict(X_test)

    # Collecting predictions
    y_true.extend(y_test)
    y_pred_nn.extend(pred_nn)
    y_pred_svr.extend(pred_svr)

# Now calculate y_true and y_pred_nn
y_true = np.array(y_true)
y_pred_nn = np.array(y_pred_nn)

# Identify the index of the outlier
outlier_index = np.argmax(np.abs(y_true - y_pred_nn))  # You can choose any predicted values array here

# Get the participant ID corresponding to the outlier index
offending_participant_id = data.index[outlier_index]

# Print the ID of the offending participant to the screen
print(f"The ID of the offending participant is: {offending_participant_id}")

# Remove the row corresponding to the outlier
data = data.drop(index=offending_participant_id)

# Reassign X and y after removing the outlier row
X = data.drop([rv], axis=1)
y = data[rv]

# Continue with your analysis using the updated X and y
# Optional normalize rv in range of 0..1, but based on whole sample leakage for leave-one-out
# y = (y - y.min()) / (y.max() - y.min())
scaler = StandardScaler()

# 5. Initialize for leave-one-out cross-validation
loo = LeaveOneOut()
y_true, y_pred_nn, y_pred_svr = [], [], []

for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Store m/std dev of training data and apply normalization to train and test data separately
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Neural Network
    model_nn = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model_nn.compile(optimizer='adam', loss='mean_squared_error')
    model_nn.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
    pred_nn = model_nn.predict(X_test).flatten()

    # SVR
    model_svr = SVR(kernel='rbf')
    model_svr.fit(X_train, y_train)
    pred_svr = model_svr.predict(X_test)

    # Collecting predictions
    y_true.extend(y_test)
    y_pred_nn.extend(pred_nn)
    y_pred_svr.extend(pred_svr)

# Scatter plot for NN predictions
plt.scatter(y_true, y_pred_nn, color='red', label='Neural Network')

# Create a DataFrame with the data
results_df = pd.DataFrame({
    'Actual Values': y_true,
    'NN Predicted Values': y_pred_nn,
    'SVR Predicted Values': y_pred_svr,
    'Lesion Volume': data['lesion_volume'].values  # Add the lesion_volume column
})

#Clamp values such that they are forced to make sense (no WAB > 100 makes sense for example)
results_df = results_df[(results_df['NN Predicted Values'] <= 100) & (results_df['SVR Predicted Values'] <= 100)]

# 6. Calculate the correlation (R) and p-value for both models
correlation_nn, p_value_nn = pearsonr(y_true, y_pred_nn)
correlation_svr, p_value_svr = pearsonr(y_true, y_pred_svr)

print(f'Neural Network - Correlation (R): {correlation_nn}, p-value: {p_value_nn}')
print(f'SVR - Correlation (R): {correlation_svr}, p-value: {p_value_svr}')

# Define the directory to save the plot
output_directory = os.path.dirname(file_path)

# Save the DataFrame to a CSV file
results_file_path = os.path.join(output_directory, 'predictions_data.csv')
results_df.to_csv(results_file_path, index=False)

# Path to the CSV file
file_path = 'predictions_data.csv'

# Read the CSV file
data = pd.read_csv(file_path)

legend_size = 14

# Normalize 'lesionsize' to scale dot sizes. Adjust as needed.
min_size = 0.5  # Minimum size of the dot
max_size = 2.0  # Maximum size of the dot
sizes = (data['Lesion Volume'] / data['Lesion Volume'].max() * (max_size - min_size) + min_size) * 100  # Scaling factor for visibility

plt.figure(figsize=(12, 10))  # Set the figure size

# Plotting NN Predicted vs. Actual Values in green
plt.scatter(data['NN Predicted Values'], data['Actual Values'], color='green', s=sizes, label='NN Predicted', alpha=0.6)

# Plotting SVR Predicted vs. Actual Values in red
plt.scatter(data['SVR Predicted Values'], data['Actual Values'], color='red', s=sizes, label='SVR Predicted', alpha=0.6)

# Adding diagonal line for perfect agreement
plt.plot([data['Actual Values'].min(), data['Actual Values'].max()], [data['Actual Values'].min(), data['Actual Values'].max()], 'k--', linewidth=2, label='Perfect Agreement')

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
