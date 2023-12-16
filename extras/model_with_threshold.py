import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from data_preprocessing.pose_constants import BODY_COLUMNS

# Read CSV data into numpy arrays
csv_path = 'data_preprocessing/output/sampleWalking1/sampleWalking1/csvFile/pose_data.csv'
df = pd.read_csv(csv_path)
data_array = df.to_numpy()

csv_path_test = '../data_preprocessing/output/sampleStanding/csvFile/pose_data.csv'
df_test = pd.read_csv(csv_path_test)
test_data_array = df_test.to_numpy()

# Define reference points
reference_points = {
    'RHipX': data_array[0, BODY_COLUMNS.index('RHipX')],
    'RHipY': data_array[0, BODY_COLUMNS.index('RHipY')],
    'LHipX': data_array[0, BODY_COLUMNS.index('LHipX')],
    'LHipY': data_array[0, BODY_COLUMNS.index('LHipY')]
}

# Initialize lists to store displacement data
displacement_data = {body_part: {'X': [], 'Y': []} for body_part in BODY_COLUMNS}
print(displacement_data)
# Calculate displacement for each frame
for i in range(1, len(data_array)):
    for body_part in BODY_COLUMNS:
        if body_part.endswith('X') or body_part.endswith('Y'):
            # Extract X and Y coordinates
            x_index = BODY_COLUMNS.index(body_part)
            y_index = x_index + 1  # Y index is next to X index

            # Calculate displacement for X and Y separately
            displacement_x = data_array[i, x_index] - reference_points[body_part] if body_part.endswith('X') else 0
            displacement_y = data_array[i, y_index] - reference_points[body_part] if body_part.endswith('Y') else 0

            # Store displacement data
            displacement_data[body_part]['X'].append(displacement_x)
            displacement_data[body_part]['Y'].append(displacement_y)

print(displacement_data)
# Placeholder for best threshold and best metric value
best_threshold = None
best_metric = 0.0

# Define threshold range (you can adjust this based on your displacement data)
thresholds_to_try = [0.5, 1.0, 1.5, 2.0, 2.5]  # Example thresholds

# Split data into folds for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for threshold in thresholds_to_try:
    # Placeholder for metric value for this threshold
    avg_metric = 0.0

    for train_index, val_index in kf.split(data_array):
        # Split data into train and validation sets
        train_data, val_data = data_array[train_index], data_array[val_index]

        # Calculate cumulative displacement for train and validation sets
        # For example, considering cumulative displacement of RHipX and RHipY
        displacements_train = np.linalg.norm(
            train_data[:, BODY_COLUMNS.index('RHipX')] - train_data[0, BODY_COLUMNS.index('RHipX')]) + \
                              np.linalg.norm(
                                  train_data[:, BODY_COLUMNS.index('RHipY')] - train_data[0, BODY_COLUMNS.index('RHipY')])
        displacements_val = np.linalg.norm(
            val_data[:, BODY_COLUMNS.index('RHipX')] - val_data[0, BODY_COLUMNS.index('RHipX')]) + \
                            np.linalg.norm(
                                val_data[:, BODY_COLUMNS.index('RHipY')] - val_data[0, BODY_COLUMNS.index('RHipY')])

        # Create labels based on the threshold
        train_labels = np.where(displacements_train > threshold, 1, 0)
        val_labels = np.where(displacements_val > threshold, 1, 0)
        # Define the neural network
        model = tf.keras.Sequential([
            layers.Input(shape=(len(BODY_COLUMNS),)),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        # Train the model
        model.fit(train_data, train_labels, epochs=100, batch_size=32, verbose=0)

        # Make predictions on validation set
        val_probabilities = model.predict(val_data)
        val_binary_labels = (val_probabilities > 0.5).astype(int)

        # Calculate accuracy as the metric
        metric = accuracy_score(val_labels, val_binary_labels)
        avg_metric += metric

        # Calculate average metric over all folds for this threshold
    avg_metric /= kf.get_n_splits()

    # Update best threshold and best metric if this threshold performs better
    if avg_metric > best_metric:
        best_metric = avg_metric
        best_threshold = threshold

print(f"Best threshold: {best_threshold}, Best metric: {best_metric}")

# Calculate cumulative displacement for RHipX and RHipY
cumulative_displacement = np.linalg.norm(np.array(displacement_data['RHipX']['X']) - displacement_data['RHipX']['X'][0]) + \
                          np.linalg.norm(np.array(displacement_data['RHipY']['Y']) - displacement_data['RHipY']['Y'][0])
# Train the final model on the entire dataset using the best threshold
final_train_labels = np.where(cumulative_displacement > best_threshold, 1, 0)

# Define the neural network
model = tf.keras.Sequential([
    layers.Input(shape=(len(BODY_COLUMNS),)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train the model on the entire dataset using the best threshold
model.fit(data_array, final_train_labels, epochs=50, batch_size=32, verbose=0)

# Make predictions on the test dataset
test_probabilities = model.predict(test_data_array)

# Apply threshold to get binary labels
test_binary_labels = (test_probabilities > best_threshold).astype(int)

# Map binary labels to 'walking' and 'standing'
label_mapping = {0: 'standing', 1: 'walking'}
test_labels_mapped = np.vectorize(label_mapping.get)(test_binary_labels)

# Adding a new column 'Label' to the test dataframe
test_df = pd.DataFrame(test_data_array, columns=BODY_COLUMNS)
test_df['Label'] = test_labels_mapped

# Count occurrences of each label in the test data
label_counts = test_df['Label'].value_counts()

# Plotting test labels for 'walking' and 'standing'
plt.figure(figsize=(8, 6))
label_counts.plot(kind='bar', color=['blue', 'green'])
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Test Labels - Walking vs Standing')
plt.xticks(rotation=0)
plt.show()

# Plotting histogram of displacement values
plt.figure(figsize=(8, 6))
plt.hist(cumulative_displacement, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Displacement')
plt.ylabel('Frequency')
plt.title('Distribution of Displacement Values')
plt.show()
