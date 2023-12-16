import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_preprocessing.pose_constants import BODY_COLUMNS

# Load standing and walking pose data into numpy arrays
csv_path_standing = '../data_preprocessing/output/sampleStanding/csvFile/pose_data.csv'
csv_path_walking = '../data_preprocessing/output/sampleWalking/csvFile/pose_data.csv'

df_standing = pd.read_csv(csv_path_standing)
df_walking = pd.read_csv(csv_path_walking)

data_array_standing = df_standing.to_numpy()
data_array_walking = df_walking.to_numpy()

# Assign labels: Standing - 0, Walking - 1
labels_standing = np.zeros(data_array_standing.shape[0])
labels_walking = np.ones(data_array_walking.shape[0])

# Combine data and labels
data_array = np.vstack((data_array_standing, data_array_walking))
labels = np.concatenate((labels_standing, labels_walking))

# Calculate displacement for each frame
displacements = np.linalg.norm(data_array[:, :len(BODY_COLUMNS) // 2] - data_array[0, :len(BODY_COLUMNS) // 2], axis=1)

# Define threshold range
thresholds_to_try = [100, 200, 300, 400, 500, 600, 700, 800, 900]

# Placeholder for best threshold and best metric value
best_threshold = None
best_metric = 0.0

# Split data into train and test sets
train_data, test_data, train_labels, test_labels = train_test_split(data_array, labels, test_size=0.2, random_state=42)

for threshold in thresholds_to_try:
    # Create labels based on the threshold
    train_labels_displacement = np.where(
        np.linalg.norm(train_data[:, :len(BODY_COLUMNS) // 2] - train_data[0, :len(BODY_COLUMNS) // 2], axis=1) > threshold,
        1, 0)
    test_labels_displacement = np.where(
        np.linalg.norm(test_data[:, :len(BODY_COLUMNS) // 2] - test_data[0, :len(BODY_COLUMNS) // 2], axis=1) > threshold,
        1, 0)

    # Define the neural network
    model = tf.keras.Sequential([
        layers.Input(shape=(len(BODY_COLUMNS),)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_data, train_labels_displacement, epochs=50, batch_size=32, verbose=0)

    # Evaluate the model on test data
    test_probabilities = model.predict(test_data)
    test_binary_labels = (test_probabilities > 0.5).astype(int)

    # Calculate accuracy as the metric
    metric = accuracy_score(test_labels_displacement, test_binary_labels)

    # Update best threshold and best metric if this threshold performs better
    if metric > best_metric:
        best_metric = metric
        best_threshold = threshold

print(f"Best threshold: {best_threshold}, Best accuracy: {best_metric}")

# Now, retrain the model on the entire dataset using the best threshold
final_labels_displacement = np.where(displacements > best_threshold, 1, 0)

# Define the neural network
model = tf.keras.Sequential([
    layers.Input(shape=(len(BODY_COLUMNS),)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the entire dataset using the best threshold
model.fit(data_array, final_labels_displacement, epochs=50, batch_size=32, verbose=0)

# Save the model for future use in detecting motion in videos
model.save('motion_detection_model')
