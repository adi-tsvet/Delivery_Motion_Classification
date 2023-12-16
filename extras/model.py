import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing.pose_constants import BODY_COLUMNS

# Read CSV data into a numpy array
csv_path = '../data_preprocessing/output/sampleWalking/csvFile/pose_data.csv'
df = pd.read_csv(csv_path)
data_array = df.to_numpy()

# Define the threshold for displacement to identify 'walking'
threshold = 10  # Hypothetical threshold value; adjust as needed

# Calculate displacement for each frame
displacements = np.linalg.norm(data_array[:, :len(BODY_COLUMNS)//2] - data_array[0, :len(BODY_COLUMNS)//2], axis=1)

# Label the data based on displacement
labels = np.where(displacements > threshold, 1, 0)

# Define the neural network
model = tf.keras.Sequential([
    layers.Input(shape=(len(BODY_COLUMNS),)),  # Input layer matching the data shape
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(data_array, labels, epochs=50, batch_size=32)

# Make predictions
test_predictions = model.predict(data_array)  # Assuming you want predictions on the whole dataset

# Plotting actual displacement values
plt.figure(figsize=(10, 6))
plt.plot(displacements, label='Actual Displacement')
plt.xlabel('Frame')
plt.ylabel('Displacement')
plt.title('Actual Displacement')
plt.legend()
plt.show()

# Plotting predicted labels
plt.figure(figsize=(10, 4))
plt.plot(labels, 'g', label='Actual Labels')
plt.plot(test_predictions, 'r', label='Predicted Labels')
plt.xlabel('Frame')
plt.ylabel('Label')
plt.title('Predicted vs Actual Labels')
plt.legend()
plt.show()