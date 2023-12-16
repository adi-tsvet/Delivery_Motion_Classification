import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from data_preprocessing.pose_constants import BODY_COLUMNS

# Read CSV data into numpy arrays
csv_path = '../data_preprocessing/output/sampleWalking/csvFile/pose_data.csv'
df = pd.read_csv(csv_path)
data_array = df.to_numpy()

csv_path_test = 'data_preprocessing/output/sampleWalking1/sampleWalking1/csvFile/pose_data.csv'
df_test = pd.read_csv(csv_path_test)
test_data_array = df_test.to_numpy()

# Calculate displacement for each frame in the training set
displacements = np.linalg.norm(data_array[:, :len(BODY_COLUMNS) // 2] - data_array[0, :len(BODY_COLUMNS) // 2], axis=1)
threshold = 500
labels = np.where(displacements > threshold, 1, 0)

# Define the neural network
model = tf.keras.Sequential([
    layers.Input(shape=(len(BODY_COLUMNS),)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train the model on the entire dataset
model.fit(data_array, labels, epochs=50, batch_size=32)

# Make predictions on the test dataset
test_probabilities = model.predict(test_data_array)

# Apply threshold to get binary labels
threshold = 0.5
test_binary_labels = (test_probabilities > threshold).astype(int)

# Map binary labels to 'walking' and 'standing'
label_mapping = {0: 'standing', 1: 'walking'}
test_labels_mapped = np.vectorize(label_mapping.get)(test_binary_labels)

# Adding a new column 'Label' to the test dataframe
test_df = pd.DataFrame(test_data_array, columns=BODY_COLUMNS)
test_df['Label'] = test_labels_mapped

# Output the dataframe with labels
print(test_df)

# # Plotting actual displacement values
# plt.figure(figsize=(10, 6))
# plt.plot(displacements, label='Actual Displacement')
# plt.xlabel('Frame')
# plt.ylabel('Displacement')
# plt.title('Actual Displacement')
# plt.legend()
# plt.show()
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



