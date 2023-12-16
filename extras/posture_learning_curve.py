import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from data_preprocessing.pose_constants import BODY_PARTS
from displacement import calculate_displacement
from tensorflow.keras import regularizers
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# List of paths for delivery and walking data
# Paths for delivery and walking data (training)
delivery_paths_train = [
    'data_preprocessing/prepared_output/delivery1/csvFile/pose_data.csv',
    'data_preprocessing/prepared_output/delivery2/csvFile/pose_data.csv',
    'data_preprocessing/prepared_output/delivery3/csvFile/pose_data.csv'
]

walking_paths_train = [
    'data_preprocessing/output/sampleWalking/csvFile/pose_data.csv',
    'data_preprocessing/prepared_output/walking2/csvFile/pose_data.csv',
    'data_preprocessing/prepared_output/walking3/csvFile/pose_data.csv',
    'data_preprocessing/output/sampleWalking1/csvFile/pose_data.csv'
]

# Paths for delivery and walking data (testing)
delivery_paths_test = [
        'data_preprocessing/prepared_output/delivery1/csvFile/pose_data.csv',
        'data_preprocessing/prepared_output/delivery4/csvFile/pose_data.csv'
]

walking_paths_test = [
    'data_preprocessing/prepared_output/walking1/csvFile/pose_data.csv',
    'data_preprocessing/output/sampleWalking1/csvFile/pose_data.csv'

]

# Calculate displacement for delivery and walking data (training)
delivery_displacements_train = pd.concat([calculate_displacement(path) for path in delivery_paths_train])
walking_displacements_train = pd.concat([calculate_displacement(path) for path in walking_paths_train])

# Calculate displacement for delivery and walking data (testing)
delivery_displacements_test = pd.concat([calculate_displacement(path) for path in delivery_paths_test])
walking_displacements_test = pd.concat([calculate_displacement(path) for path in walking_paths_test])

# Create labels for training and testing data
delivery_labels_train = np.ones(len(delivery_displacements_train))
walking_labels_train = np.zeros(len(walking_displacements_train))
delivery_labels_test = np.ones(len(delivery_displacements_test))
walking_labels_test = np.zeros(len(walking_displacements_test))

# Plotting distribution of displacement data
plt.figure(figsize=(12, 6))

# Distribution of delivery displacement data
plt.subplot(1, 2, 1)
plt.hist(delivery_displacements_train.values.flatten(), bins=50, alpha=0.7, label='Delivery')
plt.title('Delivery Displacement Distribution')
plt.xlabel('Displacement')
plt.ylabel('Frequency')
plt.legend()

# Distribution of walking displacement data
plt.subplot(1, 2, 2)
plt.hist(walking_displacements_train.values.flatten(), bins=50, alpha=0.7, label='Walking')
plt.title('Walking Displacement Distribution')
plt.xlabel('Displacement')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

# Concatenate training data and labels
train_data = pd.concat([delivery_displacements_train, walking_displacements_train])
train_labels = np.concatenate([delivery_labels_train, walking_labels_train])

# Concatenate testing data and labels
test_data = pd.concat([delivery_displacements_test, walking_displacements_test])
test_labels = np.concatenate([delivery_labels_test, walking_labels_test])


# Adjusting layout for Box Plot
num_columns = len(train_data.columns)
plt.figure(figsize=(20, 20))
for i in range(num_columns):
    plt.subplot((num_columns + 3) // 4, 4, i+1)
    train_data.iloc[:, i].plot(kind='box')
    plt.title(train_data.columns[i])
    plt.tight_layout()
plt.show()

# Adjusting layout for Histograms
plt.figure(figsize=(20, 20))
for i in range(num_columns):
    plt.subplot((num_columns + 3) // 4, 4, i+1)
    train_data.iloc[:, i].hist(bins=50)
    plt.title(train_data.columns[i])
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
plt.show()


# Correlation matrix
correlation_matrix = train_data.corr()

# Heatmap visualization
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# Training a random forest classifier
rf = RandomForestClassifier()
rf.fit(train_data, train_labels)

# Plotting feature importances
plt.figure(figsize=(12, 6))
importances = rf.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [train_data.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# Based on correlation matrix and feature importance, let's assume you decided to remove 'LElbow' and 'REye'
features_to_remove = ['REye','REar','LEye','LEar', 'Nose']

# Drop these features from your training and testing data
train_data = train_data.drop(columns=features_to_remove)
test_data = test_data.drop(columns=features_to_remove)

scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Number of samples to select
n_samples = 200

# Randomly sample 200 rows from the training data and corresponding labels
train_data_sampled = train_data_scaled.sample(n=n_samples, random_state=42)
train_labels_sampled = test_data_scaled[train_data_sampled.index]

# Randomly sample 200 rows from the testing data and corresponding labels
test_data_sampled = test_data.sample(n=n_samples, random_state=42)
test_labels_sampled = test_labels[test_data_sampled.index]


# MODEL 3
# Define a neural network model with BatchNormalization and increased dropout rate
# Function to train the model on a subset of data and return accuracies
def train_and_evaluate(train_data, train_labels, test_data, test_labels, subset_size):
    # Create a subset of data
    subset_data, _, subset_labels, _ = train_test_split(train_data, train_labels, train_size=subset_size,
                                                        random_state=42)

    # Define the model (same as before)
    model = tf.keras.Sequential([
        layers.Input(shape=(len(BODY_PARTS)-len(features_to_remove),)),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(subset_data, subset_labels, epochs=100, batch_size=32, verbose=0)

    # Evaluate the model on training subset and validation data
    train_accuracy = model.evaluate(subset_data, subset_labels, verbose=0)[1]
    validation_accuracy = model.evaluate(test_data, test_labels, verbose=0)[1]

    return train_accuracy, validation_accuracy


# Range of training sizes (e.g., 10%, 20%, ..., 100%)
training_sizes = np.linspace(0.1, 0.9, 10)

# Record accuracies
train_accuracies = []
validation_accuracies = []

for size in training_sizes:
    train_acc, val_acc = train_and_evaluate(train_data, train_labels, test_data, test_labels, size)
    train_accuracies.append(train_acc)
    validation_accuracies.append(val_acc)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(training_sizes, train_accuracies, label='Training Accuracy')
plt.plot(training_sizes, validation_accuracies, label='Validation Accuracy')
plt.title('Learning Curve')
plt.xlabel('Fraction of Training Data Used')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



