import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from data_preprocessing.pose_constants import BODY_PARTS
from displacement import calculate_displacement
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import regularizers

# List of paths for delivery and walking data
# Paths for delivery and walking data (training)
delivery_paths_train = [
    'data_preprocessing/prepared_output/delivery1/csvFile/pose_data.csv',
    'data_preprocessing/prepared_output/delivery2/csvFile/pose_data.csv',
    'data_preprocessing/prepared_output/delivery3/csvFile/pose_data.csv'
]

nondelivery_paths_train = [
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

nondelivery_paths_test = [
    'data_preprocessing/prepared_output/walking1/csvFile/pose_data.csv',
    'data_preprocessing/output/sampleWalking/csvFile/pose_data.csv'
]

# Calculate displacement for delivery and walking data (training)
delivery_displacements_train = pd.concat([calculate_displacement(path) for path in delivery_paths_train])
nondelivery_displacements_train = pd.concat([calculate_displacement(path) for path in nondelivery_paths_train])

# Calculate displacement for delivery and walking data (testing)
delivery_displacements_test = pd.concat([calculate_displacement(path) for path in delivery_paths_test])
nondelivery_displacements_test = pd.concat([calculate_displacement(path) for path in nondelivery_paths_test])

# Create labels for training and testing data
delivery_labels_train = np.ones(len(delivery_displacements_train))
nondelivery_labels_train = np.zeros(len(nondelivery_displacements_train))
delivery_labels_test = np.ones(len(delivery_displacements_test))
nondelivery_labels_test = np.zeros(len(nondelivery_displacements_test))

# Concatenate training data and labels
train_data = pd.concat([delivery_displacements_train, nondelivery_displacements_train])
train_labels = np.concatenate([delivery_labels_train, nondelivery_labels_train])

# Concatenate testing data and labels
test_data = pd.concat([delivery_displacements_test, nondelivery_displacements_test])
test_labels = np.concatenate([delivery_labels_test, nondelivery_labels_test])

# Number of samples to select
n_samples = 500

# Randomly sample 200 rows from the training data and corresponding labels
train_data_sampled = train_data.sample(n=n_samples, random_state=42)
train_labels_sampled = train_labels[train_data_sampled.index]

# Randomly sample 200 rows from the testing data and corresponding labels
test_data_sampled = test_data.sample(n=n_samples, random_state=42)
test_labels_sampled = test_labels[test_data_sampled.index]


# MODEL 3
# Define a neural network model with BatchNormalization and increased dropout rate and regularization
model = tf.keras.Sequential([
    layers.Input(shape=(len(BODY_PARTS),)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.BatchNormalization(),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

# Train the model with early stopping and validation data
history = model.fit(train_data, train_labels, epochs=100, batch_size=32, verbose=1,
                    validation_data=(test_data, test_labels))


model.save('pose_detection_model')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Posture Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Posture Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

# Plot accuracy
plt.figure(figsize=(12, 10))
# Plot precision
plt.subplot(2, 2, 1)
plt.plot(history.history['precision'], label='Training precision')
plt.plot(history.history['val_precision'], label='Validation precision')
plt.title('Training and Validation Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()

# Plot recall
plt.subplot(2, 2, 2)
plt.plot(history.history['recall'], label='Training recall')
plt.plot(history.history['val_recall'], label='Validation recall')
plt.title('Training and Validation Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

plt.tight_layout()  # Adjusts the plots to prevent overlapping
plt.show()
