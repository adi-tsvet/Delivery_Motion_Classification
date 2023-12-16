import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from data_preprocessing.pose_constants import BODY_PARTS
from displacement import calculate_displacement
from tensorflow.keras import layers, regularizers, models
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve,auc,confusion_matrix
from matplotlib_venn import venn3
import matplotlib.pyplot as plt
import seaborn as sns

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

# Concatenate training data and labels
train_data = pd.concat([delivery_displacements_train, walking_displacements_train])
train_labels = np.concatenate([delivery_labels_train, walking_labels_train])

# Concatenate testing data and labels
test_data = pd.concat([delivery_displacements_test, walking_displacements_test])
test_labels = np.concatenate([delivery_labels_test, walking_labels_test])

# Based on correlation matrix and feature importance,
features_to_remove = ['REye','REar','LEye','LEar', 'Nose']

# Drop these features from your training and testing data
train_data = train_data.drop(columns=features_to_remove)
test_data = test_data.drop(columns=features_to_remove)


# Number of samples to select
n_samples = 200

# Randomly sample 200 rows from the training data and corresponding labels
train_data_sampled = train_data.sample(n=n_samples, random_state=42)
train_labels_sampled = train_labels[train_data_sampled.index]

# Randomly sample 200 rows from the testing data and corresponding labels
test_data_sampled = test_data.sample(n=n_samples, random_state=42)
test_labels_sampled = test_labels[test_data_sampled.index]

# Define your three models as per the provided definitions
def create_model_1(train):
    # Define Model 1: "BasicNet"
    model = tf.keras.Sequential([
        layers.Input(shape=(train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def create_model_2(train):
    # Define Model 2: "StableNet"
    model = tf.keras.Sequential([
        layers.Input(shape=(train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def create_model_3(train):
    # Define Model 3: "RobustNet"
    model = tf.keras.Sequential([
        layers.Input(shape=(train.shape[1],)),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


# Compile and fit the models
def compile_and_fit(model, train_data, train_labels, test_data, test_labels, epochs=100):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model.fit(train_data, train_labels, epochs=epochs, batch_size=32,
                     validation_data=(test_data, test_labels), verbose=0)

# Initialize models
model_1 = create_model_1(train_data_sampled)
model_2 = create_model_2(train_data_sampled)
model_3 = create_model_3(train_data_sampled)

# Fit models
history_1 = compile_and_fit(model_1, train_data_sampled, train_labels_sampled, test_data_sampled, test_labels_sampled)
history_2 = compile_and_fit(model_2, train_data_sampled, train_labels_sampled, test_data_sampled, test_labels_sampled)
history_3 = compile_and_fit(model_3, train_data_sampled, train_labels_sampled, test_data_sampled, test_labels_sampled)


def evaluate_model_performance(model, test_data, test_labels):
    # Predict probabilities
    y_probs = model.predict(test_data).ravel()

    # Calculate predicted labels
    y_pred = (y_probs > 0.5).astype(int)

    # Calculate precision and recall
    precision = precision_score(test_labels, y_pred)
    recall = recall_score(test_labels, y_pred)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(test_labels, y_pred)

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(test_labels, y_probs)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc, conf_matrix, precision, recall

# Assuming models have been initialized and trained
models = [model_1, model_2, model_3]  # Replace with your actual models
roc_info = {}
conf_matrices = {}

# Evaluate each model and plot ROC curve
plt.figure(figsize=(8, 6))
for i, model in enumerate(models, start=1):
    fpr, tpr, roc_auc, conf_matrix, precision, recall = evaluate_model_performance(model, test_data, test_labels)
    roc_info[f'model_{i}'] = (fpr, tpr, roc_auc)
    conf_matrices[f'model_{i}'] = conf_matrix

    # Plot ROC curve for each model
    plt.plot(fpr, tpr, label=f'Model {i} (AUC = {roc_auc:.2f})')

    # Print precision and recall for each model
    print(f'Model {i}: Precision = {precision:.2f}, Recall = {recall:.2f}, ROC AUC = {roc_auc:.2f}')

plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Plot confusion matrices
for model_key, conf_matrix in conf_matrices.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'{model_key} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Prepare error sets for the Venn diagram
error_sets = {}
for model_key, model in enumerate(models, start=1):
    y_pred = (model.predict(test_data).ravel() > 0.5).astype(int)
    errors = np.where(y_pred != test_labels)[0]
    error_sets[f'model_{model_key}'] = set(errors)

# Plot Venn diagram
venn3(subsets=[error_sets['model_1'], error_sets['model_2'], error_sets['model_3']],
      set_labels=('Model 1', 'Model 2', 'Model 3'))
plt.title('Venn Diagram of Model Errors')
plt.show()

# Sample error analysis (Example for Model 1)
misclassified_indices_model_1 = list(error_sets['model_1'])
print("Model 1 Misclassified sample indices:", misclassified_indices_model_1[:5])

misclassified_indices_model_2 = list(error_sets['model_2'])
print("Model 2 Misclassified sample indices:", misclassified_indices_model_2[:5])

misclassified_indices_model_3 = list(error_sets['model_3'])
print("Model 3 Misclassified sample indices:", misclassified_indices_model_3[:5])