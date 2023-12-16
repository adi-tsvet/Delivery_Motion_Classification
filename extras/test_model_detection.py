import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the new CSV data
csv_path_new_data = 'data_preprocessing/output/sampleWalking1/sampleWalking1/csvFile/pose_data.csv'
df_new_data = pd.read_csv(csv_path_new_data)

# Convert the DataFrame to a numpy array
new_data_array = df_new_data.to_numpy()

# Load the trained model
model = load_model('motion_detection_model')  # Load the previously saved model

# Make predictions on the new data
predictions = model.predict(new_data_array)

# Apply threshold to get binary labels
threshold = 0.5  # You can use the threshold determined during training or adjust as needed
binary_labels = (predictions > threshold).astype(int)

# Map binary labels to 'standing' and 'walking' based on your label mapping
label_mapping = {0: 'standing', 1: 'walking'}
labels_mapped = np.vectorize(label_mapping.get)(binary_labels)

# Adding the predicted labels as a new column to the new data
df_new_data['Predicted_Label'] = labels_mapped

# Visualize the predicted labels
plt.figure(figsize=(10, 6))
plt.plot(df_new_data['Predicted_Label'], 'bo', label='Predicted Labels')
plt.xlabel('Frame')
plt.ylabel('Predicted Label')
plt.title('Predicted Motion Labels - Walking vs Standing')
plt.legend()
plt.show()
