import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming we have the following confusion matrix values for each model
conf_matrix_values = {
    'SimpleNet': {'tn': 19, 'fp': 184, 'fn': 23, 'tp': 222},
    'StableNet': {'tn': 50, 'fp': 153, 'fn': 12, 'tp': 233},
    'RobustNet': {'tn': 154, 'fp': 33, 'fn': 11, 'tp': 250}
}

# Sample size for each model
sample_size = 546
# Function to plot confusion matrix
# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix_values, model_name, ax):
    conf_matrix = np.array([[conf_matrix_values['tp'], conf_matrix_values['fn']],
                            [conf_matrix_values['fp'], conf_matrix_values['tn']]])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Positive', 'Negative'],
                yticklabels=['Positive', 'Negative'], ax=ax)
    ax.set_title(f'{model_name} Confusion Matrix')
    ax.set_ylabel('Actual Label')
    ax.set_xlabel('Predicted Label')

# Plotting each confusion matrix
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (model_name, matrix_values) in zip(axes, conf_matrix_values.items()):
    plot_confusion_matrix(matrix_values, model_name, ax)

plt.tight_layout()
plt.show()