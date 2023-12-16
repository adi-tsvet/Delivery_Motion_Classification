import json
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import Callback

from sklearn.utils.class_weight import compute_class_weight

# Define input image dimensions
img_width, img_height = 299, 299

datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True
)
# Custom Data Generator
class LogoDataGenerator(Sequence):
    def __init__(self, dataset_path, annotations, batch_size, target_size=(img_width, img_height), datagen=None):
        self.dataset_path = dataset_path
        self.annotations = annotations
        self.batch_size = batch_size
        self.target_size = target_size
        self.image_names = list(annotations.keys())
        self.datagen = datagen

    def __len__(self):
        return int(np.ceil(len(self.image_names) / self.batch_size))

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = min((idx + 1) * self.batch_size, len(self.image_names))
        batch_image_names = self.image_names[batch_start:batch_end]

        batch_images = []
        batch_labels = []
        for image_name in batch_image_names:
            processed_image = self.process_image(image_name)
            if processed_image is not None:
                batch_images.append(processed_image)
                batch_labels.append(self.process_label(image_name))

        return np.array(batch_images), np.array(batch_labels)

    def process_image(self, image_name):
        img_path = os.path.join(self.dataset_path, image_name)

        if not os.path.exists(img_path):
            # if image not found return None
            return None

        img = Image.open(img_path).convert('RGB')  # Convert image to RGB

        # Since we have multiple annotations, let's take the first one for simplicity
        # If you have multiple logos in one image, you'll need a strategy to handle them
        coords = self.annotations[image_name][0]['coordinates']  # Access the first annotation's coordinates

        # Cropping based on the annotation coordinates
        # Be sure to handle the case where the coordinates may go outside the image bounds
        x_center = coords['x']
        y_center = coords['y']
        width = coords['width']
        height = coords['height']
        left = max(x_center - width / 2, 0)
        upper = max(y_center - height / 2, 0)
        right = min(x_center + width / 2, img.width)
        lower = min(y_center + height / 2, img.height)
        img_cropped = img.crop((left, upper, right, lower))

        img_resized = img_cropped.resize(self.target_size)
        img_array = np.array(img_resized) / 255.0
        if self.datagen:
            img_array = self.datagen.random_transform(img_array)
        return img_array

    def process_label(self, image_name):
        label_info = self.annotations[image_name][0]  # Access the first annotation
        label = label_info['label']
        label_index = class_labels.index(label)  # Convert label to index
        return to_categorical(label_index, num_classes=len(class_labels))

# Function to parse JSON annotations
def parse_json_annotations(base_path):
    annotations = {}
    class_labels = []  # List to store class labels

    # Go through each class folder
    for class_folder in os.listdir(base_path):
        class_folder_path = os.path.join(base_path, class_folder)

        if os.path.isdir(class_folder_path):
            class_labels.append(class_folder)  # Add folder name to class labels

            # Read JSON files and match them with images in this class folder
            for file_name in os.listdir(class_folder_path):
                if file_name.endswith('.json'):
                    json_path = os.path.join(class_folder_path, file_name)

                    with open(json_path, 'r') as json_file:
                        data = json.load(json_file)

                        for item in data:
                            image_name = item['image']
                            for annotation in item['annotations']:
                                image_path = os.path.join(class_folder, image_name)

                                if not image_path in annotations:
                                    annotations[image_path] = []

                                annotations[image_path].append(annotation)

    return annotations, class_labels


# Prepare your data generators
train_path = 'dataset'
val_path = 'test_dataset'
# Use the function to get annotations and class labels
train_annotations, train_class_labels = parse_json_annotations(train_path)
val_annotations, val_class_labels = parse_json_annotations(val_path)

# Ensure class labels are consistent across training and validation sets
assert set(train_class_labels) == set(val_class_labels), "Class labels differ between training and validation sets"
class_labels = list(set(train_class_labels))  # Use set to remove any duplicates


train_generator = LogoDataGenerator(train_path, train_annotations, batch_size=32, datagen=datagen)
validation_generator = LogoDataGenerator(val_path, val_annotations, batch_size=32, datagen=datagen)

class DisplayRandomImagesCallback(Callback):
    def __init__(self, data_generator, class_labels, num_images=9):
        self.data_generator = data_generator
        self.class_labels = class_labels
        self.num_images = num_images

    def on_epoch_end(self, epoch, logs=None):
        # Select a random batch
        batch_index = np.random.randint(len(self.data_generator))
        images, labels = self.data_generator[batch_index]

        # Function to convert one-hot labels to class names
        def label_to_classname(label, class_labels):
            return class_labels[np.argmax(label)]

        # Plot images
        plt.figure(figsize=(12, 12))
        for i in range(min(self.num_images, len(images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            plt.title(label_to_classname(labels[i], self.class_labels))
            plt.axis('off')
        plt.suptitle(f'Random Batch of Images - Epoch: {epoch + 1}')
        plt.show()


# Instantiate the callback
#display_images_callback = DisplayRandomImagesCallback(train_generator, class_labels)



# Load and configure the InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.6)(x)
predictions = Dense(len(class_labels), activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Unfreeze more layers
for layer in base_model.layers[-50:]:  # Unfreeze more layers
    layer.trainable = True

# Adjust the learning rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,  # Lower initial learning rate
    decay_steps=10000,
    decay_rate=0.9
)


model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule),
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(), Recall()])

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Continue training with early stopping
history_fine = model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stopping],
)

def load_and_preprocess_image(image_path):
    # Load the image using PIL
    img = Image.open(image_path)

    # Convert to RGB (in case it's a grayscale image)
    img = img.convert('RGB')

    # Resize the image to match the input size expected by the model
    input_size = (224, 224)  # Adjust this size based on your model's input requirements
    img = img.resize(input_size)

    # Convert the image to an array and preprocess for the model
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    # Normalize pixel values to be in the range [0, 1]
    img_array = img_array / 255.0

    return img_array

#testing out the result -


# Use the trained model for predictions
# (Replace 'predict_image.jpg' with the path to your test image)
test_image = load_and_preprocess_image('test.png')
prediction = model.predict(test_image)

# Determine the predicted class name using the class labels
predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]
predicted_class = class_labels[predicted_class_index]
print(f"The predicted class is: {predicted_class}")

# Plot the learning curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_fine.history['accuracy'], label='Train Accuracy')
plt.plot(history_fine.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_fine.history['loss'], label='Train Loss')
plt.plot(history_fine.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.show()

# Save the model
model.save('logo_detection_model')
