import tensorflow as tf
from PIL import Image, ImageFilter, ImageOps
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Define input image dimensions
img_width, img_height = 299, 299

# Function for adding Gaussian blur
def add_gaussian_blur(img_array, blur_factor=0.5):
    # Convert the NumPy array to a PIL Image
    img = Image.fromarray((img_array * 255).astype('uint8'), 'RGB')

    # Convert palette images with transparency to RGBA
    if img.mode == 'P':
        img = img.convert('RGBA')

    # Apply Gaussian blur based on a random condition
    if np.random.rand() < blur_factor:
        img = img.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0, blur_factor)))

    # Convert back to RGB if the image was converted to RGBA
    if img.mode == 'RGBA':
        img = ImageOps.alpha_composite(Image.new("RGB", img.size), img)

    # Convert the PIL Image back to a NumPy array
    img_array_blurred = np.array(img).astype('float32') / 255.0
    return img_array_blurred

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    preprocessing_function=lambda x: add_gaussian_blur(x, blur_factor=0.5)  # Apply Gaussian blur
)


# Split data into train and validation sets
train_generator = datagen.flow_from_directory(
    'dataset',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = datagen.flow_from_directory(
    'test_dataset',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)
# Get the class labels from the train generator
class_labels = list(train_generator.class_indices.keys())

# Calculate class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes)

class_weight_dict = dict(enumerate(class_weights))

# Load and configure the InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions = Dense(len(class_labels), activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Unfreeze some of the top layers of the base model
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Learning rate scheduling
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=10000,
    decay_rate=0.9)

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Continue training with early stopping
history_fine = model.fit(
    train_generator,
    epochs=30,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stopping],
    class_weight = class_weight_dict
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
test_image = load_and_preprocess_image('../logo_detection/wallmartDelivery.png')
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
