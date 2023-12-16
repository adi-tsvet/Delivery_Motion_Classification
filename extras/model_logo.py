import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Define input image dimensions
img_width, img_height = 299, 299

# Create an ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Prepare data generators
train_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

# Load the InceptionV3 model pre-trained on ImageNet
base_model = InceptionV3(weights='imagenet', include_top=False)

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

# Create the transfer learning model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze layers from the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=20, steps_per_epoch=len(train_generator))

# Evaluate the model on a test set
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    'test_dataset',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

eval_loss, eval_accuracy = model.evaluate(test_generator)
print(f"Eval Accuracy: {eval_accuracy}")

model.save('logo_detection_model')

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
# Get the class labels from the train generator
class_labels = list(train_generator.class_indices.keys())

# Use the trained model for predictions
# (Replace 'predict_image.jpg' with the path to your test image)
test_image = load_and_preprocess_image('../logo_detection/wallmartDelivery.png')
prediction = model.predict(test_image)

# Determine the predicted class name using the class labels
predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]
predicted_class = class_labels[predicted_class_index]
print(f"The predicted class is: {predicted_class}")


