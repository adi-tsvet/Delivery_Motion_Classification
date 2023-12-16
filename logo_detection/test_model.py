from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf
# Load the logo detection model
model = load_model('logo_detection_model')

class_labels = ['Amazon', 'Fedex', 'Usps', 'Walmart']
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
test_image = load_and_preprocess_image('test3.jpeg')
prediction = model.predict(test_image)

# Determine the predicted class name using the class labels
predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]
predicted_class = class_labels[predicted_class_index]
print(f"The predicted class is: {predicted_class}")