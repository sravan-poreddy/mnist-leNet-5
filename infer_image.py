from PIL import Image
import numpy as np
from tensorflow import keras
import sys

# Load the trained LeNet-5 model
model = keras.models.load_model('tf_lenet_model.h5')

def infer_digit(image_path):
    # Load the image
    image = Image.open(image_path)
    grayscale_image = image.convert('L')
    resized_image = grayscale_image.resize((28, 28))

    # Convert the image to a NumPy array
    image_array = np.array(resized_image)

    # Preprocess the image
    preprocessed_image = image_array.reshape((1, 28, 28, 1))
    preprocessed_image = preprocessed_image.astype('float32') / 255

    # Predict the digit
    prediction = model.predict(preprocessed_image, verbose=0)
    predicted_label = np.argmax(prediction)

    return predicted_label


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python infer_image.py <image_path>")
    #     sys.exit(1)
    #
    # image_path = sys.argv[1]
    # predicted_label = infer_digit(image_path)
    # print("Predicted digit label:", predicted_label)


    predicted_label = infer_digit("./mnist_test_images/1/99921.jpg")
    print("Predicted digit label:", predicted_label)