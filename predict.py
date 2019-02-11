
import numpy as np
import matplotlib.pyplot as plt
import argparse
import utils
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

def predict(test_image_path, mean_image_path, model, model_parms):
    """
    Predicts the class of an image using the trained model
    
    # Arguments
        test_image_path: full path for the test image
        mean_image_path: full path for the mean image
        model: trained squeezenet keras model
        model_parms: dictionary of model parameters (classes and image dimensions)
    
    # Return
        The predicted class of the image
    """
    
    # Load trained model
    model = load_model(args.saved_model)
    # Load model parms
    model_parms = utils.load_model_parms(args.model_parms)
    
    # Mean image
    img_mean_array = img_to_array(load_img(args.mean_image, 
                                           target_size=(model_parms['height'], model_parms['width'])))
    # Test image
    img_test_array = img_to_array(load_img(args.test_image, 
                                           target_size=(model_parms['height'], model_parms['width'])))
    img_test_array -= img_mean_array
    img_test_batch = np.expand_dims(img_test_array, axis=0)
    
    # Predict the class of the test image
    prob = model.predict(x=img_test_batch, batch_size=1, verbose=1, steps=None)
    prediction = np.argmax(prob, axis=1)[0]
    return model_parms['classes'][prediction]


if __name__ == "__main__":
    # Parse arguements
    parser = argparse.ArgumentParser(description="SqueezeNet Prediction.")
    parser.add_argument("--test-image", type=str, default='./images/test_image.jpg', 
                        dest='test_image', help="The full path for the test image")
    parser.add_argument("--mean-image", type=str, default='./images/mean_image.jpg', 
                        dest='mean_image', help="The full path for mean image of training dataset.")
    parser.add_argument("--saved-model", type=str, default='./model/squeezenet_model.h5', 
                        dest='saved_model', help="The trained squeezenet keras model (.h5)")
    parser.add_argument("--model-parms", type=str, default='./model/model_parms.json', 
                        dest='model_parms', help="The dictionary of model params (classes and image dimensions)")
    args = parser.parse_args()
    
    # Predict the class of the image
    predicted_class = predict(args.test_image, args.mean_image, args.saved_model, args.model_parms)
    
    # Display the image and predicted class
    img_test = load_img(args.test_image)
    fig = plt.figure()
    plt.imshow(img_test)
    plt.title(predicted_class)
    plt.axis('off')
    plt.show()
