import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
import os.path
from squeezenet import SqueezeNet



if __name__ == '__main__':

    # Load model
    model = SqueezeNet(weights='downcam_weights.h5')

    # Gather image information
    image_folder = "../data/aerial_downcam/images/"
    image_names = [fn for fn in os.listdir(image_folder)]
    image_names.sort()
    num_images = len(image_names)

    # Load the images from the folder one at a time, pump through SqueezeNet, and print predictions
    for filename in image_names:
        img = image.load_img(image_folder+filename, target_size=(227, 227))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        print('Predicted:', decode_predictions(preds))
