import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
import os
import keras
from model import *
from flow import *
from data_preprocessing import *
import math
import cv2

if __name__ == '__main__':

    # File locations
    json_file =     '/home/marvin/downcam/data/test_vectors.json'
    image_test_folder =  '/home/marvin/downcam/data/image_test'
    save_path =     '/home/marvin/downcam/saved_models/inceptionNetV3_1.h5'

    # Load model
    model = class_model(input_shape=(480, 640, 3), output_classes=2)
    model.create_model(model_type="inceptionv3")
    model.add_normalize()
    model.load_weights(save_path)

    # Load labels
    image_test_names, image_test_labels = loadTest('/home/marvin/downcam/data/test_vectors.json', '/home/marvin/downcam/data/image_test')

    pred_list = []
    # Load the images from the folder one at a time, pump through SqueezeNet, and print predictions
    for i, filename in enumerate(image_test_names):
        if 2000 < i and i < 2300:
            # Get image ready
            img = image.load_img(os.path.join(image_test_folder, filename))
            cv_img = cv2.imread(os.path.join(image_test_folder, filename), 1)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            # print(image_test_labels[i])
            # Get prediction, process, print
            pred = model.predict(x)
            # score = model.evaluate(x, np.asarray(image_test_labels[i]))
            # print(pred.shape)
            angle = math.atan2(pred[0][1], pred[0][0])
            angle = angle*180/math.pi
            print('{0} angle: {1}'.format(filename,angle))
            pred_list.append(angle)

            # cv2.arrowedLine(cv_img, 100*pred[0][0], 100*pred[0][1], (0, 0, 0))
            cv2.line(cv_img,(320,240),(int(100*image_test_labels[i][0])+320, int(-100*image_test_labels[i][1])+240), (0,255,0), 5)
            cv2.line(cv_img,(320,240),(int(100*pred[0][0])+320, int(-100*pred[0][1])+240), (255,0,0), 5)

            cv2.imshow('image',cv_img)
            cv2.waitKey(500)
            #print('Predicted:', decode_predictions(preds))
