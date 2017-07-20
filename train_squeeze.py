from keras.layers import Convolution2D, GlobalAveragePooling2D
import os.path
from squeezenet import SqueezeNet
from data_preprocessing import *
from keras.layers.core import Lambda, Activation
from keras import backend as K
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3



if __name__ == '__main__':

    def lambda_normalize(x):
        return K.l2_normalize(x, axis=1)

    # Load model
    base_model = SqueezeNet(weights='imagenet')
    #base_model = InceptionV3(weights='imagenet', include_top=False)

    top_model = Sequential()
    x = Convolution2D
    x = Convolution2D(2, (1, 1), padding='valid', name='conv11')(base_model.output)
    x = Activation('relu', name='relu_conv11')(x)
    out = Lambda(lambda_normalize, name='unit_vector_normalization')(x)



    head_model = Model(input = base_model.input, output = out)


    # Gather image information
    image_folder = "../data/aerial_downcam/images/"
    image_names = [fn for fn in os.listdir(image_folder)]
    image_names.sort()
    test_images = 50
    num_images = len(image_names) - test_images# 11932
    labels = loadLabels('/home/zeon/data/aerial_downcam/unit_vectors.json')


    X_test, Y_test = genTest(image_folder, labels, test_images, num_images)

    head_model.compile(loss='mean_squared_error', optimizer='rmsprop')

    head_model.fit_generator(myGenerator(image_folder, labels, 50, num_images), samples_per_epoch=50, nb_epoch=5500)
    score = head_model.evaluate(X_test, Y_test, verbose=1)
    print('Test loss:', score)