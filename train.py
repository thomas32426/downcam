import os.path
from data_preprocessing import *
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Activation, Convolution2D, Lambda
from keras import backend as K



if __name__ == '__main__':

    def lambda_normalize(x):
        return K.l2_normalize(x, axis=1)
    # Gather image information
    image_folder = "../data/aerial_downcam/images/"
    image_names = [fn for fn in os.listdir(image_folder)]
    image_names.sort()
    test_images = 50
    num_images = len(image_names) - test_images  # 11932
    labels = loadLabels('/home/zeon/data/aerial_downcam/unit_vectors.json')

    X_test, Y_test = genTest(image_folder, labels, test_images, num_images)

    # Load model
    #base_model = SqueezeNet(weights='downcam_weights.h5', include_top = False)

    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    # x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # # let's add a fully-connected layer
    # x = Dense(1024, activation='relu')(x)
    # # and a logistic layer -- let's say we have 200 classes
    # predictions = Dense(50, activation='relu')(x)

    x = base_model.output

    x = Dense(2048, activation='relu')(x)
    # # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(2, activation='relu')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(loss='mse', optimizer='rmsprop')

    model.fit_generator(myGenerator(image_folder, labels, 1, num_images), samples_per_epoch=1, nb_epoch=10000)
    score = model.evaluate(X_test, Y_test, verbose=1)

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True



    model.compile(loss='mse', optimizer='adam')

    model.fit_generator(myGenerator(image_folder, labels, 1, num_images), samples_per_epoch=1, nb_epoch=10000)
    score = model.evaluate(X_test, Y_test, verbose=1)
    print('Test loss:', score)