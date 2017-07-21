from flow import *
from keras.layers import Conv2D, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import os



if __name__ == '__main__':

    # Gather image information
    image_folder = "../data/aerial_downcam/images/"
    json_file = '/home/zeon/data/aerial_downcam/unit_vectors.json'
    train_folder = '/home/zeon/data/aerial_downcam/train_images/'
    val_folder = '/home/zeon/data/aerial_downcam/validation_images/'
    test_folder = '/home/zeon/data/aerial_downcam/test_images/'
    batch_size = 32
    start_epochs = 10
    end_epochs = 10

    # Load labels and split into lists
    train_labels, val_labels, test_labels = loadLabels(json_file, train_folder, val_folder, test_folder)

    for e in
    # Load images
    #train_images = np.array([np.array(Image.open(train_folder+fn)) for fn in os.listdir(train_folder)])
    #val_images = np.array([np.array(Image.open(val_folder+fn)) for fn in os.listdir(val_folder)])
    #test_images = np.array([np.array(Image.open(test_folder+fn)) for fn in os.listdir(test_folder)])

    # Do augmentation and
    gen = ImageDataGenerator(rescale=1/.255)
    train_generator = gen.flow(train_images, train_labels, batch_size=batch_size, ) #target_size=(227,227)
    val_generator = gen.flow(val_images, train_labels, batch_size=batch_size, ) #target_size=(227,227)

    # Create the pre-trained base model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # Normalizing layer
    # def lambda_normalize(x):
    #     return K.l2_normalize(x, axis=1)

    # Add on the layers for the head and take base model as input
    x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    x = (Conv2D(1024, (3, 3), activation='relu'))(x)
    predictions = (Conv2D(2, (3, 3), activation='relu'))(x)

    # Define total model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(loss='mse', optimizer='rmsprop')

    # Train the model and evaluate
    model.fit_generator(train_generator,
                        steps_per_epoch=len(train_labels) // batch_size,
                        nb_epoch=start_epochs,
                        validation_data=val_generator,
                        validation_steps=len(val_labels) // batch_size)

    #score = model.evaluate(test_batches, test_labels, verbose=1)

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

    model.compile(loss='mse', optimizer='rmsprop')

    model.fit_generator(train_generator,
                        steps_per_epoch=len(train_labels) // batch_size,
                        nb_epoch=end_epochs,
                        validation_data=val_generator,
                        validation_steps=len(val_labels) // batch_size)

    #score = model.evaluate(test_generator, test_labels, verbose=1)
    #print('Test loss:', score)
    #model.save_weights('first_try.h5')