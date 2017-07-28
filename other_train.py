from flow import *
from keras.layers import Conv2D, GlobalAveragePooling2D, Input
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, Sequential
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import os
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from model import *

PATH = os.getcwd()

if __name__ == '__main__':

    # Gather image information
    json_file =     os.path.join(PATH, "..", "vectors.json")
    train_folder =  os.path.join(PATH, "..", "training_images")
    save_path =     os.path.join(PATH, "saved_models", "inceptionNetV3_1")

    batch_size = 64
    start_epochs = 4

    # Load labels and split into lists

    model = class_model(input_shape=(480, 640, 3), output_classes=2)
    model.create_model(model_type="inceptionv3")
    
    model.add_normalize()

    for l in range(6, 15):
        count = 0
        for layer in model.get_model().layers:
            if count > len(model.get_model().layers) - l:
                layer.trainable = True
            else:
                layer.trainable = False
            count += 1
        
        model.compile()

        train_names, image_labels, train_indexes, val_indexes = splitData(train_folder, json_file, old_data=True)

        checkpoint = ModelCheckpoint(str(save_path)+".h5", monitor='val_loss', save_weights_only=True, mode="min", save_best_only=True)

        train_gen = generatorWithVal(train_folder, train_names, image_labels, train_indexes, batch_size)
        val_gen = generatorWithVal(train_folder, train_names, image_labels, val_indexes, batch_size)
        model.get_model().fit_generator(train_gen,
                            steps_per_epoch=len(train_indexes) // batch_size,
                            nb_epoch=start_epochs,
                            validation_data=val_gen,
                            validation_steps=len(val_indexes) // batch_size,
                            max_q_size=128, 
                            verbose=1,
                            callbacks=[checkpoint])

    # # Create the pre-trained base model
    # base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input((480,640,3)), classes=2, pooling='avg')
    #
    # #x = Sequential()
    # # Add on the layers for the head and take base model as input
    # x = base_model.output
    # model = Model(inputs=base_model.input, outputs=x)
    # model.layers.pop()
    # # model.layers.pop()
    # model.summary()
    #
    # x = model.output
    # print(x)
    # x = (Conv2D(2, (3, 3), activation='relu'))(x)
    # x = GlobalAveragePooling2D()(x)
    # # x = (Conv2D(1024, (3, 3), activation='relu'))(x)
    # # predictions = (Conv2D(2, (3, 3), activation='relu'))(x)
    #
    # # Define total model
    # model = Model(inputs=base_model.input, outputs=x)
    #
    # # Train only the top layers (which were randomly initialized)
    # #for layer in base_model.layers:
    # #    layer.trainable = False
    # model.summary()
    #
    # # Compile the model
    # model.compile(loss='mse', optimizer='rmsprop')
    #
    # # Train the model and evaluate
    # model.fit_generator(myGenerator(train_folder, train_labels, 20),
    #                     steps_per_epoch=len(train_labels) // batch_size,
    #                     nb_epoch=start_epochs)
    #                     #validation_data=val_generator,
    #                     #validation_steps=len(val_labels) // batch_size)
    #
    # #score = model.evaluate(test_batches, test_labels, verbose=1)
    #
    # # at this point, the top layers are well trained and we can start fine-tuning
    # # convolutional layers from inception V3. We will freeze the bottom N layers
    # # and train the remaining top layers.
    #
    # # let's visualize layer names and layer indices to see how many layers
    # # we should freeze:
    # #for i, layer in enumerate(base_model.layers):
    # #    print(i, layer.name)
    #
    # # we chose to train the top 2 inception blocks, i.e. we will freeze
    # # the first 249 layers and unfreeze the rest:
    # #for layer in model.layers[:249]:
    # #    layer.trainable = False
    # #for layer in model.layers[249:]:
    # #    layer.trainable = True
    #
    # # model.compile(loss='mse', optimizer='rmsprop')
    #
    # # model.fit_generator(myGenerator(train_folder, train_labels, 20),
    # #                     steps_per_epoch=len(train_labels) // batch_size,
    # #                     nb_epoch=end_epochs)
    # #                     #validation_data=val_generator,
    # #                     #validation_steps=len(val_labels) // batch_size)
    #
    # #score = model.evaluate(test_generator, test_labels, verbose=1)
    # #print('Test loss:', score)
    # #model.save_weights('first_try.h5')