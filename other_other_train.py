from flow import loadLabels
import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.preprocessing import image
import numpy as np
import os
import random
from math import atan2



if __name__ == '__main__':

    # Gather image information
    image_folder = "../data/aerial_downcam/images/"
    json_file = '/home/zeon/data/aerial_downcam/unit_vectors.json'
    train_folder = '/home/zeon/data/aerial_downcam/train_images/'
    val_folder = '/home/zeon/data/aerial_downcam/validation_images/'
    test_folder = '/home/zeon/data/aerial_downcam/test_images/'
    batch_size = 1
    num_classes = 1
    epochs = 10
    data_augmentation = False

    # Load labels and split into lists
    train_labels, val_labels, test_labels = loadLabels(json_file, train_folder, val_folder, test_folder)

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(240,320,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Compile the model
    model.compile(loss='mse', optimizer='rmsprop')
    epoch = 0

    while epoch < epochs:
        # Load data
        train_names = [fn for fn in os.listdir(train_folder)]
        batch_images = []
        batch_labels = []

        for i in range(batch_size):
        # Choose random number for images/labels
            index = random.randrange(len(train_names))
            name = train_names[index]

            # Add image and associated label to arrays
            img = image.load_img(train_folder+name, target_size=(240,320))
            x = image.img_to_array(img)/255
            # x = np.expand_dims(x, axis=0)
            batch_images.append(x)
            #batch_labels.append(train_labels[index])
            batch_labels.append(atan2(test_labels[index][1], test_labels[index][0]))

        if i % batch_size == 0:
                final_images = np.asarray(batch_images)
                final_labels = np.asarray(batch_labels)
                batch_images = []
                batch_labels = []

        model.fit(final_images, final_labels, epochs=1) #, validation_split=0.125)
        # Train the model and evaluate
        # model.fit_generator(myGenerator(train_folder, train_labels, 20),
        #                     steps_per_epoch=len(train_labels) // batch_size,
        #                     nb_epoch=epochs)
        #                     #validation_data=val_generator,
        #                     #validation_steps=len(val_labels) // batch_size)
        print("Epoch: {0}/{1}".format(epoch, epochs))
        epoch += 1

    scores = []
    index_list = []
    out = []

    predictions = []
    test_names = [fn for fn in os.listdir(test_folder)]
    for i in range(10):
        batch_index = []
        for i in range(batch_size):
        # Choose random number for images/labels
            index = random.randrange(len(test_names))
            name = test_names[index]
            # Add image and associated label to arrays
            img = image.load_img(test_folder+name, target_size=(120,160))
            x = image.img_to_array(img)/255
            # x = np.expand_dims(x, axis=0)
            batch_images.append(x)
            #batch_labels.append(test_labels[index])
            batch_labels.append(atan2(test_labels[index][1], test_labels[index][0]))
            batch_index.append(index)


            if i % batch_size == 0:
                f_images = np.asarray(batch_images)
                f_labels = np.asarray(batch_labels)
                batch_images = []
                batch_labels = []

        score = model.evaluate(f_images, f_labels)
        pred = model.predict(f_images)
        scores.append(score)
        predictions.append(pred)
        index_list.append(batch_index)

    for i in range(len(index_list)):
        for j in range(len(index_list[i])):
            #print("Index:{0} Angle:{1} Pred:{2} Score:{3}".format(index_list[i][j], atan2(predictions[i][j][1], predictions[i][j][0]), predictions[i][j], scores[i]))
            print("Index:{0} Angle:{1} Score:{2}".format(index_list[i][j], predictions[i][j]), scores[i])
