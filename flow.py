import json
import os
from keras.preprocessing import image
import numpy as np
import random

def loadTest(labels_json, train_dir):
    train_names = [fn for fn in os.listdir(train_dir)]

    # Sort them
    train_names.sort()
    # train_numbers = [s.strip('.jpg') for s in train_names]

    # Create lists to be filled with labels
    train_labels = []

    # Load the labels
    with open(labels_json) as json_data:
        loaded = json.load(json_data)
        json_data.close()

    # Sort them by key number
    sorted_labels = sorted(loaded.items(), key=lambda x: str(x[0]))

    for i in sorted_labels:
        train_labels.append(i[1])

    # Put the labels in the lists
    # for i in sorted_labels:
    #     if str(i[0]).zfill(7)+'.jpg' in train_names:
    #         train_labels.append(float[i][1])
        
    return train_names, train_labels

def loadLabels(labels_json, train_dir):
    # '/home/zeon/data/aerial_downcam/unit_vectors.json'

    # Get names of files in the image folders
    train_names = [fn for fn in os.listdir(train_dir)]

    # Sort them
    train_names.sort()

    # Create lists to be filled with labels
    train_labels = []

    # Load the labels
    with open(labels_json) as json_data:
        loaded = json.load(json_data)
        json_data.close()

    # Sort them by key number
    
    sorted_labels = sorted(loaded.items(), key=lambda x: str(x[0]).zfill(7))

    # Put the labels in the lists
    for i in sorted_labels:
        if str(i[0]).zfill(7)+'.jpg' in train_names:
            train_labels.append(i[1])

    return train_labels

# Make a generator

def splitData(image_folder, labels_json, valRatio = 0.1):
    image_labels = loadLabels(labels_json, image_folder)

    train_names = [fn for fn in os.listdir(image_folder)]
    train_names.sort()

    total_len = len(train_names)
    random_index = [i for i in range(total_len)]
    random.shuffle(random_index)

    cut = int(total_len*(1-valRatio))
    train_indexes = random_index[:cut]
    val_indexes = random_index[cut:]

    return train_names, image_labels, train_indexes, val_indexes

def generatorWithVal(image_folder, train_names, image_labels, random_index, batch_size):
    while True:
        batch_images = []
        batch_labels = []
        for i, index in enumerate(random_index):
        # Choose random number for images/labels
            # index = random.randrange(len(train_names))
            name = train_names[index]

            # Add image and associated label to arrays
            img = image.load_img(os.path.join(image_folder, name))
            x = image.img_to_array(img)/255
            # x = np.expand_dims(x, axis=0)
            batch_images.append(x)
            batch_labels.append(image_labels[index])

            if i % batch_size == batch_size-1:
                # final_images = np.asarray(batch_images)
                # final_labels = np.asarray(batch_labels)
                # print((final_images, final_labels))
                yield (np.asarray(batch_images), np.asarray(batch_labels))
                batch_images = []
                batch_labels = []

if __name__ == '__main__':
    json_file = '/home/zeon/data/aerial_downcam/unit_vectors.json'
    train_folder = '/home/zeon/data/aerial_downcam/train_images/'
    val_folder = '/home/zeon/data/aerial_downcam/validation_images/'
    test_folder = '/home/zeon/data/aerial_downcam/test_images/'

    train_names, train_labels = loadTest('/home/marvin/downcam/data/test_vectors.json', '/home/marvin/downcam/data/image_test')

    print(len(train_names), len(train_labels))
    print(train_labels)
    print(train_names)


    # Load images
    #train_images = np.array([np.array(Image.open(train_folder+fn)) for fn in os.listdir(train_folder)])
    #val_images = np.array([np.array(Image.open(val_folder+fn)) for fn in os.listdir(val_folder)])
    #test_images = np.array([np.array(Image.open(test_folder+fn)) for fn in os.listdir(test_folder)])

    # train_labels, val_labels, test_labels = loadLabels(json_file, train_folder, val_folder, test_folder)
    # (image_array, image_labels) = myGenerator(train_folder, train_labels, 50)
    # print(image_array.shape)