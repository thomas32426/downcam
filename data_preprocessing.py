from keras.preprocessing import image
import random
import numpy as np
import json

def loadLabels(labels_json):
    # '/home/zeon/data/aerial_downcam/unit_vectors.json'
    labels = []
    with open(labels_json) as json_data:
        loaded = json.load(json_data)
        json_data.close()

    for key, value in loaded.items():
        labels.append([value[0], value[1]])
        #labels.append((atan2(value[1], value[0]) + pi)/(2*pi))
    return labels


def myGenerator(image_folder, labels, batch_size, total_images):
    # Create empty arrays to hold batches of images and labels
    # batch_images = np.zeros((batch_size, 277, 277, 3))
    # batch_labels = np.zeros((batch_size, 2))

    while True:
        batch_images = []
        batch_labels = []
        for i in range(batch_size):
        # Choose random number for images/labels
            index = random.randrange(total_images)

            # Add image and associated label to arrays
            img = image.load_img(image_folder+str(index).zfill(7)+'.jpg', target_size=(227, 227))
            x = image.img_to_array(img)/255
            # x = np.expand_dims(x, axis=0)
            batch_images.append(x)
            # batch_labels[i][0] = labels[i][0]
            # batch_labels[i][1] = labels[i][1]
            batch_labels.append(labels[index])

        batch_images = np.asarray(batch_images)
        batch_labels = np.asarray(batch_labels)
        yield batch_images, batch_labels

def genTest(image_folder, labels, how_many, num_images):
    test_images = []
    test_labels = []
    for i in range(num_images-how_many, num_images):
        img = image.load_img(image_folder + str(i).zfill(7) + '.jpg', target_size=(227, 227))
        x = image.img_to_array(img)
        test_images.append(x)
        test_labels.append(labels[i])

    test_images = np.asarray(test_images)
    test_labels = np.asarray(test_labels)
    return test_images, test_labels