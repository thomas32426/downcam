import json
import os
import operator

def loadLabels(labels_json, train_dir, val_dir, test_dir):
    # '/home/zeon/data/aerial_downcam/unit_vectors.json'

    # Get names of files in the image folders
    train_names = [fn for fn in os.listdir(train_dir)]
    val_names = [fn for fn in os.listdir(val_dir)]
    test_names = [fn for fn in os.listdir(test_dir)]

    # Sort them
    train_names.sort()
    val_names.sort()
    test_names.sort()

    # Create lists to be filled with labels
    train_labels = []
    val_labels = []
    test_labels = []

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
        elif str(i[0]).zfill(7) + '.jpg' in val_names:
            val_labels.append(i[1])
        elif str(i[0]).zfill(7) + '.jpg' in test_names:
            test_labels.append(i[1])
        #labels.append((atan2(value[1], value[0]) + pi)/(2*pi)) # For angles

    return train_labels, val_labels, test_labels

json_file = '/home/zeon/data/aerial_downcam/unit_vectors.json'
train_folder = '/home/zeon/data/aerial_downcam/train_images/'
val_folder = '/home/zeon/data/aerial_downcam/validation_images/'
test_folder = '/home/zeon/data/aerial_downcam/test_images/'

train_labels, val_labels, test_labels = loadLabels(json_file, train_folder, val_folder, test_folder)

# print(train_labels)
# print(val_labels)
# print(test_labels)
# print(len(train_labels), len(val_labels), len(test_labels))