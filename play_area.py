import random
import json
from math import atan2, pi
import os

def loadLabels(labels_json):
    # '/home/zeon/data/aerial_downcam/unit_vectors.json'
    labels = []
    with open(labels_json) as json_data:
        loaded = json.load(json_data)
        json_data.close()

    for key, value in loaded.items():
        #labels.append([value[0], value[1]])
        labels.append((atan2(value[1], value[0]) + pi) / (2 * pi))
    return labels


a = loadLabels('/home/zeon/data/aerial_downcam/unit_vectors.json')
print(a)
# def loadLabels(labels_json):
#     # '/home/zeon/data/aerial_downcam/unit_vectors.json'
#     labels = []
#     with open(labels_json) as json_data:
#         loaded = json.load(json_data)
#         json_data.close()
#
#     for key, value in loaded.items():
#         labels.append([value[0], value[1]])
#     return labels
#
# labels = loadLabels('/home/zeon/data/aerial_downcam/unit_vectors.json')
#
# for i in range(10):
#     print(labels[i][0])
# total_images = 11932
# index = random.randrange(total_images)
# print(index)