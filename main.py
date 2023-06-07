import cv2 as cv
import numpy as np
import matplotlib.pyplot as mat
from tensorflow.keras import datasets, layers, models

(training_images, training_labels), (testing_images,testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images/255, testing_images/255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for i in range(16):
    mat.subplot(4,4,i+1)
    mat.xticks([])
    mat.yticks([])
    mat.imshow(training_images[i], cmap=mat.cm.binary)
    mat.xlabel(class_names[training_labels[i][0]])

mat.show()

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

model = models.load_model('image_classification.model')



