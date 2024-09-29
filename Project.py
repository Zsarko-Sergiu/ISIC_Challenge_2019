"""  ISIC 2019 data is provided courtesy of the following sources:
  BCN_20000 Dataset: (c) Department of Dermatology, Hospital Clínic de Barcelona
  HAM10000 Dataset: (c) by ViDIR Group, Department of Dermatology, Medical University of Vienna; https://doi.org/10.1038/sdata.2018.161
  MSK Dataset: (c) Anonymous; https://arxiv.org/abs/1710.05006 ; https://arxiv.org/abs/1902.03368 """

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import tensorflow as tf
from tensorflow.keras import layers
import random
import fileReadingMethods

#GPU config
physical_devices=tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0],True)

#We have a csv file containg the names of the images given in our folder and a label set to 1 where it represents what type of diagnostic it is out of the following
#       1.Melanoma
#       2.Melanocytic nevus
#       3.Basal cell carcinoma
#       4.Actinic keratosis
#       5.Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)
#       6.Dermatofibroma
#       7.Vascular lesion
#       8.Squamous cell carcinoma
#       9.None of the others

#Each of the 9 categories above will represent a label for our training set ; The number of neurons in the output layer will also be equal to 9 for each of the labels.
#The nr of neurons in our input layer can be taken any way we want since we will just resize the images to a set pixel amount in height and width
#We have a total of 25,331 images ; The two tasks are split as follows:
#       1.classify dermoscopic images without meta-data
#       2.classify images with additional available meta-data
#


#load data
#from the csv file, take each name of the image and its classification (where "1" is written in the table) which will represent the label for that image.

ds_train=tf.data.TextLineDataset("training_cut.csv")

#for line in ds_train.skip(1).take(5):
# print(tf.strings.split(line,","))

#get dataset where each line represents the name and type of disease we have
dataset_train=fileReadingMethods.build_dataset(ds_train)
#for line in data:
#    print(line)

#convert dataset to tensor
dataset_train=fileReadingMethods.convert_to_tensor(dataset_train)


#build ds_train based on the given dataset
ds_train=fileReadingMethods.build_train_set(dataset_train)

for image,label in ds_train:
    print("image:",image)
    print("label:",label)


#
#now that we have ds_train and ds_test setup, we can start defining our model architecture

model=tf.keras.Sequential(
    [
        tf.keras.Input(shape=(128,128,3)), #200 by 200 ; 3 channels
        layers.Conv2D(32,kernel_size=3,activation='relu'),
        layers.Conv2D(64,kernel_size=3,activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        layers.Conv2D(128,kernel_size=3,activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(256,activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(9) # 9 options ; see above for explanation
    ]
)

model.compile(
    metrics=['accuracy'],
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

model.summary()

model.fit(ds_train,epochs=10,verbose=2)