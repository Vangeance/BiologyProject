import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from PIL import Image
import numpy as np
import os
import random
import re
import math
import matplotlib.pyplot as plt
from time import *

IMG_PER_VIDEO = 1441
INTERVAL = 140
CHANNEL_NUM = math.ceil(IMG_PER_VIDEO / INTERVAL)
TRAINING_SPLIT = 3

LABEL = {"Early":0, "Late":1, "NoFB":2, "WT":3}

RAW_IMG_WIDTH = 1600
RAW_IMG_HEIGHT = 1200
IMG_WIDTH = 400
IMG_HEIGHT = 400


def get_dataset():
    """
    dataset(dict) stores 4 categories, each value in dict is a list of lists of images(Integer format of filename)
    e.g. 4 videos in total, 11 images needed from each video: dataset["Early"] = [4 lists, each has 11 integers]
    :return: dataset(dict)
    """
    print("=====getting dataset...=====")

    dataset = {}
    dataset["Early"] = []
    dataset["Late"] = []
    dataset["NoFB"] = []
    dataset["WT"] = []

    for root, _, files in os.walk(".\FruitingBodydata4", topdown=False):
        if len(files) > 0:
            files = sorted([int(file[:-4]) for file in files if file.endswith("tif")])
            img_list = []
            for i in range(0, len(files), INTERVAL):
                img_list.append(files[i])
            if root.find("Early") > 0:
                dataset["Early"].append(img_list)
            elif root.find("Late") > 0:
                dataset["Late"].append(img_list)
            elif root.find("NoFB") > 0:
                dataset["NoFB"].append(img_list)
            else:
                dataset["WT"].append(img_list)
    print("dataset =", dataset)
    return dataset

def create_inputset():
    """
    select the upper left IMG_HEIGHT x IMG_WIDTH pixels
    :return: X(N, IMG_HEIGHT, IMG_WIDTH, CHANNEL_NUM), Y(N, 1)
    """
    start = time()

    dataset = get_dataset()

    print("=====creating inputset...=====")

    X_train = np.empty((1, CHANNEL_NUM, IMG_HEIGHT, IMG_WIDTH))
    Y_train = np.empty((1, 1))
    X_test = np.empty((1, CHANNEL_NUM, IMG_HEIGHT, IMG_WIDTH))
    Y_test = np.empty((1, 1))

    for category in dataset:
        for i, img_list in enumerate(dataset[category][:TRAINING_SPLIT]): # first 3 for training
            X_single = np.empty((1, IMG_HEIGHT, IMG_WIDTH))
            for j in img_list:
                img = Image.open(".\\FruitingBodydata4\\"+category+"\\"+category+"_"+str(i+1)+"\\"+str(j)+".tif")
                img_array = np.array(img.getdata()).reshape((RAW_IMG_HEIGHT, RAW_IMG_WIDTH))[:IMG_HEIGHT,:IMG_WIDTH]
                X_single = np.vstack((X_single, img_array.reshape((1, IMG_HEIGHT, IMG_WIDTH))))
            X_train = np.vstack((X_train, X_single[1:].reshape(1, X_single[1:].shape[0], X_single.shape[1], X_single.shape[2])))
        Y_train = np.vstack((Y_train, np.ones((TRAINING_SPLIT,1),dtype=int) * LABEL[category]))

        # below should be modified after given more than 1 test data
        X_single = np.empty((1, IMG_HEIGHT, IMG_WIDTH))
        for j in dataset[category][3]:
            img = Image.open(".\\FruitingBodydata4\\" + category + "\\" + category + "_4\\" + str(j) + ".tif")
            img_array = np.array(img.getdata()).reshape((RAW_IMG_HEIGHT, RAW_IMG_WIDTH))[:IMG_HEIGHT, :IMG_WIDTH]
            X_single = np.vstack((X_single, img_array.reshape((1, IMG_HEIGHT, IMG_WIDTH))))
        X_test = np.vstack((X_test, X_single[1:].reshape(1,X_single[1:].shape[0],X_single.shape[1],X_single.shape[2])))
        Y_test = np.vstack((Y_test, np.ones((1,1),dtype=int) *  LABEL[category]))

    X_train = X_train[1:]
    Y_train = Y_train[1:]
    X_test = X_test[1:]
    Y_test = Y_test[1:]

    # shuffle the input dataset
    p = np.random.permutation(len(X_train))
    X_train = X_train[p]
    Y_train = Y_train[p]
    p = np.random.permutation(len(X_test))
    X_test = X_test[p]
    Y_test = Y_test[p]

    X_train = X_train.transpose((0, 2, 3, 1))
    Y_train = tf.squeeze(Y_train, axis=1)
    X_test = X_test.transpose((0, 2, 3, 1))
    Y_test = tf.squeeze(Y_test, axis=1)

    end = time()
    print("X_train:", X_train.shape, "X_test:", X_test.shape, "Y_train:", Y_train.shape, "Y_test:", Y_test.shape)
    print("data processing runtime:", end - start)
    return X_train, X_test, Y_train, Y_test

def show_train_hitory(train, validation):
    print(train_history.history.keys())
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train_History')
    plt.ylabel('train')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def train(X_train, X_test, Y_train, Y_test):
    print("=====start training...=====")
    start = time()

    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), input_shape=(400, 400, 11), activation='relu', padding='same'))
    model.add(layers.Dropout(0.25))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(layers.Dropout(0.25))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(4, activation='softmax'))
    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])

    train_history = model.fit(X_train, Y_train, validation_split=0, batch_size=2, epochs=10, verbose=2)

    # show_train_hitory('val_sparse_categorical_accuracy', 'val_sparse_categorical_accuracy')
    # show_train_hitory('loss', 'val_loss')

    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)

    print("test_loss =", test_loss)
    print("test_acc=", test_acc)

    end = time()
    print("training runtime:", end - start)



X_train, X_test, Y_train, Y_test = create_inputset()

train(X_train, X_test, Y_train, Y_test)
