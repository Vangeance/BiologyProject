import time
from skimage.metrics import structural_similarity
import numpy as np
import re
import os
import math
from PIL import Image
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt


a = np.array([[1,2,3],
              [4,5,6]])
a = a.reshape(1,a.shape[0],a.shape[1])

b = np.empty((1,2,3))

l = [b,a,a]
c = np.vstack(l)
print(c)
print(c[1:])
print(c.shape)

print("++++++++++++++++++++++")
c[2]*=2
print(c)

img1 = Image.open("./FruitingBodydata4/Early/Early_1/1.tif")
img_arr1 = np.array(img1.getdata()).reshape((1200, 1600))

img2 = Image.open("./FruitingBodydata4/Early/Early_1/1441.tif")
img_arr2 = np.array(img2.getdata()).reshape((1200, 1600))
print(structural_similarity(img_arr1, img_arr2))

