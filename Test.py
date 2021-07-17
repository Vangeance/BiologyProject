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

d = np.array([1,2,3]).reshape((3,1))
