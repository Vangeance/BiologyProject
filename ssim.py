from skimage.metrics import structural_similarity
from PIL import Image
import numpy as np

ssim = []
for i in range(1441):
    img1 = Image.open(".\\FruitingBodydata4\\Early\\Early_1\\"+str(i+1)+".tif")
    img_arr1 = np.array(img1.getdata()).reshape((1200, 1600))
    img2 = Image.open(".\\FruitingBodydata4\\Early\\Early_1\\"+str(i+2)+".tif")
    img_arr2 = np.array(img2.getdata()).reshape((1200, 1600))
    ssim.append(structural_similarity(img_arr1, img_arr2))

print(ssim)