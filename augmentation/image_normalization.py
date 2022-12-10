import os
from os.path import exists, join
import SimpleITK as sitk
import tifffile as tiff
import csv
import json
import numpy as np
import cv2
from matplotlib import pyplot as plt
import skimage
import skimage.filters


# def my_image_normalize(image, bit_depth=16, low=0.05, high=0.95):
#     max_pixel_value = 2 ** bit_depth - 1 
#     low *= max_pixel_value 
#     high *= max_pixel_value

#     image = np.array(image, dtype=float)
#     condition = np.logical_and(np.greater_equal(image, low), np.less_equal(image, high))
#     if image[condition].shape[0] == 0:
#         min_condition_pixel = 0
#         max_condition_pixel = max_pixel_value
#     else:
#         min_condition_pixel = np.min(image[condition])
#         max_condition_pixel = np.max(image[condition])
#     image[condition] = max_pixel_value * (image[condition] - min_condition_pixel) / (max_pixel_value - min_condition_pixel)

#     return image

def my_image_normalize(image, bit_depth=16, low=0.05, high=0.95):
    max_pixel_value = 2 ** bit_depth - 1
    image = np.array(image, dtype=float)
    sorted_pixels = np.sort(image, axis=None)
    num_pixels = sorted_pixels.shape[0]
    min_condition_pixel = sorted_pixels[int(num_pixels * low)]
    max_condition_pixel = sorted_pixels[int(num_pixels * high) - 1]

    image = max_pixel_value * (image - min_condition_pixel) / (max_condition_pixel - min_condition_pixel)
    image = np.minimum(max_pixel_value, image)
    image = np.maximum(0, image)
    image = np.array(image, dtype=np.uint16)

    return image

