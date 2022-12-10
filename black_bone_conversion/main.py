import os
import argparse
import shutil
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
import sys
import pandas as pd
from skimage.restoration import denoise_nl_means, estimate_sigma
import random
import concurrent.futures
from my_multithreading import *
sys.path.append('augmentation')
from get_corner_points_utils import *
from image_preprocessing import *

os.nice(5)

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str)
parser.add_argument('--num_threads', type=int, default=10)
args = parser.parse_args()

""" input and output dir """
input_dir = os.path.join(args.input_dir, 'original_data')
input_subdirs = []
for root, dirs, files in os.walk(input_dir):
    if len(files) > 0 and files[0].endswith('.tiff'):
        input_subdir = '/'.join(root.split('/')[:-1])
        if input_subdir not in input_subdirs:
            input_subdirs.append(input_subdir)
output_dir = os.path.join(args.input_dir, 'black_bone_converted')
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
output_subdirs = []
for dir_ in input_subdirs:
    paths = dir_.split('/')
    indices = [i for i, x in enumerate(paths) if x == 'original_data']
    paths[indices[-1]] = 'black_bone_converted'
    output_subdir = '/'.join(paths)
    os.makedirs(output_subdir)
    output_subdirs.append(output_subdir)

""" hyperparameters """
# vb patch resize
resized_vb_patch_size = 505
# vb patch denoising
nl_mean_denoising_h = 8
template_window_size = 5
search_window_size = 25
# edge detection
sobel_kernel_size = 7
hyst_low_threshold = 0.5937988322
hyst_high_threshold = 0.7653667783
# image normalization on the edge gradient (gray) image
img_norm_low = 0.2906257959
img_norm_high = 0.8887703311
# erosion the edge gradient (gray) image
erosion_kernel_size = 3
erosion_iterations = 1
# related to the vertical line on the detent edge pixel
one_side_num_pixels_near_edge = 37

""" parameters that needs no change, most related to input """
# bit depth of the we trasfer the vb patch to to get further processed
original_bit_depth = 16
converted_bit_depth = 8

all_images_rst_dict = {}
for ii, spine_image_input_dir in enumerate(input_subdirs):
    """ get input vb patches and ground truth """
    spine_images = os.listdir(spine_image_input_dir)
    input_image_vbs_dict = {}
    for spine_image in spine_images:
        input_subdir = os.path.join(spine_image_input_dir, spine_image)
        input_image_vbs_dict[spine_image] = []
        for _, _, files in os.walk(input_subdir):
            for f in files:
                if f.endswith('.tiff'):
                    input_path = os.path.join(input_subdir, f)
                    input_image_vbs_dict[spine_image].append(input_path)

    num_threads = min(args.num_threads, len(spine_images))
    input_image_vbs_subdicts = split_task(input_image_vbs_dict, num_threads)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for thread_num, subdict in enumerate(input_image_vbs_subdicts):
            future = executor.submit(
                black_bone_checking_for_multi_images,
                subdict,
                resized_vb_patch_size,
                nl_mean_denoising_h,
                template_window_size,
                search_window_size,
                sobel_kernel_size,
                hyst_low_threshold,
                hyst_high_threshold,
                img_norm_low,
                img_norm_high,
                erosion_kernel_size,
                erosion_iterations,
                one_side_num_pixels_near_edge,
                original_bit_depth,
                converted_bit_depth,
                thread_num,
                output_subdirs[ii]
            )
            futures.append(future)
        images_rst_dict = {}
        for future in futures:
            dict_ = future.result()
            images_rst_dict = {**images_rst_dict, **dict_}
    all_images_rst_dict = {**all_images_rst_dict, **images_rst_dict}
df_dict = {'image': [], 'rst': []}
for image, rst in all_images_rst_dict.items():
    df_dict['image'].append(image)
    df_dict['rst'].append(rst)
output_df = pd.DataFrame(data=df_dict)
df_output_path = os.path.join(output_dir, 'black_white_bone_rsts.csv')
output_df.to_csv(df_output_path, index=False)

