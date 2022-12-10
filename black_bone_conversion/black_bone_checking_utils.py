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
import sys
import pandas as pd
sys.path.append('augmentation')
from get_corner_points_utils import *
from image_preprocessing import *
import random

"""
transfer the points in the original spine radiograph to the current one
"""
def corner_points_on_patch(image_code, annotations_df, default_rotate=0, default_expand_w=1, default_expand_h=1, square=True):
    image_annotations = annotations_df[annotations_df['Image']==image_code]
    corners_dict = {}
    for _, annotation in image_annotations.iterrows():
        corner_points = adjust_corner_points(annotation)
        corners_dict[annotation.loc['VB']] = corner_points
    corners_on_patch_dict = {}
    for vb in corners_dict:
        # use np.ones to represent the image. It doesn't matter because we only need to know the vb's and patch's corner points
        _, x_index_range, y_index_range, rotated_vb_corners = extract_vb(np.ones((1000, 1000)), corners_dict[vb], default_rotate, default_expand_w, default_expand_h, square=square)
        corners_on_patch = np.array(rotated_vb_corners) - np.array([x_index_range[0], y_index_range[0]])
        corners_on_patch_dict[vb] = corners_on_patch
    return corners_on_patch_dict

def get_endplate_patch(patch, corner_points, expanded_width_ratio=0.05, expanded_height_ratio=0.1, at_least_num_near_pixels=20):
    sup_endplate_points = corner_points[:2]
    inf_endplate_points = corner_points[2:]
    endplates_points = [sup_endplate_points, inf_endplate_points]
    endplates_patches = []
    for endplate_points in endplates_points:
        tightest_bb_x_min = min(endplate_points[0][0], endplate_points[1][0])
        tightest_bb_x_max = max(endplate_points[0][0], endplate_points[1][0])
        tightest_bb_y_min = min(endplate_points[0][1], endplate_points[1][1])
        tightest_bb_y_max = max(endplate_points[0][1], endplate_points[1][1])
        # bb_width = tightest_bb_x_max - tightest_bb_x_min
        # bb_height = tightest_bb_y_max - tightest_bb_y_min
        expanded_width = patch.shape[1] * expanded_width_ratio
        expanded_height = max(patch.shape[0] * expanded_height_ratio, at_least_num_near_pixels)
        new_bb_x_min = max(int(tightest_bb_x_min - expanded_width),  0)
        new_bb_x_max = min(int(tightest_bb_x_max + expanded_width),  patch.shape[1])
        new_bb_y_min = max(int(tightest_bb_y_min - expanded_height), 0)
        new_bb_y_max = min(int(tightest_bb_y_max + expanded_height), patch.shape[0])
        endplate_patch = patch[new_bb_y_min:new_bb_y_max, new_bb_x_min:new_bb_x_max]
        endplates_patches.append(endplate_patch)
    return endplates_patches

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
    if bit_depth == 16:
        image = np.array(image, dtype=np.uint16)
    elif bit_depth == 8:
        image = np.array(image, dtype=np.uint8)
    return image

"""
given an image, detect the horizontal edges
"""
def horizontal_edge_detector(
    image, vb, 
    kernel_size, low_threshold, high_threshold, 
    img_norm_low=0.05, img_norm_high=0.95, 
    erosion_kernel_size=5, erosion_iterations=1, 
    bit_depth=16,
):
    # image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=100)
    ################3
    # plt.figure()
    # plt.title(vb)
    # plt.imshow(image, cmap='gray')
    ################3
    # edges = cv2.Sobel(image, cv2.CV_16U, 0, 1, ksize=kernel_size)
    # print(edges)
    edges = np.abs(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size))
    edges = my_image_normalize(edges, bit_depth, img_norm_low, img_norm_high)
    edges = np.array(edges, dtype=float) / (2 ** bit_depth - 1)
    # erosion
    if erosion_kernel_size != 0:
        erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size))
        edges = cv2.erode(edges, erosion_kernel, iterations=erosion_iterations)

    hyst = skimage.filters.apply_hysteresis_threshold(edges, low=low_threshold, high=high_threshold)
    ################3
    # plt.figure()
    # plt.title(vb)
    # plt.imshow(np.array(hyst, dtype=np.uint), cmap='gray')
    ################3
    return hyst, edges

def black_bone_checking_for_prepocessed_patch(
    vb_patch, vb, 
    kernel_size, low_threshold, high_threshold, 
    one_side_num_pixels_near_edge,
    img_norm_low=0.05, img_norm_high=0.95,
    erosion_kernel_size=5, erosion_iterations=1, 
    bit_depth=16, 
):
    image = np.array(vb_patch, dtype=float)
    # num_pixels_near_edge = max(int(image.shape[0] * num_pixels_near_edge_to_image_size), at_least_num_near_pixels)
    num_pixels_near_edge = one_side_num_pixels_near_edge
    edges_detected_image, edges_gray_image = horizontal_edge_detector(
        image, vb, 
        kernel_size, low_threshold, high_threshold, 
        img_norm_low, img_norm_high,
        erosion_kernel_size, erosion_iterations, 
        bit_depth, 
    )
    img_height, img_width = image.shape
    
    edge_points_row_indices, edge_points_col_indices = np.where(edges_detected_image)
    edge_points_row_indices = np.array(edge_points_row_indices)
    edge_points_col_indices = np.array(edge_points_col_indices)
    steps = np.ones_like(edge_points_row_indices)
    near_edge_rows = steps[:, None] * np.arange(-num_pixels_near_edge, num_pixels_near_edge) + edge_points_row_indices[:, None]
    near_edge_rows = np.maximum(near_edge_rows, 0)
    near_edge_rows = np.minimum(near_edge_rows, img_height - 1)
    near_edge_cols = np.ones_like(near_edge_rows) * edge_points_col_indices[:, None]

    near_edge_pixels = image[near_edge_rows, near_edge_cols]
    edges = image[edge_points_row_indices, edge_points_col_indices]
    min_near_edge_pixels = np.min(near_edge_pixels, axis=1)
    max_near_edge_pixels = np.max(near_edge_pixels, axis=1)
    to_min_distances = edges - min_near_edge_pixels
    to_max_distances = max_near_edge_pixels - edges
    mean_min_distance = np.mean(to_min_distances)
    mean_max_distance = np.mean(to_max_distances)

    if mean_min_distance < mean_max_distance:
        rst = 'black'
    else:
        rst = 'white'

    return rst, mean_min_distance, mean_max_distance, edges_detected_image, edges_gray_image

def output_one_type_images(output_root, which_rst_type, image_codes_list, images_vbs_pixels_dict, images_vbs_rst_dict, output_suffix):
    for image_code in image_codes_list:
        vbs_pixels_dict = images_vbs_pixels_dict[image_code]
        for vb, vb_pixels_list in vbs_pixels_dict.items():
            black_bone_rst = images_vbs_rst_dict[image_code][vb]
            output_vb_dir = os.path.join(output_root, which_rst_type, image_code, vb + '_' + black_bone_rst)
            if not os.path.exists(output_vb_dir):
                os.makedirs(output_vb_dir)
            for i, vb_pixels in enumerate(vb_pixels_list):
                output_path = os.path.join(output_vb_dir, output_suffix + str(i) + '.tiff')
                tiff.imsave(output_path, vb_pixels)


def output_images_for_check(output_root, rst_types_images_dict, images_vbs_pixels_dict, images_vbs_rst_dict, output_suffix):
    for which_rst_type, image_codes_list in rst_types_images_dict.items():
        output_one_type_images(output_root, which_rst_type, image_codes_list, images_vbs_pixels_dict, images_vbs_rst_dict, output_suffix)
    

def read_patch_pixels(vb_path):
    original_img = tiff.imread(vb_path)
    if len(original_img.shape) == 3:
        index = original_img.shape.index(1)
        if index == 0:
            original_img = original_img[0, :, :]
        elif index == 1:
            original_img = original_img[:, 0, :]
        else:
            original_img = original_img[:, :, 0]
    return original_img

def bit_depth_conversion(img, old, new):
    old_max = 2 ** old - 1
    new_max = 2 ** new - 1
    img = np.array(img / old_max * new_max, dtype=np.uint8)
    return img

def patch_preprocessing(
    vb_path, 
    original_bit_depth, 
    converted_bit_depth, 
    resized_vb_patch_size, 
    nl_mean_denoising_h, 
    template_window_size,
    search_window_size
):
    ''' read and get one channel of the original vb patch '''
    original_img = read_patch_pixels(vb_path)
    original_img = bit_depth_conversion(original_img, original_bit_depth, converted_bit_depth)
    ''' resize the original vb patch '''
    resized_img = cv2.resize(original_img, (resized_vb_patch_size, resized_vb_patch_size))
    ''' denoise the original vb patch '''
    img = cv2.fastNlMeansDenoising(
        src=resized_img, dst=None, 
        h=nl_mean_denoising_h, templateWindowSize=template_window_size, searchWindowSize=search_window_size
    )
    return img

def black_bone_checking_for_original_patch(
    vb_path, vb, 
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
):
    ''' preprocessing '''
    img = patch_preprocessing(
        vb_path, 
        original_bit_depth, 
        converted_bit_depth, 
        resized_vb_patch_size, 
        nl_mean_denoising_h, 
        template_window_size,
        search_window_size
    )
    ''' white/black bone checking '''
    rst, _, _, edges_detected_image, edges_gray_image = black_bone_checking_for_prepocessed_patch(
        img, vb, 
        sobel_kernel_size, 
        hyst_low_threshold, 
        hyst_high_threshold, 
        one_side_num_pixels_near_edge,
        img_norm_low, img_norm_high,
        erosion_kernel_size, erosion_iterations,
        converted_bit_depth,
    )
    return rst

def black_bone_checking_for_spine_image(
    vb_paths, 
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
):
    rsts = []
    for vb_path in vb_paths:
        vb = vb_path.split('/')[-1].split('.')[0]
        rst = black_bone_checking_for_original_patch(
            vb_path, vb,
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
        )
        rsts.append(rst)
    ''' from the result of each vb patch, majority vote to get the final results of the whole spine image '''
    num_blacks = rsts.count('black')
    num_whites = rsts.count('white')
    if num_blacks > num_whites:
        final_rst = 'black'
    elif num_blacks < num_whites:
        final_rst = 'white'
    else:
        rand = random.random()
        if rand < 0.5:
            final_rst = 'black'
        else:
            final_rst = 'white'
    return final_rst



