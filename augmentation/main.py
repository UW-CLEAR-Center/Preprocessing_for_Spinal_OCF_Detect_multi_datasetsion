import os
from os.path import exists, join
import shutil
import SimpleITK as sitk
import tifffile as tiff
import csv
import json
from image_preprocessing import *
import numpy as np
from image_normalization import *
import sys
import pandas as pd
import threading
from sklearn.model_selection import KFold
import argparse
from get_corner_points_utils import *

os.nice(5)

""" This file is used for image augmentation """
parser = argparse.ArgumentParser()
parser.add_argument('--spine_images_dir', type=str)
parser.add_argument('--needed_vbs_dir', type=str)
parser.add_argument('--black_bone_converting', action='store_true')
parser.add_argument('--num_threads', type=int, default=10)
parser.add_argument('--num_augmentations', type=int, default=10)
args = parser.parse_args()

# ###################### parameter ###########################
needed_vbs_dir = args.needed_vbs_dir# the dir used for reference
num_threads = args.num_threads
black_bone_converting = args.black_bone_converting
num_augmentations = args.num_augmentations # number of augmented images
# needed_vbs_dir = '/home/qfdong/raship/uw_mabq_machine_learning/dataset_preprocessing/get_single_patches/test/train'
# black_bone_converting = True
# num_threads = 3
# num_augmentations = 5
image_normalizing = False
############################################################

if black_bone_converting and image_normalizing:
    save_dir = os.path.join(needed_vbs_dir, 'black_bone_converted_image_norm_augmented') # vb saved dir
elif black_bone_converting:
    save_dir = os.path.join(needed_vbs_dir, 'black_bone_converted_augmented') # vb saved dir
elif image_normalizing:
    save_dir = os.path.join(needed_vbs_dir, 'image_norm_augmented') # vb saved dir
else:
    save_dir = os.path.join(needed_vbs_dir, 'augmented') # vb saved dir

needed_black_bone_detection_rst_path = os.path.join(needed_vbs_dir, 'black_bone_converted', 'black_white_bone_rsts.csv')
black_bone_detected_rsts_df = pd.read_csv(needed_black_bone_detection_rst_path)

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)

# image_dir = '/data/Spine_ML_Data/UW_Spine_Radiograph_Data/ml_data_partition' # the directory contains all images
bit_depth = 16 # the bit depth we want the image to be
default_rotate = 0 # the base degree we want the vb to rotate
default_expand_w = 1 # basic expanding of the width of the bounding box
default_expand_h = default_expand_w # basic expanding the height of the bounding box
square = True

# annotations of all images
# only training set needs to be augmented
annotation_dir = '/home/qfdong/raship/uw_mabq_machine_learning/dataset_preprocessing/annotation_files'
# only train set needs to be augmented
annotation_file = 'train.csv'
annotation_path = os.path.join(annotation_dir, annotation_file)
annotation_df = pd.read_csv(annotation_path)
# only train set needs to be augmented
spine_orientation_annotation_file = 'train_orientations.csv'
spine_orientation_annotation_path = os.path.join(annotation_dir, spine_orientation_annotation_file)
spine_orientation_annotation_df = pd.read_csv(spine_orientation_annotation_path)

# image preprocessing related
image_norm_low = 0.05
image_norm_high = 0.95

# data augmentation related
rotate_range = (-5, 5) # the range of rotation degree
scale_x_range = (0.9, 1.1) # the range of the vb scaling along x-axis
scale_y_range = scale_x_range # the range of the vb scaling along y-axis, as we keep the vb patch as a square, this attribute won't have effects
translate_x_range = (-0.05, 0.05) # the range of the vb translation in a bounding box along x-axis
translate_y_range = translate_x_range # the range of the vb translation in a bounding box along y-axi
deformation_coeff_range = (0, 0) # the range of the vb deformation
brightness_coeff_range = (0.4, 0.6) # the range of brightness adjustment
contrast_coeff_range = (4, 8) # the range of contrast adjustment
blur_coeff_range = (0, 0) # the range of Gaussian blurring
sharpen_coeff_range = (0, 0) # the range of sharpening
noise_coeff_range = (0, 6e-4 * (2**bit_depth - 1)) # the range of noise adding
gray_inverse_prob = 0 # the probability that the augmented patch will invert gray
horizontal_flip_prob = 0 # the probabability that the augmented patch will be flipped horizontally
vertical_flip_prob = 0 # the probabability that the augmented patch will be flipped vertically

def _vb_scale_to_bb_expand(vb_scaling, default_expanding):
    return 1 / vb_scaling * (default_expanding + 1) - 1
def _vb_scale_tuple_to_bb_expand_tuple(vb_scalings, default_expanding):
    return (_vb_scale_to_bb_expand(vb_scalings[1], default_expanding),
        _vb_scale_to_bb_expand(vb_scalings[0], default_expanding))
expand_w_range = _vb_scale_tuple_to_bb_expand_tuple(scale_x_range, default_expand_w)
expand_h_range = _vb_scale_tuple_to_bb_expand_tuple(scale_y_range, default_expand_h)

# store all of papameters into a json file
param_dict = {
    'num_augmentations': num_augmentations, 
    'black_bone_converting': black_bone_converting,
    'default_rotate': default_rotate,
    'default_expand_w': default_expand_w,
    'default_expand_h': default_expand_h,
    'square': square,
    'image_normalizing': image_normalizing,
    'image_norm_low': image_norm_low,
    'image_norm_high': image_norm_high,
    'rotate_range': rotate_range,
    'scale_x_range': scale_x_range,
    'scale_y_range': scale_y_range,
    'translate_x_range': translate_x_range,
    'translate_y_range': translate_y_range,
    'deformation_coeff_range': deformation_coeff_range,
    'brightness_coeff_range': brightness_coeff_range,
    'contrast_coeff_range': contrast_coeff_range,
    'blur_coeff_range': blur_coeff_range,
    'sharpen_coeff_range': sharpen_coeff_range,
    'noise_coeff_range': noise_coeff_range,
    'gray_inverse_prob': gray_inverse_prob,
    'horizontal_flip_prob': horizontal_flip_prob,
    'vertical_flip_prob': vertical_flip_prob,
}
param_output_path = os.path.join(save_dir, 'params.json')
with open(param_output_path, 'w') as f:
    json.dump(param_dict, f)


"""
get all vbs needed augmentation
"""
# the filenames of the targeted vbs
new_dirs = []
needed_image_info_dict = {}
train_images = set()
needed_images_subdir = os.path.join(needed_vbs_dir, 'original_data')
for root, dirs, files in os.walk(needed_images_subdir):
    folders_in_path = root.split('/')
    if folders_in_path[-2] == 'original_data':
        image = folders_in_path[-1]
        if image in train_images:
            continue
        train_images.add(image)
        image_path = os.path.join(args.spine_images_dir, image + '.dcm')
        needed_image_info_dict[image] = {}
        needed_image_info_dict[image]['output_dir'] = join(save_dir, image)
        needed_image_info_dict[image]['image_path'] = image_path
        needed_image_info_dict[image]['original_vbs_dir'] = root
        needed_image_info_dict[image]['original_vbs_filenames'] = []
        for f in files:
            needed_image_info_dict[image]['original_vbs_filenames'].append(f)

def split_task(task_dict, num):
    if num == 1:
        return [task_dict]
    num_cases = len(task_dict)
    kf = KFold(n_splits=num)
    k_folds_indices = kf.split(range(num_cases))
    num_subcases = []
    for _, index in k_folds_indices:
        num_subcases.append(len(index))
    subtasks = []
    co = 0
    i = 0
    subtask_dict = {}
    for key in task_dict:
        if co >= num_subcases[i]:
            subtasks.append(subtask_dict)
            co = 0
            i += 1
            subtask_dict = {}
        subtask_dict[key] = task_dict[key]
        co += 1
    subtasks.append(subtask_dict)
    return subtasks

def data_augmentation_func(needed_image_info_dict, thread_seq_num):
    num_all_images = len(needed_image_info_dict)
    for kkk, image_id in enumerate(needed_image_info_dict):
        print('thread {}: {} out of {} starts: {}'.format(thread_seq_num, kkk, num_all_images, image_id))
        image_info_dict = needed_image_info_dict[image_id]
        orientation = spine_orientation_annotation_df[spine_orientation_annotation_df['Image']==image_id]['Orientation']
        orientation = orientation.reset_index(drop=True)
        orientation = orientation.iloc[0]
        image_file = image_info_dict['image_path']
        img = sitk.ReadImage(image_file)[:,:,0]
        npa = sitk.GetArrayViewFromImage(img)
        h, w = npa.shape
        if orientation == 'Faces User Right':
            npa = np.flip(npa, axis=1)
        # make images to the same bit_depth
        npa = regulate_bit_depth(npa, bit_depth)

        '''
        get the point annotation
        '''
        image_annotations = annotation_df[annotation_df['Image']==image_id]
        corners_dict = {}
        for _, annotation in image_annotations.iterrows():
            corner_points = adjust_corner_points(annotation)
            corners_dict[annotation.loc['VB']] = corner_points

        '''
        extract the vertebral bodies
        '''
        vb_pixel_dict = extract_spine_vbs(npa, corners_dict, default_rotate, default_expand_w, default_expand_h, square=square)

        '''
        black bone converting on the original vb pixels and output them
        '''
        vb_filenames = image_info_dict['original_vbs_filenames']
        vb_dir = image_info_dict['original_vbs_dir']
        if black_bone_converting:
            bone_gray = list(black_bone_detected_rsts_df[black_bone_detected_rsts_df['image']==image_id]['rst'])[0]
        else:
            bone_gray = None
        for vb_code, vb_pixel in vb_pixel_dict.items():
            if black_bone_converting and bone_gray == 'black':
                vb_pixel = 2**bit_depth - 1 - vb_pixel
            if image_normalizing:
                vb_pixel = my_image_normalize(vb_pixel, bit_depth, image_norm_low, image_norm_high)
            current_save_dir = os.path.join(image_info_dict['output_dir'], vb_code)
            if not exists(current_save_dir):
                os.makedirs(current_save_dir)
            output_filename = vb_code + '_nonaugmented.tiff'
            save_path = os.path.join(current_save_dir, output_filename)
            tiff.imsave(save_path, vb_pixel)

        """
        do brightness adjustment, contrast adjustment, gaussian blurring/sharpening, and gaussian noise adding to on vb patch
        """
        vb_augment_dict = affine_vb_augmentation(
            npa, 
            num_augmentations, 
            corners_dict, 
            rotate_range, 
            expand_w_range, 
            expand_h_range, 
            translate_x_range, 
            translate_y_range, 
            square=square
        )
        for vb, vb_augment in vb_augment_dict.items():
            aug_vbs = vb_patch_augmentation(
                vb_augment,
                deformation_coeff_range,
                brightness_coeff_range,
                contrast_coeff_range,
                blur_coeff_range,
                sharpen_coeff_range,
                noise_coeff_range,
                bit_depth=bit_depth,
                gray_inverse_prob=gray_inverse_prob,
                horizontal_flip_prob=horizontal_flip_prob,
                vertical_flip_prob=vertical_flip_prob,
                black_bone_converting=black_bone_converting,
                bone_gray=bone_gray,
                image_normalizing=image_normalizing,
                image_norm_low=image_norm_low,
                image_norm_high=image_norm_high
            )
            for jj, aug in enumerate(aug_vbs):
                current_save_dir = os.path.join(image_info_dict['output_dir'], vb)
                output_vb_image_name = vb + '_augmentation' + str(jj) + '.tiff'
                save_path = join(current_save_dir, output_vb_image_name)
                tiff.imsave(save_path, aug)

# data augmentation via multithreads
num_threads = min(num_threads, len(needed_image_info_dict))
needed_image_info_subdicts_list = split_task(needed_image_info_dict, num=num_threads)
threads = []
for i, subdict in enumerate(needed_image_info_subdicts_list):
    thread = threading.Thread(target=data_augmentation_func, args=(subdict, i))
    threads.append(thread)
    thread.start()
for t in threads:
    t.join()
