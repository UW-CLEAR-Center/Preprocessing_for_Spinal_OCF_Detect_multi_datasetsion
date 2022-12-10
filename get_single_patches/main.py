import pandas as pd
import os
import SimpleITK as sitk
import math
import shutil
import tifffile as tiff
import sys
from sklearn.model_selection import KFold
import argparse
import threading
sys.path.append('augmentation')
from image_preprocessing import *
from get_corner_points_utils import *
import json

os.nice(5)

""" This file is used for image augmentation """
parser = argparse.ArgumentParser()
parser.add_argument('--spine_images_dir', type=str)
parser.add_argument('--output_root', type=str)
parser.add_argument('--dataset_type', type=str)
parser.add_argument('--num_threads', type=int, default=10)
args = parser.parse_args()

output_root = args.output_root
dataset_type = args.dataset_type
images_dir = args.spine_images_dir
num_threads = args.num_threads
# output_root = '/data/Spine_ML_Data/UW_Spine_Radiograph_Data/processed_dataset/'
# dataset_type = 'train'
# # images_dir = 'test_images'
# images_dir = '/data/Spine_ML_Data/UW_Spine_Radiograph_Data/ml_data_partition/train_set'
# num_threads = 10

annotation_filename = dataset_type + '.csv'
test_mode = False
output_dir = os.path.join(output_root, dataset_type, 'original_data')
bit_depth = 16
default_rotate = 0
default_expand_w = 1
default_expand_h = default_expand_w
square = True

# read the annotations
annotation_file_dir = '/home/qfdong/raship/uw_mabq_machine_learning/dataset_preprocessing/annotation_files'
annotation_file_path = os.path.join(annotation_file_dir, annotation_filename)
annotations_df = pd.read_csv(annotation_file_path)
# print(annotations_df)

# get the orientation annotations
orientation_annotation_filename = dataset_type + '_orientations.csv'
orientation_annotation_file_path = os.path.join(annotation_file_dir, orientation_annotation_filename)
orientation_annotation_df = pd.read_csv(orientation_annotation_file_path)
# print(orientation_annotation_df)

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)


# get the image pixels
image_filenames = []
for root, dirs, files in os.walk(images_dir):
    for f in files:
        if f.endswith('.dcm'):
            image_filenames.append(f)

def split_tasks(task_list, num):
    if num == 1:
        return [task_list]
    num_cases = len(task_list)
    kf = KFold(n_splits=num)
    k_folds_indices = kf.split(range(num_cases))
    num_subcases = []
    for _, index in k_folds_indices:
        num_subcases.append(len(index))
    subtasks = []
    co = 0
    i = 0
    subtask_list = []
    for task in task_list:
        if co >= num_subcases[i]:
            subtasks.append(subtask_list)
            co = 0
            i += 1
            subtask_list = []
        subtask_list.append(task)
        co += 1
    subtasks.append(subtask_list)
    return subtasks

def patch_extracting_func(image_filenames, thread_num, test_mode=False):
    num_tasks = len(image_filenames)
    co = 0
    error_images = []
    for image_filename in image_filenames:
        try:
            print('thread {}: {} out of {} starts: {}'.format(thread_num, co, num_tasks, image_filename))
            co += 1
            image_code = image_filename.split('.')[0]
            orientation = orientation_annotation_df[orientation_annotation_df['Image']==image_code]['Orientation']
            orientation = orientation.reset_index(drop=True)
            if len(orientation) == 0:
                # unreadable images are not stored in the clean-up annotation file but may be still in the folder
                print('unreadable image')
                continue
            orientation = orientation.iloc[0]
            image_path = os.path.join(images_dir, image_filename)
            img = sitk.ReadImage(image_path)[:,:,0]
            npa = sitk.GetArrayViewFromImage(img)
            npa = regulate_bit_depth(npa, bit_depth)
            if orientation == 'Faces User Right':
                npa = np.flip(npa, axis=1)

            # get the annotations of this image
            image_annotations = annotations_df[annotations_df['Image']==image_code]
            corners_dict = {}
            for _, annotation in image_annotations.iterrows():
                corner_points = adjust_corner_points(annotation)
                corners_dict[annotation.loc['VB']] = corner_points

            # extract the vertebral bodies
            extracted_vbs = extract_spine_vbs(npa, corners_dict, default_rotate, default_expand_w, default_expand_h, square=square)

            # output the extracted patches
            if not test_mode:
                output_subdir = os.path.join(output_dir, image_code)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                for vb_name, vb_pixel in extracted_vbs.items():
                    output_path = os.path.join(output_subdir, vb_name + '.tiff')
                    tiff.imsave(output_path, vb_pixel)
        except:
            error_images.append(image_filename)
            print('error!!!!!!!')
    return error_images

if test_mode:
    error_images = patch_extracting_func(image_filenames, 1, test_mode)
    print(error_images)
    assert False
    with open('error_images.json', 'w') as f:
        json.dump(error_images, f)
else:
    num_threads = min(num_threads, len(image_filenames))
    image_filenames_sublists = split_tasks(image_filenames, num_threads)
    threads = []
    for i, sublist in enumerate(image_filenames_sublists):
        thread = threading.Thread(target=patch_extracting_func, args=(sublist, i))
        threads.append(thread)
        thread.start()
    for t in threads:
        t.join()
