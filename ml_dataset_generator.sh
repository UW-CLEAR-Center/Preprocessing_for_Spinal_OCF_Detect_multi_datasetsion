#!/bin/bash

# what type of dataset will be processed
dataset_type=test
# number of threads used
num_threads=10
# input spine image directory
spine_images_root=/data/Spine_ML_Data/UW_Spine_Radiograph_Data/ml_data_partition
# the output root
output_root=/data/Spine_ML_Data/UW_Spine_Radiograph_Data/processed_dataset
# number of augmented vbs generated
num_augmentations=10

# get patches
python get_single_patches/main.py --spine_images_dir $spine_images_root/$dataset_type --output_root $output_root --dataset_type $dataset_type --num_threads $num_threads
# black bone conversion
python black_bone_conversion/main.py --input_dir $output_root/$dataset_type --num_threads $num_threads
# augmentation
if [[ $dataset_type == train ]]
then
	python augmentation/main.py --needed_vbs_dir $output_root/$dataset_type --black_bone_converting --num_threads $num_threads --num_augmentations $num_augmentations --spine_images_dir $spine_images_root/$dataset_type
fi
