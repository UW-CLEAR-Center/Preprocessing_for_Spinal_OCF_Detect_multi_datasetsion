from sklearn.model_selection import KFold
import sys
sys.path.append('augmentation')
from get_corner_points_utils import *
from image_preprocessing import *
from black_bone_checking_utils import *

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


def black_bone_checking_for_multi_images(
    input_image_vbs_dict,
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
    output_dir
):
    rst_dict = {}
    num_tasks = len(input_image_vbs_dict)
    co = 0
    for spine_image, vb_paths in input_image_vbs_dict.items():
        print('thread {}: {} out of {} starts: {}'.format(thread_num, co, num_tasks, spine_image))
        final_rst = black_bone_checking_for_spine_image(
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
        )
        rst_dict[spine_image] = final_rst
        co += 1
    for image, black_white_rst in rst_dict.items():
        for vb_path in input_image_vbs_dict[image]:
            pixels = tiff.imread(vb_path)
            if black_white_rst == 'black':
                pixels = 2 ** original_bit_depth - 1 - pixels
            vb = vb_path.split('/')[-1]
            output_dir_ = os.path.join(output_dir, image)
            if not os.path.exists(output_dir_):
                os.makedirs(output_dir_)
            output_path = os.path.join(output_dir_, vb)
            tiff.imsave(output_path, pixels)
    return rst_dict
