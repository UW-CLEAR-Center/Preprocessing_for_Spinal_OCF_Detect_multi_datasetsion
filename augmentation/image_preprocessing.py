import math
import numpy as np
from vector_ops import *
import imutils
import cv2
from imgaug import augmenters as iaa
# from image_normalization import *

""" This file contains functions of image preprocessing"""

'''
input a array where each row is a coordinate, output a array where each row is a coordinate after bounded rotation, center is the origin
Assume the first four points are image corners
'''
def coords_bounded_rotation(points, degree):
    radian = math.radians(degree)
    M = np.array([[math.cos(radian), -math.sin(radian)],[math.sin(radian), math.cos(radian)]])
    points = np.array(points)
    new_points = M.dot(points.T).T
    xymin = np.amin(new_points[:4], axis=0)
    
    return new_points - np.minimum(np.zeros_like(xymin), xymin)

"""
bounding box translation in an image
x and y are measured in pixels
"""
def bounding_box_translation(image, corners, x, y):
    image = np.array(image.copy())
    h, w = image.shape
    corners = np.array(corners.copy())
    temp_corners = corners + np.array([x, y])
    bb_top = temp_corners[0, 1]
    bb_bottom = temp_corners[3, 1]
    bb_right = temp_corners[0, 0]
    bb_left = temp_corners[1, 0]

    if bb_left < 0:
        num_zero_cols = -bb_left
        padding_zeros = np.zeros((h, num_zero_cols))
        image = np.concatenate([padding_zeros, image], axis=1)
        temp_corners += np.array([num_zero_cols, 0])
    if bb_right > w:
        num_zero_cols = bb_right - w
        padding_zeros = np.zeros((h, num_zero_cols))
        image = np.concatenate([image, padding_zeros], axis=1)
    h, w = image.shape
    if bb_top < 0:
        num_zero_rows = -bb_top
        padding_zeros = np.zeros((num_zero_rows, w))
        image = np.concatenate([padding_zeros, image], axis=0)
        temp_corners += np.array([0, num_zero_rows])
    if bb_bottom > h:
        num_zero_rows = bb_bottom - h
        padding_zeros = np.zeros((num_zero_rows, w))
        image = np.concatenate([image, padding_zeros], axis=0)

    return image, temp_corners


"""
Given four corner points (order: suppost->supant->infant->infpost), extract the corresponding vb
1) rotate 2) scale 3) translate
input: 
    image: the array of image pixels
    corners: four corners [[],[],[],[]]
    rotate: the vb should be clockwisely rotate by this degree
    expand_w: the bounding box expand along width  by this
    expand_h: the bounding box expand along height by this
    translate_w: the bounding box should translate along width in the bounxing box
    with this percent of the width of bounding box
    translate_h: the bounding box should translate along hight in the bounxing box
    with this percent of the height of bounding box
    square: force the bounding box to be a square
output:
    vb: the array of the vb pixels
"""
def extract_vb(image, corners, rotate=0, expand_w=0.1, expand_h=0.1, translate_x=0, translate_y=0, square=True):
    if square:
        expand_h = expand_w
    suppost, supant, infant, infpost = corners[0], corners[1], corners[2], corners[3]
    image = np.array(image.copy())
    h, w = image.shape
    image_corners = [[w,0],[0,0],[0,h],[w,h]]

    # connect diagonals
    infant_suppost = get_vector(infant, suppost)
    supant_infpost = get_vector(supant, infpost)
    
    # get the bisector of the angle between two diagonals
    bisector = get_bisector_vector(infant_suppost, supant_infpost) 

    # get the angle between the bisector and the x-axis
    angle = math.degrees(math.atan2(-bisector[1], bisector[0]))

    # get the total angle that need to be rotate
    total_rotate = rotate + angle

    # rotate the image by total rotate, also document the position of the corners of image and vb
    rotated_image = imutils.rotate_bound(image, total_rotate)
    all_corners = np.concatenate([image_corners, corners], axis = 0)
    rotated_all_corners = coords_bounded_rotation(all_corners, total_rotate)
    rotated_image_corners = rotated_all_corners[:4]
    rotated_vb_corners = rotated_all_corners[4:]

    # get the tightest bounding box
    x_index_range, y_index_range = get_tightest_bounding_box(rotated_vb_corners)
    ## force the bounding box to be a square
    ## the side of the square should be the larger one
    if square:
        vb_width = x_index_range[1] - x_index_range[0]
        vb_height = y_index_range[1] - y_index_range[0]
        max_side = max(vb_width, vb_height)
        expanded_w_value = (max_side - vb_width) / 2
        expanded_h_value = (max_side - vb_height) / 2
        x_index_range[0] -= expanded_w_value
        x_index_range[1] += expanded_w_value
        y_index_range[0] -= expanded_h_value
        y_index_range[1] += expanded_h_value
        x_index_range[0] = int(x_index_range[0])
        x_index_range[1] = int(x_index_range[1])
        y_index_range[0] = int(y_index_range[0])
        y_index_range[1] = int(y_index_range[1])

    # expand the bounding box and translate
    vb_width = x_index_range[1] - x_index_range[0]
    vb_height = y_index_range[1] - y_index_range[0]
    expanded_vb_width = vb_width*(1 + expand_w)
    expanded_vb_height = vb_height*(1 + expand_h)
    expanded_w_value = (expanded_vb_width - vb_width) / 2
    expanded_h_value = (expanded_vb_height - vb_height) / 2
    translate_x_pixels = expanded_vb_width * translate_x
    translate_y_pixels = expanded_vb_height * translate_y
    x_index_range[0] -= (expanded_w_value - translate_x_pixels)
    x_index_range[1] += (expanded_w_value + translate_x_pixels)
    y_index_range[0] -= (expanded_h_value - translate_y_pixels)
    y_index_range[1] += (expanded_h_value + translate_y_pixels)
    x_index_range[0] = int(x_index_range[0])
    x_index_range[1] = int(x_index_range[1])
    y_index_range[0] = int(y_index_range[0])
    y_index_range[1] = int(y_index_range[1])

    # check if all vertices of the bounding box is inside the imagel; if not, current zero-padding
    zero_padding_left = -min(0, x_index_range[0])
    x_index_range[0] = max(0, x_index_range[0])
    zero_padding_right = -min(0, rotated_image.shape[1]- x_index_range[1])
    x_index_range[1] = min(rotated_image.shape[1], x_index_range[1])
    zero_padding_up = -min(0, y_index_range[0])
    y_index_range[0] = max(0, y_index_range[0])
    zero_padding_down = -min(0, rotated_image.shape[0]- y_index_range[1])
    y_index_range[1] = min(rotated_image.shape[0], y_index_range[1])

    zero_padding_left_mat = np.zeros([rotated_image.shape[0], zero_padding_left])
    zero_padding_right_mat = np.zeros([rotated_image.shape[0], zero_padding_right])
    zero_padding_up_mat = np.zeros([zero_padding_up, rotated_image.shape[1]+zero_padding_left+zero_padding_right])
    zero_padding_down_mat = np.zeros([zero_padding_down, rotated_image.shape[1]+zero_padding_left+zero_padding_right])

    padded_image = np.concatenate([zero_padding_left_mat, rotated_image, zero_padding_right_mat], axis=1)
    padded_image = np.concatenate([zero_padding_up_mat, padded_image, zero_padding_down_mat], axis=0)

    # extract and return the vb pixel matrix
    vb = rotated_image[y_index_range[0]:y_index_range[1], x_index_range[0]:x_index_range[1]]

    return vb, x_index_range, y_index_range, rotated_vb_corners


"""
extract all vb from an image
input: 
    image: the array of image pixels
    corners_dict: a dict whose key is vb, value is the four corners [[],[],[],[]]
    rotate: the vbs should be clockwisely rotate by this degree, can be a scale or a list
    expand_w: the bounding box expand along width  by this, can be a scaler or a list
    expand_h: the bounding box expand along height by this, can be a scaler or a list
output:
    vbi_dict: a dict whose key is the vb name, value is the array of the vb pixels
"""
def extract_spine_vbs(image, corners_dict, rotate=0, expand_w=0.1, expand_h=0.1, square=True):
    rotate = np.array(rotate) * np.ones(len(corners_dict))
    expand_w = np.array(expand_w) * np.ones(len(corners_dict))
    expand_h = np.array(expand_h) * np.ones(len(corners_dict))
    
    vb_dict = {}
    for i, vb_name in enumerate(corners_dict):
        vb_pixels, _, _, _ = extract_vb(image, corners_dict[vb_name], rotate[i], expand_w[i], expand_h[i], square=square)
        vb_dict[vb_name] = vb_pixels

    return vb_dict

def affine_vb_augmentation(image, num_augmentations, corners_dict, rotate_range, expand_w_range, expand_h_range, translate_x_range, translate_y_range, square=True):
    vb_dict = {}
    for i, vb_name in enumerate(corners_dict):
        vb_dict[vb_name] = []
        for j in range(num_augmentations):
            rotate = np.random.uniform(low=rotate_range[0], high=rotate_range[1])
            expand_w = np.random.uniform(low=expand_w_range[0], high=expand_w_range[1])
            expand_h = np.random.uniform(low=expand_h_range[0], high=expand_h_range[1])
            translate_x = np.random.uniform(low=translate_x_range[0], high=translate_x_range[1])
            translate_y = np.random.uniform(low=translate_y_range[0], high=translate_y_range[1])
            vb_pixels, _, _, _ = extract_vb(image, corners_dict[vb_name], rotate, expand_w, expand_h, translate_x, translate_y, square=square)
            vb_dict[vb_name].append(vb_pixels)

    return vb_dict

"""
do brightness adjustment, contrast adjustment, gaussian blurring/sharpening, and gaussian noise adding to on vb patch
input:
    vb_patches: pixel information of one vb patches num_batches x patch_rows x patch_cols
    brightness_coeff_range: the range of the coefficient used for brightness adjustment
    contrast_coeff_range: the range of the coefficient used for contrast adjustment
    blur_coeff_range: the range of the coefficient for or gaussian blurring
    sharpen_coeff_range: the range of the coefficient for or shapenning
    noise_coeff_range: the range of the coefficient for gaussian noise adding
    bit_depth: the bit depth of the vb patch
"""
def vb_patch_augmentation(input_vb_patches,
                          deformation_coeff_range,
                          brightness_coeff_range,
                          contrast_coeff_range,
                          blur_coeff_range,
                          sharpen_coeff_range,
                          noise_coeff_range,
                          bit_depth=16,
                          gray_inverse_prob=0,
                          horizontal_flip_prob=0,
                          vertical_flip_prob=0,
                          black_bone_converting=True,
                          bone_gray=None,
                          image_normalizing=True,
                          **kwargs
                         ):
    # cutoff = (0.5*brightness_coeff_range[0], 0.5*brightness_coeff_range[1])
    vb_patches = []
    for vb_pixel in input_vb_patches:
        if black_bone_converting:
            if bone_gray is not None and bone_gray == 'black':
                vb_pixel = 2**bit_depth - 1 - vb_pixel
        if image_normalizing:
            image_norm_low = kwargs['image_norm_low']
            image_norm_high = kwargs['image_norm_high']
            vb_pixel = my_image_normalize(vb_pixel, bit_depth, image_norm_low, image_norm_high)
        vb_patches.append(vb_pixel)
    cutoff = brightness_coeff_range
    augmentation = iaa.Sequential([
        # iaa.Multiply(brightness_coeff_range),
        # iaa.GammaContrast(contrast_coeff_range),
        # iaa.PiecewiseAffine(scale=deformation_coeff_range),
        iaa.SigmoidContrast(gain=contrast_coeff_range, cutoff=cutoff),
        # iaa.OneOf([ ## blur or sharpen
           # iaa.GaussianBlur(sigma=blur_coeff_range),
           # iaa.Sharpen(alpha=sharpen_coeff_range),
        iaa.AdditiveGaussianNoise(loc=0, scale=noise_coeff_range)
                    # ]),
    ])
    if bit_depth == 16:
        dtype = np.uint16
    elif bit_depth == 8:
        dtype = np.uint8
    aug_patches = []
    for patch in vb_patches:
        patch_ = np.array([patch], dtype=dtype)
        aug_patches.append(augmentation(images=patch_))
    aug_patches1 = []
    for patch in aug_patches:
        if np.random.uniform() > gray_inverse_prob:
            aug_patches1.append(patch)
        else:
            aug_patches1.append(2**bit_depth - 1 - patch)
    aug_patches2 = []
    for patch in aug_patches1:
        if np.random.uniform() > horizontal_flip_prob:
            aug_patches2.append(patch)
        else:
            aug_patches2.append(np.flip(patch, axis=2))
    aug_patches3 = []
    for patch in aug_patches2:
        if np.random.uniform() > vertical_flip_prob:
            aug_patches3.append(patch)
        else:
            aug_patches3.append(np.flip(patch, axis=1))
    return aug_patches3

"""
resize image
"""
def image_resize(image, width, height):
    y_len, x_len = image.shape
    x_scale = width/x_len
    y_scale = height/y_len
    resized_img = cv2.resize(image ,None,fx=x_scale, fy=y_scale)
    return resized_img

"""
regulate the bit-depth
"""
def regulate_bit_depth(image, bit_depth):
    dtype = image.dtype
    max_pixel = np.max(image)
    current_bit = math.ceil(np.log2(max_pixel + 1))
    potential_max_pixel = np.power(2, current_bit) - 1
    regulated_max_pixel = np.power(2, bit_depth) - 1

    return np.array(image / potential_max_pixel * regulated_max_pixel, dtype=dtype)
