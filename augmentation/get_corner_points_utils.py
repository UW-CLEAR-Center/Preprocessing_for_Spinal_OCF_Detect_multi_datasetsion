import numpy as np
import math

def obtain_corner_points(annotation_df):
    corner_points_headers = [
        ['SupPost X', 'SupPost Y'],
        ['SupAnt X', 'SupAnt Y'],
        ['InfAnt X', 'InfAnt Y'],
        ['InfPost X', 'InfPost Y']
    ]
    points = []
    for headers in corner_points_headers:
        point = []
        for header in headers:
            point.append(annotation_df.loc[header])
        points.append(point)
    return points

def intersection_of_two_lines(p1, p2, p3, p4):
    if p3[0] != p1[0] and p4[0] != p2[0]:
        k1 = (p3[1] - p1[1]) / (p3[0] - p1[0])
        k2 = (p4[1] - p2[1]) / (p4[0] - p2[0])
        rst_x = (k2 * p2[0] - k1 * p1[0] + p1[1] - p2[1]) / (k2 - k1)
        rst_y = k1 * (rst_x - p1[0]) + p1[1]
    elif p3[0] == p1[0] and p4[0] != p2[0]:
        k2 = (p4[1] - p2[1]) / (p4[0] - p2[0])
        rst_x = p3[0] 
        rst_y = k2 * (rst_x - p2[0]) + p2[1]
    elif p3[0] != p1[0] and p4[0] == p2[0]:
        k1 = (p3[1] - p1[1]) / (p3[0] - p1[0])
        rst_x = p4[0] 
        rst_y = k1 * (rst_x - p1[0]) + p1[1]
    else:
        print('Two diagonals cannot be parallel. Something goes wrong!')
        assert False
    return [rst_x, rst_y]

def check_corner_points_annotation_direction(corner_points):
    p1 = corner_points[0]
    p2 = corner_points[1]
    p3 = corner_points[2]
    p4 = corner_points[3]
    intersect = intersection_of_two_lines(p1, p2, p3, p4)
    p1_ = np.array(p1) - np.array(intersect)
    p2_ = np.array(p2) - np.array(intersect)
    theta1 = math.atan2(p1_[0], p1_[1])
    if theta1 < 0:
        theta1 += 2 * math.pi
    theta2 = math.atan2(p2_[0], p2_[1])
    if theta2 < 0:
        theta2 += 2 * math.pi
    if theta2 > theta1:
        return 'clockwise'
    return 'counterclockwise'


def adjust_corner_points(annotation_df):
    corner_points = obtain_corner_points(annotation_df)
    direction = check_corner_points_annotation_direction(corner_points)
    if direction == 'clockwise':
        corner_points.reverse()
    return corner_points
