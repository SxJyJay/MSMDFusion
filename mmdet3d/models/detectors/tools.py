import numpy as np
import ipdb
import math

def rotate_box(center_x, center_y, x_size, y_size, box_rot):
    x1 = int(center_x - x_size/2)
    y1 = int(center_y - y_size/2)
    x2 = int(center_x + x_size/2)
    y2 = int(center_y + y_size/2)
    # 逆时针旋转
    box_rot = -box_rot
    cos_theta = np.cos(box_rot)
    sin_theta = np.sin(box_rot)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    center_point = np.array([center_x, center_y])
    corner_points = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
    rotated_corner_points = np.dot(corner_points - center_point, rotation_matrix) + center_point
    rotated_corner_points = np.around(rotated_corner_points)

    return rotated_corner_points
    

def get_feats_in_rectangle(rotated_corner_points, featmap):
    featmap = featmap.numpy()
    min_x = max(min(rotated_corner_points[:, 0]), 0)
    max_x = min(max(rotated_corner_points[:, 0]), 179)
    min_y = max(min(rotated_corner_points[:, 1]), 0)
    max_y = min(max(rotated_corner_points[:, 1]), 179)
    min_y = min(min_y, 179)
    min_x = min(min_x, 179)
    
    feats = []
    for x in range(int(min_x), int(max_x+1)):
        # if x >= 180 or x < 0:
        #     continue
        for y in range(int(min_y), int(max_y+1)):
            # if y >= 180 or y < 0:
            #     continue
            if is_point_in_rectangle((x, y), rotated_corner_points):
                feats.append(featmap[0, :, y, x])
    if feats == []:
        feats = featmap[0, :, int(min_y): int(max_y)+1, int(min_x): int(max_x)+1]
        feats = feats.reshape(512, -1)
        feats = np.mean(feats, axis=1)
    else:
        feats = np.stack(feats, axis=0) # [num_points, 512]
        feats = np.mean(feats, axis=0) # mean pooling [512,]
    if math.isnan(feats[0]):
        ipdb.set_trace()
    return feats


def is_point_in_rectangle(p, corner_points):
    # 判断点p是否在矩形内部
    # 将矩形拆分为两个三角形，判断点p是否在两个三角形内部
    triangle1 = [corner_points[0], corner_points[1], corner_points[3]]
    triangle2 = [corner_points[0], corner_points[2], corner_points[3]]
    return is_point_in_triangle(p, triangle1) or is_point_in_triangle(p, triangle2)

def is_point_in_triangle(p, triangle):
    # 判断点p是否在三角形内部
    # 采用重心法判断点p是否在三角形内部
    x1, y1 = triangle[0]
    x2, y2 = triangle[1]
    x3, y3 = triangle[2]
    area = 0.5 * (-y2 * x3 + y1 * (-x2 + x3) + x1 * (y2 - y3) + x2 * y3)
    s = 1 / (2 * area) * (y1 * x3 - x1 * y3 + (y3 - y1) * p[0] + (x1 - x3) * p[1])
    t = 1 / (2 * area) * (x1 * y2 - y1 * x2 + (y1 - y2) * p[0] + (x2 - x1) * p[1])
    return s > 0 and t > 0 and 1 - s - t > 0