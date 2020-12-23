'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2020-12-22 15:54:50
LastEditTime: 2020-12-22 16:15:24
Contact: cgjhaha@qq.com
Description: viusalize point cloud data
'''


import os
import torch
import pickle
import pandas
import numpy as np
from pathlib import Path
import mayavi.mlab as mlab

box_colormap = [[1, 1, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 1, 0]]

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def visualize_pts(pts, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
                  color_feature=None, size=(600, 600), draw_origin=True, title=None):

    min_height_mask = pts[:, 2] > -1.5
    pts = pts[min_height_mask, :]

    fig = mlab.figure(figure=title, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    if color_feature:
        mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], -pts[:, color_feature], mode='point',
                      colormap='jet', scale_factor=1, figure=fig)
    else:
        mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point',
                      colormap='jet', scale_factor=1, figure=fig)

    if draw_origin:
        cylinder = mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cylinder', scale_mode='vector', scale_factor=3)
        cylinder.glyph.glyph_source._trfm.transform.rotate_x(90)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)

    return fig

def draw_multi_grid_range(fig, grid_size=20, bv_range=(-60, -60, 60, 60)):
    for x in range(bv_range[0], bv_range[2], grid_size):
        for y in range(bv_range[1], bv_range[3], grid_size):
            fig = draw_grid(x, y, x + grid_size, y + grid_size, fig)

    return fig

def draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5)):
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    return fig

def draw_corners3d(corners3d, fig, color=(1, 1, 1), line_width=2, cls=None, tag='', max_num=500, tube_radius=None):
    """
    :param corners3d: (N, 8, 3)
    :param fig:
    :param color:
    :param line_width:
    :param cls:
    :param tag:
    :param max_num:
    :return:
    """
    import mayavi.mlab as mlab
    num = min(max_num, len(corners3d))
    for n in range(num):
        b = corners3d[n]  # (8, 3)

        if cls is not None:
            if isinstance(cls, np.ndarray):
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%.2f' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)
            else:
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%s' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)

        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

        i, j = 0, 5
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)
        i, j = 1, 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)

    return fig

def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """

    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

def draw_scene(points, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None, title=None, confidence=0.0):

    fig = visualize_pts(points, title=title, color_feature=3)   # 2 = z-value, 3 = intensity
    fig = draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))

    if gt_boxes is not None:

        corners3d = boxes_to_corners_3d(gt_boxes)
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)

    if ref_boxes is not None:

        if ref_boxes.shape[0] > 0:

            ref_corners3d = boxes_to_corners_3d(ref_boxes)

            if ref_labels is None:

                    fig = draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), cls=ref_scores, max_num=100)

            else:

                if ref_labels.shape[0] > 0:

                    for k in range(ref_labels.min(), ref_labels.max() + 1):

                        cur_color = tuple(box_colormap[k % len(box_colormap)])
                        mask = (ref_labels == k) & (ref_scores > confidence)

                        fig = draw_corners3d(ref_corners3d[mask],
                                             fig=fig,
                                             color=cur_color,
                                             cls=ref_scores[mask],
                                             max_num=100)

    mlab.view(azimuth=-180, elevation=60.0, distance=105.0, roll=90.0)

    return fig


if __name__ == '__main__':

    # import_dir = Path(os.getcwd()).parent.parent / 'output' / 'DEMO' / 'KITTI' / 'pv_rcnn'
    import_dir = Path('/Users/dekura/chen/bei/projects/pchsd/OpenPCDet/outputs')

    data = []
    for i in os.listdir(import_dir):
        if os.path.isfile(os.path.join(import_dir, i)) and 'data_dict_' in i:
            data.append(i)

    data = sorted(data)

    predictions = []
    for i in os.listdir(import_dir):
        if os.path.isfile(os.path.join(import_dir, i)) and 'pred_dicts_' in i:
            predictions.append(i)

    predictions = sorted(predictions)

    num_samples = len(data)

    print(f'{num_samples} number of samples found')

    # df = pandas.read_csv('../../data/kitti/ImageSets/test.txt', header=None)

    # index = 5896 # has to be -1 of index in all.txt
    indexs = [0]
    # sample_id = 1
    for index in indexs:
        sample_id = f'{index+1:06d}'
        # current_recording = f'{df.values[index][0]}_{df.values[index][1]}'

        # title = f'# {sample_id}/{num_samples} - {current_recording}'

        with open(f'{import_dir}/data_dict_{sample_id}.pkl', 'rb') as f:
            data_dict = pickle.load(f)

        with open(f'{import_dir}/pred_dicts_{sample_id}.pkl', 'rb') as f:
            pred_dicts = pickle.load(f)

        scene = draw_scene(points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                            ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                            title=sample_id, confidence=0.6)

        mlab.show(stop=True)
        # index += 1 % len(data)
