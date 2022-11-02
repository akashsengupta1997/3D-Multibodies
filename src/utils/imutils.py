"""
This file contains functions that are used to perform data augmentation.
"""
import torch
import numpy as np
import scipy.misc
import cv2

from utils import constants

def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0]-1, pt[1]-1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)+1

def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1, 
                             res[1]+1], center, scale, res, invert=1))-1
    
    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], 
                                                        old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = scipy.misc.imresize(new_img, res)
    return new_img

def uncrop(img, center, scale, orig_shape, rot=0, is_rgb=True):
    """'Undo' the image cropping/resizing.
    This function is used when evaluating mask/part segmentation.
    """
    res = img.shape[:2]
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1,res[1]+1], center, scale, res, invert=1))-1
    # size of cropped image
    crop_shape = [br[1] - ul[1], br[0] - ul[0]]

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(orig_shape, dtype=np.uint8)
    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], orig_shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], orig_shape[0]) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(orig_shape[1], br[0])
    old_y = max(0, ul[1]), min(orig_shape[0], br[1])
    img = scipy.misc.imresize(img, crop_shape, interp='nearest')
    new_img[old_y[0]:old_y[1], old_x[0]:old_x[1]] = img[new_y[0]:new_y[1], new_x[0]:new_x[1]]
    return new_img

def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                  [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                  [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R,per_rdg))
    aa = (resrot.T)[0]
    return aa

def flip_img(img):
    """Flip rgb images or masks.
    channels come last, e.g. (256,256,3).
    """
    img = np.fliplr(img)
    return img

def flip_kp(kp):
    """Flip keypoints."""
    if len(kp) == 24:
        flipped_parts = constants.J24_FLIP_PERM
    elif len(kp) == 49:
        flipped_parts = constants.J49_FLIP_PERM
    kp = kp[flipped_parts]
    kp[:,0] = - kp[:,0]
    return kp

def flip_pose(pose):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    flipped_parts = constants.SMPL_POSE_FLIP_PERM
    pose = pose[flipped_parts]
    # we also negate the second and the third dimension of the axis-angle
    pose[1::3] = -pose[1::3]
    pose[2::3] = -pose[2::3]
    return pose


def batch_crop_opencv_affine(output_wh,
                             num_to_crop,
                             iuv=None,
                             joints2D=None,
                             rgb=None,
                             seg=None,
                             bbox_centres=None,
                             bbox_heights=None,
                             bbox_widths=None,
                             bbox_whs=None,
                             joints2D_vis=None,
                             orig_scale_factor=1.2,
                             delta_scale_range=None,
                             delta_centre_range=None,
                             out_of_frame_pad_val=0,
                             solve_for_affine_trans=False,
                             uncrop=False,
                             uncrop_wh=None):
    """
    :param output_wh: tuple, output image (width, height)
    :param num_to_crop: scalar int, number of images in batch
    :param iuv: (B, 3, H, W)
    :param joints2D: (B, K, 2)
    :param rgb: (B, 3, H, W)
    :param seg: (B, H, W)
    :param bbox_centres: (B, 2), bounding box centres in (vertical, horizontal) coordinates
    :param bbox_heights: (B,)
    :param bbox_widths: (B,)
    :param bbox_whs: (B,) width/height for square bounding boxes
    :param joints2D_vis: (B, K)
    :param orig_scale_factor: original bbox scale factor (pre-augmentation)
    :param delta_scale_range: bbox scale augmentation range
    :param delta_centre_range: bbox centre augmentation range
    :param out_of_frame_pad_val: padding value for out-of-frame region after affine transform
    :param solve_for_affine_trans: bool, if true use cv2.getAffineTransform() to determine
        affine transformation matrix.
    :param uncrop: bool, if true uncrop image by applying inverse affine transformation
    :param uncrop_wh: tuple, output image size for uncropping.
    :return: cropped iuv/joints2D/rgb/seg, resized to output_wh
    """
    output_wh = np.array(output_wh, dtype=np.float32)
    cropped_dict = {}
    if iuv is not None:
        if not uncrop:
            cropped_dict['iuv'] = np.zeros((iuv.shape[0], 3, int(output_wh[1]), int(output_wh[0])), dtype=np.float32)
        else:
            cropped_dict['iuv'] = np.zeros((iuv.shape[0], 3, int(uncrop_wh[1]), int(uncrop_wh[0])), dtype=np.float32)
    if joints2D is not None:
        cropped_dict['joints2D'] = np.zeros_like(joints2D)
    if rgb is not None:
        if not uncrop:
            cropped_dict['rgb'] = np.zeros((rgb.shape[0], 3, int(output_wh[1]), int(output_wh[0])), dtype=np.float32)
        else:
            cropped_dict['rgb'] = np.zeros((rgb.shape[0], 3, int(uncrop_wh[1]), int(uncrop_wh[0])), dtype=np.float32)
    if seg is not None:
        if not uncrop:
            cropped_dict['seg'] = np.zeros((seg.shape[0], int(output_wh[1]), int(output_wh[0])), dtype=np.float32)
        else:
            cropped_dict['seg'] = np.zeros((seg.shape[0], int(uncrop_wh[1]), int(uncrop_wh[0])), dtype=np.float32)

    for i in range(num_to_crop):
        if bbox_centres is None:
            assert (iuv is not None) or (joints2D is not None) or (seg is not None), "Need either IUV, Seg or 2D Joints to determine bounding boxes!"
            if iuv is not None:
                # Determine bounding box corners from segmentation foreground/body pixels from IUV map
                body_pixels = np.argwhere(iuv[i, 0, :, :] != 0)
                bbox_corners = np.concatenate([np.amin(body_pixels, axis=0),
                                               np.amax(body_pixels, axis=0)])
            elif seg is not None:
                # Determine bounding box corners from segmentation foreground/body pixels
                body_pixels = np.argwhere(seg[i] != 0)
                bbox_corners = np.concatenate([np.amin(body_pixels, axis=0),
                                               np.amax(body_pixels, axis=0)])
            elif joints2D is not None:
                # Determine bounding box corners from 2D joints
                visible_joints2D = joints2D[i, joints2D_vis[i]]
                bbox_corners = np.concatenate([np.amin(visible_joints2D, axis=0)[::-1],   # (hor, vert) coords to (vert, hor) coords
                                               np.amax(visible_joints2D, axis=0)[::-1]])
                if (bbox_corners[:2] == bbox_corners[2:]).all():  # This can happen if only 1 joint is visible in input
                    print('Only 1 visible joint in input!')
                    bbox_corners[2:] = bbox_corners[:2] + output_wh / 4.
            bbox_centre, bbox_height, bbox_width = convert_bbox_corners_to_centre_hw(bbox_corners)
        else:
            bbox_centre = bbox_centres[i]
            if bbox_whs is not None:
                bbox_height = bbox_whs[i]
                bbox_width = bbox_whs[i]
            else:
                bbox_height = bbox_heights[i]
                bbox_width = bbox_widths[i]

        if not uncrop:
            # Change bounding box aspect ratio to match output aspect ratio
            aspect_ratio = output_wh[1] / output_wh[0]
            if bbox_height > bbox_width * aspect_ratio:
                bbox_width = bbox_height / aspect_ratio
            elif bbox_height < bbox_width * aspect_ratio:
                bbox_height = bbox_width * aspect_ratio

            # Scale bounding boxes + Apply random augmentations
            if delta_scale_range is not None:
                l, h = delta_scale_range
                delta_scale = (h - l) * np.random.rand() + l
                scale_factor = orig_scale_factor + delta_scale
            else:
                scale_factor = orig_scale_factor
            bbox_height = bbox_height * scale_factor
            bbox_width = bbox_width * scale_factor
            if delta_centre_range is not None:
                l, h = delta_centre_range
                delta_centre = (h - l) * np.random.rand(2) + l
                bbox_centre = bbox_centre + delta_centre

            # Determine affine transformation mapping bounding box to output image
            output_centre = output_wh * 0.5
            if solve_for_affine_trans:
                # Solve for affine transformation using 3 point correspondences (6 equations)
                bbox_points = np.zeros((3, 2), dtype=np.float32)
                bbox_points[0, :] = bbox_centre[::-1]  # (vert, hor) coordinates to (hor, vert coordinates)
                bbox_points[1, :] = bbox_centre[::-1] + np.array([bbox_width * 0.5, 0], dtype=np.float32)
                bbox_points[2, :] = bbox_centre[::-1] + np.array([0, bbox_height * 0.5], dtype=np.float32)

                output_points = np.zeros((3, 2), dtype=np.float32)
                output_points[0, :] = output_centre
                output_points[1, :] = output_centre + np.array([output_wh[0] * 0.5, 0], dtype=np.float32)
                output_points[2, :] = output_centre + np.array([0, output_wh[1] * 0.5], dtype=np.float32)
                affine_trans = cv2.getAffineTransform(bbox_points, output_points)
            else:
                # Hand-code affine transformation matrix - easy for cropping = scale + translate
                affine_trans = np.zeros((2, 3), dtype=np.float32)
                affine_trans[0, 0] = output_wh[0] / bbox_width
                affine_trans[1, 1] = output_wh[1] / bbox_height
                affine_trans[:, 2] = output_centre - (output_wh / np.array([bbox_width, bbox_height])) * bbox_centre[::-1]  # (vert, hor) coords to (hor, vert) coords
        else:
            # Hand-code inverse affine transformation matrix - easy for UN-cropping = scale + translate
            affine_trans = np.zeros((2, 3), dtype=np.float32)
            output_centre = output_wh * 0.5
            affine_trans[0, 0] = bbox_width / output_wh[0]
            affine_trans[1, 1] = bbox_height / output_wh[1]
            affine_trans[:, 2] = bbox_centre[::-1] - (np.array([bbox_width, bbox_height]) / output_wh) * output_centre

        # Apply affine transformation inputs.
        if iuv is not None:
            cropped_dict['iuv'][i] = cv2.warpAffine(src=iuv[i].transpose(1, 2, 0),
                                                    M=affine_trans,
                                                    dsize=tuple(output_wh.astype(np.int16)) if not uncrop else uncrop_wh,
                                                    flags=cv2.INTER_NEAREST,
                                                    borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=out_of_frame_pad_val).transpose(2, 0, 1)
        if joints2D is not None:
            joints2D_homo = np.concatenate([joints2D[i], np.ones((joints2D.shape[1], 1))],
                                           axis=-1)
            cropped_dict['joints2D'][i] = np.einsum('ij,kj->ki', affine_trans, joints2D_homo)

        if rgb is not None:
            cropped_dict['rgb'][i] = cv2.warpAffine(src=rgb[i].transpose(1, 2, 0),
                                                    M=affine_trans,
                                                    dsize=tuple(output_wh.astype(np.int16)) if not uncrop else uncrop_wh,
                                                    flags=cv2.INTER_LINEAR,
                                                    borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=0).transpose(2, 0, 1)
        if seg is not None:
            cropped_dict['seg'][i] = cv2.warpAffine(src=seg[i],
                                                    M=affine_trans,
                                                    dsize=tuple(output_wh.astype(np.int16)) if not uncrop else uncrop_wh,
                                                    flags=cv2.INTER_NEAREST,
                                                    borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=0)

    return cropped_dict
