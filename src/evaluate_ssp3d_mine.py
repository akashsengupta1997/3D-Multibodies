import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import cv2
import math

import config
from utils import constants
from model_mine import Model
from experiment_config import ExperimentConfig
from networks.smpl.smpl_mine import SMPL
from utils.pose_utils import compute_similarity_transform_batch, scale_and_translation_transform_batch
from utils.renderer import Renderer
from utils.geometry import undo_keypoint_normalisation, convert_weak_perspective_to_camera_translation
from utils.cam_utils import orthographic_project_torch, check_joints2d_visibility_torch
from utils.sampling_utils import compute_vertex_uncertainties_from_samples
from datasets.ssp3d_eval_dataset import SSP3DEvalDataset

from exp_manager.config import get_config_from_file, get_arg_parser, set_config
from exp_manager.utils import pprint_dict
from exp_manager.model_io import load_model

import subsets


def evaluate_ssp3d(model,
                   eval_dataset,
                   metrics_to_track,
                   device,
                   save_path,
                   num_pred_samples,
                   vis_img_wh=512,
                   num_workers=4,
                   pin_memory=True,
                   vis_every_n_batches=1000,
                   num_samples_to_visualise=10,
                   save_per_frame_uncertainty=True,
                   extreme_crop=False):

    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 drop_last=True,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)

    smpl_neutral = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)
    smpl_male = SMPL(config.SMPL_MODEL_DIR, batch_size=1, gender='male').to(device)
    smpl_female = SMPL(config.SMPL_MODEL_DIR, batch_size=1, gender='female').to(device)

    metric_sums = {'num_datapoints': 0}
    per_frame_metrics = {}
    for metric in metrics_to_track:
        metric_sums[metric] = 0.
        per_frame_metrics[metric] = []

        if metric == 'joints3D_coco_invis_samples_dist_from_mean':
            metric_sums['num_invis_joints3Dsamples'] = 0

        if metric == 'joints3D_coco_vis_samples_dist_from_mean':
            metric_sums['num_vis_joints3Dsamples'] = 0

        elif metric == 'joints2D_l2es':
            metric_sums['num_vis_joints2D'] = 0
        elif metric == 'joints2Dsamples_l2es':
            metric_sums['num_vis_joints2Dsamples'] = 0

        elif metric == 'silhouette_ious':
            metric_sums['num_true_positives'] = 0.
            metric_sums['num_false_positives'] = 0.
            metric_sums['num_true_negatives'] = 0.
            metric_sums['num_false_negatives'] = 0.
        elif metric == 'silhouettesamples_ious':
            metric_sums['num_samples_true_positives'] = 0.
            metric_sums['num_samples_false_positives'] = 0.
            metric_sums['num_samples_true_negatives'] = 0.
            metric_sums['num_samples_false_negatives'] = 0.

    fname_per_frame = []
    pose_per_frame = []
    shape_per_frame = []
    cam_per_frame = []
    if save_per_frame_uncertainty:
        vertices_uncertainty_per_frame = []
        vertices_uncertainty_xyz_per_frame = []

    renderer_for_silh_eval = Renderer(img_res=constants.IMG_RES, faces=smpl_neutral.faces)
    renderer_for_vis = Renderer(img_res=vis_img_wh, faces=smpl_neutral.faces)
    reposed_cam_t = convert_weak_perspective_to_camera_translation(cam_wp=np.array([0.85, 0., -0.2]),
                                                                   focal_length=constants.FOCAL_LENGTH,
                                                                   resolution=vis_img_wh)
    if extreme_crop:
        rot_cam_t = convert_weak_perspective_to_camera_translation(cam_wp=np.array([0.85, 0., 0.]),
                                                                   focal_length=constants.FOCAL_LENGTH,
                                                                   resolution=vis_img_wh)

    for batch_num, samples_batch in enumerate(tqdm(eval_dataloader)):
        # if batch_num == 2:
        #     break
        # ------------------------------- TARGETS and INPUTS -------------------------------
        input = samples_batch['input'].to(device)
        target_pose = samples_batch['pose'].to(device)
        target_shape = samples_batch['shape'].to(device)
        target_gender = samples_batch['gender'][0]
        target_silhouette = samples_batch['silhouette']
        target_joints2D_coco = samples_batch['keypoints']
        target_joints2D_vis_coco = check_joints2d_visibility_torch(target_joints2D_coco,
                                                                   input.shape[-1])  # (batch_size, 17)
        fname = samples_batch['fname']

        if target_gender == 'm':
            target_smpl_output = smpl_male(body_pose=target_pose[:, 3:],
                                           global_orient=target_pose[:, :3],
                                           betas=target_shape)
            target_reposed_smpl_output = smpl_male(betas=target_shape)
        elif target_gender == 'f':
            target_smpl_output = smpl_female(body_pose=target_pose[:, 3:],
                                             global_orient=target_pose[:, :3],
                                             betas=target_shape)
            target_reposed_smpl_output = smpl_female(betas=target_shape)

        target_vertices = target_smpl_output.vertices
        target_reposed_vertices = target_reposed_smpl_output.vertices

        # ------------------------------- PREDICTIONS -------------------------------
        pred = model(input)
        # print('\nMODEL OUT')
        # for key in pred:
        #     print(key, pred[key].shape, pred[key].numel(), torch.isnan(pred[key]).sum().cpu().detach().numpy())

        # SPIN predictions as mode
        pred_cam_wp_spin = pred['spin_camera']
        pred_pose_rotmats_spin = pred['spin_local_rotmat']
        pred_glob_rotmat_spin = pred['spin_global_rotmat']
        pred_shape_spin = pred['spin_shape']
        pred_smpl_output_spin = smpl_neutral(body_pose=pred_pose_rotmats_spin,
                                             global_orient=pred_glob_rotmat_spin,
                                             betas=pred_shape_spin,
                                             pose2rot=False)
        pred_vertices_spin = pred_smpl_output_spin.vertices  # (1, 6890, 3)
        pred_joints_coco_spin = pred_smpl_output_spin.joints[:, constants.ALL_JOINTS_TO_COCO_MAP, :]  # (1, 17, 3)
        pred_vertices2D_spin_for_vis = orthographic_project_torch(pred_vertices_spin,
                                                                  pred_cam_wp_spin,
                                                                  scale_first=False)
        pred_vertices2D_spin_for_vis = undo_keypoint_normalisation(pred_vertices2D_spin_for_vis, vis_img_wh)
        pred_joints2D_coco_spin_normed = orthographic_project_torch(pred_joints_coco_spin,
                                                                    pred_cam_wp_spin,
                                                                    scale_first=False)  # (1, 17, 2)
        pred_joints2D_coco_spin = undo_keypoint_normalisation(pred_joints2D_coco_spin_normed, input.shape[-1])
        pred_reposed_vertices_spin = smpl_neutral(betas=pred_shape_spin).vertices  # (1, 6890, 3)

        pred_cam_wp_samples = pred['pred_camera'][0]  # (num_samples, 3)
        pred_pose_rotmats_samples = pred['pred_local_rotmat'][0]  # (num_samples, 23, 3, 3)
        pred_glob_rotmat_samples = pred['pred_global_rotmat'][0]  # (num_samples, 1, 3, 3)
        pred_shape_samples = pred['pred_shape'][0]  # (num_samples, 10)

        pred_smpl_output_samples = smpl_neutral(body_pose=pred_pose_rotmats_samples,
                                                global_orient=pred_glob_rotmat_samples,
                                                betas=pred_shape_samples,
                                                pose2rot=False)
        pred_vertices_samples = pred_smpl_output_samples.vertices  # (num_pred_samples, 6890, 3)

        pred_joints_coco_samples = pred_smpl_output_samples.joints[:, constants.ALL_JOINTS_TO_COCO_MAP, :]  # (num_pred_samples, 17, 3)
        pred_joints2D_coco_samples = orthographic_project_torch(pred_joints_coco_samples, pred_cam_wp_samples)  # (num_pred_samples, 17, 2)
        pred_joints2D_coco_samples = undo_keypoint_normalisation(pred_joints2D_coco_samples, input.shape[-1])

        pred_reposed_vertices_samples = smpl_neutral(body_pose=torch.zeros(num_pred_samples, 69, device=device, dtype=torch.float32),
                                                     global_orient=torch.zeros(num_pred_samples, 3, device=device, dtype=torch.float32),
                                                     betas=pred_shape_samples).vertices  # (num_pred_samples, 6890, 3)

        # ------------------------------------------------ METRICS ------------------------------------------------

        # Numpy-fying targets
        target_vertices = target_vertices.cpu().detach().numpy()
        target_reposed_vertices = target_reposed_vertices.cpu().detach().numpy()
        target_joints2D_coco = target_joints2D_coco.cpu().detach().numpy()
        target_joints2D_vis_coco = target_joints2D_vis_coco.cpu().detach().numpy()
        target_silhouette = target_silhouette.cpu().detach().numpy()

        # Numpy-fying preds
        pred_vertices_spin = pred_vertices_spin.cpu().detach().numpy()
        pred_vertices2D_spin_for_vis = pred_vertices2D_spin_for_vis.cpu().detach().numpy()
        pred_joints2D_coco_spin = pred_joints2D_coco_spin.cpu().detach().numpy()
        pred_reposed_vertices_spin = pred_reposed_vertices_spin.cpu().detach().numpy()

        pred_vertices_samples = pred_vertices_samples.cpu().detach().numpy()
        pred_joints_coco_samples = pred_joints_coco_samples.cpu().detach().numpy()
        pred_joints2D_coco_samples = pred_joints2D_coco_samples.cpu().detach().numpy()
        pred_reposed_vertices_samples = pred_reposed_vertices_samples.cpu().detach().numpy()

        # -------------- 3D Metrics with Mode and Minimum Error Samples --------------
        if 'pves' in metrics_to_track:
            pve_batch = np.linalg.norm(pred_vertices_spin - target_vertices,
                                       axis=-1)  # (bs, 6890)
            metric_sums['pves'] += np.sum(pve_batch)  # scalar
            per_frame_metrics['pves'].append(np.mean(pve_batch, axis=-1))

        if 'pves_samples_min' in metrics_to_track:
            pve_per_sample = np.linalg.norm(pred_vertices_samples - target_vertices, axis=-1)  # (num samples, 6890)
            min_pve_sample = np.argmin(np.mean(pve_per_sample, axis=-1))
            pve_samples_min_batch = pve_per_sample[min_pve_sample]  # (6890,)
            metric_sums['pves_samples_min'] += np.sum(pve_samples_min_batch)
            per_frame_metrics['pves_samples_min'].append(np.mean(pve_samples_min_batch, axis=-1, keepdims=True))  # (1,)

        # Scale and translation correction
        if 'pves_sc' in metrics_to_track:
            pred_vertices_sc = scale_and_translation_transform_batch(
                pred_vertices_spin,
                target_vertices)
            pve_sc_batch = np.linalg.norm(
                pred_vertices_sc - target_vertices,
                axis=-1)  # (bs, 6890)
            metric_sums['pves_sc'] += np.sum(pve_sc_batch)  # scalar
            per_frame_metrics['pves_sc'].append(np.mean(pve_sc_batch, axis=-1))

        if 'pves_sc_samples_min' in metrics_to_track:
            target_vertices_tiled = np.tile(target_vertices, (num_pred_samples, 1, 1))  # (num samples, 6890, 3)
            pred_vertices_samples_sc = scale_and_translation_transform_batch(
                pred_vertices_samples,
                target_vertices_tiled)
            pve_sc_per_sample = np.linalg.norm(pred_vertices_samples_sc - target_vertices_tiled, axis=-1)  # (num samples, 6890)
            min_pve_sc_sample = np.argmin(np.mean(pve_sc_per_sample, axis=-1))
            pve_sc_samples_min_batch = pve_sc_per_sample[min_pve_sc_sample]  # (6890,)
            metric_sums['pves_sc_samples_min'] += np.sum(pve_sc_samples_min_batch)
            per_frame_metrics['pves_sc_samples_min'].append(np.mean(pve_sc_samples_min_batch, axis=-1, keepdims=True))  # (1,)

        # Procrustes analysis
        if 'pves_pa' in metrics_to_track:
            pred_vertices_pa = compute_similarity_transform_batch(pred_vertices_spin, target_vertices)
            pve_pa_batch = np.linalg.norm(pred_vertices_pa - target_vertices, axis=-1)  # (bs, 6890)
            metric_sums['pves_pa'] += np.sum(pve_pa_batch)  # scalar
            per_frame_metrics['pves_pa'].append(np.mean(pve_pa_batch, axis=-1))

        if 'pves_pa_samples_min' in metrics_to_track:
            target_vertices_tiled = np.tile(target_vertices, (num_pred_samples, 1, 1))  # (num samples, 6890, 3)
            pred_vertices_samples_pa = compute_similarity_transform_batch(
                pred_vertices_samples,
                target_vertices_tiled)
            pve_pa_per_sample = np.linalg.norm(pred_vertices_samples_pa - target_vertices_tiled, axis=-1)  # (num samples, 6890)
            min_pve_pa_sample = np.argmin(np.mean(pve_pa_per_sample, axis=-1))
            pve_pa_samples_min_batch = pve_pa_per_sample[min_pve_pa_sample]  # (6890,)
            metric_sums['pves_pa_samples_min'] += np.sum(pve_pa_samples_min_batch)
            per_frame_metrics['pves_pa_samples_min'].append(np.mean(pve_pa_samples_min_batch, axis=-1, keepdims=True))  # (1,)

        if 'pve-ts' in metrics_to_track:
            pvet_batch = np.linalg.norm(pred_reposed_vertices_spin - target_reposed_vertices, axis=-1)
            metric_sums['pve-ts'] += np.sum(pvet_batch)  # scalar
            per_frame_metrics['pve-ts'].append(np.mean(pvet_batch, axis=-1))

        if 'pve-ts_samples_min' in metrics_to_track:
            pvet_per_sample = np.linalg.norm(pred_reposed_vertices_samples - target_reposed_vertices, axis=-1)  # (num samples, 6890)
            min_pvet_sample = np.argmin(np.mean(pvet_per_sample, axis=-1))
            pvet_samples_min_batch = pvet_per_sample[min_pvet_sample]  # (6890,)
            metric_sums['pve-ts_samples_min'] += np.sum(pvet_samples_min_batch)
            per_frame_metrics['pve-ts_samples_min'].append(np.mean(pvet_samples_min_batch, axis=-1, keepdims=True))  # (1,)

        # Scale and translation correction
        if 'pve-ts_sc' in metrics_to_track:
            pred_reposed_vertices_sc = scale_and_translation_transform_batch(
                pred_reposed_vertices_spin,
                target_reposed_vertices)
            pvet_scale_corrected_batch = np.linalg.norm(
                pred_reposed_vertices_sc - target_reposed_vertices,
                axis=-1)  # (bs, 6890)
            metric_sums['pve-ts_sc'] += np.sum(pvet_scale_corrected_batch)  # scalar
            per_frame_metrics['pve-ts_sc'].append(np.mean(pvet_scale_corrected_batch, axis=-1))

        if 'pve-ts_sc_samples_min' in metrics_to_track:
            target_reposed_vertices_tiled = np.tile(target_reposed_vertices, (num_pred_samples, 1, 1))  # (num samples, 6890, 3)
            pred_reposed_vertices_samples_sc = scale_and_translation_transform_batch(
                pred_reposed_vertices_samples,
                target_reposed_vertices_tiled)
            pvet_sc_per_sample = np.linalg.norm(pred_reposed_vertices_samples_sc - target_reposed_vertices_tiled, axis=-1)  # (num samples, 6890)
            min_pvet_sc_sample = np.argmin(np.mean(pvet_sc_per_sample, axis=-1))
            pvet_sc_samples_min_batch = pvet_sc_per_sample[min_pvet_sc_sample]  # (6890,)
            metric_sums['pve-ts_sc_samples_min'] += np.sum(pvet_sc_samples_min_batch)
            per_frame_metrics['pve-ts_sc_samples_min'].append(np.mean(pvet_sc_samples_min_batch, axis=-1, keepdims=True))  # (1,)

        # ---------------- 3D Sample Distance from Mean (i.e. Variance) Metrics -----------
        if 'verts_samples_dist_from_mean' in metrics_to_track:
            verts_samples_mean = pred_vertices_samples.mean(axis=0)  # (6890, 3)
            verts_samples_dist_from_mean = np.linalg.norm(pred_vertices_samples - verts_samples_mean, axis=-1)  # (num samples, 6890)
            metric_sums['verts_samples_dist_from_mean'] += verts_samples_dist_from_mean.sum()
            per_frame_metrics['verts_samples_dist_from_mean'].append(verts_samples_dist_from_mean.mean()[None])  # (1,)

        if 'joints3D_coco_samples_dist_from_mean' in metrics_to_track:
            joints3D_coco_samples_mean = pred_joints_coco_samples.mean(axis=0)  # (17, 3)
            joints3D_coco_samples_dist_from_mean = np.linalg.norm(pred_joints_coco_samples - joints3D_coco_samples_mean, axis=-1)  # (num samples, 17)
            metric_sums['joints3D_coco_samples_dist_from_mean'] += joints3D_coco_samples_dist_from_mean.sum()  # scalar
            per_frame_metrics['joints3D_coco_samples_dist_from_mean'].append(joints3D_coco_samples_dist_from_mean.mean()[None])  # (1,)

        if 'joints3D_coco_invis_samples_dist_from_mean' in metrics_to_track:
            # (In)visibility of specific joints determined by HRNet 2D joint predictions and confidence scores.
            target_joints2D_invis_coco = np.logical_not(target_joints2D_vis_coco[0])  # (17,)

            if np.any(target_joints2D_invis_coco):
                joints3D_coco_invis_samples = pred_joints_coco_samples[:, target_joints2D_invis_coco, :]  # (num samples, num invis joints, 3)
                joints3D_coco_invis_samples_mean = joints3D_coco_invis_samples.mean(axis=0)  # (num_invis_joints, 3)
                joints3D_coco_invis_samples_dist_from_mean = np.linalg.norm(joints3D_coco_invis_samples - joints3D_coco_invis_samples_mean,
                                                                            axis=-1)  # (num samples, num_invis_joints)

                metric_sums['joints3D_coco_invis_samples_dist_from_mean'] += joints3D_coco_invis_samples_dist_from_mean.sum()  # scalar
                metric_sums['num_invis_joints3Dsamples'] += np.prod(joints3D_coco_invis_samples_dist_from_mean.shape)
                per_frame_metrics['joints3D_coco_invis_samples_dist_from_mean'].append(joints3D_coco_invis_samples_dist_from_mean.mean()[None])  # (1,)
            else:
                per_frame_metrics['joints3D_coco_invis_samples_dist_from_mean'].append(np.zeros(1))

        if 'joints3D_coco_vis_samples_dist_from_mean' in metrics_to_track:
            # Visibility of specific joints determined by HRNet 2D joint predictions and confidence scores.
            joints3D_coco_vis_samples = pred_joints_coco_samples[:, target_joints2D_vis_coco[0], :]  # (num samples, num vis joints, 3)
            joints3D_coco_vis_samples_mean = joints3D_coco_vis_samples.mean(axis=0)  # (num_vis_joints, 3)
            joints3D_coco_vis_samples_dist_from_mean = np.linalg.norm(joints3D_coco_vis_samples - joints3D_coco_vis_samples_mean,
                                                                      axis=-1)  # (num samples, num_vis_joints)

            metric_sums['joints3D_coco_vis_samples_dist_from_mean'] += joints3D_coco_vis_samples_dist_from_mean.sum()  # scalar
            metric_sums['num_vis_joints3Dsamples'] += np.prod(joints3D_coco_vis_samples_dist_from_mean.shape)
            per_frame_metrics['joints3D_coco_vis_samples_dist_from_mean'].append(joints3D_coco_vis_samples_dist_from_mean.mean()[None])  # (1,)

        # -------------------------------- 2D Metrics ---------------------------
        if 'joints2D_l2es' in metrics_to_track:
            joints2D_l2e_batch = np.linalg.norm(pred_joints2D_coco_spin[:, target_joints2D_vis_coco[0], :] - target_joints2D_coco[:, target_joints2D_vis_coco[0], :],
                                                axis=-1)  # (1, num vis joints)
            assert joints2D_l2e_batch.shape[1] == target_joints2D_vis_coco.sum()

            metric_sums['joints2D_l2es'] += np.sum(joints2D_l2e_batch)  # scalar
            metric_sums['num_vis_joints2D'] += joints2D_l2e_batch.shape[1]
            per_frame_metrics['joints2D_l2es'].append(np.mean(joints2D_l2e_batch, axis=-1))  # (1,)

        if 'silhouette_ious' in metrics_to_track:
            pred_cam_t_spin = torch.stack([pred_cam_wp_spin[0, 1],
                                           pred_cam_wp_spin[0, 2],
                                           2 * constants.FOCAL_LENGTH / (constants.IMG_RES * pred_cam_wp_spin[0, 0] + 1e-9)], dim=-1).cpu().detach().numpy()
            _, pred_silhouette_spin = renderer_for_silh_eval(vertices=pred_vertices_spin[0],
                                                             camera_translation=pred_cam_t_spin.copy(),
                                                             image=np.zeros((constants.IMG_RES, constants.IMG_RES, 3)),
                                                             return_silhouette=True)
            pred_silhouette_spin = pred_silhouette_spin[None, :, :, 0].astype(np.float32)  # (1, img_wh, img_wh)

            true_positive = np.logical_and(pred_silhouette_spin, target_silhouette)
            false_positive = np.logical_and(pred_silhouette_spin, np.logical_not(target_silhouette))
            true_negative = np.logical_and(np.logical_not(pred_silhouette_spin), np.logical_not(target_silhouette))
            false_negative = np.logical_and(np.logical_not(pred_silhouette_spin), target_silhouette)
            num_tp = np.sum(true_positive, axis=(1, 2))  # (1,)
            num_fp = np.sum(false_positive, axis=(1, 2))
            num_tn = np.sum(true_negative, axis=(1, 2))
            num_fn = np.sum(false_negative, axis=(1, 2))
            metric_sums['num_true_positives'] += np.sum(num_tp)  # scalar
            metric_sums['num_false_positives'] += np.sum(num_fp)
            metric_sums['num_true_negatives'] += np.sum(num_tn)
            metric_sums['num_false_negatives'] += np.sum(num_fn)
            iou_per_frame = num_tp / (num_tp + num_fp + num_fn)
            per_frame_metrics['silhouette_ious'].append(iou_per_frame)  # (1,)

        # -------------------------------- 2D Metrics after Averaging over Samples ---------------------------
        if 'joints2Dsamples_l2es' in metrics_to_track:
            joints2Dsamples_l2e_batch = np.linalg.norm(pred_joints2D_coco_samples[:, target_joints2D_vis_coco[0], :] - target_joints2D_coco[:, target_joints2D_vis_coco[0], :],
                                                       axis=-1)  # (num_samples, num vis joints)
            assert joints2Dsamples_l2e_batch.shape[1] == target_joints2D_vis_coco.sum()

            metric_sums['joints2Dsamples_l2es'] += np.sum(joints2Dsamples_l2e_batch)  # scalar
            metric_sums['num_vis_joints2Dsamples'] += np.prod(joints2Dsamples_l2e_batch.shape)
            per_frame_metrics['joints2Dsamples_l2es'].append(np.mean(joints2Dsamples_l2e_batch)[None])  # (1,)

        if 'silhouettesamples_ious' in metrics_to_track:
            pred_cam_t_samples = torch.stack([pred_cam_wp_samples[:, 1],
                                              pred_cam_wp_samples[:, 2],
                                              2 * constants.FOCAL_LENGTH / (constants.IMG_RES * pred_cam_wp_samples[:, 0] + 1e-9)], dim=-1).cpu().detach().numpy()
            pred_silhouette_samples = []
            for i in range(num_pred_samples):
                _, silh_sample = renderer_for_silh_eval(vertices=pred_vertices_samples[i],
                                                        camera_translation=pred_cam_t_samples[i].copy(),
                                                        image=np.zeros((constants.IMG_RES, constants.IMG_RES, 3)),
                                                        return_silhouette=True)
                pred_silhouette_samples.append(silh_sample[:, :, 0].astype(np.float32))
            pred_silhouette_samples = np.stack(pred_silhouette_samples, axis=0)[None, :, :, :]  # (1, num_samples, img_wh, img_wh)
            target_silhouette_tiled = np.tile(target_silhouette[:, None, :, :], (1, num_pred_samples, 1, 1))  # (1, num_samples, img_wh, img_wh)

            true_positive = np.logical_and(pred_silhouette_samples, target_silhouette_tiled)
            false_positive = np.logical_and(pred_silhouette_samples, np.logical_not(target_silhouette_tiled))
            true_negative = np.logical_and(np.logical_not(pred_silhouette_samples), np.logical_not(target_silhouette_tiled))
            false_negative = np.logical_and(np.logical_not(pred_silhouette_samples), target_silhouette_tiled)
            num_tp = np.sum(true_positive, axis=(1, 2, 3))  # (1,)
            num_fp = np.sum(false_positive, axis=(1, 2, 3))
            num_tn = np.sum(true_negative, axis=(1, 2, 3))
            num_fn = np.sum(false_negative, axis=(1, 2, 3))
            metric_sums['num_samples_true_positives'] += np.sum(num_tp)  # scalar
            metric_sums['num_samples_false_positives'] += np.sum(num_fp)
            metric_sums['num_samples_true_negatives'] += np.sum(num_tn)
            metric_sums['num_samples_false_negatives'] += np.sum(num_fn)
            iou_per_frame = num_tp / (num_tp + num_fp + num_fn)
            per_frame_metrics['silhouettesamples_ious'].append(iou_per_frame)  # (1,)

        metric_sums['num_datapoints'] += target_pose.shape[0]

        fname_per_frame.append(fname)
        pose_per_frame.append(np.concatenate([pred_glob_rotmat_spin.cpu().detach().numpy(),
                                              pred_pose_rotmats_spin.cpu().detach().numpy()],
                                             axis=1))
        shape_per_frame.append(pred_shape_spin.cpu().detach().numpy())
        cam_per_frame.append(pred_cam_wp_spin.cpu().detach().numpy())

        # ------------------------------- VISUALISE -------------------------------
        if vis_every_n_batches is not None and batch_num % vis_every_n_batches == 0:
            vis_img = samples_batch['vis_img'].numpy()

            pred_cam_t_spin = torch.stack([pred_cam_wp_spin[0, 1],
                                           pred_cam_wp_spin[0, 2],
                                           2 * constants.FOCAL_LENGTH / (vis_img_wh * pred_cam_wp_spin[0, 0] + 1e-9)], dim=-1).cpu().detach().numpy()

            # Uncertainty Computation
            # Uncertainty computed by sampling + average distance from mean
            avg_vertices_distance_from_mean, _, avg_vertices_distance_from_mean_xyz = compute_vertex_uncertainties_from_samples(
                vertices_samples=pred_vertices_samples,
                return_separate_reposed_dims=True)
            if save_per_frame_uncertainty:
                vertices_uncertainty_per_frame.append(avg_vertices_distance_from_mean)
                vertices_uncertainty_xyz_per_frame.append(avg_vertices_distance_from_mean_xyz)

            # Render predicted meshes
            body_vis_rgb_spin = renderer_for_vis(vertices=pred_vertices_spin[0],
                                                 camera_translation=pred_cam_t_spin.copy(),
                                                 image=vis_img[0])
            body_vis_rgb_spin_rot = renderer_for_vis(vertices=pred_vertices_spin[0],
                                                     camera_translation=pred_cam_t_spin.copy() if not extreme_crop else rot_cam_t.copy(),
                                                     image=np.zeros_like(vis_img[0]),
                                                     angle=np.pi / 2.,
                                                     axis=[0., 1., 0.])

            reposed_body_vis_rgb_spin = renderer_for_vis(vertices=pred_reposed_vertices_spin[0],
                                                         camera_translation=reposed_cam_t.copy(),
                                                         image=np.zeros_like(vis_img[0]),
                                                         flip_updown=False)
            reposed_body_vis_rgb_mean_rot = renderer_for_vis(vertices=pred_reposed_vertices_spin[0],
                                                             camera_translation=reposed_cam_t.copy(),
                                                             image=np.zeros_like(vis_img[0]),
                                                             angle=np.pi / 2.,
                                                             axis=[0., 1., 0.],
                                                             flip_updown=False)

            body_vis_rgb_samples = []
            body_vis_rgb_rot_samples = []
            pred_cam_t_samples = torch.stack([pred_cam_wp_samples[:, 1],
                                              pred_cam_wp_samples[:, 2],
                                              2 * constants.FOCAL_LENGTH / (vis_img_wh * pred_cam_wp_samples[:, 0] + 1e-9)], dim=-1).cpu().detach().numpy()
            for i in range(num_samples_to_visualise):
                body_vis_rgb_samples.append(renderer_for_vis(vertices=pred_vertices_samples[i],
                                                             camera_translation=pred_cam_t_samples[i].copy(),
                                                             image=vis_img[0]))
                body_vis_rgb_rot_samples.append(renderer_for_vis(vertices=pred_vertices_samples[i],
                                                                 camera_translation=pred_cam_t_samples[i].copy() if not extreme_crop else rot_cam_t.copy(),
                                                                 image=np.zeros_like(vis_img[0]),
                                                                 angle=np.pi / 2.,
                                                                 axis=[0., 1., 0.]))

            # Save samples
            samples_save_path = os.path.join(save_path, os.path.splitext(fname[0])[0] + '_samples.npy')
            np.save(samples_save_path, pred_vertices_samples)

            # ------------------ Model Prediction, Error and Uncertainty Figure ------------------
            num_row = 6
            num_col = 6
            subplot_count = 1
            plt.figure(figsize=(20, 20))

            # Plot image and mask vis
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(vis_img[0])
            subplot_count += 1

            # Plot pred vertices 2D and body render overlaid over input
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(vis_img[0])
            plt.scatter(pred_vertices2D_spin_for_vis[0, :, 0],
                        pred_vertices2D_spin_for_vis[0, :, 1],
                        c='r', s=0.01)
            subplot_count += 1

            # Plot body render overlaid on vis image
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(body_vis_rgb_spin)
            subplot_count += 1

            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(body_vis_rgb_spin_rot)
            subplot_count += 1

            # Plot reposed body render
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(reposed_body_vis_rgb_spin)
            subplot_count += 1

            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(reposed_body_vis_rgb_mean_rot)
            subplot_count += 1

            # Plot silhouette+J2D and reposed body render
            if 'silhouette_ious' in metrics_to_track:
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.imshow(pred_silhouette_spin[0].astype(np.int16) - target_silhouette[0].astype(np.int16))
                plt.text(10, 10, s='mIOU: {:.4f}'.format(per_frame_metrics['silhouette_ious'][batch_num][0]))
            if 'joints2D_l2es' in metrics_to_track:
                plt.scatter(target_joints2D_coco[0, :, 0],
                            target_joints2D_coco[0, :, 1],
                            c=target_joints2D_vis_coco[0, :].astype(np.float32), s=10.0)
                plt.scatter(pred_joints2D_coco_spin[0, :, 0],
                            pred_joints2D_coco_spin[0, :, 1],
                            c='r', s=10.0)
                for j in range(target_joints2D_coco.shape[1]):
                    plt.text(target_joints2D_coco[0, j, 0],
                             target_joints2D_coco[0, j, 1],
                             str(j))
                    plt.text(pred_joints2D_coco_spin[0, j, 0],
                             pred_joints2D_coco_spin[0, j, 1],
                             str(j))
                plt.text(10, 30, s='J2D L2E: {:.4f}'.format(per_frame_metrics['joints2D_l2es'][batch_num][0]))
            subplot_count += 1

            if 'pves_sc' in metrics_to_track:
                # Plot PVE-SC pred vs target comparison
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.text(0.5, 0.5, s='PVE-SC')
                subplot_count += 1
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.gca().invert_yaxis()
                plt.scatter(target_vertices[0, :, 0],
                            target_vertices[0, :, 1],
                            s=0.02,
                            c='blue')
                plt.scatter(pred_vertices_sc[0, :, 0],
                            pred_vertices_sc[0, :, 1],
                            s=0.01,
                            c='red')
                plt.gca().set_aspect('equal', adjustable='box')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.gca().invert_yaxis()
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(pred_vertices_sc[0, :, 0],
                            pred_vertices_sc[0, :, 1],
                            s=0.05,
                            c=pve_sc_batch[0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-SC: {:.4f}'.format(per_frame_metrics['pves_sc'][batch_num][0]))
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.gca().invert_yaxis()
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(pred_vertices_sc[0, :, 2],  # Equivalent to Rotated 90° about y axis
                            pred_vertices_sc[0, :, 1],
                            s=0.05,
                            c=pve_sc_batch[0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-SC: {:.4f}'.format(per_frame_metrics['pves_sc'][batch_num][0]))
                subplot_count += 1

            if 'pves_pa' in metrics_to_track:
                # Plot PVE-PA pred vs target comparison
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.text(0.5, 0.5, s='PVE-PA')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.gca().invert_yaxis()
                plt.scatter(target_vertices[0, :, 0],
                            target_vertices[0, :, 1],
                            s=0.02,
                            c='blue')
                plt.scatter(pred_vertices_pa[0, :, 0],
                            pred_vertices_pa[0, :, 1],
                            s=0.01,
                            c='red')
                plt.gca().set_aspect('equal', adjustable='box')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.gca().invert_yaxis()
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(pred_vertices_pa[0, :, 0],
                            pred_vertices_pa[0, :, 1],
                            s=0.05,
                            c=pve_pa_batch[0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-PA: {:.4f}'.format(per_frame_metrics['pves_pa'][batch_num][0]))
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.gca().invert_yaxis()
                norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
                plt.scatter(pred_vertices_pa[0, :, 2],  # Equivalent to Rotated 90° about y axis
                            pred_vertices_pa[0, :, 1],
                            s=0.05,
                            c=pve_pa_batch[0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-PA: {:.4f}'.format(per_frame_metrics['pves_pa'][batch_num][0]))
                subplot_count += 1

            if 'pve-ts_sc' in metrics_to_track:
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.text(0.5, 0.5, s='PVE-T-SC')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.scatter(target_reposed_vertices[0, :, 0],
                            target_reposed_vertices[0, :, 1],
                            s=0.02,
                            c='blue')
                plt.scatter(pred_reposed_vertices_sc[0, :, 0],
                            pred_reposed_vertices_sc[0, :, 1],
                            s=0.01,
                            c='red')
                plt.gca().set_aspect('equal', adjustable='box')
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                norm = plt.Normalize(vmin=0.0, vmax=0.03, clip=True)
                plt.scatter(pred_reposed_vertices_sc[0, :, 0],
                            pred_reposed_vertices_sc[0, :, 1],
                            s=0.05,
                            c=pvet_scale_corrected_batch[0],
                            cmap='jet',
                            norm=norm)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.text(-0.6, -0.9, s='PVE-T-SC: {:.4f}'.format(per_frame_metrics['pve-ts_sc'][batch_num][0]))
                subplot_count += 1

            # Plot per-vertex uncertainties
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.text(0.5, 0.5, s='Uncertainty for\nPVE')
            subplot_count += 1

            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.gca().invert_yaxis()
            norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
            plt.scatter(pred_vertices_spin[0, :, 0],
                        pred_vertices_spin[0, :, 1],
                        s=0.05,
                        c=avg_vertices_distance_from_mean,
                        cmap='jet',
                        norm=norm)
            plt.gca().set_aspect('equal', adjustable='box')
            subplot_count += 1

            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.gca().invert_yaxis()
            norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
            plt.scatter(pred_vertices_spin[0, :, 2],
                        pred_vertices_spin[0, :, 1],
                        s=0.05,
                        c=avg_vertices_distance_from_mean,
                        cmap='jet',
                        norm=norm)
            plt.gca().set_aspect('equal', adjustable='box')
            subplot_count += 1

            # Plot vertex uncertainties in x/y/z directions
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.text(0.1, 0.1, s='Uncertainty\nfor PVE x/y/z')
            subplot_count += 1

            # x-direction
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.gca().invert_yaxis()
            norm = plt.Normalize(vmin=0.0, vmax=0.1, clip=True)
            plt.scatter(pred_vertices_spin[0, :, 0],
                        pred_vertices_spin[0, :, 1],
                        s=0.05,
                        c=avg_vertices_distance_from_mean_xyz[:, 0],
                        cmap='jet',
                        norm=norm)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.colorbar(shrink=0.5, label='Uncertainty x (m)', orientation='vertical', format='%.2f')
            subplot_count += 1

            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.gca().invert_yaxis()
            norm = plt.Normalize(vmin=0.0, vmax=0.1, clip=True)
            plt.scatter(pred_vertices_spin[0, :, 2],  # Equivalent to Rotated 90° about y axis
                        pred_vertices_spin[0, :, 1],
                        s=0.05,
                        c=avg_vertices_distance_from_mean_xyz[:, 0],
                        cmap='jet',
                        norm=norm)
            plt.gca().set_aspect('equal', adjustable='box')
            subplot_count += 1

            # y-direction
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.gca().invert_yaxis()
            norm = plt.Normalize(vmin=0.0, vmax=0.1, clip=True)
            plt.scatter(pred_vertices_spin[0, :, 0],
                        pred_vertices_spin[0, :, 1],
                        s=0.05,
                        c=avg_vertices_distance_from_mean_xyz[:, 1],
                        cmap='jet',
                        norm=norm)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.colorbar(shrink=0.5, label='Uncertainty y (m)', orientation='vertical', format='%.2f')
            subplot_count += 1

            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.gca().invert_yaxis()
            norm = plt.Normalize(vmin=0.0, vmax=0.1, clip=True)
            plt.scatter(pred_vertices_spin[0, :, 2],  # Equivalent to Rotated 90° about y axis
                        pred_vertices_spin[0, :, 1],
                        s=0.05,
                        c=avg_vertices_distance_from_mean_xyz[:, 1],
                        cmap='jet',
                        norm=norm)
            plt.gca().set_aspect('equal', adjustable='box')
            subplot_count += 1

            # z-direction
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.gca().invert_yaxis()
            norm = plt.Normalize(vmin=0.0, vmax=0.1, clip=True)
            plt.scatter(pred_vertices_spin[0, :, 0],
                        pred_vertices_spin[0, :, 1],
                        s=0.05,
                        c=avg_vertices_distance_from_mean_xyz[:, 2],
                        cmap='jet',
                        norm=norm)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.colorbar(shrink=0.5, label='Uncertainty z (m)', orientation='vertical', format='%.2f')
            subplot_count += 1

            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.gca().invert_yaxis()
            norm = plt.Normalize(vmin=0.0, vmax=0.1, clip=True)
            plt.scatter(pred_vertices_spin[0, :, 2],  # Equivalent to Rotated 90° about y axis
                        pred_vertices_spin[0, :, 1],
                        s=0.05,
                        c=avg_vertices_distance_from_mean_xyz[:, 2],
                        cmap='jet',
                        norm=norm)
            plt.gca().set_aspect('equal', adjustable='box')
            subplot_count += 1

            plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                hspace=0, wspace=0)
            plt.margins(0, 0)
            save_fig_path = os.path.join(save_path, fname[0])
            plt.savefig(save_fig_path, bbox_inches='tight')
            plt.close()

            # ------------------ Samples from Predicted Distribution Figure ------------------
            num_subplots = num_samples_to_visualise * 2 + 2
            num_row = 4
            num_col = math.ceil(num_subplots / float(num_row))

            subplot_count = 1
            plt.figure(figsize=(20, 20))

            # Plot mode prediction
            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(body_vis_rgb_spin)
            subplot_count += 1

            plt.subplot(num_row, num_col, subplot_count)
            plt.gca().axis('off')
            plt.imshow(body_vis_rgb_spin_rot)
            subplot_count += 1

            # Plot samples from predicted distribution
            for i in range(num_samples_to_visualise):
                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.imshow(body_vis_rgb_samples[i])
                subplot_count += 1

                plt.subplot(num_row, num_col, subplot_count)
                plt.gca().axis('off')
                plt.imshow(body_vis_rgb_rot_samples[i])
                subplot_count += 1

            plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            save_fig_path = os.path.join(save_path, os.path.splitext(fname[0])[0] + '_samples.png')
            plt.savefig(save_fig_path, bbox_inches='tight')
            plt.close()

        # ------------------------------- DISPLAY METRICS AND SAVE PER-FRAME METRICS -------------------------------
    print('\n--- Check Pred save Shapes ---')
    fname_per_frame = np.concatenate(fname_per_frame, axis=0)
    np.save(os.path.join(save_path, 'fname_per_frame.npy'), fname_per_frame)
    print(fname_per_frame.shape)

    pose_per_frame = np.concatenate(pose_per_frame, axis=0)
    np.save(os.path.join(save_path, 'pose_per_frame.npy'), pose_per_frame)
    print(pose_per_frame.shape)

    shape_per_frame = np.concatenate(shape_per_frame, axis=0)
    np.save(os.path.join(save_path, 'shape_per_frame.npy'), shape_per_frame)
    print(shape_per_frame.shape)

    cam_per_frame = np.concatenate(cam_per_frame, axis=0)
    np.save(os.path.join(save_path, 'cam_per_frame.npy'), cam_per_frame)
    print(cam_per_frame.shape)

    if vis_every_n_batches is not None and save_per_frame_uncertainty:
        vertices_uncertainty_per_frame = np.stack(vertices_uncertainty_per_frame, axis=0)
        np.save(os.path.join(save_path, 'vertices_uncertainty_per_frame.npy'), vertices_uncertainty_per_frame)
        print(vertices_uncertainty_per_frame.shape)

        vertices_uncertainty_xyz_per_frame = np.stack(vertices_uncertainty_xyz_per_frame, axis=0)
        np.save(os.path.join(save_path, 'vertices_uncertainty_xyz_per_frame.npy'), vertices_uncertainty_xyz_per_frame)
        print(vertices_uncertainty_xyz_per_frame.shape)

    final_metrics = {}
    for metric_type in metrics_to_track:

        if metric_type == 'joints2D_l2es':
            joints2D_l2e = metric_sums['joints2D_l2es'] / metric_sums['num_vis_joints2D']
            final_metrics[metric_type] = joints2D_l2e
            print('Check total samples:', metric_type, metric_sums['num_vis_joints2D'])
        elif metric_type == 'joints2D_l2es_best_j2d_sample':
            joints2D_l2e_best_j2d_sample = metric_sums['joints2D_l2es_best_j2d_sample'] / metric_sums['num_vis_joints2D']
            final_metrics[metric_type] = joints2D_l2e_best_j2d_sample
        elif metric_type == 'joints2Dsamples_l2es':
            joints2Dsamples_l2e = metric_sums['joints2Dsamples_l2es'] / metric_sums['num_vis_joints2Dsamples']
            final_metrics[metric_type] = joints2Dsamples_l2e
            print('Check total samples:', metric_type, metric_sums['num_vis_joints2Dsamples'])

        elif metric_type == 'silhouette_ious':
            iou = metric_sums['num_true_positives'] / \
                  (metric_sums['num_true_positives'] +
                   metric_sums['num_false_negatives'] +
                   metric_sums['num_false_positives'])
            final_metrics['silhouette_ious'] = iou
        elif metric_type == 'silhouettesamples_ious':
            samples_iou = metric_sums['num_samples_true_positives'] / \
                          (metric_sums['num_samples_true_positives'] +
                           metric_sums['num_samples_false_negatives'] +
                           metric_sums['num_samples_false_positives'])
            final_metrics['silhouettesamples_ious'] = samples_iou

        elif metric_type == 'verts_samples_dist_from_mean':
            final_metrics[metric_type] = metric_sums[metric_type] / (metric_sums['num_datapoints'] * num_pred_samples * 6890)
        elif metric_type == 'joints3D_coco_samples_dist_from_mean':
            final_metrics[metric_type] = metric_sums[metric_type] / (metric_sums['num_datapoints'] * num_pred_samples * 17)
        elif metric_type == 'joints3D_coco_invis_samples_dist_from_mean':
            if metric_sums['num_invis_joints3Dsamples'] > 0:
                final_metrics[metric_type] = metric_sums[metric_type] / metric_sums['num_invis_joints3Dsamples']
            else:
                print('No invisible 3D COCO joints!')

        else:
            if 'pves' in metric_type:
                num_per_sample = 6890
            elif 'mpjpes' in metric_type:
                num_per_sample = 14
            # print('Check total samples:', metric_type, num_per_sample, self.total_samples)
            final_metrics[metric_type] = metric_sums[metric_type] / (metric_sums['num_datapoints'] * num_per_sample)

    print('\n---- Metrics ----')
    for metric in final_metrics.keys():
        if final_metrics[metric] > 0.3:
            mult = 1
        else:
            mult = 1000
        print(metric, '{:.2f}'.format(final_metrics[metric] * mult))  # Converting from metres to millimetres

    print('\n---- Check metric save shapes ----')
    for metric_type in metrics_to_track:
        per_frame = np.concatenate(per_frame_metrics[metric_type], axis=0)
        print(metric_type, per_frame.shape)
        np.save(os.path.join(save_path, metric_type + '_per_frame.npy'), per_frame)


if __name__ == '__main__':
    use_subset = False
    num_samples = 100
    extreme_crop = False
    extreme_crop_scale = 0.5

    if extreme_crop:
        exp_dir = '../data/pretrained/ambiguous'
        checkpoint_fname = 'model_epoch_00000013.pth'
    else:
        exp_dir = '../data/pretrained/standard'
        checkpoint_fname = 'model_epoch_00000003.pth'
    print(exp_dir, checkpoint_fname)

    # Set seeds
    np.random.seed(0)
    torch.manual_seed(0)

    # Device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Model
    exp = ExperimentConfig()
    cfg_file = os.path.join(exp_dir, "expconfig.yaml")
    cfg_load = get_config_from_file(cfg_file)
    print("<- Loaded base config settings from: {0}".format(cfg_file))
    parser = get_arg_parser(type(exp), default=cfg_load)
    parsed = parser.parse_args()
    set_config(exp.cfg, vars(parsed))
    # pprint_dict(exp.cfg)
    model = Model(**exp.cfg.MODEL).to(device)
    model_path = os.path.join(exp_dir, checkpoint_fname)
    model_state_dict, stats_load, optimizer_state = load_model(model_path)
    own_state = model.state_dict()
    for name, param in model_state_dict.items():
        # print('\n', name, name[7:], param.shape, own_state[name[7:]].shape)
        try:
            own_state[name[7:]].copy_(param)
        except Exception as e:
            print(e)
            print("Unable to load: {0}".format(name[7:]))
    model.eval()

    # Setup evaluation dataset
    dataset_path = '/scratch/as2562/datasets/ssp_3d'
    dataset = SSP3DEvalDataset(dataset_path,
                               img_wh=constants.IMG_RES,
                               extreme_crop=extreme_crop,
                               extreme_crop_scale=extreme_crop_scale,
                               vis_img_wh=512)
    print("Eval examples found:", len(dataset))

    # Metrics
    metrics = ['pves', 'pves_sc', 'pves_pa', 'pve-ts', 'pve-ts_sc']
    metrics.extend([metric + '_samples_min' for metric in metrics])
    metrics.extend(['verts_samples_dist_from_mean', 'joints3D_coco_samples_dist_from_mean', 'joints3D_coco_invis_samples_dist_from_mean'])
    metrics.append('joints2D_l2es')
    metrics.append('joints2Dsamples_l2es')
    metrics.append('silhouette_ious')
    metrics.append('silhouettesamples_ious')

    save_path = '/scratch/as2562/3D-Multibodies/evaluations/ssp3d_{}_samples'.format(num_samples)
    if extreme_crop:
        save_path += '_extreme_crop_scale_{}'.format(extreme_crop_scale)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("Saving to:", save_path)

    # Run evaluation
    evaluate_ssp3d(model=model,
                   eval_dataset=dataset,
                   metrics_to_track=metrics,
                   device=device,
                   save_path=save_path,
                   num_pred_samples=num_samples,
                   num_workers=4,
                   pin_memory=True,
                   vis_every_n_batches=1000,
                   vis_img_wh=512,
                   num_samples_to_visualise=10,
                   save_per_frame_uncertainty=True,
                   extreme_crop=extreme_crop)







