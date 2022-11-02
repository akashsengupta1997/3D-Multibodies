import torch
# from torch.nn import Parameter
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import numpy as np
#
# import time

from exp_manager.base_model import BaseModel
from exp_manager.utils import auto_init_args
from exp_manager.config import get_default_args

import os
import networks.smpl.smpl as smpl
from networks.spin import ConditionalHMR
# from networks.benvp.benvp import RealNVP
# import torchgeometry as tgm

from losses import Losses
from accuracies import AccuracyMetrics
import config as global_config
from utils import constants
# from utils.kmeans import GPUKmeans
# from utils.skeleton import draw_skeleton

from utils.geometry import perspective_projection, batch_rodrigues
from utils.minofn import MinofN


class Model(BaseModel):
    def __init__(self,
                 loss_weights={
                     'loss_joint': 25.0, 'loss_angle': 1.0, 'loss_shape': 0.001,
                     'loss_skel2d': 1.0, 'loss_vertex': 0.0,
                     'loss_nll': 1e-3, 'loss_nll_reg': 1e-3,
                     'loss_skel2d_modewise': 0.1, 'loss_depth_modewise': 1.0
                 },
                 num_modes=25,
                 init_flow="",
                 COND_HMR=get_default_args(ConditionalHMR),
                 openpose_train_weight=0.0,
                 gt_train_weight=1.0,
                 log_vars=[
                     'objective',
                     'loss_nll', 'loss_nll_reg',
                     'loss_joint', 'loss_angle', 'loss_shape',
                     'loss_skel2d', 'loss_skel2d_modewise', 'loss_vertex',
                     'acc_h36m_M01_mpjpe',
                     'acc_h36m_M05_mpjpe',
                     'acc_h36m_WM05_mpjpe',
                     'acc_h36m_M10_mpjpe',
                     'acc_h36m_WM10_mpjpe',
                     'acc_h36m_M25_mpjpe',
                     'acc_h36m_WM25_mpjpe',
                     'acc_h36m_M100_mpjpe',
                     'dur_acc', 'dur_flow', 'dur_hmr', 'dur_losses',
                     'dur_minofn', 'dur_smpl'
                 ],
                 **kwargs):

        auto_init_args(self)

        super(Model, self).__init__()

        self.focal_length = constants.FOCAL_LENGTH
        d = 24 * 3
        n_hidden = 512
        num_transformations = 20
        dropout = 0.2

        # self.num_dims_flow = d

        # prior_mean = torch.zeros(d)
        # prior_var = torch.eye(d)
        # masks = (torch.rand(num_transformations, d) < 0.5).float()
        # nets = lambda: nn.Sequential(
        #     nn.Linear(d, n_hidden), nn.Dropout(dropout), nn.LeakyReLU(),
        #     nn.Linear(n_hidden, n_hidden), nn.Dropout(dropout), nn.LeakyReLU(),
        #     nn.Linear(n_hidden, d), nn.Tanh())
        #
        # nett = lambda: nn.Sequential(
        #     nn.Linear(d, n_hidden), nn.Dropout(dropout), nn.LeakyReLU(),
        #     nn.Linear(n_hidden, n_hidden), nn.Dropout(dropout), nn.LeakyReLU(),
        #     nn.Linear(n_hidden, d))

        # self.model_realnvp = RealNVP(nets, nett, masks, prior_mean, prior_var)
        self.model = ConditionalHMR(
            num_modes,
            num_transformations,
            **COND_HMR)

        self.accuracy_metrics = AccuracyMetrics()

        print("----- SPIN Layers to Optimize -----")
        for name_s, param_s in self.model.named_parameters():
            if param_s.requires_grad:
                print(name_s)

        neutral_model_path = os.path.join(global_config.SMPL_MODEL_DIR, 'SMPL_NEUTRAL.pkl')
        # self.smpl_mini = smpl.SMPL(neutral_model_path)
        self.losses = Losses()

    def forward(self, images, cache_mode=False):
        batch_size, _, _, _ = images.shape
        device = images.device

        hmr_pred = self.model(images)
        # print('\nHMR PRED')
        # for key in hmr_pred:
        #     print(key, hmr_pred[key].shape)

        return hmr_pred

        # pred_shape = hmr_pred['pred_shape']
        # pred_camera = hmr_pred['pred_camera']
        # pred_global_rotmat = hmr_pred['pred_global_rotmat']
        # pred_pose_axis = MinofN.compress_modes_into_batch(
        #     hmr_pred['pred_local_axis'])
        #
        # pred_rotmat = batch_rodrigues(pred_pose_axis.contiguous().view(-1, 3)).view(
        #     batch_size, self.num_modes, 23, 3, 3)

        # out_fpose_mode: (N, M, 24, 3, 3)
        # out_fpose_mode = torch.cat([pred_global_rotmat, pred_rotmat], dim=2)

        # # Run all modes through SMPL, after compressing
        # # (N, M, ...) -> (N * M, ...)
        # out_fpose_mode_compressed = MinofN.compress_modes_into_batch(out_fpose_mode)
        # out_shape_compressed = MinofN.compress_modes_into_batch(pred_shape)
        # out_verts_mode, out_model_joints_mode, out_model_pelvis = self.smpl_mini(
        #     out_fpose_mode_compressed,
        #     out_shape_compressed,
        #     run_mini=True)
        #
        # out_fpose_mode_compressed_zero = out_fpose_mode_compressed.clone()
        # out_fpose_mode_compressed_zero[:, 0] = torch.eye(3).to(
        #     out_fpose_mode_compressed_zero.device)
        # _, out_model_joints_mode_zero, _ = self.smpl_mini(
        #     out_fpose_mode_compressed_zero,
        #     out_shape_compressed,
        #     run_mini=True)

        # # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # # This camera translation can be used in a full perspective projection
        # out_camera_mode = MinofN.compress_modes_into_batch(pred_camera)
        # tz = 2 * self.focal_length / (global_config.IMG_RES * out_camera_mode[:, 0] + 1e-9)
        # pred_cam_t = torch.stack([out_camera_mode[:, 1],
        #                           out_camera_mode[:, 2],
        #                           tz], dim=-1)
        #
        # camera_center = torch.zeros(batch_size * self.num_modes, 2)
        # camera_rotation = torch.eye(3).unsqueeze(0).expand(
        #     batch_size * self.num_modes, -1, -1).to(device)

        # # (N * M, 49, 2)
        # projected_joints_mode = perspective_projection(
        #     out_model_joints_mode + out_model_pelvis,
        #     rotation=camera_rotation,
        #     translation=pred_cam_t,
        #     focal_length=self.focal_length,
        #     camera_center=camera_center)


        # ######## Identfy and select the min-of-n modes ########
        # mean_ids = torch.zeros(batch_size, dtype=int).to(device)
        # if self.num_modes > 1:
        #     minofn_mode_ids, _ = self.losses.select_modes(
        #         out_model_joints_mode,  # Predicted joints (N * M, 49, 3)
        #         MinofN.expand_and_compress_modes(gt_joints, self.num_modes),  # GT Sensor data
        #         MinofN.expand_and_compress_modes(gt_model_joints, self.num_modes),  # GT SMPL data
        #         MinofN.expand_and_compress_modes(has_pose_3d, self.num_modes),
        #         MinofN.expand_and_compress_modes(has_smpl, self.num_modes),
        #         self.num_modes
        #     )
        # else:
        #     minofn_mode_ids = mean_ids

        # out_model_joints_reshape = MinofN.decompress_modes_from_batch(
        #     out_model_joints_mode, batch_size, self.num_modes)
        #
        # out_verts_mode_reshape = MinofN.decompress_modes_from_batch(
        #     out_verts_mode, batch_size, self.num_modes)
        #
        # out_pelvis_mode_reshape = MinofN.decompress_modes_from_batch(
        #     out_model_pelvis, batch_size, self.num_modes)
        #
        # out_projected_joints_reshape = MinofN.decompress_modes_from_batch(
        #     projected_joints_mode, batch_size, self.num_modes)
        #
        # assert out_fpose_mode.shape[0] == \
        #        pred_camera.shape[0] == \
        #        pred_shape.shape[0] == \
        #        out_verts_mode_reshape.shape[0] == \
        #        out_model_joints_reshape.shape[0] == \
        #        out_projected_joints_reshape.shape[0], "Batch sizes don't match"

        # preds = {}


        # ######## Compute accuracy metrics ########
        #
        # with torch.no_grad():
        #     if cache_mode:
        #         dataset_eval_keys = [float(dataset_key[0])]
        #     else:
        #         dataset_eval_keys = global_config.DATASET_EVAL_KEYS
        #
        #     if not self.training:
        #         use_weight = {
        #             1: [False]
        #         }
        #
        #         for extras in [5, 10, 25]:
        #             if extras <= self.num_modes:
        #                 use_weight[extras] = [True, False]
        #     else:
        #         use_weight = {
        #             self.num_modes: [False]
        #         }
        #
        #     for eval_mode, weight_list in use_weight.items():
        #         for use_weight in weight_list:
        #             if eval_mode == 1:
        #                 eval_mode_ids = mean_ids
        #                 eval_fpose = out_fpose_mode[:, 0]
        #                 eval_betas = pred_shape[:, 0]
        #             elif eval_mode == self.num_modes:
        #                 eval_mode_ids = minofn_mode_ids
        #                 eval_fpose = out_fpose_mode[batch_ids, minofn_mode_ids]
        #                 eval_betas = pred_shape[batch_ids, minofn_mode_ids]
        #             else:
        #                 out_model_joints_vector = out_model_joints_reshape[:, :, 25:]
        #                 out_model_pelvis = (out_model_joints_vector[:, :, 2, :] + out_model_joints_vector[:, :, 3, :]) / 2
        #                 out_model_joints_vector = out_model_joints_vector - out_model_pelvis[:, :, None, :]
        #
        #                 log_weights = log_liklihood_reg
        #                 if not use_weight:
        #                     log_weights = torch.zeros_like(log_weights)
        #
        #                 # Cluster M-1 modes, and add in the mean (M=0)
        #                 start_kmeans = time.time()
        #                 km = GPUKmeans(K=eval_mode - 1)
        #                 res = km(
        #                     out_model_joints_vector.reshape(
        #                         batch_size, self.num_modes, -1)[:, 1:],
        #                     log_W=log_weights[:, 1:],
        #                     verbose=False)
        #
        #                 end_kmeans = time.time()
        #                 kmeans_time = end_kmeans - start_kmeans
        #                 print("KMEANS TIME: {0}, {1}".format(kmeans_time, eval_mode))
        #
        #                 representatives_idx = res['representatives_idx'] + 1
        #                 representatives_idx = torch.cat([
        #                     mean_ids[:, None], representatives_idx], dim=1)
        #
        #                 out_model_joints_rep = torch.gather(
        #                     out_model_joints_reshape,
        #                     1,
        #                     representatives_idx[:, :, None, None].expand(
        #                         -1, -1, *out_model_joints_reshape.shape[2:]))
        #
        #                 eval_mode_ids = representatives_idx
        #                 minofn_ids_eval, accuracy_ids = self.losses.select_modes(
        #                     MinofN.compress_modes_into_batch(out_model_joints_rep),  # Predicted joints (N * M, 49, 3)
        #                     MinofN.expand_and_compress_modes(gt_joints, eval_mode),  # GT Sensor data
        #                     MinofN.expand_and_compress_modes(gt_model_joints, eval_mode),  # GT SMPL data
        #                     MinofN.expand_and_compress_modes(has_pose_3d, eval_mode),
        #                     MinofN.expand_and_compress_modes(has_smpl, eval_mode),
        #                     eval_mode
        #                 )
        #
        #                 eval_mode_id = torch.gather(
        #                     representatives_idx,
        #                     1,
        #                     minofn_ids_eval[:, None]
        #                 )
        #                 eval_fpose = torch.gather(
        #                     out_fpose_mode,
        #                     1,
        #                     eval_mode_id[:, :, None, None, None].expand(
        #                         -1, -1, *out_fpose_mode.shape[2:]
        #                     )
        #                 )[:, 0]
        #                 eval_betas = torch.gather(
        #                     pred_shape,
        #                     1,
        #                     eval_mode_id[:, :, None].expand(
        #                         -1, -1, *pred_shape.shape[2:]
        #                     )
        #                 )[:, 0]
        #
        #             results_all_datasets = self.accuracy_metrics.compute_accuracies(
        #                 eval_fpose, eval_betas,
        #                 gt_rotmat, gt_betas,
        #                 gt_joints, gender,
        #                 dataset_key, dataset_eval_keys)
        #
        #             if use_weight:
        #                 use_weight_str = "W"
        #             else:
        #                 use_weight_str = ""
        #
        #             eval_name = "{0}M{1:02d}_ids".format(
        #                 use_weight_str, eval_mode)
        #             preds[eval_name] = eval_mode_ids
        #
        #             for d_name, results in results_all_datasets.items():
        #                 if results is not None:
        #                     metrics, result_data = results
        #                     for metric_name, metric_val in metrics.items():
        #                         if metric_val is not None:
        #                             if cache_mode:
        #                                 acc_name = "{0}M{1:02d}_{2}".format(
        #                                     use_weight_str, eval_mode, metric_name)
        #                             else:
        #                                 acc_name = "acc_{0}_{1}M{2:02d}_{3}".format(
        #                                     global_config.DATASET_LIST[d_name],
        #                                     use_weight_str,
        #                                     eval_mode,
        #                                     metric_name)
        #
        #                             preds[acc_name] = metric_val
        #
        #                 if cache_mode:
        #                     for result_name, result_datum in result_data.items():
        #                         assert result_datum.shape[0] == batch_size, "Cache dataset should be of a single dataset type"
        #                         preds[result_name] = result_datum

        # preds['out_joints_mode'] = out_model_joints_reshape
        # preds['out_verts_mode'] = out_verts_mode_reshape
        # preds['out_pelvis_mode'] = out_pelvis_mode_reshape
        # preds['out_fpose_mode'] = out_fpose_mode
        # preds['out_shape_mode'] = pred_shape
        # preds['out_projkps_mode'] = out_projected_joints_reshape
        # preds['out_cam_t'] = pred_cam_t.reshape(batch_size, self.num_modes, -1)
        # preds['out_cam_wp'] = pred_camera
        # preds['out_pose_rotmats'] = pred_rotmat
        # preds['out_glob_rotmat'] = pred_global_rotmat
        # preds['min_mode_gt'] = minofn_mode_ids

        # return preds

