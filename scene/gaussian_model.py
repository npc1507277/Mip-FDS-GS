#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
import numpy as np

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from torch_kdtree import build_kd_tree
import torch.nn.functional as F


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.R_activation = torch.exp
        self.inverse_R_activation = torch.log

        #self.scaling_activation = lambda x: F.softmax(x, dim=1)  # dim=1表示对行操作
        self.scaling_activation = torch.sigmoid
        self.scaling_inverse_activation = inverse_sigmoid

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._R = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.cam_count = 0
        self.filter_3D = None
        

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._R,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._R,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    @property
    def size(self):
        return self.get_xyz.shape[0]

    @property
    def get_scaling(self):
        scales = self.scaling_activation(self._scaling) * self.R_activation(self._R)
        scales_filter =  torch.square(scales) + torch.square(self.filter_3D).unsqueeze(-1)
        #scales_filter =  scales + self.filter_3D.unsqueeze(-1)
        scales = torch.sqrt(scales_filter)
        #scales = scales_filter
        
        return scales

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        
        opacity = self.opacity_activation(self._opacity)
        scales = self.scaling_activation(self._scaling) * self.R_activation(self._R)
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        scales_after_square = scales_square + torch.square(self.filter_3D).unsqueeze(-1) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @property
    def get_rotation_matrix(self):
        r, i, j, k = torch.unbind(self._rotation, -1)
        two_s = 2.0 / (self._rotation * self._rotation).sum(-1)
        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(self._rotation.shape[:-1] + (3, 3))

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        R = self.inverse_R_activation(1.5 * torch.sqrt(dist2)[..., None])
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._R = nn.Parameter(R.requires_grad_(True))
        self._scaling = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 3), device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.filter_3D = torch.zeros(self._xyz.shape[0], device=self._xyz.device)

    def training_setup(self, training_args):

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._R], 'lr': training_args.R_lr, "name": "R"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        l.append('R')
        for i in range(self.get_scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('filter_3D')
        return l


    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        R = self._R.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        filter_3D = self.filter_3D.unsqueeze(-1).detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, R, scale, rotation, filter_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):

        # reset opacity to by considering 3D filter
        current_opacity_with_filter = self.get_opacity
        opacities_new = torch.min(current_opacity_with_filter, torch.ones_like(current_opacity_with_filter)*0.01)
        # apply 3D filter
        scales = self.scaling_activation(self._scaling) * self.R_activation(self._R)
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        scales_after_square = scales_square + torch.square(self.filter_3D).unsqueeze(-1) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        opacities_new = opacities_new / coef[..., None]
        opacities_new = inverse_sigmoid(opacities_new)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    @torch.no_grad()
    def compute_radius(self, iteration):
        xyz = self.get_xyz.detach()
        torch_kdtree = build_kd_tree(xyz)
        k = 50
        distances, indices = torch_kdtree.query(xyz, k)
        distances = distances[:, 1:]
        distances = torch.sqrt(distances)
        scale_factor = torch.median(distances[:, 0])
        weights = torch.exp(-((distances - distances[:, :1]) / scale_factor) ** 2)
        weighted_R = torch.sum(distances * weights, dim=1) / torch.sum(weights, dim=1)
        activated_R = self.R_activation(self._R).detach()
        if iteration > 1000:
            ratio = (activated_R.squeeze() / weighted_R).detach().mean()
            return ratio * weighted_R
        else:
            return 1.2 * weighted_R


    def reset_R(self, iteration):
        R_new = self.inverse_R_activation(self.compute_radius(iteration)[..., None])
        optimizable_tensors = self.replace_tensor_to_optimizer(R_new, "R")
        self._R = optimizable_tensors["R"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        R = np.zeros((xyz.shape[0], 1))
        R[:, 0] = np.asarray(plydata.elements[0]["R"])

        filter_3D = np.zeros((xyz.shape[0])) 
        filter_3D = np.asarray(plydata.elements[0]["filter_3D"]).squeeze()

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._R = nn.Parameter(torch.tensor(R, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.filter_3D = torch.tensor(filter_3D, dtype=torch.float, device="cuda")
        self.active_sh_degree = self.max_sh_degree

    def compute_scale_loss(self):
        min_scale = torch.min(self.get_scaling, dim=1).values
        return torch.mean(min_scale)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._R = optimizable_tensors["R"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.filter_3D = self.filter_3D[valid_points_mask]
        

    def random_drop_points(self, drop_percent):
        mask = torch.rand((self.get_xyz.shape[0]), device="cuda") < drop_percent
        self.prune_points(mask)
        return mask

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densify_points(self, d):
        gen_filter_3D = d.pop("filter_3D")
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._R = optimizable_tensors["R"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        former_filter_3D = self.filter_3D
        self.filter_3D =  torch.zeros(self.size, device="cuda")
        self.filter_3D[:former_filter_3D.shape[0]] = former_filter_3D
        self.filter_3D[former_filter_3D.shape[0]:] = gen_filter_3D.squeeze()
        

    
    @torch.no_grad()
    def compute_filter_3D(self, cameras):
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0],2), device=xyz.device) * 10000.0
        filter_3D = torch.zeros(xyz.shape[0], device=xyz.device)
        for camera in cameras:
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
            
            xyz_cam = xyz @ R + T[None, :]
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            valid_depth = xyz_cam[:, 2] > 0.2
            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))   
            mask = torch.logical_and(valid_depth, in_screen)
            distance[mask,1] = ((distance[mask,0] > (z[mask]/camera.focal_x)) * camera.uid).float()
            distance[mask,0] = torch.min(distance[mask,0], z[mask]/camera.focal_x)

        if self.filter_3D == None:
            self.filter_3D = torch.zeros(xyz.shape[0], device=xyz.device)
            
        mask = distance[:,1] < 10000.0
        self.filter_3D[mask] = distance[mask,0] * (0.2**0.5)
    


    def densify_rand_organized(self, densify_mask, K=2, s_K=2):
        # K is the number of samples per point
        # s_K is the number of actual points per point to save
        stds = self.get_scaling[densify_mask].unsqueeze(1).repeat(1, K, 1)  # [N, K, 3]
        means = torch.zeros(stds.shape, device="cuda")  # [N, K, 3]
        samples = torch.normal(mean=means, std=stds)  # [N, K, 3]
        rots = build_rotation(self._rotation[densify_mask]).unsqueeze(1).repeat(1, K, 1, 1)  # [N, K, 3, 3]
        new_xyz = (
                torch.matmul(rots, samples.unsqueeze(-1)).squeeze(-1) +
                self.get_xyz[densify_mask].unsqueeze(1).repeat(1, K, 1)
        )
        new_R = self.inverse_R_activation(
            self.R_activation(self._R[densify_mask]).unsqueeze(1).repeat(1, s_K, 1) / (0.8 * s_K)
        )
        new_scaling = self._scaling[densify_mask].unsqueeze(1).repeat(1, s_K, 1)
        new_rotation = self._rotation[densify_mask].unsqueeze(1).repeat(1, s_K, 1)
        new_features_dc = self._features_dc[densify_mask].unsqueeze(1).repeat(1, s_K, 1, 1)
        new_features_rest = self._features_rest[densify_mask].unsqueeze(1).repeat(1, s_K, 1, 1)
        new_opacity = self._opacity[densify_mask].unsqueeze(1).repeat(1, s_K, 1)
        new_filter_3D = self.filter_3D[densify_mask].unsqueeze(1).repeat(1, s_K, 1) 
    
        d = {
            # "xyz": new_xyz,  # [N, K, 3]
            "xyz": new_xyz.view(-1, 3),  # [N * K, 3]
            "R": new_R.view(-1, 1),  # [N * s_K, 1]
            "f_dc": new_features_dc.view(-1, 1, 3),  # [N * s_K, 1, 3]
            "f_rest": new_features_rest.view(-1, self._features_rest.shape[1], 3),  # [N * s_K, F, 3]
            "opacity": new_opacity.view(-1, 1),  # [N * s_K, 1]
            "scaling": new_scaling.view(-1, 3),  # [N * s_K, 3]
            "rotation": new_rotation.view(-1, 4),  # [N * s_K, 4]
            "filter_3D": new_filter_3D.view(-1, 1)  # [N * s_K, 1]
        }
        return d
