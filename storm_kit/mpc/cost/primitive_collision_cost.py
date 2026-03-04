#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
import torch
import torch.nn as nn
# import torch.nn.functional as F
from ...geom.sdf.robot_world import RobotWorldCollisionPrimitive
from .gaussian_projection import GaussianProjection

class PrimitiveCollisionCost(nn.Module):
    def __init__(self, weight=None, world_params=None, robot_params=None, gaussian_params={},
                 distance_threshold=0.1,
                 violation_clip_max=0.2,
                 violation_scale=0.25,
                 tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32}):
        super(PrimitiveCollisionCost, self).__init__()
        
        self.tensor_args = tensor_args
        self.weight = torch.as_tensor(weight,**self.tensor_args)
        
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)

        robot_collision_params = robot_params['robot_collision_params']
        self.batch_size = -1
        # BUILD world and robot:
        self.robot_world_coll = RobotWorldCollisionPrimitive(robot_collision_params,
                                                             world_params['world_model'],
                                                             tensor_args=self.tensor_args,
                                                             bounds=robot_params['world_collision_params']['bounds'],
                                                             grid_resolution=robot_params['world_collision_params']['grid_resolution'])
        
        self.n_world_objs = self.robot_world_coll.world_coll.n_objs
        self.t_mat = None
        self.distance_threshold = distance_threshold
        self.violation_clip_max = float(violation_clip_max)
        self.violation_scale = float(violation_scale)
        self._dynamic_world_spheres = None
        self._empty_world_dist = None
        self._dynamic_link_dist = None
        if self.violation_clip_max <= 0.0:
            raise ValueError(f"violation_clip_max must be > 0, got {self.violation_clip_max}.")
        if self.violation_scale <= 0.0:
            raise ValueError(f"violation_scale must be > 0, got {self.violation_scale}.")

    @staticmethod
    def _shape_violation(
        dist,
        distance_threshold: float,
        violation_clip_max: float,
        violation_scale: float,
    ):
        dist = dist + float(distance_threshold)
        dist[dist <= 0.0] = 0.0
        dist[dist > float(violation_clip_max)] = float(violation_clip_max)
        dist = dist / float(violation_scale)
        return torch.sum(dist, dim=-1)

    def set_dynamic_world_spheres(self, dynamic_world_spheres):
        if dynamic_world_spheres is None:
            return False

        dynamic_world_spheres = torch.as_tensor(dynamic_world_spheres, **self.tensor_args)
        if dynamic_world_spheres.numel() == 0:
            dynamic_world_spheres = dynamic_world_spheres.reshape(0, 4)
        if dynamic_world_spheres.ndim != 2 or dynamic_world_spheres.shape[1] != 4:
            raise ValueError(
                f"dynamic_world_spheres must have shape [N, 4], got {tuple(dynamic_world_spheres.shape)}"
            )

        if (
            self._dynamic_world_spheres is None
            or self._dynamic_world_spheres.shape[0] != dynamic_world_spheres.shape[0]
        ):
            self._dynamic_world_spheres = torch.zeros(dynamic_world_spheres.shape, **self.tensor_args)
        self._dynamic_world_spheres.copy_(dynamic_world_spheres)
        return True

    def _compute_dynamic_collision_dist(self, batch_size, n_links):
        if self._dynamic_world_spheres is None or self._dynamic_world_spheres.shape[0] == 0:
            return None
        link_spheres = self.robot_world_coll.robot_coll.get_batch_robot_link_spheres()
        if link_spheres is None:
            return None

        if self._dynamic_link_dist is None or self._dynamic_link_dist.shape != (batch_size, n_links):
            self._dynamic_link_dist = torch.zeros((batch_size, n_links), **self.tensor_args)
        dist = self._dynamic_link_dist

        world_centers = self._dynamic_world_spheres[:, :3].view(1, 1, -1, 3)
        world_radii = self._dynamic_world_spheres[:, 3].view(1, 1, -1)

        for link_idx in range(n_links):
            robot_spheres = link_spheres[link_idx]
            robot_centers = robot_spheres[:, :, :3]
            robot_radii = robot_spheres[:, :, 3].unsqueeze(-1)
            center_distance = torch.norm(robot_centers.unsqueeze(2) - world_centers, dim=-1)
            signed_dist = world_radii - center_distance + robot_radii
            max_over_world = torch.max(signed_dist, dim=-1)[0]
            dist[:, link_idx] = torch.max(max_over_world, dim=-1)[0]
        return dist

    def _compute_signed_link_distance_batch(self, link_pos_batch, link_rot_batch):
        batch_size = link_pos_batch.shape[0]
        n_links = link_pos_batch.shape[1]

        if self.n_world_objs > 0:
            dist = self.robot_world_coll.check_robot_sphere_collisions(link_pos_batch, link_rot_batch)
        else:
            self.robot_world_coll.robot_coll.update_batch_robot_collision_objs(link_pos_batch, link_rot_batch)
            if (
                self._empty_world_dist is None
                or self._empty_world_dist.shape[0] != batch_size
                or self._empty_world_dist.shape[1] != n_links
            ):
                self._empty_world_dist = torch.zeros((batch_size, n_links), **self.tensor_args) - 10.0
            dist = self._empty_world_dist

        dynamic_dist = self._compute_dynamic_collision_dist(batch_size, n_links)
        if dynamic_dist is not None:
            dist = torch.maximum(dist, dynamic_dist)
        return dist

    def forward(
        self,
        link_pos_seq,
        link_rot_seq,
        return_violation=False,
        violation_distance_threshold=None,
        violation_clip_max=None,
        violation_scale=None,
    ):

        
        inp_device = link_pos_seq.device
        batch_size = link_pos_seq.shape[0]
        horizon = link_pos_seq.shape[1]
        n_links = link_pos_seq.shape[2]

        if(self.batch_size != batch_size):
            self.batch_size = batch_size
            self.robot_world_coll.build_batch_features(self.batch_size * horizon, clone_pose=True, clone_points=True)

        link_pos_batch = link_pos_seq.view(batch_size * horizon, n_links, 3)
        link_rot_batch = link_rot_seq.view(batch_size * horizon, n_links, 3, 3)
        dist = self._compute_signed_link_distance_batch(link_pos_batch, link_rot_batch)
        dist = dist.view(batch_size, horizon, n_links)  # , self.n_world_objs)
        violation_default = self._shape_violation(
            dist.clone(),
            self.distance_threshold,
            self.violation_clip_max,
            self.violation_scale,
        )

        cost = self.weight * violation_default
        cost = cost.to(inp_device)

        if return_violation:
            if violation_distance_threshold is None:
                threshold = self.distance_threshold
            else:
                threshold = float(violation_distance_threshold)

            if violation_clip_max is None:
                clip_max = self.violation_clip_max
            else:
                clip_max = float(violation_clip_max)
                if clip_max <= 0.0:
                    raise ValueError(f"violation_clip_max must be > 0, got {clip_max}.")

            if violation_scale is None:
                scale = self.violation_scale
            else:
                scale = float(violation_scale)
                if scale <= 0.0:
                    raise ValueError(f"violation_scale must be > 0, got {scale}.")

            if (
                threshold == self.distance_threshold
                and clip_max == self.violation_clip_max
                and scale == self.violation_scale
            ):
                violation_unweighted = violation_default
            else:
                violation_unweighted = self._shape_violation(
                    dist,
                    threshold,
                    clip_max,
                    scale,
                )
            return cost, violation_unweighted.to(inp_device)

        return cost
