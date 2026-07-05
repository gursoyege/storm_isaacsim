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
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
# import torch.nn.functional as F
from .gaussian_projection import GaussianProjection
from ..model.integration_utils import build_fd_matrix
class FiniteDifferenceCost(nn.Module):
    def __init__(self, tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32}, weight=1.0, order=1, gaussian_params={}, vec_weight=None, **kwargs):
        super(FiniteDifferenceCost, self).__init__()

        self.order = order
        for _ in range(order):
            weight *= weight
        self.weight = weight
        self.tensor_args = tensor_args
        # Tier 2d: optional per-DOF weight (e.g. upweight low-torque-limit wrist joints).
        # None means "all ones", preserved as a tensor lazily once forward() knows d_act,
        # since the model dof count isn't known yet at construction time.
        self._vec_weight_cfg = vec_weight
        self.vec_weight = None
        # build FD matrix
        
        self.fd_mat = None
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)
        self.t_mat = None
    def forward(self, ctrl_seq, dt):
        """
        ctrl_seq: [B X H X d_act]
        """
        dt[dt == 0.0] = 0.0 #dt[-1]
        dt = 1 / dt
        
        #dt = dt / torch.max(dt)
        dt = torch.abs(dt)
        
        #print(dt)
        dt[dt == float("Inf")] = 0

        dt[dt > 10] = 10
        #dt = dt / torch.max(dt)
        
        dt[dt != dt] = 0.0
        #for _ in range(self.order-1):
        #    dt = dt * dt
        #print(dt)
        inp_device = ctrl_seq.device
        ctrl_seq = ctrl_seq.to(**self.tensor_args)
        
        B, H, _ = ctrl_seq.shape
        H = H - self.order
        dt = dt[:H]
        #
        if(self.fd_mat is None or self.fd_mat.shape[0] != H):
            self.fd_mat = build_fd_matrix(H,device=self.tensor_args['device'], dtype=self.tensor_args['dtype'], order=self.order, PREV_STATE=True)
            
        
        
        diff = torch.matmul(self.fd_mat,ctrl_seq)

        res = torch.abs(diff)

        # NOTE: upstream STORM did `cost = res[:,:,-1]` here, which only ever reads the *last*
        # d_act index -- for cost.smooth (called with the velocity state slice, d_act=n_dofs)
        # that means only the last joint (the wrist) was ever penalized; joints 1-(n_dofs-1)
        # silently contributed nothing. Verified empirically with synthetic per-joint ramps.
        # Sum across all DOFs (optionally per-DOF weighted) instead.
        d_act = res.shape[-1]
        if self.vec_weight is None or self.vec_weight.shape[0] != d_act:
            if self._vec_weight_cfg is None:
                vw = torch.ones(d_act, **self.tensor_args)
            else:
                vw = torch.as_tensor(self._vec_weight_cfg, **self.tensor_args)
                if vw.numel() != d_act:
                    raise ValueError(f"FiniteDifferenceCost vec_weight has {vw.numel()} entries, expected {d_act}.")
            self.vec_weight = vw
        cost = torch.sum(res * self.vec_weight, dim=-1)

        cost[cost < 0.0001] = 0.0
        cost = self.weight * cost 
        
        
        return cost
