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
from .gaussian_projection import GaussianProjection

class BoundCost(nn.Module):
    def __init__(self, tensor_args={'device':torch.device('cpu'), 'dtype':torch.float64},
                 bounds=[], weight=1.0, gaussian_params={}, bound_thresh=0.1):
        super(BoundCost, self).__init__()
        self.tensor_args = tensor_args
        self.weight = torch.as_tensor(weight, **tensor_args)
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)

        self.bounds = torch.as_tensor(bounds, **tensor_args)
        self.raw_bounds = self.bounds.clone()
        self.bnd_range = (self.raw_bounds[:,1] - self.raw_bounds[:,0]) / 2.0
        self.t_mat = None
        self.bound_thresh = float(bound_thresh)
        bound_thresh_abs = self.bound_thresh * self.bnd_range
        self.bounds[:,1] = self.raw_bounds[:,1] - bound_thresh_abs
        self.bounds[:,0] = self.raw_bounds[:,0] + bound_thresh_abs

    def _compute_violation(self, state_batch, lower, upper):
        bound_mask = torch.logical_and(state_batch < upper, state_batch > lower)
        sq = torch.minimum(torch.square(state_batch - lower), torch.square(upper - state_batch))
        sq[bound_mask] = 0.0
        return torch.sqrt(torch.sum(sq, dim=-1))

    def forward(self, state_batch, return_violation=False, violation_bound_thresh=None):
        inp_device = state_batch.device

        violation_default = self._compute_violation(state_batch, self.bounds[:,0], self.bounds[:,1])
        cost = self.weight * self.proj_gaussian(violation_default)
        cost = cost.to(inp_device)

        if return_violation:
            if violation_bound_thresh is None:
                violation_unweighted = violation_default
            else:
                violation_bound_thresh = float(violation_bound_thresh)
                violation_thresh_abs = violation_bound_thresh * self.bnd_range
                lower = self.raw_bounds[:,0] + violation_thresh_abs
                upper = self.raw_bounds[:,1] - violation_thresh_abs
                violation_unweighted = self._compute_violation(state_batch, lower, upper)
            return cost, violation_unweighted.to(inp_device)

        return cost
