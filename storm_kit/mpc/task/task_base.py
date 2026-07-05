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
import numpy as np

from ...mpc.utils.state_filter import JointStateFilter
from ...mpc.utils.mpc_process_wrapper import ControlProcess

class BaseTask(): 
    def __init__(self, tensor_args={'device':"cpu", 'dtype':torch.float32}):
        self.tensor_args = tensor_args
        self.prev_qdd_des = None
    def init_aux(self):
        self.state_filter = JointStateFilter(filter_coeff=self.exp_params['state_filter_coeff'], dt=self.exp_params['control_dt'])

        self.command_filter = JointStateFilter(filter_coeff=self.exp_params['cmd_filter_coeff'], dt=self.exp_params['control_dt'])
        self.control_process = ControlProcess(self.controller)
        self.n_dofs = self.controller.rollout_fn.dynamics_model.n_dofs
        self.zero_acc = np.zeros(self.n_dofs)
        # Tier 3a: command_filter (above) was constructed but never invoked anywhere in this
        # file or anywhere else in storm_kit -- confirmed by grep across the library. Opt-in
        # only (default off, identical behavior to before) since the existing default
        # cmd_filter_coeff.acceleration=0.0 would freeze the published acceleration at its
        # initial value forever if this were silently turned on under an unchanged config.
        self.enable_command_filter = bool(self.exp_params.get('enable_command_filter', False))
        if self.enable_command_filter and float(self.exp_params['cmd_filter_coeff'].get('acceleration', 0.0)) <= 0.0:
            raise ValueError(
                "enable_command_filter=True requires cmd_filter_coeff['acceleration'] > 0.0 "
                "(0.0 means 'always keep the old value', i.e. acceleration freezes forever)."
            )
        
    def get_rollout_fn(self, **kwargs):
        raise NotImplementedError
    
    def init_mppi(self, **kwargs):
        raise NotImplementedError
    
    def update_params(self, **kwargs):
        self.controller.rollout_fn.update_params(**kwargs)
        self.control_process.update_params(**kwargs)
        return True


    def get_command(self, t_step, curr_state, control_dt, WAIT=False):

        # predict forward from previous action and previous state:
        #self.state_filter.predict_internal_state(self.prev_qdd_des)

        if(self.state_filter.cmd_joint_state is None):
            curr_state['velocity'] *= 0.0
        filt_state = self.state_filter.filter_joint_state(curr_state)
        state_tensor = self._state_to_tensor(filt_state)

        if(WAIT):
            next_command, val, info, best_action = self.control_process.get_command_debug(t_step, state_tensor.numpy(), control_dt=control_dt)
        else:
            next_command, val, info, best_action = self.control_process.get_command(t_step, state_tensor.numpy(), control_dt=control_dt)

        # FIX: this used to unconditionally treat next_command as an acceleration and call
        # integrate_acc() regardless of control_space. For control_space='vel' or 'pos', the
        # action is a velocity or position value, not an acceleration -- integrate_acc would
        # have double-integrated it as if it were one, which is not just "less smooth" but
        # semantically meaningless. Dispatch on the same control_space the rollout/cost
        # evaluation already uses (tensor_step_vel/tensor_step_pos in integration_utils.py),
        # via the now-completed integrate_vel/integrate_pos (state_filter.py).
        control_space = self.exp_params.get('control_space', 'acc')
        if control_space == 'acc':
            qdd_des = next_command
            self.prev_qdd_des = qdd_des
            cmd_des = self.state_filter.integrate_acc(qdd_des)
        elif control_space == 'vel':
            cmd_des = self.state_filter.integrate_vel(next_command)
        elif control_space == 'pos':
            cmd_des = self.state_filter.integrate_pos(next_command)
        else:
            raise NotImplementedError(
                f"BaseTask.get_command() does not support control_space='{control_space}'."
            )

        if self.enable_command_filter:
            # Filters the published (position, velocity, acceleration) dict in place against
            # STORM's own command_filter, per-key EMA via cmd_filter_coeff. This is the more
            # "architecturally correct" place to smooth qdd_des than the bridge-level filter --
            # it makes STORM's own next-replan belief state consistent with what is actually
            # published, instead of filtering only at the point of downstream consumption.
            cmd_des = self.command_filter.filter_joint_state(cmd_des)

        return cmd_des



    def _state_to_tensor(self, state):
        state_tensor = np.concatenate((state['position'], state['velocity'], state['acceleration']))

        state_tensor = torch.tensor(state_tensor)
        return state_tensor
    def get_current_error(self, curr_state):
        state_tensor = self._state_to_tensor(curr_state).to(**self.controller.tensor_args).unsqueeze(0)

        
        ee_error,_ = self.controller.rollout_fn.current_cost(state_tensor)
        ee_error = [x.detach().cpu().item() for x in ee_error]
        return ee_error

    @property
    def mpc_dt(self):
        return self.control_process.mpc_dt
    @property
    def opt_dt(self):
        return self.control_process.opt_dt
    
    def close(self):
        self.control_process.close()
    @property
    def top_trajs(self):
        return self.control_process.top_trajs
    @property
    def top_values(self):
        return self.control_process.top_values
    @property
    def effective_sample_size(self):
        return self.control_process.effective_sample_size

