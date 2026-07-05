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
import numpy as np
import copy

class AlphaBetaFilter(object):
    def __init__(self, filter_coeff=0.4):
        self.raw_state = None
        self.filter_coeff = filter_coeff
    def filter(self, raw_state):
        self.raw_state = (1 - self.filter_coeff) * self.raw_state + self.filter_coeff * raw_state

    def two_state_filter(self, raw_state1, raw_state2):
        new_state = self.filter_coeff * raw_state1 + (1 - self.filter_coeff) * raw_state2

        return new_state

class RobotStateFilter(object):
    def __init__(self, filter_keys=['position', 'velocity','acceleration'], filter_coeff={'position':0.1, 'velocity':0.1,'acceleration':0.1},dt=0.1):
        self.prev_filtered_state = None
        self.filtered_state = None
        self.filter_coeff = filter_coeff
        self.filter_keys = filter_keys
        self.dt = dt
    def filter_state(self, raw_state, dt=None):
        dt = self.dt if dt is None else dt
        if(self.filtered_state is None):
            self.filtered_state = copy.deepcopy(raw_state)
            if('acceleration' in self.filter_keys):
                self.filtered_state['acceleration'] = 0.0* raw_state['position']
            #return self.filtered_state
        self.prev_filtered_state = copy.deepcopy(self.filtered_state)
        for k in self.filter_keys:
            if(k in raw_state.keys()):
                self.filtered_state[k] = self.filter_coeff[k] * raw_state[k] + (1.0 - self.filter_coeff[k]) * self.filtered_state[k]
        if('acceleration' in self.filter_keys):# and 'acceleration' not in raw_state):
            self.filtered_state['acceleration'] = (self.filtered_state['velocity'] - self.prev_filtered_state['velocity']) / dt
        return self.filtered_state
        
class JointStateFilter(object):
    
    def __init__(self, raw_joint_state=None, filter_coeff=0.4, dt=0.1, filter_keys=['position','velocity','acceleration']):
        self.cmd_joint_state = copy.deepcopy(raw_joint_state)

        self.filter_coeff = {}
        if not isinstance(filter_coeff,dict):
            for k in filter_keys:
                self.filter_coeff[k] = filter_coeff
        else:
            self.filter_coeff = filter_coeff
        self.dt = dt
        self.filter_keys = filter_keys
        self.prev_cmd_qdd = None
        # Dedicated "what did I command last" memory for integrate_pos's finite difference --
        # NOT the same thing as cmd_joint_state['position'], which filter_joint_state blends
        # towards the real measured position every tick (state_filter_coeff['position']=0.1 in
        # franka_reacher.yml, not 0 like velocity/acceleration). Differencing against that
        # contaminates the derivative with measurement-tracking lag instead of giving a clean
        # commanded-trajectory slope. integrate_vel doesn't need an analogous tracker: velocity's
        # own filter_coeff is 0.0, which already fully protects cmd_joint_state['velocity'] from
        # this same contamination (verified: filter_joint_state's blend formula degenerates to
        # "keep the old value" exactly when coeff=0).
        self.prev_cmd_position = None
    def filter_joint_state(self, raw_joint_state):
        if(self.cmd_joint_state is None):
            self.cmd_joint_state = copy.deepcopy(raw_joint_state)
            return self.cmd_joint_state

        for k in self.filter_keys:
            self.cmd_joint_state[k] = self.filter_coeff[k] * raw_joint_state[k] + (1.0 - self.filter_coeff[k]) * self.cmd_joint_state[k]

        return self.cmd_joint_state

    def forward_predict_internal_state(self, dt=None):
        if(self.prev_cmd_qdd is None):
            return
        dt = self.dt if dt is None else dt 
        self.cmd_joint_state['acceleration'] = self.prev_cmd_qdd
        self.cmd_joint_state['velocity'] = self.cmd_joint_state['velocity'] + self.prev_cmd_qdd * dt
        self.cmd_joint_state['position'] = self.cmd_joint_state['position'] + self.cmd_joint_state['velocity'] * dt
        

    def predict_internal_state(self, qdd_des=None, dt=None):
        if(qdd_des is None):
            return
        dt = self.dt if dt is None else dt 
        self.cmd_joint_state['acceleration'] = qdd_des
        self.cmd_joint_state['velocity'] = self.cmd_joint_state['velocity'] + qdd_des * dt
        self.cmd_joint_state['position'] = self.cmd_joint_state['position'] + self.cmd_joint_state['velocity'] * dt
        

    def integrate_jerk(self, qddd_des, raw_joint_state, dt=None):
        dt = self.dt if dt is None else dt 
        self.filter_joint_state(raw_joint_state)
        self.cmd_joint_state['acceleration'] = self.cmd_joint_state['acceleration'] + qddd_des * dt
        self.cmd_joint_state['velocity'] = self.cmd_joint_state['velocity'] + self.cmd_joint_state['acceleration'] * dt
        self.cmd_joint_state['position'] = self.cmd_joint_state['position'] + self.cmd_joint_state['velocity'] * dt
        self.prev_cmd_qdd = self.cmd_joint_state['acceleration']
        return self.cmd_joint_state

    def integrate_acc(self, qdd_des, raw_joint_state=None, dt=None):
        dt = self.dt if dt is None else dt
        if(raw_joint_state is not None):
            self.filter_joint_state(raw_joint_state)
        self.cmd_joint_state['acceleration'] = qdd_des
        self.cmd_joint_state['velocity'] = self.cmd_joint_state['velocity'] + qdd_des * dt
        self.cmd_joint_state['position'] = self.cmd_joint_state['position'] + self.cmd_joint_state['velocity'] * dt
        self.prev_cmd_qdd = self.cmd_joint_state['acceleration']
        return self.cmd_joint_state

    def integrate_vel(self, qd_des, raw_joint_state=None, dt=None):
        """FIXED: previously never updated 'acceleration' at all (left stale), inconsistent
        with tensor_step_vel's rollout semantics (acceleration = finite-difference of velocity).
        raw_joint_state now defaults to None (matching integrate_acc's convention) so callers
        that already filtered the measured state earlier in the same tick (e.g.
        BaseTask.get_command()) don't accidentally apply the EMA position/velocity blend twice.
        """
        dt = self.dt if dt is None else dt
        if(raw_joint_state is not None):
            self.filter_joint_state(raw_joint_state)
        prev_velocity = copy.deepcopy(self.cmd_joint_state['velocity'])
        self.cmd_joint_state['velocity'] = qd_des
        self.cmd_joint_state['position'] = self.cmd_joint_state['position'] + self.cmd_joint_state['velocity'] * dt
        self.cmd_joint_state['acceleration'] = (self.cmd_joint_state['velocity'] - prev_velocity) / dt
        return self.cmd_joint_state

    def integrate_pos(self, q_des, raw_joint_state=None, dt=None):
        """FIXED: previously raised NotImplementedError outright (acceleration finite-difference
        was never written). Now computes velocity and acceleration via backward finite
        differences, consistent with tensor_step_pos's rollout semantics (velocity = d/dt
        position, acceleration = d^2/dt^2 position) -- both noise-amplifying operations, since
        q_des is the raw, unintegrated MPPI action in this control_space; see
        franka_reacher_bench_pos.yml's header comment for the full implication of that.

        Differences against self.prev_cmd_position (a dedicated tracker), NOT
        cmd_joint_state['position']: the latter gets blended toward the real measured position
        every tick by filter_joint_state (state_filter_coeff['position']=0.1, not 0), which
        would otherwise contaminate this derivative with measurement-tracking lag instead of
        giving a clean commanded-trajectory slope -- confirmed via a standalone test where this
        produced acceleration values >100x past the hardware ceiling before this fix.
        """
        dt = self.dt if dt is None else dt
        if(raw_joint_state is not None):
            self.filter_joint_state(raw_joint_state)
        prev_position = self.prev_cmd_position if self.prev_cmd_position is not None else q_des
        prev_velocity = copy.deepcopy(self.cmd_joint_state['velocity'])
        new_velocity = (q_des - prev_position) / dt
        self.cmd_joint_state['acceleration'] = (new_velocity - prev_velocity) / dt
        self.cmd_joint_state['velocity'] = new_velocity
        self.cmd_joint_state['position'] = q_des
        self.prev_cmd_position = copy.deepcopy(q_des)
        return self.cmd_joint_state
