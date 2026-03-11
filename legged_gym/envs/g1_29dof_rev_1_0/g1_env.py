from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch


class G1WithHandRobot(LeggedRobot):

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
        obs layout: [ang_vel(3), gravity(3), commands(3), dof_pos(43), dof_vel(43), actions(43), sin_phase(1), cos_phase(1)]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        num_dof = self.num_dof  # 43 for g1_29dof_with_hand

        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.  # commands
        noise_vec[9:9+num_dof] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+num_dof:9+2*num_dof] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*num_dof:9+3*num_dof] = 0.  # previous actions
        noise_vec[9+3*num_dof:9+3*num_dof+2] = 0.  # sin/cos phase

        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

        return super()._post_physics_step_callback()

    def compute_observations(self):
        """Computes observations.
        obs = [ang_vel(3), gravity(3), commands(3), dof_pos(43), dof_vel(43), actions(43), sin_phase(1), cos_phase(1)]
        total = 140
        privileged_obs = [lin_vel(3), ang_vel(3), gravity(3), commands(3), dof_pos(43), dof_vel(43), actions(43), sin_phase(1), cos_phase(1)]
        total = 143
        """
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)

        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,          # 3
            self.projected_gravity,                               # 3
            self.commands[:, :3] * self.commands_scale,           # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 43
            self.dof_vel * self.obs_scales.dof_vel,               # 43
            self.actions,                                         # 43
            sin_phase,                                            # 1
            cos_phase,                                            # 1
        ), dim=-1)

        self.privileged_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,          # 3
            self.base_ang_vel * self.obs_scales.ang_vel,          # 3
            self.projected_gravity,                               # 3
            self.commands[:, :3] * self.commands_scale,           # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 43
            self.dof_vel * self.obs_scales.dof_vel,               # 43
            self.actions,                                         # 43
            sin_phase,                                            # 1
            cos_phase,                                            # 1
        ), dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res

    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))

    def _reward_alive(self):
        return 1.0

    def _reward_contact_no_vel(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))

    def _reward_hip_pos(self):
        # left_hip_roll(1), left_hip_yaw(2), right_hip_roll(7), right_hip_yaw(8)
        # joint order from URDF: pitch(0), roll(1), yaw(2), knee(3), ankle_pitch(4), ankle_roll(5),
        #                         pitch(6), roll(7), yaw(8), ...
        return torch.sum(torch.square(self.dof_pos[:, [1, 2, 7, 8]]), dim=1)
