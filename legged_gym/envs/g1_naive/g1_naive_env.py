import torch
import numpy as np

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from legged_gym.envs.g1_29dof_rev_1_0.g1_env import G1WithHandRobot


class G1NaiveRobot(G1WithHandRobot):
    """G1 29-DOF env for the MLP+Decoder (NaivePolicy) actor.

    Actor observation layout (1026 dims):
      [0:3]    ang_vel  * obs_scales.ang_vel           (current frame)
      [3:32]   dof_pos  * obs_scales.dof_pos           (current frame, relative to default)
      [32:61]  dof_vel  * obs_scales.dof_vel           (current frame)
      [61:90]  actions                                 (last actions, current frame)
      [90:93]  projected_gravity                       (current frame)
      [93:96]  commands * commands_scale               (vx, vy, wz)
      ── history for decoder (oldest→newest within each block) ──
      [96:126]   his_ang_vel    10 × 3  = 30
      [126:416]  his_dof_pos    10 × 29 = 290
      [416:706]  his_dof_vel    10 × 29 = 290
      [706:996]  his_actions    10 × 29 = 290
      [996:1026] his_gravity    10 × 3  = 30

    Privileged obs (1029 dims) = actor obs + base_lin_vel (3).
    """

    HISTORY_LEN = 10
    NUM_DOF = 29

    def _get_noise_scale_vec(self, cfg):
        """Noise only on the 96-dim MLP input portion; history is zero-noise."""
        noise_vec = torch.zeros(self.cfg.env.num_observations, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        ns = self.cfg.noise.noise_scales
        nl = self.cfg.noise.noise_level

        # [0:96] MLP input
        noise_vec[0:3]   = ns.ang_vel  * nl * self.obs_scales.ang_vel
        noise_vec[3:32]  = ns.dof_pos  * nl * self.obs_scales.dof_pos
        noise_vec[32:61] = ns.dof_vel  * nl * self.obs_scales.dof_vel
        noise_vec[61:90] = 0.   # actions
        noise_vec[90:93] = ns.gravity * nl
        noise_vec[93:96] = 0.   # commands
        # [96:1026] history — no additional noise (it accumulated naturally)

        return noise_vec

    def _init_buffers(self):
        super()._init_buffers()
        H = self.HISTORY_LEN
        D = self.NUM_DOF
        ne = self.num_envs
        dev = self.device

        self.his_ang_vel = torch.zeros(ne, H * 3,  device=dev)
        self.his_dof_pos = torch.zeros(ne, H * D,  device=dev)
        self.his_dof_vel = torch.zeros(ne, H * D,  device=dev)
        self.his_actions = torch.zeros(ne, H * D,  device=dev)
        self.his_gravity = torch.zeros(ne, H * 3,  device=dev)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.his_ang_vel[env_ids] = 0.
        self.his_dof_pos[env_ids] = 0.
        self.his_dof_vel[env_ids] = 0.
        self.his_actions[env_ids] = 0.
        self.his_gravity[env_ids] = 0.

    def compute_observations(self):
        D = self.NUM_DOF

        # ── Compute scaled current-frame quantities ────────────────────────────
        ang_vel = self.base_ang_vel * self.obs_scales.ang_vel              # [N, 3]
        dof_pos = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos  # [N, 29]
        dof_vel = self.dof_vel * self.obs_scales.dof_vel                   # [N, 29]
        actions = self.actions                                             # [N, 29]
        gravity = self.projected_gravity                                   # [N, 3]
        cmd     = self.commands[:, :3] * self.commands_scale              # [N, 3]

        # ── Shift history buffers (drop oldest frame, append current) ─────────
        self.his_ang_vel = torch.roll(self.his_ang_vel, shifts=-3,  dims=1)
        self.his_ang_vel[:, -3:]  = ang_vel

        self.his_dof_pos = torch.roll(self.his_dof_pos, shifts=-D,  dims=1)
        self.his_dof_pos[:, -D:]  = dof_pos

        self.his_dof_vel = torch.roll(self.his_dof_vel, shifts=-D,  dims=1)
        self.his_dof_vel[:, -D:]  = dof_vel

        self.his_actions = torch.roll(self.his_actions, shifts=-D,  dims=1)
        self.his_actions[:, -D:]  = actions

        self.his_gravity = torch.roll(self.his_gravity, shifts=-3,  dims=1)
        self.his_gravity[:, -3:]  = gravity

        # ── MLP input (96 dims) ───────────────────────────────────────────────
        mlp_input = torch.cat([ang_vel, dof_pos, dof_vel, actions, gravity, cmd], dim=-1)

        # ── Actor obs (1026 dims) ─────────────────────────────────────────────
        self.obs_buf = torch.cat([
            mlp_input,
            self.his_ang_vel,
            self.his_dof_pos,
            self.his_dof_vel,
            self.his_actions,
            self.his_gravity,
        ], dim=-1)

        # ── Privileged obs (1029 dims) ────────────────────────────────────────
        self.privileged_obs_buf = torch.cat([
            self.obs_buf,
            self.base_lin_vel * self.obs_scales.lin_vel,
        ], dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
