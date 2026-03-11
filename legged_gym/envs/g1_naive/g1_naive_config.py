import os
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# Pretrained decoder weights (.pth = state_dict only, no custom class needed at load time)
_DECODER_PATH = os.path.join(LEGGED_GYM_ROOT_DIR, 'pretrained', 'model', 'model_decoder_native.pth')
_DECODER_PATH = os.path.normpath(_DECODER_PATH)


class G1NaiveCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.8]  # x,y,z [m]
        default_joint_angles = {
            # --- legs ---
            'left_hip_pitch_joint':   -0.1,
            'left_hip_roll_joint':     0.0,
            'left_hip_yaw_joint':      0.0,
            'left_knee_joint':         0.3,
            'left_ankle_pitch_joint':  -0.2,
            'left_ankle_roll_joint':   0.0,
            'right_hip_pitch_joint':   -0.1,
            'right_hip_roll_joint':    0.0,
            'right_hip_yaw_joint':     0.0,
            'right_knee_joint':        0.3,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint':  0.0,
            # --- waist ---
            'waist_yaw_joint':   0.0,
            'waist_roll_joint':  0.0,
            'waist_pitch_joint': 0.0,
            # --- left arm ---
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint':  0.0,
            'left_shoulder_yaw_joint':   0.0,
            'left_elbow_joint':          0.0,
            'left_wrist_roll_joint':     0.0,
            'left_wrist_pitch_joint':    0.0,
            'left_wrist_yaw_joint':      0.0,
            # --- right arm ---
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint':  0.0,
            'right_shoulder_yaw_joint':   0.0,
            'right_elbow_joint':          0.0,
            'right_wrist_roll_joint':     0.0,
            'right_wrist_pitch_joint':    0.0,
            'right_wrist_yaw_joint':      0.0,
        }

    class env(LeggedRobotCfg.env):
        num_envs = 1024
        # Actor obs = MLP input (96) + decoder history (930) = 1026
        #   MLP input:  ang_vel(3) + dof_pos(29) + dof_vel(29) + actions(29) + gravity(3) + cmd(3) = 96
        #   History:    his_ang_vel(30) + his_dof_pos(290) + his_dof_vel(290) + his_actions(290) + his_gravity(30) = 930
        num_observations = 1026
        # Privileged obs appends base_lin_vel(3)
        num_privileged_obs = 1029
        num_actions = 29

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {
            'hip_yaw':   100,
            'hip_roll':  100,
            'hip_pitch': 100,
            'knee':      150,
            'ankle':      40,
            'waist':     200,
            'shoulder':   80,
            'elbow':      50,
            'wrist':      30,
        }
        damping = {
            'hip_yaw':   2,
            'hip_roll':  2,
            'hip_pitch': 2,
            'knee':      4,
            'ankle':     2,
            'waist':     5,
            'shoulder':  2,
            'elbow':     2,
            'wrist':     1,
        }
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof_rev_1_0.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        only_positive_rewards = False

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18


class G1NaiveCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.3
        # Actor MLP hidden dims: 96 -> 512 -> 256 -> 64
        actor_hidden_dims = [512, 256]
        # Critic MLP hidden dims: 1029 -> 512 -> 256 -> 128 -> 1
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
        # Path to pretrained decoder state dict (.pth)
        decoder_path = _DECODER_PATH

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.005

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticNaive"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1_naive'
