from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Robot


from legged_gym.envs.g1_29dof_rev_1_0.g1_config import G1RoughCfg as G129DofCfg, G1RoughCfgPPO as G129DofCfgPPO
from legged_gym.envs.g1_29dof_rev_1_0.g1_env import G1WithHandRobot

from legged_gym.envs.g1_naive.g1_naive_config import G1NaiveCfg, G1NaiveCfgPPO
from legged_gym.envs.g1_naive.g1_naive_env import G1NaiveRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
task_registry.register( "g1_29dof_rev_1_0", G1WithHandRobot, G129DofCfg(), G129DofCfgPPO())
task_registry.register( "g1_naive", G1NaiveRobot, G1NaiveCfg(), G1NaiveCfgPPO())
