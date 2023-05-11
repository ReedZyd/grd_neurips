from .normal_mujoco import MuJoCoNormalEnv
# from .normal_dm import DMNormalEnv
from .ep_rews import create_EpisodicRewardsEnv


mujoco_list = [
    'Ant-v2', 'HalfCheetah-v2', 'Walker2d-v2', 'Humanoid-v2',
    'Reacher-v2', 'Swimmer-v2', 'Hopper-v2', 'HumanoidStandup-v2'
]

dm_control_list = [
    'point_mass', 'reacher'
]

envs_collection = {
    # MuJoCo envs
    **{
        mujoco_name : 'mujoco'
        for mujoco_name in mujoco_list
    },
    # DM Control envs
    **{
        dm_control_name : 'dm_control'
        for dm_control_name in dm_control_list
    }
}

def make_env(args):
    normal_env = {
        'mujoco': MuJoCoNormalEnv,
        # 'dm_control': DMNormalEnv,
    }[envs_collection[args.env]]

    return {
        'normal': normal_env,
        'ep_rews': create_EpisodicRewardsEnv(normal_env)
    }[args.env_type](args)
