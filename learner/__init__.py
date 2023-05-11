from .mujoco import MuJoCoLearner

def create_learner(args):
    return {
        'mujoco': MuJoCoLearner,
    }[args.env_category](args)
