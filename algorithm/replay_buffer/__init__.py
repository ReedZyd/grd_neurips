from .mujoco_buffer import (ReplayBuffer_CAUSAL, ReplayBuffer_Ground,
                            ReplayBuffer_IRCR, ReplayBuffer_RRD,
                            ReplayBuffer_Transition)


def create_buffer(args):
    if args.env_category=='mujoco' or args.env_category == 'dm_control':
        if args.alg=='ircr':
            return ReplayBuffer_IRCR(args)
        if args.alg=='rrd':
            return ReplayBuffer_RRD(args)
        if args.alg == 'causal':
            return ReplayBuffer_CAUSAL(args)
        if args.alg == 'ground':
            return ReplayBuffer_Ground(args)
        return ReplayBuffer_Transition(args)
    else:
        raise NotImplementedError
