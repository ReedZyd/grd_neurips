from algorithm import basis_algorithm_collection

def IRCR(args):
    # The algorithmic components of IRCR is implemented in the replay buffer.
    basis_alg_class = basis_algorithm_collection[args.basis_alg]
    
    class IRCRReturnDecom(basis_alg_class):
        def get_reward(self, obs, obs_next, act):
            return 0
    
    return IRCRReturnDecom(args)
