import numpy as np
import tensorflow as tf
from utils.tf_utils import get_vars
from algorithm import basis_algorithm_collection

def ground(args):
    basis_alg_class = basis_algorithm_collection[args.basis_alg]
    class Upper(basis_alg_class):
        def __init__(self, args):
            super().__init__(args)

            self.train_info_r = {}
            self.train_info_q = {**self.train_info_q, **self.train_info_r}
            self.train_info = {**self.train_info, **self.train_info_r}

        def get_reward(self, obs, obs_next, act):
            return 0
    return Upper(args)
