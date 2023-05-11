import copy
import numpy as np

class Trajectory:
    def __init__(self, init_obs):
        self.ep = {
            'obs': [copy.deepcopy(init_obs)],
            'rews': [],
            'acts': [],
            'done': [],
            'dense_reward': [],
        }
        self.length = 0
        self.sum_rews = 0

    def store_transition(self, info):
        self.ep['acts'].append(copy.deepcopy(info['acts']))
        self.ep['obs'].append(copy.deepcopy(info['obs_next']))
        self.ep['rews'].append(copy.deepcopy(info['rews']))
        self.ep['done'].append(copy.deepcopy(np.float32(info['done'])))
        self.ep['dense_reward'].append(copy.deepcopy(np.float32(info['dense_reward'])))
        self.length += 1
        self.sum_rews += info['rews']

        if info['real_done']:
            for key in self.ep.keys():
                self.ep[key] = np.array(self.ep[key])

    def sample(self):
        idx = np.random.randint(self.length)
        info = {
            'obs': copy.deepcopy(self.ep['obs'][idx]),
            'obs_next': copy.deepcopy(self.ep['obs'][idx+1]),
            'acts': copy.deepcopy(self.ep['acts'][idx]),
            'rews': [copy.deepcopy(self.ep['rews'][idx])],
            'done': [copy.deepcopy(self.ep['done'][idx])],
            'dense_reward': [copy.deepcopy(self.ep['dense_reward'][idx])],
        }
        return info

    def causal_sample_rrd(self, sample_size, store_coef=False):
        idx = np.random.choice(self.length, sample_size, replace=(sample_size>self.length))
        info = {
            'causal_obs': self.ep['obs'][idx],
            'causal_obs_next': self.ep['obs'][idx+1],
            'causal_acts': self.ep['acts'][idx],
            'causal_rews': [self.sum_rews/self.length],
            'causal_done': np.array(self.ep['done'][idx])[:, np.newaxis],
        }
        if store_coef:
            if (sample_size<=self.length) and (self.length>1):
                info['causal_var_coef'] = [1.0-float(sample_size)/self.length]
            else:
                # We do not handle the case with (sample_size>self.length).
                info['causal_var_coef'] = [1.0 if self.length>1 else 0.0]
        return info
    def causal_sample_seq(self, max_length):
        extend = (max_length != self.length)
        info = {
            'causal_obs': list(self.ep['obs'][:-1]) + [self.ep['obs'][-1]]*(max_length-self.length) if extend else self.ep['obs'][:-1],
            'causal_obs_next': list(self.ep['obs'][1:]) + [self.ep['obs'][-1]]*(max_length-self.length) if extend else self.ep['obs'][1:],
            'causal_acts': list(self.ep['acts']) + [self.ep['acts'][-1]]*(max_length-self.length) if extend else self.ep['acts'],
            'causal_rews': [self.sum_rews/self.length],
            # TODO check if the last reward should be calculated
            'causal_done': np.array(list(self.ep['done'][:-1])+ [0.0] + [1.0]*(max_length-self.length))[:, np.newaxis],
        }
        return info

class ReplayBuffer_CAUSAL:
    def __init__(self, args):
        self.args = args
        self.ep_counter = 0
        self.ep_num = 0
        self.step_counter = 0
        self.buffer_size = self.args.buffer_size

        self.ep = []
        self.ram_idx = []
        self.length = 0
        self.head_idx = 0
        self.in_head = True
        if args.apply_accurate_loss:
            self.sample_batch = self.sample_batch_seq
            self.max_length = args.test_timesteps
            assert args.causal_bias_correction == False
        else:
            self.sample_batch = self.sample_batch_rrd
    def store_transition(self, info):
        if self.in_head:
            new_ep = Trajectory(info['obs'])
            self.ep.append(new_ep)
        self.ep[-1].store_transition(info)
        self.ram_idx.append(self.ep_counter)
        self.length += 1

        if self.length>self.buffer_size:
            del_len = self.ep[0].length
            self.ep.pop(0)
            self.head_idx += 1
            self.length -= del_len
            self.ep_num -= 1
            self.ram_idx = self.ram_idx[del_len:]

        self.step_counter += 1
        self.in_head = info['real_done']
        if info['real_done']:
            self.ep_counter += 1
            self.ep_num += 1

    def sample_batch_rrd(self, batch_size=-1, causal_batch_size=-1, causal_sample_size=-1):
        if batch_size==-1: batch_size = self.args.batch_size
        if causal_batch_size==-1: causal_batch_size = self.args.causal_batch_size
        if causal_sample_size==-1: causal_sample_size = self.args.causal_sample_size
        batch = dict(obs=[], obs_next=[], acts=[], rews=[], done=[], dense_reward=[], 
        causal_obs=[], causal_obs_next=[], causal_acts=[], causal_rews=[], causal_done=[])
        if self.args.causal_bias_correction:
            batch['causal_var_coef'] = []

        for i in range(batch_size):
            idx = self.ram_idx[np.random.randint(self.length)]-self.head_idx
            info = self.ep[idx].sample()
            for key in info.keys():
                batch[key].append(info[key])

        for i in range(causal_batch_size//causal_sample_size):
            idx = self.ram_idx[np.random.randint(self.length)]-self.head_idx
            info = self.ep[idx].causal_sample_rrd(causal_sample_size, store_coef=self.args.causal_bias_correction)
            for key in info.keys():
                batch[key].append(info[key])

        return batch
    def sample_batch_seq(self, batch_size=-1, causal_batch_size=-1, causal_sample_size=-1):
        if batch_size==-1: batch_size = self.args.batch_size
        if causal_batch_size==-1: causal_batch_size = self.args.causal_batch_size
        if causal_sample_size==-1: causal_sample_size = self.args.causal_sample_size
        batch = dict(obs=[], obs_next=[], acts=[], rews=[], done=[], dense_reward=[], 
        causal_obs=[], causal_obs_next=[], causal_acts=[], causal_rews=[], causal_done=[])

        for i in range(batch_size):
            idx = self.ram_idx[np.random.randint(self.length)]-self.head_idx
            info = self.ep[idx].sample()
            for key in info.keys():
                batch[key].append(info[key])

        idx = np.random.choice(self.ep_num, causal_batch_size//causal_sample_size, replace=False)
        length = [self.ep[i].length for i in idx]
        max_length = np.array(length).max()
        for i in idx:
            # ep_ = self.ep[i]
            info = self.ep[i].causal_sample_seq(max_length)
            for key in info.keys():
                batch[key].append(info[key])
        return batch