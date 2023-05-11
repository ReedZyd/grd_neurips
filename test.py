import os
import numpy as np
from envs import make_env
from utils.os_utils import make_dir

class Tester:
    def __init__(self, args):
        self.args = args
        self.env = make_env(args)
        self.info = []
        self.cycle_num = 0
        if args.save_rews:
            make_dir('log/rews', clear=False)
            # self.rews_record = {}
            # self.rews_record[args.env] = []
            self.rews_record = [[]]
            self.rews_record_cycle = []
        self.epoch_num = 0
    def test_rollouts(self):
        rewards_sum = 0.0
        rews_List, V_pred_List = [], []
        for _ in range(self.args.test_rollouts):
            rewards = 0.0
            obs = self.env.reset()
            for timestep in range(self.args.test_timesteps):
                action, info = self.args.agent.step(obs, explore=False, test_info=True)
                self.args.logger.add_dict(info)
                if 'test_eps' in self.args.__dict__.keys():
                    # the default testing scheme of Atari games
                    if np.random.uniform(0.0, 1.0)<=self.args.test_eps:
                        action = self.env.action_space.sample()
                obs_next, reward, done, info = self.env.step(action)
                if self.args.save_rews:
                    self.rews_record[-1].append([
                        info['dense_reward'], 
                        self.args.agent.get_reward(obs, obs_next, action)
                        ])
                obs = obs_next
                rewards += reward
                if done: 
                    if self.args.save_rews:
                        self.rews_record[-1] += [[-999., -999] for i in range(self.args.test_timesteps-timestep-1)]
                        self.rews_record[-1] = np.stack(self.rews_record[-1])
                        self.rews_record.append([])
                    break
            rewards_sum += rewards
            self.args.logger.add_dict(info)

        # if self.args.save_rews:
        #     step = self.args.learner.step_counter
        #     rews = rewards_sum/self.args.test_rollouts
        #     self.rews_record[self.args.env].append((step, rews))

    def cycle_summary(self):
        self.test_rollouts()
        if self.args.save_rews:
            self.rews_record_cycle.append(np.stack(self.rews_record[:-1]))
            self.rews_record = [[]]
        if self.args.alg == "causal" and self.cycle_num % 10 == 0:
            causal_structure = self.args.agent.get_structure()
            for key, value in causal_structure.items():
                log_folder = 'log/causal_structure/' + self.args.env + '/' +self.args.alg + '/' + self.args.tag + self.args.timestamp
                os.makedirs(log_folder, exist_ok=True)
                np.save(log_folder + '/' + str(self.cycle_num) +key+'.npy', value)
        self.cycle_num += 1
    def epoch_summary(self):
        if self.args.save_rews:
        
            log_folder = 'log/rews/' + self.args.env + '/' +self.args.alg + '/' + self.args.tag + self.args.timestamp
            os.makedirs(log_folder, exist_ok=True)
            np.save(log_folder + '/' + str(self.epoch_num) +'.npy', np.stack(self.rews_record_cycle[1:]))
            self.rews_record_cycle = [[]]

    def final_summary(self):
        pass
