import gym
import numpy as np
from utils.os_utils import remove_color

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class MuJoCoNormalEnv():
    def __init__(self, args):
        self.args = args
        self.env = gym.make(args.env).env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.render = self.env.render
        self.get_obs = self.env._get_obs


        self.norm_flg = self.args.obs_norm
        init_obs = (self.env.reset()).copy()
        self.max_value = init_obs.max()
        self.min_value = init_obs.min()
        print(self.max_value, self.min_value)


        self.acts_dims = list(self.action_space.shape)
        self.obs_dims = list(self.observation_space.shape)

        self.action_scale = np.array(self.action_space.high)
        for value_low, value_high in zip(list(self.action_space.low), list(self.action_space.high)):
            assert abs(value_low+value_high)<1e-3, (value_low, value_high)

        self.reset()
        self.env_info = {
            'Steps': self.process_info_steps, # episode steps
            'Rewards@green': self.process_info_rewards, # episode cumulative rewards
            'dense_reward':self.process_info_dense_reward

        }
        self.target_vel = 4.0 # Ant
        if self.args.new:
            if "Ant" in self.args.env or \
                    "HalfCheetah" in self.args.env or \
                        "Walker2d" in self.args.env or \
                            "Swimmer" in self.args.env or "Hopper" in self.args.env:
                self.step_and_get_reward = self.get_new_reward_1
            elif "Humanoid" in self.args.env and "Standup" not in self.args.env:
                self.step_and_get_reward = self.get_new_reward_2
            else:
                self.step_and_get_reward = self.get_new_reward_3
        else:
            self.step_and_get_reward = self.get_new_reward_3

    def norm_obs(self, obs):
        return (obs - self.min_value) / (self.max_value - self.min_value)
    def get_new_reward_1(self, action):

        posbefore = self.env.unwrapped.sim.data.qpos[0]
        obs, reward, done, info = self.env.step(action*self.action_scale)
        posafter = self.env.unwrapped.sim.data.qpos[0]
        forward_vel = (posafter - posbefore) / self.env.unwrapped.dt
        reward -= forward_vel # remove the term already in reward
        reward += -1 * abs(forward_vel - self.target_vel)
        return obs, reward, done, info
    def get_new_reward_2(self, action):
        posbefore = mass_center(self.env.unwrapped.model, self.env.unwrapped.sim)
        obs, reward, done, info = self.env.step(action*self.action_scale)
        posafter = mass_center(self.env.unwrapped.model, self.env.unwrapped.sim)
        forward_vel = 1.25 * (posafter - posbefore) / self.env.unwrapped.dt
        reward -= forward_vel # remove the term already in reward
        reward += -1.25 * abs(forward_vel - self.target_vel)
        return obs, reward, done, info
        
    def get_new_reward_3(self, action):
        obs, reward, done, info = self.env.step(action*self.action_scale)
        return obs, reward, done, info
    def process_info_steps(self, obs, reward, info):
        self.steps += 1
        return self.steps
    def process_info_dense_reward(self, obs, reward, info):
        return reward
    def process_info_rewards(self, obs, reward, info):
        self.rewards += reward
        return self.rewards

    def process_info(self, obs, reward, info):
        return {
            remove_color(key): value_func(obs, reward, info)
            for key, value_func in self.env_info.items()
        }
    
    def env_step(self, action):
        obs, reward, done, info = self.step_and_get_reward(action)
        info = self.process_info(obs, reward, info)
        if self.norm_flg:
            obs = self.norm_obs(obs)
        self.last_obs = obs.copy()
        if self.steps==self.args.test_timesteps: 
            done = True
        return obs, reward, done, info

    def step(self, action):
        obs, reward, done, info = self.env_step(action)

        return obs, reward, done, info

    def reset_ep(self):
        self.steps = 0
        self.rewards = 0.0
        self.last_obs = (self.env.reset()).copy()
        if self.norm_flg:
            self.last_obs = self.norm_obs(self.last_obs)
    def reset(self):
        self.reset_ep()
        return self.last_obs.copy()
    