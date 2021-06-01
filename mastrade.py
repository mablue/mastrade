# from agent_env import agentEnv
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import gym
from gym import spaces
from gym.utils import seeding


class agentEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, ropes, stock=1):
        self.ropes = ropes
        self.size = 4

        # init
        # self.agent_pos = 0
        # self.agent_last_pos = 0
        # self.agent_stock = ropes.iat[-1, 0]
        # self.agent_last_pos_stock = ropes.iat[-1, 0]
        # self.time = 0
        # self.last_time = 0
        # actions
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float32
        )
        self.viewer = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.agent_last_pos = self.agent_pos
        self.agent_pos = action
        self.agent_stock = self.ropes.iat[-1, self.agent_pos]
        self.agent_last_pos_stock = self.ropes.iat[-1, self.agent_last_pos]
        self.last_time = self.time
        self.time += 1
        # updating ropes:
        for i in range(3):
            self.ropes[i] = self.ropes[i] * self.agent_stock / \
                self.ropes.iat[-1, self.agent_pos]
        # Account for the boundaries of the grid

        # Are we at the left of the grid?
        done = True if self.time == 100 else False
        reward = 0.01 * self.agent_last_pos_stock - self.agent_stock

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return np.array([self.agent_pos]).astype(np.float32), reward, done, info

    def reset(self):
        self.agent_pos = 0
        self.agent_last_pos = 0
        self.agent_stock = self.ropes.iat[-1, 0]
        self.agent_last_pos_stock = self.ropes.iat[-1, 0]
        self.time = 0
        self.last_time = 0

        return np.array([self.agent_pos]).astype(np.float32)

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(2000, 500)

            for i in range(3):
                xs = pd.Series(range(100))
                ys = self.ropes.iloc[:, i]
                xys = list(zip(xs*20, ys*5))
                self.track = rendering.make_polyline(xys)
                self.track.set_linewidth(1)
                self.track.set_color(*np.random.rand(3))

                self.viewer.add_geom(self.track)

            x = self.agent_pos
            y = self.agent_stock
            self.agent = rendering.make_circle()
            self.agent.set_color(0.5, 0.5, 0.5)
            self.agent.add_attr(rendering.Transform(translation=(x*20, y*5)))

            self.viewer.add_geom(self.agent)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        pass

# --------------------------------------------------------------


df = pd.DataFrame(np.random.randint(
    1, 100, size=(100, 4)))
env = agentEnv(ropes=df)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2500)
model.save("ppo_cartpole")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(obs, rewards, dones, info)
    env.render()
