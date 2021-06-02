import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering


class MonkeyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, ropes, stock):
        self.ropes = ropes
        self.size = 4
        self.stock = stock
        # init
        # self.Monkey_pos = 0
        # self.Monkey_last_pos = 0
        # self.Monkey_price = ropes.iat[-1, 0]
        # self.Monkey_last_pos_price = ropes.iat[-1, 0]
        # self.time = 0
        # self.last_time = 0
        # actions
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=reset)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.Monkey_pos = 0
        self.Monkey_last_pos = 0
        self.Monkey_price = self.ropes.iat[-1, 0]
        self.Monkey_last_pos_price = self.ropes.iat[-1, 0]
        self.time = 0
        self.last_time = 0

        return np.array([self.Monkey_pos]).astype(np.float32)

    def step(self, action):
        self.Monkey_last_pos = self.Monkey_pos
        self.Monkey_pos = action

        self.Monkey_price = self.ropes.iat[-1, self.Monkey_pos]
        self.Monkey_last_pos_price = self.ropes.iat[-1, self.Monkey_last_pos]
        self.stock = self.stock / self.Monkey_price

        self.last_time = self.time
        self.time += 1
        # updating ropes:
        # for i in range(3):
        #     self.ropes[i] = self.ropes[i] * self.Monkey_price / \
        #         self.ropes.iat[-1, self.Monkey_pos]

        done = True if self.time == 100 else False
        reward = 0.01 * self.Monkey_last_pos_price - self.Monkey_price
        if done:
            env.reset()
        # Optionally we can pass additional info, we are not using that for now
        info = {'time': self.time}

        return np.array([self.Monkey_pos]).astype(np.float32), reward, done, info

    def render(self, mode='human'):

        if self.viewer is None:
            self.viewer = rendering.Viewer(2000, 500)
            for i in range(3):
                xs = pd.Series(range(100))
                ys = self.ropes.iloc[:, i]
                xys = list(zip(xs*20, ys*5))
                self.track = rendering.make_polyline(xys)
                self.track.set_linewidth(1)
                self.track.set_color(*np.random.rand(3))

                self.viewer.add_geom(self.track)
        else:

            x = self.time
            y = self.Monkey_price
            self.Monkey = rendering.make_circle()
            self.Monkey.set_color(*np.random.rand(3))

            self.Monkey.add_attr(rendering.Transform(translation=(x*20, y*5)))
            self.viewer.add_geom(self.Monkey)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        pass

# --------------------------------------------------------------


df = pd.DataFrame(np.random.randint(
    1, 100, size=(100, 4)))


env = MonkeyEnv(ropes=df, stock=1000)  # for example we have 1000$ in start

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2500)
model.save("ppo_monkey")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_monkey")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(obs, rewards, dones, info)
    env.render()
