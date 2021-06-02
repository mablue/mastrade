import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
MONKEY_HIGH = 10000
NUMBER_OF_ROPES = 30


class monkeyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, ropes, monkey_high, n):
        self.ropes = ropes
        self.monkey_high = monkey_high
        self.viewer = None
        self.n = n

        # actions
        self.action_space = spaces.Discrete(self.n)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.monkey_last_pos = 0
        self.monkey_pos = 0
        self.rope_high = 1
        self.last_rope_high = 1
        self.monkey_last_high = MONKEY_HIGH
        self.monkey_high = MONKEY_HIGH
        self.last_time = 0
        self.time = 0

        return np.array([self.monkey_pos]).astype(np.float32)

    def step(self, action):

        self.monkey_last_pos = self.monkey_pos
        self.monkey_pos = action

        if self.monkey_last_pos != self.monkey_pos:
            self.monkey_last_high = self.monkey_high
            self.monkey_high = self.monkey_high / \
                self.ropes.iat[-1, self.monkey_pos]
        # self.rope_high = self.ropes.iat[-1, self.monkey_pos]
        # self.last_rope_high = self.ropes.iat[-1, self.monkey_last_pos]
        # self.monkey_last_high = self.monkey_high * self.last_rope_high
        # self.monkey_high = self.monkey_high * self.rope_high
        self.time += 1
        done = True if self.time == 100 else False
        reward = 0.01 * (self.monkey_high - self.monkey_last_high)

        info = {
            "monkey_last_pos": self.monkey_last_pos,
            "monkey_pos": self.monkey_pos,

            "rope_high": self.rope_high,
            "last_rope_high": self.last_rope_high,

            "monkey_last_high": self.monkey_last_high,
            "monkey_high": self.monkey_high,

            "last_time": self.last_time,
            "time": self.time,

        }

        return np.array([self.monkey_pos]).astype(np.float32), reward, done, info

    def render(self, mode='human'):

        if self.viewer is None:

            self.viewer = rendering.Viewer(2000, 500)

            for i in range(self.n-1):
                xs = pd.Series(range(100))
                ys = self.ropes.iloc[:, i]
                xys = list(zip(xs*20, ys*5))
                self.track = rendering.make_polyline(xys)
                self.track.set_linewidth(1)
                self.track.set_color(*np.random.rand(3))
                self.viewer.add_geom(self.track)

        else:
            self.monkey = rendering.make_circle(radius=5)
            x = self.time
            y = self.monkey_high
            self.monkey.set_color(*np.random.rand(3))
            self.monkey.add_attr(rendering.Transform(translation=(x*20, y*5)))
            self.viewer.add_geom(self.monkey)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        pass

# --------------------------------------------------------------


df = pd.DataFrame(
    np.random.randint(
        1, 100, size=(100, NUMBER_OF_ROPES)
    ),
    columns=list(range(1, NUMBER_OF_ROPES+1))
)
df.insert(loc=0, column=0, value=[1]*100)


# for example we have 1USDT in start(MONKEY_HIGH=1)
env = monkeyEnv(ropes=df, monkey_high=MONKEY_HIGH, n=30)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2500)
model.save("ppo_monkey")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_monkey")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    print(info['monkey_high'], 1)
    if info['time'] >= 100:
        env.reset()
