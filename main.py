import time
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import talib as ta

MONKEY_HIGH = 1
NUMBER_OF_ROPES = 4
DoGym = False


class monkeyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, monkey_high, n):

        self.monkey_high = monkey_high
        self.viewer = None
        self.n = n+1

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

        self.ropes = pd.DataFrame(
            np.random.randint(
                1, 100, size=(100, NUMBER_OF_ROPES)
            ),
            columns=list(range(1, NUMBER_OF_ROPES+1))
        )
        # let assume that is an stable coin like usdt that have 50$ price
        self.ropes.insert(loc=0, column=0, value=[50]*100)

        # for idx, rope in enumerate(self.ropes):
        #     self.ropes[idx] = ta.SMA(rope, 14)
        if self.viewer != None:
            self.viewer.close()
            self.viewer = None
        # print(self.ropes.head())

        return np.array([self.monkey_pos]).astype(np.float32)

    def step(self, action):

        self.monkey_last_pos = self.monkey_pos
        self.monkey_pos = action

        # if self.monkey_last_pos != self.monkey_pos:
        self.monkey_last_high = self.monkey_high
        self.monkey_high = self.ropes.iat[self.time, self.monkey_pos]

        self.time += 1
        done = True if self.time == 100 else False
        reward = 0.01*(self.monkey_high - self.monkey_last_high)

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

        x = self.time - 1
        y = self.monkey_high
        if self.viewer is None:

            self.viewer = rendering.Viewer(2000, 500)

            for i in range(self.n):
                xs = pd.Series(range(100))
                ys = self.ropes.iloc[:, i]
                xys = list(zip(xs*20, ys*5))
                self.track = rendering.make_polyline(xys)
                self.track.set_linewidth(2)
                self.track.set_color(*np.random.rand(3))
                self.viewer.add_geom(self.track)

            self.monkey = rendering.make_circle(radius=5)

            self.monkey.set_color(0, 0, 0)
            self.monkeyT = rendering.Transform(translation=(x*20, y*5))
            self.monkey.add_attr(self.monkeyT)
            self.viewer.add_geom(self.monkey)
        else:
            self.monkeyT.set_translation(x*20, y*5)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        pass

# --------------------------------------------------------------


# symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD']
# df = pd.DataFrame([1]*100)

# for i in range(len(symbols)):
#     symbol = symbols[i]
#     s = yf.download(tickers=symbol, period='22h',
#                     interval='15m')['Close']
#     print(s)
#     df.insert(loc=i+1, column=i+1, value=s)

# scaler = MinMaxScaler()

# x = df.values  # returns a numpy array
# x_scaled = scaler.fit_transform(x)
# df = pd.DataFrame(x_scaled)


# for example we have 50$ in start
env = monkeyEnv(monkey_high=MONKEY_HIGH, n=NUMBER_OF_ROPES)
model = None
if DoGym:
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("ppo_monkey")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_monkey")

obs = env.reset()
while True:
    time.sleep(0.1)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    print(info['monkey_high'], info['monkey_pos'])
    if info['time'] >= 100:
        env.reset()
