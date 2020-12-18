import gym
import envs
from algo.ddpg_td3 import DDPG

def main():
    env = gym.make('Pushing2D-v0')
    algo = DDPG(env, 'ddpg_log_td3.txt')
    algo.train(50000, hindsight=False)


if __name__ == '__main__':
    main()
