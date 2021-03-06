import time
import gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from color_board_env import ColorBoardEnv, EMBEDDING_SIZE, COLOR_MAP

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizer_v2.adam import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import argparse
import shutil
import glob
import tensorflow
import tensorflow.python

from stable_baselines3.common.monitor import Monitor

parser = argparse.ArgumentParser()
parser.add_argument('--seed', '-s', help='Seed to be used in generating random numbers', type=lambda x: x if x is None else int(x), default=None)
parser.add_argument('--sleep_period', '-S', help='Sleep period between steps made by the agent in seconds', type=float, default=0.)
parser.add_argument('--text_encoder', '-e', help='Huggingface encoder to be used in encoding instructions', type=str, default='google/bert_uncased_L-2_H-128_A-2')
parser.add_argument('--results_dir', '-r', type=str, default='./training_results')
parser.add_argument('--weights_path', '-o', type=str, default='./weights.bin')
parser.add_argument('--visualize', '-v', dest='visualize', action='store_true', default=True)
parser.add_argument('--no-visualize', '-nv', dest='visualize', action='store_false')
parser.add_argument('--learning_rate', '-lr', type=float, default=2.5e-4)
parser.add_argument('--max_eps', type=float, default=.35)
parser.add_argument('--min_eps', type=float, default=.075)
parser.add_argument('--gamma', type=float, default=.995)
parser.add_argument('--window_length', type=int, default=1)
parser.add_argument('--total_steps', type=int, default=100000)
parser.add_argument('--max_steps_per_episode', type=int, default=8)
parser.add_argument('--num_rows', '-R', type=int, default=3)
parser.add_argument('--num_columns', '-C', type=int, default=3)

args = parser.parse_args()

if not os.path.exists(args.results_dir):
  os.makedirs(args.results_dir)

WINDOW_LENGTH = args.window_length

class Main():
    def __init__(self, weights_path=args.weights_path):
        self.weights_path = weights_path

        self.max_steps_per_episode = args.max_steps_per_episode
        self.total_steps = args.total_steps
        self.samp_freq = 10
        self.max_episodes = 2
        self.env = gym.make('ColorBoardEnv-v1', seed=args.seed, num_rows=args.num_rows, num_columns=args.num_columns, sleep_period_between_steps=args.sleep_period, text_encoder=args.text_encoder, max_steps_per_episode=args.max_steps_per_episode)
        print('ColorBoardEnv-v1 env initialised')

        self.env = Monitor(self.env, os.path.join(args.results_dir, '{}'.format(int(time.time()))), allow_early_resets=True)
        print('Env wrapped in monitor')

        self.num_actions = self.env.action_space.n
        self.dqn_model = self.build_model()

        self.load_model()
        cbs = [ModelIntervalCheckpoint(self.weights_path, verbose=1, interval=1000)]
        self.dqn_model.fit(self.env, nb_steps=self.total_steps, nb_max_episode_steps=self.max_steps_per_episode,
            visualize=args.visualize, log_interval=100, callbacks=cbs)
        print('Win rate: {:.4f}, Loss rate: {:.4f} (out of {} episodes)'.format(self.env.win_rate, self.env.loss_rate, (self.env.wins + self.env.losses)))
        self.env.stop()

    def build_model(self):
        model = Sequential()
        input_shape = (WINDOW_LENGTH, 1, EMBEDDING_SIZE + (self.env.num_rows * self.env.num_columns) + (self.env.num_rows * self.env.num_columns * len(COLOR_MAP)))
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(768, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_actions, activation='linear'))
        model.summary()

        memory = SequentialMemory(limit=self.total_steps, window_length=WINDOW_LENGTH)
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
            value_max=args.max_eps, value_min=args.min_eps, value_test=.05, nb_steps=self.total_steps)

        dqn = DQNAgent(model=model, policy=policy, gamma=args.gamma, memory=memory,
            nb_actions=self.num_actions, train_interval=4, delta_clip=1.)
        dqn.compile(Adam(learning_rate=args.learning_rate), metrics=['mae'])
        return dqn

    def save_model(self):
        print('Saving model...')
        self.dqn_model.save_weights(self.weights_path, overwrite=True)

    def load_model(self):
        print('Loading model...')
        if os.path.exists(self.weights_path + '.index'):
            try:
              self.dqn_model.load_weights(self.weights_path)
              return True
            except:
              fs = glob.glob(args.weights_path + '.*')
              for f in fs:
                os.unlink(f)
              return False
        return False


if __name__ == '__main__':
    Main()
