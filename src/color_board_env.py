from copy import deepcopy
import gym
from gym.envs.registration import register
from gym import spaces
import time
import numpy as np
import sys
from termcolor import colored, cprint
from collections import namedtuple
import torch
from transformers import BertTokenizer, BertModel
import humanize

registration = register(
    id='ColorBoardEnv-v1',
    entry_point='color_board_env:ColorBoardEnv',
    max_episode_steps=None,
)

REWARDS = {
    'win': 100,
    'invalid_stop': -5,
    'invalid_move': -2.5,
    'move': -1,
    'lose': -5,
}

COLOR_MAP = {
    'red': 1,
    'green': 2,
    'yellow': 3,
    'blue': 4,
    'magenta': 5,
    'cyan': 6,
    'white': 7,
}

def to_color_id_one_hot_array(color_id, size=len(COLOR_MAP)):
    arr = np.zeros((size,))
    arr[color_id - 1] = 1
    return arr

def position_to_one_hot_array(position, shape):
    r, c = shape
    arr = np.zeros((r * c,))
    idx = np.ravel_multi_index(position, shape)
    arr[idx] = 1
    return arr


COLORS = list(COLOR_MAP.keys())
COLOR_CODES = list(COLOR_MAP.values())
EMBEDDING_SIZE = 128
DEFAULT_BOARD_ROWS = 3
DEFAULT_BOARD_COLS = 3
T = '◢◣'
D = '◀▶'

has_cuda = False # torch.cuda.is_available()

Position = namedtuple('Position', ['row', 'col'])
class Position(Position):
    def __repr__(self):
        return '({} row, {} column)'.format(humanize.ordinal(self.row + 1), humanize.ordinal(self.col + 1))

class ColorBoardEnv(gym.Env):
    def __init__(self, seed=None, sleep_period_between_steps=0., text_encoder='google/bert_uncased_L-2_H-128_A-2', num_rows=DEFAULT_BOARD_ROWS, num_columns=DEFAULT_BOARD_COLS, max_steps_per_episode=8):
        print('Starting up ColorBoard environment...')
        self.start_time = time.time()
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        self.sleep_period_between_steps = sleep_period_between_steps
        self.max_steps_per_episode = max_steps_per_episode
        self.num_rows = num_rows
        self.num_columns = num_columns

        self.tokenizer = BertTokenizer.from_pretrained(text_encoder)
        self.model = BertModel.from_pretrained(text_encoder)
        if has_cuda:
            self.model = self.model.to('cuda')

        self.step_count = 0
        self.reset_count = 0

        self.rewards = []
        self.action_space = spaces.Discrete(5)
        self.action_info_map = {
            0: 'up',
            1: 'right',
            2: 'down',
            3: 'left',
            4: 'stop',
        }
        self.observation_space = spaces.Box(
            low=0, high=len(COLOR_MAP), shape=(EMBEDDING_SIZE + (self.num_rows * self.num_columns),))
        self.reward_range = (-5, 5)

        self.__board = None
        self.__flattened_board = None
        self.__player_position = None
        self.__instruction = None
        self.__target_color = None
        self.__instruction_embedding = None
        self.__embedding_cache = {}
        self.wins = self.losses = 0

    def reset(self):
        self.rewards.clear()
        self.step_count = 0
        self.reset_count += 1
        self.__init_game()
        self.render('human')

        return self.__observe()

    def __init_game(self):
        self.__board = np.random.randint(low=1, high=len(COLOR_MAP) + 1, size=(self.num_rows, self.num_columns))
        one_hotter = np.vectorize(to_color_id_one_hot_array, signature='() -> (n)')
        self.__flattened_board = one_hotter(self.__board.reshape(self.num_columns * self.num_rows)).reshape((1, self.num_rows * self.num_columns * len(COLOR_MAP)))
        self.__player_position = Position(np.random.randint(self.num_rows), np.random.randint(self.num_columns))
        self.__target_color = COLORS[np.random.choice(np.unique(self.__board)) - 1]
        self.__instruction = '{}'.format(self.__target_color)

        if self.__instruction not in self.__embedding_cache:
            inputs = self.tokenizer(self.__instruction, return_tensors="pt")
            if has_cuda:
                inputs = inputs.to('cuda')
            outputs = self.model(**inputs)
            self.__instruction_embedding = outputs.last_hidden_state[:,1:-1,:].mean(dim=1).cpu().detach().numpy()
            self.__embedding_cache[self.__instruction] = self.__instruction_embedding
        else:
            self.__instruction_embedding = self.__embedding_cache[self.__instruction]

        msg = '🤖 starting position: {}; 🤖 target color: {}'.format(self.__player_position, self.__target_color)
        print('-' * (len(msg) + 2))
        msg = msg[:-len(self.__target_color)] + colored(self.__target_color, self.__target_color)
        print(msg)

    def step(self, action):
        self.step_count += 1

        time.sleep(self.sleep_period_between_steps)
        action_str = self.action_info_map[action]

        time_up = self.step_count >= self.max_steps_per_episode
        __current_row, __current_col = self.__player_position
        __current_color = COLORS[self.__board[__current_row, __current_col] - 1]
        __next = deepcopy(self.__player_position)
        invalid_move = False
        invalid_stop = False

        done = False
        if action_str == 'up':
            if __current_row == 0:
                invalid_move = True
            else:
                __next = Position(row=__current_row - 1, col=__current_col)
        elif action_str == 'down':
            if __current_row == (self.num_rows - 1):
                invalid_move = True
            else:
                __next = Position(row=__current_row + 1, col=__current_col)
        elif action_str == 'right':
            if __current_col == (self.num_columns - 1):
                invalid_move = True
            else:
                __next = Position(row=__current_row, col=__current_col + 1)
        elif action_str == 'left':
            if __current_col == 0:
                invalid_move = True
            else:
                __next = Position(row=__current_row, col=__current_col - 1)
        elif action_str == 'stop':
            if self.__target_color == __current_color:
                done = True
            else:
                invalid_move = invalid_stop = True

        print('🤖 transitioned from {} -> {} with {} action'.format(self.__player_position, __next, action_str))
        self.__player_position = __next

        # TODO: Perhaps allow the system to combine rewards for multi-state steps?
        if time_up and not done:
            reward = REWARDS['lose']
            cprint('🤖 loses!', 'white', 'on_red')
            self.losses += 1
        if invalid_stop:
            reward = REWARDS['invalid_stop']
        elif invalid_move:
            reward = REWARDS['invalid_move']
        elif done:
            reward = REWARDS['win']
            cprint('🤖 wins!', 'white', 'on_green')
            self.wins += 1
        else:
            reward = REWARDS['move']

        self.rewards.append(reward)
        obs = self.__observe()
        info = {}

        return obs, reward, done, info

    '''
        (instruction_embeddings, player_position, board)
    '''
    def __observe(self):
        obs = np.concatenate((
            self.__instruction_embedding, 
            position_to_one_hot_array(self.__player_position, (self.num_rows, self.num_columns)).reshape((1, self.num_rows * self.num_columns)), 
            self.__flattened_board
        ), axis=1)
        return obs

    def render(self, mode):
        if mode != 'human':
            return

        print('•' + '-' * self.num_columns * 2 + '•')
        for r in range(self.num_rows):
            print('|', end='')
            for c in range(self.num_columns):
                pos_color_code = self.__board[r, c]
                pos_color = COLORS[pos_color_code - 1]
                on = 'on_{}'.format(pos_color)
                if self.__is_player_position(Position(r, c)):
                    cprint(T, 'grey', on, end='')
                else:
                    cprint('  ', '{}'.format(pos_color), on, end='')
            print('|')
        print('•' + '-' * self.num_columns * 2 + '•')

    def stop(self):
        pass

    def __is_player_position(self, pos: Position):
        return pos.row == self.__player_position.row and pos.col == self.__player_position.col

    @property
    def player_position(self):
        return self.__player_position

    @property
    def instruction(self):
        return self.__instruction

    @property
    def win_rate(self):
        if self.losses + self.wins == 0:
            return 0.
        return self.wins / (self.wins + self.losses)

    @property
    def loss_rate(self):
        if self.losses + self.wins == 0:
            return 0.
        return self.losses / (self.wins + self.losses)
