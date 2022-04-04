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

registration = register(
    id='ColorBoardEnv-v1',
    entry_point='color_board_env:ColorBoardEnv',
    max_episode_steps=None,
)

Position = namedtuple('Position', ['row', 'col'])

REWARDS = {
    'win': 5,
    'invalid_stop': -5,
    'move': -1,
    'invalid_move': -1,
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
COLORS = list(COLOR_MAP.keys())
COLOR_CODES = list(COLOR_MAP.values())
EMBEDDING_SIZE = 128
BOARD_ROWS = 4
BOARD_COLS = 4
T = '◢◣'
D = '◀▶'

class ColorBoardEnv(gym.Env):
    def __init__(self, seed=None, sleep_period_between_steps=0., text_encoder='google/bert_uncased_L-2_H-128_A-2', max_steps_per_episode=100):
        print('Starting up ColorBoard environment...')
        self.start_time = time.time()
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        self.sleep_period_between_steps = sleep_period_between_steps
        self.max_steps_per_episode = max_steps_per_episode

        self.tokenizer = BertTokenizer.from_pretrained(text_encoder)
        self.model = BertModel.from_pretrained(text_encoder)

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
            low=0, high=len(COLOR_MAP), shape=(EMBEDDING_SIZE + (BOARD_ROWS * BOARD_COLS),))
        self.reward_range = (-5, 5)

        self.__board = None
        self.__flattened_board = None
        self.__player_position = None
        self.__instruction = None
        self.__target_color = None
        self.__instruction_embedding = None

    def reset(self):
        self.rewards.clear()
        self.step_count = 0
        self.reset_count += 1
        self.__init_game()

        return self.__observe()

    def __init_game(self):
        self.__board = np.random.randint(low=1, high=len(COLOR_MAP) + 1, size=(BOARD_ROWS, BOARD_COLS))
        self.__flattened_board = self.__board.reshape((1, BOARD_COLS * BOARD_ROWS))
        self.__player_position = Position(np.random.randint(BOARD_ROWS), np.random.randint(BOARD_COLS))
        self.__target_color = COLORS[np.random.choice(np.unique(self.__board)) - 1]
        self.__instruction = 'Go to a {} position'.format(self.__target_color)

        inputs = self.tokenizer(self.__instruction, return_tensors="pt")
        outputs = self.model(**inputs)
        self.__instruction_embedding = outputs.last_hidden_state[:,:,:].mean(dim=1).detach().numpy()

        print('🤖 starting position: ({}, {}); 🤖 target color: {}'.format(self.__player_position[0], self.__player_position[1], self.__target_color))

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
            if __current_row == (BOARD_ROWS - 1):
                invalid_move = True
            else:
                __next = Position(row=__current_row + 1, col=__current_col)
        elif action_str == 'right':
            if __current_col == (BOARD_COLS - 1):
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

        if not time_up:
            self.__player_position = __next

        if time_up:
            reward = REWARDS['lose']
            cprint('🤖 loses!', 'white', 'on_red')
        if invalid_stop:
            reward = REWARDS['invalid_stop']
        elif invalid_move:
            reward = REWARDS['invalid_move']
        elif done:
            reward = REWARDS['win']
            cprint('🤖 wins!', 'white', 'on_green')
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
        return np.concatenate((self.__instruction_embedding, np.array(self.__player_position).reshape((1, 2)), self.__flattened_board), axis=1)

    def render(self, mode):
        if mode != 'human':
            return

        print('•' + '-' * BOARD_COLS * 2 + '•')
        for r in range(BOARD_ROWS):
            print('|', end='')
            for c in range(BOARD_COLS):
                pos_color_code = self.__board[r, c]
                pos_color = COLORS[pos_color_code - 1]
                on = 'on_{}'.format(pos_color)
                if self.__is_player_position(Position(r, c)):
                    cprint(T, 'grey', on, end='')
                else:
                    cprint('  ', '{}'.format(pos_color), on, end='')
            print('|')
        print('•' + '-' * BOARD_COLS * 2 + '•')

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
