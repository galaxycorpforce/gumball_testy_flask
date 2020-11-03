import enum
import numpy as np

import random
import math
from sklearn.preprocessing import LabelBinarizer
import math
#from dask_ml.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

#TARGETS: Sally, Jane, Kimberlly

"""


#If you take an odd number of gumballs you can only give to sally or kim.
#Kim has a higher reward but you must give her gumballs divisible by 3

#If you take an even number of gumballs you can give to jane

Restrict actions based on number of gumballs.

restrict targets based on action taken.

state is how many gumballs left.

"""

class Actions(enum.Enum):
	Take_1_Gumballs = 0
	Take_2_Gumballs = 1
	Take_3_Gumballs = 2
	Take_4_Gumballs = 3
	Take_5_Gumballs = 4

	def gumballs_for_action(self):
		if self == Actions.Take_1_Gumballs:
			return 1
		if self == Actions.Take_2_Gumballs:
			return 2
		if self == Actions.Take_3_Gumballs:
			return 3
		if self == Actions.Take_4_Gumballs:
			return 4
		if self == Actions.Take_5_Gumballs:
			return 5


cammy  = 'cammy'
sally  = 'sally'
jane   = 'jane'
kim	= 'kim'

days_liked ={'cammy':[0,2,3], 'sally':[1,2,3], 'jane':[4,5,6], 'kim':[0,4,5], }
liked_song ={'cammy':"Everybody's gonna love today, gonna love today ", 'sally':"I'm hooked on a feeling ", 'jane':"I feel nice, like sugar and spice ", 'kim':"Oh, oh, running out of breath ", }
hated_song ={'cammy':"What If I Never Get Over You", 'sally':"Someone You Loved", 'jane':"Remember You Young", 'kim':"I Guess I Just Feel Like", }

names_list = [cammy, sally, jane, kim]
days_of_week = range(7)

names_encoder = LabelEncoder().fit(names_list)
PLAYERS = ['p1', 'p2']
LAST_PIECE_REWARD = 3

class State:

	def __init__(self):
		self.buckets = {'cammy':0, 'sally':0, 'jane':0, 'kim':0}
		self.gumballs = 100
		self.reward = 0
		self.turns = 0
		self.day_of_week = random.randint(0, 7)
		self.max_gumballs_to_take = random.randint(1, 5)

		self.song = self.get_song_for_today()
		self.target_index = 0
		self.current_girl = names_list[self.target_index]
		self.next_girl = names_list[self.target_index]
		self.player_index = 0

	def reset(self):
		self.buckets = {'cammy':0, 'sally':0, 'jane':0, 'kim':0}
		self.computer_buckets = {'cammy':0, 'sally':0, 'jane':0, 'kim':0}
		self.gumballs = 100
		self.reward = 0
		self.turns = 0
		self.day_of_week = random.randint(0, 7)
		self.max_gumballs_to_take = random.randint(1, 5)
		self.song = self.get_song_for_today()
		self.target_index = 0
		self.current_girl = names_list[self.target_index]
		self.next_girl = names_list[self.target_index]
		self.player_index = 0

	def step(self, action):
		self.player_index += 1
		action = Actions(action)
		self.current_girl = names_list[self.target_index]

		self.target_index += 1
		self.target_index = self.target_index % len(names_list)
		self.next_girl = names_list[self.target_index]

		gumballs_taken = action.gumballs_for_action()

		self.gumballs -= gumballs_taken
		self.buckets[self.current_girl] += gumballs_taken

		done = self.gumballs <= 0
		base_reward = gumballs_taken

		#Kim gives bonus
		if self.current_girl == kim:
			base_reward *= 3

		if liked_song[self.current_girl] == self.song:
			base_reward *= 2
		if hated_song[self.current_girl] == self.song:
			base_reward *= -2

		if done:
#			base_reward += self.calculate_final_reward()
			base_reward += LAST_PIECE_REWARD

		self.song = self.get_song_for_today()
		self.max_gumballs_to_take = 1
		if self.gumballs > 1:
			self.max_gumballs_to_take = random.randint(1, min(5, self.gumballs))

		state, info = self.get_observation()
		return state, base_reward, done, info

	def get_observation(self):
		state = self.encode_field_state()
		state = flatten(state)
		player = PLAYERS[self.player_index % 2]
		info = {'valid_actions':self.valid_onehot_moves(self.get_avail_actions()), 'song':self.song, 'player': player}
		return state, info

	def valid_onehot_moves(self, avail_moves):
		moves = np.zeros(5)
		for move in avail_moves:
			moves[move.value] = 1
#        moves[np.nonzero(moves==0)] = -500
#		moves[np.nonzero(moves==0)] = -math.inf
#		moves[np.nonzero(moves==0)] = -4000000
#		moves[np.nonzero(moves==1)] = 0
		return moves


	def get_song_for_today(self):
		girl = np.random.choice(names_list)
		girl_days_liked = days_liked[girl]
		if self.day_of_week in girl_days_liked:
			return liked_song[girl]
		else:
			return hated_song[girl]


	def calculate_final_reward(self):
		min_gumballs = self.buckets[kim]
		max_gumballs = self.buckets[kim]

		min_gumballs = min(min_gumballs, self.buckets[cammy])
		min_gumballs = min(min_gumballs, self.buckets[sally])
		min_gumballs = min(min_gumballs, self.buckets[jane])

		max_gumballs = max(max_gumballs, self.buckets[cammy])
		max_gumballs = max(max_gumballs, self.buckets[sally])
		max_gumballs = max(max_gumballs, self.buckets[jane])

		distance = abs(max_gumballs - min_gumballs)
		return (300 - distance**2)


	def simulate_sequence(self):
		done = False
		winner = None
		steps = 0
		rewards = 0
		self.reset()
		while done == False:
			steps += 1
#			print()
			action = self.sample_actions()

			state, base_reward, done, info = self.step(action)

			rewards += base_reward
		print('final reward is: %d' % (rewards,))

	def encode_field_state(self):
		#
		encode = []
		encode.append(names_encoder.transform([self.next_girl]))
		encode.append(names_encoder.transform([cammy]))
		encode.append(names_encoder.transform([sally]))
		encode.append(names_encoder.transform([jane]))
		encode.append(names_encoder.transform([kim]))
		encode.append(self.gumballs/100.0)
		encode.append(self.buckets[cammy]/100.0)
		encode.append(self.buckets[sally]/100.0)
		encode.append(self.buckets[jane]/100.0)
		encode.append(self.buckets[kim]/100.0)

		return encode

	def get_avail_actions(self):
		if self.gumballs == 0:
			return []
		actions = []
		for i in range(self.max_gumballs_to_take):
			actions.append(Actions(i))
		return actions

	def bool_to_int(self, value):
		return 1 if value else 0

	def sample_actions(self):
		return np.random.choice(self.get_avail_actions())

	@property
	def shape(self):
		return (len(self.encode_field_state()), )


class GumballEnv():
	metadata = {'render.modes': ['human']}

	def __init__(self):
		# if network exists, use that decide moves. Otherwise use random move.
		self._state = State()




	def step(self, action_as_int):
#		print('p1 availables: ', self._state.get_valid_moves_for_player(position='a', is_p1_perspective=True))
#		p1_action = get_sample_action('p1', self._state.sample_actions(position='a', is_p1_perspective=True))
		p1_action = Actions(action_as_int)
		if p1_action not in self._state.get_avail_actions():
				p1_action = self._state.sample_actions()
				print('Ilegal action selected:', Actions(action_as_int))
				print('using action instead:', Actions(p1_action))
#		p1_action = get_sample_action('p1', p1_action)
		obs, reward, done, info = self._state.step(p1_action)

#		print('valid_onehot', obs['valid_onehot_player'])
		return obs, reward, done, info


	def reset(self):
		self._state.reset()
		state, info = self._state.get_observation()
		return state, 0, False, info

	def get_current_transcript(self):
		return self._state.state_transcript

	def get_current_transcript(self):
		return self._state.state_transcript

	def sample_actions(self):
		return self._state.sample_actions().value

	def seed(self, seed=None):
		self.np_random, seed1 = seeding.np_random(seed)
		seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
		return [seed1, seed2]

	def set_network(computer_network):
		self.computer_network = computer_network

class RewardScaler():
	"""
		Bring rewards to a reasonable scale for PPO.
		This is incrediably important and effects performance drastically
	"""
	def reward(self, reward):
		return reward * 0.01


def make_env():
	env = GumballEnv()

	return env

def make_gumball_env():
	def _init():
		env = make_env()
		return env
	return _init

def flatten(items):
    new_items = []
    for x in items:
        if isinstance(x, list) or isinstance(x, np.ndarray):
            new_items.extend(x)
        else:
            new_items.append(x)
    return new_items
