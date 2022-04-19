from gym_connect_four.connect_four_env import Player
import random
import numpy as np
from collections import defaultdict
from statistics import mean


class MonteCarloESPlayer(Player):

	def __init__(self, env: 'ConnectFourEnv', name='SemiGradientSarsaPlayer', alpha=0.01, epsilon=0.1, discount=0.99):
		super().__init__(env, name)
		self.alpha = alpha
		self.epsilon = epsilon
		self.discount = discount
		self.pi = defaultdict(lambda: random.choices(list(self.env.available_moves()))[0])
		self.qvals = defaultdict(float)
		self.returns = defaultdict(list)
		self.starting_move = True
		self.states = []
		self.actions = []
		self.rewards = []

	def get_next_action(self, state: np.ndarray) -> int:
		action = self.__pi__(state)
		if self.starting_move:
			self.starting_move = False
		return action

	def __pi__(self, state):
		if self.starting_move:
			action = random.choices(list(self.env.available_moves()))[0]
			# print("starting", action)
		else:
			action =  self.pi[tuple(map(tuple, state.tolist()))]
			# print("pi", action)
		return action

	def __update_pi__(self, state):
		val = 0
		best_action = None
		for action in self.env.available_moves_for_state(state):
			if best_action is None or self.qvals[(state, action)] > val:
				val = self.qvals[(state, action)]
				best_action = action
		self.pi[tuple(map(tuple, state.tolist()))] = best_action

	def learn(self, state, action: int, state_next, reward: int, done: bool) -> None:
		if done:
			g = 0
			self.rewards.append(reward)
			index = len(self.states) - 1
			while index >= 0:
				st = self.states[index]
				at = self.actions[index]
				rt = self.rewards[index]
				g = self.discount * g + rt
				if st not in self.states[:index - 1]:
					self.returns[(st, at)].append(g)
					self.qvals[(st, at)] = mean(self.returns[(st, at)])
		else:
			self.states.append(state)
			self.actions.append(action)
			self.rewards.append(reward)

	def save_model(self, model_prefix: str = None):
		pass

	def load_model(self, model_prefix: str = None):
		pass

	def reset(self, episode: int = 0, side: int = 1) -> None:
		"""
		Allows a player class to reset it's state before each round
			Parameters
			----------
			episode : which episode we have reached
			side : 1 if the player is starting or -1 if the player is second
		"""
		# self.weights = np.zeros(len(self.features))
		# We don't want to do anything here because we want the player to learn
		self.states = []
		self.actions = []
		self.rewards = []
		pass