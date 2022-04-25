from gym_connect_four.connect_four_env import Player
import random
import numpy as np
from agents.features import ADJACENT_FEATURES


def tab_aggregation(s):
	return s


class SemiGradientSarsaPlayer(Player):

	def __init__(self, env: 'ConnectFourEnv', name='SemiGradientSarsaPlayer', alpha=0.01, epsilon=0.1, discount=0.99, features=ADJACENT_FEATURES):
		super().__init__(env, name)
		self.alpha = alpha
		self.epsilon = epsilon
		self.discount = discount
		self.features = features
		self.weights = np.zeros(len(self.features))

	def get_next_action(self, state: np.ndarray) -> int:
		return self.__pi__(state)

	def __approx_q_val__(self, state, action):
		value = 0
		for i in range(len(self.features)):
			value += self.features[i](state, action) + self.weights.item(i)
		return value

	def __delta__(self, state, action):
		vals = []
		for i in range(len(self.features)):
			vals.append(self.features[i](state, action))
		return np.array(vals)

	def __pi__(self, state):
		choices = {}
		for action in self.env.available_moves():
			choices[action] = self.__approx_q_val__(state, action)
		max_value = max(choices.values())
		best_choices = [key for key, value in choices.items() if value == max_value]
		random_choice = random.choices(list(self.env.available_moves()))[0]
		greedy_choice = random.choices(best_choices)[0]
		action = random.choices([greedy_choice, random_choice], weights=[1 - self.epsilon, self.epsilon])[0]
		return action

	def learn(self, state, action: int, state_next, reward: int, done: bool) -> None:
		if done:
			change = self.alpha * (reward - self.__approx_q_val__(state, action))
			self.weights = self.weights + change * self.__delta__(state, action)
			self.weights = self.weights + self.alpha * (reward - self.__approx_q_val__(state, action))
			return
		a_prime = self.__pi__(state)
		change = self.alpha * (reward + self.discount * self.__approx_q_val__(state_next, a_prime) - self.__approx_q_val__(state, action))
		self.weights = self.weights + change * self.__delta__(state, action)
		# print(self.weights)

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
		pass