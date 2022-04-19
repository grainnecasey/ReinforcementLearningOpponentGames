from gym_connect_four.connect_four_env import ConnectFourEnv
from gym_connect_four.connect_four_env import RandomPlayer
from agents.sarsa_player import SemiGradientSarsaPlayer, create_features
from agents.monte_carlo_es_player import MonteCarloESPlayer
import matplotlib.pyplot as plt
import numpy as np
import statistics
import math

env = ConnectFourEnv(board_shape=(12, 14))

player1 = RandomPlayer(env, "Dexter-Bot")
# player2 = SemiGradientSarsaPlayer(env, "SemiGradientSarsa", features=create_features(7))
player2 = MonteCarloESPlayer(env, "MonteCarloES")

# result = env.run(player1, player2, render=True)
# reward = result.value
# print(reward)

episodes = 100
trials = 10

rewards_by_trial = []

for trial in range(trials):
	rewards = 0
	cumulative_rewards = []
	for ep in range(episodes):
		if ep % 100 == 0:
			print("trial", trial, "ep", ep)
		result = env.run(player1, player2, render=False)
		rewards += result.value
		cumulative_rewards.append(rewards)
	rewards_by_trial.append(cumulative_rewards)


average_cumulative_rewards = np.average(np.array(rewards_by_trial), axis=0)
plt.plot(average_cumulative_rewards, color='r', label='Cumulative Reward')
error = 1.96* statistics.pstdev(average_cumulative_rewards)/math.sqrt(episodes)
plt.fill_between(range(episodes), [x-error for x in average_cumulative_rewards],
				 [x+error for x in average_cumulative_rewards], color='b', alpha=0.2)
plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Monte Carlo ES Cumulative Rewards Playing Connect 4 Against Random Player")
plt.show()