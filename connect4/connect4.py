from gym_connect_four.connect_four_env import ConnectFourEnv, ResultType
from gym_connect_four.connect_four_env import RandomPlayer
from agents.sarsa_player import SemiGradientSarsaPlayer
from agents.features import create_adj_features, create_adj_x_features, ADJ_1_AND_2_CONNECTION_FEATURES, ADJ_1_2_3_CONNECTION_FEATURES, create_one_player_adj_x_features
from agents.monte_carlo_es_player import MonteCarloESPlayer
from agents.DQNPlayer import NNPlayer
import matplotlib.pyplot as plt
import numpy as np
import statistics
import math
from collections import defaultdict
import random

board_1 = (6, 7)
board_2 = (12, 14)
board_3 = (24, 28)
board_4 = (150, 175)

BOARD = board_1

connect_size = {board_1: 4, board_2: 8, board_3: 16, board_4: 100}

env = ConnectFourEnv(board_shape=BOARD, win_condition=connect_size[BOARD])

# features = create_one_player_adj_x_features(BOARD[1], 1, 1) + create_one_player_adj_x_features(BOARD[1], 2, 1) + create_one_player_adj_x_features(BOARD[1], 3, 1)
features = create_adj_features(7)

randomPlayer = RandomPlayer(env, "Dexter-Bot")

# label = "connect 3 features for just agent"
file_name = '../data/against_random_loss_0_random_start.csv'

episodes = 1000
trials = 3

# trainedPlayer = MonteCarloESPlayer(env, "MonteCarloES")
# player2 = SemiGradientSarsaPlayer(env, "Sarsa", features)
# trainedPlayer = SemiGradientSarsaPlayer(env, "SemiGradientSarsa", features=features)
trainedPlayer = NNPlayer(env, "DQNPlayer")

# result = env.run(player1, player2, render=True)
# reward = result.value
# print(reward)

# rewards_by_trial = []
#
# for trial in range(trials):
# 	rewards = 0
# 	cumulative_rewards = []
# 	for ep in range(episodes):
# 		if ep % 100 == 0:
# 			print("trial", trial, "ep", ep)
# 		result = env.run(randomPlayer, player2, render=False)
# 		rewards += result.value
# 		cumulative_rewards.append(rewards)
# 	rewards_by_trial.append(cumulative_rewards)


rewards_by_trial = []

win_1_count = 0
win_2_count = 0

wins_dict = defaultdict(int)

players = [trainedPlayer, randomPlayer]

for trial in range(trials):
	ep_rewards = []
	rewards = 0
	wins = 0
	losses = 0
	draws = 0
	for ep in range(episodes):
		random.shuffle(players)
		result = env.run(*players, render=False)
		rewards += result.value
		ep_rewards.append(rewards)
		wins += max(0, result.value)
		losses += max(0, -result.value)
		draws += (abs(result.value) + 1) % 2
		if result == ResultType.WIN1:
			print("winner", players[0].name)
			wins_dict[players[0]] += 1
		elif result == ResultType.WIN2:
			print("winner", players[1].name)
			wins_dict[players[1]] += 1
	rewards_by_trial.append(ep_rewards)

print(wins_dict)

#
# for trial in range(trials):
# 	episode_rewards = []
# 	rewards = 0
# 	for ep in range(episodes):
# 		if ep % 100 == 0:
# 			print("training trial", trial, "ep", ep)
# 		# player2 = MonteCarloESPlayer(env, "MonteCarloES")
# 		players = [trainedPlayer, randomPlayer]
# 		first_player = random.randrange(2)
# 		second_player = 1 if first_player == 0 else 0
# 		result = env.run(players[first_player], players[second_player])
# 		if result == ResultType.WIN1:
# 			win_1_count += 1
# 			wins[players[first_player].name] += 1
# 		elif result == ResultType.WIN2:
# 			win_2_count += 1
# 			wins[players[second_player].name] += 1
# 		if ep % 100 == 0:
# 			print("first player", players[first_player].name)
# 			print("reward: ", result)
# 		# if first_player == 0 and result == ResultType.WIN1:
# 		# 	rewards += 1
# 		# elif first_player == 1 and result == ResultType.WIN2:
# 		# 	rewards += 1
# 		if first_player == 0:
# 			rewards += result.value
# 		else:
# 			if result.value == 0.5:
# 				rewards += result.value
# 			else:
# 				rewards -= result.value
# 		episode_rewards.append(rewards)
# 	rewards_by_trial.append(episode_rewards)
#
# print("win 1", win_1_count, "win 2", win_2_count)
# print(wins)

average_cumulative_rewards = np.average(np.array(rewards_by_trial), axis=0)

plt.plot(average_cumulative_rewards, label="against random")
error = 1.96* statistics.pstdev(average_cumulative_rewards)/math.sqrt(episodes)
plt.fill_between(range(episodes), [x-error for x in average_cumulative_rewards],
				 [x+error for x in average_cumulative_rewards], alpha=0.2)

plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Trained MC Agent")
plt.show()

with open(file_name, 'a') as fd:
	fd.write("trained agent" + "," + " ".join([str(x) for x in average_cumulative_rewards.tolist()]) + "\n")


features_to_test = {"adjacent features for both players": create_adj_features(BOARD[1]),
					"connection 2 features for both players": create_adj_features(BOARD[1]) + create_adj_x_features(BOARD[1], 2),
					"connection 3 features for both players": create_adj_features(BOARD[1]) + create_adj_x_features(BOARD[1], 2) + create_adj_x_features(BOARD[1], 3),
					"adjacent features for just agent": create_one_player_adj_x_features(BOARD[1], 1, -1),
					"connection 2 features for just agent": create_one_player_adj_x_features(BOARD[1], 1, -1) + create_one_player_adj_x_features(BOARD[1], 2, -1),
					"connection 3 features for just agent": create_one_player_adj_x_features(BOARD[1], 1, -1) + create_one_player_adj_x_features(BOARD[1], 2, -1) + create_one_player_adj_x_features(BOARD[1], 3, -1)}


# def get_features_to_test(player):
# 	return {"adjacent features for both players": create_adj_features(BOARD[1]),
# 						"connection 2 features for both players": create_adj_features(BOARD[1]) + create_adj_x_features(BOARD[1], 2),
# 						"connection 3 features for both players": create_adj_features(BOARD[1]) + create_adj_x_features(BOARD[1], 2) + create_adj_x_features(BOARD[1], 3),
# 						"adjacent features for just agent": create_one_player_adj_x_features(BOARD[1], 1, player),
# 						"connection 2 players for just agent": create_one_player_adj_x_features(BOARD[1], 1, player) + create_one_player_adj_x_features(BOARD[1], 2, player),
# 						"connection 3 players for just agent": create_one_player_adj_x_features(BOARD[1], 1, player) + create_one_player_adj_x_features(BOARD[1], 2, player) + create_one_player_adj_x_features(BOARD[1], 3, player)}


# rewards_by_trial = defaultdict(list)
#
# players = {}
#
# for label in features_to_test.keys():
# 	players[label] = SemiGradientSarsaPlayer(env, "Sarsa " + label, features=features_to_test[label])
#
# wins = defaultdict(int)
#
# for trial in range(trials):
# 	rewards = defaultdict(int)
# 	cumulative_rewards = defaultdict(list)
# 	# player2 = MonteCarloESPlayer(env, "MonteCarloES")
# 	# player2 = SemiGradientSarsaPlayer(env, "SemiGradientSarsa", features=features)
# 	for ep in range(episodes):
# 		if ep % 100 == 0:
# 			print("trial", trial, "ep", ep)
# 		for label in features_to_test.keys():
# 			game_players = [players[label], randomPlayer]
# 			first_player = random.randrange(2)
# 			second_player = 1 if first_player == 0 else 0
# 			result = env.run(game_players[first_player], game_players[second_player])
# 			if result == ResultType.WIN1:
# 				win_1_count += 1
# 				wins[game_players[first_player].name] += 1
# 			elif result == ResultType.WIN2:
# 				win_2_count += 1
# 				wins[game_players[second_player].name] += 1
# 			if first_player == 0 and result == ResultType.WIN1:
# 				rewards[label] += 1
# 			elif first_player == 1 and result == ResultType.WIN2:
# 				rewards[label] += 1
# 			cumulative_rewards[label].append(rewards[label])
# 	for label in features_to_test.keys():
# 		rewards_by_trial[label].append(cumulative_rewards[label])
#
# print(rewards_by_trial)
# print(wins)
#
#
# for label in features_to_test.keys():
# 	average_cumulative_rewards = np.average(np.array(rewards_by_trial[label]), axis=0)
#
# 	with open(file_name, 'a') as fd:
# 		fd.write(label + "," + " ".join([str(x) for x in average_cumulative_rewards.tolist()]) + "\n")
#
# 	plt.plot(average_cumulative_rewards, label=label)
# 	error = 1.96* statistics.pstdev(average_cumulative_rewards)/math.sqrt(episodes)
# 	plt.fill_between(range(episodes), [x-error for x in average_cumulative_rewards],
# 					 [x+error for x in average_cumulative_rewards], alpha=0.2)
#
# plt.legend()
# plt.xlabel("Episodes")
# plt.ylabel("Rewards")
# plt.title("Sarsa Agent Against Random with Random Starts")
# plt.show()