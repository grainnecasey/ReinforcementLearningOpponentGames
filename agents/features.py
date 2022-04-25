import numpy as np

def column_adj_player_features(col, player):
	def F(state, action):
		val = 0
		for index in list(reversed(range(state.shape[0]))):
			if state[index][col] == 0:
				if index > 0:
					if state[index-1][col] == player:
						val = 1
				if col > 0:
					if state[index][col-1] == player:
						val = 1
				if col < state.shape[1] - 1:
					if state[index][col+1] == player:
						val = 1
		return val
	return F


def create_adj_features(cols: int):
	features = []
	for col in range(cols):
		features.append(column_adj_player_features(col, 1))
		features.append(column_adj_player_features(col, -1))
	return features


ADJACENT_FEATURES = create_adj_features(7)


def adjacent_connect_x_features(col, player, x):
	def F(state, action):
		# Test rows
		val = 0
		for index in list(reversed(range(state.shape[0]))):
			if state[index][col] == 0:
				if col - x >= 0:
					connection = sum(state[index][col-x:col])
				elif col + x < state.shape[1]:
					connection = sum(state[index][col:col+x])
				else:
					connection = 0
				if connection == player * x:
					val = 1
				break

		# Test columns
		# print(state)
		for index in list(reversed(range(state.shape[0]))):
			if state[index][col] == 0:
				# print("index", index, "col", col, "player", player, "x", x)
				column = state[:, col]
				# print("column", column)
				x_set = column[index-x:index]
				connection = sum(x_set)
				# print("connection", connection)
				if connection == player * x:
					val = 1
				break

		# Test diagonal
		for index in list(reversed(range(state.shape[0]))):
			if state[index][col] == 0:
				value = 0
				for i in range(x):
					if index+i < state.shape[0] and col+i < state.shape[1]:
						value += state[index+i][col+i]
				# print("connection", value)
				if value == x * player:
					val = 1
				break

		# Test reverse diagonal
		for index in list(reversed(range(state.shape[0]))):
			if state[index][col] == 0:
				value = 0
				for i in range(x):
					if index - i >= 0 and col - i >= 0:
						value += state[index - i][col - i]
				if value == x * player:
					val = 1
				break
		return val
	return F


def create_adj_x_features(cols: int, connection: int):
	features = []
	for col in range(cols):
		features.append(adjacent_connect_x_features(col, 1, connection))
		features.append(adjacent_connect_x_features(col, -1, connection))
	return features


def create_one_player_adj_x_features(cols: int, connection: int, player: int):
	features = []
	for col in range(cols):
		features.append(adjacent_connect_x_features(col, player, connection))
	return features


ADJ_1_AND_2_CONNECTION_FEATURES = create_adj_features(7) + create_adj_x_features(7, 2)

ADJ_1_2_3_CONNECTION_FEATURES = create_adj_features(7) + create_adj_x_features(7, 2) + create_adj_x_features(7, 3)
