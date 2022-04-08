import gym
from gym_connect_four.connect_four_env import RandomPlayer
from gym_connect_four.connect_four_env import ConnectFourEnv

env = ConnectFourEnv(board_shape=(6, 7))

player1 = RandomPlayer(env, "Dexter-Bot")
player2 = RandomPlayer(env, "Deedee-Bot")
result = env.run(player1, player2, render=True)
reward = result.value
print(reward)