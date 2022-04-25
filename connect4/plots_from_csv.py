import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statistics
import math

plots_against_random = pd.read_csv("../data/sarsa_features_cumulative_reward_4.csv")
plots_against_trained = pd.read_csv("../data/sarsa_features_against_trained_4.csv")

print(plots_against_random)
print(plots_against_random.columns)

cmap = plt.cm.get_cmap('hsv', len(plots_against_random.index) + 1)

greens = ['greenyellow', 'olive', 'limegreen', 'forestgreen', 'mediumspringgreen', 'palegreen']
reds = ['lightcoral', 'brown', 'maroon', 'red', 'coral', 'chocolate']

for index, row in plots_against_random.iterrows():
	color = np.random.rand(3,)
	rewards = [float(x) for x in row['cumulative_rewards'].split()]
	plt.plot(rewards, label=row['label'] + "against random", color=greens.__getitem__(index))
	error = 1.96 * statistics.pstdev(rewards) / math.sqrt(len(rewards))
	plt.fill_between(range(len(rewards)), [x - error for x in rewards],
					 [x + error for x in rewards], color=greens.__getitem__(index), alpha=0.2)

plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Sarsa Features Comparison on Connect 4")
plt.show()

for index, row in plots_against_random.iterrows():
	color = np.random.rand(3,)
	rewards = [float(x) for x in row['cumulative_rewards'].split()]
	plt.plot(rewards, label=row['label'] + "against trained", color=reds.__getitem__(index))
	error = 1.96 * statistics.pstdev(rewards) / math.sqrt(len(rewards))
	plt.fill_between(range(len(rewards)), [x - error for x in rewards],
					 [x + error for x in rewards], color=reds.__getitem__(index), alpha=0.2)

plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Sarsa Features Comparison on Connect 4")
plt.show()