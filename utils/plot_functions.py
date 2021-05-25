import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils_functions import get_features


# TODO: win_rates
win_rates = []

t = np.arange(len(win_rates['Score']))
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Timesteps')
ax1.set_ylabel('Score', color=color)
ax1.plot(t, win_rates['Score'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Rank', color=color)  # we already handled the x-label with ax1
ax2.plot(t, win_rates['Rank'], color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Performance of G-net vs 3 Greedy Agents')
plt.show()


def plot_features(features):
    fig, axs = plt.subplots(3, 4, figsize=(20, 10))
    for i in range(3):
        for j in range(4):
            sns.heatmap(features[i * 4 + j], ax=axs[i, j], cmap='Blues',
                        vmin=0, vmax=1, linewidth=2, linecolor='black', cbar=False)


def get_example_features(env):
    observation = {}
    observation['step'] = 104
    observation['index'] = 0
    observation['geese'] = [[46, 47, 36, 37, 48, 59, 58, 69],
                            [5, 71, 72, 6, 7, 73, 62, 61, 50, 51, 52, 63, 64, 53, 54],
                            [12, 11, 21, 20, 19, 8, 74, 75, 76, 65, 55, 56, 67, 1],
                            [23, 22, 32, 31, 30, 29, 28, 17, 16, 27, 26, 15, 14, 13, 24]]
    observation['food'] = [45, 66]
    prev_heads = [47, 71, 11, 22]
    return get_features(observation, env.configuration, prev_heads)


##features = get_example_features()
##plot_features(features.cpu().detach().numpy())