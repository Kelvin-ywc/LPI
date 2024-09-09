import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

vectors = np.loadtxt('../MID/tasks_array.txt')

tsne = TSNE(n_components=2, random_state=0, perplexity=1)
words = ['appliance', 'sports', 'outdoor', 'electronic', 'accessory', 'indoor', 'kitchen', 'furniture', 'vehicle', 'food', 'animal', 'person']

Y = tsne.fit_transform(vectors)

# colors = cm.rainbow(np.linspace(0, 1, Y.shape[0]))
colors = plt.get_cmap('tab20')(range(12))
for dataset, color, label in zip(Y, colors, words):
    plt.scatter(dataset[0], dataset[1], color=color, label=label, s=120)

plt.xlabel("X")
plt.ylabel("Y")
plt.legend(ncol=4, loc=(0,2/3))
plt.savefig('../MID/task_visual_4.svg')
# for dataset, label in zip(Y, words):
#     plt.annotate(label, (dataset[0], dataset[1]), textcoords='offset points',xytext=(0,10), ha='center')
plt.show()

# plt.savefig('../MID/task_visual.png')