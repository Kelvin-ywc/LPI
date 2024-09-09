import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import numpy as np
from openai import OpenAI
api_key = "sk-MazpnWiEWQhrgtP8526a79F8D7254a5894296d2d81Ea6c7a"
api_base = "https://oneapi.xty.app/v1"

client = OpenAI(api_key=api_key, base_url=api_base)

def get_embedding(text, model="text-embedding-3-large"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


tsne = TSNE(n_components=2, random_state=0, perplexity=1)
words = ['appliance', 'sports', 'outdoor', 'electronic', 'accessory', 'indoor', 'kitchen', 'furniture', 'vehicle', 'food', 'animal', 'person']
vectors = [get_embedding(word) for word in words]
vectors = np.array(vectors)

np.savetxt('../MID/tasks_array.txt', vectors)

Y = tsne.fit_transform(vectors)

# colors = cm.rainbow(np.linspace(0, 1, Y.shape[0]))
colors = plt.get_cmap('tab20')(range(12))
for dataset, color, label in zip(Y, colors, words):
    plt.scatter(dataset[0], dataset[1], color=color, label=label)

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.savefig('../MID/task_visual.png')
# for dataset, label in zip(Y, words):
#     plt.annotate(label, (dataset[0], dataset[1]), textcoords='offset points',xytext=(0,10), ha='center')
plt.show()

# plt.savefig('../MID/task_visual.png')