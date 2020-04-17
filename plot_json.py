import json
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join


with open("output/task3_reward.json") as f:
    reward = json.load(f)


sns.lineplot(x=range(len(reward)), y=[sum(x) for x in reward])
plt.savefig(join('output', 'task3_reward.pdf'))
plt.show()

with open("output/task3_loss.json") as f:
    loss = json.load(f)

sns.lineplot(x=range(len(loss)), y=loss)
plt.savefig(join('output', 'task3_loss.pdf'))
plt.show()
