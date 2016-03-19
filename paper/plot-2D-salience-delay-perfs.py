import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

possible_cues = {"[3, 0]" : "A", "[3, 1]" : "B" , "[3, 2]" : "C", "[2, 0]" : "D", "[2, 1]" : "E", "[1, 0]" : "F"}


perfs = np.load("icann_2D_perf_salience_delay_allcues.npy")

saliences = np.arange(21)*2
delays = np.arange(21)*2
figs = 0
fig = plt.figure()
for c in range(len(possible_cues)):
    ax = fig.add_subplot(210+(c%2), projection='3d')
    ax.set_title(possible_cues.keys()[c])
    ax.set_xlabel("Salience")
    ax.set_ylabel("Delay (ms)")
    ax.set_zlabel("Performance")
    for i in range(saliences.size):
        ax.plot(np.zeros(21)+saliences[i], delays, perfs[:][possible_cues.keys()[c]].mean(axis=2)[i,:])
    for i in range(delays.size):
        ax.plot(saliences, np.zeros(21)+delays[i], perfs[:][possible_cues.keys()[c]].mean(axis=2)[:,i])
    if (c+1)%2 == 0:
        plt.tight_layout()
        plt.savefig("Icann-2D-salience-delay-"+str(figs)+".pdf", dpi=100)
        figs += 1
        fig = plt.figure(figs+1)
plt.show()
