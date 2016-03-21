import numpy as np
import matplotlib.pyplot as plt

# Dictionnary are not sorted
possible_cues = {"[3, 0]" : "A / D",
                 "[3, 1]" : "A / C" ,
                 "[3, 2]" : "A / B",
                 "[2, 0]" : "B / D",
                 "[2, 1]" : "B / C",
                 "[1, 0]" : "C / D"}

# Sorted list such that they appear in the order want
cues = ["[3, 0]", "[3, 1]", "[3, 2]",
        "[2, 0]", "[2, 1]", "[1, 0]"]


perfs = np.load("icann_2D_perf_salience_delay_allcues.npy")
X = np.arange(21)*2
Y = np.arange(21)*2

fig = plt.figure(figsize=(11,7))
for i,cue in enumerate(cues):
    ax = plt.subplot(2,3,1+i)
    ax.set_title("Stimuli %s" % possible_cues[cue], x=0.5, y=0.875, color="w", fontsize=14)
    ax.set_xlabel("Salience (%)")
    ax.set_ylabel("Delay (ms)")
    Z = np.zeros((X.size, Y.size))
    for j in range(X.size):
        Z[j] = perfs[:][cue].mean(axis=2)[j,:]
    levels = 6
    plt.contourf(X, Y, Z, levels, alpha=.75, cmap=plt.cm.gray, aspect=1, vmin=0, vmax=1)
    C = plt.contour(X, Y, Z, levels, colors='black', linewidth=.5, aspect=1, vmin=0, vmax=1)
    plt.clabel(C, inline=1, fontsize=10)

plt.tight_layout()
plt.savefig("perf-vs-delay-vs-salience.pdf")
plt.show()
