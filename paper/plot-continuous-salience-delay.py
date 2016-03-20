import numpy as np
import matplotlib.pyplot as plt

all_perf = np.load("Feb_mean_perf_salience_delay_allcues.npy")
saliences = np.arange(36)*7.0/100
sals_for_plot = saliences*100/7
delays = np.arange(41)

possible_cues = {"[3, 0]" : "A / D",
                 "[3, 1]" : "A / C" ,
                 "[3, 2]" : "C / B",
                 "[2, 0]" : "C / D",
                 "[2, 1]" : "C / B",
                 "[1, 0]" : "C / D"}
cue_set1 = ([3, 0], [2, 0], [1, 0])
cue_set2 = ([3, 1], [2, 1])
cue_set3 = ([3, 2], [])

rgb = ('r','g','b')

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def gethalfperf(points):
    idx = np.array([])
    prox = 0
    if 1 :
#    while idx.size == 0 :
        prox += 0.05
        idx = np.argwhere(np.isclose(points, 0.5, atol=prox)).reshape(-1)
    return idx

def eachset(cueset, sb):
    i = 0
    plt.figure(1)
    ax = plt.subplot(3,1,sb)
    ax.set_xlim((0, sals_for_plot[-1]+5))
    ax.set_ylim((0, 1.2))
    ax.set_yticks( [0.0, 0.5, 1.0])
    
    ax.set_ylabel('Performance', fontsize=12)
    for st in cueset:
        if not st : continue 
        stim = str(st)
        clr = rgb[i]
        perfs = all_perf[stim]["salience"][0]
        nearidx = find_nearest(perfs.mean(axis=1), 0.5)
        ax.plot((sals_for_plot[nearidx],sals_for_plot[nearidx]), (0,perfs.mean(axis=1)[nearidx]), c=clr, ls='dashed')
        ax.plot(sals_for_plot, perfs.mean(axis=1), c=clr, lw=1, label='Stimuli '+ possible_cues[stim] ) #+ '('+stim+')')
        ax.plot(sals_for_plot, perfs.mean(axis=1)+perfs.var(axis=1), c=clr, lw=.5)
        ax.plot(sals_for_plot, perfs.mean(axis=1)-perfs.var(axis=1), c=clr, lw=.5)
        ax.fill_between(sals_for_plot, perfs.mean(axis=1)+perfs.var(axis=1), perfs.mean(axis=1)-perfs.var(axis=1), color=clr, alpha=.1)
        i += 1
    ax.plot((0,sals_for_plot[-1]), (0.5,0.5), c='black', lw=1)
    ax.legend(loc="lower left", frameon=False, prop={'size':6})

    i = 0
    plt.figure(2)
    ax = plt.subplot(3,1,sb)
    ax.set_xlim((0, delays[-1]+5))
    ax.set_ylim((0, 1.2))
    ax.set_yticks( [0.0, 0.5, 1.0])
    ax.set_ylabel('Performance', fontsize=12)
    for st in cueset:
        if not st : continue 
        stim = str(st)
        clr = rgb[i]
        perfs = all_perf[stim]["delay"][0]
        nearidx = find_nearest(perfs.mean(axis=1), 0.5)
        ax.plot((delays[nearidx],delays[nearidx]), (0,perfs.mean(axis=1)[nearidx]), c=clr, ls='dashed')
        ax.plot(delays , perfs.mean(axis=1), c=clr, lw=1, label='Stimuli '+ possible_cues[stim] ) # + '('+stim+')')
        ax.plot(delays, perfs.mean(axis=1)+perfs.var(axis=1), c=clr, lw=.5)
        ax.plot(delays, perfs.mean(axis=1)-perfs.var(axis=1), c=clr, lw=.5)
        ax.fill_between(delays, perfs.mean(axis=1)+perfs.var(axis=1), perfs.mean(axis=1)-perfs.var(axis=1), color=clr, alpha=.1)
        i += 1
    ax.plot((0,delays[-1]), (0.5,0.5), c='black', lw=1)
    ax.legend(loc="lower left", frameon=False, prop={'size':6})

eachset(cue_set1, 1)
eachset(cue_set2, 2)
eachset(cue_set3, 3)

plt.figure(1)
plt.suptitle("Change in performance with relative salience", fontsize=20)
#plt.tight_layout()
plt.xlabel(' Relative difference in salience ', fontsize=12)
plt.savefig("perf-vs-salience.pdf", dpi=50)
plt.figure(2)
plt.suptitle("Change in performance with delay in presentation", fontsize=20)
#plt.tight_layout()
plt.xlabel('Delay(ms) in presentation of higher rewarding cue', fontsize=12)
plt.savefig("perf-vs-delay.pdf", dpi=100)
plt.show()
