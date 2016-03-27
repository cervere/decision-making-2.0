# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License.
#
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------
# References:
#
# * Interaction between cognitive and motor cortico-basal ganglia loops during
#   decision making: a computational study. M. Guthrie, A. Leblois, A. Garenne,
#   and T. Boraud. Journal of Neurophysiology, 109:3025–3040, 2013.
# -----------------------------------------------------------------------------
import numpy as np
from cytmodel import *


tau     = 0.01
clamp   = Clamp(min=0, max=1000)
sigmoid = Sigmoid(Vmin=0, Vmax=20, Vh=16, Vc=3)

getWeights = False

CTX = AssociativeStructure(
                 tau=tau, rest=- 3.0, noise=0.010, activation=clamp )
STR = AssociativeStructure(
                 tau=tau, rest=  0.0, noise=0.001, activation=sigmoid )
STN = Structure( tau=tau, rest=-10.0, noise=0.001, activation=clamp )
GPI = Structure( tau=tau, rest=+10.0, noise=0.030, activation=clamp )
THL = Structure( tau=tau, rest=-40.0, noise=0.001, activation=clamp )
structures = (CTX, STR, STN, GPI, THL)

def weights(shape):
    Wmin, Wmax = 0.25, 0.75
    N = np.random.normal(0.5, 0.005, shape)
    N = np.minimum(np.maximum(N, 0.0),1.0)
    return (Wmin+(Wmax-Wmin)*N)

if getWeights:
    W = weights(4)
else:
    WS = np.load("cog-weights.npy")
    W = WS[0]

connections = [
    OneToOne( CTX.cog.V, STR.cog.Isyn, W,   gain=+1.0 ),
    OneToOne( CTX.mot.V, STR.mot.Isyn, weights(4),   gain=+1.0 ),
    OneToOne( CTX.ass.V, STR.ass.Isyn, weights(4*4), gain=+1.0 ),
    CogToAss( CTX.cog.V, STR.ass.Isyn, weights(4),   gain=+0.2 ),
    MotToAss( CTX.mot.V, STR.ass.Isyn, weights(4),   gain=+0.2 ),
    OneToOne( CTX.cog.V, STN.cog.Isyn, np.ones(4),   gain=+1.0 ),
    OneToOne( CTX.mot.V, STN.mot.Isyn, np.ones(4),   gain=+1.0 ),
    OneToOne( STR.cog.V, GPI.cog.Isyn, np.ones(4),   gain=-2.0 ),
    OneToOne( STR.mot.V, GPI.mot.Isyn, np.ones(4),   gain=-2.0 ),
    AssToCog( STR.ass.V, GPI.cog.Isyn, np.ones(4),   gain=-2.0 ),
    AssToMot( STR.ass.V, GPI.mot.Isyn, np.ones(4),   gain=-2.0 ),
    OneToAll( STN.cog.V, GPI.cog.Isyn, np.ones(4),   gain=+1.0 ),
    OneToAll( STN.mot.V, GPI.mot.Isyn, np.ones(4),   gain=+1.0 ),
    OneToOne( GPI.cog.V, THL.cog.Isyn, np.ones(4),   gain=-0.5 ),
    OneToOne( GPI.mot.V, THL.mot.Isyn, np.ones(4),   gain=-0.5 ),
    OneToOne( THL.cog.V, CTX.cog.Isyn, np.ones(4),   gain=+1.0 ),
    OneToOne( THL.mot.V, CTX.mot.Isyn, np.ones(4),   gain=+1.0 ),
    OneToOne( CTX.cog.V, THL.cog.Isyn, np.ones(4),   gain=+0.4 ),
    OneToOne( CTX.mot.V, THL.mot.Isyn, np.ones(4),   gain=+0.4 ),
]


cues_mot = np.array([0,1,2,3])
cues_cog = np.array([0,1,2,3])
cues_value = np.ones(4) * 0.5
cues_reward = np.array([3.0,2.0,1.0,0.0])/3.0

def start_trial(salience_off=0):
    CTX.mot.Iext = 0
    CTX.cog.Iext = 0
    CTX.ass.Iext = 0

def first_stimulus(cues=[], mot=[], salience=0):
    m1 = mot[0]
    c1 = cues[0]
    v = 7
    noise = 0.01
    CTX.mot.Iext[m1]  = v + np.random.normal(0,v*noise)
    CTX.cog.Iext[c1]  = v + np.random.normal(0,v*noise) + salience
    CTX.ass.Iext[c1*4+m1] = v + np.random.normal(0,v*noise)

def second_stimulus(cues=[], mot=[]):
    m2 = mot[1]
    c2 = cues[1]
    v = 7
    noise = 0.01
    CTX.mot.Iext[m2]  = v + np.random.normal(0,v*noise)
    CTX.cog.Iext[c2]  = v + np.random.normal(0,v*noise)
    CTX.ass.Iext[c2*4+m2] = v + np.random.normal(0,v*noise)

def stop_trial():
    CTX.mot.Iext = 0
    CTX.cog.Iext = 0
    CTX.ass.Iext = 0

    
def iterate(dt):
    global connections, structures

    # Flush connections
    for connection in connections:
        connection.flush()

    # Propagate activities
    for connection in connections:
        connection.propagate()

    # Compute new activities
    for structure in structures:
        structure.evaluate(dt)

def reset():
    global cues_values, structures
    cues_value = np.ones(4) * 0.5
    for structure in structures:
        structure.reset()

dt = 0.001
def run_session(stim=[], mot=[], delay=0, salience_off=0):
    for j in range(1):
        reset()
        for i in xrange(0,500):
            iterate(dt)
        first_stimulus(stim, mot, salience_off)
        for i in xrange(500,2500):
            if i == 500 + delay:
                second_stimulus(stim, mot)
            iterate(dt)
        stop_trial()
        for i in xrange(2500,3000):
            iterate(dt)

shape = ["A", "B", "C", "D"]
pos = ["Up", "Down", "Left", "Right"]

np.random.shuffle(cues_cog)
c1, c2 = [0, 3] #cues_cog[:2]
cues = [max(c1, c2), min(c1, c2)]
np.random.shuffle(cues_mot)
mot = cues_mot[:2]
stim_title = shape[cues[0]] + "(" + pos[cues_mot[0]] + ") - "+ shape[cues[1]] + "(" + pos[cues_mot[1]] + ")"  


# -----------------------------------------------------------------------------
from display import *

dtype = [ ("CTX", [("mot", float, 4), ("cog", float, 4), ("ass", float, 16)]),
          ("STR", [("mot", float, 4), ("cog", float, 4), ("ass", float, 16)]),
          ("GPI", [("mot", float, 4), ("cog", float, 4)]),
          ("THL", [("mot", float, 4), ("cog", float, 4)]),
          ("STN", [("mot", float, 4), ("cog", float, 4)])]

plots = ["ideal", "salience", "delay"]
def plot(stn, i):
    history = np.zeros(3000, dtype=dtype)
    history["CTX"]["mot"]   = CTX.mot.history[:3000]
    history["CTX"]["cog"]   = CTX.cog.history[:3000]
    history["CTX"]["ass"]   = CTX.ass.history[:3000]
    history["STR"]["mot"] = STR.mot.history[:3000]
    history["STR"]["cog"] = STR.cog.history[:3000]
    history["STR"]["ass"] = STR.ass.history[:3000]
    history["STN"]["mot"]      = STN.mot.history[:3000]
    history["STN"]["cog"]      = STN.cog.history[:3000]
    history["GPI"]["mot"]      = GPI.mot.history[:3000]
    history["GPI"]["cog"]      = GPI.cog.history[:3000]
    history["THL"]["mot"] = THL.mot.history[:3000]
    history["THL"]["cog"] = THL.cog.history[:3000]
    display_ctx(stim_title+": "+stn, history, 3.0, filename="decisiontime-"+plots[i-1]+".pdf")

run_session(cues, mot)
plot("Ideal Scenario", 1)
run_session(cues, mot, salience_off=1.96)
plot("Salience difference - 28%", 2)
run_session(cues, mot, delay=35)
plot("Delay - 35 ms", 3)

plt.show()
