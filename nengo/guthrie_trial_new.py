import nengo
import nengolib
import numpy as np
import matplotlib.pyplot as plt
from nengo.processes import PresentInput

inps = 4
arrpop = 200
pop = 50
tau=0.01
cmin=0
cmax=1000
Vmin=0
Vmax=20
Vh=16
Vc=3
bw=1
Ixt = 7
v = 7
Wmin, Wmax = 0.25, 0.75

rest = np.array([3, 0, -10, 40, 10]) #[CTX, STR, GPI, THL, STN] 



def clamp(y, ind):
    x = y+rest[ind]
    if x < cmin: return cmin
    if x > cmax: return cmax
    return x

def sigmoid(Vo, ind):
    V = Vo + rest[ind]
    return Vmin + (Vmax-Vmin)/(1.0+np.exp((Vh-V)/Vc))

def weights(shape):
    N = np.random.normal(0.5, 0.005, shape)
    N = np.minimum(np.maximum(N, 0.0),1.0)
    return (Wmin+(Wmax-Wmin)*N)

def OneToOne(gain=1.0):
    return bw*gain

def OneToAll(gain=1.0):
    return np.array([[bw*gain]*inps for i in range(inps)])

def CogToAsc(gain=1.0, norm=True):
    if norm: wts_n = weights((inps,inps*inps))
    else : wts_n = np.ones((inps,inps*inps))
    wts  = np.zeros((inps,inps*inps))
    for i in range(inps):
        asc = np.zeros((inps, inps))
        asc[i] = np.ones(inps)
        wts[i] = asc.reshape((inps*inps))
    lc = [a*b*gain for a,b in zip(wts_n,wts)]
    return np.array(lc)

def MotToAsc(gain=1.0, norm=True):
    if norm : wts_n = weights((inps,inps*inps))
    else : wts_n = np.ones((inps,inps*inps))
    wts = np.zeros((inps, inps*inps))
    for i in range(inps):
        asc = np.zeros((inps, inps))
        asc[:,i] = np.ones(inps)
        wts[i] = asc.reshape((inps*inps))
    lc = [a*b*gain for a,b in zip(wts_n,wts)]
    return np.array(lc)

def fill(stim):
    set = []
    for i in stim:
        set.append([-1, -1])
        set.append(i)
        set.append(i)
        set.append(i)
        set.append(i)
        set.append([-1, -1])
    return np.array(set)

def precogstim(chan):
    stim = np.zeros(inps)
    if chan[0] < 0 or chan[1] < 0 : # Trial reset
        return stim
    noise = 0.01
    stim[chan[0]] = Ixt + np.random.normal(0,v*noise)
    stim[chan[1]] = Ixt + np.random.normal(0,v*noise)
    return stim

def premotstim(chan):
    stim = np.zeros(inps)
    if chan[0] < 0 or chan[1] < 0 : # Trial reset
        return stim
    noise = 0.01
    stim[chan[0]] = Ixt + np.random.normal(0,v*noise)
    stim[chan[1]] = Ixt + np.random.normal(0,v*noise)
    return stim

def ascstim(asc):
    c1, c2, m1, m2 = asc
    stim = np.zeros((inps, inps))
    if c1 < 0 or c2 < 0 or m1 < 0 or m2 < 0: # Trial reset
        return stim.reshape(inps*inps)
    noise = 0.01
    stim[c1, m1] = Ixt + np.random.normal(0,v*noise)
    stim[c2, m2] = Ixt + np.random.normal(0,v*noise)
    return stim.reshape(inps*inps)


CTX_noise = 0.01
STR_noise = 0.001
STN_noise = 0.001
GPI_noise = 0.03
THL_noise = 0.001

radius = 70*1.42   

#cogstim=np.array([[0, 1]])
#motstim=np.array([[3, 2]])

model = nengo.Network('Action Selection')
model.config[nengo.Ensemble].neuron_type = nengo.Direct()#Sigmoid(tau_ref=0.001125)

W = weights(4)#np.array([ 0.53677366,  0.56135171,  0.49578237,  0.49318303])

asctoasc = np.zeros((inps*inps, inps*inps))

def runTrial(cogstim, motstim):
    global asctoasc
    with model:
        CTX_COG = nengo.networks.EnsembleArray(n_ensembles = inps, n_neurons = arrpop*4, radius=radius)
        STR_COG = nengo.networks.EnsembleArray(n_ensembles = inps, n_neurons = arrpop, radius=radius)
        GPI_COG = nengo.networks.EnsembleArray(n_ensembles = inps, n_neurons = arrpop, radius=radius)
        STN_COG = nengo.networks.EnsembleArray(n_ensembles = inps, n_neurons = arrpop, radius=radius)
        THL_COG = nengo.networks.EnsembleArray(n_ensembles = inps, n_neurons = arrpop, radius=radius)
    
        CTX_MOT = nengo.networks.EnsembleArray(n_ensembles = inps, n_neurons = arrpop*4, radius=radius)
        STR_MOT = nengo.networks.EnsembleArray(n_ensembles = inps, n_neurons = arrpop, radius=radius)
        GPI_MOT = nengo.networks.EnsembleArray(n_ensembles = inps, n_neurons = arrpop, radius=radius)
        STN_MOT = nengo.networks.EnsembleArray(n_ensembles = inps, n_neurons = arrpop, radius=radius)
        THL_MOT = nengo.networks.EnsembleArray(n_ensembles = inps, n_neurons = arrpop, radius=radius)
    
        # Assuming both are 2D arrays of [inps, inps] represented in 1D array of size inps*inps
        # Each row corresponds to a cue and each column to a position
    
        CTX_ASC = nengo.networks.EnsembleArray(n_ensembles = inps*inps, n_neurons = pop*inps*inps, radius=radius)
        STR_ASC = nengo.networks.EnsembleArray(n_ensembles = inps*inps, n_neurons = pop*inps, radius=radius)
    
        CTX_COG_out = CTX_COG.add_output('func_cog_ctx', lambda x: clamp(x+CTX_noise,0))
        STR_COG_out = STR_COG.add_output('func_cog_str', lambda V: sigmoid(V+STR_noise,1))
        GPI_COG_out = GPI_COG.add_output('func_cog_gpi', lambda x: clamp(x+GPI_noise,2))
        STN_COG_out = STN_COG.add_output('func_cog_stn', lambda x: clamp(x+STN_noise,4))
        THL_COG_out = THL_COG.add_output('func_cog_thl', lambda x: clamp(x+THL_noise,3))
    
        CTX_MOT_out = CTX_MOT.add_output('func_mot_ctx', lambda x: clamp(x+CTX_noise,0))
        STR_MOT_out = STR_MOT.add_output('func_mot_str', lambda V: sigmoid(V+STR_noise,1))
        GPI_MOT_out = GPI_MOT.add_output('func_mot_gpi', lambda x: clamp(x+GPI_noise,2))
        STN_MOT_out = STN_MOT.add_output('func_mot_stn', lambda x: clamp(x+STN_noise,4))
        THL_MOT_out = THL_MOT.add_output('func_mot_thl', lambda x: clamp(x+THL_noise,3))
    
        CTX_ASC_out = CTX_ASC.add_output('func_asc_ctx', lambda x: clamp(x+CTX_noise,0))
        STR_ASC_out = STR_ASC.add_output('func_asc_str', lambda V: sigmoid(V+STR_noise,1))
    
    #np.array([ 0.56135171,  0.53677366,  0.49578237,  0.49318303])
    
        conn = nengo.Connection(CTX_COG_out, STR_COG.input, transform=W, synapse = tau)
        nengo.Connection(CTX_MOT_out, STR_MOT.input, transform=weights(inps), synapse = tau)
        asctoasc[:] = 0
        np.fill_diagonal(asctoasc, weights(inps*inps))
        nengo.Connection(CTX_ASC_out, STR_ASC.input, transform=asctoasc, synapse = tau)
        nengo.Connection(CTX_COG_out, STR_ASC.input, transform=CogToAsc(gain=0.2).transpose(), synapse = tau)
        nengo.Connection(CTX_MOT_out, STR_ASC.input, transform=MotToAsc(gain=0.2).transpose(), synapse = tau)
        nengo.Connection(STR_COG_out, GPI_COG.input, transform=OneToOne(gain=-2.0), synapse = tau)
        nengo.Connection(STR_MOT_out, GPI_MOT.input, transform=OneToOne(gain=-2.0), synapse = tau)
        nengo.Connection(STR_ASC_out, GPI_COG.input, transform=CogToAsc(gain=-2.0, norm=False), synapse = tau)
        nengo.Connection(STR_ASC_out, GPI_MOT.input, transform=MotToAsc(gain=-2.0, norm=False), synapse = tau)
        nengo.Connection(GPI_COG_out, THL_COG.input, transform=OneToOne(gain=-0.5), synapse = tau)
        nengo.Connection(GPI_MOT_out, THL_MOT.input, transform=OneToOne(gain=-0.5), synapse = tau)
        nengo.Connection(THL_COG_out, CTX_COG.input, transform=OneToOne(gain=1.0), synapse = tau)
        nengo.Connection(THL_MOT_out, CTX_MOT.input, transform=OneToOne(gain=1.0), synapse = tau)
        nengo.Connection(CTX_COG_out, THL_COG.input, transform=OneToOne(gain=0.4), synapse = tau)
        nengo.Connection(CTX_MOT_out, THL_MOT.input, transform=OneToOne(gain=0.4), synapse = tau)
        nengo.Connection(CTX_COG_out, STN_COG.input, transform=OneToOne(gain=1.0), synapse = tau)
        nengo.Connection(CTX_MOT_out, STN_MOT.input, transform=OneToOne(gain=1.0), synapse = tau)
        nengo.Connection(STN_COG_out, GPI_COG.input, transform=OneToAll(gain=1.0), synapse = tau)
        nengo.Connection(STN_MOT_out, GPI_MOT.input, transform=OneToAll(gain=1.0), synapse = tau)
    
        stim_cog = nengo.Node([-1, -1])
        stim_mot = nengo.Node([-1, -1])
        stim_asc = nengo.Node([-1, -1, -1, -1])
    
        stim_cog.output = PresentInput(inputs=fill(cogstim), presentation_time=0.5)
        stim_mot.output = PresentInput(inputs=fill(motstim), presentation_time=0.5)
        stim_asc.output = PresentInput(inputs=np.hstack([fill(cogstim), fill(motstim)]), presentation_time=0.5)
    
        stim_asc_r = nengo.Ensemble(pop*(inps*inps),inps*inps, radius=20)
        nengo.Connection(stim_asc, stim_asc_r, function=ascstim, synapse = None)
        nengo.Connection(stim_asc_r, CTX_ASC.input, synapse = None)
        nengo.Connection(stim_cog, CTX_COG.input, function=precogstim, synapse = None)
        nengo.Connection(stim_mot, CTX_MOT.input, function=premotstim, synapse = None)
    
        weights_p = nengo.Probe(conn, 'weights', synapse=None, sample_every=3.0)
        ctxcog = nengo.Probe(CTX_COG.output, synapse=None, sample_every=2.0)
        ctxmot = nengo.Probe(CTX_MOT.output, synapse=None, sample_every=2.0)
        strcog = nengo.Probe(STR_COG.output, synapse=None, sample_every=2.0)

    with nengo.Simulator(model) as sim:
        sim.reset()
        sim.run(3.0)
    t = sim.trange()
    return sim.data[ctxcog], sim.data[ctxmot], sim.data[strcog], sim.data[weights_p]


np.set_printoptions(suppress=True)
cues_mot = np.array([0,1,2,3])
cues_cog = np.array([0,1,2,3])
cues_value = np.ones(4) * 0.5
cues_reward = np.array([3.0,2.0,1.0,0.0])/3.0

threshold  = 40
alpha_c    = 0.025
alpha_LTP  = 0.004
alpha_LTD  = 0.002

P=[]
R=[]

def learn(ctxcog, ctxmot, strcog, debug=True):
    global P, R, W
    # A motor decision has been made
    c1, c2 = cues_cog[:2]
    m1, m2 = cues_mot[:2]
    mot_choice = np.argmax(ctxmot)
    cog_choice = np.argmax(ctxcog)
    # Only the motor decision can designate the chosen cue
    if mot_choice == m1:
        choice = c1
    else:
        choice = c2
    if choice == min(c1,c2):
        P.append(1)
    else:
        P.append(0)
    reward = np.random.uniform(0,1) < cues_reward[choice]
    R.append(reward)
    error = reward - cues_value[choice]
    # Update cues values
    cues_value[choice] += error* alpha_c
    # Learn
    lrate = alpha_LTP if error > 0 else alpha_LTD
    dw = error * lrate * strcog[choice]
    W[choice] = W[choice] + dw * (W[choice]-Wmin)*(Wmax-W[choice])
    if not debug: return
    # Just for displaying ordered cue
    oc1,oc2 = min(c1,c2), max(c1,c2)
    if choice == oc1:
        print "Choice:          [%d] / %d  (good)" % (oc1,oc2)
    else:
        print "Choice:           %d / [%d] (bad)" % (oc1,oc2)
    print "Reward (%3d%%) :   %d" % (int(100*cues_reward[choice]),reward)
    print "Mean performance: %.3f" % np.array(P)[-20:].mean()
    print "Mean reward:      %.3f" % np.array(R).mean()



for i in range(10):    
    np.random.shuffle(cues_cog)
    np.random.shuffle(cues_mot)
    c1,c2 = cues_cog[:2]
    m1,m2 = cues_mot[:2]
    ctxcog, ctxmot, strcog, wt = runTrial(np.array([[c1, c2]]), np.array([[m1, m2]]))
    learn(ctxcog[0], ctxmot[0], strcog[0], debug=False)
    print "cues (%d, %d) shown at (%d, %d)" % (c1, c2, m1, m2)
    print "Motor Cortex activity - %s" % str(ctxmot)
    print "Weights after trial %d : %s" % (i, str(wt))
np.save(P, "performance.npy")
np.save(W, "weights.npy")
