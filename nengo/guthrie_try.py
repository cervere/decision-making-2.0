import nengo
import numpy as np
from nengo.processes import PresentInput



inps = 4
arrpop = 200
pop = 50
tau=0.01
min=0
max=1000
Vmin=0
Vmax=20
Vh=16
Vc=3
bw=1
Ixt = 7
v = 7
rest = np.array([3, 0, -10, 40, 10]) #[CTX, STR, GPI, THL, STN] 

def clamp(y, ind):
    x = y+rest[ind]
    if x < min: return min
    if x > max: return max
    return x

def sigmoid(Vo, ind):
    V = Vo + rest[ind]
    return Vmin + (Vmax-Vmin)/(1.0+np.exp((Vh-V)/Vc))


def weights(shape):
    Wmin, Wmax = 0.25, 0.75
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

CTX_noise = 0.01
STR_noise = 0.001
STN_noise = 0.001
GPI_noise = 0.03
THL_noise = 0.001

radius = 70*1.42   







#stim = np.array([[2, 3], [1, 3], [1, 2], [0, 1], [0, 2], [0, 3]]*20)
cogstim=np.array([[0, 1]])
motstim=np.array([[3, 2]])
ascstim=np.vstack([cogstim, motstim])

model = nengo.Network('Action Selection')
model.config[nengo.Ensemble].neuron_type = nengo.Direct()#Sigmoid(tau_ref=0.001125)
with model:
    
    #Every node/structure has this property - rest, this needs to be added in every iteration of the loop
    #rest = nengo.Node([3, 0, -10, 40, 10])

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
    nengo.Connection(CTX_COG_out, STR_COG.input, transform=np.array([ 0.53677366, 0.56135171,   0.49578237,  0.49318303]), synapse = tau)
    nengo.Connection(CTX_MOT_out, STR_MOT.input, transform=weights(inps), synapse = tau)
    asctoasc = np.zeros((inps*inps, inps*inps))
    np.fill_diagonal(asctoasc, weights((inps*inps, inps*inps)))
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

    print(asctoasc)
    

    stim_cog = nengo.Node([-1, -1])
    stim_mot = nengo.Node([-1, -1])
    stim_asc = nengo.Node([-1, -1, -1, -1])


    rv = np.zeros(3000)
    rv[0:100] = -1
    rv[-100:] = -1
    stim_cog.output = PresentInput(inputs=fill(cogstim), presentation_time=0.5)
    stim_mot.output = PresentInput(inputs=fill(motstim), presentation_time=0.5)
    stim_asc.output = PresentInput(inputs=np.hstack([fill(cogstim), fill(motstim)]), presentation_time=0.5)

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

    stim_asc_r = nengo.Ensemble(pop*(inps*inps),inps*inps, radius=20)

    nengo.Connection(stim_asc, stim_asc_r, function=ascstim, synapse = None)
    nengo.Connection(stim_asc_r, CTX_ASC.input, synapse = None)
    nengo.Connection(stim_cog, CTX_COG.input, function=precogstim, synapse = None)
    nengo.Connection(stim_mot, CTX_MOT.input, function=premotstim, synapse = None)
    #nengo.Connection(stim_asc, CTX_ASC.input, function=ascstim, synapse = None)

#print(help(nengo.Node))






print sum(ens.n_neurons for ens in model.all_ensembles)