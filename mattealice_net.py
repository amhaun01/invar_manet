import numpy as np
import pyphi
from pyphi.convert import nodes2indices as n2i
from datetime import datetime
import pickle
import scipy.io
import random
from pyphi import Direction
cause, effect = Direction.CAUSE, Direction.EFFECT


pyphi.config.PARTITION_TYPE = 'TRI'
pyphi.config.MEASURE = 'BLD'
#pyphi.config.TIE_BREAKING = ''
#pyphi.config.PHI_SHORTCUT = True
#pyphi.config.USE_PROD = False

LP = '/Users/amhaun01/Documents/localOutput/pyphispace5/'

# Weights matrix

sg = 1 # grid self-connection
gc = 0.25 # grid lateral connection
gc2 = gc/2
bc1 = 0.1 # grid feedback from detectors

ic1 = 2.5 # detector excitatory driver
nc = -4.3 # detector inhibition
bc2 = 0.15 # detector feedback from invariant
scD = 0.5 # detector/invariant default self-connection

ic2 = 3 # invariant excitatory driver
scI = .15

ei = 1 # External inputs
all_weights = np.array([
                        [sg, gc, 0, 0, 0, 0, 0, 0,   gc2, 0, 0, 0, 0, 0, 0,     bc1, 0, 0, 0,    0,     ei, 0, 0, 0, 0, 0, 0, 0], # A
                        [gc, sg, gc, 0, 0, 0, 0, 0,  gc2, gc2, 0, 0, 0, 0, 0,   bc1, 0, 0, 0,    0,     0, ei, 0, 0, 0, 0, 0, 0], # B
                        [0, gc, sg, gc, 0, 0, 0, 0,  0, gc2, gc2, 0, 0, 0, 0,   0, bc1, 0, 0,    0,     0, 0, ei, 0, 0, 0, 0, 0], # C
                        [0, 0, gc, sg, gc, 0, 0, 0,  0, 0, gc2, gc2, 0, 0, 0,   0, bc1, 0, 0,    0,     0, 0, 0, ei, 0, 0, 0, 0], # D
                        [0, 0, 0, gc, sg, gc, 0, 0,  0, 0, 0, gc2, gc2, 0, 0,   0, 0, bc1, 0,    0,     0, 0, 0, 0, ei, 0, 0, 0], # E
                        [0, 0, 0, 0, gc, sg, gc, 0,  0, 0, 0, 0, gc2, gc2, 0,   0, 0, bc1, 0,    0,     0, 0, 0, 0, 0, ei, 0, 0], # F
                        [0, 0, 0, 0, 0, gc, sg, gc,  0, 0, 0, 0, 0, gc2, gc2,   0, 0, 0, bc1,    0,     0, 0, 0, 0, 0, 0, ei, 0], # G
                        [0, 0, 0, 0, 0, 0, gc, sg,   0, 0, 0, 0, 0, 0, gc2,     0, 0, 0, bc1,    0,     0, 0, 0, 0, 0, 0, 0, ei], # H
                        [gc, gc, 0, 0, 0, 0, 0, 0,   sg, gc, 0, 0, 0, 0, 0,     0, 0, 0, 0,      0,     ei, ei, 0, 0, 0, 0, 0, 0], # I
                        [0, gc, gc, 0, 0, 0, 0, 0,   gc, sg, gc, 0, 0, 0, 0,    0, 0, 0, 0,      0,     0, ei, ei, 0, 0, 0, 0, 0], # J
                        [0, 0, gc, gc, 0, 0, 0, 0,   0, gc, sg, gc, 0, 0, 0,    0, 0, 0, 0,      0,     0, 0, ei, ei, 0, 0, 0, 0], # K
                        [0, 0, 0, gc, gc, 0, 0, 0,   0, 0, gc, sg, gc, 0, 0,    0, 0, 0, 0,      0,     0, 0, 0, ei, ei, 0, 0, 0], # L
                        [0, 0, 0, 0, gc, gc, 0, 0,   0, 0, 0, gc, sg, gc, 0,    0, 0, 0, 0,      0,     0, 0, 0, 0, ei, ei, 0, 0], # M
                        [0, 0, 0, 0, 0, gc, gc, 0,   0, 0, 0, 0, gc, sg, gc,    0, 0, 0, 0,      0,     0, 0, 0, 0, 0, ei, ei, 0], # N
                        [0, 0, 0, 0, 0, 0, gc, gc,   0, 0, 0, 0, 0, gc, sg,     0, 0, 0, 0,      0,     0, 0, 0, 0, 0, 0, ei, ei], # O
                        [ic1, ic1, 0, 0, 0, 0, 0, 0,  nc, nc, 0, 0, 0, 0, 0,    scD, 0, 0, 0,     bc2,    0, 0, 0, 0, 0, 0, 0, 0], # P
                        [0, 0, ic1, ic1, 0, 0, 0, 0,  0, nc, nc, nc, 0, 0, 0,   0, scD, 0, 0,     bc2,    0, 0, 0, 0, 0, 0, 0, 0], # Q
                        [0, 0, 0, 0, ic1, ic1, 0, 0,  0, 0, 0, nc, nc, nc, 0,   0, 0, scD, 0,     bc2,    0, 0, 0, 0, 0, 0, 0, 0], # R
                        [0, 0, 0, 0, 0, 0, ic1, ic1,  0, 0, 0, 0, 0, nc, nc,    0, 0, 0, scD,     bc2,    0, 0, 0, 0, 0, 0, 0, 0], # S
                        [0, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0,    ic2, ic2, ic2 ,ic2,  scI,    0, 0, 0, 0, 0, 0, 0, 0], # T
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # a
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # b
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # c
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # d
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # e
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # f
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # g
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # h
                    ])  #A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R,  S, T, a, b, c, d, e, f, g, h
subsystem_labels = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T')


#incns = [4,5,11,12,13,17,19];turnons = [4,17,19]
#incns = [15,16,17,18,19];turnons = [17,19]
incns = [4,5,11,12,13,15,16,17,18,19];turnons = [4,17,19]
incns = [x for x in range(0,20)]

weights = []
#sv = 0
for x in incns:
    #weights.append(all_weights[x][range(sv,20)])
    weights.append([all_weights[x][y] for y in incns])
weights = np.array(weights)
# Gate function: nodes 8 to 14 has threshold => 2, all the rest >= 1
                      #A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, a, b, c, d, e, f, g, h
threshold = 1/4
offsets = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2])
offsets = np.array([offsets[x] for x in incns])
ntypes = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
ntypes = np.array([ntypes[x] for x in incns])

nN = len(offsets)

pset = pyphi.utils.powerset(np.arange(nN))

newtpm = np.zeros([2**nN,nN])
#indslist = [[] for x in range(0,len(pset))]
for inds in pset:
    istate = np.zeros(nN)
    for y in range(0,len(inds)):
        istate[inds[y]] = 1

    sw = np.zeros(nN,dtype='f')
    swN = np.zeros(nN)
    for z in range(0,len(weights)):
        #this should make the inputs -1/1 for the grid nodes, and 0/1 for the higher levels
        inpt = (2/(1+ntypes[z]))*istate - (1-ntypes[z])
        if offsets[z]==0:
            inpt = (inpt + ntypes)/(1 + ntypes)

        sw[z] = sum(inpt*weights[z])

        #swN[z] = (1-(sw[z]/sum(weights[z]))/10)/(1 + np.exp(-(sw[z] - offsets[z])/threshold))
        if ntypes[z]==0:
            swN[z] = 1/(1 + np.exp(-(sw[z] - offsets[z])/threshold))
        if ntypes[z]==1:
            doff = sw[z]/sum(weights[z]) - ic2/sum(weights[z])
            swN[z] = min(0.99,1/((1+doff)**8 + np.exp(-(sw[z] - offsets[z])/threshold)))



    #print(inpt,sw,swN)
    V = 0;
    for v in range(0,nN):
        V = V + istate[v]*2**v
    newtpm[int(V)] = tuple(swN)
#    indslist[int(V)] = inds

#tpm = tuple(newtpm)


cm = np.zeros(shape=weights.shape)
for x in range(0,cm.shape[0]):
    for y in range(0,cm.shape[1]):
        cm[x,y] = int(abs(weights[y,x])>0)

network = pyphi.Network(tuple(newtpm),cm,[subsystem_labels[x] for x in incns])

nameformat = 'iv_manet5.0'
fname = LP + '/pklout/' + nameformat + '_network.pkl'
outputf = open(fname,'wb')
pickle.dump(network,outputf)

"""
#FP = '/Users/andrew/Dropbox (Personal)/phipy/pyphispace5/manet/'
#fname = FP + nameformat + '_weightmat.mat'
#scipy.io.savemat(fname, mdict={'weights':all_weights})

cstate = [0 for x in range(0,nN)]
subsystem_off = pyphi.Subsystem(network,cstate,range(network.size))

cstate = [0 for x in range(0,nN)]
for x in turnons:
    for y in range(0,nN):
        if x==incns[y]:
            cstate[y] = 1
subsystem_on = pyphi.Subsystem(network,cstate,range(network.size))


C_off = subsystem_off.concept((7,))
print('')
C_on = subsystem_on.concept((7,))
print('')

C_off = subsystem_off.concept((9,))
print('')
C_on = subsystem_on.concept((9,))
print('')

C_off = subsystem_off.concept((0,7))
print('')
C_on = subsystem_on.concept((0,7))
print('')

C_off = subsystem_off.concept((7,9))
print('')
C_on = subsystem_on.concept((7,9))
print('')

C_off = subsystem_off.concept((3,7,9))
print('')
C_on = subsystem_on.concept((3,7,9))
print('')

C_off = subsystem_off.concept((0,7,9))
print('')
C_on = subsystem_on.concept((0,7,9))
print('')
#print(C_off)
#print(C_on)
mips = []
#mips.append(subsystem_off.find_mip(cause,(5,),(2,),0))
#mips.append(subsystem_off.find_mip(cause,(5,),(2,3),0))
#mips.append(subsystem_off.find_mip(cause,(5,),(2,3,4),0))
#mips.append(subsystem_off.find_mip(cause,(5,),(0,1),0))

#mips.append(subsystem_on.find_mip(cause,(4,),(0,),0))
#mips.append(subsystem_on.find_mip(cause,(4,),(0,1),0))
#mips.append(subsystem_on.find_mip(cause,(4,),(0,1,2),0))
#mips.append(subsystem_on.find_mip(cause,(4,),(0,1,2,3),0))
"""
