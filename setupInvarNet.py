import pyphi
import numpy as np
from utils import Experiment
import pickle
pyphi.config.PARTITION_TYPE = 'TRI'
pyphi.config.MEASURE = 'BLD'

# Weights matrix
network = pickle.load(open('iv_manet5.0_network.pkl','rb'))

nN = network.cm.shape[0]
elements = list(range(nN))

cstate = [0 for x in elements];

nameformat = 'iv_manet'

experiment = Experiment(nameformat, '5.0', network, cstate)
experiment.initialize()

print('success!')
