import sys
import numpy as np
import pickle

states_dir = sys.argv[1]
nstates = int(sys.argv[2])
lsize = int(sys.argv[3])
path_to_pickle = sys.argv[4]
temp = float(sys.argv[5])
spin_size = lsize**2

def get_spins(nstates, spin_size):
    new = np.loadtxt(states_dir + '/states{}.txt'.format(0), skiprows=1, delimiter=',', dtype=int)
    new = new.reshape(spin_size,)
    for i in range(1,nstates):
        old = np.loadtxt(states_dir + '/states{}.txt'.format(i), skiprows=1, delimiter=',', dtype=int)
        old = old.reshape(spin_size,)
        new = np.vstack((old,new))

    return new

spins = get_spins(nstates, spin_size)
filehandler = open(path_to_pickle + "/Ising_loukas_L{:n}_T={:.2f}.pkl".format(lsize, temp),"wb")
pickle.dump(spins, filehandler)
