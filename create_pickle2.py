import sys
import numpy as np
import pickle

states_dir = sys.argv[1]
nstates = int(sys.argv[2])
lsize = int(sys.argv[3])
path_to_pickle = sys.argv[4]
temp = float(sys.argv[5])
spin_size = lsize**2

spins = np.loadtxt(states_dir + '/' + "1states0.txt", delimiter=',', skiprows=1).reshape(-1)
for i in range(nstates-1):
    spins2 = np.loadtxt(states_dir + '/' + "{:}states0.txt".format(i+2), delimiter=',', skiprows=1).reshape(-1)
    spins = np.column_stack((spins, spins2))

spins = spins.T

filehandler = open(path_to_pickle + "/Ising_loukas_L{:n}_T={:.2f}.pkl".format(lsize, temp),"wb")
pickle.dump(spins, filehandler)
