import numpy as np
import pickle
from thermodynamic_methods import *
import time

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['lines.linewidth'] = 1.5
# mpl.rcParams['lines.marker'] = "o"
mpl.rcParams['axes.grid'] = True


# import matplotlib.gridspec as gs
# import matplotlib.cm as cm
# import seaborn as sns
# import ml_style as style
# import matplotlib as mpl
# mpl.rcParams.update(style.style)



### from rbm-train file
# This is the format of the pickle file
# pickle.dump([dbm_models, true_examples, dbm_L1_fantasy, dbm_L1_reconstructions,
            # image_shape, num_to_plot, batch_size, num_epochs, monte_carlo_steps,
            # initial, coefficient, num_fantasy_steps, lmbda,num_hidden_units,
            # ], open(save_file_name, "wb"))

def create_j(nvis, temp):
    """ Returns shape nxn. Specific J for 'typewriter' type of spin configuration. """
    l = int(np.sqrt(nvis))
    pos1,pos2, pos3, pos4 = 1, l-1, l*(l-1), nvis-1
    return (np.eye(nvis,nvis,k=pos1) + np.eye(nvis,nvis,k=-pos1) \
             + np.eye(nvis,nvis,k=pos2) + np.eye(nvis,nvis,k=-pos2) \
             + np.eye(nvis,nvis,k=pos3) + np.eye(nvis,nvis,k=-pos3) \
             + np.eye(nvis, nvis,k=pos4) + np.eye(nvis, nvis,k=-pos4)) #/ (temp)

def plot(x1, y1, x2, y2, t, title, saveTitle=False):
    plt.figure()
    plt.plot(t, y1, 'b.-')
    plt.plot(t, y2, 'r.-')
    plt.plot(t , x1, 'g,-')
    # plt.plot(t , x2, 'go-')
    plt.xlabel('Temperature (T)')
    plt.ylabel(title)
    plt.title('{:} against Temperature'.format(title))
    plt.legend(["Rbm Data (hidden modes = 16)", "Rbm Data (hidden modes = 64)", "Real Data"], loc=1, fontsize="small")
    if not saveTitle:
        plt.show()
    else:
        plt.savefig(title + "(join)")


def transformation_binary_to_ising(data):
    return 2*data - 1

def w_histogram(weights, fname=None):
    fig = plt.figure()

    ax = fig.add_subplot(311)
    ax.hist(np.ravel(weights), bins=100)
    ax.set_title("W")
    # ax = fig.add_subplot(312)
    # ax.hist(self.b, bins=100)
    # ax.set_title("visible bias")
    # ax = fig.add_subplot(313)
    # ax.hist(self.c, bins=100)
    # ax.set_title("hidden bias")

    if fname is not None:
        plt.savefig(fname, bbox_inches ="tight")
    else:
        plt.show()
        # plt.show(block=False)
        # plt.pause(3)
        # plt.close()


# file = "DBM_ising_training_data-L=40.pkl"
# temperatures = np.linspace(0.25,4.0,16) # temperatures

#file = "ising_loukas_data-L=4.pkl"
#temperatures = np.arange(1,4.1,0.1) # temperatures

# file = "ising_loukas2_data-L=4.pkl"
# temperatures = np.arange(1,4.01,0.01) # temperatures


temperatures = np.arange(1,4.01,0.01) # temperatures
def get_observables(file):

    # Load from file
    with open(file, 'rb') as file:
        pickle_model = pickle.load(file)


    m1list, m2list = [], []
    x1list, x2list = [], []
    e1list, e2list = [], []
    c1list, c2list = [], []
    i = 0
    for temp in temperatures:
        rbm = pickle_model[0].get('{:.2f}'.format(temp))
        # get the weights
        weights = rbm._connected_weights(0)[0]
        # w_histogram(weights)

        # print("Temperature: ", temp, float(temp))
        # if i == 120:
        #     w_histogram(weights)
        # i += 1

        real_data = pickle_model[1].get('{:.2f}'.format(temp))
        rbm_data = pickle_model[2].get('{:.2f}'.format(temp))
        real_data = transformation_binary_to_ising(real_data)
        rbm_data = transformation_binary_to_ising(rbm_data)

        N, nvis = real_data.shape
        J = create_j(nvis, temp)

        m1, m2 = avg_magnetisation_per_spin(real_data), avg_magnetisation_per_spin(rbm_data)
        x1, x2 = magnetic_susceptibility_per_spin(real_data, temp), magnetic_susceptibility_per_spin(rbm_data, temp)
        m1list.append(m1)
        m2list.append(m2)
        x1list.append(x1)
        x2list.append(x2)

        E1, E2 = avg_energy(real_data, J), avg_energy(rbm_data, J)
        C1, C2 = heat_capacity(real_data, J, temp, E1) , heat_capacity(rbm_data, J, temp, E2)
        e1list.append(E1 / nvis)
        e2list.append(E2 / nvis)
        c1list.append(C1 / nvis)
        c2list.append(C2 / nvis)

    return m1list, m2list, x1list, x2list, e1list, e2list, c1list, c2list

t0 = time.time()

run1 = get_observables("ising_loukas4.0(cd)_data-L=8.pkl")
run2 = get_observables("ising_loukas4(cd)_data-L=8.pkl")

t1 = time.time()
print("Took approximately {:.0f} minute(s).".format((t1-t0) / 60))

plot(run1[0], run1[1], run2[0], run2[1], temperatures, "Magnetisation", saveTitle=True)
plot(run1[2], run1[3], run2[2], run2[3], temperatures, "Susceptibility", saveTitle=True)
plot(run1[4], run1[5], run2[4], run2[5], temperatures, "Energy", saveTitle=True)
plot(run1[6], run1[7], run2[6], run2[7], temperatures, "Capacity", saveTitle=True)
