import os
import pickle
import numpy as np
import sys
# print("Python version: ", sys.version)

# for plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.cm as cm
import seaborn as sns
import ml_style as style
import matplotlib as mpl
# print("Matplolib version: ", mpl.__version__)

mpl.rcParams.update(style.style)

# for Boltzmann machines
from paysage import preprocess as pre
from paysage.layers import BernoulliLayer, GaussianLayer
from paysage.models import BoltzmannMachine
from paysage import batch
from paysage import fit
from paysage import optimizers
from paysage import samplers
from paysage import backends as be
from paysage import schedules
from paysage import penalties as pen

# fix random seed to ensure deterministic behavior
be.set_seed(137)

def unpack_data(path_to_data, data_name):
    """
    Get the data from a pickled file.

    Args:
        path_to_data (str)
        data_name (str)

    Returns:~
        numpy.ndarray
    """
    # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
    # pickle reads the file and returns the Python object (1D array, compressed bits)
    with open(os.path.join(path_to_data, data_name), 'rb') as infile:
        data = pickle.load(infile)
    # Decompress array and reshape for convenience
    # data = np.unpackbits(data).reshape(-1, 16).astype('int')
    data = data.reshape(-1, 64).astype('int')

    return data

def Load_Ising_Dataset():
    """
    Loads the Ising dataset.

    Args:
        None

    Returns:
        dict[numpy.ndarray]
    """
    L = 8 # linear system size
    # T = np.linspace(0.25,4.0,16) # temperatures
    T = np.arange(1, 4.01, 0.01) # temperatures
    T_c = 2.26 # critical temperature in the TD limit

    # path to data directory
    path_to_data = 'data'

    # LK
    data_dict = {}
    # print(T)
    for temp in T:
        print("Temperature: ", temp)
        file_name = "Ising_loukas_L{:}_T={:.2f}.pkl".format(L, temp)
        data = unpack_data(path_to_data, file_name)
        print("Shape of data for T = {:}:".format(temp),data.shape)
        np.random.shuffle(data)
        data_dict['%.2f' % temp] = data

    return data_dict

def ADAM_optimizer(initial, coefficient):
    """
    Convenience function to set up an ADAM optimizer.

    Args:
        initial (float): learning rate to start with
        coefficient (float): coefficient that determines the rate of
            learning rate decay (larger -> faster decay)

    Returns:
        ADAM

    """
    # define learning rate attenuation schedule
    # LK
    # learning_rate = schedules.PowerLawDecay(initial=initial, coefficient=coefficient)
    learning_rate = schedules.Constant(initial=initial)

    return optimizers.ADAM(stepsize=learning_rate)


def train_model(model, data, num_epochs, monte_carlo_steps):
    """
    Train a model.

    Args:
        model (BoltzmannMachine)
        data (Batch)
        num_epochs (int)
        monte_carlo_steps (int)

    Returns:
        None

    """
    is_deep = model.num_layers > 2
    model.initialize(data,method='glorot_normal')
    opt = ADAM_optimizer(initial,coefficient)
    if is_deep:
        print("layerwise pretraining")
        pretrainer=fit.LayerwisePretrain(model,data)
        pretrainer.train(opt, num_epochs, method=fit.pcd, mcsteps=monte_carlo_steps, init_method="glorot_normal")
        # reset the optimizer using a lower learning rate
        opt = ADAM_optimizer(initial/10.0, coefficient)
    print("use persistent contrastive divergence to fit the model")
    trainer=fit.SGD(model,data)
    # LK
    trainer.train(opt,num_epochs, method=fit.cd, mcsteps=monte_carlo_steps)
    # trainer.train(opt,num_epochs,method=fit.pcd,mcsteps=monte_carlo_steps)

def compute_reconstructions(model, data):
    """
    Computes reconstructions of the input data.
    Input v -> h -> v' (one pass up one pass down)

    Args:
        model: a model
        data: a tensor of shape (num_samples, num_visible_units)

    Returns:
        tensor of shape (num_samples, num_visible_units)

    """
    recons = model.compute_reconstructions(data).get_visible()
    return be.to_numpy_array(recons)

def compute_fantasy_particles(model,num_fantasy,num_steps,mean_field=True):
    """
    Draws samples from the model using Gibbs sampling Markov Chain Monte Carlo .
    Starts from randomly initialized points.

    Args:
        model: a model
        data: a tensor of shape (num_samples, num_visible_units)
        num_steps (int): the number of update steps
        mean_field (bool; optional): run a final mean field step to compute probabilities

    Returns:
        tensor of shape (num_samples, num_visible_units)

    """
    schedule = schedules.Linear(initial=1.0, delta=1 / (num_steps-1))
    fantasy = samplers.SequentialMC.generate_fantasy_state(model,
                                                           num_fantasy,
                                                           num_steps,
                                                           schedule=schedule,
                                                           beta_std=0.0,
                                                           beta_momentum=0.0)
    if mean_field:
        fantasy = model.mean_field_iteration(1, fantasy)
    fantasy_particles = fantasy.get_visible()
    return be.to_numpy_array(fantasy_particles)


def plot_image_grid(image_array, shape, vmin=0, vmax=1, cmap=cm.gray_r,
                    row_titles=None, filename=None):
    """
    Plot a grid of images.

    Args:
        image_array (numpy.ndarray)
        shape (tuple)
        vmin (optional; float)
        vmax (optional; float)
        cmap (optional; colormap)
        row_titles (optional; List[str])
        filename (optional; str)

    Returns:
        None

    """
    array = be.to_numpy_array(image_array)
    nrows, ncols = array.shape[:-1]
    f = plt.figure(figsize=(2*ncols, 2*nrows))
    grid = gs.GridSpec(nrows, ncols)
    axes = [[plt.subplot(grid[i,j]) for j in range(ncols)] for i in range(nrows)]
    for i in range(nrows):
        for j in range(ncols):
            sns.heatmap(np.reshape(array[i][j], shape),
                ax=axes[i][j], cmap=cmap, cbar=False, vmin=vmin, vmax=vmax)
            axes[i][j].set(yticks=[])
            axes[i][j].set(xticks=[])

    if row_titles is not None:
        for i in range(nrows):
            axes[i][0].set_ylabel(row_titles[i], fontsize=36)

    plt.tight_layout()
    plt.show(f)
    if filename is not None:
        f.savefig(filename)
    plt.close(f)

image_shape = (8,8) # 40x40=1600 spins in every configuration
num_to_plot = 5000 # of data points to plot

# parameters the user needs to choose
batch_size = 50 # batch size
num_epochs = 5 # training epochs
monte_carlo_steps = 20 # number of MC sampling steps
initial = 1E-2 # initial learning rate
coefficient = 1.0 # controls learning rate decay
num_fantasy_steps = 80 # MC steps when drawing fantasy particles
lmbda = 1E-6 # stength of the L1 penalty
num_hidden_units = [128] # hidden layer units

# load data
data =  Load_Ising_Dataset()

# preallocate data dicts
dbm_L1_reconstructions = {}
dbm_L1_fantasy = {}
true_examples = {}
dbm_models = {}

for phase in data:
    print('training in the T = {} phase'.format(phase))

    # set up an object to read minibatch of the data
    transform = pre.Transformation()
    batch_reader = batch.in_memory_batch(data[phase], batch_size, train_fraction=0.95, transform=transform)
    batch_reader.reset_generator(mode='train')

    ##### Bernoulli RBM
    dbm_L1 = BoltzmannMachine(
            [BernoulliLayer(batch_reader.ncols)] + \
            [BernoulliLayer(n) for n in num_hidden_units]
            )

    # add an L1 penalty to the weights
    for j_, conn in enumerate(dbm_L1.connections):
        conn.weights.add_penalty({'matrix': pen.l1_penalty(lmbda)})

    # train the model
    train_model(dbm_L1, batch_reader, num_epochs, monte_carlo_steps)

    # store model
    dbm_models[phase] = dbm_L1

    # reset the generator to the beginning of the validation set
    batch_reader.reset_generator(mode='validate')
    examples = batch_reader.get(mode='validate') # shape (batch_size, 1600)
    true_examples[phase] = examples[:num_to_plot]

    # compute reconstructions
    reconstructions = compute_reconstructions(dbm_L1, true_examples[phase])
    dbm_L1_reconstructions[phase] = reconstructions

    # compute fantasy particles
    fantasy_particles = compute_fantasy_particles(dbm_L1,
                                                  num_to_plot,
                                                  num_fantasy_steps,
                                                  mean_field=False)
    dbm_L1_fantasy[phase] = fantasy_particles

    # plot results and save fig
    # reconstruction_plot = plot_image_grid(
    #         np.array([
    #                 true_examples[phase],
    #                 dbm_L1_reconstructions[phase],
    #                 dbm_L1_fantasy[phase]
    #                 ]),
    #         image_shape, vmin=0, vmax=1,
    #         row_titles=["data", "reconst", "fantasy"],
    #         filename='DBM_Ising-'+phase+'.png')

# save data
save_file_name = './ising_loukas4.2(cd)_data-L=8.pkl'
pickle.dump([dbm_models, true_examples, dbm_L1_fantasy, dbm_L1_reconstructions,
            image_shape, num_to_plot, batch_size, num_epochs, monte_carlo_steps,
            initial, coefficient, num_fantasy_steps, lmbda,num_hidden_units,
            ], open(save_file_name, "wb"))
