import numpy as np

def avg_magnetisation_per_spin(data):
    N = data.shape[1]

    return 1/N *np.mean(np.abs(np.sum(data, axis=1)))

def avg_magnetisation_squared(data):
    return np.mean(np.sum(data, axis=1)**2)

def magnetic_susceptibility_per_spin(data, T):
    N = data.shape[1]

    return 1/N * 1/T**2 * (avg_magnetisation_squared(data) - (N*avg_magnetisation_per_spin(data))**2)

    # return (avg_magnetisation_squared_per_spin(data) *  avg_magnetisation_per_spin(data)**2)  / T

def avg_energy(data, J):
    """
    Data has dimensions b x N where b is the number of batches and N is the
    number of spins.
    """
    energy = 0
    for configuration in data:
        energy += -configuration @ J @ configuration.T
    return energy / data.shape[0] /2

def avg_energy_squared(data, J):
    energy_squared = 0
    for configuration in data:
        energy_squared += (configuration @ J @ configuration)**2

    return energy_squared / data.shape[0] / 4

def heat_capacity(data, J, T, avg_energy):
    return (avg_energy_squared(data, J) - avg_energy**2) / T**2

# def avg_energy2(data, J):
#     """
#     Data has dimensions b x N where b is the number of batches and N is the
#     number of spins.
#     """
#     return -np.mean(data @ J @ data.T) * data.shape[1]
#
# def avg_energy_squared2(data, J):
#     return np.mean(data @ J @ data.T @ data @ J @ data.T) * data.shape[1]
#
# def heat_capacity2(data, J, T, avg_energy):
#     # return np.mean(np.mean(data @ J @ data.T @ data @ J @ data.T, axis=1) - np.mean(data @ J @ data.T, axis=1)) * data.shape[1] / T**2
#     return (avg_energy_squared2(data, J) - avg_energy**2) / T**2
#
# def aux_energy(conf, J):
#     sum = 0
#     for i in range(conf.size):
#         for j in range(conf.size):
#             sum += -conf[i] * J[i,j] * conf[j]
#
#     return sum
