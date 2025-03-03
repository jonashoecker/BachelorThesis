"""
        Bachelor Thesis : Jonas Hoecker

        Ising Model Implementation


"""

import numpy as np
import math as m
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from tqdm import tqdm

# random matrix generation function
def random_matrix_generation(row, col):
    return np.random.choice([-1, 1], size = (row, col))

# get energy of model
def get_energy(model):
    # coupling constant (fixed to 1)
    Jcoupling = 1

    # initiate energy variable
    energy = 0

    # get lengths of the model (rows/columns)
    rows, cols = model.shape

    # iterate over every particle
    for i in range(rows):
        for j in range(cols):
            # spin value of the particle
            s = model[i][j]

            # get neighbors
            neighbors = [
                model[(i + 1) % rows][j], #below
                model[(i - 1) % rows][j], #above
                model[i][(j + 1) % cols], #right
                model[i][(j - 1) % cols], #left
            ]

            # add to total energy (- Jcoupling * s_i * s_j)
            energy += - Jcoupling * s * sum(neighbors)

    # divide energy by two (each bound counted twice)
    energy /= 2

    return energy

# get magnetization of model
def get_magnetization(model):
    # get lengths of the model (rows/columns)
    rows, cols = model.shape

    # get total number of particle in the system
    n = rows * cols

    # initiate the sum of spins variable
    spins_sum = 0

    # iterate over every particle
    for i in range(rows):
        for j in range(cols):
            # spin value of the particle
            s = model[i][j]

            # add spin to total sum
            spins_sum += s

    # get magnetization (spin sum / nb of particles)
    magnetization = spins_sum / n

    return abs(magnetization)

# get weigth of model
def get_weigth(model, temperature):
    # define energy of model
    energy = get_energy(model)

    # fix boltzmann constant to 1
    kb = 1

    # define beta (1/kb * 1/T)
    beta = 1 / (kb * temperature)

    # define weight
    weight = m.exp(- beta * energy)

    return weight

# spin-flip metropolis algorithm
def spin_flip(model, temperature):
    # fix boltzmann constant to 1
    kb = 1

    # define beta (1/kb * 1/T)
    beta = 1 / (kb * temperature)

    # get lengths of the model (rows/columns)
    rows, cols = model.shape

    # Pick a random spin
    row, col = np.random.randint(0, rows), np.random.randint(0, cols)
    s = model[row, col]

    # Compute local energy change ΔE
    neighbors = (
        model[(row + 1) % rows, col] +
        model[(row - 1) % rows, col] +
        model[row, (col + 1) % cols] +
        model[row, (col - 1) % cols]
    )
    # Energy change if flipped
    dE = 2 * s * neighbors

    # Metropolis criterion
    if dE < 0 or np.random.rand() < np.exp(-beta * dE):
        model[row, col] *= -1  # Flip spin
    return model

# simulation for fixed temperature returning average magnetization
def calc(model, temperature):
    # initiate array to store magnetizations
    magnetization = np.array([])
    evolution = np.array([])

    # repeat calculations multiple times
    # tqdm evolution bar
    for i in tqdm(range(100000), desc = 'Simulation') : 
        # attempt multiple flips 
        for j in range(5):
            model = spin_flip(model, temperature)
        # store magnetization
        magnetization = np.append(magnetization, get_magnetization(model))
        evolution = np.append(evolution, i)

    stabilisation_index = find_most_stable_segment(magnetization)
    
    for i in range(len(magnetization)-1):
        if (magnetization[i] < magnetization[stabilisation_index]) and (magnetization[i+1] > magnetization[stabilisation_index]):
            correct_start_index = i
            break
        elif (magnetization[i] > magnetization[stabilisation_index]) and (magnetization[i+1] < magnetization[stabilisation_index]):
            correct_start_index = i
            break

    correct_start_index = find_correct_start_index(magnetization)
    print(correct_start_index)

    """
    # Write results to a text file
    with open("Magnetization_evolution_T={temperature}.txt", "w") as f:
        f.write("Evolution\tMagnetization\n")
        for e, M in zip(evolution, magnetization):
            f.write(f"{e}\t{M}\n")
    """

    error_value = analysis(magnetization)
    print(error_value)

    return magnetization[stabilisation_index], error_value

    """
    # Plot the magnetization evolution
    plt.plot(evolution, magnetization)
    plt.xlabel("Iteration")
    plt.ylabel("Magnetization")
    plt.axhline(y=magnetization[stabilisation_index], color='r', linestyle='--', label="Stabilization Point")
    plt.title(f"Ising Model Magnetization (T = {temperature})")
    plt.show()
    """

def find_correct_start_index(array):
    stabilisation_index = find_most_stable_segment(array)
    
    for i in range(len(array) - 1):
        if (array[i] - array[stabilisation_index]) * (array[i + 1] - array[stabilisation_index]) < 0:
            return i
    
    return None

def find_most_stable_segment(array, stable_window=7000):
    min_std = float('inf')
    best_index = 0
    
    for i in range(len(array) - stable_window):
        window_std = np.std(array[i:i + stable_window])
        if window_std < min_std:
            min_std = window_std
            best_index = i
    
    return best_index

def sqrt_fct(x, a, b):
    return a * np.sqrt(x) + b

def analysis(array):

    # remove first random uncorrect elements
    array = array[find_correct_start_index(array):]

    errors, blocks = binning(array)

    final_value = np.argmax(errors)

    final_error = errors[final_value]

    """

    plt.axhline(y=error[final_value], color='r', linestyle='--', label='Convergence value')


    plt.plot(blocks, error, marker='.')
    plt.xscale("log")
    plt.xlabel('Block size')
    plt.ylabel('Standard deviation')
    plt.title('Blocking analysis')
    plt.show()
    """

    return final_error

def binning(array):
    # ensure to be numpy array
    array = np.array(array)

    # define array the store # of blocks
    length = []

    # define list of standard deviation
    standard_dev = [np.std(array)]

    # add first length of array
    length.append(len(array))

    # loop until length of array is 1
    while len(array) > 1:
        new_array = []

        # iterate over pair of elements
        for i in range(0, len(array) - 1, 2):
            mean_inter = np.mean([array[i], array[i+1]])
            new_array.append(mean_inter)

        # handle odd-length arrays (keep last element)
        if len(array) % 2 == 1:
            new_array.append(array[-1])

        length.append(len(array))

        array = np.array(new_array)
        standard_dev.append(np.std(array))

    return np.array(standard_dev), np.array(length)

# values of magnetizations vs. temperature
def mag_vs_temp(model):
    # Arrays to store temperature and magnetization values
    temperatures = np.linspace(1, 5, 20)
    temperatures = np.append(temperatures, [10, 20])
    magnetizations = []
    errors = []
    # Loop over temperatures
    for T in temperatures:
        mag, err = calc(model, T)  # Compute magnetization
        magnetizations.append(mag)
        errors.append(err)
    # Write results to a text file
    with open("magnetization_vs_temp_40x40.txt", "w") as f:
        f.write("Temperature\tMagnetization\tError\n")
        for T, M, E in zip(temperatures, magnetizations, errors):
            f.write(f"{T}\t{M}\t{E}\n")


def main():
    # generate model
    model = random_matrix_generation(40, 40)

    mag_vs_temp(model)

main()
    



