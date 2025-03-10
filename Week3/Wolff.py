import numpy as np
import matplotlib.pyplot as plt
import math as m
from tqdm import tqdm

# random matrix generation function
def random_matrix_generation(row, col):
	return np.random.choice([-1, 1], size=(row, col))

# Stack implentation
class Stack:
	def __init__(self):
		self.stack = []

	def push(self, item):
		# add element on top of the stack
		self.stack.append(item)

	def pop(self):
		# remove top element of the stack
		if not self.is_empty():
			return self.stack.pop()
		return "Stack is empty"

	def peek(self):
		# show first element of the stack
		if not self.is_empty():
			return self.stack[-1]

	def is_empty(self):
		# checks if the stack is empty
		return len(self.stack) == 0

	def size(self):
		# returns the size of the stack
		return len(self.stack)

# wolf algorithm for cluster spin flip
def wolff(model, temperature):
	# define constants
	J_val = 1
	kB_val = 1

	# define the probability coefficient
	K_val = J_val / (kB_val * temperature)
	probability = 1 - m.exp(- 2 * K_val)

	# get lengths of the model (rows/columns)
	rows, cols = model.shape

	# randomly choose a lattice site
	row, col = np.random.randint(0, rows), np.random.randint(0, cols)
	initial_spin = model[row][col]

	# initiate stack and add lattice site
	in_line = Stack()
	in_line.push((row, col))

	# create visited set not to add same sites
	visited = set()
	visited.add((row, col))

	while not in_line.is_empty():
		# get spin of interest
		r, c = in_line.pop()

		# find neighbors
		neighbors = [
			((r - 1) % rows, c),
			((r + 1) % rows, c),
			(r, (c - 1) % cols),
			(r, (c + 1) % cols)
		]

		# iterate over every neighbors
		for nr, nc in neighbors:
			if (
				# check if they have same spin
				model[nr, nc] == initial_spin
				# check if they are already visited
				and (nr, nc) not in visited
				# probability argument
				and np.random.rand() < probability
			):
				# if passes -> add in stack and visited spin history
				in_line.push((nr, nc))
				visited.add((nr, nc))

	# flip all spins in the cluster
	for r, c in visited:
		model[r][c] *= -1

	return model

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

	# get magnetization (spin sum / # particles)
	magnetization = spins_sum / n



	return abs(magnetization)


def calc(model, temperature):
    # initiate array to store magnetization
    magnetization = np.array([])
    evolution = np.array([])

    # repeat calculations multiple times
    # tqdm evolution bar
    for i in tqdm(range(50000), desc='Simulation'):
        # cluster flipping
        model = wolff(model, temperature)
        # store magnetization
        magnetization = np.append(magnetization, get_magnetization(model))
        evolution = np.append(evolution, i)

    stabilisation_index = find_most_stable_segment(magnetization)
    print("STABILIZATION INDEX = ", stabilisation_index)

    for i in range(len(magnetization) - 1):
        if (magnetization[i] < magnetization[stabilisation_index] and magnetization[i + 1] > magnetization[stabilisation_index]):
            correct_start_index = i
            break
        elif (magnetization[i] > magnetization[stabilisation_index] and magnetization[i + 1] < magnetization[stabilisation_index]):
            correct_start_index = i
            break

    correct_start_index = find_correct_start_index(magnetization)
    print(correct_start_index)

    """
    # plot the magnetization evolution
    plt.plot(evolution, magnetization)
    plt.xlabel("Iteration")
    plt.ylabel("Magnetization")
    plt.axhline(y=magnetization[stabilisation_index], color='r', linestyle='--', label='Stabilisation Point')
    plt.title(f"Ising Model Magnetization (T = {temperature})")
    plt.legend()
    plt.show()
    """

    return np.mean(magnetization[stabilisation_index:]), analysis(magnetization)


def find_correct_start_index(array):
    stabilisation_index = find_most_stable_segment(array)
    
    for i in range(len(array) - 1):
        if (array[i] - array[stabilisation_index]) * (array[i + 1] - array[stabilisation_index]) < 0:
            return i
    
    return None

def find_most_stable_segment(array, stable_window=9000):
    min_std = float('inf')
    best_index = 0
    
    for i in range(len(array) - stable_window):
        window_std = np.std(array[i:i + stable_window])
        if window_std < min_std:
            min_std = window_std
            best_index = i
    
    return best_index

def analysis(array):
    # remove first random uncorrect elements

    if find_correct_start_index(array) is None:
        array = array[find_most_stable_segment(array):]
    else:
        array = array[find_correct_start_index(array):]

    errors, blocks = binning(array)

    final_value = np.argmax(errors)

    final_error = errors[final_value]

    

    plt.axhline(y=errors[final_value], color='r', linestyle='--', label='Convergence value')


    """
    plt.plot(blocks, errors, marker='.')
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
    with open("wolff_magnetization_vs_temp_120x120.txt", "w") as f:
        f.write("Temperature\tMagnetization\tError\n")
        for T, M, E in zip(temperatures, magnetizations, errors):
            f.write(f"{T}\t{M}\t{E}\n")


def main():
	# generate model
	model = random_matrix_generation(120, 120)

	mag_vs_temp(model)
	

main()
