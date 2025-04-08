
import numpy as np
import matplotlib.pyplot as plt
import math as m
from tqdm import tqdm
from scipy.optimize import curve_fit

# random matrix generation function
def random_matrix_generation(row, col):
	return np.random.choice([-1, 1], size=(row, col))

# Stack implementation
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
		return 'Stack is empty'

	def peek(self):
		# show first element of the stack
		if not self.is_empty():
			return self.stack[-1]

	def is_empty(self):
		# check if the stack is empty
		return len(self.stack) == 0

	def size(self):
		# returns the size of the stack
		return len(self.stack)

# calculate structure factor
def structure_factor(cluster, qx, N):

	# introduce the complex sum
	sum_exp = 0.0 + 0.0j

	# sum of exp(i * q * r)
	# only take into account the x component of the q vector
	for(x, y) in cluster:
		sum_exp += np.exp(1j * qx * x)

	# get size of cluster
	cluster_len = len(cluster)

	# check that cluster is not empty
	if (cluster_len == 0):
		# return zero to avoid division by zero
		return 0.0

	# compute structure factor
	S_q = (np.abs(sum_exp)**2) / (N * cluster_len)

	return S_q

# wolff algorithm for cluster spin flip
def wolff(model, temperature):
	# define constants
	J_val = 1
	kB_val = 1

	# define the probability coefficient
	K_val = J_val / (kB_val * temperature)
	probability = 1 - m.exp(-2 * K_val)

	# get lengths of the model (rows/cols)
	rows, cols = model.shape

	# randomly choose a lattice site
	row, col = np.random.randint(0, rows), np.random.randint(0, cols)
	initial_spin = model[row][col]

	# initiate stak and add lattice site
	in_line = Stack()
	in_line.push((row, col))

	# create visited set not to add same sites
	visited = set()
	visited.add((row, col))

	# wolff decision
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

	# get size of the cluster
	size = len(visited)

	# define wavevector (only on x)
	q_x = []
	for nx in range(cols):
		qx = 2 * np.pi * nx/ cols
		q_x.append(qx)

	# introduce structure factor as array
	S_qs = []

	# compute structure factor as array
	for qx in q_x:
		S_q = structure_factor(visited, qx, rows * cols)
		S_qs.append(S_q)

	return model, size, S_qs

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


# Compute <m^4>
def get_m4(model):
	N = model.size
	M = np.sum(model) / N

	m4 = M**4

	return m4

# Compute Binder Cumulant
def binder_cumulant(m2, m4):
	return 1 - m4 / (3 * m2**2)

# Apply Jacknife Analysis
def jacknife_binder(m2_vals, m4_vals):
	# get length of array
	n = len(m2_vals)
	jacknife_U = np.zeros(n)

	# Compute jacknife estimation by removing one element
	for i in range(n):
		m2_jack = np.delete(m2_vals, i).mean()
		m4_jack = np.delete(m4_vals, i).mean()
		jacknife_U[i] = binder_cumulant(m2_jack, m4_jack)

	# Get Jacknife mean
	mean_jack = jacknife_U.mean()

	# Get Jacknife error
	variance_jack = (n-1)/n * np.sum((jacknife_U - mean_jack)**2)
	error_jack = np.sqrt(variance_jack)

	return mean_jack, error_jack

# Apply Jacknife to Scalars
def jacknife_scalar(samples):
	n = len(samples)
	jack_vals = []

	for i in range(n):
		jack_sample = np.delete(samples, i)
		jack_vals.append(np.mean(jack_sample))

	jack_vals = np.array(jack_vals)
	mean_jack = np.mean(jack_vals)
	var_jack = (n - 1) / n * np.sum((jack_vals - mean_jack) ** 2)
	error_jack = np.sqrt(var_jack)

	return mean_jack, error_jack



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

def jacknife(samples):
	# get size of the samples matrix
	n_q, n_samples = samples.shape

	# introduce means and errors arrays
	means = np.zeros(n_q)
	errors = np.zeros(n_q)

	for i in range(n_q):
		# get interested sample
		q_samples = samples[i]
		jacknife_means = []

		for j in range(n_samples):
			reduced = np.delete(q_samples, j)
			jacknife_mean = np.mean(reduced)
			jacknife_means.append(jacknife_mean)

		jacknife_means = np.array(jacknife_means)

		mean_i = np.mean(jacknife_means)
		var_i = (n_samples - 1)/n_samples * np.sum((jacknife_means - mean_i)**2)
		error_i = np.sqrt(var_i)

		means[i] = mean_i
		errors[i] = error_i

	return means, errors


# running wolff algorithm multiple times
def calc(model, temperature, N):
	# initiate array to store magnetization
	magnetization = []
	S_q_samples = []
	for _ in tqdm(range(50000), desc='Wolff flipping'):
		model, _, S_qs = wolff(model, temperature)
		magnetization.append(get_magnetization(model))
		S_q_samples.append(S_qs)

	# find stabilisation index
	stabilisation_index = find_most_stable_segment(magnetization)

	S_q_samples = S_q_samples[stabilisation_index:]

	# keep only stabilized data
	S_q_samples = np.array(S_q_samples)

	# transpose to get (n_q, n_samples) matrix
	S_q_samples = S_q_samples.T

	# apply jacknife analysis to structure factor
	S, e_S = jacknife(S_q_samples)

	# apply jacknife analysis to magnetization
	Mag, e_Mag = jacknife_scalar(magnetization)

	return S, e_S, Mag, e_Mag

def main():
    N = 30
    #temperatures = [round(x, 2) for x in np.arange(1, 4, 0.15)]
    #temperatures = np.linspace(1.5, 2.2680, 10)
    temperatures = np.linspace(1.5, 2.22, 20)
    model = random_matrix_generation(N, N)

    # write results in a txt file
    with open('lower_Tc_structure_factor_data.txt', 'w') as f:
    	f.write('# T\tqx\tS_q\terror\tM\tM_error\n')

    	for T in temperatures:
    		model_T = model.copy()
    		S_vals, Serr_vals, Mag, e_Mag = calc(model_T, T, N)

    		q_x = [2 * np.pi * nx / N for nx in range(N)]

    		for q, S, err in zip(q_x, S_vals, Serr_vals):
    			f.write(f'{T:.2f} {q:.2f} {S:.6f} {err:.6f} {Mag:.6f} {e_Mag:.6f}\n')


    print("Results save in 'lower_Tc_structure_factor_data.txt'")

main()







