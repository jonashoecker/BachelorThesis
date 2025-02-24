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

	# select random row/column
	row = np.random.randint(0, rows)
	col = np.random.randint(0, cols)

	# get energy before flip
	energy_before = get_energy(model)

	# generate new model with spin flipped
	new_model = np.copy(model)
	new_model[row][col] *= -1

	# get energy after flip
	energy_after = get_energy(new_model)

	# check if model switches
	# if weight(after) > weight(before) -> always switch
	if get_weigth(new_model, temperature) > get_weigth(model, temperature):
		model = new_model
	# if weight(after) > weight(before) -> apply exp(- beta * dE) probability
	else:
		# random variable from 0 to 1
		r = np.random.rand()

		# calculate weight difference
		difference = energy_after - energy_before

		# define probability
		probability = m.exp(- beta * difference)

		# if r < probability -> accept flip (otherwise not)
		if (r < probability):
			model = new_model

	return model

# simulation for fixed temperature returning average magnetization
def calc(model, temperature):
	# initiate array to store magnetizations
	magnetization = np.array([])

	# repeat calculations multiple times
	# for i in range(10):
	# tqdm evolution bar
	for i in tqdm(range(10), desc = 'Simulation') : 
		# attempt multiple flips 
		for j in range(5000):
			model = spin_flip(model, temperature)
		# store magnetization
		magnetization = np.append(magnetization, get_magnetization(model))

	return np.mean(magnetization)

# color map representings spins configuration
def spinmap(model, temperature):

	# call simulation
	calc(model, temperature)

	# Define color map :  blue -1, red 1
	cmap = mcolors.ListedColormap(['blue', 'red'])

	# Display
	plt.imshow(model, cmap=cmap, origin='upper')

	# Create patches for legend
	patch_neg = mpatches.Patch(color='blue', label='Spin -1')
	patch_pos = mpatches.Patch(color='red', label='Spin 1')

	# Add a custom legend that includes the temperature in its title
	plt.legend(handles=[patch_neg, patch_pos], title=f"Temperature: {temperature}", loc='upper right', frameon=True)

	# Remove axis ticks
	plt.xticks([])
	plt.yticks([])

	plt.show()

# values of magnetizations vs. temperature
def mag_vs_temp(model):
    # Arrays to store temperature and magnetization values
    temperatures = np.linspace(1, 5, 20)
    temperatures = np.append(temperatures, 10)
    temperatures = np.append(temperatures, 20)
    magnetizations = [calc(model, T) for T in temperatures]
	# Write results to a text file
    with open("magnetization_vs_temp_12x12.txt", "w") as f:
        f.write("Temperature\tMagnetization\n")
        for T, M in zip(temperatures, magnetizations):
            f.write(f"{T}\t{M}\n")

def main():
	# generate model
	model = random_matrix_generation(30, 30)

	spinmap(model, 4)
	#mag_vs_temp(model)

main()


