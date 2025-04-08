import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Define constants
# size of lattice
N = 40
# number of steps
steps = 10000
# temperature
temperatures = np.linspace(2.0, 3.0, 60)
# energy constants
J_val = 1
kB_val = 1

# function to enforce periodic boundary conditions (pbc)
def pbc(x):
	return x % N

# get distance between two sites with pbc
def distance(a, b):
	# get distance
	dx = pbc(abs(a[0] - b[0]))
	# check that it is not shorter from other way
	dx = min(dx, N - dx)

	# return along x-axis
	return dx

# get neighbors
def neighbors(site):
	x, y = site
	return [((x, y), (pbc(x+1), y)),
			((x, y), (pbc(x-1), y)),
			((x, y), (x, pbc(y+1))),
			((x, y), (x, pbc(y-1)))]

# order tuples not to double them
def order(a, b):
	return (b, a) if a > b else (a, b)

def worm(T, corr_steps):
	# Worm algorithm
	# define probability coefficient
	K_val = J_val / (kB_val * T)
	probability = np.tanh(K_val)

	# create dictionnary to store bonds
	bonds = {}

	# enforce observables
	G = defaultdict(int)

	def single_worm(update_G=False):
		nonlocal bonds, G
		# pick random start site (define as tuple)
		x0 = (random.randint(0, N-1), random.randint(0, N-1))
		# define site as head
		head = x0
		# define moved boolean variable
		moved = False

		# iterate while head hasn't joined tail
		while True:
			# choose random neighbor
			x, y = head
			# choose a random direction
			direct = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
			# get neighbor with the direction
			neighbor = (pbc(x + direct[0]), pbc(y + direct[1]))

			# define as bond
			bond = order(head, neighbor)
			# check if bond is already present in configuration
			current = bonds.get(bond, 0)

			# no already existing bond
			if current == 0:
				if random.random() < probability:
					# apply bond to existing bonds
					bonds[bond] = 1
					head = neighbor
					moved = True
			# if bond already exists
			else:
				# remove bond
				bonds.pop(bond)
				head = neighbor
				moved = False

			# close worm if head reaches tail
			if moved and head == x0:
				break

			if update_G and moved:
				r = distance(x0, head)
				G[r] += 1

	# First step: Autocorrelation
	for _ in tqdm(range(corr_steps), desc='Auto-Correlating', leave=False):
		single_worm(update_G=False)

	# Second step: Measurements
	for _ in tqdm(range(steps), desc=f'Worms T={T:.2f}', leave=False):
		single_worm(update_G=True)

	# normalize correlation function
	tot = sum(G.values())
	for r in G:
		G[r] /= tot

	return G

# run for multiple T and save to one file
output_file = "correlation_results_all_temperatures.txt"

with open(output_file, "w") as f:
	f.write("# T\t r\t G(r)\n")
	for T in temperatures:
		# define auto-correlation steps
		corr_steps = 20000 if abs(T - 2.268) < 0.3 else 5000
		# run worm algorithm
		G = worm(T, corr_steps)
		for r in sorted(G.keys()):
			f.write(f"{T:.2f}\t{r:.6f}\t{G[r]:.6e}\n")

print(f"results saved in '{output_file}'")



