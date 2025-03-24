import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Define constants
# size of lattice
N = 30
# number of steps
steps = 3000000
# temperature
temperature =  2
# constants
J_val = 1
kB_val = 1

# define probability coefficient
K_val = J_val / (kB_val * temperature)
probability = np.tanh(K_val)

# create dictionnary to store bonds
bonds = {}

# enforce observables
G = defaultdict(int)


# function to enforce periodic boundary conditions (pbc)
def pbc(x):
	return x % N

# get distance between two sites with pbc
def distance(a, b):
	# get distance
	dx = pbc(abs(a[0] - b[0]))
	dy = pbc(abs(a[1] - b[1]))
	# check that it is not shorter from other way
	dx = min(dx, N - dx)
	dy = min(dy, N - dy)

	# return euclidean distance as int
	return np.sqrt(dx**2 + dy**2)

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

# Worm algorithm
for _ in tqdm(range(steps), desc='Worms'):
	# pick random start site (define as tuple)
	x0 = (random.randint(0, N-1), random.randint(0, N-1))
	# define site as head
	head = x0

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
		# if bond already exists
		else:
				# remove bond
				bonds.pop(bond)
				head = neighbor

		# close worm if head reaches tail
		if head == x0:
			break

		# compute correlation
		r = distance(x0, head)
		G[r] += 1

	# normalize correlation function
	tot = sum(G.values())
	for r in G:
		G[r] /= tot

# get values to plot
r_val = sorted(G.keys())
G_val = [G[r] for r in r_val]

# plot results
plt.figure(figsize=(8, 6))
plt.plot(r_val, G_val, marker='o', linestyle='-', linewidth=2, markersize=5, label=f"T = {temperature}")
plt.xlabel("Distance $r$", fontsize=14)
plt.ylabel("Correlation $G(r)$", fontsize=14)
plt.title("Two-point Correlation Function from Worm Algorithm", fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
# plt.xscale('log'); plt.yscale('log')
plt.show()



