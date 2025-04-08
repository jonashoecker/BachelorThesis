import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# exponential fit
def exponential_fit(r, A, xi, mag2):
	return A * np.exp(-r / xi) + mag2

# read data
def read_data(filename):
	data = {}
	with open(filename, "r") as f:
		for line in f:
			if line.startswith('#'):
				continue
			T, r, Gr = map(float, line.strip().split())
			data.setdefault(T, []).append((r, Gr))

	return data

# extract xi as function of T
def extract_xi(data):
	T_vals = []
	xi_vals = []

	for T in sorted(data.keys()):
		values = data[T][1:-1] # skip first and last point
		r, Gr = zip(*values)
		# normalize
		Gr = np.array(Gr)
		Gr /= np.sum(Gr)

		popt, _ = curve_fit(exponential_fit, r, Gr, p0=[1.0, 1.0, 0.01], maxfev=50000)
		xi = popt[1]
		T_vals.append(T)
		xi_vals.append(xi)

	return np.array(T_vals), np.array(xi_vals)

def main():
	filename = 'correlation_results_all_temperatures.txt'
	data = read_data(filename)
	T_vals, xi_vals = extract_xi(data)

	T_c = 2.2680  # critical temperature

	# plot correlation length
	plt.figure(figsize=(8, 6))
	plt.plot(T_vals, xi_vals, 'o-', color='#f1680e', label='Correlation length $\\xi$', markersize=6, linewidth=2)
	plt.axvline(x=T_c, color='#0ec8f1', linestyle='--', linewidth=2, label='$T_c = 2.2680$')

	plt.xlabel("Temperature $T$", fontsize=14)
	plt.ylabel("Correlation length $\\xi$", fontsize=14)
	plt.title('Worm algorithm: Correlation length', fontsize=16)
	plt.grid(True, linestyle='--', alpha=0.6)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.xlim(min(T_vals), max(T_vals))
	plt.legend(fontsize=12)
	plt.tight_layout()
	plt.show()

main()


