import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# define lorentzian forme of structure factor for curve_fit
def lorentzian_structure_factor(q, xi):
	return 1 / (1 + q**2 * xi**2)

# read data
def read_data(filename):
	data = {}
	with open(filename, "r") as f:
		for line in f:
			if line.startswith("#"):
				continue
			T, q, S, err = map(float, line.strip().split())
			data.setdefault(T, []).append((q, S, err))

	# sort q for each temperature
	for T in data:
		data[T].sort(key=lambda x: x[0])

	return data

def analysis(data):
	# introduce variables
	T_array = []
	xi_array = [] 
	xi_error_array = []

	for i, (T, values) in enumerate(sorted(data.items())):
		q_vals, S_vals, errors = zip(*values)
		# list to numpy arrays
		q_vals = np.array(q_vals)
		S_vals = np.array(S_vals)
		errors = np.array(errors)

		#normalization (propagation error mode)
		S0 = S_vals[0]
		e0 = errors[0]
		S_vals /= S0
		errors = S_vals * np.sqrt((errors / S_vals)**2 + (e0 / S0)**2)

		# take into account only small q's for fit
		q_fit = q_vals[:6]
		S_fit = S_vals[:6]
		# scipy fitting
		popt, pcov = curve_fit(lorentzian_structure_factor, q_fit, S_fit)
		# extract correlation length
		xi = popt[0]
		xi_error = np.sqrt(pcov[0, 0])

		# append results
		T_array.append(T)
		xi_array.append(xi)
		xi_error_array.append(xi_error)

	return T_array, xi_array, xi_error_array

def plot(T_val, xi_val, errors):
    plt.errorbar(
        T_val, xi_val, yerr=errors,
        fmt='o', capsize=4, capthick=1.5,
        markersize=6, markerfacecolor='white', markeredgewidth=1.5,
        color='royalblue', ecolor='black', linewidth=1.5
    )
    plt.xlabel("Temperature $T$", fontsize=14)
    plt.ylabel("Correlation length $\\xi$", fontsize=14)
    plt.title("Correlation Length vs Temperature", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


def main():

	data = read_data("long_structure_factor_data.txt")
	T_val, xi_val, errors = analysis(data)
	plot(T_val, xi_val, errors)

main()





