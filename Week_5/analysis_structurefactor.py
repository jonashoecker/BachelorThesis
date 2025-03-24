import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# define lorentzian form of structure factor for curve_fit
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
	# define markers style
	markers = ['o', 's', 'D', '^']
	point_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
	fit_colors   = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896']

	plt.figure(figsize=(10, 6))

	# iterate over every temperature values
	for i, (T, values) in enumerate(sorted(data.items())):
		q_vals, S_vals, errors = zip(*values)
		# list to numpy arrays
		q_vals = np.array(q_vals)
		S_vals = np.array(S_vals)
		errors = np.array(errors)

		# normalization (propagation error mode)
		S0 = S_vals[0]
		e0 = errors[0]
		S_vals /= S0
		errors = S_vals * np.sqrt((errors / S_vals)**2 + (e0 / S0)**2)

		# take into account only small q's for fit
		q_fit = q_vals[:6]
		S_fit = S_vals[:6]
		# scipy fitting
		popt, _ = curve_fit(lorentzian_structure_factor, q_fit, S_fit)
		# extract correlation length
		xi = popt[0]

		# print correlation length
		print(f"T = {T:.2f} → ξ ≈ {xi:.3f}")

		# plot calculations with errorbar
		plt.errorbar(q_vals, S_vals, yerr=errors, fmt=markers[i % len(markers)],
             color=point_colors[i % len(point_colors)], capsize=3, label=f'T = {T:.2f}')

		# plot fit
		q_model = np.linspace(0, max(q_vals), 200)
		S_model = lorentzian_structure_factor(q_model, xi)
		plt.plot(q_model, S_model, linestyle='--', color=fit_colors[i % len(fit_colors)], label=fr'Fit T = {T:.2f}')

	plt.xlabel(r'$q_x$')
	plt.ylabel(r'$\langle S(q_x) \rangle$')
	plt.title('Structure factor with lorentzian fitting')
	plt.legend()
	plt.grid(True, linestyle=':', alpha=0.7)
	plt.xlim(0,3)
	plt.show()

data = read_data("structure_factor_data.txt")
analysis(data)










