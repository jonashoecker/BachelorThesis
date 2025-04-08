import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# define lorentzian form of structure factor for curve_fit
def lorentzian_structure_factor(q, xi):
	return 1 / (1 + q**2 * xi**2)

# exponential fit
def exponential_fit(r, A, xi):
	return A * np.exp(-r / xi)

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

# read lower Tc data
def read_lower_data(filename):
	data = {}
	with open(filename, 'r') as f:
		for line in f:
			if line.startswith('#'):
				continue
			T, q, S, S_err, M, M_err = map(float, line.strip().split())
			data.setdefault(T, []).append((q, S, S_err, M, M_err))

	# sort q for each temperature
	for T in data:
		data[T].sort(key= lambda x: x[0])

	return data

def analysis(data, fit):
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

		
		# normalization (propagation error mode)
		S0 = S_vals[0]
		e0 = errors[0]
		S_vals /= S0
		errors = S_vals * np.sqrt((errors / S_vals)**2 + (e0 / S0)**2)

		# take into account only small q's for fit
		q_fit = q_vals[:6]
		S_fit = S_vals[:6]
		# scipy fitting
		if fit == 0:
			popt, pcov = curve_fit(lorentzian_structure_factor, q_fit, S_fit, maxfev=5000)
		elif fit == 1:
			popt, pcov = curve_fit(exponential_fit, q_fit, S_fit)
		# extract correlation length
		xi = popt[0]
		xi_error = np.sqrt(pcov[0, 0])

		# append results
		T_array.append(T)
		xi_array.append(xi)
		xi_error_array.append(xi_error)

	return T_array, xi_array, xi_error_array

def analysis_lower_Tc(data):
	# introduce variables
	T_array = []
	xi_array = [] 
	xi_error_array = []

	for T in sorted(data.keys()):
		values = data[T]
		q_vals, S_vals, S_errs, M_vals, M_errs = zip(*values)
		# list to numpy arrays
		q_vals = np.array(q_vals)
		S_vals = np.array(S_vals)
		S_errs = np.array(S_errs)
		M_vals = np.array(M_vals)
		M_errs = np.array(M_errs)

		# take mean of magnetization
		M = np.mean(M_vals)
		M_err = np.mean(M_errs)

		# define volume
		vol = 30*30
		#vol = 1

		# transform S(q) to real-space correlation function C(r)
		C_vals = np.fft.ifft(S_vals).real

		# error of correlation function
		C_errs = np.sqrt(S_errs**2 + (2 * M * M_err)**2)

		# normalization
		C0 = C_vals[0]
		C_vals /= C0
		C_errs /= C0

		# subtract spontaneous magnetization squared
		C_vals -= M**2

		C_vals *= 100

		# take into account only small q's for fit
		q_fit = q_vals[:6]
		C_fit = C_vals[:6]
		err_fit = C_errs[:6]

		popt, pcov = curve_fit(exponential_fit, q_fit, C_fit, sigma=err_fit, maxfev=10000)
		xi = popt[0]
		xi_err = np.sqrt(pcov[0, 0])
		print(xi)

		T_array.append(T)
		xi_array.append(xi)
		xi_error_array.append(xi_err)

	return np.array(T_array), np.array(xi_array), np.array(xi_error_array)

def plot(T_val, xi_val, errors, color, marker, label):
    plt.errorbar(
        T_val, xi_val, yerr=errors,
        fmt=marker, capsize=4, capthick=1.5,
        markersize=3, markeredgewidth=1.5,
        color=color, ecolor='black', linewidth=1.5, label=label)


# attempt to extract exponent nu
def extract_nu(T_val, xi_val):
	# define critical temperature
	T_c = 2.2680

	nu_vals = []

	for i in range(len(T_val)):
		nu = np.log(1 / xi_val[i]) / np.log(abs(T_val[i] - T_c))
		nu_vals.append(float(nu))  # convert from np.float64 to float

	"""
	# Print
	print("Estimated ν values for T > T_c:")
	for T, nu in zip(T_val, nu_vals):
		print(f"T = {T:.4f}, ν ≈ {nu:.4f}")

	# Compute mean
	mean = np.mean(nu_vals)
	print('---------')
	print(f'Mean exponent ν ≈ {mean:.4f}')
	"""

def main():


	data = read_lower_data("lower_Tc_structure_factor_data.txt")
	T_val, xi_val, xi_err = analysis_lower_Tc(data)
	plot(T_val, xi_val, xi_err, '#49a9ca', 'o', 'Lower Tc calculation')
	

	data = read_data("upper_Tc_structure_factor_data.txt")
	T_val, xi_val, errors = analysis(data, 0)
	plot(T_val, xi_val, errors, '#8a49ca', '^', 'First calculation')

	data = read_data("closer_upper_Tc_structure_factor_data.txt")
	T_val, xi_val, errors = analysis(data, 0)
	plot(T_val, xi_val, errors, '#82ca49', 'D', 'Second calculation')


	# plot critical temperature line
	T_c = 2.2680  # critical temperature
	plt.axvline(x=T_c, color='#f1560e', linestyle='--', linewidth=2, label='$T_c = 2.2680$')

	# show plot
	plt.xlabel("Temperature $T$", fontsize=14)
	plt.ylabel("Correlation length $\\xi$", fontsize=14)
	plt.title("Correlation Length vs Temperature", fontsize=16)
	plt.grid(True, linestyle='--', alpha=0.6)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.xlim(1, 4)
	#plt.ylim(-0.1, 0.5)
	plt.tight_layout()
	plt.legend()
	plt.show()

	# define critical temperature
	T_c = 2.2680

	# define arrays for high temp (T > Tc)
	T_above = []
	xi_above = []

	for i in range(len(T_val)):
		if T_val[i] > T_c:
			T_above.append(T_val[i])
			xi_above.append(xi_val[i])

	extract_nu(T_above, xi_above)

main()