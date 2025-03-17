"""
        Bachelor Thesis : Jonas Hoecker

        Ising Model Implementation

        Critical Temperature Determination
"""


import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

file_names = [
    'wolff_algo_size100.txt',
    'wolff_algo_size160.txt',
    'wolff_algo_size140.txt'
]

colors = ['darkorange', 'forestgreen', 'crimson']

plt.figure(figsize=(12, 7))  # Slightly larger figure

for idx, file_name in enumerate(file_names):
    temperature = []
    U_values = []
    U_errors = []

    # Extract the spin configuration size from the filename
    size = file_name.split('size')[-1].split('.')[0]

    with open(file_name, "r") as file:
        next(file)  # Skip header
        for line in file:
            temp, _, _, m2, m4 = line.strip().split("\t")
            temperature.append(float(temp))
            U_values.append(float(m2))
            U_errors.append(float(m4))

    # Convert lists to NumPy arrays
    temperature = np.array(temperature)
    U_values = np.array(U_values)
    U_errors = np.array(U_errors)

    # Plot with improved markers and line visibility
    plt.errorbar(temperature, U_values, yerr=U_errors,
                 label=f'Spin {size}x{size} Matrix',
                 marker='o',
                 markersize=6,
                 linewidth=2,
                 color=colors[idx],
                 markerfacecolor='white',
                 markeredgewidth=1.2,
                 alpha=0.85,
                 ecolor='black',
                 elinewidth=1,
                 capsize=3)

# Add title and axis labels
plt.title("Binder Cumulant to Determine Critical Temperature", fontsize=16, fontweight='bold', color='black')
plt.xlabel("Temperature", fontsize=14, color='black')
plt.ylabel("Binder Cumulant, $U$", fontsize=14, color='black')

# Show T_c
plt.axvline(x=2.2680, linestyle='--', color='red', linewidth=2, alpha=0.8, label='Critical temperature')
plt.text(2.272, 0.05, r'$T_c = 2.2680$ K', fontsize=12, color='red',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

# Show legend
plt.legend(title="Spin Config Size", fontsize=12, title_fontsize=13, loc='best', frameon=True)

# Add gridlines
plt.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.7)  # Major grid
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)  # Minor grid
plt.minorticks_on()  # Enable minor ticks

# Enhance layout
plt.tight_layout()

# Show the graph
plt.show()
