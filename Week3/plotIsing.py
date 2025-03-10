"""
        Bachelor Thesis : Jonas Hoecker

        Ising Model Implementation


"""

import matplotlib.pyplot as plt

#file_names = [ 'magnetization_vs_temp_25x25.txt',
	#'magnetization_vs_temp_30x30.txt',
	#'magnetization_vs_temp_40x40.txt']

file_names = [
    'magnetization_vs_temp_25x25.txt',
    'magnetization_vs_temp_30x30.txt',
    'magnetization_vs_temp_40x40.txt'
]

# Define distinct colors
colors = ['blue', 'green', 'red']

plt.figure(figsize=(10, 6))

for idx, file_name in enumerate(file_names):
    temperature = []
    magnetization = []
    error = []
    
    # Extract the spin configuration size from the filename (e.g., '5x5')
    size = file_name.split('_')[-1].split('.')[0]
    
    with open(file_name, "r") as file:
        next(file)  # Skip the header line
        for line in file:
            temp, mag, err = line.strip().split("\t")
            temperature.append(float(temp))
            magnetization.append(float(mag))
            error.append(float(err))
    
    # Plot each dataset with markers and label
    plt.errorbar(temperature, magnetization, yerr = error,
             label=f"Spin Config {size}",
             color=colors[idx],
             marker='o',
             markersize=4,
             linewidth=2,
             ecolor='black',
             elinewidth=1,
             capthick=1.5)

# Add title and axis labels
plt.title("Metropolis Algorithm", fontsize=14)
plt.xlabel("Temperature", fontsize=12)
plt.ylabel("Magnetization", fontsize=12)

# Include a legend to indicate spin configuration sizes
plt.legend(title="Spin Config Size")

# Enhance layout and show gridlines
plt.grid(True)
plt.tight_layout()
plt.show()
