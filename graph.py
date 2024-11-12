import os
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
from scipy.optimize import curve_fit

pix_to_um = 0.12048  # Conversion factor from pixels to micrometers
bead_diameter_um = 1.9  # diameter of bead in micrometers
viscosity = 0.932  # viscosity of water in centipoise at 20°C
temperature = 296.5  # temperature in Kelvin
accepted_boltzmann_constant = 1.38e-23  # accepted value of Boltzmann constant in J/K

# Convert viscosity to SI units (1 centipoise = 0.001 Pascal-second)
viscosity_pa_s = viscosity * 0.001

# Calculate the bead radius in meters
bead_radius_m = (bead_diameter_um * 1e-6) / 2

def load_data_as_tuples(folder_path):
    """Load all .txt files in the specified folder into a dictionary with arrays of (x, y) tuples."""
    data_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            key = os.path.splitext(filename)[0]
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()[2:]  # Skip the first two header lines
                data = [(float(line.split()[0]), float(line.split()[1])) for line in lines]
            data_dict[key] = data
    return data_dict

folder_path = '/Users/dingshengliu/Desktop/Thermal Motion Lab/Thermal Motion Lab/Data'
data_dict = load_data_as_tuples(folder_path)

def msd(x, y):
    return (x * pix_to_um) ** 2 + (y * pix_to_um) ** 2

mean_squared = {}
time_intervals = []
step_sizes = []

# Calculate MSD for each time step and collect step sizes
for key in data_dict:
    mean_squared[key] = [0]
    for c in range(1, len(data_dict[key])):
        x_diff = data_dict[key][c][0] - data_dict[key][0][0]
        y_diff = data_dict[key][c][1] - data_dict[key][0][1]
        
        # Append step size in micrometers
        x_diff1 = data_dict[key][c][0] - data_dict[key][c-1][0]
        y_diff1 = data_dict[key][c][1] - data_dict[key][c-1][1]
        step_size = np.sqrt((x_diff1 * pix_to_um) ** 2 + (y_diff1 * pix_to_um) ** 2)
        step_sizes.append(step_size)
        
        mean_squared[key].append(msd(x_diff, y_diff))

# Generate time values (in seconds) based on 2 frames per second
time_intervals = [i / 2 for i in range(len(mean_squared[key]))]

# Plot MSD vs Time and calculate diffusion coefficient D, Boltzmann constant k, percent error, and reduced chi-squared for each trial
plt.figure()
results = {}
k_values = []

for key in mean_squared:
    # Perform a linear fit to MSD vs. time
    slope, intercept, r_value, p_value, std_err = linregress(time_intervals, mean_squared[key])
    diffusion_coefficient = slope / 4  # D = slope / 4 for MSD in 2D

    # Convert D from um^2/s to m^2/s for Boltzmann calculation
    diffusion_coefficient_m2_per_s = diffusion_coefficient * 1e-12

    # Calculate Boltzmann constant k for this trial
    k_trial = (diffusion_coefficient_m2_per_s * 6 * np.pi * viscosity_pa_s * bead_radius_m) / temperature
    percent_error = abs((k_trial - accepted_boltzmann_constant) / accepted_boltzmann_constant) * 100

    # Calculate reduced chi-squared
    predicted_msd = [slope * t + intercept for t in time_intervals]
    chi_squared = np.sum([(observed - predicted) ** 2 / predicted for observed, predicted in zip(mean_squared[key], predicted_msd)])
    degrees_of_freedom = len(mean_squared[key]) - 2  # 2 fitted parameters (slope and intercept)
    reduced_chi_squared = chi_squared / degrees_of_freedom

    # Store results for each trial
    results[key] = {
        "Diffusion Coefficient (D)": diffusion_coefficient,
        "Boltzmann Constant (k)": k_trial,
        "Percent Error (%)": percent_error,
        "Reduced Chi-Squared": reduced_chi_squared
    }
    
    # Print intermediate results for debugging
    print(f"\nTrial {key} Intermediate Results:")
    print(f"  Slope of MSD fit: {slope:.3e}")
    print(f"  Diffusion Coefficient (D) in um^2/s: {diffusion_coefficient:.3e}")
    print(f"  Diffusion Coefficient (D) in m^2/s: {diffusion_coefficient_m2_per_s:.3e}")
    print(f"  Boltzmann Constant (k) calculated: {k_trial:.3e} J/K")
    print(f"  Percent Error in k: {percent_error:.2f}%")
    print(f"  Reduced Chi-Squared: {reduced_chi_squared:.3f}")

    # Plot each trial's MSD data and fit
    plt.plot(time_intervals, mean_squared[key], label=key)
    plt.plot(time_intervals, predicted_msd, '--', label=f"{key} fit")
    k_values.append(k_trial)


# Plotting details for MSD vs Time
plt.legend(loc="best")
plt.xlabel("Time (s)")
plt.ylabel("Mean Squared Distance (um^2)")
plt.title("Mean Squared Distance vs. Time for Brownian Motion Trials")

# Show MSD plot
plt.show()

# Display results summary
print("\nResults Summary:")
for key, metrics in results.items():
    print(f"{key}: Diffusion Coefficient (D) = {metrics['Diffusion Coefficient (D)']:.3e} um^2/s, "
          f"Boltzmann Constant (k) = {metrics['Boltzmann Constant (k)']:.3e} J/K, "
          f"Percent Error = {metrics['Percent Error (%)']:.2f}%, "
          f"Reduced Chi-Squared = {metrics['Reduced Chi-Squared']:.3f}")




# PART 2 a: Rayleigh Distribution Fit
bins1 = 35
# Probability Distribution of Step Sizes following lab instructions
def rayleigh_pdf_fit(x, scale):
    return (x / scale) * np.exp(-x**2 / (2 * scale))

plt.figure()
plt.hist(step_sizes, bins=bins1, density=True, alpha=0.6, color='b', edgecolor='black')

# Fit the Rayleigh distribution using curve_fit
hist, bin_edges = np.histogram(step_sizes, bins=bins1, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
popt, pcov = curve_fit(rayleigh_pdf_fit, bin_centers, hist)
scale_fit = popt[0]
D_fit = scale_fit / (2 * 0.5)  # Calculate diffusion coefficient D from scale parameter with 0.5 s interval

# Calculate Boltzmann constant from probability distribution method
k_from_distribution = D_fit * 6 * np.pi * viscosity_pa_s * bead_radius_m * (10**-12)/ temperature
percent_error_distribution = abs((k_from_distribution - accepted_boltzmann_constant) / accepted_boltzmann_constant) * 100
print(f"\nBoltzmann Constant from Probability Distribution Method: {k_from_distribution:.3e} J/K")
print(f"Percent Error: {percent_error_distribution:.2f}%")

# Plot the Rayleigh fit
x_values = np.linspace(0, max(step_sizes), 100)
plt.plot(x_values, rayleigh_pdf_fit(x_values, scale_fit), 'r-', label="Rayleigh Fit (Probability Distribution)")


# Part 2 b: maximum likelihood estimate
D_mle = 0
for i in range(len(step_sizes)):
    D_mle += step_sizes[i] ** 2
D_mle /= 2 * len(step_sizes) 

k_mle = D_mle * 6 * np.pi * viscosity_pa_s * bead_radius_m * (10**-12)/ temperature
percent_error_mle = abs((k_mle - accepted_boltzmann_constant) / accepted_boltzmann_constant) * 100
print(f"\nBoltzmann Constant from Maximum Likelihood Estimate: {k_mle:.3e} J/K")
print(f"Percent Error: {percent_error_mle:.2f}%")

plt.plot(x_values, rayleigh_pdf_fit(x_values, D_mle), 'g-', label="Rayleigh Fit (MLE)")

# Plotting details for Probability Distribution
plt.xlabel("Step Size (um)")
plt.ylabel("Probability Density")
plt.title("Probability Distribution of Step Sizes")
plt.legend(loc="best")

# Show probability distribution plot
plt.show()