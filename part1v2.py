import os
import matplotlib.pyplot as plt
import numpy as np

# Constants
pix_to_um = 0.12048  # Conversion factor from pixels to micrometers
bead_diameter_um = 1.9  # Diameter of bead in micrometers
viscosity = 0.932  # Viscosity of water in centipoise at 20°C
temperature = 296.5  # Temperature in Kelvin
accepted_boltzmann_constant = 1.38e-23  # Accepted value of Boltzmann constant in J/K

# Uncertainties
uncertainty_position_um = 1e-1  # Uncertainty in position (in micrometers)
viscosity_uncertainty_percent = 0.05 / 1.00  # Given viscosity uncertainty is ±0.05 centipoise
viscosity_pa_s_err = viscosity * viscosity_uncertainty_percent * 0.001  # Convert to Pa·s
temperature_err = 0.5  # Uncertainty in temperature in Kelvin
bead_radius_m_err = 0.05e-6  # Uncertainty in bead radius in meters (±0.05 μm)

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
                data = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            data.append((x, y))
                        except ValueError:
                            continue  # Skip lines that cannot be parsed
            data_dict[key] = data
    return data_dict

# Replace this with your actual data folder path
folder_path = '/Users/dingshengliu/Desktop/Thermal Motion Lab/Thermal Motion Lab/Data'  # Update this path
data_dict = load_data_as_tuples(folder_path)

def msd(s_x, s_y):
    """Calculate Mean Squared Displacement (MSD) in micrometers squared."""
    return s_x ** 2 + s_y ** 2

def delta_msd(s_x, s_y):
    """
    Calculate uncertainty in MSD using uncertainties in s_x and s_y.
    δMSD = sqrt( (2 * s_x * δs_x)^2 + (2 * s_y * δs_y)^2 )
    """
    δs_x = uncertainty_position_um
    δs_y = uncertainty_position_um
    return np.sqrt((2 * s_x * δs_x) ** 2 + (2 * s_y * δs_y) ** 2)

# MSD calculations with uncertainties
mean_squared = {}
delta_msd_values = {}
time_intervals = []

# Time intervals (assuming 2 frames per second)
num_points = max(len(data_dict[key]) for key in data_dict)
time_intervals = [i / 2 for i in range(num_points)]

# Processing each trial
for key in data_dict:
    msd_values = []
    delta_msd_list = []
    initial_x, initial_y = data_dict[key][0]
    for c in range(len(data_dict[key])):
        x_diff = data_dict[key][c][0] - initial_x
        y_diff = data_dict[key][c][1] - initial_y

        # Convert positions from pixels to micrometers
        s_x = x_diff * pix_to_um
        s_y = y_diff * pix_to_um

        # Calculate MSD and its uncertainty
        msd_value = msd(s_x, s_y)
        msd_values.append(msd_value)
        delta_msd_value = delta_msd(s_x, s_y)
        delta_msd_list.append(delta_msd_value)

    mean_squared[key] = msd_values
    delta_msd_values[key] = delta_msd_list

# Plot and regression
plt.figure(figsize=(10, 6))
results = {}
k_values = []
delta_k_values = []

def linear_func(t, slope, intercept):
    return slope * t + intercept

for key in mean_squared:
    # Prepare data for fitting
    t_data = np.array(time_intervals[1:])
    msd_data = np.array(mean_squared[key][1:])
    sigma_data = np.array(delta_msd_values[key][1:])

    # Perform weighted linear regression using uncertainties
    weights = 1 / sigma_data**2
    W = np.sum(weights)
    W_x = np.sum(weights * t_data)
    W_y = np.sum(weights * msd_data)
    W_xx = np.sum(weights * t_data * t_data)
    W_xy = np.sum(weights * t_data * msd_data)

    denominator = W * W_xx - W_x ** 2
    if denominator == 0:
        print(f"Cannot perform linear regression for trial {key} due to zero denominator.")
        continue

    # Calculate slope (m) and intercept (b) with their uncertainties
    slope = (W * W_xy - W_x * W_y) / denominator
    intercept = (W_y * W_xx - W_x * W_xy) / denominator

    slope_err = np.sqrt(W / denominator)
    intercept_err = np.sqrt(W_xx / denominator)

    # Diffusion coefficient (D) and its uncertainty
    diffusion_coefficient = slope / 2
    diffusion_coefficient_err = slope_err / 2

    # Convert D to SI units (from μm²/s to m²/s)
    diffusion_coefficient_m2_per_s = diffusion_coefficient * 1e-12
    diffusion_coefficient_m2_per_s_err = diffusion_coefficient_err * 1e-12

    # Calculate gamma (viscous drag coefficient)
    gamma = 6 * np.pi * viscosity_pa_s * bead_radius_m

    # Boltzmann constant (k) and its uncertainty
    k_trial = (diffusion_coefficient_m2_per_s * gamma) / temperature
    # Error propagation for k_trial
    k_trial_err = k_trial * np.sqrt(
        (diffusion_coefficient_m2_per_s_err / diffusion_coefficient_m2_per_s) ** 2 +
        (viscosity_pa_s_err / viscosity_pa_s) ** 2 +
        (bead_radius_m_err / bead_radius_m) ** 2 +
        (temperature_err / temperature) ** 2
    )

    percent_error = abs((k_trial - accepted_boltzmann_constant) / accepted_boltzmann_constant) * 100

    # Reduced chi-squared
    predicted_msd = linear_func(t_data, slope, intercept)
    chi_squared = np.sum(((msd_data - predicted_msd) / sigma_data) ** 2)
    degrees_of_freedom = len(msd_data) - 2
    reduced_chi_squared = chi_squared / degrees_of_freedom

    # Collect results for each trial
    results[key] = {
        "Diffusion Coefficient (D)": diffusion_coefficient,
        "D Uncertainty": diffusion_coefficient_err,
        "Boltzmann Constant (k)": k_trial,
        "k Uncertainty": k_trial_err,
        "Percent Error (%)": percent_error,
        "Reduced Chi-Squared": reduced_chi_squared
    }

    print(f"\nTrial {key}:")
    print(f"  Slope: {slope:.3e} ± {slope_err:.3e} μm²/s")
    print(f"  Intercept: {intercept:.3e} ± {intercept_err:.3e} μm²")
    print(f"  Diffusion Coefficient (D): {diffusion_coefficient:.3e} ± {diffusion_coefficient_err:.3e} μm²/s")
    print(f"  Boltzmann Constant (k): {k_trial:.3e} ± {k_trial_err:.3e} J/K")
    print(f"  Percent Error: {percent_error:.2f}%")
    print(f"  Reduced Chi-Squared: {reduced_chi_squared:.3f}")

    plt.errorbar(t_data, msd_data, yerr=sigma_data, label=f'Trial {key}')
    plt.plot(t_data, predicted_msd, '--', label=f'Trial {key} Fit')

    k_values.append(k_trial)
    delta_k_values.append(k_trial_err)

# Weighted average of the Boltzmann constant
k_values = np.array(k_values)
delta_k_values = np.array(delta_k_values)
weights_k = 1 / delta_k_values ** 2
k_weighted_avg = np.sum(k_values * weights_k) / np.sum(weights_k)
delta_k_weighted_avg = np.sqrt(1 / np.sum(weights_k))

print(f"\nWeighted Average Boltzmann Constant (k): {k_weighted_avg:.3e} ± {delta_k_weighted_avg:.3e} J/K")
print(f"Accepted Boltzmann Constant (k): {accepted_boltzmann_constant:.3e} J/K")
percent_error_avg = abs((k_weighted_avg - accepted_boltzmann_constant) / accepted_boltzmann_constant) * 100
print(f"Percent Error of Weighted Average: {percent_error_avg:.2f}%")

# Plot formatting
plt.xlabel('Time (s)')
plt.ylabel('Mean Squared Displacement (μm²)')
plt.title('MSD vs Time with Weighted Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
