import os
import matplotlib.pyplot as plt
import numpy as np

# Constants
pix_to_um = 0.12048  # Conversion factor from pixels to micrometers
bead_diameter_um = 1.9  # Diameter of bead in micrometers
viscosity = 0.932  # Viscosity of water in centipoise at 20°C
temperature = 296.5  # Temperature in Kelvin
accepted_boltzmann_constant = 1.38e-23  # Accepted value of Boltzmann constant in J/K
uncertainty_x = 1e-1  # Uncertainty in x direction (in micrometers)
uncertainty_y = 1e-1  # Uncertainty in y direction (in micrometers)

# Convert viscosity to SI units (1 centipoise = 0.001 Pascal-second)
viscosity_pa_s = viscosity * 0.001
viscosity_pa_s_err = 0.05 * 0.001  # Uncertainty in viscosity in Pa·s
temperature_err = 0.5  # Uncertainty in temperature in Kelvin

# Calculate the bead radius in meters
bead_radius_m = (bead_diameter_um * 1e-6) / 2
bead_radius_m_err = 0.05e-6  # Uncertainty in radius in meters

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

# Replace this with your actual data folder path
folder_path = '/Users/dingshengliu/Desktop/Thermal Motion Lab/Thermal Motion Lab/Data'  # Update this path
data_dict = load_data_as_tuples(folder_path)

def msd(x, y):
    return (x * pix_to_um) ** 2 + (y * pix_to_um) ** 2

def delta_msd(msd_value):
    """Calculate uncertainty in MSD using x and y uncertainties."""
    delta_s = np.sqrt(uncertainty_x**2 + uncertainty_y**2)
    return np.sqrt(8 * msd_value * (delta_s) ** 2)

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
    for c in range(len(data_dict[key])):
        x_diff = data_dict[key][c][0] - data_dict[key][0][0]
        y_diff = data_dict[key][c][1] - data_dict[key][0][1]
        
        # Calculate MSD and its uncertainty
        msd_value = msd(x_diff, y_diff)
        msd_values.append(msd_value)
        delta_msd_list.append(delta_msd(msd_value))
        
    mean_squared[key] = msd_values
    delta_msd_values[key] = delta_msd_list

# Plot and regression
plt.figure()
results = {}
k_values = []
delta_k_values = []

for key in mean_squared:
    # Prepare data for fitting
    t_data = np.array(time_intervals[1:])
    msd_data = np.array(mean_squared[key][1:])
    sigma_data = np.array(delta_msd_values[key][1:])
    
    # Perform weighted linear regression using uncertainties
    weights = 1 / sigma_data**2
    
    # Use np.polyfit with weights and obtain covariance matrix
    coeffs, cov = np.polyfit(t_data, msd_data, 1, w=weights, cov=True)
    m, b = coeffs
    m_err, b_err = np.sqrt(np.diag(cov))
    
    # Diffusion coefficient (D) and its uncertainty
    diffusion_coefficient = m / 4
    diffusion_coefficient_err = m_err / 4
    
    # Convert D to SI units (from um^2/s to m^2/s)
    diffusion_coefficient_m2_per_s = diffusion_coefficient * 1e-12
    diffusion_coefficient_m2_per_s_err = diffusion_coefficient_err * 1e-12

    # Boltzmann constant (k) and its uncertainty
    k_trial = (diffusion_coefficient_m2_per_s * 6 * np.pi * viscosity_pa_s * bead_radius_m) / temperature
    # Error propagation formula for multiplication/division
    k_trial_err = k_trial * 6 * np.sqrt(
        (diffusion_coefficient_m2_per_s_err / diffusion_coefficient_m2_per_s) ** 2 +
        (viscosity_pa_s_err / viscosity_pa_s) ** 2 +
        (bead_radius_m_err / bead_radius_m) ** 2 +
        (temperature_err / temperature) ** 2
    )
    
    percent_error = abs((k_trial - accepted_boltzmann_constant) / accepted_boltzmann_constant) * 100
    
    # Reduced chi-squared
    predicted_msd = m * t_data + b
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
    print(f"  Slope: {m:.3e} ± {m_err:.3e}")
    print(f"  Diffusion Coefficient (D): {diffusion_coefficient:.3e} ± {diffusion_coefficient_err:.3e} um^2/s")
    print(f"  Boltzmann Constant (k): {k_trial:.3e} ± {k_trial_err:.3e} J/K")
    print(f"  Percent Error: {percent_error:.2f}%")
    print(f"  Reduced Chi-Squared: {reduced_chi_squared:.3f}")
    
    plt.errorbar(t_data, msd_data, yerr=sigma_data, label=f'Trial {key}')
    plt.plot(t_data, predicted_msd, '--', label=f'Trial {key} fit')
    
    k_values.append(k_trial)
    delta_k_values.append(k_trial_err)

# Weighted average of the Boltzmann constant
k_values = np.array(k_values)
delta_k_values = np.array(delta_k_values)
weights_k = 1 / delta_k_values ** 2
k_weighted_avg = np.sum(k_values * weights_k) / np.sum(weights_k)
delta_k_weighted_avg = np.sqrt(1 / np.sum(weights_k))

print(f"\nWeighted Average Boltzmann Constant (k): {k_weighted_avg:.3e} ± {delta_k_weighted_avg:.3e} J/K")

# Plot formatting
plt.xlabel('Time (s)')
plt.ylabel('Mean Squared Displacement (μm²)')
plt.title('MSD vs Time with Weighted Linear Regression')
plt.legend()
plt.show()