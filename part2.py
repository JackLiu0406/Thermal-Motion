import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

pix_to_um = 0.12048  # Conversion factor from pixels to micrometers
bead_diameter_um = 1.9  # diameter of bead in micrometers
viscosity = 0.932  # viscosity of water in centipoise at 23.5°C
temperature = 296.5  # temperature in Kelvin
accepted_boltzmann_constant = 1.38e-23  # accepted value of Boltzmann constant in J/K
time_uncertainty = 0.03

# Convert viscosity to SI units (1 centipoise = 0.001 Pascal-second)
viscosity_pa_s = viscosity * 0.001
# Calculate the bead radius in meters
bead_radius_m = (bead_diameter_um * 1e-6) / 2

# errors
bead_radium_m_error = 0.05 * 1e-6
viscosity_pa_s_error = 0.05 * 0.001 
temperature_error = 0.5
x_uncertainty = np.sqrt(2)*1e-1  # Uncertainty in x direction (in micrometers)
y_uncertainty = np.sqrt(2)*1e-1  # Uncertainty in y direction (in micrometers)

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

folder_path = 'Data'
data_dict = load_data_as_tuples(folder_path)

def msd(x, y):
    return (x * pix_to_um) ** 2 + (y * pix_to_um) ** 2

# Probability Distribution of Step Sizes following lab instructions
def rayleigh_pdf_fit(x, scale):
    return (x / scale) * np.exp(-x**2 / (2 * scale))

def k_calc(D, D_error):
    k = D * 6 * np.pi * viscosity_pa_s * bead_radius_m * (10**-12)/ temperature
    k_error = k*np.sqrt((D_error/D)**2 + (bead_radium_m_error/bead_radius_m)**2 + (viscosity_pa_s_error/viscosity_pa_s)**2 + (temperature_error/temperature)**2)
    return k, k_error

def delta_msd(x, y):
    """Calculate uncertainty in MSD using x and y uncertainties."""
    x_2 = 2*x*x_uncertainty
    y_2 = 2*y*y_uncertainty
    r_2 = np.sqrt(x_2**2 + y_2**2)
    return r_2

step_sizes = []
step_sizes_error = []

for key in data_dict:
    for c in range(1, len(data_dict[key])):
        # Append step size in micrometers
        x_diff1 = data_dict[key][c][0] - data_dict[key][c-1][0]
        y_diff1 = data_dict[key][c][1] - data_dict[key][c-1][1]
        step_size = np.sqrt(msd((x_diff1), (y_diff1)))
        error = delta_msd(x_diff1, y_diff1)
        step_sizes_error.append(error)
        step_sizes.append(step_size)


# part 2a: probability distribution method   
bins1 = 30

# Fit the Rayleigh distribution using curve_fit
hist, bin_edges = np.histogram(step_sizes, bins=bins1, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
popt, pcov = curve_fit(rayleigh_pdf_fit, bin_centers, hist)
scale_fit = popt[0]
D1 = scale_fit / (2 * 0.5)  # Calculate diffusion coefficient D from scale parameter with 0.5 s interval4

D1_std_dev = D1*np.sqrt((np.sqrt(pcov[0, 0])/scale_fit)**2 + (time_uncertainty/0.5)**2)
k1, k1_std_dev = k_calc(D1, D1_std_dev)
k1_precent_error = abs((k1 - accepted_boltzmann_constant) / accepted_boltzmann_constant) * 100

print(f"\n\nDiffusion Coefficient from Probability Distribution Method: {D1:.3e} um^2/s")
print(f"Standard Deviation of D_Fit: {D1_std_dev:.3e}")
print(f"\nBoltzmann Constant from Probability Distribution Method: {k1:.3e} J/K")
print(f"Standard Deviation of k_Fit: {k1_std_dev:.3e}")
print(f"Percent Error: {k1_precent_error:.2f}%")

# Plot the Rayleigh fit
plt.figure()
plt.hist(step_sizes, bins=bins1, density=True, alpha=0.6, color='b', edgecolor='black')
x_values = np.linspace(0, max(step_sizes), 100)
plt.plot(x_values, rayleigh_pdf_fit(x_values, scale_fit), 'r-', label="Rayleigh Fit (Probability Distribution)")





# Part 2 b: maximum likelihood estimate
s = 0
s_error = 0
for i in range(len(step_sizes)):
    s += step_sizes[i] ** 2
    s_error += step_sizes_error[i] ** 2
s /= 2 * len(step_sizes) 
s_error = np.sqrt(s_error) / (2 * len(step_sizes))

D2 = s / (2 * 0.5)
D2_std_dev = D2*np.sqrt((s_error/s)**2 + (time_uncertainty/0.5)**2)
k2, k2_std_dev = k_calc(D2, D2 * 0.05)
k2_percent_error = abs((k2 - accepted_boltzmann_constant) / accepted_boltzmann_constant) * 100

print(f"\n\n\nDiffusion Coefficient from Maximum Likelihood Estimate: {D2:.3e} um^2/s")
print(f"Standard Deviation of D_MLE: {D2_std_dev:.3e}")
print(f"\nBoltzmann Constant from Maximum Likelihood Estimate: {k2:.3e} J/K")
print(f"Standard Deviation of k_MLE: {k2_std_dev:.3e}")
print(f"Percent Error: {k2_percent_error:.2f}%\n\n")



plt.plot(x_values, rayleigh_pdf_fit(x_values, D2), 'g-', label="Rayleigh Fit (MLE)")
# Plotting details for Probability Distribution
plt.xlabel("Step Size (um)")
plt.ylabel("Probability Density")
plt.title("Probability Distribution of Step Sizes")
plt.legend(loc="best")
plt.show()



# residual for curve fit
# Predicted values using the fit
predicted_counts = rayleigh_pdf_fit(bin_centers, *popt)
# Calculate residuals
residuals = hist - predicted_counts

# Plot the residuals
#plt.bar(bin_centers, residuals, width=np.diff(bin_edges), alpha=0.5, label="Residuals")
plt.axhline(0, color='black', linestyle='--', linewidth=1)

# Plot dots at the actual value locations and straight lines going down to those points
for center, residual in zip(bin_centers, residuals):
    plt.plot(center, residual, 'ro')
    plt.vlines(center, 0, residual, colors='r', linestyles='dotted')

plt.title("Residuals of the Rayleigh Distribution Fit")
plt.xlabel("Step Size (um)")
plt.ylabel("Residual")
plt.show()