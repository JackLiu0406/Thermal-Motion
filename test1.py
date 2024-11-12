import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import rayleigh
import functions  # Assuming you have a module named 'functions' with necessary functions
from sklearn.metrics import r2_score

# Constants and Conversion Factors
conversion_factor = 0.12048  # microns per pixel
T = 296.5  # Temperature in Kelvin
viscosity = 0.932*0.001 # Viscosity in SI units (Pa.s)
radius = 1.9e-6 / 2  # Radius of the bead in meters
k_B = 1.38e-23  # Boltzmann constant

# Uncertainties
position_uncertainty = 0.1
time_uncertainty = 0.03
viscosity_uncertainty = 0.05e-3
radius_uncertainty = 0.1e-6 / 2
T_uncertainty = 0.5

# Function to validate data based on MSD consistency
def validate_data(m,b, time_intervals, mean_squared_displacement, r_squared_threshold):
    # Calculate R-squared value
    fit_values = m * time_intervals + b
    ss_res = np.sum((mean_squared_displacement - fit_values) ** 2)
    ss_tot = np.sum((mean_squared_displacement - np.mean(mean_squared_displacement)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Check if R-squared meets the threshold
    if r_squared < r_squared_threshold:
        return False
    else:
        return True


# Function to calculate MSD and step sizes from data
def calculate_msd_step_size(x_pixels, y_pixels):
    x_microns = x_pixels * conversion_factor
    y_microns = y_pixels * conversion_factor

    # Calculate the displacement (distance traveled) at each time step
    mean_squared_displacement = (x_microns - x_microns[0]) ** 2 + (y_microns-y_microns[0]) ** 2
    time_intervals = np.arange(1, len(mean_squared_displacement) + 1) * 0.5  # Time between frames

    # Uncertainty in displacement and MSD
    displacement_uncertainty = np.sqrt(2) * position_uncertainty
    msd_uncertainty = 2 * mean_squared_displacement * displacement_uncertainty

    return time_intervals, mean_squared_displacement, msd_uncertainty

# Function to handle data loading and analysis for all sets
def analyze_all_sets(num_sets):
    all_step_sizes = np.array([])
    all_mean_squared_displacements = []  # List to store MSD arrays from each set
    all_time_intervals = []              # List to store time intervals from each set

    # Create a figure for plotting MSD vs Time for individual valid sets
    plt.figure(figsize=(10, 8))

    valid_set_count = 0  # To count the number of valid data sets
    k_values = []
    k_uncertainties = []

    for i in range(1, num_sets + 1):
        data = np.loadtxt(f"{i}.txt", skiprows=2)
        x_pixels = data[:, 0]
        y_pixels = data[:, 1]

        # Calculate MSD and step sizes
        time_intervals, mean_squared_displacement, msd_uncertainty = calculate_msd_step_size(x_pixels, y_pixels)

        # Fit MSD to calculate D and k (part 1 calculations)
        m, sm, b, sb = functions.linear_fit_with_uncertainty(time_intervals, mean_squared_displacement)
        chi_squared, reduced_chi_squared = functions.calculate_chi_squared(
            time_intervals, mean_squared_displacement, msd_uncertainty, T_uncertainty, m, b)

        valid_set_count += 1
        plt.plot(time_intervals, mean_squared_displacement, label=f'Set {i}')
        all_step_sizes = np.concatenate((all_step_sizes, mean_squared_displacement))
        all_mean_squared_displacements.append(mean_squared_displacement)
        all_time_intervals.append(time_intervals)

        print(f"Set {i} - Slope (m): {m:.5e} ± {sm:.5e}")
        print(f"Set {i} - Intercept (b): {b:.5f} ± {sb:.5f}")
        print(f"Set {i} - Reduced Chi-Squared: {reduced_chi_squared:.5f}")

        # Calculate D and k
        D = (m / 4) * 1e-12  # Convert µm²/s to m²/s
        stokes_drag = 6 * np.pi * viscosity * radius
        k_calculated = D * stokes_drag / T
        D_uncertainty = (sm / 4) * 1e-12  # Convert µm²/s to m²/s

        # Error propagation for k uncertainty
        k_calculated_uncertainty = np.sqrt(
            (stokes_drag / T * D_uncertainty) ** 2 +
            (D * 6 * np.pi * radius / T * viscosity_uncertainty) ** 2 +
            (D * 6 * np.pi * viscosity / T * radius_uncertainty) ** 2 +
            (D * stokes_drag / T ** 2 * T_uncertainty) ** 2
        )

        k_values.append(k_calculated)
        k_uncertainties.append(k_calculated_uncertainty)

        R = 8.3145
        Na_calculated = R / k_calculated

        k_accepted = 1.38e-23
        avogadro = 6.022e23
        percent_difference = np.abs((k_calculated - k_accepted) / k_accepted) * 100
        avogadro_error = np.abs((Na_calculated - avogadro) / avogadro) * 100

        print(f"Set {i} - Calculated Boltzmann Constant (k): {k_calculated:.4e} J/K ± {k_calculated_uncertainty:.4e}")
        print(f"Set {i} - Percent Difference compared to accepted value of k: {percent_difference} %")
        print(f"Set {i} - Avogadro's Percent Error: {avogadro_error:.4e} %\n")

    if valid_set_count == 0:
        print("No valid data sets found.")
        return np.array([])

    # Finalize the MSD vs Time plot for all valid sets
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Squared Displacement ($\mu m^2$)')
    plt.title('MSD vs Time for Valid Sets')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate the weighted average of k
    k_values = np.array(k_values)
    k_uncertainties = np.array(k_uncertainties)
    weights = 1 / k_uncertainties ** 2
    weighted_k = np.sum(k_values * weights) / np.sum(weights)
    weighted_k_uncertainty = np.sqrt(1 / np.sum(weights))

    # Print the weighted average k and its uncertainty
    print(f"Weighted Average Boltzmann Constant (k): {weighted_k:.4e} J/K ± {weighted_k_uncertainty:.4e}")

    return all_step_sizes

def fit_rayleigh_distribution(all_step_sizes, time_interval):
    N = len(all_step_sizes)
    if N == 0:
        print("No step sizes to analyze.")
        return

    sum_r2 = np.sum(all_step_sizes ** 2)

    # Compute Maximum Likelihood Estimation (MLE) for sigma and D
    sigma_estimated_mle = np.sqrt(sum_r2 / (2 * N))
    D_estimated_mle = sigma_estimated_mle ** 2 / (2 * time_interval)

    # Calculate uncertainties for MLE
    delta_sigma_mle = sigma_estimated_mle / (2 * np.sqrt(N))  # δσ_MLE = σ_MLE / (2√N)
    delta_D_mle = D_estimated_mle / np.sqrt(N)                # δD_MLE = D_MLE / √N

    # Create histogram data and plot
    bins = int(np.sqrt(N))  # Using the square-root rule for bin selection
    hist_counts, bin_edges = np.histogram(all_step_sizes, bins=bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)

    # Normalize histogram for plotting
    hist_density = hist_counts / (N * bin_widths)
    #hist_density = hist_counts

    plt.hist(all_step_sizes, bins=bins, density=True, alpha=0.6,
             color='lightblue', edgecolor='black', label='Step Sizes Histogram')

    # Rayleigh distribution PDF definition
    def rayleigh_pdf(r, sigma):
        return (r / sigma ** 2) * np.exp(-r ** 2 / (2 * sigma ** 2))

    # Fit Rayleigh distribution using curve_fit
    sigma_guess = np.std(all_step_sizes)  # Initial guess for sigma
    popt, pcov = curve_fit(rayleigh_pdf, bin_centers, hist_density, p0=[sigma_guess])
    sigma_estimated_cf = popt[0]
    sigma_uncertainty_cf = np.sqrt(np.diag(pcov))[0]

    # Compute D from curve_fit estimate
    D_estimated_cf = sigma_estimated_cf ** 2 / (2 * time_interval)

    # Calculate uncertainty in D from curve_fit
    delta_D_cf = (sigma_estimated_cf / time_interval) * sigma_uncertainty_cf  # δD_cf = (σ_cf / Δt) * δσ_cf

    # Plot the fitted Rayleigh distributions
    r_values = np.linspace(0, np.max(all_step_sizes), 1000)
    plt.plot(r_values, rayleigh_pdf(r_values, sigma_estimated_cf), 'r-', lw=2,
             label=f'Rayleigh Fit (curve_fit), σ={sigma_estimated_cf:.2f} µm')
    plt.plot(r_values, rayleigh_pdf(r_values, sigma_estimated_mle), 'g--', lw=2,
             label=f'Rayleigh Fit (MLE), σ={sigma_estimated_mle:.2f} µm')

    plt.xlabel('Step size (microns)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.title('Step Size Histogram with Rayleigh Distribution Fits')
    plt.show()

    # Convert D_estimated to SI units (m^2/s)
    D_estimated_cf_SI = D_estimated_cf * 1e-12
    D_estimated_mle_SI = D_estimated_mle * 1e-12

    # Calculate Boltzmann constant k using D_estimated_cf
    stokes_drag = 6 * np.pi * viscosity * radius
    k_calculated_cf = (stokes_drag * D_estimated_cf_SI) / T

    # Calculate Boltzmann constant k using D_estimated_mle
    k_calculated_mle = (stokes_drag * D_estimated_mle_SI) / T

    # Propagate uncertainties for k (Curve Fit)
    delta_k_cf = np.sqrt(
        (stokes_drag / T * delta_D_cf * 1e-12) ** 2 +
        (D_estimated_cf_SI * 6 * np.pi * radius / T * viscosity_uncertainty) ** 2 +
        (D_estimated_cf_SI * 6 * np.pi * viscosity / T * radius_uncertainty) ** 2 +
        (D_estimated_cf_SI * stokes_drag / T ** 2 * T_uncertainty) ** 2  # Corrected to use D_estimated_cf_SI
    )

    # Propagate uncertainties for k (MLE)
    delta_k_mle = np.sqrt(
        (stokes_drag / T * delta_D_mle * 1e-12) ** 2 +
        (D_estimated_mle_SI * 6 * np.pi * radius / T * viscosity_uncertainty) ** 2 +
        (D_estimated_mle_SI * 6 * np.pi * viscosity / T * radius_uncertainty) ** 2 +
        (D_estimated_mle_SI * stokes_drag / T ** 2 * T_uncertainty) ** 2
    )

    error_cf = np.abs((k_calculated_cf - k_B) / k_B) * 100
    error_mle = np.abs((k_calculated_mle - k_B) / k_B) * 100

    # ---- Reduced Chi-Squared Calculation -----

    # Calculate expected counts for chi-squared calculation (curve_fit)
    expected_counts_cf = rayleigh_pdf(bin_centers, sigma_estimated_cf) * bin_widths * N
    valid_bins_cf = expected_counts_cf >= 5
    O_cf = hist_counts[valid_bins_cf]
    E_cf = expected_counts_cf[valid_bins_cf]
    weighted_average_variance_cf = 1 / (np.sum(1 / E_cf) / len(E_cf))
    chi_squared_cf = np.sum((O_cf - E_cf) ** 2 / weighted_average_variance_cf**2)
    dof_cf = len(O_cf) - 1
    reduced_chi_squared_cf = chi_squared_cf / dof_cf

    # Calculate expected counts for chi-squared calculation (MLE)
    expected_counts_mle = rayleigh_pdf(bin_centers, sigma_estimated_mle) * bin_widths * N
    valid_bins_mle = expected_counts_mle >= 5
    O_mle = hist_counts[valid_bins_mle]
    E_mle = expected_counts_mle[valid_bins_mle]
    dof_mle = len(O_mle) - 1
    weighted_average_variance_mle = 1/(np.sum(1 / E_cf) / len(E_mle))
    chi_squared_mle = np.sum((O_mle - E_mle) ** 2 / weighted_average_variance_mle ** 2)
    reduced_chi_squared_mle = chi_squared_mle / dof_mle

    residuals_cf = hist_counts - expected_counts_cf
    residuals_mle = hist_counts - expected_counts_mle
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1]})

    # Residuals for curve_fit
    ax1.errorbar(bin_centers[valid_bins_cf], residuals_cf[valid_bins_cf],
                 yerr=np.sqrt(O_cf), fmt='o', color='blue',
                 ecolor='lightblue', elinewidth=1, capsize=2, label='Residuals (curve_fit)')
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_ylabel('Residuals (Counts)')
    ax1.set_title(f'Residuals for Rayleigh Fit using curve_fit')
    ax1.legend()
    ax1.grid(True)

    # Residuals for MLE
    ax2.errorbar(bin_centers[valid_bins_mle], residuals_mle[valid_bins_mle],
                 yerr=np.sqrt(O_mle), fmt='o', color='green',
                 ecolor='lightgreen', elinewidth=1, capsize=2, label='Residuals (MLE)')
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel('Step size (µm)')
    ax2.set_ylabel('Residuals (Counts)')
    ax2.set_title(f'Residuals for Rayleigh Fit using MLE')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    # Print results with uncertainties and Reduced Chi-Squared
    print(f"Using curve_fit:")
    print(f"Estimated sigma (σ): {sigma_estimated_cf:.4f} µm ± {sigma_uncertainty_cf:.4f} µm")
    print(f"Estimated Diffusion Coefficient (D): {D_estimated_cf:.4e} µm²/s ± {delta_D_cf:.4e} µm²/s")
    print(f"Calculated Boltzmann Constant (k): {k_calculated_cf:.4e} J/K ± {delta_k_cf:.4e} J/K")
    print(f"Percent error compared to accepted value of k: {error_cf:.2f}%")
    print(f"Reduced Chi-Squared (curve_fit): {reduced_chi_squared_cf:.2f}\n")  # Added line

    print(f"Using Maximum Likelihood Estimation:")
    print(f"Estimated sigma (σ): {sigma_estimated_mle:.4f} µm ± {delta_sigma_mle:.4f} µm")
    print(f"Estimated Diffusion Coefficient (D): {D_estimated_mle:.4e} µm²/s ± {delta_D_mle:.4e} µm²/s")
    print(f"Calculated Boltzmann Constant (k): {k_calculated_mle:.4e} J/K ± {delta_k_mle:.4e} J/K")
    print(f"Percent error compared to accepted value of k: {error_mle:.2f}%")
    print(f"Reduced Chi-Squared (MLE): {reduced_chi_squared_mle:.2f}")

    return D_estimated_cf_SI, sigma_estimated_cf, sigma_uncertainty_cf, D_estimated_mle_SI, sigma_estimated_mle


if __name__ == '__main__':
    num_sets = 14

    all_step_sizes = analyze_all_sets(num_sets)

    # Only proceed if there are valid step sizes
    if len(all_step_sizes) > 0:
        D_estimated_cf_SI, sigma_estimated_cf, sigma_uncertainty_cf, D_estimated_mle_SI, sigma_estimated_mle = fit_rayleigh_distribution(all_step_sizes, 0.5)  # Time interval = 0.5s
    else:
        print("No valid data to fit Rayleigh distribution.")
