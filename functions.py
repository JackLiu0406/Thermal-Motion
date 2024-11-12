import numpy as np
import matplotlib.pyplot as plt

def linear_fit_with_uncertainty(x, y):
    N = len(x)
    # Calculate sums required for slope (m) and intercept (b)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x ** 2)
    sum_xy = np.sum(x * y)

    # Calculate the slope (m) and intercept (b)
    delta = N * sum_x2 - sum_x ** 2
    m = (N * sum_xy - sum_x * sum_y) / delta
    b = (sum_y * sum_x2 - sum_x * sum_xy) / delta

    # Calculate the variance of the y(x) values
    s2_yx = np.sum((y - (m * x + b)) ** 2) / (N - 2)

    # Calculate uncertainties in the slope and intercept
    sm = np.sqrt(N * s2_yx / delta)
    sb = np.sqrt(s2_yx * sum_x2 / delta)

    return m, sm, b, sb

def plot_fit_and_residuals(x, y, y_uncertainty, x_uncertainty, m, sm, b, sb):
    # Create figure with two subplots: one for the fit, one for the residuals
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(6, 8))

    # Plot the Time vs. Mean Squared Displacement with the best-fit line on the first subplot
    time_fit = np.linspace(min(x), max(x), 100)
    msd_fit = m * time_fit + b
    ax1.errorbar(x, y, xerr=x_uncertainty, yerr=y_uncertainty, fmt='o', label='Data with error bars')
    ax1.plot(time_fit, msd_fit, color='red', label=f'Fit: MSD = ({m:.2e} ± {sm:.2e})t + ({b:.2e} ± {sb:.2e})')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Mean Squared Displacement ($\mu m^2$)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Mean Squared Displacement vs. Time with Linear Fit')

    # Calculate residuals
    y_expected = m * x + b
    residuals = y - y_expected

    # Plot the residuals on the second subplot
    ax2.errorbar(x, residuals, xerr=x_uncertainty, yerr=y_uncertainty, fmt='o', label='Residuals with error bars')
    ax2.axhline(0, color='red', linestyle='--', label='Zero residual')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Residuals ($\mu m^2$)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Residuals')

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()

def calculate_chi_squared(x, y, y_uncertainty, x_uncertainty, m, b):
    # Calculate the expected y values
    y_expected = m * x + b

    # Combine uncertainties in x and y
    total_uncertainty = np.sqrt(y_uncertainty ** 2 + (m * x_uncertainty) ** 2)

    residuals = (y - y_expected) / total_uncertainty
    chi_squared = np.sum(residuals ** 2)

    # Degrees of freedom
    dof = len(y) - 2
    reduced_chi_squared = chi_squared / dof

    return chi_squared, reduced_chi_squared
