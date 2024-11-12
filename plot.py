import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
# data = pd.read_csv('/Users/dingshengliu/Desktop/EngSci/Y2 S1/Physics Lab/V_I_Data_Op1csv.csv')
data = pd.read_csv('/Users/dingshengliu/Desktop/EngSci/Y2 S1/Physics Lab/V_I_Data_Op1csv.csv')

# Extract values
x = data['x'].values.reshape(-1, 1) * 1000  # Convert x to mA
y = data['y'].values  # y values in V
uncertainty_x = data['uncertainty_x'].values * 1000  # Uncertainty in x in mA
uncertainty_y = data['uncertainty_y'].values  # Uncertainty in y in V

# Calculate weights for weighted linear regression
weights = 1 / uncertainty_y

# Perform weighted linear regression
coeffs, cov = np.polyfit(x.flatten(), y, deg=1, w=weights, cov=True)
slope = coeffs[0]
intercept = coeffs[1]

# Calculate uncertainties in slope and intercept from covariance matrix
uncertainty_slope = np.sqrt(cov[0, 0])
uncertainty_intercept = np.sqrt(cov[1, 1])

# Generate predicted y values
y_pred = slope * x.flatten() + intercept

# Calculate residuals
residuals = y - y_pred

# Calculate chi-squared
chi2 = np.sum(((y - y_pred) / uncertainty_y) ** 2)

# Calculate R-squared value
r2 = np.corrcoef(y, y_pred)[0, 1] ** 2

# Plot the data with error bars and the line of best fit
plt.figure(figsize=(10, 6))
plt.errorbar(x.flatten(), y, xerr=uncertainty_x, yerr=uncertainty_y, fmt='o',
             color='blue', ecolor='gray', elinewidth=1, capsize=3, label='Data Points')
plt.plot(x, y_pred, color='red', label='Line of Best Fit')

# Adding labels and title
plt.xlabel('Current (mA)')
plt.ylabel('Voltage (V)')
plt.title('Voltage vs. Current for Circuit Option 1')
plt.legend()

# Display slope and intercept with uncertainties on the graph
textstr = (
    f'Slope: {slope * 1000:.2f} ± {uncertainty_slope * 1000:.2f} V/A\n'
    f'Intercept: {intercept:.3f} ± {uncertainty_intercept:.3f} V\n'
    f'Chi-squared: {chi2:.2f}'
)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
plt.grid()
plt.show()

# Plot the residuals
plt.figure(figsize=(10, 6))
plt.errorbar(x.flatten(), residuals, yerr=uncertainty_y, fmt='o',
             color='green', ecolor='gray', elinewidth=1, capsize=3, label='Residuals')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Current (mA)')
plt.ylabel('Residuals (V)')
plt.title('Residuals of Voltage vs. Current Fit')
plt.legend()
plt.grid()
plt.show()

# Print the calculated values
print(f"Slope: {slope * 1000:.2f} ± {uncertainty_slope * 1000:.2f} V/A")
print(f"Intercept: {intercept:.4f} ± {uncertainty_intercept:.4f} V")
print(f"R-squared: {r2:.3f}")
print(f"Chi-squared: {chi2:.2f}")