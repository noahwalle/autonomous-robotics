import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "lines.linewidth": 2,
    "lines.markersize": 7,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Load calibration data
r, z = np.loadtxt('partA/ankh.csv', skiprows=1, delimiter=',').T

DEAD_ZONE_THRESHOLD = 50

# Define linear sensor model
def h(x, a, b):
    return (a + b * x) * (x >= DEAD_ZONE_THRESHOLD)

# Fit model only to valid region
params, _ = curve_fit(h, r, z)
z_fit = h(r, *params)

# Calculate residuals on valid region
residuals = z - h(r, *params)

# Remove outliers from residuals (>2.5 std dev)
mask = np.abs(residuals) < 2 * np.std(residuals)
params_refined, _ = curve_fit(h, r[mask], z[mask])
residuals_refined = z[mask] - h(r[mask], *params_refined)
sensor_variance = np.var(residuals)

# Final model and fit
# z_fit_refined = h(r, *params_refined)

def main():
    # Print results
    print("Sensor model: z = {:.4f} + {:.4f} * r".format(*params))
    print("Estimated sensor noise variance:", sensor_variance)

    # Plot full data and fit
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Raw data and fit
    axs[0].plot(r, z, '.', label='Raw data')
    axs[0].plot(r, z_fit, '-', label='Fitted model', color='orange')
    axs[0].axhline(0, color='red', linestyle='--', label='Dead zone threshold')
    axs[0].set_ylabel('Sensor Output (z)')
    # axs[0].set_title('Ankh Sensor Calibration')
    axs[0].legend()

    # Residuals
    axs[1].plot(r[mask], residuals_refined, 'x', color='purple')
    axs[1].axhline(0, linestyle='--', color='black')
    axs[1].set_xlabel('Distance (m)')
    axs[1].set_ylabel('Residual')
    # axs[1].set_title('Residuals After Fitting')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()