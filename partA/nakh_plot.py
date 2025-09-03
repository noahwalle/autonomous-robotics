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

# Load data using numpy (skip header)
r, z = np.loadtxt("partA/nakh.csv", delimiter=",", skiprows=1).T

DEAD_ZONE_THRESHOLD = 50

# Define linear sensor model
def h(r, a, b):
    return (a + b * r) * (r >= DEAD_ZONE_THRESHOLD)

params, _ = curve_fit(h, r, z)
z_fit = h(r, *params)

# Calculate residuals on valid region
residuals = z - h(r, *params)

# Remove outliers from residuals (>2.5 std dev)
mask = np.abs(residuals) < 4 * np.std(residuals)
params_refined, _ = curve_fit(h, r[mask], z[mask])
residuals_refined = z[mask] - h(r[mask], *params_refined)
sensor_variance = np.var(residuals_refined)

# Final model and fit
z_fit_refined = h(r, *params_refined)


if __name__ == "__main__":
    # Print results
    print("Sensor model: z = {:.4f} + {:.4f} * r".format(*params_refined))
    print("Estimated sensor noise variance:", sensor_variance)


    fig, ax = plt.subplots(2, sharex=True)
    ax[0].scatter(r, z, label="Z", color='blue', s=5)
    ax[0].scatter(r[mask], z[mask], label="Z", s=5)
    ax[0].plot(r, z_fit_refined, label="Z", color='orange', linewidth=1)
    ax[0].set_ylabel("z (m)")
    # ax[0].set_title("Drone Position")

    ax[1].scatter(r[mask], residuals_refined, label="Residuals", color='blue', s=5)
    ax[1].set_xlabel("r (m)")
    ax[1].set_ylabel("z (m)")

    plt.tight_layout()
    plt.show()