import numpy as np
import matplotlib.pyplot as plt

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

# Load training data
time, x, y, z1, z2 = np.loadtxt("partA/training.csv", delimiter=",", skiprows=1).T

# Initial position
x0, y0 = x[0], y[0]

# Initial speed estimate
dt = time[1] - time[0]
dx = x[1] - x[0]
dy = y[1] - y[0]
v0 = np.sqrt(dx**2 + dy**2) / dt # - 0.55
heading = np.arctan2(dy, dx)

# Dead-reckoning loop
dead_x = [x0]
dead_y = [y0]

for i in range(1, len(time)):
    dt = time[i] - time[i - 1]
    x_new = dead_x[-1] + v0 * dt * np.cos(heading)
    y_new = dead_y[-1] + v0 * dt * np.sin(heading)
    dead_x.append(x_new)
    dead_y.append(y_new)

def main():
    print(f"Initial position: ({x0:.2f}, {y0:.2f})")
    print(f"Initial speed: {v0:.4f} mm/s, heading: {np.degrees(heading):.2f}°")
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="Ground Truth", color='green', alpha=0.6)
    plt.plot(dead_x, dead_y, label="Dead Reckoning", linestyle='--', color='orange')
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    # plt.title("Dead-Reckoning Path vs Ground Truth")
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()