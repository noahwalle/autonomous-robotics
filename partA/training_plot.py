import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

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

# Load data using numpy (skip header)
time, x, y, z1, z2 = np.loadtxt("partA/training.csv", delimiter=",", skiprows=1).T

# Compute time steps
dt = np.diff(time)
dx = np.diff(x)
dy = np.diff(y)

dt_0 = time[1] - time[0]

# Compute velocity and acceleration using dt_0
pos = np.column_stack((x, y))
vel = np.diff(pos, axis=0) / dt_0
acc = np.diff(vel, axis=0) / dt_0

# Stack process noise components as in ekfdemo.py
W = np.vstack([
    0.5 * dt_0**2 * acc[:,0],
    0.5 * dt_0**2 * acc[:,1],
    dt_0 * acc[:,0],
    dt_0 * acc[:,1],
])

Q = np.cov(W.T)

def g(X):
    return np.array([X[0] + X[0] * dt,
                     X[1] + X[1] * dt,
                     X[2],
                     X[3]])

A_matrices = np.array([
    [[1, 0, dt_i, 0],
     [0, 1, 0, dt_i],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]
    for dt_i in dt
])

Xn = np.vstack([x[:-1], y[:-1], dx/dt, dy/dt]).T
Xn_1 = np.vstack([x[:-2], y[:-2], dx[:-1]/dt[:-1], dy[:-1]/dt[:-1]]).T

g_Xn_1 = np.array([A_matrices[i] @ Xn_1[i] for i in range(len(Xn_1))])

W = Xn[1:] - g_Xn_1

d = np.sqrt(x**2 + y**2)
v = np.diff(d) / dt
a = np.diff(v) / dt[1:]

# acceleration variance
sigma_a2 = np.var(a)

Q = W @ W.T

# Use EKF motion model for prediction
def gv_func(xv, dt):
    x, y, dx, dy = xv
    return np.array([
        x + dx * dt,
        y + dy * dt,
        dx,
        dy
    ])

X = np.vstack([x[:-1], y[:-1], dx/dt, dy/dt]).T  # shape (N-1, 4)

# Predict next state using EKF motion model
X_pred = np.array([gv_func(X[i], dt[i]) for i in range(len(X))])

speed_model = np.sqrt(X_pred[:,2]**2 + X_pred[:,3]**2)
speed_est = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt

# Add random noise to every entry in X_pred
noise = np.random.normal(0, 10, X_pred.shape)
X_pred_noisy = X_pred + noise
speed_model_noisy = np.sqrt(X_pred_noisy[:,2]**2 + X_pred_noisy[:,3]**2)

def main():
    print(f"Process noise covariance Q:\n{Q}")
    print("\nAcceleration variance o^2: {:.6f} m^2/s^4".format(sigma_a2))

    plt.figure(figsize=(7, 4))
    # Plot measured path
    plt.plot(x, y, label="Measured Path (data)", color='blue')
    # Plot predicted noisy path
    plt.plot(X_pred_noisy[:,0], X_pred_noisy[:,1], label="Predicted Path (model)", color='orange')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()