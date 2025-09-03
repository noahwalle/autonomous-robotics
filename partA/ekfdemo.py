"""
Demo EKF skeleton.

M.P. Hayes UCECE

See ekf.py and section 11.4.1 in ENMT482 course reader for details
"""

import numpy as np
from ekf import EKF
import matplotlib.pyplot as plt
from ankh_plot import h, params, sensor_variance

t, x, y, z1, z2 = np.loadtxt('partA/training.csv', skiprows=1, delimiter=',').T

# Beacon positions
xr1, yr1 = 0, 0
xr2, yr2 = 1000, 0

# Define initial mean of state vector xv0
# Define initial covariance of state vector P0
# Define motion model matrix A

# === Initial state ===
dt_0 = t[1] - t[0]
dx_0 = x[1] - x[0]
dy_0 = y[1] - y[0]
theta = np.arctan2(dy_0, dx_0)

xv0 = np.array([x[0], y[0], dx_0 / dt_0, dy_0 / dt_0])
P0 = np.diag([0.01, 0.01, 0.01, 0.01])  # position + velocity cov

# === Motion model matrix A ===
A = np.array([
    [1, 0, dt_0, 0],
    [0, 1, 0, dt_0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])


def gv_func(xv, uv):
    """
    Motion model

    Parameters
    ----------
    `xv` state vector
    `uv` control vector

    Returns
    -------
    `xvp` predicted state vector

    """
    # Define motion model
    x, y, dx_0, dy_0 = xv
    return np.array([
        x + dx_0 * dt_0,
        y + dy_0 * dt_0,
        dx_0,
        dy_0])



def hv_func(xv):
    """
    Sensor model in terms of state vector

    Parameters
    ----------
    `xv` state vector

    Returns
    -------
    `zv` predicted measurement vector

    """
    # Define sensor model
    x, y, dx, dy = xv
    r1 = np.sqrt((x - xr1)**2 + (y - yr1)**2)
    r2 = np.sqrt((x - xr2)**2 + (y - yr2)**2)

    z1 = h(r1, *params)
    z2 = h(r2, *params)

    return np.array([z1, z2])


def C_func(xv):
    """
    Sensor model Jacobian evaluated at state estimate

    Parameters
    ----------
    `xv` state vector estimate to linearise about

    Returns
    -------
    `C` Jacobian matrix evaluated at state estimate

    """
    # Calculate Jacobian matrix to linearise sensor model
    x, y, dx, dy = xv
    dx_01, dy_01 = x - xr1, y - yr1
    dx_02, dy_02 = x - xr2, y - yr2
    r1 = np.sqrt(dx_01**2 + dy_01**2)
    r2 = np.sqrt(dx_02**2 + dy_02**2)
    return np.array([
        [dx_01 / r1, dy_01 / r1, 0, 0],
        [dx_02 / r2, dy_02 / r2, 0, 0]])


N = len(t)

# Measurements
measurements = np.array((z1, z2))
measurements = np.column_stack((z1, z2))  # shape = (N, 2)

# There is no control input
uv = 0

ekf = EKF(xv0, P0)

# === Save history ===
xv_hist = np.zeros((N, 4))

# === Process noise covariance matrix Q ===
pos = np.column_stack((x, y))
vel = np.diff(pos, axis=0) / dt_0
acc = np.diff(vel, axis=0) / dt_0

W = np.vstack([
    0.5 * dt_0**2 * acc[:,0],
    0.5 * dt_0**2 * acc[:,1],
    dt_0 * acc[:,0],
    dt_0 * acc[:,1],
])

Q = W @ W.T

for n in range(N):
    ekf.predict(uv, A, gv_func, Q)

    zv = measurements[n]

    # TODO: Handle measurement outliers

    # === Sensor noise R ===
    R = np.diag([sensor_variance, sensor_variance])

    C = C_func(ekf.xv)
    ekf.update(zv, C, hv_func, R)

    # Save history of state mean and covariance for plotting
    xv_hist[n] = ekf.xv

def main():
    print(xv_hist[-1]) # Final state estimate
    print(ekf.P)      # Final state covariance
    # === Plot EKF vs ground truth ===
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="Ground truth", color='green')
    plt.plot(xv_hist[:, 0], xv_hist[:, 1], label="EKF estimate", color='blue')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    # plt.title("EKF Estimated Path vs Ground Truth")
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()