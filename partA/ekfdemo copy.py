"""
Demo EKF skeleton.

M.P. Hayes UCECE

See ekf.py and section 11.4.1 in ENMT482 course reader for details
"""

import numpy as np
from ekf import EKF
import matplotlib.pyplot as plt
from nakh_plot import h, params, sensor_variance  # fixed import
from filter_test_data import t, z1, z2  # added import

# Load time, ground truth and sensor measurements
# t, z1, z2 = np.loadtxt('partA/test.csv', skiprows=1, delimiter=',').T

# Beacon positions
xr1, yr1 = 0.0, 0.0
xr2, yr2 = 1000.0, 0.0

# Invert calibrated sensor model h(r) -> z using bisection
def invert_h(zval, rmin=0.0, rmax=2000.0, tol=1e-6, maxit=60):
    # handle edge cases
    if zval <= h(rmin, *params):
        return rmin
    if zval >= h(rmax, *params):
        return rmax
    a, b = rmin, rmax
    fa, fb = h(a, *params) - zval, h(b, *params) - zval
    for _ in range(maxit):
        m = 0.5 * (a + b)
        fm = h(m, *params) - zval
        if abs(fm) < tol:
            return m
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)

# Trilateration for two circles (returns one solution)
def trilaterate(x0, y0, r0, x1, y1, r1):
    dx, dy = x1 - x0, y1 - y0
    d = np.hypot(dx, dy)
    if d == 0:
        return x0, y0
    # no solution: return point on line between beacons proportional to r0/(r0+r1)
    if d > r0 + r1 or d < abs(r0 - r1):
        t = r0 / max(r0 + r1, 1e-6)
        return x0 + t * dx, y0 + t * dy
    a = (r0**2 - r1**2 + d**2) / (2 * d)
    h = np.sqrt(max(0.0, r0**2 - a**2))
    xm = x0 + a * dx / d
    ym = y0 + a * dy / d
    rx = -dy * (h / d)
    ry = dx * (h / d)
    p1 = (xm + rx, ym + ry)
    p2 = (xm - rx, ym - ry)
    # choose the point with larger y (or closer to midpoint)
    midx, midy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    if np.hypot(p1[0] - midx, p1[1] - midy) < np.hypot(p2[0] - midx, p2[1] - midy):
        return p1
    return p2

# Use first sensor measurements to get initial position estimate
r1_0 = invert_h(z1[0])
r2_0 = invert_h(z2[0])
x_init, y_init = trilaterate(xr1, yr1, r1_0, xr2, yr2, r2_0)

# initial velocity set to zero (or estimate from data if available)
vx0, vy0 = 0.0, 0.0

# initial state and covariance
xv0 = np.array([x_init, y_init, vx0, vy0])
P0 = np.diag([0.5, 0.5, 1.0, 1.0])

# === Motion model matrix A ===
dt_0 = t[1] - t[0]
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

# === Build sensor-only positions for all timesteps (invert ranges -> trilaterate) ===
pos_est = []
for n in range(len(t)):
    r1n = invert_h(z1[n])
    r2n = invert_h(z2[n])
    xn, yn = trilaterate(xr1, yr1, r1n, xr2, yr2, r2n)
    pos_est.append((xn, yn))
pos_est = np.array(pos_est)  # shape (N,2)

# Measurements (keep as before)
measurements = np.column_stack((z1, z2))  # shape = (N, 2)

# === Process noise covariance matrix Q ===
# Estimate velocities/accelerations from sensor-only positions
if len(pos_est) > 2:
    vel = np.diff(pos_est, axis=0) / dt_0
    if len(vel) > 1:
        acc = np.diff(vel, axis=0) / dt_0
        # Build W as in EKF demo (shape 4 x K)
        W = np.vstack([
            0.5 * dt_0**2 * acc[:, 0],
            0.5 * dt_0**2 * acc[:, 1],
            dt_0 * acc[:, 0],
            dt_0 * acc[:, 1],
        ])
        # If enough samples compute covariance, else fallback
        if W.shape[1] >= 2:
            Q = np.cov(W, bias=False)
        else:
            Q = np.diag([1e-3, 1e-3, 1e-3, 1e-3])
    else:
        Q = np.diag([1e-3, 1e-3, 1e-3, 1e-3])
else:
    Q = np.diag([1e-3, 1e-3, 1e-3, 1e-3])

# small regularisation to keep Q positive definite
Q += np.eye(4) * 1e-9

ekf = EKF(xv0, P0)

# === Save history ===
xv_hist = np.zeros((N, 4))

# control vector (gv_func currently ignores uv, so zeros are fine)
uv = np.array([0.0, 0.0])

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
    # === Plot EKF vs sensor-only estimate ===
    plt.figure(figsize=(8, 5))
    plt.plot(pos_est[:,0], pos_est[:,1], label="Sensor-only estimate", color='green')
    plt.plot(xv_hist[:, 0], xv_hist[:, 1], label="EKF estimate", color='blue')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()