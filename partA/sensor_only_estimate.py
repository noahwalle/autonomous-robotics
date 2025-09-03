import numpy as np
import matplotlib.pyplot as plt
from ankh_plot import params
import nakh_plot
import filter_test_data
# Beacon positions (mm)
b1 = np.array([0.0, 0.0])
b2 = np.array([1000.0, 0.0])

def trilaterate(z1, z2, params, prev_pos=None):
    b_cal, a_cal = params
    x1, y1 = b1
    x2, y2 = b2
    r1 = (z1 - b_cal) / a_cal # inverse sensor
    r2 = (z2 - b_cal) / a_cal # inverse sensor

    d = np.hypot(x2 - x1, y2 - y1) # distance between beacons

    if d < 1e-9:
        return (np.nan, np.nan), 'degenerate_centers'

    if d > r1 + r2 + 1e-9 or d < abs(r1 - r2) - 1e-9:
        t = np.clip(r1 / d, 0.0, 1.0)
        proj = (x1+t*(x2-x1), y1+t*(y2-y1))
        return proj, 'no_intersection_projected'

    a = (r1**2 - r2**2 + d**2) / (2 * d)

    xm = x1 + a * (x2 - x1) / d
    ym = y1 + a * (y2 - y1) / d

    h_sq = max(0.0, r1**2 - a**2)
    if h_sq == 0.0:
        return (xm, ym), 'one_intersection'
    
    h = np.sqrt(h_sq)

    rx = -(y2 - y1) * (h / d)
    ry =  (x2 - x1) * (h / d)

    sol1 = np.array([xm + rx, ym + ry])
    sol2 = np.array([xm - rx, ym - ry])

    if prev_pos is not None:
        prev = np.array(prev_pos)
        dA = np.linalg.norm(sol1 - prev)
        dB = np.linalg.norm(sol2 - prev)
        result = sol1 if dA <= dB else sol2
        return (result[0], result[1]), 'chosen_by_prev'
    else:
        result = sol1 if sol1[1] >= sol2[1] else sol2
        return (result[0], result[1]), 'chosen_by_y'
        

# Load training data
time, x, y, z1, z2 = np.loadtxt("partA/training.csv", delimiter=",", skiprows=1).T
prev_pos = np.array([200.0, 100.0])  # starting guess

result = []
messages = []

z1 = filter_test_data.z1
z2 = filter_test_data.z2
prev_pos = None

# Estimate drone position at each timestep
for i in range(len(z1)):
    (xs, ys), msg = trilaterate(z1[i], z2[i], nakh_plot.params, prev_pos=prev_pos)
    result.append((xs, ys))
    messages.append(msg)
    prev_pos = (xs, ys)

result = np.array(result)

def main():
    plt.figure(figsize=(8, 5))
    plt.plot(result[:,0], result[:,1], label="Sensor-only path", linewidth=1.5)
    # plt.plot(x, y, label="Ground Truth", linewidth=1.5)
    plt.scatter(*b1, color='red', label='Beacon 1 (0,0)')
    plt.scatter(*b2, color='green', label='Beacon 2 (1000,0)')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    # plt.title("Drone Path from Sensor-only Trilateration")
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()