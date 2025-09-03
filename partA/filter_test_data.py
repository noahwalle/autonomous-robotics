import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from nakh_plot import h, params
    
t, z1, z2 = np.loadtxt("partA/test.csv", delimiter=",", skiprows=1).T

for i in range(len(z1)-1):
    if abs(z1[i+1] - z1[i]) > 5:
        z1[i] = z1[i-1]
    if abs(z2[i+1] - z2[i]) > 5:
        z2[i] = z2[i-1]

if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.scatter(t, z1, label="Z1", color='blue', s=5)
    ax.scatter(t, z2, label="Z2", color='orange', s=5)
    plt.show()