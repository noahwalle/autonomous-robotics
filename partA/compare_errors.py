import numpy as np
import matplotlib.pyplot as plt
import ekfdemo
import sensor_only_estimate
import dead_reckoning

plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 6,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "lines.linewidth": 1,
    "lines.markersize": 7,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

fig, ax = plt.subplots(3, sharex=True)

time_steps = np.arange(len(ekfdemo.xv_hist))
e_x_EKF = ekfdemo.xv_hist[:, 0] - ekfdemo.x
e_y_EKF = ekfdemo.xv_hist[:, 1] - ekfdemo.y

e_x_SO = sensor_only_estimate.result[:, 0] - ekfdemo.x
e_y_SO = sensor_only_estimate.result[:, 1] - ekfdemo.y

e_x_DR = dead_reckoning.dead_x - ekfdemo.x
e_y_DR = dead_reckoning.dead_y - ekfdemo.y

ax[0].plot(time_steps, e_x_EKF, label='EKF')
ax[0].plot(time_steps, e_x_SO, label='Sensor-only')
ax[0].plot(time_steps, e_x_DR, label='Dead Reckoning')
ax[0].set_ylabel('Error x [m]')
# ax[0].set_title("X Error")
ax[0].legend()


ax[1].plot(time_steps, e_y_EKF, label='EKF')
ax[1].plot(time_steps, e_y_SO, label='Sensor-only')
ax[1].plot(time_steps, e_y_DR, label='Dead Reckoning')
ax[1].set_ylabel('Error y [m]')
# ax[1].set_title("Y Error")
ax[1].legend()

euclid_EKF = np.sqrt(e_x_EKF ** 2 + e_y_EKF ** 2)
euclid_SO = np.sqrt(e_x_SO ** 2 + e_y_SO ** 2)
euclid_DR = np.sqrt(e_x_DR ** 2 + e_y_DR ** 2)

ax[2].plot(time_steps, euclid_EKF, label='EKF')
ax[2].plot(time_steps, euclid_SO, label='Sensor-only')
ax[2].plot(time_steps, euclid_DR, label='Dead Reckoning')
ax[2].set_xlabel('Time [s]')
ax[2].set_ylabel('Error [m]')
# ax[2].set_title("Error Magnitude")
ax[2].legend()

plt.show()