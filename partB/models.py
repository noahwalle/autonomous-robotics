"""Particle filter sensor and motion model implementations.

M.P. Hayes and M.J. Edwards,
Department of Electrical and Computer Engineering
University of Canterbury
"""

import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan2, sqrt, exp
from numpy import random
from utils import gauss, wraptopi, angle_difference


def motion_model(particle_poses, speed_command, odom_pose, odom_pose_prev, dt):
    """Apply motion model and return updated array of particle_poses.

    Parameters
    ----------

    particle_poses: an M x 3 array of particle_poses where M is the
    number of particles.  Each pose is (x, y, theta) where x and y are
    in metres and theta is in radians.

    speed_command: a two element array of the current commanded speed
    vector, (v, omega), where v is the forward speed in m/s and omega
    is the angular speed in rad/s.

    odom_pose: the current local odometry pose (x, y, theta).

    odom_pose_prev: the previous local odometry pose (x, y, theta).

    dt is the time step (s).

    Returns
    -------
    An M x 3 array of updated particle_poses.

    """

    M = particle_poses.shape[0]
    v, omega = speed_command

    # Process noise 
    noise_std_x = 0.01
    noise_std_y = 0.01
    noise_std_theta = 0.01

    # Compute odometry delta in world frame
    dx_world = odom_pose[0] - odom_pose_prev[0]
    dy_world = odom_pose[1] - odom_pose_prev[1]
    dtheta_odom = angle_difference(odom_pose[2], odom_pose_prev[2])
    theta_odom_prev = odom_pose_prev[2]

    #Rotate world delta into odometry (robot-local) frame
    dx_odom = cos(-theta_odom_prev) * dx_world - sin(-theta_odom_prev) * dy_world
    dy_odom = sin(-theta_odom_prev) * dx_world + cos(-theta_odom_prev) * dy_world

    #Apply motion to each particle
    for m in range(M):
        x, y, theta = particle_poses[m]

        # Rotate odometry delta into particle's frame
        dx_global = cos(theta) * dx_odom - sin(theta) * dy_odom
        dy_global = sin(theta) * dx_odom + cos(theta) * dy_odom

        # Add process noise
        dx_noisy = dx_global + random.normal(0, noise_std_x)
        dy_noisy = dy_global + random.normal(0, noise_std_y)
        dtheta_noisy = dtheta_odom + random.normal(0, noise_std_theta)

        # Update particle pose
        particle_poses[m, 0] = x + dx_noisy
        particle_poses[m, 1] = y + dy_noisy
        particle_poses[m, 2] = wraptopi(theta - dtheta_noisy)

    return particle_poses


def sensor_model(particle_poses, beacon_pose, beacon_loc):
    """Apply sensor model and return particle weights.

    Parameters
    ----------
    
    particle_poses: an M x 3 array of particle_poses (in the map
    coordinate system) where M is the number of particles.  Each pose
    is (x, y, theta) where x and y are in metres and theta is in
    radians.

    beacon_pose: the measured pose of the beacon (x, y, theta) in the
    robot's camera coordinate system.

    beacon_loc: the pose of the currently visible beacon (x, y, theta)
    in the map coordinate system.

    Returns
    -------
    An M element array of particle weights.  The weights do not need to be
    normalised.

    """

    M = particle_poses.shape[0]
    particle_weights = np.zeros(M)
    
    # TODO.  For each particle calculate its weight based on its pose,
    # the relative beacon pose, and the beacon location.

    # Sensor noise standard deviations
    beacon_std_x = 0.1
    beacon_std_y = 0.1
    beacon_std_theta = 0.1

    for m in range(M):
        x, y, theta = particle_poses[m]

        # Transform beacon location from map frame to robot (particle) frame
        dx = beacon_loc[0] - x
        dy = beacon_loc[1] - y
        # Rotate into particle's frame
        beacon_pred_x = cos(-theta) * dx - sin(-theta) * dy
        beacon_pred_y = sin(-theta) * dx + cos(-theta) * dy
        beacon_pred_theta = wraptopi(beacon_loc[2] - theta)

        # Compute error between predicted and measured beacon pose
        err_x = beacon_pose[0] - beacon_pred_x
        err_y = beacon_pose[1] - beacon_pred_y
        err_theta = angle_difference(beacon_pose[2], beacon_pred_theta)

        # Weight: product of Gaussians for each component
        w_x = gauss(err_x, beacon_std_x)
        w_y = gauss(err_y, beacon_std_y)
        w_theta = gauss(err_theta, beacon_std_theta)
        particle_weights[m] = w_x * w_y * w_theta

    return particle_weights
