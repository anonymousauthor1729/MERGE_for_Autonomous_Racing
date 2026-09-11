import numpy as np
from os.path import join


#########################################################################################################
# GLOBAL CONFIGURATION
# This file contains simulation constants, vehicle parameters, and controller gains.
#########################################################################################################

# --- File I/O Settings ---
# Timestamp used for versioning saved trajectory files
FILEDATE = "2026-02-03_18-43"

# Default path to the "Expert/Optimal" trajectory pickle file & Map csv file
TRAJ_DATA_PATH = join("data_gym_ros_exp", f"TDList_Learnt_{FILEDATE}.pkl")
SAVE_DATA_PATH = join("data_gym_ros_exp", f"TDList_Learnt_{FILEDATE}_GYM_ROS.pkl")
RACE_TRACK_PATH = join("f1tenth_racetracks", "Oschersleben", "Oschersleben_centerline.csv")

# --- Visualization Settings ---
# LaTeX-formatted labels for the 7-state vehicle model vector
# Used by matplotlib for axis labeling in 'trajectory_handler.py'
STATE_LABELS = [
    r"$X$ [m]",              # X Position
    r"$Y$ [m]",              # Y Position
    r"$\delta$ [rad]",       # Steering Angle
    r"$v$ [m/s]",            # Longitudinal Velocity
    r"$\psi$ [rad]",         # Yaw Angle
    r"$\dot{\psi}$ [rad/s]", # Yaw Rate
    r"$\beta$ [rad]"         # Vehicle Slip Angle
]

STATE_NAMES = [
    "X Coordinate", 
    "Y Coordinate", 
    "Steering Angle", 
    "Linear Velocity",
    "Orientation", 
    "Angular Velocity",
    "Slip"
]

# --- Controller Tuning Parameters ---
# Gains and thresholds for the Switching Stanley Controller.
CONTROLLER_PARAMS = {
    'lf': 0.15875,  # Distance from Center of Gravity (CG) to Front Axle [m]
 
    # Dynamic Lookahead: Lookahead distance = Factor * Velocity
    # Increase this if the car oscillates at high speeds.
    "look_ahead_progress_factor": 1e-3,

    # Lookahead factor for velocity (multiplied with the above value)
    "vel_look_ahead_factor": 0.5,

    # Stanley Gain (k): Determines how aggressively the car corrects Cross-Track Error (CTE).
    # Higher k = faster convergence to path but potential for oscillation.
    # "k_stanley_factor": 9.5e-2,
    "k_stanley_factor": 1e-3,

    # Low Velocity Threshold [m/s]
    # The Stanley controller has a singularity at v=0 (division by zero).
    # Steering commands are suppressed below this speed.
    "velocity_threshold": 1e-2
}

#########################################################################################################
#########################################################################################################