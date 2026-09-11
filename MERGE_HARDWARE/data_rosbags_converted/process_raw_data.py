#!/usr/bin/env python3

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from collections import defaultdict

import numpy as np
from pathlib import Path
from os.path import join
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R

from trajectory_handler import WaypointTrajectoryUtils, \
    traj_data_dict_get, save_traj_data, load_traj_data


#########################################################################################################
#########################################################################################################


def find_rosbags(root_dir):
    """
    Recursively scans the root directory to find all ROS 2 bag directories containing SQLite3 storage (.db3).
    """
    root = Path(root_dir)
    bags = []

    # Iterate over subdirectories and check for SQLite3 rosbag storage files
    for d in root.iterdir():
        if d.is_dir() and any(f.suffix == ".db3" for f in d.iterdir()):
            bags.append(str(d))
    return bags


def read_ros2_bag(bag_path: str):
    """
    Opens and deserializes a ROS 2 bag file using the sequential reader API.
    Extracts all messages grouped by topic along with timestamps converted to seconds.
    """
    reader = rosbag2_py.SequentialReader()

    # Configure the underlying SQLite3 storage reader
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id="sqlite3"
    )

    # Configure Common Data Representation (CDR) deserialization options
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )

    reader.open(storage_options, converter_options)
    
    # Retrieve active topic names and message types to look up ROS 2 interface classes
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    data = defaultdict(list)
    
    # Iterate through all serialized records and deserialize dynamically
    while reader.has_next():
        topic, raw_data, t = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(raw_data, msg_type)
        # Convert timestamp from nanoseconds to fractional seconds
        data[topic].append((t * 1e-9, msg))
    return data


def read_map_data(map_path: str, csv_filename: str = None) -> WaypointTrajectoryUtils:
    """
    Loads raw track geometry and lane boundary widths from a MATLAB .mat structure,
    instantiates the WaypointTrajectoryUtils spline parameterizer, and optionally exports to CSV.
    """
    track_mat_data = loadmat(map_path)
    
    # Extract track 2D centerline coordinates (transposing to shape: N, 2)
    waypoints_xy = track_mat_data['track'][0][0][0].T
    
    # Extract maximum lateral lane clearance across both track boundary sides
    waypoints_width = np.max(np.column_stack((
        np.squeeze(track_mat_data['track'][0][0][3]),
        np.squeeze(track_mat_data['track'][0][0][4])
    )), axis=-1)

    # Initialize spline interpolation and progress projection utility
    WT = WaypointTrajectoryUtils(
        waypoint_density=10.,
        waypoints=waypoints_xy,
        waypoints_width=waypoints_width,
        smoothness_params=np.full(2, 0.1)
    )

    # Optionally persist interpolated centerline waypoints as CSV for ROS deployment
    if csv_filename is not None:
        np.savetxt(f"{csv_filename}", waypoints_xy, fmt="%.15f", 
                   delimiter=",", header="x_m\ty_m", comments="")
    return WT


#########################################################################################################
#########################################################################################################


def extract_amcl_xyyaw(amcl_data):
    """
    Extracts timestamps, 2D coordinates (x, y), and converts 3D quaternion orientations
    from AMCL pose estimates into planar Euler yaw angles (psi).
    """
    out = []

    for t, msg in amcl_data:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        # Convert orientation quaternion [x, y, z, w] to intrinsic Euler angle (yaw)
        yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
        out.append([t, p.x, p.y, yaw])

    return np.array(out)


def extract_odom_velocity(odom_data):
    """
    Extracts timestamps, longitudinal forward speed (v), and body-frame yaw rates (omega)
    from filtered odometry messages.
    """
    out = []

    for t, msg in odom_data:
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        out.append([t, v, w])

    return np.array(out)


#########################################################################################################
#########################################################################################################        


def interpolate_trajectory_data(amcl_xyyaw: np.ndarray, filt_vel: np.ndarray):
    """
    Resamples asynchronous pose estimation (AMCL) and velocity telemetry (Odometry)
    onto a unified time array, builds complete 7-dimensional vehicle state matrices,
    and computes track progress and cross-track deviations.
    """
    # Identify the overlapping global temporal boundaries
    t_start = min(amcl_xyyaw[0, 0], filt_vel[0, 0])
    t_end = max(amcl_xyyaw[-1, 0], filt_vel[-1, 0])
    num_t_steps = max(amcl_xyyaw.shape[0], filt_vel.shape[0])
    t_steps = np.linspace(t_start, t_end, num_t_steps)
    
    TD = traj_data_dict_get()
    TD["t_steps"] = t_steps - t_steps[0]   # Zero-align time to start of recorded run
    TD["states"] = np.zeros((num_t_steps, 7))
    
    # 1D linear temporal interpolation of discrete telemetry channels
    TD["states"][:, 0] = np.interp(t_steps, amcl_xyyaw[:, 0], amcl_xyyaw[:, 1])  # X [m]
    TD["states"][:, 1] = np.interp(t_steps, amcl_xyyaw[:, 0], amcl_xyyaw[:, 2])  # Y [m]
    TD["states"][:, 3] = np.interp(t_steps, filt_vel[:, 0], filt_vel[:, 1])        # v [m/s]
    TD["states"][:, 4] = np.interp(t_steps, amcl_xyyaw[:, 0], amcl_xyyaw[:, 3])  # Yaw (psi) [rad]
    TD["states"][:, 5] = np.interp(t_steps, filt_vel[:, 0], filt_vel[:, 2])        # Yaw rate (psi_dot) [rad/s]
    
    # Project positions onto the centerline to determine normalized progress s in [0, 1] and tracking deviations
    TD["progress"], closest_points = WT.find_progress_value(TD["states"][:, :2])
    TD["deviations"] = np.linalg.norm(TD["states"][:, :2] - closest_points, axis=-1)
    
    # Cast output values to NumPy arrays for compatibility with downstream optimization
    TD["t_steps"] = np.array(TD["t_steps"])
    TD["states"] = np.array(TD["states"])
    TD["progress"] = np.array(TD["progress"])
    TD["deviations"] = np.array(TD["deviations"])
    return TD


def create_trajectory_data(WT: WaypointTrajectoryUtils, bag_path: str, plot_flag: bool = True):
    """
    Extracts trajectory data from a single ROS 2 bag, aligns telemetry, identifies wrap-around points
    where lap progress jumps (s ~ 1.0 -> 0.0), splits the session into individual closed-circuit laps,
    and logs lap time and velocity statistics.
    """
    # Read raw bag topics and extract kinematic state channels
    data = read_ros2_bag(bag_path)
    amcl_xyyaw = extract_amcl_xyyaw(data["/amcl_pose"])
    filt_vel   = extract_odom_velocity(data["/odometry/filtered"])
    TD_All = interpolate_trajectory_data(amcl_xyyaw, filt_vel)

    # Detect lap boundary wrap-around transitions where progress difference exceeds threshold (0.75)
    splitting_indices = np.argwhere(
        np.abs(np.diff(TD_All["progress"])) > 0.75
    ).squeeze()
    splitting_indices = np.clip(
        splitting_indices + 1, 0, len(TD_All["t_steps"])
    )
    
    lap_times = []
    TDList_Demo = []
    # Account for single-lap or multi-lap segment arrays
    num_splits = len(splitting_indices) if splitting_indices.ndim > 0 else (1 if splitting_indices.size > 0 else 0)

    # Slice unified session recording into individual laps
    for i in range(num_splits + 1):
        start = 0 if i == 0 else (splitting_indices[i-1] if splitting_indices.ndim > 0 else int(splitting_indices))
        stop = -1 if i == num_splits else (splitting_indices[i] if splitting_indices.ndim > 0 else int(splitting_indices))
        TD = traj_data_dict_get()

        TD["t_steps"] = TD_All["t_steps"][start:stop]
        TD["progress"] = TD_All["progress"][start:stop]
        TD["states"] = TD_All["states"][start:stop]
        TD["deviations"] = TD_All["deviations"][start:stop]
        
        # Zero-align time baseline for each isolated lap
        TD["t_steps"] -= TD["t_steps"][0]
        TDList_Demo.append(TD)
        
        # Exclude partial warm-up or cooldown transitions from lap-time benchmark statistics
        if 0 < i < num_splits: 
            lap_times.append(np.max(TD["t_steps"]))

    # Optional trajectory visualization
    if plot_flag:
        WT.plot_trajectory_data(
            TDList_demo=TDList_Demo,
            x_axis_time=False,
        )

    # Print summary telemetry statistics
    print(f"[INFO]: Speed Stats -> Max = {round(np.max(TD_All['states'][:, 3]), 4)} m/s")
    print(f"[INFO]: Speed Stats -> Mean = {round(np.mean(TD_All['states'][:, 3]), 4)} m/s")
    if len(lap_times) > 0:
        print(f"[INFO]: Lap Time Stats -> Best = {round(np.min(lap_times), 4)} s")
        print(f"[INFO]: Lap Time Stats -> Mean = {round(np.mean(lap_times), 4)} s")
    print("")
    return TDList_Demo


#########################################################################################################
#########################################################################################################


if __name__ == "__main__":

    # Configure input directories and target map paths
    BAG_PATH = "DEMO_BAGS"
    MAP_PATH = join("MAP_DATA", "Track_Data_Jan_21.mat")
    
    TDList_Demo = []
    
    # Initialize track geometry and export normalized centerline progress model
    WT = read_map_data(
        map_path=MAP_PATH, 
        csv_filename=join("MAP_DATA", "Track_Data_Jan_21.txt")
    )

    # Process all demonstration rosbags and compile isolated lap trajectories
    for bag_path in find_rosbags(BAG_PATH):
        TDList_Demo += create_trajectory_data(
            WT, bag_path, plot_flag=False
        )
    
    # Save the processed list of demonstration laps to disk using pickle
    save_traj_data(
        TDList=TDList_Demo, 
        filepath=join("DEMO_BAGS", "Track_Data_Jan_21.pkl")
    )

    # Render spatial trajectories and state evolutions across all extracted demonstration laps
    WT.plot_trajectory_data(
        TDList_demo=TDList_Demo,
        x_axis_time=False,
    )


#########################################################################################################
#########################################################################################################