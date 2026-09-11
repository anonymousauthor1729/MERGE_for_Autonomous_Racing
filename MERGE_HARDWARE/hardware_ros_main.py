import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseWithCovarianceStamped, Pose

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import numpy as np
from scipy.interpolate import splev, splrep
from scipy.spatial.transform import Rotation as R

from hardware_ros_config import *
from trajectory_utils.waypoint import WayPoint
from trajectory_utils.data import traj_data_dict_get, save_traj_data, load_traj_data, print_details


#########################################################################################################
#########################################################################################################

class SwitchingController(Node):
    """
    ROS2 node implementing a Stanley-based trajectory tracking controller
    with velocity and look-ahead switching based on vehicle speed.
    Tailored for real-time deployment on physical F1TENTH hardware.
    """

    def __init__(
        self,
        wheelbase_front=CONTROLLER_PARAMS["lf"],
        k_stanley_factor=CONTROLLER_PARAMS["k_stanley_factor"],
        velocity_threshold=CONTROLLER_PARAMS["velocity_threshold"],
        vel_look_ahead_factor=CONTROLLER_PARAMS["vel_look_ahead_factor"],
        look_ahead_progress_factor=CONTROLLER_PARAMS["look_ahead_progress_factor"],
    ):
        """
        Initialize controller parameters, low-latency QoS profiles, ROS interfaces, and internal state.
        """
        super().__init__("switching_controller_node")

        # --- Controller Kinematic & Tuning Parameters ---
        self.lf = wheelbase_front                                   # Distance from center of mass to front axle (m)
        self.vel_thresh = velocity_threshold                       # Deadband velocity threshold to suppress static steering twitch
        self.k_stanley_factor = k_stanley_factor                   # Proportional scaling coefficient for Stanley cross-track gain
        self.vel_look_ahead_factor = vel_look_ahead_factor         # Look-ahead velocity interpolation horizon multiplier
        self.look_ahead_progress_factor = look_ahead_progress_factor # Spatial look-ahead horizon gain scaled by current speed

        # Timer handle for periodic RViz trajectory visualization
        self.waypoint_vis_timer = None
        self.reset()
        
        # --- Hardware-Optimized Quality of Service (QoS) Profile ---
        # High-throughput, lowest-latency configuration for real-time sensor streams (e.g., EKF/SLAM state estimates)
        qos_fast = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Drop delayed packets rather than stalling for retransmissions
            durability=DurabilityPolicy.VOLATILE,       # Do not persist samples for late-joining subscribers
            history=HistoryPolicy.KEEP_LAST,            # Retain only the most recent sample
            depth=1                                     # Minimum queue depth to prevent processing stale state updates
        )

        # --- ROS 2 Interfaces ---
        # Publisher for low-level Ackermann steering and throttle commands sent to the VESC/chassis microcontroller
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )

        # Subscriber for filtered state estimation from onboard sensor fusion (EKF/Fast-LIO)
        self.odom_subscriber = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, qos_fast
        )

        # Publisher for RViz visualization of the reference path
        self.vis_publisher = self.create_publisher(
            Marker, '/visual_waypoints', 10
        )

        # Subscriber to receive manual 2D Pose Estimates from RViz to trigger and synchronize initial hardware state
        self.initial_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.set_initial_pose_callback,
            10
        )

        # High-resolution periodic timer (100 Hz) to record telemetry for post-run empirical verification
        self.log_data_timer = self.create_timer(
            1e-2, self.log_callback
        )
        self.get_logger().info("SwitchingController node initialized.")


    def reset(self):
        """
        Reset controller internal state, trajectory buffers, and progress monitors.
        """
        self.t = 0.0                      # Elapsed running duration (s)
        self.start = 0.0                  # Hardware clock reference at start of deployment (s)
        self.progress = 0.0               # Normalized curvilinear distance along track centerline: s in [0, 1]

        self.k_stanley = 1.0              # Dynamically computed cross-track gain
        self.look_ahead_progress = 0.0    # Dynamic spatial look-ahead progress offset

        # State representations: [x, y, delta, v, yaw, yaw_rate, ...]
        self.X = np.zeros(7, dtype="float")      # Estimated vehicle physical state vector
        self.X_des = np.zeros(7, dtype="float")  # Synthesized target state queried from optimal splines
        self.TDCar = traj_data_dict_get()        # Data structure logging hardware state trajectories

        self.first_callback = True        # Timestamp synchronization flag for the first received feedback cycle
        self.running = False              # Engagement interlock; remains False until primed via RViz initial pose
        self.get_logger().debug("Controller state reset.")


    def publish_waypoints_viz(self):
        """
        Publish the loaded reference trajectory as an RViz LINE_STRIP Marker for visual sanity checks.
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1             # Line strip diameter in meters

        # Color profile: Solid Green
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Populate marker geometry using discrete spatial waypoints
        for pt in self.WP.waypoints:
            p = Point()
            p.x = float(pt[0])
            p.y = float(pt[1])
            p.z = 0.0
            marker.points.append(p)
        self.vis_publisher.publish(marker)


    def set_initial_pose_callback(self, msg):
        """
        Callback triggered when a 2D Pose Estimate is published from RViz.
        Synchronizes coordinate frames, resets internal buffers, and engages active hardware control.
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Convert quaternion orientation to planar Euler yaw angle
        q = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        yaw = R.from_quat(q).as_euler('zyx')[0]

        # Reset states and initialize vehicle pose
        self.reset()
        self.X[0] = x
        self.X[1] = y
        self.X[4] = yaw
        self.running = True               # Engage the active tracking controller
        self.get_logger().info(
            f"Initial pose set: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}. GO!"
        )


    def log_callback(self):
        """
        100 Hz logging timer storing time, normalized spatial progress, and multi-state telemetry.
        """
        if not self.running:
            return
        self.TDCar["t_steps"].append(self.t)
        self.TDCar["progress"].append(self.progress)
        self.TDCar["states"].append(np.copy(self.X))


    def get_traj_splines(self, opt_traj_data_path, waypoint_density=10):
        """
        Load synthesized offline trajectory profiles, parameterize continuous path and velocity splines,
        and adapt controller tracking gains to mean velocity expectations.
        """
        # Load trajectory data from disk
        self.TDOpt = load_traj_data(opt_traj_data_path)
        self.TDOpt["states"] = self.TDOpt["states"][:, :-1]

        # Parameterize reference velocity profile as a 1D cubic B-spline over normalized progress
        self.V_tck = splrep(
            self.TDOpt["progress"],
            self.TDOpt["states"][:, 3],
            s=1.0
        )

        # Initialize spatial waypoint handler for closest-point progress projection and path tangents
        self.WP = WayPoint(
            waypoints=self.TDOpt["states"][:, :2],
            waypoint_density=waypoint_density,
            smoothness_params=np.ones(2),
            plot_waypoints_flag=False
        )

        # Scale Stanley gain proportionally to average reference velocity
        self.k_stanley = (
            np.mean(self.TDOpt["states"][:, 3]) * self.k_stanley_factor
        )

        # [LOGGING ADDITION] Log track statistics
        track_len = self.WP.cumulative_dists[-1]
        num_pts = len(self.WP.waypoints)
        self.get_logger().info(
            f"Track Loaded: {track_len:.2f}m long with {num_pts} waypoints."
        )

        # Launch background timer to publish visualization markers to RViz periodically (0.5 Hz)
        self.waypoint_vis_timer = self.create_timer(
            2.0, self.publish_waypoints_viz
        )


    def stanley_ctrl_modified(self):
        """
        Compute front wheel steering command using a modified Stanley path tracking control law.
        Projects position to the front axle and balances lateral cross-track error with tangent heading error.
        """
        # Transform center-of-mass coordinates forward to the vehicle's front axle center
        front_axle = self.X[:2] + self.lf * np.array(
            [np.cos(self.X[4]), np.sin(self.X[4])]
        )

        # Compute displacement vector from the target reference position to the vehicle's front axle
        heading_error_vec = front_axle - self.X_des[:2]
        
        # Calculate cross-track error (CTE) as the normal scalar projection relative to desired heading
        cte = np.dot(
            heading_error_vec,
            np.array([-np.sin(self.X_des[4]), np.cos(self.X_des[4])])
        )

        steering_angle = 0.0
        # Guard against zero-speed division singularities
        if self.X[3] > self.vel_thresh:
            # Velocity-dependent non-linear cross-track error correction
            cte_term = np.arctan2(self.k_stanley * cte, self.X[3])
            
            # Heading error bounded within [-pi, pi] to avoid discontinuous wraparound
            heading_term = ((self.X_des[4] - self.X[4]) + np.pi) % (2*np.pi) - np.pi
            
            # Stanley steering feedback control law
            steering_angle = heading_term - cte_term

        # Update dynamic progress look-ahead horizon proportional to forward speed
        self.look_ahead_progress = self.look_ahead_progress_factor * self.X[3]
        return steering_angle


    def odom_callback(self, msg: Pose):
        """
        Real-time feedback loop processing filtered odometry messages,
        updating state vectors, checking run termination, and triggering control computation.
        """
        # (Optional: automatic lap termination interlock can be uncommented here if needed)
        """
        if self.progress >= 1.0 and self.running:
            self.running = False
            self.get_logger().info("Lap completed. Stopping controller.")
            self.print_results()
        """

        # Enforce safety-stop commands with zero velocity if controller is disengaged
        if not self.running:
            stop_msg = AckermannDriveStamped()
            stop_msg.drive.speed = 0.0
            self.drive_publisher.publish(stop_msg)
            return

        # Extract planar position, orientation, and body-frame velocities from incoming odometry
        q = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        self.X[0] = msg.pose.pose.position.x
        self.X[2] = self.X_des[2]                                  # Retain previous commanded steering angle
        self.X[1] = msg.pose.pose.position.y
        self.X[3] = msg.twist.twist.linear.x                      # Longitudinal vehicle speed
        self.X[4] = R.from_quat(q).as_euler('zyx')[0]             # Heading angle (yaw)
        self.X[5] = msg.twist.twist.angular.z                     # Yaw rate

        # Establish precise time baseline on initial active cycle
        now = self.get_clock().now().nanoseconds / 1e9
        if self.first_callback:
            self.start = now
            self.first_callback = False

        self.t = now - self.start
        self.step_controller()


    def step_controller(self):
        """
        Execute an instantaneous control update:
        1. Find closest spatial progress on reference curve.
        2. Evaluate B-spline targets at the spatial look-ahead point.
        3. Solve modified Stanley steering law.
        4. Broadcast low-level Ackermann drive actuation messages.
        """
        # Determine normalized progress s in [0, 1] via orthogonal spatial projection onto waypoints
        self.progress = self.WP.find_progress_value(self.X[:2])[0]

        # Spline Lookups: Query target 2D coordinates [X_des, Y_des] at forward look-ahead progress
        self.X_des[:2] = np.array([
            splev(self.progress + self.look_ahead_progress, tck)
            for tck in self.WP.waypoints_tck[:2]
        ])

        # Query tangent vector derivatives to compute target trajectory orientation
        desired_phi = np.arctan2(*[
            splev(self.progress + self.look_ahead_progress, tck)
            for tck in self.WP.waypoints_dot_tck[:2][::-1]
        ])

        # Control Calculation: Compute target steering angle and lookup target speed
        self.X_des[2] = self.stanley_ctrl_modified()
        self.X_des[3] = splev(
            self.progress + self.vel_look_ahead_factor * self.look_ahead_progress,
            self.V_tck
        )
        # Unwrap heading angle to maintain smooth continuity across +/- pi boundaries
        self.X_des[4] = np.unwrap([self.X_des[4], desired_phi])[-1]

        # [LOGGING ADDITION] Periodic Telemetry (Throttled to 2 seconds)
        # Allows you to verify speed and error without spamming the console
        self.get_logger().info(
            f"Speed: {self.X[3]:.1f}/{self.X_des[3]:.1f} m/s | "
            f"Steer: {self.X_des[2]:.2f} rad | "
            f"Prog: {self.progress*100:.1f}%",
            throttle_duration_sec=0.5
        )

        # Assemble and publish the final Ackermann drive message to hardware actuator drivers
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = float(self.X_des[2])
        drive_msg.drive.speed = float(self.X_des[3])
        self.drive_publisher.publish(drive_msg)


#########################################################################################################
#########################################################################################################


def summarise(TDCar:dict, race_track_path:str, waypoint_density=10):
    """
    Compute tracking performance metrics (cross-track deviations, completion lap times)
    and print summary statistics following experiment termination.
    """
    # Cast trajectory tracking histories into NumPy arrays for vectorized operations
    TDCar["t_steps"] = np.array(TDCar["t_steps"])
    TDCar["progress"] = np.array(TDCar["progress"])
    TDCar["states"] = np.array(TDCar["states"])[:, :-1]

    # Load ground truth track layout and waypoints
    WP = WayPoint(
        path_to_waypts=race_track_path,
        waypoint_density=waypoint_density,
        smoothness_params=np.ones(2),
        plot_waypoints_flag=False
    )
    
    # Identify orthogonal nearest-neighbor coordinates along the track centerline
    nearest_waypoints = np.vstack([
        WP.find_progress_value(x[:2])[1]
        for x in TDCar["states"]
    ])
    
    # Calculate Euclidean cross-track error across all logged timestamps
    deviations = np.linalg.norm(
        nearest_waypoints - TDCar["states"][:, :2],
        axis=-1
    )
    TDCar["deviations"] = deviations

    # Output quantitative evaluation metrics to standard output
    print("\n" + "="*30)
    print(" LAP COMPLETED - RESULTS ")
    print("="*30)
    print_details([TDCar])
    print("="*30 + "\n")
    
        
#########################################################################################################
#########################################################################################################


def main(args=None):
    """
    Main entry point for physical hardware deployment:
    Initializes rclpy, loads the reference trajectory, starts the event loop,
    and safely logs telemetry to disk upon node shutdown.
    """
    rclpy.init(args=args)
    SC = SwitchingController()

    try:
        # Load offline synthesized optimal trajectory parameters
        SC.get_traj_splines(
            opt_traj_data_path=TRAJ_DATA_PATH,
        )
        print("[INFO]: Trajectories loaded. Starting SwitchingController...")
        rclpy.spin(SC)

    except Exception as e:
        print(f"Exiting due to exception: {e}")

    finally:
        # Process performance metrics and write telemetry logs to disk upon process termination
        summarise(
            TDCar=SC.TDCar,
            race_track_path=RACE_TRACK_PATH
        )
        
        save_traj_data(
            TDList=[SC.TDCar],
            filepath=SAVE_DATA_PATH,
        )


#########################################################################################################
#########################################################################################################


if __name__ == "__main__":
    main()


#########################################################################################################
#########################################################################################################