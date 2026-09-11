import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseWithCovarianceStamped, Pose

import numpy as np
from scipy.interpolate import splev, splrep
from scipy.spatial.transform import Rotation as R

from f1tenth_gym_ros_config import *
from trajectory_utils.waypoint import WayPoint
from trajectory_utils.data import traj_data_dict_get, save_traj_data, load_traj_data, print_details


#########################################################################################################
#########################################################################################################


class SwitchingController(Node):
    """
    ROS 2 node implementing a modified Stanley path-tracking controller for autonomous racing.
    
    Features:
    - Speed-adaptive cross-track error (CTE) gain scheduling.
    - Dynamic look-ahead progress projection along spline-interpolated race trajectories.
    - Decoupled lateral steering and longitudinal speed tracking.
    - RViz marker broadcasting and high-frequency telemetry logging.
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
        Initialize controller gains, physical vehicle dimensions, ROS 2 topics, and state buffers.
        """
        super().__init__("switching_controller_node")

        # --- Geometric & Kinematic Controller Parameters ---
        self.lf = wheelbase_front                                   # Distance from Center of Gravity (CG) to front axle [m]
        self.vel_thresh = velocity_threshold                        # Min linear velocity to prevent Stanley singularity at v=0 [m/s]
        self.k_stanley_factor = k_stanley_factor                    # Scaling factor for baseline Stanley cross-track gain
        self.vel_look_ahead_factor = vel_look_ahead_factor          # Multiplier for velocity spline evaluation horizon
        self.look_ahead_progress_factor = look_ahead_progress_factor  # Speed-dependent scaling factor for spatial look-ahead

        self.waypoint_vis_timer = None
        self.reset()

        # --- ROS 2 Publishers & Subscribers ---
        # Publishes Ackermann steering and speed commands to the vehicle / simulator
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, '/drive', 1
        )

        # Receives state estimation / odometry from the ego vehicle (100 Hz typical)
        self.odom_subscriber = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_callback, 1
        )

        # Publishes the target trajectory as an RViz visual marker line strip
        self.vis_publisher = self.create_publisher(
            Marker, '/visual_waypoints', 10
        )

        # Listens for 2D Pose Estimate tool triggered from RViz to reset/start the run
        self.initial_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.set_initial_pose_callback,
            10
        )

        # High-frequency data logger timer running at 100 Hz (dt = 0.01s)
        self.log_data_timer = self.create_timer(
            1e-2, self.log_callback
        )
        self.get_logger().info("SwitchingController node initialized.")


    def reset(self):
        """
        Reset time counters, spatial progress, states, and telemetry data buffers.
        """
        self.t = 0.0                        # Elapsed run time [s]
        self.start = 0.0                    # Wall clock timestamp at run initiation [s]
        self.progress = 0.0                 # Normalized progress along track manifold s in [0, 1]

        self.k_stanley = 1.0                # Baseline Stanley gain (updated dynamically upon loading trajectory)
        self.look_ahead_progress = 0.0      # Current spatial look-ahead distance in progress domain

        # State vector representation: [x, y, delta (steering), v (speed), yaw, yaw_rate, a (accel)]
        self.X = np.zeros(7, dtype="float")
        self.X_des = np.zeros(7, dtype="float")
        self.TDCar = traj_data_dict_get()   # Telemetry dictionary for post-lap evaluation

        self.first_callback = True         # Flag to synchronize starting clock on first odom tick
        self.running = False                # Execution flag; controlled by initial pose trigger and lap completion
        self.get_logger().debug("Controller state reset.")


    def publish_waypoints_viz(self):
        """
        Broadcast the reference trajectory line strip marker to RViz on /visual_waypoints.
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1                 # Line width [m]
        marker.color.r = 0.0
        marker.color.g = 1.0                 # Visualized as green path
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Populate marker geometry with discrete trajectory waypoints
        for pt in self.WP.waypoints:
            p = Point()
            p.x = float(pt[0])
            p.y = float(pt[1])
            p.z = 0.0
            marker.points.append(p)
        self.vis_publisher.publish(marker)


    def set_initial_pose_callback(self, msg):
        """
        Triggered when RViz /initialpose is published.
        Initializes vehicle states, clears past telemetry, and starts closed-loop tracking.
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Convert quaternion orientation from geometry message to 2D yaw (Euler ZYX)
        q = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        yaw = R.from_quat(q).as_euler('zyx')[0]

        # Reset buffers and seed initial vehicle pose
        self.reset()
        self.X[0] = x
        self.X[1] = y
        self.X[4] = yaw
        self.running = True
        self.get_logger().info(
            f"Initial pose set: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}. GO!"
        )


    def log_callback(self):
        """
        Periodic 100 Hz telemetry logging of time, progress, and full kinematic state vector.
        """
        if not self.running:
            return
        self.TDCar["t_steps"].append(self.t)
        self.TDCar["progress"].append(self.progress)
        self.TDCar["states"].append(np.copy(self.X))


    def get_traj_splines(self, opt_traj_data_path, waypoint_density=10):
        """
        Load offline optimized trajectory (e.g. synthesized from MERGE), parameterize 
        B-spline curves for coordinates/velocities, and initialize the waypoint manager.
        """
        self.TDOpt = load_traj_data(opt_traj_data_path)
        self.TDOpt["states"] = self.TDOpt["states"][:, :-1]

        # Generate 1D cubic spline representation of target velocity profile along track progress
        self.V_tck = splrep(
            self.TDOpt["progress"],
            self.TDOpt["states"][:, 3],
            s=1.0                           # Smoothing factor
        )

        # Build parameterized waypoint manager with arc-length progress mapping
        self.WP = WayPoint(
            waypoints=self.TDOpt["states"][:, :2],
            waypoint_density=waypoint_density,
            smoothness_params=np.ones(2),
            plot_waypoints_flag=False
        )

        # Scale Stanley gain proportionally to average racing speed across the trajectory
        self.k_stanley = (
            np.mean(self.TDOpt["states"][:, 3]) * self.k_stanley_factor
        )

        # [LOGGING ADDITION] Log track statistics
        track_len = self.WP.cumulative_dists[-1]
        num_pts = len(self.WP.waypoints)
        self.get_logger().info(
            f"Track Loaded: {track_len:.2f}m long with {num_pts} waypoints."
        )

        # Periodic timer (0.5 Hz) to continuously display path in RViz
        self.waypoint_vis_timer = self.create_timer(
            2.0, self.publish_waypoints_viz
        )


    def stanley_ctrl_modified(self):
        """
        Modified Stanley lateral control law.
        Projects vehicle position to front axle, computes cross-track error (CTE) and heading error 
        relative to the desired Frenet reference pose, and synthesizes the target steering command.
        """
        # 1. Kinematic projection from vehicle center-of-mass to the front axle center
        front_axle = self.X[:2] + self.lf * np.array(
            [np.cos(self.X[4]), np.sin(self.X[4])]
        )

        # 2. Vector from reference waypoint to the front axle position
        heading_error_vec = front_axle - self.X_des[:2]
        
        # 3. Project error vector onto reference trajectory normal vector to compute signed CTE
        cte = np.dot(
            heading_error_vec,
            np.array([-np.sin(self.X_des[4]), np.cos(self.X_des[4])])
        )

        # 4. Stanley control law: delta = heading_error - arctan(k * cte / (v + epsilon))
        steering_angle = 0.0
        if self.X[3] > self.vel_thresh:
            # Non-linear damping term for cross-track error
            cte_term = np.arctan2(self.k_stanley * cte, self.X[3])
            
            # Heading error wrapped to [-pi, pi] to avoid phase-wrap discontinuity
            heading_term = ((self.X_des[4] - self.X[4]) + np.pi) % (2*np.pi) - np.pi
            
            # Combined lateral control law
            steering_angle = heading_term - cte_term

        # Speed-dependent dynamic look-ahead progress computation
        self.look_ahead_progress = self.look_ahead_progress_factor * self.X[3]
        return steering_angle


    def odom_callback(self, msg: Pose):
        """
        Primary odometry subscriber callback. Extracts rigid-body 2D poses and twists, 
        evaluates lap termination, manages clock offsets, and triggers control updates.
        """
        # Stop car if lap progress completes (progress normalized to [0.0, 1.0])
        if self.progress >= 1.0 and self.running:
            self.running = False
            self.get_logger().info("Lap completed. Stopping controller.")

        # Zero-velocity brake command when inactive or run finished
        if not self.running:
            stop_msg = AckermannDriveStamped()
            stop_msg.drive.speed = 0.0
            self.drive_publisher.publish(stop_msg)
            return

        # Extract orientation quaternion from message and convert to yaw
        q = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        
        # Synchronize vehicle state feedback: [x, y, delta_prev, v_x, yaw, yaw_rate, a_x]
        self.X[0] = msg.pose.pose.position.x
        self.X[2] = self.X_des[2]            # Retain previous steering command
        self.X[1] = msg.pose.pose.position.y
        self.X[3] = msg.twist.twist.linear.x # Longitudinal linear body velocity [m/s]
        self.X[4] = R.from_quat(q).as_euler('zyx')[0] # Vehicle heading (yaw) [rad]
        self.X[5] = msg.twist.twist.angular.z        # Yaw rate [rad/s]

        # Initialize start reference time on first odometry callback received
        now = self.get_clock().now().nanoseconds / 1e9
        if self.first_callback:
            self.start = now
            self.first_callback = False

        self.t = now - self.start
        
        # Execute lateral and longitudinal control loop
        self.step_controller()


    def step_controller(self):
        """
        Execute one controller cycle:
        1. Find closest normalized spatial progress along reference path.
        2. Query reference position, heading, and velocity from splines.
        3. Compute steering angle and commanded velocity.
        4. Publish AckermannDriveStamped command message.
        """
        # Project current vehicle position (x, y) onto the track spline to obtain progress s
        self.progress = self.WP.find_progress_value(self.X[:2])[0]

        # Query target Cartesian coordinates [x_des, y_des] at look-ahead progress horizon
        self.X_des[:2] = np.array([
            splev(self.progress + self.look_ahead_progress, tck)
            for tck in self.WP.waypoints_tck[:2]
        ])

        # Compute desired tangent heading angle phi from spatial path derivatives (dy/ds / dx/ds)
        desired_phi = np.arctan2(*[
            splev(self.progress + self.look_ahead_progress, tck)
            for tck in self.WP.waypoints_dot_tck[:2][::-1]
        ])

        # Compute steering command via modified Stanley formulation
        self.X_des[2] = self.stanley_ctrl_modified()
        
        # Query target speed from the velocity spline with dedicated velocity look-ahead
        self.X_des[3] = splev(
            self.progress + self.vel_look_ahead_factor * self.look_ahead_progress,
            self.V_tck
        )
        
        # Unwrap heading target to ensure continuous phase matching without jump discontinuities
        self.X_des[4] = np.unwrap([self.X_des[4], desired_phi])[-1]

        # [LOGGING ADDITION] Periodic Telemetry (Throttled to 2 seconds)
        # Allows you to verify speed and error without spamming the console
        self.get_logger().info(
            f"Spd: {self.X[3]:.1f}/{self.X_des[3]:.1f} m/s | "
            f"Prog: {self.progress*100:.1f}%",
            throttle_duration_sec=2.0
        )

        # Assemble and broadcast Ackermann control frame to the chassis
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
    Compute lap-completion metrics:
    - Formats recorded arrays.
    - Computes Euclidean lateral deviation from nearest track centerline waypoints.
    - Prints summary metrics (mean speed, lap time, tracking error).
    """
    TDCar["t_steps"] = np.array(TDCar["t_steps"])
    TDCar["progress"] = np.array(TDCar["progress"])
    TDCar["states"] = np.array(TDCar["states"])[:, :-1]

    # Reconstruct ground-truth track geometry for deviation measurement
    WP = WayPoint(
        path_to_waypts=race_track_path,
        waypoint_density=waypoint_density,
        smoothness_params=np.ones(2),
        plot_waypoints_flag=False
    )
    
    # Calculate nearest point on the track centerline for every logged position sample
    nearest_waypoints = np.vstack([
        WP.find_progress_value(x[:2])[1]
        for x in TDCar["states"]
    ])
    
    # Compute L2 norm (Euclidean distance) for lateral deviation across all time steps
    deviations = np.linalg.norm(
        nearest_waypoints - TDCar["states"][:, :2],
        axis=-1
    )
    TDCar["deviations"] = deviations

    # Output quantitative evaluation tables to the console
    print("\n" + "="*30)
    print(" LAP COMPLETED - RESULTS ")
    print("="*30)
    print_details([TDCar])
    print("="*30 + "\n")
    
        
#########################################################################################################
#########################################################################################################


def main(args=None):
    """
    ROS 2 entrypoint:
    1. Initializes rclpy context.
    2. Instantiates controller node and binds spline references.
    3. Enters spinning loop to process callbacks.
    4. Post-processes, summarizes, and serializes collected run data upon termination.
    """
    rclpy.init(args=args)
    SC = SwitchingController()

    try:
        # Load offline synthesized trajectory and construct continuous spline functions
        SC.get_traj_splines(
            opt_traj_data_path=TRAJ_DATA_PATH,
        )
        print("[INFO]: Trajectories loaded. Starting SwitchingController...")
        rclpy.spin(SC)

    except Exception as e:
        print(f"Exiting due to exception: {e}")

    finally:
        # Compute and print lap performance benchmarks
        summarise(
            TDCar=SC.TDCar,
            race_track_path=RACE_TRACK_PATH
        )
        
        # Save telemetry array dictionaries for paper/results plotting
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