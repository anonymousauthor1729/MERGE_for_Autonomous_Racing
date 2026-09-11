import numpy as np
from .gym_car import Car


#########################################################################################################
#########################################################################################################


def traj_data_dict_get():
    """Returns a standardized dictionary for logging simulation rollouts."""
    return {
        "name": "",
        "states": [],        # Full vehicle state history
        "t_steps": [],       # Time stamps
        "progress": [],      # s-coordinates (0 to 1)
        "ctrl_inputs": [],   # U vectors
        "deviations": [],    # Lateral error history
    }


#########################################################################################################
#########################################################################################################


class GeomtricControllersHub():
    """
    A collection of geometric steering laws for path tracking.
    Supports PID and Pure Pursuit architectures.
    """
    def __init__(self, car_dt, car_lf, car_lr, kp=0., ki=0., kd=0., 
                 k_stanley=0., vel_thresh=0., look_ahead_progress=0.):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.k_stanley = k_stanley
        self.vel_thresh = vel_thresh
        self.look_ahead_progress = look_ahead_progress
        self.lf, self.lr = car_lf, car_lr
        self.dt = car_dt
        self.reset()
    
    def reset(self):
        """Clears controller memory (integral and derivative terms)."""
        self.integral_error = 0.0
        self.prev_error = 0.0

    def pid_ctrl(self, X:np.ndarray, desired_xy:np.ndarray, desired_phi:float):
        """Computes steering angle based on a PID law over cross-track error."""
        # 1. Calculate positional error vector
        pos_error = desired_xy - X[:2]
        # 2. Project error onto the track normal to find Cross-Track Error (CTE)
        error = np.dot(pos_error, np.array([-np.sin(desired_phi), np.cos(desired_phi)]))
        
        P = self.kp * error
        self.integral_error += error * self.dt
        if abs(error) > 0.5: self.integral_error = 0.0 # Anti-windup reset
        I = self.ki * self.integral_error
        
        D = self.kd * (error - self.prev_error) / self.dt
        self.prev_error = error
        
        steering_angle = 0.
        if X[3] > self.vel_thresh:
            # Add heading error (Proportional) to the CTE-PID correction
            phi_error = (desired_phi - X[4] + np.pi) % (2 * np.pi) - np.pi
            desired_omega = phi_error + (P + I + D)
            # Convert yaw-rate command to steering angle using bicycle kinematics
            steering_angle = np.arctan(desired_omega * (self.lf + self.lr) / X[3])
        return steering_angle
    
    def pure_pursuit_ctrl(self, X:np.ndarray, desired_xy:np.ndarray, desired_phi:float):
        """Computes steering angle based on the geometry of a circular arc to the target."""
        error = desired_xy - X[:2]
        # Project target into the vehicle's local frame
        local_y = np.dot(error, np.array([-np.sin(X[4]), np.cos(X[4])]))
        # L2-norm is the look-ahead distance
        gamma = (2.0 * local_y) / np.linalg.norm(error)
        
        steering_angle = 0.
        if X[3] > self.vel_thresh: 
            steering_angle = np.arctan(gamma * (self.lr + self.lf))
        return steering_angle


#########################################################################################################
#########################################################################################################


def simulate_controllers(C:Car, progress_func, desired_xy_func, desired_phi_func, 
                         velocity_ctrl_func, steering_ctrl_fun, completion_thresh, 
                         simulation_duration, velocity_tube_width=1e-1, steering_tube_width=1e-2):
    """
    Main simulation wrapper that performs a single lap rollout.
    Implements soft saturation (tanh) for smooth longitudinal and lateral control.
    """
    TDCar = traj_data_dict_get()
    TDCar["name"] = "Car"
    desired_phi = C.X[4]
    last_lap_time = 0.
    
    for _ in range(int(simulation_duration / C.dt)):
        # 1. Sync simulation with current progress on the track
        progress = progress_func(C.X[:2])[0]
        desired_xy = desired_xy_func(progress)
        # 2. Smoothly unwrap the heading to prevent 2*pi jumping during integration
        desired_phi = np.unwrap([desired_phi, desired_phi_func(progress)])[-1]
        
        # 3. High-level controller target selection
        desired_steering_angle = steering_ctrl_fun(C.X, desired_xy, desired_phi)
        desired_velocity = velocity_ctrl_func(progress)
        
        # 4. Low-level Actuator Dynamics
        # tanh((error/width)^3) provides a smooth ramp that saturates at C.max
        acceleration = np.tanh((1.5 * (desired_velocity - C.X[3]) / 
                                velocity_tube_width) ** 3) * C.a_max
        steering_velocity = np.tanh((1.5 * (desired_steering_angle - C.X[2]) 
                                     / steering_tube_width) ** 3) * C.sv_max
        
        U = np.array([steering_velocity, acceleration])
        C.step_sim(U) # Physics step
        
        # 5. Data Logging
        TDCar["t_steps"].append(C.t)
        TDCar["progress"].append(progress)
        TDCar["states"].append(np.copy(C.X))
        TDCar['ctrl_inputs'].append(np.copy(C.U))
        
        # Termination condition (end of track)
        if 1 - progress <= completion_thresh and last_lap_time == 0.: break

    # Convert list logs to efficient numpy arrays
    TDCar["t_steps"] = np.array(TDCar["t_steps"])
    TDCar["progress"] = np.array(TDCar["progress"])
    TDCar['ctrl_inputs'] = np.array(TDCar['ctrl_inputs'])
    TDCar["states"] = np.array(TDCar["states"])
    return TDCar


#########################################################################################################
#########################################################################################################