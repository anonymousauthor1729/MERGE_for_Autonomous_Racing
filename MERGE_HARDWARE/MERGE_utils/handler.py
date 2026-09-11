import numpy as np
from .segmenter import GMM_Segmentation_Handler


#########################################################################################################
#########################################################################################################


def generate_opt_trajectory(TDList: list, 
                            num_segs: int, 
                            acceleration_max: float,
                            safety_width: float, 
                            safety_gain: float, 
                            switch_gain: float, 
                            track_length: float,
                            downsample: int = 3,
                            filedate: str = None,
                            waypoints_plotter = None,
                            ):
    """
    End-to-end wrapper for the MERGE trajectory synthesis pipeline.
    
    This function takes raw expert demonstrations, extracts their spatial variance,
    segments the track using a Gaussian Mixture Model, and uses Viterbi Dynamic 
    Programming to stitch together a globally optimal, physically feasible racing line.
    
    Args:
        TDList (list): List of dictionaries containing the expert demonstrations.
        num_segs (int): Number of 'Smart Bins' (GMM components) to partition the track into.
        acceleration_max (float): Physical longitudinal acceleration limit (m/s^2).
        safety_width (float): Maximum allowed lateral deviation from centerline (m).
        safety_gain (float): Scaling factor for the lateral safety reward.
        switch_gain (float): Scaling factor for the transition acceleration penalty.
        track_length (float): Total length of the track centerline (m).
        downsample (int, optional): Step size for GMM feature fitting. Defaults to 3.
        filedate (str, optional): Date string used for saving the plot. Defaults to None.
        waypoints_plotter (callable, optional): Custom plotting function for track boundaries.
        
    Returns:
        dict: The synthesized optimal trajectory containing states and normalized progress.
    """
    
    # 1. Initialize the MERGE Segmenter
    GH = GMM_Segmentation_Handler(
        num_segs=num_segs,
        downsample=downsample
    )
    
    # 2. Re-parameterize time-domain demonstrations into spatial Frenet coordinates
    X_all, X_all_dot, deviations, features = \
        GH.preprocess_trajectories(
            TDList=TDList
        )
    
    # 3. Discover variance-driven 'Smart Bins'
    feature_labels, weights, cluster_labels = \
        GH.fit_gmm_model(
            features=features
        )
    
    # 4. Global Optimization via Physics-Aware Switching (Viterbi Trellis)
    best_indices, cluster_scores = \
        GH.compute_optimal_indices(
            X_all=X_all,
            weights=weights, 
            X_all_dot=X_all_dot,
            deviations=deviations, 
            cluster_labels=cluster_labels, 
            safety_width=safety_width, 
            track_length=track_length,
            safety_gain=safety_gain,
            switch_gain=switch_gain,
            acceleration_max=acceleration_max,
        )
    
    # 5. Smooth C-infinity convex blending of the chosen expert sequence
    TDOpt = \
        GH.blend_trajectories(
            X_all=X_all, 
            weights=weights, 
            best_indices=best_indices
        )
    
    # 6. Generate the comprehensive 5-panel MERGE analysis plot
    GH.plot_gmm_results(
        X_all=X_all, 
        features=features, 
        weights=weights, 
        best_indices=best_indices,
        cluster_scores=cluster_scores,
        cluster_labels=cluster_labels, 
        feature_labels=feature_labels, 
        waypoints_plotter=waypoints_plotter,
        filedate=filedate,
    )
    
    return TDOpt


#########################################################################################################
#########################################################################################################


class SwitchingController():
    """
    Modified Stanley Controller for high-speed tracking of the MERGE trajectory.
    
    Unlike standard path trackers, this controller incorporates an anticipative 
    lookahead (to compensate for high-speed actuation latency) and is specifically 
    designed to track the velocity-scheduled profiles synthesized by the MERGE GMM.
    """

    def __init__(
        self,
        k_stanley: float,
        wheelbase_front: float,
        velocity_threshold: float,
        vel_look_ahead_factor: float,
        look_ahead_progress_factor: float,
    ):
        """
        Args:
            k_stanley (float): The base control gain for cross-track error correction.
            wheelbase_front (float): Distance from the center of mass to the front axle (m).
            velocity_threshold (float): Minimum velocity required to activate steering (prevents division by zero).
            vel_look_ahead_factor (float): Gain for velocity-dependent lookahead.
            look_ahead_progress_factor (float): Time-scaling factor (t_lookahead / L_track) 
                                                for spatial anticipative tracking.
        """
        self.lf = wheelbase_front
        self.k_stanley = k_stanley
        self.vel_thresh = velocity_threshold
        self.vel_look_ahead_factor = vel_look_ahead_factor
        
        # Maps to the paper's equation: s_ref(t) = s_curr(t) + v(t) * (t_lookahead / L_track)
        self.look_ahead_progress_factor = look_ahead_progress_factor
        self.look_ahead_progress = 0.
        

    def stanley_ctrl_modified(self, X: np.ndarray, desired_xy: np.ndarray, desired_phi: float):
        """
        Computes the optimal steering angle based on lateral deviation and heading error.
        
        State Vector Assumption (X):
            X[0], X[1]: Current Cartesian X, Y coordinates
            X[3]: Longitudinal Velocity (v)
            X[4]: Yaw/Heading Angle (psi)
            
        Args:
            X (np.ndarray): The current vehicle state vector.
            desired_xy (np.ndarray): The [X, Y] coordinates of the anticipative reference point.
            desired_phi (float): The desired tangent heading angle at the reference point.
            
        Returns:
            float: The commanded steering angle (delta) in radians.
        """
        
        # 1. Anticipative Spatial Lookahead
        # Calculates how far ahead on the normalized progress curve the controller should target
        # based on current speed, effectively compensating for physical steering servo delay.
        self.look_ahead_progress = self.look_ahead_progress_factor * X[3]
        
        # 2. Kinematic Front Axle Projection
        # Stanley control is governed by the error at the front axle, not the center of mass.
        front_axle = X[:2] + self.lf * np.array(
            [np.cos(X[4]), np.sin(X[4])]
        )

        # 3. Cross-Track Error (CTE) Calculation
        # Vector from the target point on the track to the car's front axle
        heading_error_vec = front_axle - desired_xy
        
        # Project this error vector onto the track's normal vector [-sin(phi), cos(phi)]
        # This yields the precise lateral deviation perpendicular to the racing line.
        cte = np.dot(
            heading_error_vec,
            np.array([-np.sin(desired_phi), np.cos(desired_phi)])
        )

        steering_angle = 0.0
        
        # 4. Control Law Execution
        # Only compute steering if the car is moving to avoid singularity (1/v -> infinity)
        if X[3] > self.vel_thresh:
            
            # Non-linear proportional control for lateral deviation
            # np.arctan2 provides numerical stability over standard np.arctan
            cte_term = np.arctan2(self.k_stanley * cte, X[3])
            
            # Heading error correction, normalized strictly to [-pi, pi]
            heading_term = ((desired_phi - X[4]) + np.pi) % (2*np.pi) - np.pi
            
            # Final Steering Law: delta = (psi_desired - psi_current) - arctan(k * e / v)
            # The sign of cte_term depends on the track's normal vector orientation.
            steering_angle = heading_term - cte_term

        return steering_angle
    

#########################################################################################################
#########################################################################################################