import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from collections import defaultdict
import matplotlib.patches as patches
from scipy.interpolate import splrep, splev, splder

from .segmenter import GMM_Segmentation_Handler

#########################################################################################################
#########################################################################################################


def plot_wavefront_snapshot(voxel_size, visited_voxels, valid_neighbors, raw_start_med,
                            curr_voxel, ref_traj, TDList=[], filedate=None):
    
    fig, ax = plt.subplots(figsize=(14, 14))    
    # --- Plotting Grid and Voxels ---
    ax.set_aspect('equal')

    for i, TD in enumerate(TDList):
        ax.plot(TD["states"][:, 0] - raw_start_med[0], TD["states"][:, 1] - raw_start_med[1], zorder=1,
                color="#8B6477", alpha=0.5, linewidth=2, linestyle='-.', 
                label='Demonstrations' if i == 0 else "")
    
    # Draw visited voxels
    for v in visited_voxels:
        rect = patches.Rectangle((v[0]*voxel_size - voxel_size/2, v[1]*voxel_size - voxel_size/2), 
                                 voxel_size, voxel_size, linewidth=1, edgecolor="#999EA5", zorder=0,
                                 facecolor='#E2E8F0', label='Visited' if v == visited_voxels[0] else "")
        ax.add_patch(rect)
        
    # Draw candidate neighbors
    for v in valid_neighbors:
        rect = patches.Rectangle((v[0]*voxel_size - voxel_size/2, v[1]*voxel_size - voxel_size/2),voxel_size, 
                                 voxel_size, zorder=2, linewidth=2, edgecolor='#10B981', facecolor='#D1FAE5', 
                                 hatch='//', label='Candidates' if v == valid_neighbors[0] else "")
        ax.add_patch(rect)

    # Draw current voxel
    rect = patches.Rectangle((curr_voxel[0]*voxel_size - voxel_size/2, curr_voxel[1]*voxel_size - voxel_size/2), 
                             voxel_size, voxel_size,  zorder=3, linewidth=2, edgecolor='#4F46E5', 
                             facecolor='#C7D2FE', label='Current Voxel')
    ax.add_patch(rect)

    # --- Plotting Vectors ---
    current_coord = np.array(curr_voxel) * voxel_size
    
    # Candidate evaluation arrows
    for n in valid_neighbors:
        n_coord = np.array(n) * voxel_size
        move_vec = n_coord - current_coord
        ax.arrow(current_coord[0], current_coord[1], move_vec[0]*0.7, move_vec[1]*0.7, 
                 head_width=0.01, color='#059669', linewidth=1.5, zorder=4)

    # Reference Trajectory Line
    ref_x = [p[0] for p in ref_traj]
    ref_y = [p[1] for p in ref_traj]
    ax.plot(ref_x, ref_y, color='#374151', marker='o', linewidth=4, 
            label='Extracted Trajectory', zorder=5)

    # Clean up plot
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend(loc='upper left', framealpha=0.9)
    # plt.title("Reference Trajectory using Wavefront Expansion")
    
    plt.tight_layout()
    if filedate is not None:
        plt.savefig(join("data_dummy", f"Ref_Traj_Gen_{filedate}.png"), dpi=300, bbox_inches='tight')
    plt.show()


#########################################################################################################
#########################################################################################################


def generate_reference_trajectory(TDList: list, voxel_size: float, state_space_dim:int,
                                  plot_snapshot_index=None, filedate=None):
    """
    Generates a reference trajectory by tracing the most densely populated 
    voxels connected from start to finish, computing local medians.
    
    Args:
        TDList: List of demonstration dictionaries.
        voxel_size: Float defining the spatial size of a voxel.
        
    Returns:
        np.ndarray: The reference trajectory (M, D).
    """
    D = state_space_dim
    
    # Find the median of the starting and ending points
    start_pts = np.array([TD["states"][0, :D] for TD in TDList])
    end_pts = np.array([TD["states"][-1, :D] for TD in TDList])
    raw_start_med = np.median(start_pts, axis=0)
    raw_end_med = np.median(end_pts, axis=0)
    
    # Spatial Hashing & Voting Map
    # Map stores: {'points': list of all states here}
    voxel_map = defaultdict(lambda: {'points': []})
    
    for demo_idx, TD in enumerate(TDList):
        states = TD["states"][:, :D] - raw_start_med[np.newaxis]
        # Convert continuous coordinates to discrete voxel tuples
        voxels = np.floor(states / voxel_size).astype(int)
        
        for v, x in zip(voxels, states):
            v_tuple = tuple(v)
            voxel_map[v_tuple]['points'].append(x)
            # Sets prevent duplicate counting!
            
    # Build KD-Tree on strictly populated voxels
    unique_voxels = np.array(list(voxel_map.keys()))
    tree = cKDTree(unique_voxels)
    
    # Safety feature: The mathematical median might fall in an empty gap between demos.
    # Query the KD-Tree to snap to the absolutely closest *populated* voxel.
    _, start_idx = tree.query(np.floor(np.zeros_like(raw_start_med) / voxel_size).astype(int))
    _, end_idx = tree.query(np.floor(raw_end_med - raw_start_med / voxel_size).astype(int))
    
    curr_voxel = tuple(unique_voxels[start_idx])
    end_voxel = tuple(unique_voxels[end_idx])
    
    # Traversal Setup
    ref_traj = []
    visited = set()
    
    # In D-dimensions, the diagonal distance to a neighbor is sqrt(D).
    # Adding a small epsilon guarantees we catch all adjacent corner-sharing voxels.
    search_radius = np.sqrt(D) + 5e-2

    # 5. The Voxel-Marching Loop
    count = 0
    while True:
        visited.add(curr_voxel)
        
        # Compute and save the spatial median of the raw points inside this voxel
        local_median = np.median(voxel_map[curr_voxel]['points'], axis=0)
        ref_traj.append(local_median)
        
        # Termination condition: We reached the target destination
        if curr_voxel == end_voxel: break
            
        # Find all neighbor voxels within the grid
        idx_list = tree.query_ball_point(curr_voxel, r=search_radius)
        neighbors = [tuple(unique_voxels[i]) for i in idx_list]
        
        # Filter out where we've already been, and don't count ourselves
        valid_neighbors = [n for n in neighbors if n not in visited and n != curr_voxel]
        if not valid_neighbors: break # Traversal hit a dead end
            
        # Add ALL valid neighbors to the visited set immediately.
        # This prevents backtracking or zigzagging here again.
        for n in valid_neighbors: visited.add(n)
        visited.add(curr_voxel)
        
        # Pool the raw coordinates of every point inside this entire frontier
        pooled_points = []
        for n in valid_neighbors:
            pooled_points.extend(voxel_map[n]['points'])
            
        # Take the spatial median of the pooled points.
        # This finds the true physical center-of-mass of the advancing wave.
        wavefront_median = np.median(pooled_points, axis=0)
        
        # Snap the new median back to the nearest POPULATED voxel,
        # but penalize voxels that require a sharp change in direction.
        neighbor_coords = np.array(valid_neighbors) * voxel_size
        current_coord = np.array(curr_voxel) * voxel_size
        
        # Calculate Spatial Distance Cost (original logic)
        dist_costs = np.linalg.norm(neighbor_coords - wavefront_median, axis=1)
        
        # Calculate Angular Penalty (Inertia)
        if len(ref_traj) > 1:
            # Vector of our previous movement
            prev_vec = ref_traj[-1] - ref_traj[-2]
            prev_norm = np.linalg.norm(prev_vec)
            
            if prev_norm > 1e-6:
                prev_dir = prev_vec / prev_norm
                
                # Vectors to all candidate neighbors
                move_vecs = neighbor_coords - current_coord
                move_norms = np.linalg.norm(move_vecs, axis=1)
                
                # Avoid division by zero
                move_norms[move_norms == 0] = 1e-6
                move_dirs = move_vecs / move_norms[:, np.newaxis]
                
                # Dot product ranges from 1 (same direction) to -1 (opposite)
                # We want to penalize negative or zero dot products.
                angle_penalties = 1.0 - np.dot(move_dirs, prev_dir)
            else:
                angle_penalties = np.zeros(len(valid_neighbors))
        else:
            # No previous direction on the first step
            angle_penalties = np.zeros(len(valid_neighbors))
            
        # Combine costs
        # The weight determines how strictly the algorithm resists turning.
        # Scaling by voxel_size keeps the units roughly comparable to distance.
        inertia_weight = 0.5 * voxel_size 
        total_costs = dist_costs + (inertia_weight * angle_penalties)
        
        # Plot the wavefront snapshot if needed
        if plot_snapshot_index is not None and count==plot_snapshot_index: 
            plot_wavefront_snapshot(
                voxel_size=voxel_size, visited_voxels=list(visited), 
                valid_neighbors=valid_neighbors, curr_voxel=curr_voxel,
                ref_traj=ref_traj, raw_start_med=raw_start_med, 
                TDList=TDList, filedate=filedate,
            )
        count +=1 
        
        # Select the voxel with the lowest combined cost
        curr_voxel = valid_neighbors[np.argmin(total_costs)]
        
    return np.array(ref_traj) + raw_start_med[np.newaxis]


#########################################################################################################
#########################################################################################################


def generate_opt_trajectory(TDList: list, 
                            num_segs: int, 
                            acceleration_max: float,
                            safety_width: float, 
                            safety_gain: float, 
                            switch_gain: float, 
                            track_length: float,
                            downsample: int = 1,
                            filedate: str = None,
                            progress_scaler: float = 1e3, 
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
        acceleration_max (float): Physical acceleration limit (m/s).
        safety_width (float): Maximum allowed lateral deviation from centerline (m).
        safety_gain (float): Scaling factor for the lateral safety reward.
        switch_gain (float): Scaling factor for the transition acceleration penalty.
        track_length (float): Total length of the track centerline (m).
        downsample (int, optional): Step size for GMM feature fitting. Defaults to 3.
        progres_scaler (float): For scaling the progress axis for better clustering
        filedate (str, optional): Date string used for saving the plot. Defaults to None.
        waypoints_plotter (callable, optional): Custom plotting function for track boundaries.
        
    Returns:
        dict: The synthesized optimal trajectory containing states and normalized progress.
    """
    
    # 1. Initialize the MERGE Segmenter
    GH = GMM_Segmentation_Handler(
        num_segs=num_segs,
    )
    
    # 2. Re-parameterize time-domain demonstrations into spatial Frenet coordinates
    X_all, X_all_dot, deviations, features = \
        GH.preprocess_trajectories(
            TDList=TDList, 
            progress_scaler=progress_scaler,
        )
    
    # 3. Discover variance-driven 'Smart Bins'
    feature_labels, weights, cluster_labels = \
        GH.fit_gmm_model(
            features=features,
            downsample=downsample
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
            X_all_dot=X_all_dot,            
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
    def __init__(
        self,
        states: np.ndarray,
        progress: np.ndarray,
        smoothness_params:np.ndarray,
        plot_flag:bool=False
    ):
        
        num_states = states.shape[-1]
        self.X_tck = [
            splrep(progress, states[:, i], s=smoothness_params[i])
            for i in range(num_states)
        ]
        self.X_dot_tck = [
            splder(tck) for tck in self.X_tck
        ]
    
        # Evaluate splines
        progress_grid = np.linspace(0, 1, 1000)
        X = np.stack([splev(progress_grid, tck) for tck in self.X_tck], axis=1)
        
        # Create figure (if needed)
        if not plot_flag: return
        fig, axes = plt.subplots(num_states, figsize=(17, 3 * num_states))

        # handle single state case
        if num_states == 1: axes = np.array([axes])

        for i in range(num_states):
            label = f"State {i}"

            # --- State ---
            axes[i].plot(progress_grid, X[:, i], lw=2)
            axes[i].set_title(f"{label}")
            axes[i].set_xlabel("Progress")
            axes[i].set_ylabel("Value")
            axes[i].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()
        

    def get_states(self, progress:float):
       return np.array([splev(progress, tck) for tck in self.X_tck])
   

    def get_states_dot(self, progress:float):
       return np.array([splev(progress, tck) for tck in self.X_dot_tck])
   
   
#########################################################################################################
#########################################################################################################