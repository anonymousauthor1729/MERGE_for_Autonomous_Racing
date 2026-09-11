import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from matplotlib.gridspec import GridSpec
from scipy.interpolate import splrep, splder, splev


#########################################################################################################
#########################################################################################################

# Default LaTeX axis labels for kinematic and dynamic vehicle state variables:
# X [m], Y [m], Steering Angle (delta) [rad], Forward Velocity (v) [m/s],
# Heading (psi) [rad], Yaw Rate (psi_dot) [rad/s], Sideslip Angle (beta) [rad]
STATE_LABELS = [
    r"\(X\) [m]", r"\(Y\) [m]", 
    r"\(\delta\) [rad]", r"\(v\) [m/s]", 
    r"\(\psi\) [rad]", r"\(\dot{\psi}\) [rad/s]", 
    r"\(\beta\) [rad]"
]


class WaypointTrajectoryUtils:
    """
    Utility handler for track waypoint geometry parameterization,
    spatial progress calculation via orthogonal projection, boundary generation,
    and multi-state trajectory visualization.
    """
    def __init__(self, 
                 path_to_waypts: str = None,
                 waypoints: np.ndarray = None, 
                 waypoint_density: float = 1., 
                 waypoints_width: float = 0.,
                 smoothness_params: np.ndarray = None,
                 plot_waypoints_flag: bool = True,
                ):
        """
        Initializes the waypoint track model:
        1. Reads or receives coarse Cartesian waypoints (X, Y).
        2. Computes cumulative arc length and resamples waypoints at a uniform density.
        3. Normalizes track progress to s in [0, 1].
        4. Builds a KD-Tree and periodic B-splines for spatial projection and tangent evaluations.
        """
        # Load raw waypoint coordinates from CSV if an array is not provided directly
        if waypoints is None:
            if path_to_waypts is None:
                raise AttributeError(
                    "Either waypoints (np.ndarray) or path_to_waypts (str) of the waypoints csv file must be provided!"
                )
            waypoints = np.loadtxt(
                path_to_waypts, delimiter=",", skiprows=1, usecols=(0, 1)
            )

        # Compute cumulative Euclidean arc-length distances along the discrete centerline
        cumulative_dists = np.insert(
            np.cumsum(
                np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
            ), 0, 0
        )
        
        # Discretize arc-length into a finely spaced uniform grid based on waypoint_density
        cumulative_dists_fine = np.linspace(
            0., cumulative_dists[-1],
            int(cumulative_dists[-1] * waypoint_density),
            endpoint=True
        )

        # Calculate normalized curvilinear progress coordinate: s in [0, 1]
        progress = cumulative_dists_fine / cumulative_dists_fine[-1]
        
        # Resample waypoints linearly along the new fine arc-length grid
        waypoints = np.column_stack([
            np.interp(cumulative_dists_fine, cumulative_dists, waypoints[:, i])
            for i in range(waypoints.shape[1])
        ])
        
        # Interpolate or assign track boundaries / lane width across the progress domain
        if np.isscalar(waypoints_width):
            self.waypoints_width = float(waypoints_width)
        else:
            waypoints_width = np.interp(cumulative_dists_fine, cumulative_dists, 
                                        np.asarray(waypoints_width))

        # Set default cubic spline smoothing weights if none are specified
        if smoothness_params is None:
            smoothness_params = np.ones(waypoints.shape[-1])
        
        # Store centerline coordinates, widths, and spatial lookup structures
        self.waypoints = waypoints
        self.waypoints_width = waypoints_width
        self.waypoints_tree = KDTree(waypoints)                     # KD-Tree for nearest-neighbor spatial queries
        self.cumulative_dists = cumulative_dists_fine               # Arc-length lookup table (meters)
        
        # Fit periodic 1D cubic B-splines parameterizing coordinates against normalized progress s
        self.waypoints_tck = [
            splrep(progress, waypoints[:, i], s=smoothness_params[i], per=True) 
            for i in range(waypoints.shape[-1])
        ]
        # Evaluate analytical first derivatives (tangent vectors) of the centerline splines
        self.waypoints_dot_tck = [
            splder(state_tck) for state_tck in self.waypoints_tck
        ]

        # Optional initial sanity visualization of fitted splines and discrete centerline
        if plot_waypoints_flag:
            _, ax = plt.subplots(1, 1, figsize=(15, 15))
            ax.plot(
                splev(progress, self.waypoints_tck[1]), 
                splev(progress, self.waypoints_tck[0]), 
                linewidth=2, label="Spline waypoints"
            )
            self.plot_waypoints(ax)
            plt.axis("equal")
            plt.legend()
            plt.show()


    def find_progress_value(self, car_pos: np.ndarray):
        """
        Projects arbitrary 2D vehicle positions (X, Y) onto the track centerline:
        1. Finds the nearest discrete waypoint via KD-Tree.
        2. Computes the local segment tangent vector.
        3. Projects the position error vector along the tangent.
        4. Derives the continuous normalized track progress s in [0, 1] and Cartesian projection.
        """
        car_pos = np.asarray(car_pos)
        single_input = False

        # Support single point [X, Y] as well as batched queries (N, 2)
        if car_pos.ndim == 1:
            car_pos = car_pos[None, :]   # Reshape to (1, 2)
            single_input = True

        # Query KDTree for nearest waypoint vertex index
        _, nearest_idx = self.waypoints_tree.query(car_pos)  # (N,)
        next_idx = (nearest_idx + 1) % len(self.waypoints)
        
        # Calculate unit tangent vector along the forward waypoint segment
        tangent_vec = (
            self.waypoints[next_idx] - self.waypoints[nearest_idx]
        )  # (N, 2)
        tangent_norm = np.linalg.norm(tangent_vec, axis=1, keepdims=True)
        tangent_vec = tangent_vec / tangent_norm

        # Compute displacement from closest discrete waypoint to car position
        error_vec = car_pos - self.waypoints[nearest_idx]  # (N, 2)
        
        # Scalar projection of positional displacement along the centerline tangent
        proj_along_tangent = np.sum(error_vec * tangent_vec, axis=1)  # (N,)
        
        # Add tangent projection to cumulative distance and normalize by total track length
        progress = (
            proj_along_tangent + self.cumulative_dists[nearest_idx]
        ) / self.cumulative_dists[-1]

        # Prevent negative progress indexing around start/finish threshold
        progress = np.maximum(progress, 0.0)
        
        # Compute exact Cartesian coordinate of the closest orthogonal projection on centerline
        true_closest_waypoint = (
            self.waypoints[nearest_idx] +
            proj_along_tangent[:, None] * tangent_vec
        )  # (N, 2)

        if single_input:
            return progress[0], true_closest_waypoint[0]
        return progress, true_closest_waypoint


    def plot_waypoints(self, ax):
        """
        Plots the track centerline and, if track width > 0, constructs and renders
        the orthogonal left and right track boundaries.
        """
        # Render track centerline (swapping axes: Y on horizontal, X on vertical)
        ax.plot(self.waypoints[:, 1], self.waypoints[:, 0], 
                color="grey", lw=1, alpha=0.75, linestyle="--")
        ax.set_title("XY Trajectory")
        ax.set_xlabel("Y [m]")
        ax.set_ylabel("X [m]")
        ax.grid(True, alpha=0.3)
        plt.axis("equal")
        
        # Skip boundary generation if track width is unassigned or zero
        if np.any(self.waypoints_width) <= 0.: return
        
        # Compute discrete tangent gradients along the path
        grads = np.gradient(self.waypoints, axis=0)
        
        # Derive 2D normal unit vectors orthogonal to tangent vectors [-dy, dx]
        normals = np.column_stack((-grads[:, 1], grads[:, 0]))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-7
        
        # Scale normal unit vectors by half track width (lateral safety envelope)
        if np.isscalar(self.waypoints_width):
            normals *= self.waypoints_width / 2.0
        else:
            normals *= self.waypoints_width[:, None] / 2.0
        
        # Compute Cartesian coordinates for outer left and right boundaries
        left_bound = self.waypoints + normals
        right_bound = self.waypoints - normals
        
        # Render left and right track boundaries
        ax.plot(left_bound[:, 1], left_bound[:, 0], 
                color="black", lw=1, alpha=0.5)
        ax.plot(right_bound[:, 1], right_bound[:, 0], 
                color="black", lw=1, alpha=0.5)

        
    def plot_trajectory_data(self, 
                             TDList_learnt=[], 
                             TDList_demo=[], 
                             TD_learnt_name: str = "Proposed Method", 
                             TD_demo_name: str = "Demonstrations", 
                             x_axis_time: bool = True, 
                             plot_waypoints_flag = True, 
                             filedate: str = None, 
                             title = None, 
                            ):
        """
        Generates comprehensive comparative trajectory plots:
        - Left subplot: 2D XY track geometry showing boundaries, raw demonstrations, and synthesized paths.
        - Right subplots: Stacked state time-series / progress-series comparison for each state dimension.
        """
        # Determine total state dimensions from recorded trajectory dictionaries
        states_count_list = [TD["states"].shape[-1] for TD in TDList_learnt + TDList_demo]
        num_states = max(*states_count_list) if len(states_count_list) > 1 else states_count_list[0]

        # Dynamically append generic labels if state dimensions exceed default STATE_LABELS
        global STATE_LABELS
        if num_states > len(STATE_LABELS):
            STATE_LABELS += [f"State {i}" for i in range(len(STATE_LABELS), num_states)]

        # Initialize multi-panel figure layout using GridSpec
        fig = plt.figure(figsize=(18, 3.5 * num_states))
        gs = GridSpec(num_states, 2, figure=fig, width_ratios=[1.2, 1], wspace=0.1)

        # --- 2D Spatial Subplot (Spans all rows in the left column) ---
        ax_xy = fig.add_subplot(gs[:, 0])
        color_demo = '#FFB07F'   # Light orange for baseline/expert demonstrations
        color_hero = '#1f77b4'   # Steel blue for synthesized/evaluated trajectory

        # Render track centerline and boundaries on the spatial plot
        if self.waypoints is not None and plot_waypoints_flag:
            self.plot_waypoints(ax=ax_xy)

        # Plot all baseline/expert demonstration paths
        for i, TD in enumerate(TDList_demo):
            lbl = TD_demo_name if i == 0 else "_nolegend_"
            ax_xy.plot(TD["states"][:, 1], TD["states"][:, 0],
                       color=color_demo, lw=1.5, alpha=0.75, linestyle='-', label=lbl)

        # Plot the synthesized/optimized trajectory path
        for i, TD in enumerate(TDList_learnt):
            lbl = TD_learnt_name if i == 0 else "_nolegend_"
            ax_xy.plot(TD["states"][:, 1], TD["states"][:, 0],
                       color=color_hero, lw=3, alpha=0.9, label=lbl, zorder=10)
        ax_xy.legend(loc='upper right', framealpha=0.9, fancybox=True)

        # --- State Evolution Subplots (Stacked vertically in the right column) ---
        axes_states = []
        xlabel = "Time [s]" if x_axis_time else "Track Progress [0-1]"

        for j in range(num_states):
            ax = fig.add_subplot(gs[j, 1])
            axes_states.append(ax)

            # Plot state profiles across all demonstrations
            for TD in TDList_demo:
                if np.all(TD["states"][:, j] == 0.): continue
                x_axis = TD["t_steps"] if x_axis_time else TD["progress"]
                ax.plot(x_axis, TD["states"][:, j],
                        color=color_demo, lw=1.5, alpha=0.75)

            # Plot state profile of the synthesized trajectory
            for TD in TDList_learnt:
                if np.all(TD["states"][:, j] == 0.): continue
                x_axis = TD["t_steps"] if x_axis_time else TD["progress"]
                ax.plot(x_axis, TD["states"][:, j],
                        color=color_hero, lw=2.0, alpha=0.95)

            # Align state labels on the right y-axis for clean readability
            ax.set_ylabel(STATE_LABELS[j])
            ax.yaxis.set_label_position("right")
            
            # Show x-axis tick labels only on the bottom-most state subplot
            if j == num_states - 1: ax.set_xlabel(xlabel)
            else: ax.set_xticklabels([])

        # Configure figure-level suptitle and spacing
        if title is None: title = f"{TD_demo_name} and {TD_learnt_name}"
        fig.suptitle(f"F1tenth Gym Ros: {title}", fontsize=27, fontweight="bold")
        plt.subplots_adjust(top=0.93)

        # Save figure to disk if a filedate/timestamp identifier is provided
        if filedate is not None:
            title = title.strip(" ").replace(" ", "_")
            plt.savefig(os.path.join("OUTPUTS", f"{title}_{filedate}.png"), 
                        dpi=500, bbox_inches='tight')
        plt.show()


#########################################################################################################
#########################################################################################################


def traj_data_dict_get():
    """
    Returns an empty standard trajectory data dictionary template.
    Fields include:
    - 'name': Run identifier or controller name.
    - 'states': Array of vehicle state vectors over time.
    - 't_steps': Elapsed timestamps.
    - 'progress': Spatial track progress values s in [0, 1].
    - 'deviations': Cross-track lateral error from centerline waypoints.
    """
    traj_data_dict = {
        "name": "",
        "states": [],
        "t_steps": [],
        "progress": [],
        "deviations": [],
    }
    return traj_data_dict


def save_traj_data(TDList: list, filepath: str):
    """
    Serializes and saves a list of trajectory data dictionaries to disk using pickle.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(TDList, f)
    print(f"[INFO]: Trajectory Data saved succesfully -> {filepath}")


def load_traj_data(filepath: str):
    """
    Loads and deserializes a list of trajectory data dictionaries from a pickle file.
    """
    with open(filepath, 'rb') as f:
        TDList = pickle.load(f)
    print(f"[INFO]: Trajectory Data loaded succesfully -> {filepath}")
    return TDList


#########################################################################################################
#########################################################################################################


def set_publication_style():
    """
    Applies high-resolution, publication-grade styling parameters to Matplotlib:
    - Serif / Times New Roman typography.
    - Scaled font and tick sizes for academic paper formatting.
    - Uniform grid styling with semi-transparent dashed lines.
    """
    plt.rcdefaults()
    plt.rcParams['font.family'] = 'serif' 
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.titlesize'] = 25
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.75
    plt.rcParams['grid.linestyle'] = '--'


#########################################################################################################
#########################################################################################################