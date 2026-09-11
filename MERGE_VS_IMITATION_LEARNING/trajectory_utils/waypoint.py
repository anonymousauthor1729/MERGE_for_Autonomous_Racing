import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.interpolate import splrep, splder, splev

from .plotter import plot_waypoints


#########################################################################################################
#########################################################################################################


def extract_waypoints_info(waypoints, waypoint_density):
    cumulative_dists = np.insert(
        np.cumsum(
            np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
            ), 0, 0
    )
    cumulative_dists_fine = np.linspace(
        0., cumulative_dists[-1],
        int(cumulative_dists[-1] * waypoint_density),
        endpoint=True
    )
    progress = cumulative_dists_fine / cumulative_dists_fine[-1]
    waypoints = np.column_stack([
        np.interp(cumulative_dists_fine, cumulative_dists, waypoints[:, i])
        for i in range(waypoints.shape[1])
    ])
    return progress, waypoints, cumulative_dists_fine


#########################################################################################################
#########################################################################################################


class WayPoint:
    def __init__(self, 
                 path_to_waypts: str=None,
                 waypoints: np.ndarray=None, 
                 waypoint_density: float = 1., 
                 waypoints_width: np.ndarray = 0.,
                 smoothness_params: np.ndarray = None,
                 plot_waypoints_flag: bool = False,
                ):
        
        if waypoints is None:
            if path_to_waypts is None:
                raise AttributeError(
                    "Either waypoints (np.ndarray) or path_to_waypts (str) \
                    of the waypoints csv file must be provided!"
                )
            waypoints = self.load_waypoints(path_to_waypts)

        if smoothness_params is None:
            smoothness_params = np.ones(waypoints.shape[-1])

        progress, self.waypoints, self.cumulative_dists = extract_waypoints_info(waypoints, waypoint_density)
        self.waypoints_tree = KDTree(self.waypoints)
        self.waypoints_width = waypoints_width

        self.waypoints_tck = [
            splrep(progress, self.waypoints[:, i], s=smoothness_params[i]) 
            for i in range(self.waypoints.shape[-1])
        ]
        self.waypoints_dot_tck = [
            splder(state_tck) for state_tck in self.waypoints_tck
        ]

        self.waypoints = np.column_stack([
            splev(progress, tck) for tck in self.waypoints_tck
        ])
        
        if plot_waypoints_flag:
            _, ax = plt.subplots(1, 1, figsize=(15, 7))
            plot_waypoints(self.waypoints, self.waypoints_width, ax)

            ax.plot(
                splev(progress, self.waypoints_tck[0]), 
                splev(progress, self.waypoints_tck[1]), 
                linewidth=2, label="Spline waypoints"
            )
            plt.axis("equal")
            plt.legend()
            plt.show()


    def find_progress_value(self, pos: np.ndarray):
        pos = np.asarray(pos)
        single_input = False

        if pos.ndim == 1:
            pos = pos[None, :] 
            single_input = True

        _, nearest_idx = self.waypoints_tree.query(pos)
        next_idx = (nearest_idx + 1) % len(self.waypoints)
        tangent_vec = (
            self.waypoints[next_idx] - self.waypoints[nearest_idx]
        )
        tangent_norm = np.linalg.norm(tangent_vec, axis=1, keepdims=True)
        tangent_vec = tangent_vec / tangent_norm

        error_vec = pos - self.waypoints[nearest_idx]
        proj_along_tangent = np.sum(error_vec * tangent_vec, axis=1)
        progress = (
            proj_along_tangent + self.cumulative_dists[nearest_idx]
        ) / self.cumulative_dists[-1]

        progress = np.maximum(progress, 0.0)
        true_closest_waypoint = (
            self.waypoints[nearest_idx] +
            proj_along_tangent[:, None] * tangent_vec
        )

        if single_input:
            return progress[0], true_closest_waypoint[0]
        return progress, true_closest_waypoint

    
    def compute_deviations(self, pos: np.ndarray):
        nearest_waypoints = self.find_progress_value(pos)[1]
        deviations = np.linalg.norm(nearest_waypoints - pos, axis=-1)
        return deviations


    def load_waypoints(self, filepath:str):
        return np.loadtxt(
            filepath, delimiter=",", skiprows=1, usecols=(0, 1)
        )
    
    
    def save_waypoints(self, filepath:str):
        np.savetxt(
            filepath, self.waypoints, header="x_m y_m", fmt="%.15f", delimiter=","
        )
        print(f"[INFO]: Waypoints saved succesfully -> {filepath}")


#########################################################################################################
#########################################################################################################
