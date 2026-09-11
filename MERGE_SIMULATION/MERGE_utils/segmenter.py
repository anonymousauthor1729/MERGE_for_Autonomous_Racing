import numpy as np
from os.path import join
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
from sklearn.mixture import GaussianMixture
from matplotlib.colors import ListedColormap, Normalize


#########################################################################################################
#########################################################################################################


def traj_data_dict_get():
    """
    Initializes and returns a standardized dictionary structure for trajectory data.
    
    Returns:
        dict: A dictionary containing empty lists for name, states, t_steps, 
              progress, and lateral deviations.
    """
    traj_data_dict = {
        "name": "",
        "states": [],
        "t_steps": [],
        "progress": [],
        "deviations": [],
        "ctrl_inputs":[],
    }
    return traj_data_dict


#########################################################################################################
#########################################################################################################


class GMM_Segmentation_Handler(GaussianMixture):
    """
    Core handler for the MERGE (Multi-Expert Regime's Gaussian Ensemble) algorithm.
    Inherits from sklearn's GaussianMixture to discover variance-driven 'Smart Bins' 
    and uses Dynamic Programming (Viterbi) to find the optimal, dynamically 
    feasible sequence of expert trajectories.
    """
    
    def __init__(self, num_segs, downsample):
        """
        Args:
            num_segs (int): The number of GMM components (Smart Bins) to fit.
            downsample (int): Step size for downsampling features during GMM fitting 
                              to speed up computation.
        """
        super().__init__(
            n_components=num_segs, 
            covariance_type='full', 
            random_state=42
        )
        self.num_segs = num_segs
        self.downsample = downsample
        self.sort_idx = np.arange(num_segs)
    

    def preprocess_trajectories(self, TDList):
        """
        Spatially aligns and normalizes a list of time-varying expert demonstrations.
        Converts time-derivatives to spatial-derivatives to decouple geometry from speed.
        
        Args:
            TDList (list of dict): List of trajectory dictionaries.
            
        Returns:
            X_all (np.ndarray): Spatially interpolated states (Demos x Progress x States).
            X_all_dot (np.ndarray): Time derivatives of states (Demos x Progress x States).
            deviations (np.ndarray): Interpolated lateral deviations (Demos x Progress).
            features (np.ndarray): Formatted feature matrix for GMM fitting [Progress, Deviations].
        """
        self.num_demos = len(TDList)
        self.num_states = TDList[0]["states"].shape[-1]
        
        # Use the 25th percentile length to establish a baseline resolution
        self.num_t_steps = int(np.quantile([len(TD["states"]) for TD in TDList], 0.25))
        self.progress = np.linspace(0., 1., self.num_t_steps, endpoint=True)

        # Interpolate states to the normalized progress coordinate (s)
        X_all = np.stack([
            np.column_stack([np.interp(
                self.progress, TD["progress"], TD["states"][:, i]
                ) for i in range(self.num_states)]
            ) for TD in TDList
        ], axis=0)

        # Interpolate lateral deviations
        deviations = np.vstack([np.interp(
                self.progress, TD["progress"], TD["deviations"]
            ) for TD in TDList]
        )
        
        # Interpolate time steps
        t_steps = np.vstack([np.interp(
                self.progress, TD["progress"], TD["t_steps"]
            ) for TD in TDList]
        )
        
        # Compute differences
        dx = np.diff(X_all, axis=1)
        dt = np.diff(t_steps, axis=1)[:, :, np.newaxis]
        
        # Compute time derivatives (e.g., velocities/accelerations) safely
        X_all_dot = np.concatenate([
            np.divide(dx, dt, out=np.zeros_like(dx), where=dt!=0),
            np.zeros((self.num_demos, 1, self.num_states)) # Pad final step
        ], axis=1)

        # Scale progress to match the magnitude of deviations for balanced GMM fitting
        self.progress_feature_factor = self.num_segs * np.max(deviations)
        progress_feature = self.progress * self.progress_feature_factor
        
        # Construct symmetric feature space (mirroring deviations)
        features = np.column_stack([
            np.tile(progress_feature, self.num_demos * 2), 
            np.append(deviations.flatten(), -deviations.flatten())
        ])
        return X_all, X_all_dot, deviations, features
    

    def fit_gmm_model(self, features):
        """
        Fits the GMM to the progress-variance feature space to identify track regimes,
        sorting the resulting clusters sequentially along the track.
        
        Args:
            features (np.ndarray): The [progress, deviation] features.
            
        Returns:
            labels (np.ndarray): Hard cluster assignments for each point.
            marginal_weights (np.ndarray): Continuous GMM responsibilities (gamma_k).
            cluster_labels (np.ndarray): The dominant bin index at each progress step.
        """
        labels = self.fit_predict(features[::self.downsample])
        x0 = np.asarray(features[:self.num_t_steps, 0]).reshape(-1, 1)
        
        # Sort clusters logically from start to finish line
        self.sort_idx = np.argsort(self.means_[:, 0])
        mus = self.means_[:, 0][self.sort_idx]
        sigmas = np.sqrt(self.covariances_[:, 0, 0])[self.sort_idx]
        
        # Remap labels to match the sorted indices
        inverse = np.empty_like(self.sort_idx)
        inverse[self.sort_idx] = np.arange(len(self.sort_idx))
        labels = inverse[labels]

        # Calculate marginal spatial responsibilities gamma_k(s)
        likelihood = norm.pdf(x0, loc=mus[None, :], scale=sigmas[None, :])
        normalization = likelihood.sum(axis=1, keepdims=True)
        marginal_weights = likelihood / normalization
        cluster_labels = np.argmax(marginal_weights, axis=1)
                
        return labels, marginal_weights, cluster_labels
  
    
    def compute_optimal_indices(self, X_all, X_all_dot, deviations, cluster_labels, track_length, 
                                weights, safety_width, acceleration_max, safety_gain, switch_gain):
        """
        Viterbi Dynamic Programming with Exact Tensor Transition Physics.
        Finds the globally optimal sequence of experts by evaluating exact blended 
        accelerations across an N x N transition matrix.
        
        Returns:
            best_indices (np.ndarray): The optimal expert index for each bin.
            cluster_scores (np.ndarray): The base performance scores of each expert per bin.
        """
        mus = self.means_[:, 0][self.sort_idx] / self.progress_feature_factor
        covs = self.covariances_[:, 0, 0][self.sort_idx] / (self.progress_feature_factor**2)
        
        # Calculate the normalized spatial derivative of the weights: L_k(s) - bar{L}(s)
        gauss_exp_term = (mus[:, np.newaxis] - self.progress) / covs[:, np.newaxis]
        gauss_exp_term -= np.einsum('tk,kt->t', weights, gauss_exp_term)[np.newaxis, :]
        
        cluster_scores = np.zeros((self.num_demos, self.num_segs))
        
        # --- 1. INTRA-BIN REWARDS ---
        for k in range(self.num_segs):
            mask = (cluster_labels == k)
            if not np.any(mask): continue
            
            # Reward speed, penalize clipping the safety width bounds
            modified_lat_error = safety_gain * (safety_width - deviations[:, mask])
            scores = X_all[:, mask, 3] * (
                np.log(1 + np.clip(modified_lat_error, a_min=1e-7-1, a_max=None))
            )
            cluster_scores[:, k] = np.mean(scores, axis=1)
        
        # --- 2. VITERBI TRELLIS INITIALIZATION ---
        dp = np.zeros((self.num_segs, self.num_demos))
        parent = np.zeros((self.num_segs, self.num_demos), dtype=int)
        dp[0] = cluster_scores[:, 0]

        # --- 3. DYNAMIC PROGRAMMING OVER TRANSITIONS ---
        for k in range(1, self.num_segs):
            prev_dp = dp[k - 1]
            
            # Isolate the spatial transition window between bin centers
            t_mask = (self.progress >= mus[k-1]) & (self.progress <= mus[k])
            if not np.any(t_mask):
                t_mask = (cluster_labels == k-1) | (cluster_labels == k)
                
            # Slice physical profiles strictly for the transition window
            V = X_all[:, t_mask, 3]            # Shape: (N, T)
            V_dot = X_all_dot[:, t_mask, 3]    # Shape: (N, T)
            w_prev = weights[t_mask, k-1]      # Shape: (T,)
            w_curr = weights[t_mask, k]        # Shape: (T,)
            
            # Analytical derivatives of the mixing weights (gamma')
            dw_prev = w_prev * gauss_exp_term[k-1, t_mask]  
            dw_curr = w_curr * gauss_exp_term[k, t_mask]    
            
            # --- 4. EXACT TENSOR TRANSITION PHYSICS (N x N x T) ---
            # j = previous demo (bin k-1), i = current candidate demo (bin k)
            V_j = V[:, np.newaxis, :]          
            V_i = V[np.newaxis, :, :]          
            V_dot_j = V_dot[:, np.newaxis, :]  
            V_dot_i = V_dot[np.newaxis, :, :]  
            
            # Exact Blended Velocity: v_opt = gamma_prev * v_j + gamma_curr * v_i
            v_blend = w_prev * V_j + w_curr * V_i
            
            # Exact Blended Acceleration: a_opt = a_inherent + a_switching
            a_inherent = w_prev * V_dot_j + w_curr * V_dot_i
            a_switching = (dw_prev * V_j + dw_curr * V_i) * (v_blend / track_length)
            a_total = np.abs(a_inherent + a_switching)
            
            # Extract the worst-case instantaneous acceleration for each (j, i) pair
            max_a_trans = np.max(a_total, axis=2) # Shape: (N, N)
            
            # --- 5. BELLMAN UPDATE ---
            modified_acc_error = switch_gain * (acceleration_max - max_a_trans)
            transition_reward = np.log(1 + np.clip(modified_acc_error, a_min=1e-7-1, a_max=None)) 
            
            scores_transition = prev_dp[:, np.newaxis] + transition_reward 
            parent[k, :] = np.argmax(scores_transition, axis=0)
            dp[k, :] = cluster_scores[:, k] + np.max(scores_transition, axis=0)
            
        # --- 6. BACKTRACK OPTIMAL PATH ---
        best_indices = np.zeros(self.num_segs, dtype=int)
        best_indices[-1] = np.argmax(dp[-1])
        for k in range(self.num_segs - 1, 0, -1):
            best_indices[k - 1] = parent[k, best_indices[k]]            
        return best_indices, cluster_scores
    

    def blend_trajectories(self, X_all, weights, best_indices):
        """
        Synthesizes the final trajectory using a smooth C-infinity convex 
        combination of the selected expert states weighted by GMM responsibilities.
        
        Args:
            X_all (np.ndarray): Full tensor of all expert states.
            weights (np.ndarray): Continuous GMM responsibilities.
            best_indices (list): The optimal expert selected for each bin.
            
        Returns:
            dict: Standardized trajectory dictionary of the synthesized path.
        """
        # Vectorized blending: sum over bins (k) of weight * optimal_expert_state
        X_blended = np.einsum('tk,ktd->td', weights, X_all[best_indices])
        TDOpt = traj_data_dict_get()
        TDOpt.update({
            "name": "Mixed", 
            "states": X_blended, 
            "progress": self.progress
        })
        return TDOpt
           

    def plot_gmm_results(self, 
                        X_all, 
                        features, 
                        weights, 
                        cluster_labels, 
                        feature_labels,
                        best_indices=None,
                        cluster_scores=None,
                        waypoints_plotter=None, 
                        filedate: str = None):
        """
        Generates the comprehensive 5-panel visualization of the MERGE pipeline.
        Includes spatial clustering, feature variance, covariance ellipses, 
        Viterbi DP trellis, and smooth activation weights.
        """
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in range(self.num_segs)]
        custom_cmap = ListedColormap(colors)
        
        fig = plt.figure(figsize=(17, 23)) 
        gs = GridSpec(5, 1, height_ratios=[2.5, 1, 1, 1, 1], hspace=0.3)
        
        # --- 1. XY PATH (Spanning top rows) ---
        plot_index = 0
        ax_xy = fig.add_subplot(gs[plot_index])
        switch_axis = False
        if waypoints_plotter is not None: 
            switch_axis = waypoints_plotter(ax_xy) 
            
        for x in X_all: 
            if switch_axis: X, Y = x[:, 1], x[:, 0]
            else: X, Y = x[:, 0], x[:, 1]
            ax_xy.scatter(X, Y, c=cluster_labels, cmap=custom_cmap,
                          s=50, alpha=0.25, edgecolors='none')

        ax_xy.set_title(f"({chr(97 + plot_index)}) Segmented Track Map")
        ax_xy.axis("equal")

        # --- 2. VARIANCE ENVELOPE ---
        plot_index += 1
        ax_f = fig.add_subplot(gs[plot_index])
        ax_f.scatter(features[::self.downsample, 0], features[::self.downsample, 1], 
                     alpha=0.25, c=feature_labels, cmap=custom_cmap, s=1)
        
        ax_f.set_title(f"({chr(97 + plot_index)}) Track Complexity (Trajectory Deviations)")
        ax_f.set_ylabel("Lateral Deviation [m]")
        ax_f.yaxis.set_label_position("right")
        ax_f.grid(True)
        ax_f.set_xticklabels([]) 
        
        # --- 3. GMM CLUSTERS & COVARIANCES ---
        plot_index += 1
        ax_g = fig.add_subplot(gs[plot_index])
        for i, idx in enumerate(self.sort_idx):
            draw_ellipse(self.means_[idx], self.covariances_[idx], ax=ax_g, 
                        alpha=0.35, color=colors[i], zorder=1)
            ax_g.scatter(self.means_[idx, 0], self.means_[idx, 1], 
                        color=colors[i], marker='x', s=100, lw=2)
        
        ax_g.set_xticklabels([]) 
        ax_g.set_ylabel("Lateral Deviation [m]")
        ax_g.yaxis.set_label_position("right")
        ax_g.set_title(f"({chr(97 + plot_index)}) Learned Clusters from Deviations")

        # --- 4. OPTIMAL SWITCHING TRELLIS (Viterbi DP) ---
        plot_index += 1
        ax_t = fig.add_subplot(gs[plot_index])
        
        if best_indices is not None and cluster_scores is not None:
            num_bins = cluster_scores.shape[1]
            num_controllers = cluster_scores.shape[0]
            norm = Normalize(vmin=np.percentile(cluster_scores, 10), 
                             vmax=np.max(cluster_scores))
            
            X_grid, Y_grid = np.meshgrid(np.arange(num_bins), np.arange(num_controllers))
            sc = ax_t.scatter(X_grid.flatten(), Y_grid.flatten(), 
                            c=cluster_scores.flatten(), cmap='RdYlGn', norm=norm,
                            s=100, alpha=0.6, edgecolors='grey')
            
            # Plot the selected global optimal sequence path
            ax_t.plot(np.arange(num_bins), best_indices, 
                    color='blue', linewidth=2, marker='o', 
                    label='Optimal Sequence', alpha=0.4)
            
            ax_t.scatter(np.arange(num_bins), best_indices, 
                        color='none', edgecolors='blue', s=200, 
                        linewidth=2, alpha=0.4)

            ax_t.set_title(f"({chr(97 + plot_index)}) Optimal Switching Trellis")
            ax_t.set_ylabel("Demonstrations per Bin")
            ax_t.legend(loc='lower center')
            ax_t.set_xticklabels([])
            ax_t.set_yticklabels([])
            ax_t.grid(True)
            
            # Add colorbar for score context
            cbar = fig.colorbar(sc, ax=ax_t, fraction=0.025, pad=0.01)
            cbar.set_label('Demonstration Score', rotation=270, labelpad=20)
            cbar.outline.set_visible(False)

        # --- 5. ACTIVATION WEIGHTS (gamma) ---
        plot_index += 1
        ax_w = fig.add_subplot(gs[plot_index])
        for i in range(self.num_segs):
            ax_w.plot(self.progress, weights[:, i], color=colors[i], lw=2)
            ax_w.fill_between(self.progress, 0, weights[:, i], color=colors[i], alpha=0.25)
            
        ax_w.set_title(f"({chr(97 + plot_index)}) Activation Weights for Smooth Transitions")
        ax_w.set_ylabel("Marginalised Gaussians")
        ax_w.set_xlabel("Normalised Track Progress")
        ax_w.yaxis.set_label_position("right")
        ax_w.set_xlim(0, 1)
        ax_w.grid(True)
        
        # Title for the plot
        fig.suptitle("GMM-based Smart Binning & Switching Optimization", 
                     fontsize=27, fontweight="bold")
        
        plt.subplots_adjust(top=0.94)
        if filedate is not None:
            plt.savefig(join("data_dummy", f"OPT_TRAJ_GMM_{filedate}.png"), dpi=300, bbox_inches='tight')
        plt.show()


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """
    Helper function to draw an ellipse representing the covariance of a Gaussian.
    Draws 1-sigma and 2-sigma contours based on Singular Value Decomposition.
    """
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    for nsig in np.linspace(1, 2, 3, endpoint=True):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, 
                            angle=angle, **kwargs))


#########################################################################################################
#########################################################################################################