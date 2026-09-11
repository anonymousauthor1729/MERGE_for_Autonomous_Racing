import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec


#########################################################################################################
#########################################################################################################


STATE_NAMES = [
    "X Coordinate", "Y Coordinate", 
    "Steering Angle", "Linear Velocity",
    "Orientation", "Angular Velocity",
    "Slip"
]

STATE_DOT_NAMES = [
    "Velocity along X", "Velocity along Y", 
    "Steering Velocity", "Linear Acceleration",
    "Orientation Velocity", "Angular Acceleration",
    "Rate of change of Slip"
]


STATE_LABELS = [
    r"$X$ [m]", r"$Y$ [m]", 
    r"$\delta$ [rad]", r"$v$ [m/s]", 
    r"$\psi$ [rad]", r"$\omega$ [rad/s]", 
    r"$\beta$ [rad]"
]

STATE_DOT_LABELS = [
    r"$\dot{X}$ [m/s]", r"$\dot{Y}$ [m/s]", 
    r"$\dot{\delta}$ [rad/s]", r"$\dot{v}$ [m/s$^2$]", 
    r"$\dot{\psi}$ [rad/s]", r"$\dot{\omega}$ [rad/s$^2$]", 
    r"$\dot{\beta}$ [rad/s]"
]

#########################################################################################################
#########################################################################################################


def plot_waypoints(waypoints, waypoints_width, ax):
    lims = np.max(np.abs(
        waypoints - np.mean(waypoints, axis=0)[None]
    ), axis=0)

    if lims[1] > lims[0]:
        switch_axis = True
        X = waypoints[:, 1]
        Y = waypoints[:, 0]
        ax.set_xlabel("Y [m]")
        ax.set_ylabel("X [m]")
    else:
        switch_axis = False
        X = waypoints[:, 0]
        Y = waypoints[:, 1]
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")

    ax.plot(X, Y, color="grey", lw=1, alpha=0.85, linestyle="--")
    ax.set_title("(a) XY Trajectory of the Car")
    ax.grid(True, alpha=0.3)
    plt.axis("equal")
    
    if waypoints_width <= 0.: return
    grads = np.gradient(waypoints, axis=0)
    normals = np.column_stack((-grads[:, 1], grads[:, 0]))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-7
    normals *= waypoints_width / 2.0
    
    if switch_axis: normals = normals[:, ::-1]
    ax.plot(X + normals[:, 0], Y + normals[:, 1], color="black", lw=1, alpha=0.5)
    ax.plot(X - normals[:, 0], Y - normals[:, 1], color="black", lw=1, alpha=0.5)
    return switch_axis


#########################################################################################################
#########################################################################################################


def plot_trajectory_data(waypoints=None,
                         waypoints_width=None,
                         TDList_learnt=[], 
                         TDList_demo=[],
                         plot_state_indices = [3],
                         plot_state_dot_indices = [],
                         TD_learnt_name: str = "Proposed Method",
                         TD_demo_name: str = "Demonstrations",
                         x_axis_time: bool = True,
                         filedate: str = None,
                         title: str = None,
                        ):

    global STATE_LABELS, STATE_NAMES
    num_states = len(plot_state_indices)
    num_states_dot = len(plot_state_dot_indices)
    num_extra_plots = 1 if len(TDList_learnt) == 0 else 2
    
    if num_states > len(STATE_LABELS): 
        STATE_LABELS += [f"State {i}" for i in range(len(STATE_LABELS), num_states)]
    
    fig = plt.figure(
        figsize=(17, num_extra_plots*10 + (num_states + num_states_dot)*3.5)
    )
    gs = GridSpec(
        num_states + num_states_dot + num_extra_plots, 1, 
        height_ratios=[3]*num_extra_plots + [1]*(num_states + num_states_dot), hspace=0.27
    )
    
    ax_xy = fig.add_subplot(gs[0])
    if num_extra_plots == 2: ax_xy_v = fig.add_subplot(gs[1])
    color_demo = '#FFB07F'
    color_hero = '#1f77b4'

    switch_axis = False
    if waypoints is not None and waypoints_width is not None:
        switch_axis = plot_waypoints(waypoints, waypoints_width, ax_xy)
        if num_extra_plots == 2: plot_waypoints(waypoints, waypoints_width, ax_xy_v)

    for i, TD in enumerate(TDList_demo):
        lbl = TD_demo_name if i == 0 else "_nolegend_"
        if switch_axis: X, Y = TD["states"][:, 1], TD["states"][:, 0]
        else: X, Y = TD["states"][:, 0], TD["states"][:, 1]
        ax_xy.plot(X, Y, color=color_demo, lw=1.5, alpha=0.85, linestyle='-', label=lbl)

    for i, TD in enumerate(TDList_learnt):
        lbl = TD_learnt_name if i == 0 else "_nolegend_"
        if switch_axis: X, Y = TD["states"][:, 1], TD["states"][:, 0]
        else: X, Y = TD["states"][:, 0], TD["states"][:, 1]
        ax_xy.plot(X, Y, color=color_hero, lw=3, alpha=0.9, label=lbl, zorder=10)
        
        norm = Normalize(
            vmin=np.percentile(TD["states"][:, 3], 10), vmax=np.max(TD["states"][:, 3])
        )
        sc = ax_xy_v.scatter(
            X, Y, c=TD["states"][:, 3], cmap='RdYlGn', norm=norm, s=50, alpha=0.25, edgecolors='grey'
        )
        
    ax_xy.legend(loc='upper right', framealpha=0.9, fancybox=True)
    ax_xy.axis("equal")
    
    if num_extra_plots == 2: 
        ax_xy_v.set_title("(b) Car's Velocity Profile (Deployed)")
        cbar = fig.colorbar(sc, ax=ax_xy_v, fraction=0.025, pad=0.01)
        cbar.set_label('Velocity Magnitude [m/s]', rotation=270, labelpad=25)
        cbar.outline.set_visible(False)
        ax_xy_v.axis("equal")

    xlabel = "Time [s]" if x_axis_time else "Normalised Track Progress"
    for i, j in enumerate(plot_state_indices):
        ax = fig.add_subplot(gs[i+num_extra_plots])

        for TD in TDList_demo:
            x_axis = TD["t_steps"] if x_axis_time else TD["progress"]
            ax.plot(x_axis, TD["states"][:, j], color=color_demo, lw=1.5, alpha=0.85)

        for TD in TDList_learnt:
            x_axis = TD["t_steps"] if x_axis_time else TD["progress"]
            ax.plot(x_axis, TD["states"][:, j], color=color_hero, lw=2.0, alpha=0.95)

        ax.set_ylabel(STATE_LABELS[j])
        ax.set_title(f"({chr(97+i+num_extra_plots)}) {STATE_NAMES[j]}")
        ax.yaxis.set_label_position("right")
        if i != len(plot_state_indices) - 1: ax.set_xticklabels([])

    for i, j in enumerate(plot_state_dot_indices):
        ax.set_xticklabels([])
        ax = fig.add_subplot(gs[i+num_extra_plots+num_states])

        for TD in TDList_learnt:
            if len(TD["t_steps"]) == 0: continue
            x_axis = TD["t_steps"] if x_axis_time else TD["progress"]
            states_dot_j = np.gradient(TD["states"][:, j], TD["t_steps"])
            ax.plot(x_axis, states_dot_j, color=color_hero, lw=2.0, alpha=0.95)

        for TD in TDList_demo:
            if len(TD["t_steps"]) == 0: continue
            x_axis = TD["t_steps"] if x_axis_time else TD["progress"]
            states_dot_j = np.gradient(TD["states"][:, j], TD["t_steps"])
            ax.plot(x_axis, states_dot_j, color=color_demo, lw=1.5, alpha=0.85)

        ax.set_ylabel(STATE_DOT_LABELS[j])
        ax.set_title(f"({chr(97+i+num_extra_plots+num_states)}) {STATE_DOT_NAMES[j]}")
        ax.yaxis.set_label_position("right")

    ax.set_xlabel(xlabel)
    if title is None: title = TD_demo_name + (
        f" and {TD_learnt_name}" if len(TDList_learnt) else ""
    )        
    fig.suptitle(f"{title}", fontsize=27, fontweight="bold")
    if num_extra_plots == 2: plt.subplots_adjust(top=0.94)
    else: plt.subplots_adjust(top=0.92)

    if filedate is not None:
        title = title.strip(" ").replace(" ", "_")
        plt.savefig(join("data_dummy", f"{title}_{filedate}.png"), dpi=300, bbox_inches='tight')
    plt.show()
    

#########################################################################################################
#########################################################################################################


def set_publication_style():
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
    plt.rcParams['grid.alpha'] = 0.85
    plt.rcParams['grid.linestyle'] = '--'


#########################################################################################################
#########################################################################################################
