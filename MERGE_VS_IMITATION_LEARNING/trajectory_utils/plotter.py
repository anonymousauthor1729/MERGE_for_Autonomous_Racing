import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


#########################################################################################################
#########################################################################################################


STATE_NAMES = [
    "X", "Y", 
    "Z", "Velocity",
]

STATE_DOT_NAMES = [
    "Velocity Along X", "Velocity Along Y", 
    "Velocity Along Z", "Acceleration",
]

STATE_LABELS = [
    r"$X$ [m]", r"$Y$ [m]", r"$Z$ [m]", r"$v$ [m/s]"
]

STATE_DOT_LABELS = [
    r"$\dot{X}$ [m]", r"$\dot{Y}$ [m]", 
    r"$\dot{Z}$ [m]", r"$\dot{v}$ [m/s$^2$]"
]


#########################################################################################################
#########################################################################################################


def plot_waypoints(waypoints, waypoints_width, ax):
    X = waypoints[:, 0]
    Y = waypoints[:, 1]
    
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.plot(X, Y, color="k", lw=1.5, alpha=0.75, linestyle="--")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    
    if waypoints_width <= 0.: return
    
    grads = np.gradient(waypoints, axis=0)
    normals = np.column_stack((-grads[:, 1], grads[:, 0]))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-7
    normals *= waypoints_width / 2.0
    
    ax.plot(X + normals[:, 0], Y + normals[:, 1], color="black", lw=1.5, alpha=0.5)
    ax.plot(X - normals[:, 0], Y - normals[:, 1], color="black", lw=1.5, alpha=0.5)


#########################################################################################################
#########################################################################################################


def plot_trajectory_data(waypoints=None,
                         waypoints_width=0.,
                         TDList_learnt=[], 
                         TDList_demo=[],
                         plot_state_indices=[0, 1, 3],
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
    
    fig = plt.figure(figsize=(24, max(12, (num_states + num_states_dot) * 3.5)))
    gs = GridSpec(num_states + num_states_dot, 2, width_ratios=[1, 1], hspace=0.2, wspace=0.05)
    ax_states = fig.add_subplot(gs[:, 0])
        
    color_demo = '#FFB07F'
    color_hero = '#1f77b4'

    if waypoints is not None:
        plot_waypoints(waypoints, waypoints_width, ax_states)

    for i, TD in enumerate(TDList_demo):
        lbl = TD_demo_name if i == 0 else "_nolegend_"
        X, Y = TD["states"][:, 0], TD["states"][:, 1]
        ax_states.plot(X, Y, color=color_demo, lw=1.5, alpha=0.75, linestyle='-', label=lbl)
        ax_states.scatter(X[0], Y[0], color=color_demo, marker='o', s=60, zorder=3, edgecolors='k')
        ax_states.scatter(X[-1], Y[-1], color=color_demo, marker='X', s=60, zorder=3, edgecolors='k')

    for i, TD in enumerate(TDList_learnt):
        lbl = TD_learnt_name if i == 0 else "_nolegend_"
        X, Y = TD["states"][:, 0], TD["states"][:, 1]
        ax_states.plot(X, Y, color=color_hero, lw=3, alpha=0.9, label=lbl, zorder=10)
        ax_states.scatter(X[0], Y[0], color=color_hero, marker='o', s=60, zorder=3, edgecolors='k')
        ax_states.scatter(X[-1], Y[-1], color=color_hero, marker='X', s=60, zorder=3, edgecolors='k')
            
    ax_states.set_title("(a) XY Trajectory of the End-Effector")
    ax_states.legend(framealpha=0.9, fancybox=True)
    ax_states.axis("equal")

    xlabel = "Time [s]" if x_axis_time else "Normalised Trajectory Progress"
    letter_offset = 97
    
    for i, j in enumerate(plot_state_indices):
        ax = fig.add_subplot(gs[i, 1])

        for TD in TDList_demo:
            x_axis = TD["t_steps"] if x_axis_time else TD["progress"]
            ax.plot(x_axis, TD["states"][:, j], color=color_demo, lw=1.5, alpha=0.75)

        for TD in TDList_learnt:
            x_axis = TD["t_steps"] if x_axis_time else TD["progress"]
            ax.plot(x_axis, TD["states"][:, j], color=color_hero, lw=2.0, alpha=0.95)

        ax.set_ylabel(STATE_LABELS[j])
        ax.set_title(f"({chr(letter_offset + i)}) {STATE_NAMES[j]}")
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        if i < len(plot_state_indices)-1: ax.set_xticklabels([])
    
    if len(plot_state_dot_indices) != 0: ax.set_xticklabels([])
    
    for i, j in enumerate(plot_state_dot_indices):
        ax = fig.add_subplot(gs[i+num_states, 1])

        for TD in TDList_demo:
            x_axis = TD["t_steps"] if x_axis_time else TD["progress"]
            states_dot_j = np.gradient(TD["states"][:, j], TD["t_steps"])
            ax.plot(x_axis, states_dot_j, color=color_demo, lw=2.0, alpha=0.75)

        for TD in TDList_learnt:
            x_axis = TD["t_steps"] if x_axis_time else TD["progress"]
            states_dot_j = np.gradient(TD["states"][:, j], TD["t_steps"])
            ax.plot(x_axis, states_dot_j, color=color_hero, lw=2.0, alpha=0.95)

        ax.set_ylabel(STATE_DOT_LABELS[j])
        ax.set_title(f"({chr(65+i+num_states)}) {STATE_DOT_NAMES[j]}")
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        if i < len(plot_state_dot_indices)-1: ax.set_xticklabels([])
    
    ax.set_xlabel(xlabel)
    if title is None:
        if len(TDList_learnt) == 0: title = f"{TD_demo_name}"
        else: title = f"{TD_demo_name} and {TD_learnt_name}"
        
    fig.suptitle(f"{title}", fontsize=27, fontweight="bold")
    plt.subplots_adjust(top=0.9)

    if filedate is not None:
        title_safe = title.strip(" ").replace(" ", "_")
        plt.savefig(join("data_dummy", f"{title_safe}_{filedate}.png"), dpi=300, bbox_inches='tight')
    plt.show()


#########################################################################################################
#########################################################################################################


def plot_multi_algo_trajectories(
    TDList:list, palette:dict={},
    plot_state_indices=[],
    plot_state_dot_indices=[],
    state_lims = None,
    state_dot_lims = None,
    waypoints_width=0.,
    waypoints=None,
    x_axis_time=True,
    filedate=None,
    title=None,
    ):

    global STATE_LABELS, STATE_NAMES, STATE_DOT_LABELS, STATE_DOT_NAMES
    unsafe = "#D62728"

    grouped = {}
    for TD in TDList:
        name = TD.get("name", "Unknown") or "Unknown"
        grouped.setdefault(name, []).append(TD)

    algos = list(grouped.keys())
    num_algos = len(algos)
    num_states = len(plot_state_indices)
    num_states_dot = len(plot_state_dot_indices)
    total_rows = 1 + num_states + num_states_dot

    fig = plt.figure(figsize=(6 * num_algos, max(13, total_rows * 4)))
    gs = GridSpec(total_rows, num_algos, height_ratios=[1.5, 1], hspace=0.3, wspace=0.2)
    xlabel = "Time [s]" if x_axis_time else "Progress"

    for col, algo in enumerate(algos):
        ax = fig.add_subplot(gs[0, col])
        color = palette.get(algo, "gray")

        if waypoints is not None:
            plot_waypoints(waypoints, waypoints_width, ax)

        for TD in grouped[algo]:
            states = np.asarray(TD["states"])
            ax.plot(states[:, 0], states[:, 1], color=color, alpha=0.8)
            ax.scatter(states[0, 0], states[0, 1], color=color, marker='o', s=60, zorder=3, edgecolors='k')
            ax.scatter(states[-1, 0], states[-1, 1], color=color, marker='X', s=60, zorder=3, edgecolors='k')
        
        ax.set_xlabel(STATE_LABELS[0])
        if col == 0: ax.set_ylabel(STATE_LABELS[1])
        else: ax.set_ylabel('')
        
        ax.set_title(f"{algo}", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for i, j in enumerate(plot_state_indices):
        for col, algo in enumerate(algos):
            ax = fig.add_subplot(gs[1 + i, col])
            color = palette.get(algo, "gray")

            for TD in grouped[algo]:
                x_axis = TD["t_steps"] if x_axis_time else TD["progress"]
                ax.plot(x_axis, TD["states"][:, j], color=color, lw=2.5, alpha=0.8)

            if col == 0: ax.set_ylabel(STATE_LABELS[j])
            if i == num_states - 1: ax.set_xlabel(xlabel)
            else: ax.set_xticklabels([])

            if len(state_lims) != 0:
                ax.axhline(state_lims[i][0], linestyle='--', color=unsafe, alpha=0.75)
                ax.axhline(state_lims[i][1], linestyle='--', color=unsafe, alpha=0.75)
                ax.axhspan(state_lims[i][0], ax.get_ylim()[0], color=unsafe, alpha=0.1)
                ax.axhspan(state_lims[i][1], ax.get_ylim()[1], color=unsafe, alpha=0.1)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    for i, j in enumerate(plot_state_dot_indices):
        for col, algo in enumerate(algos):
            ax = fig.add_subplot(gs[1 + num_states + i, col])
            color = palette.get(algo, "gray")

            for TD in grouped[algo]:
                x_axis = TD["t_steps"] if x_axis_time else TD["progress"]
                dt = np.diff(TD["t_steps"])
                dt = np.where(dt <= 0, 1e-6, dt)

                states_dot = np.diff(TD["states"][:, j]) / dt
                ax.plot(x_axis[:-1], states_dot, color=color, lw=2.5, alpha=0.8)

            if col == 0: ax.set_ylabel(f"{STATE_DOT_NAMES[j]}: {STATE_DOT_LABELS[j]}")
            if i == num_states_dot - 1: ax.set_xlabel(xlabel)
            else: ax.set_xticklabels([])

            if len(state_dot_lims) != 0:
                ax.axhline(state_dot_lims[i][0], linestyle='--', color=unsafe, alpha=0.75)
                ax.axhline(state_dot_lims[i][1], linestyle='--', color=unsafe, alpha=0.75)
                ax.axhspan(state_dot_lims[i][0], ax.get_ylim()[0], color=unsafe, alpha=0.1)
                ax.axhspan(state_dot_lims[i][1], ax.get_ylim()[1], color=unsafe, alpha=0.1)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    if title is None:
        fig.suptitle(title, fontsize=30, fontweight="bold")
    plt.subplots_adjust(top=0.9)

    if filedate is not None:
        plt.savefig(join("data_dummy", f"Trajectory_Comparison_{filedate}.png"), dpi=300, bbox_inches='tight')
    plt.show()


#########################################################################################################
#########################################################################################################


def plot_segmented(ax, xy, mask, color, lw=2):
    start = None
    for i in range(len(mask)):
        
        if mask[i] and start is None:
            start = i
        
        elif not mask[i] and start is not None:
            ax.plot(xy[start:i, 0], xy[start:i, 1], color=color, linewidth=lw)
            start = None

    if start is not None:
        ax.plot(xy[start:, 0], xy[start:, 1], color=color, linewidth=lw)


#########################################################################################################
#########################################################################################################


def plot_multi_demos(
    TDList,
    w_safe=0.02,
    waypoints=None,
    waypoints_width=0.,
    filedate=None,
    title=None,
    num_rows = 2
    ):

    num_demos = len(TDList)
    cols = int(np.ceil(num_demos/num_rows))
    fig, axes = plt.subplots(num_rows, cols, figsize=(6 * cols, max(17, num_rows * 4)), 
                             gridspec_kw={'hspace': 0.3, 'wspace': 0.2})
    
    axes = np.array(axes).reshape(-1)
    safe_color = "#4C72B0"   # blue
    unsafe_color = "#D62728" # red
    ax_last = axes[-1]

    for i, TD in enumerate(TDList):
        ax = axes[i]
        states = np.asarray(TD["states"])
        deviations = TD.get("deviations", np.zeros(len(states)))

        safe_mask = deviations <= w_safe
        unsafe_mask = deviations > w_safe

        if waypoints is not None:
            plot_waypoints(waypoints, waypoints_width, ax)

        plot_segmented(ax, states, safe_mask, safe_color, lw=4)
        plot_segmented(ax, states, unsafe_mask, unsafe_color, lw=4)
        ax.scatter(states[0, 0], states[0, 1], color=safe_color, marker='o', s=60, edgecolors='black', zorder=3)
        ax.scatter(states[-1, 0], states[-1, 1], color=safe_color, marker='X', s=70, edgecolors='black', zorder=3)

        plot_segmented(ax_last, states, safe_mask, safe_color, lw=4)
        plot_segmented(ax_last, states, unsafe_mask, unsafe_color, lw=4)
        ax_last.scatter(states[0, 0], states[0, 1], color=safe_color, marker='o', s=60, edgecolors='black', zorder=3)
        ax_last.scatter(states[-1, 0], states[-1, 1], color=safe_color, marker='X', s=70, edgecolors='black', zorder=3)
        ax_last.set_xlabel(STATE_LABELS[0])
        
            
        if i % cols == 0: ax.set_ylabel(STATE_LABELS[1])
        else: ax.set_ylabel('')
        ax.set_title(f"Demo {i+1}")

        ax.axis("equal")
        # ax.spines["top"].set_visible(False)
        # ax.spines["right"].set_visible(False)

    if waypoints is not None:
        plot_waypoints(waypoints, waypoints_width, ax_last)

    if i % cols == 0: ax_last.set_ylabel(STATE_LABELS[1])
    else: ax_last.set_ylabel('')
    ax_last.set_title(f"Demo All")

    ax_last.axis("equal")
    # ax_last.spines["top"].set_visible(False)
    # ax_last.spines["right"].set_visible(False)
    
    for j in range(num_demos+1, len(axes)):
        fig.delaxes(axes[j])

    if title is not None: 
        fig.suptitle(title, fontsize=22, fontweight="bold")
    plt.subplots_adjust(top=0.9)

    if filedate is not None:
        plt.savefig(
            join("data_dummy", f"Demonstrations_{filedate}.png"),
            dpi=300, bbox_inches='tight'
        )
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
    plt.rcParams['grid.alpha'] = 0.75
    plt.rcParams['grid.linestyle'] = '--'


#########################################################################################################
#########################################################################################################