<div align="center">

# MERGE: Multi-Expert Regimes Gaussian Ensemble

**Structure-aware trajectory synthesis from heterogeneous and imperfect demonstrations for high-speed autonomous racing.**

<p>
  <img src="https://img.shields.io/badge/Platform-F1TENTH-blue" alt="Platform">
  <img src="https://img.shields.io/badge/ROS%202-Humble%20%7C%20Iron-22314E" alt="ROS 2">
  <img src="https://img.shields.io/badge/Python-%3E%3D3.8-3776AB" alt="Python">
  <img src="https://img.shields.io/badge/License-TBD-lightgrey" alt="License">
</p>

</div>

---

## Overview

**MERGE (Multi-Expert Regimes Gaussian Ensemble)** is an end-to-end framework for synthesizing safe, continuous, and dynamically consistent racing trajectories from heterogeneous, noisy, and potentially sub-optimal demonstrations.

Rather than averaging demonstrations, MERGE treats them as **localized driving expertise** and selectively recombines the most useful behaviors across the track. Track geometry is decoupled from temporal execution, while spatial safety and transition feasibility are explicitly considered during trajectory synthesis.

</br>
<img width="6023" height="2327" alt="METHODLOGY_ICRA" src="https://github.com/user-attachments/assets/7e866aa8-5c17-4f96-b4eb-7fd6331ae83e" />
</br>

## Repository Structure

```text
.
├── MERGE_HARDWARE/
│   ├── data_dummy/                  # Mock data for headless testing
│   ├── data_hardware_exp/           # Physical vehicle telemetry and experiments
│   ├── data_rosbags_converted/
│   │   ├── DEMO_BAGS/               # Raw multi-expert ROS 2 bags (.db3)
│   │   └── MAP_DATA/                # Track centerlines and boundary models
│   ├── f1tenth_gym_model/           # Vehicle dynamics and actuation configuration
│   ├── MERGE_utils/                 # GMM, DP and trajectory synthesis
│   └── trajectory_utils/            # Frenet projection, KDTree and visualization
│
├── MERGE_SIMULATION/
│   ├── data_dummy/                  # Synthetic test data
│   ├── data_gym_playground/         # Interactive simulation outputs
│   ├── data_gym_ros_exp/            # Multi-lap benchmark logs
│   ├── f1tenth_gym_model/           # Dynamic single-track vehicle models
│   ├── f1tenth_racetracks/          # F1TENTH track maps
│   ├── MERGE_utils/                 # Simulation-specific MERGE handlers
│   └── trajectory_utils/            # Spline and plotting utilities
│
└── MERGE_VS_IMITATION_LEARNING/
    ├── data_1/                      # Evaluation batch 1
    ├── data_2/                      # Evaluation batch 2
    ├── data_dummy/                  # Reference demonstration subsets
    ├── MERGE_utils/                 # Benchmark handlers
    └── trajectory_utils/            # Comparison metrics
```

---

## Module Overview

### `MERGE_HARDWARE`

Complete ROS 2 pipeline for physical deployment.

- Converts and processes logged ROS 2 SQLite3 bags.
- Uses `/amcl_pose` and `/odometry/filtered` telemetry.
- Provides the real-time switching/tracking controller.
- Uses low-latency ROS 2 QoS settings for communication with the vehicle/VESC through `/drive`.

### `MERGE_SIMULATION`

Simulation and verification environment built around the F1TENTH Gym ecosystem.

- Supports multiple global racetracks, including Oschersleben and other standard F1TENTH maps.
- Uses a dynamic single-track vehicle model.
- Includes collision checking, friction parameterization, and tire-dynamics models.
- Provides multi-lap evaluation and trajectory visualization.

### `MERGE_VS_IMITATION_LEARNING`

Benchmark suite for comparing MERGE with imitation-learning baselines: Behavior Cloning, ILEED and PACER.

Evaluation includes:

- Lap/completion time
- Lateral deviation
- Constraint violations
- Acceleration
- Cumulative reward
- Behavior under heterogeneous demonstrations and distribution shift

---

## State Representation

All logged demonstrations and synthesized vehicle trajectories use the following **7-dimensional state**:

| Index | Symbol | Description | Units |
|:---:|:---:|---|:---:|
| 0 | $X$ | Global X position (map frame) | m |
| 1 | $Y$ | Global Y position (map frame) | m |
| 2 | $\delta$ | Front-wheel steering angle | rad |
| 3 | $v$ | Longitudinal forward velocity | m/s |
| 4 | $\psi$ | Heading / yaw | rad |
| 5 | $\dot{\psi}$ | Body-frame yaw rate | rad/s |
| 6 | $\beta$ | Vehicle sideslip angle | rad |

The synthesized reference is represented spatially as a function of progress $s$, enabling trajectory generation to be separated from execution timing.

---

## Installation

### Requirements

- Ubuntu 22.04 LTS
- ROS 2 Humble or Iron
- Python >= 3.8
- F1TENTH simulation/deployment environment

Windows users may use **WSL2** where appropriate.

### Python dependencies

```bash
pip install numpy scipy matplotlib scikit-learn
```

For ROS 2 bag processing:

```bash
pip install rosbag2_py rosidl_runtime_py
```

> **Note:** ROS 2 Python packages are normally installed through the corresponding ROS 2 distribution. Ensure your ROS 2 environment is sourced before running ROS-dependent modules.

---

## Quick Start

### 1. Process Demonstration Bags

Convert, align, and split multi-lap demonstrations into a unified Frenet representation:

```bash
cd MERGE_HARDWARE
python3 data_extractor.py
```

The processed demonstrations can then be supplied to the MERGE synthesis pipeline.

---

### 2. Generate the Optimal Trajectory

```python
from MERGE_utils.handler import generate_opt_trajectory
from trajectory_utils.data import load_traj_data

# Load processed multi-expert demonstrations
demonstrations = load_traj_data(
    "data_rosbags_converted/DEMO_BAGS/Track_Data_Jan_21.pkl"
)

# Offline MERGE trajectory synthesis
optimal_trajectory = generate_opt_trajectory(
    TDList=demonstrations,
    num_segs=5,
    acceleration_max=9.51,
    safety_width=0.35,
    safety_gain=3.5,
    switch_gain=3.5
)
```

### Main synthesis parameters

| Parameter | Description |
|---|---|
| `num_segs` | Number of GMM operational regimes |
| `acceleration_max` | Maximum allowed acceleration |
| `safety_width` | Lateral safety bound |
| `safety_gain` | Weight controlling the spatial safety barrier |
| `switch_gain` | Weight controlling transition/switching behavior |

The exact values should be selected according to the track, vehicle configuration, and experimental setup.

---

### 3. Deploy on Hardware

Launch the ROS 2 tracking node:

```bash
ros2 run f1tenth_control switching_controller_node
```

MERGE trajectory synthesis is performed offline. Online execution tracks the resulting continuous trajectory using the adaptive Stanley controller.

---

## Trajectory Synthesis Details

### Spatial Representation

Each demonstration is transformed from

$\tau_i(t)$

to a spatial representation

$\tau_i(s)=\{X_i(s),v_i(s)\}, \qquad s\in[0,1].$

This allows demonstrations with different speeds and execution times to be compared in a common spatial coordinate system.

### GMM Regime Discovery

MERGE constructs progress-dependent variance features and fits a $K$-component GMM. The posterior responsibility

$\gamma_k(s) = \frac{\pi_k p(s|k)}{\sum_j\pi_jp(s|j)}$

provides smooth regime membership and blending weights.

### Dynamic Programming

For consecutive regimes, MERGE evaluates transition-level spatial and acceleration rewards. Unsafe transitions are assigned a reward of $-\infty$, preventing them from entering the feasible DP solution.

The optimal expert sequence is obtained through Bellman recursion:

$DP(k,i)=\max_j\left[DP(k-1,j)+R_{e,k}^{(j\rightarrow i)}+R_{a,k}^{(j\rightarrow i)}\right].$

### Continuous Synthesis

The selected demonstrations are blended using the GMM responsibilities and subsequently represented by continuous splines for deployment.

---

## Simulation

The simulation pipeline is intended for controlled verification before hardware deployment.

The simulation stack supports dynamic vehicle modeling, friction/tire parameterization, collision checking, and multi-lap evaluation.

---

## Hardware Deployment

MERGE has been deployed on a physical F1TENTH platform.

The controller operates online while trajectory synthesis remains an offline optimization step. This separates the computationally heavier expert selection process from the high-frequency tracking loop.

---

## Evaluation

MERGE can be evaluated against imitation-learning and model-based control baselines using:

- **Lateral deviation** — adherence to the synthesized/reference racing line.
- **Acceleration** — smoothness and actuation feasibility.
- **Completion time** — execution efficiency.
- **Constraint satisfaction** — percentage of trajectory samples satisfying prescribed limits.
- **Total reward** — cumulative trajectory quality under the defined MERGE objective.

The benchmark suite contains experiments designed to expose the failure modes of averaging-based imitation under heterogeneous demonstrations.

---

<!-- ## Citation

If you use MERGE, its trajectory-synthesis pipeline, or the associated F1TENTH experiments in your research, please cite the corresponding paper/repository once the citation information is finalized.

```bibtex
@article{MERGE,
  title   = {MERGE: Multi-Expert Regimes Gaussian Ensemble},
  author  = {<Authors>},
  journal = {<Venue>},
  year    = {<Year>}
}
```

Relevant foundational work includes the F1TENTH platform, Stanley tracking, Gaussian mixture models, and Dynamic Programming.

---

## Acknowledgements

This project builds on the open-source F1TENTH ecosystem and related autonomous-racing and trajectory-tracking research.

---

<div align="center">

**MERGE — From heterogeneous demonstrations to globally consistent, feasible racing trajectories.** -->

</div>
