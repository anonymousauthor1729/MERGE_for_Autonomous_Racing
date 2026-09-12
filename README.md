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

**MERGE** is a trajectory-synthesis framework that recombines heterogeneous demonstrations into a single safe, continuous, and dynamically consistent trajectory.

Instead of averaging conflicting demonstrations, MERGE:

- aligns demonstrations by **spatial progress** rather than execution time;
- discovers localized driving regimes from **cross-track variance**;
- selects a globally consistent sequence of expert behaviors using **constraint-aware Dynamic Programming**;
- smoothly blends neighboring experts using **GMM responsibilities**; and
- produces a continuous trajectory for **simulation and physical F1TENTH deployment**.

The complete methodology is summarized below.

</br>
<img width="6023" height="2327" alt="METHODLOGY_ICRA" src="https://github.com/user-attachments/assets/7e866aa8-5c17-4f96-b4eb-7fd6331ae83e" />
</br>

---

## Hardware Deployment

MERGE trajectory synthesis is performed **offline**, while trajectory tracking runs online at the vehicle control rate. The hardware stack handles sensing, localization, trajectory projection, spline evaluation, control, and VESC actuation.

</br>
<img width="6023" height="2327" alt="HW_PIPIELINE_ICRA" src="https://github.com/user-attachments/assets/218cc677-771a-4828-9f6b-07e6940e176b" />
</br>

The physical deployment uses ROS 2 telemetry together with the vehicle's localization and state-estimation stack. The resulting continuous trajectory is tracked using a velocity-scheduled Stanley controller.

---

## Method at a Glance

| Stage | Purpose | Main component |
|---|---|---|
| **Spatial alignment** | Remove differences in execution timing | Frenet re-parameterization |
| **Regime discovery** | Identify localized driving behavior | Variance-aware GMM |
| **Expert selection** | Find a globally feasible combination | Bellman Dynamic Programming |
| **Trajectory synthesis** | Avoid discontinuous expert switching | Probabilistic blending |
| **Execution** | Track the synthesized trajectory | Continuous splines + adaptive Stanley |

The key optimization is performed **between neighboring operational regimes**, where candidate expert transitions are evaluated using spatial safety and switching-induced acceleration constraints.

---

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

### Where to look

| Directory | Role |
|---|---|
| `MERGE_HARDWARE` | ROS 2 processing and physical F1TENTH deployment |
| `MERGE_SIMULATION` | Dynamic simulation and multi-lap experiments |
| `MERGE_VS_IMITATION_LEARNING` | Baseline comparison and evaluation |

---

## Data Representation

Demonstrations are converted from time-indexed vehicle trajectories to a common spatial representation using normalized progress

\[
s \in [0,1].
\]

This separates **trajectory geometry** from the speed at which it was executed and enables different demonstrations to be compared and recombined at corresponding locations along the track.

The vehicle state used throughout the logged demonstrations and synthesized trajectories is:

| Index | Symbol | Description | Units |
|:---:|:---:|---|:---:|
| 0 | $X$ | Global X position (map frame) | m |
| 1 | $Y$ | Global Y position (map frame) | m |
| 2 | $\delta$ | Front-wheel steering angle | rad |
| 3 | $v$ | Longitudinal forward velocity | m/s |
| 4 | $\psi$ | Heading / yaw | rad |
| 5 | $\dot{\psi}$ | Body-frame yaw rate | rad/s |
| 6 | $\beta$ | Vehicle sideslip angle | rad |

---

## Implementation

### `MERGE_HARDWARE`

ROS 2 pipeline for physical F1TENTH experiments.

- Processes logged ROS 2 SQLite3 bags.
- Uses `/amcl_pose` and `/odometry/filtered` telemetry.
- Provides the real-time switching/tracking controller.
- Communicates with the vehicle through `/drive` using low-latency ROS 2 QoS settings.
- Uses the F1TENTH vehicle dynamics and actuation configuration contained in `f1tenth_gym_model/`.

### `MERGE_SIMULATION`

Simulation and verification environment.

- Supports F1TENTH racetracks including Oschersleben.
- Uses a dynamic single-track vehicle model.
- Includes collision checking and vehicle/tire-dynamics configuration.
- Provides multi-lap execution and trajectory visualization.

### `MERGE_VS_IMITATION_LEARNING`

Evaluation suite for comparison against:

- Behavioral Cloning (BC)
- ILEED
- PACER

The benchmark utilities compute trajectory quality, constraint satisfaction, execution time, and reward-based metrics.

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

> **Note:** ROS 2 Python packages are normally installed through the corresponding ROS 2 distribution. Source the appropriate ROS 2 environment before running ROS-dependent modules.

---

## Quick Start

### 1. Process Demonstrations

Convert and align the logged multi-expert demonstrations:

```bash
cd MERGE_HARDWARE
python3 data_extractor.py
```

This produces the processed trajectory representation used by the synthesis pipeline.

### 2. Generate a Trajectory

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

### Main parameters

| Parameter | Description |
|---|---|
| `num_segs` | Number of GMM operational regimes |
| `acceleration_max` | Maximum allowed acceleration |
| `safety_width` | Lateral safety bound |
| `safety_gain` | Weight controlling the spatial safety objective |
| `switch_gain` | Weight controlling transition/switching behavior |

The values above reproduce the demonstrated example configuration; tune them according to the track and vehicle setup.

### 3. Run the Hardware Controller

```bash
ros2 run f1tenth_control switching_controller_node
```

Trajectory synthesis remains offline. The online node receives the vehicle state, evaluates the continuous reference, and tracks it through the adaptive Stanley controller.

---

## Evaluation

The repository contains simulation, hardware, and imitation-learning benchmark experiments.

| Metric | What it measures |
|---|---|
| **Lateral deviation** | Adherence to the synthesized/reference racing line |
| **Acceleration** | Smoothness and actuation feasibility |
| **Completion time** | Racing performance |
| **Constraint satisfaction** | Fraction of trajectory samples within prescribed limits |
| **Total reward** | Overall quality under the MERGE objective |

The experiments are designed around a central failure mode of heterogeneous imitation: **combining demonstrations by averaging can destroy useful local behavior and violate trajectory constraints**. MERGE instead evaluates candidate expert transitions before constructing the final continuous trajectory.

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

---

## Acknowledgements

This project builds on the open-source F1TENTH ecosystem and related autonomous-racing and trajectory-tracking research.

---
-->
