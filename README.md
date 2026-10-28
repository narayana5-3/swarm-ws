# Risk-Prioritized Multi-AUV Swarm for Autonomous Underwater Infrastructure Inspection

A simulation-based multi-Autonomous Underwater Vehicle (AUV) framework for collaborative underwater infrastructure inspection. This project integrates probabilistic 3D occupancy mapping, deep-learning-based crack segmentation, uncertainty-aware damage-risk mapping, CBBA-derived task allocation, and risk-aware hybrid metaheuristic path planning.

**Technology stack:** ROS 2 Jazzy · Gazebo Harmonic · Python · SciPy

**Inspection environment:** Simulated underwater dam and pipe structures

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Software Requirements](#software-requirements)
- [Installation and Setup](#installation-and-setup)
- [Experimental Results](#experimental-results)
- [Limitations](#limitations)
- [Future Work](#future-work)

## Overview

Underwater infrastructure such as dams and pipelines requires periodic inspection to identify cracks and potential structural defects. Manual inspection can be costly and difficult, motivating the development of autonomous inspection systems.

This project investigates a collaborative inspection framework using three simulated AUVs: `auv0`, `auv1`, and `auv2`. Each vehicle collects camera and range observations and uses odometry for mapping and navigation.

The system combines spatial mapping and visual crack detection to estimate structural damage risk. This risk information guides inspection-task assignment and path planning, while a dashboard displays relevant mission information and damage-prognosis outputs.

### Key Features

- **Multi-AUV simulation:** Three BlueROV2-derived vehicle models in a custom underwater Gazebo environment.
- **Collaborative 3D mapping:** Bayesian log-odds voxel occupancy grids with changed-voxel peer updates.
- **Deep-learning crack detection:** Two-stage training using real DeepCrack photographs and simulation-derived crack images.
- **Damage-risk estimation:** Fusion of damage probability, observation uncertainty, and confirmed structural geometry.
- **Adaptive task allocation:** CBBA-derived sequential auction with distributed max-consensus.
- **Hybrid path planning:** Improved Nutcracker Optimization Algorithm (INOA) with Sequential Quadratic Programming (SQP) refinement.
- **Damage prognosis:** Severity-based urgency estimates and linear growth extrapolation when previous inspection data are available.
- **Integrated dashboard:** Visualization of camera feeds, shared risk maps, swarm status, and prognosis outputs.

## Methodology

### 1. Simulation and Vehicle Configuration

The simulation uses ROS 2 Jazzy and Gazebo Harmonic. The environment contains a dam wall, a pipe structure, and seafloor terrain.

Each AUV uses a custom BlueROV2-derived box-hull model with a reported mass of 44.6 kg and five axis-aligned thrusters for surge, sway, and heave actuation.

| Parameter | Configuration |
|---|---|
| Number of AUVs | 3 |
| Camera resolution | 640 × 480 pixels |
| Camera update rate | 15 Hz |
| Camera horizontal field of view | 90° |
| Range sensor | Gazebo GPU LiDAR configured for sonar-like observations |
| Range samples | 64 horizontal × 4 vertical |
| Range horizontal field of view | 120° |
| Range update rate | 10 Hz |
| Range interval | 1–40 m |

These parameters describe the simulation configuration, not a physical underwater sonar instrument.

### 2. Collaborative Occupancy Mapping

Each AUV maintains a probabilistic 3D occupancy grid with a voxel resolution of **0.5 m**.

The mapping component uses Bayesian log-odds updates to represent occupied and free space. Range observations update voxels along a sensor ray, and odometry provides vehicle pose information for spatial mapping.

Key parameters:

- Free-space update: `-0.4`
- Occupied-hit update: `+0.85`
- Log-odds clamp: `[-10, 10]`
- Changed-voxel broadcast threshold: `0.05`
- Simulated peer communication radius: `60 m`

Agents exchange changed voxel evidence. Each receiving agent retains the latest known snapshot from each peer rather than repeatedly accumulating duplicate copies of a peer's previous map update.

### 3. Crack Segmentation

The camera-based segmentation pipeline produces pixel-wise crack probability maps.

Three architectures are compared in the report:

- **Small U-Net:** Compact encoder–decoder trained from scratch.
- **ResNet-34 U-Net:** U-Net decoder using an ImageNet-pretrained ResNet-34 encoder.
- **SegFormer-B0:** Transformer-based semantic segmentation model.

The training workflow consists of two stages:

1. **Initial training:** Use real, pixel-annotated crack photographs from the public DeepCrack dataset.
2. **Simulation-domain fine-tuning:** Fine-tune the model using Gazebo camera images with synthetic crack annotations, including undamaged frames.

Training combines binary cross-entropy loss and Dice loss to address crack-pixel class imbalance.

Performance is evaluated using Dice score, Intersection over Union (IoU), precision, and recall.

### 4. Damage Probability and Risk Mapping

Pixel-wise crack probabilities are approximately associated with range-sensor azimuth columns and fused into a damage-probability grid. Observation counts are used to estimate uncertainty:

\[
U(\mathbf{v}) = \exp\left(-\frac{N(\mathbf{v})}{3.0}\right)
\]

The resulting risk score combines damage probability and uncertainty while restricting risk to the vicinity of confirmed structural geometry:

\[
R(\mathbf{v}) =
P_{\mathrm{damage}}(\mathbf{v}) U(\mathbf{v})
\mathbb{1}[\mathbf{v}\text{ lies within the structural mask}]
\]

The structural mask uses occupancy probability greater than `0.6` and a dilation radius of six voxels, approximately 3 m.

The camera-to-range association is an approximation; it is not a calibrated full 3D camera projection.

### 5. CBBA-Derived Adaptive Task Allocation

The task allocator uses a **sequential single-task auction with distributed max-consensus**, derived from the Consensus-Based Bundle Algorithm (CBBA).

- Extract candidate inspection targets from local maxima in the risk map.
- Retain task peaks above a risk threshold of `0.05`.
- Calculate task values from accumulated local risk and normalize the highest value to `100`.
- Compute each AUV's bid using task value, additional route distance, and a deterministic tie-break offset.
- Exchange bids between agents within `60 m` and perform max-consensus for up to `20` rounds.
- Confirm the highest agreed agent–task bid and update the selected AUV's route.
- Repeat until the task pool is exhausted or each AUV reaches its capacity of four tasks.

**Implementation distinction:** This is CBBA-derived allocation, not a full implementation of the original CBBA bundle-building and task-release process.

### 6. Risk-Aware Hybrid Path Planning

The path planner adapts an **Improved Nutcracker Optimization Algorithm (INOA)–SQP hybrid** to the fused occupancy and risk grids.

The path cost considers:

- Path length.
- Collision penalty based on occupancy probability.
- Turning-angle smoothness.
- Mean risk along the path.
- Excursions beyond environment boundaries.

Additional optimization mechanisms include:

- Adaptive fitness smoothing.
- Rank-dependent Metropolis acceptance.
- **Chaotic Dynamic Lens Opposition-Based Learning (CDL-LOBL)** when a candidate stagnates for five iterations.
- Bounded **Sequential Quadratic Programming (SLSQP)** refinement when the global best fails to improve for 15 iterations.
- Cubic-spline smoothing followed by collision verification, with fallback to a verified piecewise-linear path if smoothing introduces a collision.

The reported live configuration uses a population size of `8` and a maximum of `15` optimization iterations to reduce planning time.

### 7. Damage Prognosis

Damage detections are aggregated using connected-component labeling on voxels with damage probability greater than `0.5` within confirmed structure, where occupancy probability exceeds `0.6`. Each connected region is represented by its peak-severity voxel.

Two prognosis modes are supported:

- **Previous inspection available:** Estimate damage severity growth rate and linearly extrapolate time to a critical severity threshold of `0.9`.
- **No comparable previous inspection:** Assign a severity-based urgency band.

| Severity threshold | Urgency band | Suggested reinspection |
|---|---|---|
| ≥ 0.90 | Critical | 7 days |
| ≥ 0.75 | High | 30 days |
| ≥ 0.60 | Moderate | 90 days |
| ≥ 0.50 | Low | 180 days |

These intervals are illustrative engineering heuristics, not calibrated structural-safety criteria. Linear extrapolation is an approximate estimate and does not model nonlinear crack propagation.

## Software Requirements

The documented core stack includes:

- Ubuntu with **ROS 2 Jazzy**
- **Gazebo Harmonic**
- ROS–Gazebo integration packages required by the workspace
- Python and **SciPy** for optimization and SLSQP refinement
- The machine-learning framework and dependencies used by the crack-segmentation implementation
- DeepCrack data, simulation-generated evaluation data, and any model checkpoints required by the scripts

The project report does not list the exact versions of every Python dependency or the repository's package and launch-file names. Verify those details against the actual repository before reproducing the full system.

## Installation and Setup

### 1. Install prerequisites

Install ROS 2 Jazzy, Gazebo Harmonic, and the ROS–Gazebo integration required by your workspace. Follow the installation instructions for the versions used by the project.

### 2. Build the ROS 2 workspace

From the repository's ROS 2 workspace root, the standard build sequence is:

```bash
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

These commands assume a conventional ROS 2 workspace with packages under `src/`. Adapt them if the repository has a different layout or uses an isolated Python environment.

### 3. Launch the simulation

Use the package and launch-file names actually provided in the repository. The general ROS 2 launch syntax is:

```bash
ros2 launch <package_name> <launch_file.launch.py>
```

The names shown above are placeholders, not ready-to-run commands. Check the project's launch files and documentation for the correct entry point and the required order for starting Gazebo, AUV controllers, mapping, perception, risk fusion, task allocation, path planning, and the dashboard.

### 4. Run crack-model evaluation

Use the training and evaluation scripts included in the repository. Keep training and held-out evaluation file lists separate. Record the model architecture, checkpoint, dataset split, threshold, and dependency versions when reproducing metrics.

## Experimental Results

### Optimizer Benchmark

The NOA-derived optimizer was compared with the Grey Wolf Optimizer (GWO) and Particle Swarm Optimization (PSO) on five selected CEC 2017 functions: **F1, F3, F5, F9, and F13**.

Each experiment used 10 dimensions and 10 independent trials per algorithm and function. Statistical comparison used the Wilcoxon rank-sum test.

| Function | Baseline | p-value | Reported result |
|---|---|---:|---|
| F1 | GWO | 0.821 | Not statistically significant |
| F3 | GWO | 0.019 | NOA-derived method significantly better |
| F5 | PSO | 0.496 | Not statistically significant |
| F9 | PSO | 0.151 | Not statistically significant |
| F13 | PSO | 0.0025 | NOA-derived method significantly better |

The NOA-derived optimizer showed statistically significant improvements on two of the five selected benchmark functions. These tests assess numerical optimization performance, not underwater path safety or quality directly.

### Task Allocation

The CBBA-derived allocator was compared with greedy nearest-task-first and value-sorted round-robin assignment over 30 trials using simulation-captured risk grids and randomized AUV start positions. Each allocator was limited to four tasks per AUV.

| Allocator | Risk value captured | Workload standard deviation | Travel/value |
|---|---:|---:|---:|
| CBBA-derived | 1185.9 ± 7.8 | 1.77 | 0.171 |
| Greedy | 899.9 ± 80.3 | 31.89 | 0.114 |
| Round-robin | 1186.5 ± 7.9 | 0.65 | 0.413 |

In the reported trials, the CBBA-derived allocator captured substantially more risk value than the greedy baseline and had a much smaller workload standard deviation. Round-robin achieved a marginally higher raw risk value and lower workload imbalance, but had a higher travel-cost-per-value ratio. The results show a trade-off between risk captured, travel efficiency, and workload balance.

### Crack Segmentation

| Evaluation set | Images | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| DeepCrack pretrained, held-out real images | 80 | 0.824 | 0.718 | 0.780 | 0.900 |
| Simulation-fine-tuned, crack-containing subset | 18 | 0.740 | 0.598 | 0.602 | 0.986 |
| Simulation-fine-tuned, full held-out set | 33 | 0.586 | — | — | — |

The simulation-fine-tuned model achieved high recall on the 18 crack-containing images, while its lower precision indicates additional predicted crack pixels outside reference masks. The full-set Dice score should be interpreted separately because that set includes undamaged images and the evaluation's empty-mask convention affects aggregation.

These metrics measure agreement with image annotations. They do not measure crack depth, structural integrity, or engineering safety.

### Integrated Simulation

The three AUVs and main software components were run together in a simulated session. The dashboard displayed the shared risk map, heuristic prognosis, camera feeds, and swarm status.

The report recorded:

- **Real-time factor:** approximately `0.20–0.26` on the tested machine.
- **Task re-auction interval:** `300` simulated seconds.

Runtime may vary with hardware and simulation load.

## Limitations

- The framework has been evaluated in simulation; physical AUV trials are outside the current scope.
- GPU LiDAR generates sonar-like range observations but does not simulate full acoustic propagation.
- Inter-agent communication uses a software distance constraint, not a physical acoustic modem network with latency, bandwidth, and packet loss.
- Camera crack probabilities are associated approximately with range-sensor azimuth columns rather than through calibrated 3D camera-to-range projection.
- Occupancy mapping does not include pose-graph optimization or loop closure.
- Ocean-current effects are not included in the path-planning cost.
- Optimization benchmarking covers five selected CEC 2017 functions, not the complete suite.
- Damage urgency levels are heuristic; linear prognosis requires validation against repeated inspections and engineering labels.
- The integrated simulation ran below real time on the reported machine.

## Future Work

Potential extensions include:

- Validate vehicle and sensor models on physical AUV platforms.
- Introduce realistic sonar and acoustic communication models.
- Improve camera-to-range projection using calibrated 3D geometry.
- Incorporate ocean-current estimates into route planning.
- Expand optimizer benchmarking and measure map accuracy against known ground truth.
- Add mission-level coverage accounting.
- Validate damage prognosis using repeated inspections and expert engineering labels.

## Dataset and Licensing Notes

The crack-segmentation workflow uses the public DeepCrack dataset for initial training and simulation-derived images with synthetic crack annotations for fine-tuning. Follow the original dataset's access terms and license. Document the licenses for any additional datasets, model checkpoints, and third-party dependencies before redistributing them.
