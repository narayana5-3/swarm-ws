# HANDOFF: Risk-Prioritized Distributed Belief-Fusion Swarm
## Migrating to ROS2 + Gazebo + Project DAVE

**Read this whole document before doing anything.** This is a complete
context transfer for continuing this project in a new conversation. The
previous conversation built a full working pipeline in HoloOcean; this
document explains what's being kept, what's being replaced, and exactly
what to build next.

---

## 1. Project Goal (unchanged)

A multi-AUV swarm system for detecting structural damage (cracks) in
underwater infrastructure (dams, pipelines). Architecture: multiple AUVs
collaboratively map structure, independently detect damage via AI, share/
fuse beliefs over a simulated acoustic network, and use risk-prioritized,
uncertainty-driven task allocation to decide where to inspect next --
synthesized into a digital twin visualization with a degradation prognosis
("how long until this becomes a big problem").

**Deliverables, in priority order:**
1. **Live demonstration** (top priority) -- a judging panel needs to watch
   the swarm move, detect damage, and auto-reallocate tasks live, with a
   web dashboard showing damage images + time-to-critical estimates.
2. Research paper (secondary priority, still in progress).

---

## 2. THE PIVOT: why this document exists

The project was originally built in **HoloOcean** (Unreal Engine-based
underwater simulator). That work is real, validated, and produced genuine
results (see Section 6). However: **the live demo requirement is now
"the simulation must run and be visibly shown inside Gazebo."** HoloOcean
cannot satisfy that (it renders in its own Unreal Engine viewport, not
inside Gazebo). HoloOcean is being dropped as the primary/demo simulator.

**Confirmed via research (Nov 2025 sources) that this is actually
feasible without losing capability**: Project DAVE
(https://field-robotics-lab.github.io/dave.doc/, docs at
https://dave-ros2.notion.site/) is an actively maintained, GSoC-funded
(2024 AND 2025) underwater robotics simulator built specifically for
**ROS2 Jazzy + Gazebo Harmonic** -- the exact ROS2/Gazebo combo already
installed on the target machine. As of a November 2025 GSoC project, DAVE
shipped a **physics-based, GPU-accelerated multibeam sonar plugin**
(CUDA-optimized) -- genuine sonar simulation, not a compromise. This means
Gazebo can host this project's actual sonar-based architecture, not a
downgraded version of it.

**New target stack: ROS2 Jazzy Jalisco + Gazebo Harmonic + Project DAVE,
on Ubuntu 24.04 (WSL2).**

---

## 3. What's REUSABLE (build on this, don't rebuild it)

These files are simulator-agnostic -- pure algorithms/math/ML operating on
data structures, not HoloOcean API calls. They should be brought into the
new chat and used as-is or with minor interface adaptation.

### Crack detection (`damage_detection/` folder -- entire thing reusable)
- `model.py`, `model_factory.py` -- three architectures behind one
  interface: `small_unet` (from scratch), `transfer_unet` (pretrained
  ResNet34 encoder via `segmentation_models_pytorch`), `vit_segformer`
  (pretrained SegFormer via HuggingFace `transformers`). Fully tested.
- `crack_injection.py` -- procedural crack generator, composites onto ANY
  source image with pixel-accurate ground truth masks. Not HoloOcean-
  specific at all -- point it at Gazebo-captured frames once available.
- `train.py` -- training loop, BCE+Dice loss, augmentation, proper held-out
  validation (saves the held-out file list explicitly -- this was added
  after catching a real data-leakage bug, don't remove it). Supports
  `--init_from_checkpoint` for the two-stage pretrain-then-finetune
  workflow, with auto-selected learning rate (1e-3 from scratch, 1e-4 when
  warm-starting -- this was a real bug found via testing: warm-starting at
  the wrong LR destabilized an already-converged checkpoint).
- `eval_visualize.py` -- honest evaluation against held-out data, auto-
  detects the held-out list. Never evaluate on the training folder itself.
- `compare_algorithms.py` -- multi-trial statistical comparison (Wilcoxon
  rank-sum) across all model configs, including an optional 4th arm for a
  DeepCrack-pretrained checkpoint.
- `prepare_deepcrack.py` -- converts the real DeepCrack dataset (537 real
  pixel-annotated crack photos) into the `images/`+`masks/` format the
  rest of the pipeline expects.
- `real_dataset/deepcrack_dataset.zip` -- the actual prepared DeepCrack
  data, ready to use, no download needed.

**Known real result from testing**: DeepCrack-pretrained + fine-tuned
`transfer_unet` scored 0.9153 Dice (single run, promising but not yet
statistically validated across trials) vs. 0.8292+/-0.0311 for ImageNet-only
pretraining (5-trial mean). A 4-trial comparison run was in progress when
this handoff was written -- check if it finished, the results are useful
either way.

### Task allocation, path planning, optimization (all pure math, fully reusable)
- `cbba_task_allocation.py` -- risk-prioritized task allocation. NOTE: this
  is NOT a literal implementation of the original CBBA paper -- after
  finding a chain of real bugs in the bundle/release mechanism, it was
  rewritten as a sequential single-task auction with proper distributed
  max-consensus. Also fixed: task value normalization (real risk grids
  produced values ~1000x larger than synthetic test values, breaking
  travel-cost weighting) and task extraction (switched from connected-
  component labeling to local-maxima peak detection, since risk-masked
  regions are topologically one continuous band and connected-components
  collapsed everything into one mega-task).
- `iterative_task_allocation.py` -- repeats CBBA across multiple dispatch
  rounds until the full detected backlog is covered, agents' positions
  advancing between rounds.
- `path_planning.py` -- risk-aware path planner derived from Luo et al.
  2026 (INOA-SQP paper -- the base paper for this component). Plans
  through REAL occupancy grids (not synthetic obstacles) with a risk-score
  bonus. Two real bugs found and fixed: collision-check resolution scaling
  with path length (a fixed-count resample could step over thin obstacles
  on long paths), and the safety-rollback mechanism actually verifying
  both candidate paths instead of assuming one was safe.
- `noa_optimizer.py` -- the generic, reusable core of the NOA+DHA+CDL-LOBL+
  SQP optimizer, factored out of `path_planning.py` so the planner and the
  benchmark below provably use the same algorithm.
- `baseline_optimizers.py` -- GWO and PSO baselines (the two strongest
  performers against INOA-SQP in the base paper itself).
- `benchmark_cec2017.py` -- validates the optimizer against GWO/PSO on real
  CEC2017 functions (via `opfunu` library -- needs `pip install opfunu
  "setuptools<81"`, there's a real dependency fragility with pkg_resources
  on newer setuptools). Confirmed authentic against known optimum values.

### Belief fusion / mapping (algorithm reusable, sensor-format glue needs adapting)
- `occupancy_mapping.py` -- log-odds Bayesian occupancy grid + distributed
  delta-compressed acoustic-mesh fusion (the actual novel architecture
  contribution). The `VoxelGrid` class and fusion logic (`AgentLocalMapper`)
  are reusable directly. The `sonar_hits_world()` function assumes
  HoloOcean's specific `[range_bin, azimuth_bin]` sonar array format --
  this needs to be adapted once you know what format DAVE's multibeam
  sonar plugin actually publishes (likely a ROS message type, possibly
  PointCloud2 or a custom sonar image -- check DAVE's docs/topics).
- `damage_probability_mapping.py` -- same story: `AgentDamageMapper`'s
  fusion logic is reusable, the camera-column-to-sonar-azimuth association
  logic needs adapting to DAVE's actual sensor formats.
- `digital_twin.py` -- interactive Plotly visualization + longitudinal
  differential change detection. Operates on saved `.npy` grids, no
  simulator dependency at all.
- `degradation_estimator.py` -- the "how long until this is a big problem"
  estimator (trend-based when a prior inspection exists, severity-
  heuristic fallback otherwise). Pure grid math, no simulator dependency.
  Real bug fixed: switched from local-maxima to connected-component
  labeling for detection extraction (opposite fix from cbba's, and for a
  good reason -- see the file's own docstring for why the two cases differ).

### Demo/dashboard tooling (concept reusable, needs a web version)
- `live_dashboard.py` -- `render_dashboard()` is a pure function (state ->
  image), fully tested. Currently renders for `cv2.imshow` (a desktop
  window). **New requirement is a webpage, not a desktop window** -- either
  adapt this to serve frames via Flask/FastAPI, or build a proper HTML/JS
  dashboard that reads the same underlying data (damage images, prognosis
  JSON). The panel layout/content design (camera+overlay, risk map, swarm
  status, prognosis list) is a good reference for what the webpage should
  show even if the rendering tech changes.
- `animate_task_allocation.py`, `build_demo_reel.py` -- video assembly
  tools, still useful for producing a recorded backup of the live demo,
  not simulator-dependent (operate on saved data).

---

## 4. What's being REPLACED (HoloOcean-specific, don't reuse directly)

- `swarm_sim.py` -- HoloOcean scenario building, `env.tick()`/`env.act()`
  calls. Gets replaced by: a Gazebo world file (SDF) + ROS2 launch files +
  DAVE vehicle model instantiation. **One general lesson worth carrying
  over**: this file's spawn-point discovery approach (don't guess world
  coordinates -- probe and read back the real spawn location) was the fix
  for a real, time-consuming bug. Whatever the Gazebo/DAVE equivalent of
  "where does my vehicle actually spawn" is, verify it directly rather
  than assuming, the same way.
- `live_mission.py`, `mock_holoocean_env.py` -- the orchestration LOGIC
  (patrol -> detect -> live CBBA redirect -> dashboard update) is worth
  reading as a reference for what the new ROS2 nodes need to do, but the
  plumbing (`env.tick()`/`env.act()` synchronous loop) needs to become
  ROS2 node callbacks (`rclpy` subscribers/publishers/timers, async).
  **Testing lesson worth carrying over**: this was validated against a
  mock environment before ever touching real hardware/sim -- build
  something equivalent (a mock ROS2 publisher setup, or `launch_testing`)
  before trusting new ROS2 nodes on demo day.
- `check_builtin_scenario.py` -- HoloOcean-specific diagnostic, not needed.

**PROJECT_SUMMARY.md and README.md** (in the same folder) contain the
full detailed history of every bug found and fixed across the whole
project -- worth bringing along as reference material, especially for the
paper's methods/limitations section, even though the simulator itself is
changing.

---

## 5. What needs to be BUILT FRESH

1. **ROS2 package scaffolding** -- proper package structure, launch files.
2. **Gazebo world + DAVE vehicle setup** -- get a multi-AUV DAVE scenario
   actually running and visible in Gazebo. This is the new "prove it
   moves" milestone (the role `swarm_sim.py` played originally). Need to
   find/adapt a dam/pipe-like structure -- check DAVE's example worlds
   first before building one from scratch.
3. **Sensor format adapters** -- bridge DAVE's actual sonar/camera topic
   formats into the reusable occupancy/damage-detection logic above.
4. **ROS2 nodes**: perception (occupancy + live crack detection), task
   allocation (CBBA, triggered live on new detections), control (waypoint
   following / redirect execution).
5. **Web dashboard** -- Flask/FastAPI app serving damage images +
   `degradation_estimator.py`'s prognosis output, live-updating during
   the demo.
6. **Gazebo-domain crack detector fine-tuning** -- once Gazebo is capturing
   real frames, re-run `crack_injection.py` on those frames + `train.py
   --init_from_checkpoint <deepcrack_checkpoint>` (see Section 2 above).

---

## 6. Validated results from the HoloOcean phase (still legitimate reference data)

- Occupancy mapping: converged to a recognizable structure on real sonar
  data, ~99.9%+ communication savings via delta compression.
- Crack detection: 0.91 Dice / 0.87 IoU on genuinely held-out real
  synthetic-injected data; 0.9153 Dice with DeepCrack real-photo
  pretraining (single run, needs multi-trial validation).
- CBBA: 292 real detected risk points, balanced allocation across 4
  agents after fixing the bugs described above.
- Path planning: genuine zero-penetration obstacle avoidance, verified
  against a real (not accidentally-impassable) test obstacle.

These numbers are real and citable. They just came from HoloOcean, which
won't be the demo simulator going forward -- still valid as an initial
validation phase / fallback / comparison point for the paper.

---

## 7. How to start the new chat

1. Upload this document first.
2. Upload the reusable-files package (accompanying zip -- everything
   listed in Section 3, packaged together).
3. State clearly: "Continuing this project, building the ROS2+Gazebo+DAVE
   simulation layer, using these existing files as-is where possible."
4. First concrete milestone to ask for: get a DAVE multi-AUV scenario
   running and visibly moving in Gazebo -- the equivalent of this
   project's very first HoloOcean milestone. Everything else builds on
   confirming that works first.
