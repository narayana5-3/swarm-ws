"""
Live per-agent waypoint-following controller.

Reuses path_planning.py's PlanningEnvironment/plan_path (NOA-SQP,
completely unchanged) against the same run_dir/occupancy + run_dir/
damage_probability files occupancy_mapping_node/belief_fusion_node already
write -- PlanningEnvironment already expects exactly this file layout, so
zero new environment-loading code was needed.

Live-tuning note: the paper-scale defaults (POP_SIZE=40, MAX_ITERS=150)
took 73s for a SINGLE waypoint leg on this machine -- unusable for a judge
watching a live demo. Measured on this exact grid: pop_size=15/max_iters=30
plans the same leg collision-free in ~6s. Live defaults below reflect that
real measurement, not a guess; offline/paper-quality planning still uses
this file's own POP_SIZE/MAX_ITERS constants directly if called without
overrides.

Planning runs in a background thread per replan trigger (not the executor
callback itself) since even ~6s/leg x several waypoints would otherwise
block every other subscription this node has. A per-agent lock drops any
new assigned_waypoints message that arrives while a replan is already in
flight for that agent, rather than spawning a competing thread -- confirmed
live that without this guard, multiple uncancelled planning threads pile
up for the same agent (cbba_allocator re-auctions every 15s, faster than a
4-leg route reliably finishes computing under load), each diluting the
others' CPU share so none of them ever converges.

Thrust allocation is direct world-to-body rotation (yaw only -- roll/pitch
are near-zero at the low speeds this controller drives) onto the AUV's 5
axis-aligned thrusters (see models/auv/model.sdf): surge_left/right get
equal force for pure surge (no yaw-turning needed since sway gives
holonomic lateral motion directly), heave_front/rear get equal force plus
a constant feedforward matching the hull's measured ~2% negative buoyancy
(mass 44.6kg vs neutral 43.75kg -- see model.sdf's own comment) so the
vehicle holds depth at a waypoint instead of slowly sinking, the behavior
confirmed during M1 verification.
"""

import os
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float64

from occupancy_mapping.sensor_adapter import quaternion_to_rpy_deg
from swarm_control.path_planning import PlanningEnvironment, plan_path

LIVE_POP_SIZE = 15
LIVE_MAX_ITERS = 30

# Net downward force at rest: (44.6 - 43.75) kg * 9.81 m/s^2 -- see model.sdf.
# The exact feedforward left the vehicle slowly sinking in practice (confirmed
# live: settled on the seafloor after several minutes with no depth-error
# correction active yet) -- hydrodynamic drag/CG-offset effects the simple
# mass-difference estimate doesn't capture apparently tip the real balance
# slightly negative. A 50% margin trades a small upward bias (bounded, safe)
# for the alternative failure mode (unbounded sinking to the seafloor).
GRAVITY_FEEDFORWARD_N = (44.6 - 43.75) * 9.81 * 1.5

WAYPOINT_ARRIVAL_THRESHOLD_M = 2.0
MAX_THRUST_N = 30.0
SURGE_GAIN = 4.0
SWAY_GAIN = 4.0
HEAVE_GAIN = 6.0


class AgentControlState:
    def __init__(self):
        self.position = None
        self.yaw_deg = None
        self.route = []       # list of np.array([x, y, z]) waypoints
        self.route_index = 0
        self.planning_lock = threading.Lock()


class SwarmControlNode(Node):
    def __init__(self):
        super().__init__("swarm_control_node")

        self.declare_parameter("num_auvs", 3)
        self.declare_parameter("run_dir", os.path.expanduser("~/swarm_ws/live_run"))
        self.declare_parameter("control_rate_hz", 5.0)

        num_auvs = self.get_parameter("num_auvs").value
        self.run_dir = os.path.expanduser(self.get_parameter("run_dir").value)
        control_rate = self.get_parameter("control_rate_hz").value

        self.agent_names = [f"auv{i}" for i in range(num_auvs)]
        self.state = {name: AgentControlState() for name in self.agent_names}
        self.thrust_pubs = {}

        for name in self.agent_names:
            self.create_subscription(
                Odometry, f"/{name}/odom",
                self._make_odom_cb(name), qos_profile_sensor_data)
            self.create_subscription(
                PoseArray, f"/{name}/assigned_waypoints",
                self._make_waypoints_cb(name), 10)
            self.thrust_pubs[name] = {
                joint: self.create_publisher(
                    Float64, f"/model/{name}/joint/{joint}/cmd_thrust", 10)
                for joint in ("surge_left_joint", "surge_right_joint", "sway_joint",
                              "heave_front_joint", "heave_rear_joint")
            }

        self.create_timer(1.0 / control_rate, self._control_tick)

        self.get_logger().info(
            f"swarm_control_node: controlling {self.agent_names}, "
            f"planning against {self.run_dir}")

    def _make_odom_cb(self, name):
        def cb(msg):
            p = msg.pose.pose.position
            o = msg.pose.pose.orientation
            st = self.state[name]
            st.position = np.array([p.x, p.y, p.z])
            st.yaw_deg = quaternion_to_rpy_deg(o.x, o.y, o.z, o.w)[2]
        return cb

    def _make_waypoints_cb(self, name):
        def cb(msg):
            targets = [np.array([p.position.x, p.position.y, p.position.z])
                       for p in msg.poses]
            if not targets:
                return
            st = self.state[name]
            if not st.planning_lock.acquire(blocking=False):
                # A replan for this agent is already in flight. cbba_allocator
                # re-auctions every 15s; NOA-SQP planning legs measured 6-30s
                # each under contention, so a 4-leg route can easily still be
                # computing when the next assignment arrives. Without this
                # guard, every new message spawned an uncancelled thread on
                # top of the last -- confirmed live via `ps -T`: 3 separate
                # waves of planning threads had piled up for the same agent,
                # each diluting the others' CPU share so NONE of them ever
                # finished. Stale target sets are simply dropped; the agent
                # finishes its current route before considering a newer one.
                self.get_logger().info(f"[{name}] still planning previous route, dropping this update.")
                return
            threading.Thread(
                target=self._replan, args=(name, targets), daemon=True).start()
        return cb

    def _replan(self, name, targets):
        st = self.state[name]
        try:
            if st.position is None:
                return  # no pose yet -- can't plan a start point
            try:
                env = PlanningEnvironment(self.run_dir, agent_name=name)
            except FileNotFoundError as exc:
                self.get_logger().warn(f"[{name}] planning skipped, no occupancy grid yet: {exc}")
                return

            leg_start = st.position
            route = []
            for goal in targets:
                result = plan_path(leg_start, goal, env,
                                    pop_size=LIVE_POP_SIZE, max_iters=LIVE_MAX_ITERS)
                route.extend(result["final_path"][1:])  # skip duplicate leg_start
                leg_start = goal

            st.route = route
            st.route_index = 0
            self.get_logger().info(f"[{name}] replanned route: {len(route)} points, "
                                    f"{len(targets)} task waypoints")
        finally:
            st.planning_lock.release()

    def _control_tick(self):
        for name in self.agent_names:
            st = self.state[name]
            if st.position is None or st.yaw_deg is None:
                continue

            surge_force = sway_force = 0.0
            heave_force = GRAVITY_FEEDFORWARD_N / 2.0  # per-thruster share

            if st.route and st.route_index < len(st.route):
                target = st.route[st.route_index]
                delta_world = target - st.position
                dist = np.linalg.norm(delta_world)

                if dist <= WAYPOINT_ARRIVAL_THRESHOLD_M:
                    st.route_index += 1
                else:
                    yaw_rad = np.radians(st.yaw_deg)
                    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
                    # World -> body rotation (yaw only, see module docstring).
                    body_dx = cos_y * delta_world[0] + sin_y * delta_world[1]
                    body_dy = -sin_y * delta_world[0] + cos_y * delta_world[1]
                    body_dz = delta_world[2]

                    surge_force = float(np.clip(SURGE_GAIN * body_dx, -MAX_THRUST_N, MAX_THRUST_N))
                    sway_force = float(np.clip(SWAY_GAIN * body_dy, -MAX_THRUST_N, MAX_THRUST_N))
                    heave_force += float(np.clip(HEAVE_GAIN * body_dz, -MAX_THRUST_N, MAX_THRUST_N))

            pubs = self.thrust_pubs[name]
            pubs["surge_left_joint"].publish(Float64(data=surge_force))
            pubs["surge_right_joint"].publish(Float64(data=surge_force))
            pubs["sway_joint"].publish(Float64(data=sway_force))
            pubs["heave_front_joint"].publish(Float64(data=heave_force))
            pubs["heave_rear_joint"].publish(Float64(data=heave_force))


def main(args=None):
    rclpy.init(args=args)
    node = SwarmControlNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
