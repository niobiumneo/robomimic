"""MuJoCo contact descriptors for explicit task-relevant collision pairs.

Native mujoco bindings are required for geometry queries. This module does not
import robosuite until action replay is requested.
"""
from fnmatch import fnmatchcase
import numpy as np


def native_model_data(sim):
    return getattr(sim.model, "_model", sim.model), getattr(sim.data, "_data", sim.data)


def geometry_catalog(model):
    import mujoco
    result = {}
    for i in range(model.ngeom):
        if not (model.geom_contype[i] or model.geom_conaffinity[i]):
            continue
        body = int(model.geom_bodyid[i])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "world"
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        name = name or "@body/{}/geom/{}".format(body_name, i - int(model.body_geomadr[body]))
        result[name] = {"id": i, "body": body_name}
    return result


def resolve_pairs(model, specification):
    """Resolve explicit globs; body:foo matches collision geoms on body foo."""
    catalog = geometry_catalog(model)
    pairs = set()
    for group in specification["groups"]:
        sides = []
        for side in ("a", "b"):
            matches = [name for name, value in catalog.items() if any(
                fnmatchcase(value["body"], pattern[5:]) if pattern.startswith("body:")
                else fnmatchcase(name, pattern) for pattern in group[side])]
            if not matches:
                raise ValueError("No geometry matches {}. Use inspect-cami-c-geoms output".format(group[side]))
            sides.append(matches)
        for a in sides[0]:
            for b in sides[1]:
                ia, ib = catalog[a]["id"], catalog[b]["id"]
                if ia != ib and model.geom_bodyid[ia] != model.geom_bodyid[ib]:
                    pairs.add(tuple(sorted((a, b))))
    if not pairs:
        raise ValueError("No distinct-body geometry pairs selected")
    return sorted(pairs)


class ContinuousContactProbe:
    """Query selected contacts without feeding privileged data to the policy.

    ``sample_forces`` returns [signed gap, sum Fn, sum ||Ft||] in m,N,N.
    ``sample`` returns a one-physics-step impulse estimate, not a control-period
    integral. Use ``step_with_contact`` for actual recorded control intervals.
    Tangential force is wrench[1:3]; wrench[3:6] is torque, never friction force.
    """
    def __init__(self, model, data, pair_names, max_gap=.02):
        import mujoco
        if not isinstance(model, mujoco.MjModel) or not isinstance(data, mujoco.MjData):
            raise TypeError("Physics extraction requires native MuJoCo, not mujoco_py")
        if not hasattr(mujoco, "mj_geomDistance") or not np.isfinite(max_gap) or max_gap <= 0:
            raise ValueError("Positive max_gap and MuJoCo mj_geomDistance are required")
        self.mj, self.model, self.data = mujoco, model, data
        self.max_gap = max_gap
        catalog = geometry_catalog(model)
        self.pairs = [tuple(catalog[n]["id"] for n in pair) for pair in pair_names]
        if not self.pairs:
            raise ValueError("At least one explicit pair is required")
        self.lookup = {tuple(sorted(p)) for p in self.pairs}

    def gap(self):
        # Keep signed soft-contact penetration. The witness segment also lets
        # us reject a known CCD failure: zero distance between separated geoms.
        # Never reinterpret that failure as measured touching contact.
        values, valid = [], True
        tolerance = max(1e-5, 10. * float(self.model.opt.ccd_tolerance))
        for a, b in self.pairs:
            witness = np.zeros(6)
            distance = float(self.mj.mj_geomDistance(self.model, self.data, a, b,
                                                   self.max_gap, witness))
            length = float(np.linalg.norm(witness[3:] - witness[:3]))
            if not np.isfinite(distance) or not np.isfinite(length):
                valid = False
            elif distance < self.max_gap - tolerance and abs(abs(distance) - length) > tolerance:
                valid = False
            values.append(distance)
        self.gap_valid = valid
        return min(values) if valid else 0.  # Placeholder accompanied by validity=0.

    def contacts(self):
        """One row per selected solver contact, in native geom1 -> geom2 order."""
        rows = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if tuple(sorted((int(contact.geom1), int(contact.geom2)))) not in self.lookup:
                continue
            if contact.efc_address < 0:
                continue
            wrench = np.zeros(6, dtype=np.float64)
            self.mj.mj_contactForce(self.model, self.data, i, wrench)
            frame = np.asarray(contact.frame).reshape(3, 3).copy()
            if not np.isfinite(wrench).all():
                raise ValueError("Nonfinite solver contact force")
            rows.append(dict(geom=np.array([contact.geom1, contact.geom2], dtype=np.int32),
                gap=float(contact.dist), position=np.asarray(contact.pos).copy(),
                frame=frame, wrench=wrench,
                tangential_force_world=frame[1:].T @ wrench[1:3]))
        return rows

    def sample_forces(self):
        rows = self.contacts()
        gap = min([self.gap()] + [r["gap"] for r in rows])
        return np.asarray([gap, sum(max(0., r["wrench"][0]) for r in rows),
                           sum(np.linalg.norm(r["wrench"][1:3]) for r in rows)], dtype=np.float64)

    def sample(self):
        result = self.sample_forces()
        result[1:] *= float(self.model.opt.timestep)
        return result.astype(np.float32)


def step_with_contact(env, action, probe, record_contacts=True):
    """Integrate selected contact loading over one high-level action.

    robosuite 1.5.2 calls sim.step2 (lite physics) or sim.step. Wrap those
    instance methods temporarily, sample after each solver call, and restore
    them even on exceptions. No global MuJoCo callback or dynamics changes.
    RK4 is rejected because its internal-stage quadrature is not exposed here.

    Multiple contacts use sum of nonnegative normal impulses and sum of
    tangential impulse magnitudes across points and substeps. This loading
    aggregate avoids vector cancellation, but is NOT the magnitude of the net
    tangential impulse. Raw vectors are retained for alternative aggregation.

    An impulse is force integrated over time, in N*s (momentum units), not
    a force in N. Magnitude ignores friction direction and is invariant to
    rotations of the local tangent axes. It can be nonzero with no sliding
    because static friction can transmit load. Tangential speed is not used.
    """
    if int(probe.model.opt.integrator) == int(probe.mj.mjtIntegrator.mjINT_RK4):
        raise ValueError("Contact integration requires Euler or implicit integrator, not RK4")
    sim = env.env.sim
    start = float(probe.data.time)
    gap = float(probe.gap())
    gap_valid = probe.gap_valid
    totals = np.zeros(2, dtype=np.float64)
    rows, times = [], []
    originals, own = {}, {}

    def wrap(original):
        def step(*args, **kwargs):
            before = float(probe.data.time)
            result = original(*args, **kwargs)
            dt = float(probe.data.time) - before
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError("Simulator substep did not advance time")
            contacts = probe.contacts()
            # Numerical integral of solver forces, not velocity * duration.
            # Sum local magnitudes so opposite contact loads do not cancel.
            totals[0] += dt * sum(max(0., r["wrench"][0]) for r in contacts)
            totals[1] += dt * sum(np.linalg.norm(r["wrench"][1:3]) for r in contacts)
            for row in contacts if record_contacts else []:
                row.update(substep=len(times), time=before, dt=dt)
                rows.append(row)
            times.append((before, dt))
            return result
        return step

    try:
        for name in ("step", "step2"):
            originals[name] = getattr(sim, name)
            own[name] = name in vars(sim)
            setattr(sim, name, wrap(originals[name]))
        outcome = env.step(action)
    finally:
        for name, original in originals.items():
            if own[name]:
                setattr(sim, name, original)
            else:
                delattr(sim, name)
    end = float(probe.data.time)
    duration = end - start
    if not times or not np.isclose(sum(t[1] for t in times), duration, rtol=1e-6, atol=1e-10):
        raise RuntimeError("Unsupported simulator stepping path; no complete impulse measurement")
    return outcome, dict(physics=np.r_[gap, totals].astype(np.float32),
        valid=np.asarray([gap_valid, True, True], dtype=np.uint8),
        mean_force=totals / duration, time=start, dt=duration, substeps=len(times),
        contacts=rows)


def make_replay_env(data_group):
    import json
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    if ObsUtils.OBS_KEYS_TO_MODALITIES is None:
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": [], "rgb": []}})
    metadata = data_group.attrs.get("env_args")
    if metadata is None:
        raise ValueError("Replay requires robomimic env_args metadata")
    metadata = json.loads(metadata)
    return EnvUtils.create_env_from_metadata(env_meta=metadata, render=False,
        render_offscreen=False, use_image_obs=False)


def reset_demo(env, demo):
    if "states" not in demo or "model_file" not in demo.attrs:
        raise ValueError("Replay requires states and per-demo model_file; use --raw-dataset if necessary")
    state = {"states": demo["states"][0], "model": demo.attrs["model_file"]}
    if "ep_meta" in demo.attrs:
        state["ep_meta"] = demo.attrs["ep_meta"]
    env.reset()
    env.reset_to(state)
    # XML reset initializes controllers before the stored qpos is restored.
    # For full episodes, initialize their references at the restored start.
    # Mid-episode controller memory cannot be reconstructed this way.
    for robot in env.env.robots:
        controllers = getattr(robot, "part_controllers", None)
        if controllers is None:
            controller = getattr(robot, "controller", None)
            controllers = {} if controller is None else {"arm": controller}
        for controller in controllers.values():
            controller.update(force=True)
            if hasattr(controller, "update_initial_joints"):
                controller.update_initial_joints(controller.joint_pos)
            controller.reset_goal()


def replay_descriptors(env, demo, specification, max_gap=.02, tolerance=1e-3,
                       *, replay_mode="action"):
    """Integrate solver contact impulses along replayed action intervals.

    ``action`` is strict open-loop replay and rejects trajectory drift.
    ``state`` restores each recorded state before applying its action, refreshes
    controller goals, and reports the one-step drift instead. Its gap is from
    the recorded geometry; its impulses are state-conditioned simulator
    estimates, not recovered original measurements. No force-times-dt shortcut
    is used. Neither mode can restore unrecorded solver or controller memory.
    """
    if replay_mode not in ("action", "state") or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Replay mode must be action or state, with a finite nonnegative tolerance")
    reset_demo(env, demo)
    sim = env.env.sim
    model, data = native_model_data(sim)
    states = np.asarray(demo["states"])
    actions = demo["actions"]
    state_width = 1 + model.nq + model.nv
    if (states.ndim != 2 or states.shape[0] not in (len(actions), len(actions) + 1)
            or states.shape[1] != state_width or not np.isfinite(states).all()):
        raise ValueError("Recorded states must be finite [T,1+nq+nv] or [T+1,1+nq+nv]")
    if not np.isfinite(actions[()]).all():
        raise ValueError("Recorded actions must be finite")
    if len(states) > 1 and np.any(np.diff(states[:, 0]) <= 0):
        raise ValueError("Recorded simulator times must increase between action intervals")
    pairs = resolve_pairs(model, specification)
    probe = ContinuousContactProbe(model, data, pairs, max_gap)
    values, masks, durations, errors = [], [], [], []

    def qpos_error(index):
        error = float(np.max(np.abs(data.qpos - states[index, 1:1 + model.nq])))
        if not np.isfinite(error):
            raise ValueError("Nonfinite replay joint configuration at t={}".format(index))
        return error

    def check_drift(error, index):
        if error > tolerance:
            raise ValueError("Replay diverged at t={}: qpos max error {:.6g} > {}. "
                             "Match the collection environment, record descriptors directly, or explicitly "
                             "use --replay-mode state for simulator-estimated impulses."
                             .format(index, error, tolerance))

    check_drift(qpos_error(0), 0)
    for t, action in enumerate(actions):
        if replay_mode == "state":
            # Do not let open-loop drift accumulate or move a descriptor away
            # from the source images. Warm starts are not in flattened states;
            # use a deterministic zero warm start for each estimated interval.
            data.qacc_warmstart[:] = 0.
            env.reset_to({"states": states[t]})
            for robot in env.env.robots:
                controllers = getattr(robot, "part_controllers", None)
                if controllers is None:
                    controller = getattr(robot, "controller", None)
                    controllers = {} if controller is None else {"arm": controller}
                for controller in controllers.values():
                    controller.update(force=True)
                    controller.reset_goal()
            if qpos_error(t) > 1e-10:
                raise ValueError("Failed to restore recorded joint configuration at t={}".format(t))
        sim.forward()
        _, measured = step_with_contact(env, action, probe, record_contacts=False)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("Nonfinite simulator state after action t={}".format(t))
        if t + 1 < len(states):
            recorded_dt = float(states[t + 1, 0] - states[t, 0])
            if not np.isclose(measured["dt"], recorded_dt, rtol=1e-6, atol=1e-9):
                raise ValueError("Replay action interval {} does not match recorded interval {} at t={}"
                                 .format(measured["dt"], recorded_dt, t))
            error = qpos_error(t + 1)
            errors.append(error)
            if replay_mode == "action":
                check_drift(error, t + 1)
        values.append(measured["physics"])
        masks.append(measured["valid"])
        durations.append(measured["dt"])
    source = ("state_aligned_action_simulation" if replay_mode == "state"
              else "action_replay_substep_integrated_contact")
    return np.asarray(values), {"pairs": pairs, "max_qpos_replay_error": max(errors, default=0.),
                               "mean_qpos_replay_error": float(np.mean(errors)) if errors else 0.,
                               "checked_transitions": len(errors),
                               "transitions_over_tolerance": sum(error > tolerance for error in errors),
                               "replay_mode": replay_mode, "replay_tolerance": tolerance,
                               "gap_semantics": ("recorded_state_geometry" if replay_mode == "state"
                                                 else "replayed_state_geometry"),
                               "impulse_semantics": ("state_conditioned_simulator_estimate" if replay_mode == "state"
                                                     else "open_loop_simulator_reconstruction"),
                               "mujoco_version": probe.mj.__version__,
                               "robosuite_version": getattr(env, "version", None),
                               "source": source, "_valid": np.asarray(masks),
                               "interval_dt": float(durations[0]),
                               "interval_dt_range": [float(min(durations)), float(max(durations))]}
