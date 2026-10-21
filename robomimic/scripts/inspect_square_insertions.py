"""Look closely at how the square nut ends up on the post, in demonstrations or in saved rollouts.

Replays every episode of a robomimic HDF5 (a demonstration dataset, or the ``rollouts.hdf5`` that
rollout_best.py writes) by restoring each recorded simulator state, and reads the contacts and poses
there. It needs the same ``states`` and ``model_file`` that descriptor reconstruction needs, and the
simulator it was recorded in:

    python -m robomimic.scripts.inspect_square_insertions \
        --dataset dataset/square_image_84_contactnets_state.hdf5 --csv square_insertions.csv

Per episode it reports:
  * whether the task counted as a success (robosuite's own rule) and, at that moment, whether the
    post is really inside the nut's hole, how tilted the nut is, and how high it is above the table;
  * how the nut was carried: lifted, or slid along the table while held (counted from the first lift,
    so that shifting the nut while closing the gripper does not count), and whether it was above the
    post when the post first came inside its outline;
  * how long and how hard the nut pressed on the post, and how often the gripper's fingers, hand and
    arm touch the table, including how long and how hard the fingertips pressed on it before both
    fingers held the nut.
Fingertips touching the table while they grasp a handle that lies on it are normal, so those counts are
for comparing episodes, not for rejecting them. Nothing here reads a policy or a CaMI label.

A rollouts.hdf5 from rollout_best.py works too. It keeps the state after the last action in
``next_states``, which is where the step that first satisfies the success rule is, so that state is
replayed as well. Rollouts that stopped before their first action have no states and are skipped.

robosuite's success rule only asks that the nut's centre is within 3 cm of the post and below 5 cm above
the table; it never checks that the nut is flat or that the post passes through the hole.
"""
import argparse
import csv
import json
import math
import os
import sys

import h5py
import numpy as np

# Nut dimensions (m) from robosuite's square-nut.xml, used when the model does not expose them.
SQUARE_NUT = dict(hole_half=0.02275, outer_half=0.04375, half_thickness=0.01)
PEG_HALF = 0.016
CATEGORIES = ("nut", "peg", "table", "finger", "hand", "arm", "other")
FLAG_NAMES = ("success_without_post_in_hole", "tilted_at_success", "dragged_on_table", "arrived_beside_post",
              "long_contact_with_post")
# The flags that mean the nut was forced onto the post; the others only describe how it was handled.
JAM_FLAGS = ("success_without_post_in_hole", "tilted_at_success", "arrived_beside_post")


def geom_category(body_name):
    """Which part of the scene a collision geom belongs to, from the name of its body."""
    name = body_name or ""
    if name.startswith("SquareNut"):
        return "nut"
    if name == "peg1":
        return "peg"
    if name == "table":
        return "table"
    if "finger" in name:
        return "finger"
    if name.startswith("gripper0"):
        return "hand"
    if name.startswith("robot0"):
        return "arm"
    return "other"


class Scene:
    """Reads one recorded state at a time from a simulator model.

    ``set_state(state)`` must put the simulator in that state and run the forward pass, so that contacts
    exist. ``eef_site`` is the id of the gripper's grasp site, as robosuite's success rule uses.
    """

    def __init__(self, model, data, set_state, table_z, eef_site=None):
        import mujoco
        self.mj, self.model, self.data, self.set_state = mujoco, model, data, set_state
        self.table_z, self.eef_site = float(table_z), eef_site
        self.nut_body = self._body("SquareNut_main")
        self.peg_body = self._body("peg1")
        self.category = np.array([geom_category(self._body_name(int(b))) for b in model.geom_bodyid], dtype=object)
        peg_geoms = np.flatnonzero(model.geom_bodyid == self.peg_body)
        self.peg_pos = np.array(model.body_pos[self.peg_body])  # the post is welded to the world
        self.peg_half = float(model.geom_size[peg_geoms[0]][0]) if len(peg_geoms) else PEG_HALF
        self.peg_top = (float(self.peg_pos[2] + model.geom_pos[peg_geoms[0]][2] + model.geom_size[peg_geoms[0]][2])
                        if len(peg_geoms) else float("nan"))
        self.dims = self._nut_dimensions()
        self.finger_of = {}  # a finger's pad body counts as the finger it is attached to (the Panda has two bodies per finger)
        for body in {int(b) for b in model.geom_bodyid}:
            if geom_category(self._body_name(body)) == "finger":
                root = body
                while geom_category(self._body_name(int(model.body_parentid[root]))) == "finger":
                    root = int(model.body_parentid[root])
                self.finger_of[body] = root

    def _body(self, name):
        index = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, name)
        if index < 0:
            raise ValueError("The scene has no body named {}; this tool is for the Square task".format(name))
        return index

    def _body_name(self, index):
        return self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_BODY, index)

    def _nut_dimensions(self):
        """Hole and outline half-widths and half-thickness from the four bar geoms, else the asset's values."""
        try:
            bars = [self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_GEOM, "SquareNut_g{}".format(i)) for i in range(4)]
            pos, size = self.model.geom_pos[bars], self.model.geom_size[bars]
            axis = np.argmax(np.abs(pos[:, :2]), axis=1)          # the bar's offset axis (x or y)
            offset = np.abs(pos[np.arange(4), axis])
            width = size[np.arange(4), axis]
            return dict(hole_half=float((offset - width).min()), outer_half=float((offset + width).max()),
                        half_thickness=float(size[:, 2].max()))
        except Exception:
            return dict(SQUARE_NUT)

    def read(self):
        """Poses and contact counts at the current state."""
        m, d, c = self.model, self.data, self.category
        out = dict(nut_pos=np.array(d.xpos[self.nut_body]), nut_rot=np.array(d.xmat[self.nut_body]).reshape(3, 3),
                   eef=np.array(d.site_xpos[self.eef_site]) if self.eef_site is not None else np.full(3, np.nan))
        counts = dict(nut_table=0, nut_peg=0, finger_table=0, hand_table=0, arm_table=0)
        fingers, force, finger_table_force = set(), 0.0, 0.0
        wrench = np.zeros(6)
        for i in range(d.ncon):
            contact = d.contact[i]
            if contact.dist > 1e-3:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            pair = {c[a], c[b]}
            if pair == {"nut", "table"}:
                counts["nut_table"] += 1
            elif pair == {"nut", "peg"}:
                counts["nut_peg"] += 1
                self.mj.mj_contactForce(m, d, i, wrench)
                force += max(0.0, float(wrench[0]))
            elif pair == {"nut", "finger"}:
                fingers.add(self.finger_of[int(m.geom_bodyid[b if c[b] == "finger" else a])])
            elif pair == {"finger", "table"}:
                counts["finger_table"] += 1
                self.mj.mj_contactForce(m, d, i, wrench)
                finger_table_force += max(0.0, float(wrench[0]))
            elif pair == {"hand", "table"}:
                counts["hand_table"] += 1
            elif pair == {"arm", "table"}:
                counts["arm_table"] += 1
        out.update(counts, nut_finger=len(fingers), nut_peg_force=force, finger_table_force=finger_table_force)
        return out


def scene_from_env(env):
    """The adapter for a robomimic environment wrapper around robosuite."""
    from robomimic.CAMI.continuous_mujoco import native_model_data
    sim = env.env.sim
    model, data = native_model_data(sim)

    def set_state(state):
        env.reset_to({"states": state})
        sim.forward()
    site = env.env.robots[0].eef_site_id
    return Scene(model, data, set_state, env.env.table_offset[2], site["right"] if isinstance(site, dict) else site)


def extract(scene, states):
    """Per-step features of one episode, as arrays with one row per recorded state."""
    rows = []
    for state in states:
        scene.set_state(state)
        rows.append(scene.read())
    features = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
    features["peg_pos"] = scene.peg_pos
    return features, dict(table_z=scene.table_z, peg_top=scene.peg_top, peg_half=scene.peg_half, **scene.dims)


def to_nut_frame(rot, vectors):
    """World vectors expressed along the nut's own axes (R^T v); ``rot`` is one matrix or one per vector."""
    return np.einsum("...ij,...i->...j", rot, vectors)


def post_through_hole(nut_pos, nut_rot, peg_xy, peg_half, hole_half, tolerance=1e-3):
    """Whether the post's four corners lie inside the nut's square hole, seen along the nut's own axes."""
    corners = np.asarray(peg_xy) + peg_half * np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    local = to_nut_frame(nut_rot, np.c_[corners - np.asarray(nut_pos)[:2], np.zeros(4)])
    return bool(np.abs(local[:, :2]).max() <= hole_half + tolerance)


def analyze(f, geometry, max_tilt_deg=30., seat_tolerance=.005, drag_speed=.001, reach_scale=10., lift_height=.02):
    """Metrics for one episode from its features (see extract).

    robosuite spawns the nut 6 cm in the air, so the first steps are its fall; everything about lifting starts
    at the first step it touches the table. Sliding is counted from the first step after that when the nut is
    more than ``lift_height`` above lying flat (from that touch if it never is): shifting the nut while the
    fingers close on it is not what jams it onto the post. The post's arrival is judged at the start of the
    last unbroken stretch, ending at the first success, in which the post is inside the nut's outline.
    """
    pos, rot, eef, peg = f["nut_pos"], f["nut_rot"], f["eef"], np.asarray(f["peg_pos"])
    T = len(pos)
    z_rest = geometry["table_z"] + geometry["half_thickness"]  # nut lying flat on the table
    tilt = np.degrees(np.arccos(np.clip(rot[:, 2, 2], -1., 1.)))
    near = (np.abs(pos[:, 0] - peg[0]) < .03) & (np.abs(pos[:, 1] - peg[1]) < .03) & (pos[:, 2] < geometry["table_z"] + .05)
    far_enough = 1. - np.tanh(reach_scale * np.linalg.norm(eef - pos, axis=1)) < .6
    success = near & (far_enough | np.isnan(eef).any(axis=1))
    first = int(np.argmax(success)) if success.any() else None
    at = first if first is not None else T - 1
    seated_flags = np.array([post_through_hole(pos[t], rot[t], peg[:2], geometry["peg_half"], geometry["hole_half"])
                             for t in range(T)])
    seated = seated_flags & (tilt <= max_tilt_deg) & (pos[:, 2] - z_rest <= seat_tolerance)
    # carrying
    speed = np.r_[0., np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)]
    held = f["nut_finger"] > 0
    drag = (f["nut_table"] > 0) & held & (speed > drag_speed)
    on_table = f["nut_table"] > 0
    settled = int(np.argmax(on_table)) if on_table.any() else 0  # the spawn drop ends when the nut first touches the table
    height = pos[:, 2] - z_rest
    lifted = (height > lift_height) & (np.arange(T) >= settled)
    carried_from = int(np.argmax(lifted)) if lifted.any() else settled
    # the first step at which both fingers hold the nut, and what the fingertips did to the table before it
    grasped = f["nut_finger"] >= 2
    grasp = int(np.argmax(grasped)) if grasped.any() else None
    before = slice(0, grasp if grasp is not None else T)
    # where the post came inside the nut's outline for the last time before the success: above it, or beside it?
    local = to_nut_frame(rot, np.c_[peg[:2] - pos[:, :2], np.zeros(T)])
    inside_outline = np.abs(local[:, :2]).max(axis=1) <= geometry["outer_half"] + geometry["peg_half"]
    if inside_outline[at]:
        enter = at
        while enter > 0 and inside_outline[enter - 1]:
            enter -= 1
    else:
        enter = int(np.argmax(inside_outline)) if inside_outline.any() else None
    margin = (float(pos[enter, 2] - geometry["half_thickness"] - geometry["peg_top"]) if enter is not None else float("nan"))
    return dict(
        steps=T, success=first is not None, first_success=first, seated_at_end=bool(seated[-1]),
        success_without_hole=bool(first is not None and not seated_flags[first]),
        tilted_at_success=bool(first is not None and tilt[first] > max_tilt_deg),
        tilt_deg=float(tilt[at]), height_above_rest=float(pos[at, 2] - z_rest),
        post_through_hole=bool(seated_flags[at]), xy_offset=float(np.linalg.norm(pos[at, :2] - peg[:2])),
        lift=float(height[settled:].max()), drag_steps=int(drag.sum()), drag_after_lift_steps=int(drag[carried_from:].sum()),
        clearance_at_post=margin, nut_table_held_steps=int(((f["nut_table"] > 0) & held).sum()), held_steps=int(held.sum()),
        grasp_step=grasp, finger_table_steps=int((f["finger_table"] > 0).sum()),
        finger_table_before_grasp_steps=int((f["finger_table"][before] > 0).sum()),
        finger_table_peak_force=float(np.max(f["finger_table_force"], initial=0.)),
        finger_table_peak_force_before_grasp=float(np.max(f["finger_table_force"][before], initial=0.)),
        hand_table_steps=int((f["hand_table"] > 0).sum()),
        arm_table_steps=int((f["arm_table"] > 0).sum()), nut_peg_steps=int((f["nut_peg"] > 0).sum()),
        peak_post_force=float(f["nut_peg_force"].max()))


def flags(m, min_drag_steps=5, min_post_contact_steps=40):
    out = []
    if m["success_without_hole"]:
        out.append("success_without_post_in_hole")
    if m["tilted_at_success"]:
        out.append("tilted_at_success")
    if m["drag_after_lift_steps"] >= min_drag_steps:
        out.append("dragged_on_table")
    if m["clearance_at_post"] == m["clearance_at_post"] and m["clearance_at_post"] < 0:
        out.append("arrived_beside_post")
    if m["nut_peg_steps"] >= min_post_contact_steps:
        out.append("long_contact_with_post")
    return out


def ranks(values):
    """Ranks from 0, with equal values sharing the average of the ranks they span."""
    values = np.asarray(values, float)
    order = np.argsort(values, kind="stable")
    result, start = np.empty(len(values)), 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        result[order[start:stop + 1]] = (start + stop) / 2.
        start = stop + 1
    return result


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(ranks(a[ok]), ranks(b[ok]))[0, 1])


def replayable(demo):
    return "states" in demo and demo["states"].shape[0] > 0 and "model_file" in demo.attrs


def episodes(path):
    """(names, metadata, skipped) of the episodes with replayable states in a robomimic HDF5.

    ``skipped`` are the episodes that have no states or no model_file, such as a rollout that failed before
    its first action.
    """
    with h5py.File(path, "r") as f:
        names = sorted(f["data"], key=lambda x: int(x.split("_")[-1]))
        usable = [n for n in names if replayable(f["data"][n])]
        meta = json.loads(f["data"].attrs.get("cami_c", "{}"))
    if not usable:
        raise ValueError("{} has no episodes with both states and a model_file; descriptor reconstruction needs "
                         "them too (use the original raw demonstrations)".format(path))
    return usable, meta, [n for n in names if n not in usable]


def replay_states(demo):
    """The recorded states, plus the state after the last action when ``next_states`` records it.

    A rollout stops at the action after which the task counts as done, so the state that satisfies the success
    rule is only in ``next_states``. It is added only when ``next_states`` is the same trajectory shifted by one.
    """
    states = demo["states"][()]
    if "next_states" in demo:
        later = demo["next_states"][()]
        if later.shape == states.shape and np.allclose(later[:-1], states[1:]):
            return np.concatenate([states, later[-1:]])
    return states


def build_report(path, limit=None, max_tilt_deg=30., seat_tolerance=.005, min_drag_steps=5, min_post_contact_steps=40,
                 make_scene=None, progress=False, drag_speed=.001, lift_height=.02):
    from robomimic.CAMI import continuous_mujoco as physics
    names, meta, skipped = episodes(path)
    names = names[:limit] if limit else names
    results = {}
    with h5py.File(path, "r") as f:
        env = physics.make_replay_env(f["data"])
        try:
            recorded = {n: bool(f["data"][n].attrs["success"]) for n in names if "success" in f["data"][n].attrs}
            for index, name in enumerate(names):
                demo = f["data"][name]
                physics.reset_demo(env, demo)
                scene = (make_scene or scene_from_env)(env)
                features, geometry = extract(scene, replay_states(demo))
                metrics = analyze(features, geometry, max_tilt_deg, seat_tolerance, drag_speed=drag_speed, lift_height=lift_height)
                metrics["flags"] = flags(metrics, min_drag_steps, min_post_contact_steps)
                if name in recorded:
                    metrics["recorded_success"] = recorded[name]
                results[name] = metrics
                if progress:
                    print("  {} ({}/{})".format(name, index + 1, len(names)), flush=True)
        finally:
            close = getattr(env, "close", None) or getattr(getattr(env, "env", None), "close", None)
            if close:
                close()
    drift = {n: d for n, d in (meta.get("demos") or {}).items() if n in results and "transitions_over_tolerance" in d}
    thresholds = dict(max_tilt_deg=max_tilt_deg, seat_tolerance=seat_tolerance, min_drag_steps=min_drag_steps,
                      min_post_contact_steps=min_post_contact_steps, drag_speed=drag_speed, lift_height=lift_height)
    return dict(dataset=str(path), episodes=results, drift=drift, skipped=skipped, thresholds=thresholds)


def summarize(report):
    eps = report["episodes"]
    n = len(eps)
    values = lambda key: np.array([m[key] for m in eps.values()], dtype=float)
    frac = lambda mask: float(np.mean(mask)) if n else float("nan")
    has = lambda flag: np.array([flag in m["flags"] for m in eps.values()])
    success = values("success").astype(bool)
    no_hole = values("success_without_hole").astype(bool)
    tilted = values("tilted_at_success").astype(bool)
    summary = dict(
        episodes=n, successes=int(success.sum()), success_with_post_in_hole=int((success & ~no_hole).sum()),
        success_without_post_in_hole=int(no_hole.sum()), tilted_at_success=int(tilted.sum()),
        seated_at_end=int(values("seated_at_end").astype(bool).sum()), never_succeeded=int((~success).sum()),
        median_height_above_rest_at_success=float(np.median(values("height_above_rest")[success])) if success.any() else float("nan"),
        long_post_contact=frac(has("long_contact_with_post")), median_post_contact_steps=float(np.median(values("nut_peg_steps"))),
        median_peak_post_force=float(np.median(values("peak_post_force"))),
        lifted_over_3cm=frac(values("lift") > .03), median_lift=float(np.median(values("lift"))),
        dragged_after_lift=frac(has("dragged_on_table")), median_drag_after_lift_steps=float(np.median(values("drag_after_lift_steps"))),
        dragged_at_any_time=frac(values("drag_steps") > 0), median_drag_steps=float(np.median(values("drag_steps"))),
        arrived_beside_post=frac(has("arrived_beside_post")),
        grasped=frac(np.array([m["grasp_step"] is not None for m in eps.values()])),
        finger_table_before_grasp=frac(values("finger_table_before_grasp_steps") > 0),
        median_finger_table_before_grasp_steps=float(np.median(values("finger_table_before_grasp_steps"))),
        median_finger_table_peak_force_before_grasp=float(np.median(values("finger_table_peak_force_before_grasp"))),
        median_finger_table_peak_force=float(np.median(values("finger_table_peak_force"))),
        finger_table=frac(values("finger_table_steps") > 0), hand_table=frac(values("hand_table_steps") > 0),
        arm_table=frac(values("arm_table_steps") > 0), median_tilt_at_success=float(np.median(values("tilt_deg")[success])) if success.any() else float("nan"),
        flagged={name: m["flags"] for name, m in eps.items() if m["flags"]})
    recorded = [(m["success"], m["recorded_success"]) for m in eps.values() if "recorded_success" in m]
    if recorded:
        summary["recorded_vs_replayed_success_mismatch"] = int(sum(a != b for a, b in recorded))
    if report["drift"]:
        over = np.array([report["drift"][k]["transitions_over_tolerance"] / max(1, report["drift"][k]["checked_transitions"])
                         for k in report["drift"]])
        summary["drift_over_tolerance_median"] = float(np.median(over))
        flagged = np.array([bool(eps[k]["flags"]) for k in report["drift"]])
        summary["drift_over_tolerance_flagged_vs_not"] = (float(np.median(over[flagged])) if flagged.any() else float("nan"),
                                                           float(np.median(over[~flagged])) if (~flagged).any() else float("nan"))
        for key in ("nut_peg_steps", "drag_after_lift_steps", "tilt_deg", "peak_post_force"):
            summary["drift_vs_" + key] = spearman(over, [eps[k][key] for k in report["drift"]])
    summary["skipped"] = list(report.get("skipped", []))
    summary["thresholds"] = dict(report.get("thresholds", {}))
    return summary


def number(value, spec, scale=1.):
    """``value`` formatted, or n/a when there is nothing to report (an empty group gives NaN)."""
    return format(value * scale, spec) if value == value and abs(value) != float("inf") else "n/a"


def print_summary(summary):
    p = lambda *a: print(*a, flush=True)
    n = summary["episodes"]
    p("Episodes: {} | counted as success: {} (post inside the nut's hole at that step: {}, post not in the hole: {}) | "
      "never succeeded: {}".format(n, summary["successes"], summary["success_with_post_in_hole"],
                                   summary["success_without_post_in_hole"], summary["never_succeeded"]))
    p("At the first success: median nut height above flat-on-the-table {} cm, median tilt {} deg, tilted past the limit in {} "
      "episodes. Nut flat and seated at the last step: {} of {}".format(
          number(summary["median_height_above_rest_at_success"], ".1f", 100), number(summary["median_tilt_at_success"], ".1f"),
          summary["tilted_at_success"], summary["seated_at_end"], n))
    if "recorded_vs_replayed_success_mismatch" in summary:
        p("Success recorded in the file differs from the replayed rule in {} episodes".format(
            summary["recorded_vs_replayed_success_mismatch"]))
    limits = summary.get("thresholds", {})
    drag_limit, contact_limit = limits.get("min_drag_steps", 5), limits.get("min_post_contact_steps", 40)
    p("\nCarrying")
    p("  nut lifted more than 3 cm above lying flat after it landed: {:.0%} (median lift {:.1f} cm)".format(
        summary["lifted_over_3cm"], 100 * summary["median_lift"]))
    p("  slid along the table while held, any time: {:.0%} (median {:.0f} steps); after the first lift for {}+ steps: "
      "{:.0%} (median {:.0f} steps)".format(summary["dragged_at_any_time"], summary["median_drag_steps"], drag_limit,
                                           summary["dragged_after_lift"], summary["median_drag_after_lift_steps"]))
    p("  post came inside the nut's outline with the nut not above it: {:.0%}".format(summary["arrived_beside_post"]))
    p("  nut in contact with the post for {}+ steps in all: {:.0%} (median {:.0f} steps, median peak force {:.1f} N)".format(
        contact_limit, summary["long_post_contact"], summary["median_post_contact_steps"], summary["median_peak_post_force"]))
    p("\nTable contact (share of episodes with any step)")
    p("  fingers {:.0%} | hand {:.0%} | arm {:.0%}   (fingertips on the table while grasping a flat handle are normal)".format(
        summary["finger_table"], summary["hand_table"], summary["arm_table"]))
    p("  both fingers held the nut in {:.0%} of episodes; fingertips on the table before that: {:.0%} "
      "(median {:.0f} steps, median peak force {:.1f} N; {:.1f} N at any time)".format(
          summary["grasped"], summary["finger_table_before_grasp"], summary["median_finger_table_before_grasp_steps"],
          summary["median_finger_table_peak_force_before_grasp"], summary["median_finger_table_peak_force"]))
    p("  (forces are computed at each restored pose with no actuator force, so compare them between episodes; they are not what the "
      "robot applied)")
    if "drift_over_tolerance_median" in summary:
        a, b = summary["drift_over_tolerance_flagged_vs_not"]
        p("\nReplay drift (share of transitions over tolerance): median {}; flagged episodes {}, others {}".format(
            number(summary["drift_over_tolerance_median"], ".0%"), number(a, ".0%"), number(b, ".0%")))
        p("  rank correlation of drift with: " + ", ".join(
            "{} {}".format(k[len("drift_vs_"):], number(v, "+.2f")) for k, v in summary.items() if k.startswith("drift_vs_")))
    flagged = summary["flagged"]
    p("\nFlagged episodes: {} of {}".format(len(flagged), n))
    for reason in FLAG_NAMES:
        names = [k for k, v in flagged.items() if reason in v]
        p("  {:<28} {:>4}  {}".format(reason, len(names), ", ".join(names[:12]) + (" ..." if len(names) > 12 else "")))
    jammed = [k for k, v in flagged.items() if set(v) & set(JAM_FLAGS)]
    p("  raising a jam flag ({}): {} of {}".format(", ".join(JAM_FLAGS), len(jammed), n))
    if summary.get("skipped"):
        p("\nSkipped, with no replayable states: {} ({}{})".format(
            len(summary["skipped"]), ", ".join(summary["skipped"][:12]), " ..." if len(summary["skipped"]) > 12 else ""))


def jsonable(value):
    """``value`` with numpy scalars as numbers and non-finite numbers as None, which strict JSON parsers accept."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def write_csv(path, report):
    fields = ["episode", "steps", "success", "first_success", "success_without_hole", "tilted_at_success", "seated_at_end", "tilt_deg",
              "height_above_rest", "post_through_hole", "xy_offset", "lift", "drag_steps", "drag_after_lift_steps", "clearance_at_post",
              "held_steps", "nut_table_held_steps", "grasp_step", "finger_table_steps", "finger_table_before_grasp_steps",
              "finger_table_peak_force", "finger_table_peak_force_before_grasp", "hand_table_steps", "arm_table_steps", "nut_peg_steps",
              "peak_post_force", "flags"]
    with open(path, "x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for name, m in report["episodes"].items():
            writer.writerow(dict(m, episode=name, flags=";".join(m["flags"])))


def mask_episodes(report, exclude=JAM_FLAGS):
    """The episodes that raise none of the ``exclude`` flags (by default the jam flags).

    Refuses when the report skipped episodes, since they would fall out of a mask without having been judged,
    and when no episode is left.
    """
    if report.get("skipped"):
        raise ValueError("{} episodes have no replayable states, so a mask would silently leave them out".format(len(report["skipped"])))
    clean = [n for n, m in report["episodes"].items() if not set(m["flags"]) & set(exclude)]
    if not clean:
        raise ValueError("Every episode raises one of {}; there is nothing to write".format(", ".join(exclude)))
    return clean


def write_mask(path, report, key, exclude=JAM_FLAGS):
    """Add mask/<key> listing ``mask_episodes``. Refuses to replace an existing mask."""
    clean = mask_episodes(report, exclude)
    with h5py.File(path, "r+") as f:
        masks = f.require_group("mask")
        if key in masks:
            raise ValueError("mask/{} already exists in {}".format(key, path))
        masks.create_dataset(key, data=np.asarray(clean, dtype="S"))
    return clean


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Demonstration HDF5, or rollouts.hdf5 from rollout_best.py")
    parser.add_argument("--limit", type=int, help="Only the first N episodes")
    parser.add_argument("--max-tilt-deg", type=float, default=30., help="Tilt of the nut that counts as tilted (a nut held by its handle hangs about 10-20 deg)")
    parser.add_argument("--seat-tolerance", type=float, default=.005, help="Largest height (m) above lying flat for a seated nut")
    parser.add_argument("--min-drag-steps", type=int, default=5, help="Steps of sliding on the table while held, after the first lift, that flag an episode")
    parser.add_argument("--min-post-contact-steps", type=int, default=40, help="Steps of contact between nut and post that flag an episode")
    parser.add_argument("--drag-speed", type=float, default=.001, help="Distance (m) the nut moves in one step for it to count as sliding")
    parser.add_argument("--lift-height", type=float, default=.02, help="Height (m) above lying flat from which the nut counts as lifted")
    parser.add_argument("--csv", help="Write one row per episode to this new file")
    parser.add_argument("--json", help="Write the full report to this new file")
    parser.add_argument("--write-mask", metavar="KEY", help="Add mask/KEY to the dataset, listing the episodes that raise none of --mask-flags")
    parser.add_argument("--mask-flags", nargs="+", choices=FLAG_NAMES + ("all",),
                        help="Flags that keep an episode out of the mask (default: the jam flags; 'all' for every flag)")
    parser.add_argument("--quiet", action="store_true", help="Do not print progress")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    for name in ("min_drag_steps", "min_post_contact_steps"):
        if getattr(args, name) < 1:
            parser.error("--{} must be at least 1".format(name.replace("_", "-")))
    for name in ("max_tilt_deg", "seat_tolerance", "drag_speed", "lift_height"):
        if not getattr(args, name) > 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    for target in (args.csv, args.json):
        if target and os.path.exists(target):
            parser.error("refusing to replace an existing file: " + target)
    if args.mask_flags and args.write_mask is None:
        parser.error("--mask-flags only applies with --write-mask")
    if args.write_mask is not None:
        if not args.write_mask:
            parser.error("--write-mask needs a name for the mask")
        if args.limit:
            parser.error("--write-mask lists the episodes that raise none of --mask-flags, so it needs every episode analyzed; drop --limit")
        if not os.access(args.dataset, os.W_OK):
            parser.error("{} cannot be written to, so it cannot get a mask".format(args.dataset))
        with h5py.File(args.dataset, "r") as f:
            if "mask/" + args.write_mask in f:
                parser.error("mask/{} already exists in {}".format(args.write_mask, args.dataset))
        skipped = episodes(args.dataset)[2]
        if skipped:
            parser.error("{} episodes have no replayable states ({}), so a mask would silently leave them out".format(
                len(skipped), ", ".join(skipped[:5])))
    report = build_report(args.dataset, args.limit, args.max_tilt_deg, args.seat_tolerance, args.min_drag_steps,
                          args.min_post_contact_steps, progress=not args.quiet, drag_speed=args.drag_speed,
                          lift_height=args.lift_height)
    summary = summarize(report)
    print_summary(summary)
    clean = None
    if args.write_mask is not None:
        exclude = FLAG_NAMES if args.mask_flags and "all" in args.mask_flags else tuple(args.mask_flags or JAM_FLAGS)
        clean = mask_episodes(report, exclude)  # decided before anything is written, so a failure leaves no files behind
    if args.csv:
        write_csv(args.csv, report)
    if args.json:
        with open(args.json, "x") as stream:
            json.dump(jsonable(dict(summary=summary, **report)), stream, indent=2, allow_nan=False)
    if clean is not None:
        write_mask(args.dataset, report, args.write_mask, exclude)
        print("Wrote mask/{} with {} episodes".format(args.write_mask, len(clean)))
    return report, summary


if __name__ == "__main__":
    main(sys.argv[1:])
