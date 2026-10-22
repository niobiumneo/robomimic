"""The pieces of robosuite's Square success rule, and what stands between a rollout that looks done and a success.

robosuite (NutAssembly._check_success) counts a rollout as a success when, at one control step,
  * the nut's centre is within XY_LIMIT of the post in x and in y (a box, not a circle),
  * the nut's centre is less than HEIGHT_LIMIT above the table top, and
  * the gripper has let go: 1 - tanh(REACH_SCALE * d) < REACH_LIMIT, with d the distance from the grip site to the
    nut's centre, which means d > RELEASE_DISTANCE (4.24 cm).
It never looks at the nut's tilt. Holding the handle does not block it: a handle grasp keeps d around 6 to 7 cm.

So a nut that sits on the post can still fail: resting on the post's top or rim (its centre is 11 to 14 cm above the
table), wedged tilted and too high, held and tilted over the post, or with the hand still within 4.24 cm of it.
``RuleTracker`` follows a rollout and says which of these it ended in, and which came closest to a success.
Only numpy is needed.
"""
import math

import numpy as np

XY_LIMIT = 0.03
HEIGHT_LIMIT = 0.05
REACH_SCALE = 10.
REACH_LIMIT = 0.6
RELEASE_DISTANCE = math.atanh(1. - REACH_LIMIT) / REACH_SCALE  # 0.042365 m
HOLDING_DISTANCE = 0.10      # without contact information, a hand this close to the nut's centre counts as holding it
POST_TOP_HEIGHT = 0.10       # a nut centre this high above the table is on the post's top, not down over it
NEAR_POST = 0.10             # a nut within this of the post in the table plane is "beside" it

# What each reason says about the state a rollout ENDED in (Failure_Reason) or about the state that came closest (Closest_Reason).
REASONS = {
    "success": "the rule was met",
    "gripper_not_released": "the nut is down over the post but the hand is within 4.2 cm of it",
    "resting_on_post_top": "the nut is over the post, 10 cm or more above the table, and no finger touches it",
    "held_above_post": "the nut is over the post, above the 5 cm limit, and a finger touches it",
    "wedged_above_limit": "the nut is over the post, between 5 and 10 cm above the table, and no finger touches it",
    "beside_post": "the nut is within 10 cm of the post but more than 3 cm off in x or y",
    "far_from_post": "the nut is more than 10 cm from the post",
    "simulator_error": "the simulator raised an error",
}


def components(nut_pos, nut_rot, eef_pos, peg_pos, table_z, reach_scale=REACH_SCALE):
    """The quantities of the rule, for one state or (with a leading axis) for every step of a rollout.

    A missing grip position (NaN) counts as released, so scenes without a gripper can still be analyzed.
    """
    nut_pos, eef_pos, peg_pos = (np.asarray(x, float) for x in (nut_pos, eef_pos, peg_pos))
    dx, dy = nut_pos[..., 0] - peg_pos[..., 0], nut_pos[..., 1] - peg_pos[..., 1]
    height = nut_pos[..., 2] - table_z
    hand = np.linalg.norm(eef_pos - nut_pos, axis=-1)
    reach = 1. - np.tanh(reach_scale * hand)
    xy_ok = (np.abs(dx) < XY_LIMIT) & (np.abs(dy) < XY_LIMIT)
    z_ok = height < HEIGHT_LIMIT
    released = (reach < REACH_LIMIT) | np.isnan(hand)
    tilt = np.degrees(np.arccos(np.clip(np.asarray(nut_rot, float)[..., 2, 2], -1., 1.)))
    return dict(dx=dx, dy=dy, height=height, hand=hand, reach=reach, xy_ok=xy_ok, z_ok=z_ok, released=released,
                tilt=tilt, success=xy_ok & z_ok & released)


def violation(c):
    """How far a state is from meeting the rule: the largest of the three shortfalls, each as a fraction of its limit.

    At or below 0 means the rule is met; the state of a rollout with the smallest violation is the one that came closest.
    """
    xy = np.maximum(np.abs(c["dx"]), np.abs(c["dy"])) / XY_LIMIT - 1.
    z = (c["height"] - HEIGHT_LIMIT) / HEIGHT_LIMIT
    hand = np.where(np.isnan(c["hand"]), -1., (RELEASE_DISTANCE - c["hand"]) / RELEASE_DISTANCE)
    return np.maximum(np.maximum(xy, z), hand)


def failure_reason(c):
    """Why one state (scalars) does not meet the rule: a key of REASONS.

    "Held" means a finger touches the nut when the state says how many do (``fingers_on_nut``, which the live probe
    reads from the simulator's contacts); without that it falls back on the hand being within HOLDING_DISTANCE.
    """
    if bool(c["success"]):
        return "success"
    if c["xy_ok"] and c["z_ok"]:
        return "gripper_not_released"
    if c["xy_ok"]:
        holding = c["fingers_on_nut"] > 0 if "fingers_on_nut" in c else c["hand"] <= HOLDING_DISTANCE
        if holding:
            return "held_above_post"
        return "resting_on_post_top" if c["height"] >= POST_TOP_HEIGHT else "wedged_above_limit"
    return "beside_post" if math.hypot(c["dx"], c["dy"]) <= NEAR_POST else "far_from_post"


def plain(c):
    return {key: (value.item() if hasattr(value, "item") else value) for key, value in c.items()}


class RuleTracker:
    """Follows a rollout step by step: the state it ended in, the state that came closest to the rule, and whether it ever met it."""

    def __init__(self):
        self.last = self.best = self.step = None
        self.best_violation, self.steps, self.mismatches, self.succeeded = math.inf, 0, 0, False

    def update(self, c, environment_success=None):
        """``c`` is ``components`` of the state after a step (scalars); ``environment_success`` is the environment's own verdict."""
        c = plain(c)
        self.last = c
        self.succeeded = self.succeeded or bool(c["success"])
        v = float(violation(c))
        if v < self.best_violation:
            self.best, self.best_violation, self.step = c, v, self.steps
        if environment_success is not None and bool(c["success"]) != bool(environment_success):
            self.mismatches += 1
        self.steps += 1

    def result(self):
        """Rollout statistics; {} if no step was seen.

        Failure_Reason says what the rollout ended in (what its video ends on); Closest_Reason what stopped the state that came
        closest to the rule. A rollout that met the rule at any step is "success" for both.
        """
        if self.last is None:
            return {}
        last = self.last
        return {"Failure_Reason": "success" if self.succeeded else failure_reason(last),
                "Final_Nut_Height_cm": 100. * last["height"], "Final_XY_Offset_cm": 100. * math.hypot(last["dx"], last["dy"]),
                "Final_Hand_Distance_cm": 100. * last["hand"], "Final_Tilt_deg": last["tilt"],
                "Closest_Reason": "success" if self.succeeded else failure_reason(self.best), "Closest_Step": self.step,
                "Rule_Mismatch_Steps": self.mismatches}


def reason_counts(records):
    """{reason: number of rollouts} for the records that carry a Failure_Reason."""
    counts = {}
    for record in records:
        reason = record.get("Failure_Reason")
        if reason is not None:
            counts[reason] = counts.get(reason, 0) + 1
    return counts
