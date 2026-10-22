"""The Square success rule and the reasons a rollout that looks done is not a success.

The pure functions are tested on made-up numbers; the rule itself is checked against robosuite's own verdict on nuts
placed in the real simulator (skipped when robosuite cannot make the environment).
"""
import argparse
import math
import os
import statistics
from types import SimpleNamespace

import numpy as np
import pytest

from robomimic.utils import square_success as ss

TABLE = .82
PEG = np.array([.23, .10, .85])
EYE = np.eye(3)


def state(dx=0., dy=0., height=.01, hand=.40, rot=EYE):
    """The components of a nut ``dx``, ``dy`` from the post, ``height`` above the table, with the grip site ``hand`` away."""
    nut = np.array([PEG[0] + dx, PEG[1] + dy, TABLE + height])
    return ss.components(nut, rot, nut + [0., 0., hand], PEG, TABLE)


def test_the_release_distance_is_where_the_reach_term_reaches_its_limit():
    assert ss.RELEASE_DISTANCE == pytest.approx(0.042365, abs=1e-6)
    assert 1 - math.tanh(10 * ss.RELEASE_DISTANCE) == pytest.approx(ss.REACH_LIMIT)


@pytest.mark.parametrize("changes, expected", [
    (dict(), True),
    (dict(dx=.0299, dy=-.0299), True),            # a box: a corner 4.2 cm away passes
    (dict(dx=.0301), False), (dict(dy=-.0301), False),
    (dict(height=.0499), True), (dict(height=.0501), False),
    (dict(hand=.0426), True), (dict(hand=.0423), False),
    (dict(hand=.067), True),                      # holding the handle (6 to 7 cm) does not block a success
])
def test_the_rule_is_the_three_conditions(changes, expected):
    assert bool(state(**changes)["success"]) is expected


def test_the_rule_ignores_the_nuts_tilt():
    tilt = np.array([[1, 0, 0], [0, math.cos(math.radians(25)), -math.sin(math.radians(25))],
                     [0, math.sin(math.radians(25)), math.cos(math.radians(25))]])
    c = state(rot=tilt)
    assert bool(c["success"]) and c["tilt"] == pytest.approx(25.)


def test_a_missing_hand_counts_as_released_and_is_never_a_reason_to_fail():
    nut = np.array([PEG[0], PEG[1], TABLE + .01])
    c = ss.components(nut, EYE, np.full(3, np.nan), PEG, TABLE)
    assert bool(c["success"]) and ss.violation(c) <= 0
    high = ss.components(nut + [0, 0, .12], EYE, np.full(3, np.nan), PEG, TABLE)
    assert ss.failure_reason(high) == "resting_on_post_top"  # no hand to be near it


def test_the_violation_is_not_positive_exactly_when_the_rule_is_met():
    rng = np.random.default_rng(0)
    n = 4000
    nut = np.c_[PEG[0] + rng.uniform(-.08, .08, n), PEG[1] + rng.uniform(-.08, .08, n), TABLE + rng.uniform(0, .16, n)]
    eef = nut + rng.normal(0, .05, (n, 3))
    c = ss.components(nut, np.tile(EYE, (n, 1, 1)), eef, PEG, TABLE)
    assert c["success"].sum() > 100 and (~c["success"]).sum() > 100
    assert ((ss.violation(c) <= 0) == c["success"]).all()


@pytest.mark.parametrize("changes, reason", [
    (dict(), "success"),
    (dict(hand=.03), "gripper_not_released"),
    (dict(height=.12, hand=.40), "resting_on_post_top"),
    (dict(height=.12, hand=.07), "held_above_post"),
    (dict(height=.07, hand=.40), "wedged_above_limit"),
    (dict(height=.07, hand=.067), "held_above_post"),
    (dict(dx=.06), "beside_post"),
    (dict(dx=.06, dy=.06, height=.12), "beside_post"),  # it is the post's position that matters, not the height
    (dict(dx=-.3), "far_from_post"),
])
def test_each_way_of_missing_the_rule_has_its_reason(changes, reason):
    assert ss.failure_reason(state(**changes)) == reason
    assert reason in ss.REASONS


def with_fingers(c, fingers):
    return dict(scalar(c), fingers_on_nut=fingers)


def test_held_means_a_finger_touches_the_nut_when_the_state_says_how_many_do():
    over_post = state(height=.12, hand=.07)   # the hand is within 10 cm, as it is just after letting go of a nut on the post
    assert ss.failure_reason(over_post) == "held_above_post"                      # no contact information: the distance proxy
    assert ss.failure_reason(with_fingers(over_post, 0)) == "resting_on_post_top"  # open hand hovering over a nut on the post
    assert ss.failure_reason(with_fingers(over_post, 1)) == "held_above_post"
    assert ss.failure_reason(with_fingers(over_post, 2)) == "held_above_post"
    loose = state(height=.07, hand=.07)
    assert ss.failure_reason(with_fingers(loose, 0)) == "wedged_above_limit"
    assert ss.failure_reason(with_fingers(state(height=.07, hand=.40), 1)) == "held_above_post"  # touching though the site is far
    # contact does not matter for the other reasons
    assert ss.failure_reason(with_fingers(state(hand=.03), 2)) == "gripper_not_released"
    assert ss.failure_reason(with_fingers(state(dx=.06), 2)) == "beside_post"


def test_a_nut_near_the_post_is_measured_by_its_distance_not_by_the_larger_offset():
    assert ss.failure_reason(state(dx=.08, dy=.08)) == "far_from_post"      # 11.3 cm away
    assert ss.failure_reason(state(dx=.06, dy=.06)) == "beside_post"        # 8.5 cm away
    assert ss.failure_reason(state(dy=-.08)) == "beside_post"


def test_every_reason_has_a_description_and_the_error_is_one_of_them():
    assert set(ss.REASONS) == {"success", "gripper_not_released", "resting_on_post_top", "held_above_post", "wedged_above_limit",
                               "beside_post", "far_from_post", "simulator_error"}
    assert all(text for text in ss.REASONS.values())
    assert "at the end" not in ss.REASONS["far_from_post"] and "never" not in ss.REASONS["far_from_post"]


def test_the_reason_boundaries():
    assert ss.failure_reason(state(height=.0999, hand=.4)) == "wedged_above_limit"
    assert ss.failure_reason(state(height=.1001, hand=.4)) == "resting_on_post_top"
    assert ss.failure_reason(state(height=.12, hand=.0999)) == "held_above_post"
    assert ss.failure_reason(state(height=.12, hand=.1001)) == "resting_on_post_top"
    assert ss.failure_reason(state(dx=.0999 / math.sqrt(2) + .0, dy=.0999 / math.sqrt(2))) == "beside_post"
    assert ss.failure_reason(state(dx=.1001)) == "far_from_post"


def scalar(c):
    return {key: (value.item() if hasattr(value, "item") else value) for key, value in c.items()}


def test_the_tracker_reports_what_the_rollout_ended_in_and_what_came_closest():
    follow = ss.RuleTracker()
    # far, then 6 cm beside the post, then resting on top of it, then the nut is knocked away
    for c in (state(dx=-.3), state(dx=.06), state(height=.14, hand=.35), state(dx=.2, height=.01), state(dx=-.3)):
        follow.update(scalar(c))
    result = follow.result()
    assert result["Failure_Reason"] == "far_from_post"      # the end of its video
    assert result["Closest_Reason"] == "beside_post" and result["Closest_Step"] == 1  # 6 cm off is nearer than 9 cm too high
    assert result["Final_Nut_Height_cm"] == pytest.approx(1.) and result["Final_XY_Offset_cm"] == pytest.approx(30.)
    off_in_y = ss.RuleTracker()
    off_in_y.update(scalar(state(dx=.03, dy=-.04)))
    assert off_in_y.result()["Final_XY_Offset_cm"] == pytest.approx(5.)  # the distance, not either offset
    assert result["Final_Hand_Distance_cm"] == pytest.approx(40.) and result["Final_Tilt_deg"] == pytest.approx(0.)
    assert result["Rule_Mismatch_Steps"] == 0
    stays = ss.RuleTracker()
    for c in (state(dx=.06), state(height=.14, hand=.35)):
        stays.update(scalar(c))
    assert stays.result()["Failure_Reason"] == "resting_on_post_top" and stays.result()["Final_Nut_Height_cm"] == pytest.approx(14.)


def test_a_rollout_that_met_the_rule_once_is_a_success_whatever_it_ended_in():
    follow = ss.RuleTracker()
    assert follow.result() == {}
    for c in (state(dx=.2), state(), state(dx=.2)):
        follow.update(scalar(c))
    result = follow.result()
    assert result["Failure_Reason"] == "success" and result["Closest_Reason"] == "success" and result["Closest_Step"] == 1
    assert result["Final_XY_Offset_cm"] == pytest.approx(20.)


def test_disagreements_with_the_environments_own_verdict_are_counted():
    follow = ss.RuleTracker()
    for c, environment in ((state(), True), (state(), False), (state(dx=.2), False), (state(dx=.2), True), (state(), None)):
        follow.update(scalar(c), environment)
    assert follow.result()["Rule_Mismatch_Steps"] == 2


def test_reason_counts_only_count_records_that_have_one():
    records = [{"Failure_Reason": "beside_post"}, {"Failure_Reason": "beside_post"}, {"Failure_Reason": "success"}, {"Return": 1}]
    assert ss.reason_counts(records) == {"beside_post": 2, "success": 1}


# ---- the real simulator -----------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def square():
    """A real NutAssemblySquare environment (robomimic wrapper), or skip."""
    return make_square(hard_reset=False)


def make_square(hard_reset):
    """A real NutAssemblySquare environment (robomimic wrapper); the test is skipped only when robosuite or MuJoCo is not installed."""
    try:
        import mujoco  # noqa: F401
        import robosuite  # noqa: F401
    except ImportError as exc:
        pytest.skip("robosuite or MuJoCo is not installed: {}".format(exc))
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    from robomimic.envs.env_base import EnvType
    from robosuite.controllers import load_composite_controller_config
    if ObsUtils.OBS_KEYS_TO_MODALITIES is None:
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": [], "rgb": []}})
    env = EnvUtils.create_env(
        EnvType.ROBOSUITE_TYPE, "NutAssemblySquare", render=False, render_offscreen=False, use_image_obs=False,
        robots="Panda", controller_configs=load_composite_controller_config(controller="BASIC", robot="Panda"),
        control_freq=20, ignore_done=True, reward_shaping=False, hard_reset=hard_reset, use_object_obs=True,
        use_camera_obs=False, seed=0)
    env.reset()
    return env


@pytest.fixture(autouse=True)
def no_display(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "disabled")


def place(env, position, quat=(1., 0., 0., 0.)):
    sim = env.env
    sim.sim.data.set_joint_qpos(sim.nuts[sim.nut_id].joints[0], np.concatenate([position, quat]))
    sim.sim.forward()


@pytest.mark.parametrize("label, offset, height, reason", [
    ("seated", (0, 0), .01, "success"),
    ("seated, 2 cm off", (.02, -.02), .01, "success"),
    ("resting on the post", (0, 0), .14, "resting_on_post_top"),
    ("caught part way", (0, 0), .07, "wedged_above_limit"),
    ("beside the post", (.06, 0), .01, "beside_post"),
    ("far away", (-.3, 0), .01, "far_from_post"),
])
def test_the_rule_and_its_reasons_agree_with_robosuites_own_verdict(square, label, offset, height, reason):
    from robomimic.scripts.inspect_square_insertions import probe_from_env
    square.reset()
    sim = square.env
    peg = np.array(sim.sim.data.body_xpos[sim.peg1_body_id])
    probe = probe_from_env(square)
    place(square, [peg[0] + offset[0], peg[1] + offset[1], sim.table_offset[2] + height])
    c = scalar(probe.read())
    assert bool(c["success"]) == bool(square.is_success()["task"]) == (reason == "success"), label
    assert ss.failure_reason(c) == reason, label
    assert c["height"] == pytest.approx(height, abs=1e-6)


def test_a_real_rollout_reports_what_came_closest_and_agrees_with_the_environment_at_every_step(square, monkeypatch):
    from robomimic.scripts import rollout_best as helper
    from robomimic.scripts.inspect_square_insertions import probe_from_env

    class ZeroPolicy:
        policy = SimpleNamespace(global_config=SimpleNamespace(use_goals=False))

        def start_episode(self):
            pass

        def __call__(self, ob, goal=None):
            return np.zeros(7)

    monkeypatch.setattr(square, "render", lambda **kwargs: np.zeros((8, 8, 3), np.uint8), raising=False)
    stats, trajectory, initial, frames = helper.collect_episode(
        ZeroPolicy(), square, horizon=6, camera_names=["agentview"], video_skip=1, probe_factory=probe_from_env)
    assert stats["Horizon"] == 6 and stats["Success_Rate"] == 0 and len(frames) == 6
    assert stats["Rule_Mismatch_Steps"] == 0  # the rule as written never disagreed with robosuite's own check
    assert stats["Failure_Reason"] in ("beside_post", "far_from_post")  # the nut starts away from the post
    assert 0 <= stats["Closest_Step"] < 6 and stats["Closest_Reason"] in ("beside_post", "far_from_post")
    assert stats["Final_Nut_Height_cm"] == pytest.approx(1., abs=.5)
    assert stats["Final_XY_Offset_cm"] > 3 and stats["Final_Hand_Distance_cm"] > 5
    # without a probe nothing is added
    stats, *_ = helper.collect_episode(ZeroPolicy(), square, horizon=2, camera_names=["agentview"], video_skip=1)
    assert "Failure_Reason" not in stats


@pytest.mark.parametrize("hand, expected", [(.03, False), (.0423, False), (.0426, True), (.067, True), (.25, True)])
def test_the_release_condition_agrees_with_robosuite_for_a_hand_at_each_distance(square, hand, expected):
    """Holding the handle (6 to 7 cm) does not block a success; a hand within 4.24 cm of the nut does."""
    from robomimic.scripts.inspect_square_insertions import probe_from_env
    square.reset()
    sim = square.env
    peg = np.array(sim.sim.data.body_xpos[sim.peg1_body_id])
    probe = probe_from_env(square)
    nut = np.array([peg[0], peg[1], sim.table_offset[2] + .01])
    place(square, nut)
    site = sim.robots[0].eef_site_id
    site = site["right"] if isinstance(site, dict) else site
    sim.sim.data.site_xpos[site] = nut + [0., 0., hand]  # read by robosuite's check and by the probe alike, until the next forward pass
    c = scalar(probe.read())
    assert bool(c["success"]) == bool(square.is_success()["task"]) == expected
    assert ss.failure_reason(c) == ("success" if expected else "gripper_not_released")


def test_a_nut_on_the_post_with_no_finger_on_it_is_resting_not_held(square):
    from robomimic.scripts.inspect_square_insertions import probe_from_env
    square.reset()
    sim = square.env
    peg = np.array(sim.sim.data.body_xpos[sim.peg1_body_id])
    probe = probe_from_env(square)
    place(square, [peg[0], peg[1], sim.table_offset[2] + .14])
    c = scalar(probe.read())
    assert c["fingers_on_nut"] == 0 and ss.failure_reason(c) == "resting_on_post_top"


def test_the_probe_reads_the_new_simulator_when_robosuite_rebuilds_it_on_reset():
    """With hard_reset robosuite makes a new simulator on every reset: a probe made before the reset would read the old one."""
    from robomimic.scripts import rollout_best as helper
    from robomimic.scripts.inspect_square_insertions import probe_from_env
    env = make_square(hard_reset=True)

    class ZeroPolicy:
        policy = SimpleNamespace(global_config=SimpleNamespace(use_goals=False))

        def start_episode(self):
            pass

        def __call__(self, ob, goal=None):
            return np.zeros(7)

    old = env.env.sim
    env.render = lambda **kwargs: np.zeros((8, 8, 3), np.uint8)
    stats, trajectory, initial, frames = helper.collect_episode(
        ZeroPolicy(), env, horizon=3, camera_names=["agentview"], video_skip=1, probe_factory=probe_from_env)
    sim = env.env
    assert sim.sim is not old  # the reset really built a new simulator
    nut_z = sim.sim.data.body_xpos[sim.obj_body_id["SquareNut"]][2]
    assert stats["Final_Nut_Height_cm"] / 100 == pytest.approx(nut_z - sim.table_offset[2], abs=1e-9)
    assert stats["Rule_Mismatch_Steps"] == 0
