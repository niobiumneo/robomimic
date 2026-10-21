"""Square-nut insertion report: geometry helpers, the metrics, contacts read from a small MuJoCo scene, and the CLI."""

import json
from types import SimpleNamespace

import h5py
import mujoco
import numpy as np
import pytest

from robomimic.CAMI import continuous_mujoco as physics
from robomimic.scripts import inspect_square_insertions as insertions

TABLE_Z = .8
FLAT = .8095  # the nut's centre when it lies on the table, pressed in by half a millimetre so the contact exists
BAR = dict(hole_half=.02275, outer_half=.04375, half_thickness=.01)
PEG_XY = np.array([.2, 0.])

XML = """<mujoco><option gravity="0 0 0"/><worldbody>
<body name="table" pos="0 0 .7"><geom name="table_top" type="box" size=".5 .5 .1"/></body>
<body name="peg1" pos=".2 0 .9"><geom name="peg_geom" type="box" size=".016 .016 .1"/></body>
<body name="SquareNut_main" pos="0 0 .8095"><freejoint/>
  <geom name="SquareNut_g0" type="box" pos=".03325 0 0" size=".0105 .04375 .01" mass=".05"/>
  <geom name="SquareNut_g1" type="box" pos="-.03325 0 0" size=".0105 .04375 .01" mass=".05"/>
  <geom name="SquareNut_g2" type="box" pos="0 .03325 0" size=".04375 .0105 .01" mass=".05"/>
  <geom name="SquareNut_g3" type="box" pos="0 -.03325 0" size=".04375 .0105 .01" mass=".05"/></body>
<body name="gripper0_right_rightfinger" pos="0 0 1.5"><freejoint/>
  <geom name="pad_a" type="sphere" size=".01" mass=".01"/><geom name="pad_b" type="sphere" size=".01" pos="0 .005 0" mass=".01"/></body>
<body name="gripper0_right_leftfinger" pos="0 .1 1.5"><freejoint/><geom name="pad_c" type="sphere" size=".01" mass=".01"/></body>
<body name="gripper0_right_hand" pos="0 .3 1.5"><freejoint/><geom name="palm" type="box" size=".02 .02 .02" mass=".1"/></body>
<body name="robot0_link5" pos="0 .6 1.5"><freejoint/><geom name="forearm" type="box" size=".02 .02 .02" mass=".1"/></body>
</worldbody></mujoco>"""


def quat(angle_deg=0., axis=(1., 0., 0.)):
    half = np.radians(angle_deg) / 2
    return np.r_[np.cos(half), np.sin(half) * np.asarray(axis, float)]


def state(nut, tilt=0., held=True, hand=(0., .3, 1.5), arm=(0., .6, 1.5), finger_on_table=False):
    """qpos of the five free bodies: the nut, two fingertips, the palm and a forearm.

    When held, the fingertips rest on the nut's two opposite bars (the first finger has two collision geoms,
    as a real finger has). ``finger_on_table`` instead presses the first fingertip into the table beside the nut.
    """
    nut = np.asarray(nut, float)
    if held:
        first, second = nut + [.03325, 0., .0195], nut + [-.03325, 0., .0195]
    else:
        first, second = np.array([-.1, 0., FLAT]) if finger_on_table else np.array([0., 0., 1.5]), np.array([0., .1, 1.5])
    return np.concatenate([np.r_[nut, quat(tilt)], np.r_[first, quat()], np.r_[second, quat()],
                           np.r_[hand, quat()], np.r_[arm, quat()]])


def make_scene(table_z=TABLE_Z):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    def set_state(values):
        data.qpos[:] = values
        mujoco.mj_forward(model, data)
    return insertions.Scene(model, data, set_state, table_z)


def clean_episode():
    """Lifted, carried above the post and lowered over it."""
    rows = [state((0, 0, FLAT)) for _ in range(10)]
    rows += [state((0, 0, FLAT + (1.1 - FLAT) * k / 14)) for k in range(1, 15)]
    rows += [state((.2 * k / 19, 0, 1.1)) for k in range(1, 20)]
    rows += [state((.2, 0, 1.1 - (1.1 - FLAT) * k / 14)) for k in range(1, 15)]
    return np.array(rows)


def jam_episode():
    """Dragged along the table into the post with the palm resting on the table: robosuite still calls it a success."""
    rows = [state((.222 * k / 40, 0, FLAT), hand=(0., .3, .8195)) for k in range(41)]
    rows += [state((.222, 0, FLAT), hand=(0., .3, .8195)) for _ in range(29)]
    return np.array(rows)


def tilted_episode():
    """Lowered onto the post at 45 degrees and left there."""
    rows = [state((.2, 0, 1.1), tilt=45., held=False) for _ in range(10)]
    rows += [state((.2, 0, 1.1 - (1.1 - .83) * k / 20), tilt=45., held=False) for k in range(1, 21)]
    return np.array(rows)


def idle_episode():
    return np.array([state((0, 0, FLAT), held=False) for _ in range(20)])


def grasp_episode():
    """A fingertip pressed on the table for 6 steps before both fingers hold the nut, which is then lifted."""
    rows = [state((0, 0, FLAT), held=False, finger_on_table=True) for _ in range(6)]
    rows += [state((0, 0, FLAT)) for _ in range(6)]
    rows += [state((0, 0, FLAT + (1.1 - FLAT) * k / 10)) for k in range(1, 11)]
    return np.array(rows)


def early_slide_episode():
    """The nut is shifted 5 cm along the table while the fingers close on it, then lifted and lowered over the post."""
    rows = [state((.05 * k / 20, 0, FLAT)) for k in range(21)]
    rows += [state((.05, 0, FLAT + (1.1 - FLAT) * k / 10)) for k in range(1, 11)]
    rows += [state((.05 + .15 * k / 15, 0, 1.1)) for k in range(1, 16)]
    rows += [state((.2, 0, 1.1 - (1.1 - FLAT) * k / 12)) for k in range(1, 13)]
    return np.array(rows)


def late_slide_episode():
    """Lifted, put back on the table short of the post and slid onto it."""
    rows = [state((0, 0, FLAT + (1.1 - FLAT) * k / 10)) for k in range(11)]
    rows += [state((.1 * k / 8, 0, 1.1)) for k in range(1, 9)]
    rows += [state((.1, 0, 1.1 - (1.1 - FLAT) * k / 8)) for k in range(1, 9)]
    rows += [state((.1 + .1 * k / 25, 0, FLAT)) for k in range(1, 26)]
    return np.array(rows)


def test_geom_category_follows_robosuite_body_names():
    category = insertions.geom_category
    assert category("SquareNut_main") == "nut" and category("peg1") == "peg" and category("table") == "table"
    assert category("gripper0_right_rightfinger") == "finger"
    assert category("gripper0_right_hand") == "hand" and category("robot0_link5") == "arm"
    assert category("mount0_base") == "other" and category(None) == "other"


def test_post_through_hole_looks_along_the_nuts_own_axes():
    identity, peg = np.eye(3), PEG_XY
    check = lambda pos, rot=identity: insertions.post_through_hole(pos, rot, peg, .016, .02275)
    assert check([.2, 0, .83])
    assert check([.2, .004, .83])           # 6.4 mm of clearance on a side
    assert not check([.22, 0, .83])        # a corner of the post is 3.6 cm from the nut centre
    z45 = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4), 0], [np.sin(np.pi / 4), np.cos(np.pi / 4), 0], [0, 0, 1]])
    assert check([.2, 0, .83], z45)         # the post's diagonal (2.26 cm) still fits the hole's half-width
    assert not insertions.post_through_hole([.2, 0, .83], identity, peg, .0245, .02275)  # a fatter post cannot


def test_spearman_ignores_missing_values_and_flat_inputs():
    assert insertions.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.)
    assert insertions.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.)
    assert insertions.spearman([1, 2, np.nan, 4, 5], [1, 2, 9, 4, 5]) == pytest.approx(1.)
    assert np.isnan(insertions.spearman([1, 1, 1, 1], [1, 2, 3, 4]))
    # tied values share a rank, so the order they appear in does not matter
    tied = insertions.spearman([0, 0, 0, 1], [1, 2, 3, 4])
    assert tied == pytest.approx(3 / np.sqrt(15)) and insertions.spearman([0, 0, 0, 1], [3, 1, 2, 4]) == pytest.approx(tied)
    np.testing.assert_allclose(insertions.ranks([5, 1, 5, 5, 2]), [3, 0, 3, 3, 1])
    assert np.isnan(insertions.spearman([1, 2], [1, 2]))


def test_scene_reads_the_geometry_of_the_nut_and_post():
    scene = make_scene()
    assert scene.dims == pytest.approx(BAR)
    assert scene.peg_half == pytest.approx(.016) and scene.peg_top == pytest.approx(1.)
    np.testing.assert_allclose(scene.peg_pos, [.2, 0, .9])


def test_scene_needs_the_square_task_bodies():
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body name='a'><geom size='.1'/></body></worldbody></mujoco>")
    with pytest.raises(ValueError, match="SquareNut_main"):
        insertions.Scene(model, mujoco.MjData(model), lambda s: None, 0.)


def test_scene_counts_each_kind_of_contact():
    scene = make_scene()
    scene.set_state(state((0, 0, FLAT), hand=(0., .3, .8195), arm=(.3, .3, .8195)))
    held = scene.read()
    # two fingers hold the nut although the first one touches it with two geoms
    assert held["nut_table"] > 0 and held["nut_finger"] == 2 and held["nut_peg"] == 0
    assert held["hand_table"] > 0 and held["arm_table"] > 0 and held["finger_table"] == 0
    np.testing.assert_allclose(held["nut_pos"], [0, 0, FLAT])
    assert np.isnan(held["eef"]).all()  # no grasp site in this scene
    scene.set_state(state((.222, 0, 1.2), held=False))
    free = scene.read()
    assert free["nut_table"] == free["nut_peg"] == free["nut_finger"] == free["hand_table"] == free["arm_table"] == 0
    assert free["nut_peg_force"] == 0 and free["finger_table_force"] == 0
    scene.set_state(state((.222, 0, FLAT), held=False))
    pressed = scene.read()
    assert pressed["nut_peg"] > 0 and pressed["nut_peg_force"] > 0
    scene.set_state(state((.2, 0, FLAT), held=False))   # the post is inside the hole with room to spare
    assert scene.read()["nut_peg"] == 0


def test_a_fingertip_on_the_table_is_counted_apart_from_the_hand_and_has_a_force():
    scene = make_scene()
    scene.set_state(state((0, 0, FLAT), held=False, finger_on_table=True))
    read = scene.read()
    assert read["finger_table"] > 0 and read["hand_table"] == 0 and read["nut_finger"] == 0
    assert read["finger_table_force"] > 0


def analyze(rows, **kwargs):
    scene = make_scene()
    features, geometry = insertions.extract(scene, rows)
    return insertions.analyze(features, geometry, **kwargs)


def test_a_clean_insertion_is_lifted_and_seated():
    m = analyze(clean_episode())
    assert m["success"] and m["post_through_hole"] and not m["success_without_hole"] and not m["tilted_at_success"]
    assert m["seated_at_end"] and m["tilt_deg"] == pytest.approx(0, abs=1e-6)
    assert m["lift"] == pytest.approx(1.1 - FLAT - .0005, abs=1e-3) and m["drag_steps"] == 0 == m["drag_after_lift_steps"]
    assert m["clearance_at_post"] > .05 and m["nut_peg_steps"] == 0 and m["hand_table_steps"] == 0
    assert m["grasp_step"] == 0 and m["finger_table_before_grasp_steps"] == 0
    assert insertions.flags(m) == []


def test_jamming_the_nut_along_the_table_into_the_post_is_flagged():
    m = analyze(jam_episode())
    assert m["success"] and m["success_without_hole"] and not m["post_through_hole"] and not m["seated_at_end"]
    assert m["drag_steps"] >= 30 and m["drag_after_lift_steps"] == m["drag_steps"]  # never lifted: counted from the start
    assert m["clearance_at_post"] < 0 and m["lift"] < .01
    assert m["nut_peg_steps"] >= 20 and m["peak_post_force"] > 0
    assert m["hand_table_steps"] == 70 and m["arm_table_steps"] == 0
    assert set(insertions.flags(m, min_post_contact_steps=20)) == {
        "success_without_post_in_hole", "dragged_on_table", "arrived_beside_post", "long_contact_with_post"}
    assert "long_contact_with_post" not in insertions.flags(m, min_post_contact_steps=200)


def test_a_nut_that_hangs_over_the_post_at_a_steep_angle_counts_as_tilted():
    m = analyze(tilted_episode())
    assert m["success"] and m["tilt_deg"] == pytest.approx(45., abs=1e-4) and m["tilted_at_success"]
    assert m["post_through_hole"] and not m["seated_at_end"]
    assert insertions.flags(m) == ["tilted_at_success"]
    assert not analyze(tilted_episode(), max_tilt_deg=50.)["tilted_at_success"]


def test_a_seated_nut_must_be_close_to_lying_flat():
    rows = np.array([state((.2, 0, FLAT + .03), held=False) for _ in range(10)])
    m = analyze(rows)
    assert m["success"] and m["post_through_hole"] and not m["seated_at_end"]
    assert analyze(rows, seat_tolerance=.05)["seated_at_end"]


def test_an_episode_that_never_reaches_the_post_does_not_succeed():
    m = analyze(idle_episode())
    assert not m["success"] and m["first_success"] is None and not m["success_without_hole"]
    assert not m["tilted_at_success"] and np.isnan(m["clearance_at_post"])  # the post never came inside the nut's outline
    assert m["grasp_step"] is None and m["finger_table_before_grasp_steps"] == 0
    assert insertions.flags(m) == []


def test_the_release_distance_is_part_of_the_success_rule():
    scene = make_scene()
    rows = np.array([state((.2, 0, .83), held=False) for _ in range(6)])
    features, geometry = insertions.extract(scene, rows)
    features["eef"] = np.tile([.2, 0., .85], (6, 1))    # gripper 2 cm from the nut: still holding it
    assert not insertions.analyze(features, geometry)["success"]
    features["eef"] = np.tile([.2, 0., .95], (6, 1))
    assert insertions.analyze(features, geometry)["success"]


def test_fingertips_on_the_table_before_the_grasp_are_timed_and_measured():
    m = analyze(grasp_episode())
    assert m["grasp_step"] == 6 and m["finger_table_before_grasp_steps"] == 6 and m["finger_table_steps"] == 6
    assert m["finger_table_peak_force_before_grasp"] > 0
    assert m["finger_table_peak_force"] == pytest.approx(m["finger_table_peak_force_before_grasp"])
    # contact after the grasp is not counted as before it
    rows = np.concatenate([grasp_episode(), [state((0, 0, 1.1), held=False, finger_on_table=True)] * 3])
    later = analyze(rows)
    assert later["finger_table_before_grasp_steps"] == 6 and later["finger_table_steps"] == 9


def test_sliding_the_nut_while_the_fingers_close_is_not_counted_as_dragging_it_onto_the_post():
    early, late = analyze(early_slide_episode()), analyze(late_slide_episode())
    assert early["drag_steps"] >= 15 and early["drag_after_lift_steps"] == 0
    assert "dragged_on_table" not in insertions.flags(early)
    assert late["drag_after_lift_steps"] >= 20 and late["drag_steps"] == late["drag_after_lift_steps"]
    assert "dragged_on_table" in insertions.flags(late)
    assert "dragged_on_table" not in insertions.flags(late, min_drag_steps=30)


def test_episodes_need_states_and_a_model_file(tmp_path):
    path = tmp_path / "plain.hdf5"
    with h5py.File(path, "w") as f:
        f.create_group("data/demo_0")["actions"] = np.zeros((3, 7))
    with pytest.raises(ValueError, match="states and a model_file"):
        insertions.episodes(str(path))


def write_dataset(path, episodes, drift=None, recorded=None, later=None, empty=()):
    """A robomimic file; ``later`` gives next_states for each episode, ``empty`` the names of episodes with no states."""
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "NutAssemblySquare", "type": 1, "env_kwargs": {}})
        if drift is not None:
            data.attrs["cami_c"] = json.dumps({"demos": drift})
        for index, rows in enumerate(episodes):
            demo = data.create_group("demo_{}".format(index))
            demo["states"] = rows
            if later is not None and later[index] is not None:
                demo["next_states"] = later[index]
            demo.attrs["model_file"] = "<mujoco/>"
            if recorded is not None:
                demo.attrs["success"] = recorded[index]
        for name in empty:
            demo = data.create_group(name)
            demo["states"] = np.zeros((0, 35))
            demo.attrs["model_file"] = "<mujoco/>"
            demo.attrs["rollout_error"] = "MujocoException: instability"


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.setattr(physics, "make_replay_env", lambda group: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(physics, "reset_demo", lambda env, demo: None)
    monkeypatch.setattr(insertions, "scene_from_env", lambda env: make_scene())


def test_report_over_a_dataset_flags_the_jammed_episode_only(tmp_path, replay):
    path = tmp_path / "demos.hdf5"
    write_dataset(path, [clean_episode(), jam_episode(), tilted_episode(), idle_episode()],
                  recorded=[True, True, True, False])
    report = insertions.build_report(str(path), min_post_contact_steps=20)
    flagged = {name: m["flags"] for name, m in report["episodes"].items() if m["flags"]}
    assert set(flagged) == {"demo_1", "demo_2"} and "dragged_on_table" in flagged["demo_1"]
    assert report["skipped"] == [] and report["thresholds"]["min_post_contact_steps"] == 20
    summary = insertions.summarize(report)
    assert summary["episodes"] == 4 and summary["successes"] == 3 and summary["never_succeeded"] == 1
    assert summary["success_with_post_in_hole"] == 2 and summary["success_without_post_in_hole"] == 1
    assert summary["tilted_at_success"] == 1 and summary["seated_at_end"] == 1
    assert summary["recorded_vs_replayed_success_mismatch"] == 0
    assert summary["dragged_after_lift"] == pytest.approx(.25) and summary["dragged_at_any_time"] == pytest.approx(.25)
    assert summary["long_post_contact"] == pytest.approx(.25) and summary["arrived_beside_post"] == pytest.approx(.25)
    assert summary["hand_table"] == pytest.approx(.25) and summary["grasped"] == pytest.approx(.5)
    assert report["episodes"]["demo_0"]["recorded_success"] is True
    assert insertions.summarize(insertions.build_report(str(path), limit=1))["episodes"] == 1


def test_the_state_after_the_last_action_is_replayed_when_next_states_records_it(tmp_path, replay):
    rows = clean_episode()
    first = analyze(rows)["first_success"]
    # a rollout stops at the action after which the task counts as done: that state is only in next_states
    states, following = rows[:first], rows[1:first + 1]
    path = tmp_path / "rollouts.hdf5"
    write_dataset(path, [states, states, states], recorded=[True, True, True], later=[following, None, rows[2:first + 2]])
    episodes = insertions.build_report(str(path))["episodes"]
    assert episodes["demo_0"]["success"] and episodes["demo_0"]["first_success"] == first and episodes["demo_0"]["steps"] == first + 1
    assert not episodes["demo_1"]["success"] and episodes["demo_1"]["steps"] == first  # no next_states to add
    assert not episodes["demo_2"]["success"] and episodes["demo_2"]["steps"] == first   # not the same trajectory: not trusted
    summary = insertions.summarize(insertions.build_report(str(path)))
    assert summary["recorded_vs_replayed_success_mismatch"] == 2


def test_rollouts_without_states_are_listed_as_skipped_and_never_judged(tmp_path, replay, capsys):
    path = tmp_path / "rollouts.hdf5"
    write_dataset(path, [clean_episode(), jam_episode()], empty=["demo_2"])
    report = insertions.build_report(str(path))
    assert set(report["episodes"]) == {"demo_0", "demo_1"} and report["skipped"] == ["demo_2"]
    summary = insertions.summarize(report)
    insertions.print_summary(summary)
    assert "Skipped, with no replayable states: 1 (demo_2)" in capsys.readouterr().out
    # a mask would drop them without judging them
    with pytest.raises(ValueError, match="no replayable states"):
        insertions.write_mask(str(path), report, "x")
    with pytest.raises(SystemExit):
        insertions.main(["--dataset", str(path), "--write-mask", "x", "--quiet"])
    with h5py.File(path) as f:
        assert "mask" not in f


def test_report_compares_replay_drift_with_the_flagged_episodes(tmp_path, replay):
    path = tmp_path / "demos.hdf5"
    drift = {"demo_{}".format(i): dict(checked_transitions=100, transitions_over_tolerance=n)
             for i, n in enumerate([5, 60, 40, 3])}
    write_dataset(path, [clean_episode(), jam_episode(), tilted_episode(), idle_episode()], drift=drift)
    summary = insertions.summarize(insertions.build_report(str(path)))
    assert summary["drift_over_tolerance_median"] == pytest.approx(.225)
    flagged, others = summary["drift_over_tolerance_flagged_vs_not"]
    assert flagged == pytest.approx(.5) and others == pytest.approx(.04)
    assert summary["drift_vs_drag_after_lift_steps"] > .5


def test_the_printed_report_names_the_limits_it_used_and_the_fingertip_timing(tmp_path, replay, capsys):
    path = tmp_path / "demos.hdf5"
    write_dataset(path, [grasp_episode(), clean_episode()])
    insertions.main(["--dataset", str(path), "--min-drag-steps", "7", "--min-post-contact-steps", "33", "--quiet"])
    text = capsys.readouterr().out
    assert "for 7+ steps" in text and "for 33+ steps" in text
    assert "both fingers held the nut in 100% of episodes; fingertips on the table before that: 50% (median 3 steps" in text


def test_cli_prints_and_writes_the_outputs(tmp_path, replay, capsys):
    path = tmp_path / "demos.hdf5"
    write_dataset(path, [clean_episode(), jam_episode(), tilted_episode(), idle_episode()])
    csv_path, json_path = tmp_path / "out.csv", tmp_path / "out.json"
    report, summary = insertions.main(["--dataset", str(path), "--csv", str(csv_path), "--json", str(json_path),
                                       "--write-mask", "no_jam", "--min-post-contact-steps", "20", "--quiet"])
    text = capsys.readouterr().out
    assert "post not in the hole: 1" in text and "Flagged episodes: 2 of 4" in text
    assert "raising a jam flag" in text and "Wrote mask/no_jam with 2 episodes" in text
    rows = csv_path.read_text().splitlines()
    assert rows[0].startswith("episode,steps,success") and len(rows) == 5
    assert "finger_table_peak_force_before_grasp" in rows[0] and "grasp_step" in rows[0]
    assert any(row.startswith("demo_1,") and "dragged_on_table" in row for row in rows)
    saved = json.loads(json_path.read_text())
    assert saved["summary"]["successes"] == 3 and set(saved["episodes"]) == {"demo_0", "demo_1", "demo_2", "demo_3"}
    with h5py.File(path) as f:
        assert [b.decode() for b in f["mask/no_jam"][()]] == ["demo_0", "demo_3"]
    # none of the outputs may replace a file or mask that exists
    with pytest.raises(SystemExit):
        insertions.main(["--dataset", str(path), "--csv", str(csv_path), "--quiet"])
    with pytest.raises(SystemExit):  # refused before any episode is replayed
        insertions.main(["--dataset", str(path), "--write-mask", "no_jam", "--quiet"])
    assert "already exists" in capsys.readouterr().err
    with pytest.raises(ValueError, match="already exists"):  # and again at the end, if the file changed in between
        insertions.write_mask(str(path), report, "no_jam")


def test_a_mask_leaves_out_the_jam_flags_unless_told_otherwise(tmp_path, replay):
    path = tmp_path / "demos.hdf5"
    write_dataset(path, [clean_episode(), early_slide_episode(), late_slide_episode(), tilted_episode()])
    report = insertions.build_report(str(path), min_post_contact_steps=1)
    flags = {name: set(m["flags"]) for name, m in report["episodes"].items()}
    assert "dragged_on_table" in flags["demo_2"] and "dragged_on_table" not in flags["demo_1"]
    assert flags["demo_0"] == set() and "tilted_at_success" in flags["demo_3"]
    # the default mask leaves out only the jam flags, "everything" any flag, and a named flag only that one
    default = insertions.write_mask(str(path), report, "default")
    everything = insertions.write_mask(str(path), report, "everything", insertions.FLAG_NAMES)
    only_tilt = insertions.write_mask(str(path), report, "tilt", ("tilted_at_success",))
    assert default == [n for n in flags if not flags[n] & set(insertions.JAM_FLAGS)]
    assert everything == [n for n in flags if not flags[n]]
    assert only_tilt == [n for n in flags if "tilted_at_success" not in flags[n]]
    assert "demo_3" not in default and "demo_3" not in only_tilt and "demo_0" in everything


def test_cli_chooses_the_flags_that_keep_an_episode_out_of_the_mask(tmp_path, replay):
    path = tmp_path / "demos.hdf5"
    write_dataset(path, [clean_episode(), jam_episode(), tilted_episode()])
    insertions.main(["--dataset", str(path), "--write-mask", "only_tilt", "--mask-flags", "tilted_at_success", "--quiet"])
    insertions.main(["--dataset", str(path), "--write-mask", "all", "--mask-flags", "all", "--quiet"])
    with h5py.File(path) as f:
        assert [b.decode() for b in f["mask/only_tilt"][()]] == ["demo_0", "demo_1"]
        assert [b.decode() for b in f["mask/all"][()]] == ["demo_0"]
    with pytest.raises(SystemExit):
        insertions.main(["--dataset", str(path), "--write-mask", "bad", "--mask-flags", "not_a_flag", "--quiet"])


def test_writing_a_mask_needs_at_least_one_acceptable_episode(tmp_path, replay):
    path = tmp_path / "demos.hdf5"
    write_dataset(path, [jam_episode(), tilted_episode()])
    with pytest.raises(ValueError, match="nothing to write"):
        insertions.main(["--dataset", str(path), "--write-mask", "none", "--quiet"])
    with h5py.File(path) as f:
        assert "mask" not in f


def test_a_mask_cannot_be_written_from_a_partial_run(tmp_path, replay, capsys):
    path = tmp_path / "demos.hdf5"
    write_dataset(path, [clean_episode(), idle_episode()])
    with pytest.raises(SystemExit):
        insertions.main(["--dataset", str(path), "--write-mask", "partial", "--limit", "1", "--quiet"])
    assert "--limit" in capsys.readouterr().err
    with h5py.File(path) as f:
        assert "mask" not in f
