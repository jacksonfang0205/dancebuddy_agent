"""Motion layers and the agent's sandbox: no model calls, no Blender."""

import json
from pathlib import Path

import numpy as np
import pytest

from dancebuddy.agent.tools.dance.plan import default_plan
from dancebuddy.agent.tools.dance.checks import check_curves
from dancebuddy.agent.tools.dance.retarget import retarget
from dancebuddy.agent.tools.perception.video import build_dance_score, load_keypoints
from dancebuddy.evaluation.synth import make

ROOT = Path(__file__).resolve().parent.parent
RIG = json.loads((ROOT / "assets" / "plush_soft_rig.json").read_text())


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = make(tmp_path_factory.mktemp("synth"))
    kp = load_keypoints(d / "keypoints.npz")
    return kp, build_dance_score(kp)


def _run(synth, layers):
    kp, score = synth
    plan = default_plan(score)
    plan["layers"] = layers
    curves = retarget(kp, RIG, beats=score["beats"], score=score, plan=plan)
    return curves, check_curves(curves, score, RIG, plan=plan)


def test_spin_layer_turns_whole_turns(synth):
    d = synth[1]["duration_s"]
    _, res = _run(synth, [{"type": "spin", "start_s": 0, "end_s": d, "rate": 0.5},
                          {"type": "still", "start_s": 0, "end_s": d}])
    req = res["checks"]["requests"]
    assert req["passed"]
    assert res["checks"]["completeness"]["passed"]  # units off the copy are not held to it


def test_freeze_holds_the_pose(synth):
    curves, res = _run(synth, [{"type": "freeze", "start_s": 2.0, "end_s": 3.0}])
    fps = curves["fps"]
    q = np.asarray(curves["channels"]["root.quat"])[int(2 * fps):int(3 * fps)]
    assert np.allclose(q, q[0], atol=1e-4)
    assert res["checks"]["requests"]["passed"]


def test_shell_refuses_destructive_commands():
    from dancebuddy.sandbox import run_terminal
    assert "error" in run_terminal("rm -rf runs")
    assert "error" in run_terminal("git push origin main")
    assert run_terminal("echo hi")["exit_code"] == 0
