"""Checks on a copy (curves from retarget.py), reported per move unit so the agent knows where to revise.

  accents      each dancer accent should have a visible character hit (a stop after fast motion) within 0.1 s
  completeness in each unit the character should move about as much (relative to its own average) as the dancer
               (relative to hers); much less means the copy ate the move
  in_frame     the slide plus the body tilt must keep the character inside the picture
  range        body tilt / turn within what the character can do (motion spec "can_do")
  requests     with plan layers: did each requested motion happen (spin turns, bounce height, sway, still, freeze);
               units the user took off the copy (still / freeze) skip the accent and completeness checks
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dancebuddy.agent.tools.perception.beats import detect_hits, match, smooth

# frame_margin: how far (x body height) the top of the body may drift sideways before the hair leaves the picture:
# the camera (rig_soft.py: 45 mm lens at 3 body heights, portrait) sees +-0.68 heights, the hair is +-0.35 wide
THRESHOLDS = {"accent_recall": 0.6, "completeness": 0.6, "frame_margin": 0.32, "tol_s": 0.1}


def _rot_series(q: list) -> Rot:
    q = np.asarray(q, float)
    return Rot.from_quat(q[:, [1, 2, 3, 0]])


def character_speed(curves: dict) -> np.ndarray:
    """Per-frame motion of the character (deg/s of its main bones + m/s of its root, as one number)."""
    ch, fps = curves["channels"], float(curves["fps"])
    n = int(curves["n_frames"])
    sp = np.zeros(n)
    for name, w in (("root.quat", 1.0), ("body_upper.quat", 0.7), ("arm_L_1.quat", 0.5), ("arm_R_1.quat", 0.5),
                    ("arm_L_3.quat", 0.3), ("arm_R_3.quat", 0.3)):
        if name in ch:
            r = _rot_series(ch[name])
            d = (r[1:] * r[:-1].inv()).magnitude()
            sp += w * np.degrees(np.r_[d[:1], d]) * fps
    if "root.loc" in ch:
        loc = np.asarray(ch["root.loc"], float)
        v = np.linalg.norm(np.diff(loc, axis=0), axis=1) * fps
        sp += 100.0 * np.r_[v[:1], v]  # 1 m/s ~ 100 deg/s
    if "root.squash" in ch:
        sq = np.asarray(ch["root.squash"], float)
        sp += 100.0 * np.abs(np.gradient(sq)) * fps
    return smooth(sp, 1.0)


def _yaw(curves: dict, rig: dict) -> np.ndarray:
    """(N,) degrees the whole body has turned about the vertical, unwrapped."""
    m = Rot.from_matrix(np.array(next(b["matrix"] for b in rig["bones"] if b["name"] == rig.get("root_bone", "root"))))
    x = (m * _rot_series(curves["channels"]["root.quat"]) * m.inv()).apply([1, 0, 0])
    return np.degrees(np.unwrap(np.arctan2(x[:, 1], x[:, 0])))


def check_requests(curves: dict, rig: dict, plan: dict | None) -> list[dict]:
    """One entry per plan layer: what was asked, what the curves show, passed."""
    fps, n = float(curves["fps"]), int(curves["n_frames"])
    height = float(rig.get("height_m", 1.4))
    ch = curves["channels"]
    yaw = _yaw(curves, rig)
    sp = character_speed(curves)
    out = []
    for ly in (plan or {}).get("layers") or []:
        a, b = int(round(ly["start_s"] * fps)), min(n - 1, int(round(ly["end_s"] * fps)))
        if b <= a or a >= n:
            out.append({"layer": ly, "passed": False, "found": "the window is empty or after the end of the clip "
                        f"({n / fps:.2f} s)"})
            continue
        kind, rec = ly["type"], {"layer": ly}
        if kind == "spin":
            turns = abs(yaw[min(n - 1, b + int(0.3 * fps))] - yaw[max(0, a - int(0.3 * fps))]) / 360.0
            want = max(1, round(float(ly.get("rate", 0.5)) * (ly["end_s"] - ly["start_s"])))
            found = f"{turns:.2f} turns (asked about {want})"
            if turns < 0.8 * want:
                found += ("; the dancer's own turns are added to the spin: spin her way, or add a still layer "
                          "to spin without the copy")
            rec.update(found=found, passed=bool(turns >= 0.8 * want))
        elif kind == "bounce":
            lift = np.asarray(ch.get("root.loc", np.zeros((n, 3))), float)[a:b + 1, 1] / height
            amp = float(np.percentile(lift, 95) - np.percentile(lift, 5))
            want = float(ly.get("size", 0.06))
            rec.update(found=f"hops {amp:.3f} body heights (asked {want:.3f})", passed=bool(amp >= 0.6 * want))
        elif kind == "sway":
            m = Rot.from_matrix(np.array(next(bb["matrix"] for bb in rig["bones"]
                                              if bb["name"] == rig.get("root_bone", "root"))))
            up = (m * _rot_series(ch["root.quat"]) * m.inv()).apply([0, 0, 1])[a:b + 1]
            side = np.degrees(np.arctan2(up[:, 0], up[:, 2]))
            amp = float(np.percentile(side, 95) - np.percentile(side, 5)) / 2
            want = float(ly.get("size", 12.0))
            rec.update(found=f"rocks +-{amp:.1f} deg (asked +-{want:.0f})", passed=bool(amp >= 0.5 * want))
        elif kind == "freeze":
            moving = float(np.median(sp[a + 1:b + 1])) if b > a + 1 else 0.0
            rec.update(found=f"median motion {moving:.1f} (0 = still)", passed=bool(moving < 5.0))
        elif kind == "still":
            rec.update(found="copy switched off in the window", passed=True)
        out.append(rec)
    return out


def check_curves(curves: dict, score: dict, rig: dict, spec: dict | None = None, plan: dict | None = None) -> dict:
    fps = float(curves["fps"])
    n = int(curves["n_frames"])
    height = float(rig.get("height_m", 1.4))
    tol = max(1, int(round(THRESHOLDS["tol_s"] * fps)))
    sp = character_speed(curves)
    hits = [f for f, st in detect_hits(sp, fps, sigma=0.7, scale=max(float(np.percentile(sp, 90)), 1e-6)) if st >= 0.25]
    accents = [a for a in score["accents"] if a < n]

    dancer = np.asarray(score.get("motion_energy", np.ones(n)), float)[:n]
    m = Rot.from_matrix(np.array(next(b["matrix"] for b in rig["bones"] if b["name"] == rig.get("root_bone", "root"))))
    up = (m * _rot_series(curves["channels"]["root.quat"]) * m.inv()).apply([0, 0, 1])  # bone frame -> armature
    loc = np.asarray(curves["channels"].get("root.loc", np.zeros((n, 3))), float)
    reach = np.abs(loc[:, 0]) + 0.7 * height * np.abs(up[:, 0])  # sideways offset of the top of the body
    out_frames = np.nonzero(reach > THRESHOLDS["frame_margin"] * height)[0]
    tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))
    can = (spec or {}).get("parts", {}).get("body", {}).get("can_do", {})
    tilt_max = float(can.get("tilt_deg", 25)) + 5

    off_copy = np.zeros(n, bool)  # frames the user took off the copy (still / freeze layers)
    for ly in (plan or {}).get("layers") or []:
        if ly["type"] in ("still", "freeze"):
            off_copy[int(round(ly["start_s"] * fps)):int(round(ly["end_s"] * fps)) + 1] = True
    accents = [a for a in accents if not off_copy[a]]
    matched, missed, _ = match(accents, hits, tol)
    recall = len(matched) / max(1, len(accents)) if accents else 1.0

    units = []
    for u in score["units"]:
        s, e = u["start"], min(n, u["end"] + 1)
        if e <= s:
            continue
        skip_copy = off_copy[s:e].mean() > 0.5
        ratio = float((sp[s:e].mean() / (sp.mean() + 1e-9)) / (dancer[s:e].mean() / (dancer.mean() + 1e-9) + 1e-9))
        miss = [int(a) for a in missed if s <= a < e]
        outs = [int(f) for f in out_frames if s <= f < e]
        problems = []
        if miss:
            problems.append(f"no visible hit on {len(miss)} of {len([a for a in accents if s <= a < e])} accents "
                            f"(frames {miss})")
        if skip_copy:
            ratio = 1.0
        if ratio < THRESHOLDS["completeness"] and not skip_copy:
            problems.append(f"the character moves {ratio:.2f}x as much as the dancer here (relative to each one's "
                            "average): the copy eats this move")
        if outs:
            worst = max(outs, key=lambda f: reach[f])
            problems.append(f"the character leaves the picture on {len(outs)} frames; at the worst frame ({worst}) "
                            f"the slide is {abs(loc[worst, 0]):.2f} m and the body tilt moves its top "
                            f"{0.7 * height * abs(up[worst, 0]):.2f} m sideways (limit together "
                            f"{THRESHOLDS['frame_margin'] * height:.2f} m)")
        if tilt[s:e].max() > tilt_max:
            problems.append(f"body tilt reaches {tilt[s:e].max():.0f} deg, beyond what the character can do")
        units.append({"unit_id": u["id"], "time_s": [u["start_s"], u["end_s"]], "dancer": u.get("label"),
                      "dancer_regions": u.get("dominant_regions"), "completeness": round(ratio, 2),
                      "missed_accents": miss, "frames_out_of_picture": len(outs), "problems": problems})
    checks = {
        "accents": {"value": round(recall, 3), "threshold": THRESHOLDS["accent_recall"],
                    "passed": recall >= THRESHOLDS["accent_recall"]},
        "completeness": {"passed": all(x["completeness"] >= THRESHOLDS["completeness"] for x in units),
                         "lowest": min((x["completeness"] for x in units), default=1.0)},
        "in_frame": {"passed": len(out_frames) == 0, "frames_out": int(len(out_frames))},
        "range": {"passed": float(tilt.max()) <= tilt_max, "max_tilt_deg": round(float(tilt.max()), 1)},
    }
    req = check_requests(curves, rig, plan)
    if req:
        checks["requests"] = {"passed": bool(all(r["passed"] for r in req)), "layers": req}
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks,
            "units": [x for x in units if x["problems"]], "all_units": units}


def rank(res: dict) -> tuple:
    """Higher is better: passed first, then fewer problems, then accent recall."""
    n_prob = sum(len(u["problems"]) for u in res["units"])
    return (res["passed"], -n_prob, res["checks"]["accents"]["value"])
