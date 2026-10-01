"""Study report: what a paper should state about how RAWR shaped what participants heard.

The operator's decisions are part of the manipulation, so a study should report the
review policy, how often replies were changed or withheld, how long the operator took,
and whether the spoken replies showed the requested behavior (the manipulation check).
build_report() computes all of this from a session database; render_markdown() fills
docs/reporting_template.md with it. tools/study_report.py writes both files.
"""

import json
import re
import sqlite3
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Optional

from antagonist_robot.conversation.avct_manager import CATEGORY_DEFINITIONS

TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "docs" / "reporting_template.md"
SPOKEN = ("sent", "auto_sent")
DISPOSITIONS = ["sent", "auto_sent", "tempered", "intensified", "regenerated", "withheld_session_ended", "pending"]
MONITOR_DIMENSIONS = ("privacy", "discrimination", "manipulation", "psych_harm", "insulting")


def _j(text, default):
    try:
        value = json.loads(text) if text else default
    except (TypeError, ValueError):
        return default
    return default if value is None else value


def _pct(n: int, total: int) -> Optional[float]:
    return round(100 * n / total, 1) if total else None


def _summary(values: list) -> dict:
    """n, mean, sd, median, quartiles, min, max of the non-missing values."""
    v = sorted(x for x in values if x is not None)
    if not v:
        return {"n": 0}
    out = {"n": len(v), "mean": round(statistics.fmean(v), 2), "median": round(statistics.median(v), 2),
           "min": v[0], "max": v[-1]}
    out["sd"] = round(statistics.stdev(v), 2) if len(v) > 1 else None
    if len(v) > 1:
        q = statistics.quantiles(v, n=4, method="inclusive")
        out["q1"], out["q3"] = round(q[0], 2), round(q[2], 2)
    return out


def _hold_reason(reason: str) -> str:
    """Group hold reasons that carry numbers (e.g. 'judge fidelity 3/10 below 4 ...')."""
    if reason.startswith("judge fidelity"):
        return "judge: fidelity below threshold"
    return re.sub(r"\s*\(.*\)$", "", reason)


def _held_for(cands: list, events: list) -> dict:
    """candidate_id -> every reason it waited for the operator.

    Databases written before the final reasons were stored at decision time keep only
    the reasons known when the reply was generated; the later ones come from the events.
    """
    out = {c["candidate_id"]: {_hold_reason(r) for r in _j(c.get("blocked_reasons_json"), [])} for c in cands}
    for e in events:
        if e["candidate_id"] not in out:
            continue
        p = _j(e["payload_json"], {})
        if e["event"] == "monitor_block":
            reason = "monitor unavailable" if p.get("error") else "monitor: clear psychosocial risk"
        elif e["event"] == "fidelity_block":
            reason = ("fidelity judge unavailable" if p.get("error") else
                      "judge: reply refuses the requested behavior" if p.get("refused") else
                      "judge: fidelity below threshold")
        elif e["event"] == "hold":
            reason = "held by operator"
        else:
            continue
        out[e["candidate_id"]].add(reason)
    return out


def _category(code: Optional[str]) -> str:
    if code in CATEGORY_DEFINITIONS:
        return f"{code} ({CATEGORY_DEFINITIONS[code]['name']})"
    return code or "?"


def _condition(row: dict) -> str:
    mods = " ".join(_j(row.get("modifiers_json"), []))
    return f"polar {row.get('polar_level'):+d}, {_category(row.get('category'))}, intensity {row.get('subtype')}" + (
        f", {mods}" if mods else "")


def _distinct(values) -> list:
    out = []
    for v in values:
        if v not in out:
            out.append(v)
    return out


def build_report(db_path: str, session_ids: Optional[list] = None, participant_ids: Optional[list] = None,
                 title: str = "RAWR study report") -> dict:
    """Compute the study report from a RAWR session database (read-only)."""
    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    def rows(query, args=()):
        return [dict(r) for r in conn.execute(query, args).fetchall()]

    sessions = rows("SELECT * FROM sessions ORDER BY start_time")
    if session_ids:
        sessions = [s for s in sessions if s["session_id"] in session_ids]
    if participant_ids:
        sessions = [s for s in sessions if s["participant_id"] in participant_ids]
    ids = [s["session_id"] for s in sessions]
    marks = ", ".join("?" * len(ids)) or "NULL"
    turns = rows(f"SELECT * FROM turns WHERE session_id IN ({marks}) ORDER BY session_id, turn_number", ids)
    cands = rows(f"SELECT * FROM candidates WHERE session_id IN ({marks}) ORDER BY candidate_id", ids)
    events = rows(f"SELECT * FROM operator_events WHERE session_id IN ({marks}) ORDER BY event_id", ids)
    conn.close()

    configs = [_j(s.get("config_snapshot"), {}) for s in sessions]
    starts = {e["session_id"]: _j(e["payload_json"], {}) for e in events if e["event"] == "session_start"}
    ends = {e["session_id"]: _j(e["payload_json"], {}) for e in events if e["event"] == "session_end"}

    # --- sessions -----------------------------------------------------------------
    session_rows, durations = [], []
    for s in sessions:
        minutes = None
        if s.get("end_time"):
            minutes = round((datetime.fromisoformat(s["end_time"]) -
                             datetime.fromisoformat(s["start_time"])).total_seconds() / 60, 1)
            durations.append(minutes)
        session_rows.append({
            "session_id": s["session_id"], "participant_id": s["participant_id"], "start_time": s["start_time"],
            "duration_min": minutes, "initial_condition": _condition(s),
            "spoken_turns": sum(t["session_id"] == s["session_id"] for t in turns),
            "candidates": sum(c["session_id"] == s["session_id"] for c in cands),
            "end_reason": ends.get(s["session_id"], {}).get("reason", "not ended (crash or still running)"),
        })

    # --- setup (from the configuration each session recorded) -----------------------
    def setting(path):
        values = []
        for cfg in configs:
            v = cfg
            for key in path.split("."):
                v = v.get(key) if isinstance(v, dict) else None
            values.append(v)
        return _distinct(values)

    caps = _distinct(json.dumps(starts.get(i, {}).get("capabilities", {}), sort_keys=True) for i in ids)
    caps = [json.loads(c) for c in caps]

    def ran(get, path):
        values = _distinct(get(c) for c in caps if c)
        return values if any(v is not None for v in values) else setting(path)

    setup = {
        "robot_backend": setting("robot.backend"),
        "robot": _distinct(c.get("robot") for c in caps),
        "robot_speech": _distinct(c.get("speech") for c in caps),
        "stop_speech": _distinct(c.get("interrupt") for c in caps),
        "nonverbal_cues": setting("robot.expressions"),
        "llm_provider": setting("llm.provider_name"), "llm_model": setting("llm.model"),
        "llm_temperature": setting("llm.temperature"), "llm_max_tokens": setting("llm.max_tokens"),
        # what actually ran (recorded at session start), else the configuration file
        "monitor_enabled": ran(lambda c: c.get("monitor"), "monitor.enabled"),
        "monitor_model": setting("monitor.model"),
        "judge_enabled": ran(lambda c: c.get("fidelity", {}).get("judge"), "fidelity.judge_enabled"),
        "judge_model": setting("fidelity.judge_model"),
        "judge_blocks_below": ran(lambda c: c.get("fidelity", {}).get("block_below"),
                                  "fidelity.block_auto_send_below"),
        "detector_enabled": ran(lambda c: c.get("fidelity", {}).get("detector"), "fidelity.detector_enabled"),
        "review_mode": setting("operator.review_mode"), "hold_seconds": setting("operator.hold_seconds"),
        "auto_release_below_risk": setting("operator.block_auto_send_at"),
        "model_can_end_session": setting("operator.model_can_end_session"),
        "review_policy_changes": [_j(e["payload_json"], {}) | {"session_id": e["session_id"]}
                                  for e in events if e["event"] == "review_policy"],
    }

    # --- conditions ---------------------------------------------------------------------
    by_condition = Counter(_condition(t) for t in turns)
    changes = Counter(_j(e["payload_json"], {}).get("source", "operator")
                      for e in events if e["event"] == "settings_change")
    conditions = {
        "spoken_turns_by_condition": [{"condition": k, "turns": n, "percent": _pct(n, len(turns))}
                                      for k, n in by_condition.most_common()],
        "condition_changes": dict(changes),
        "turns_spoken_below_requested_level": sum(
            1 for t in turns if t.get("requested_polar_level") is not None
            and t["polar_level"] is not None and abs(t["polar_level"]) < abs(t["requested_polar_level"])),
    }

    # --- operator review -------------------------------------------------------------------
    disp = Counter(c.get("disposition") or "pending" for c in cands)
    by = Counter(c.get("decided_by") for c in cands if c.get("decided_by"))
    held_for = _held_for(cands, events)
    reasons = Counter(r for rs in held_for.values() for r in rs)
    held = sum(1 for rs in held_for.values() if rs)
    event_counts = Counter(e["event"] for e in events)
    review = {
        "candidates_generated": len(cands), "spoken_turns": len(turns),
        "candidates_per_spoken_turn": round(len(cands) / len(turns), 2) if turns else None,
        "dispositions": [{"disposition": d, "n": disp.get(d, 0), "percent": _pct(disp.get(d, 0), len(cands))}
                         for d in DISPOSITIONS if disp.get(d, 0) or d != "pending"],
        "decided_by": dict(by),
        "candidates_held_for_operator": held, "held_percent": _pct(held, len(cands)),
        "hold_reasons": [{"reason": r, "n": n} for r, n in reasons.most_common()],
        "operator_decision_ms": _summary([c.get("review_ms") for c in cands if c.get("decided_by") == "operator"]),
        "review_wait_per_spoken_turn_ms": _summary([t.get("latency_review_ms") for t in turns]),
        "operator_events": dict(sorted(event_counts.items())),
        "stop_speech_used": event_counts.get("stop_speech", 0),
        "speech_interrupted_turns": sum(1 for t in turns if t.get("speech_completed") == 0),
    }

    # --- manipulation check (judge on the replies participants heard) ------------------------
    spoken = [c for c in cands if c.get("disposition") in SPOKEN]
    judged = [(c, _j(c.get("fidelity_judge_json"), {})) for c in spoken]
    judged = [(c, j) for c, j in judged if j and not j.get("error") and j.get("fidelity") is not None]
    detector = [_j(c.get("fidelity_detector_json"), {}) for c in spoken]
    detector = [d for d in detector if d and d.get("p_faithful") is not None]
    block_below = next((v for v in setup["judge_blocks_below"] if v is not None), None)

    def check(group):
        fid = [j["fidelity"] for _, j in group]
        return {
            "n_judged": len(group), "fidelity": _summary(fid),
            "category_match_percent": _pct(sum(j.get("matched_category") == c.get("category") for c, j in group),
                                           len(group)),
            "below_threshold": sum(f < block_below for f in fid) if block_below is not None else None,
            "refusals": sum(bool(j.get("refused")) for _, j in group),
            "judged_neutral": sum(j.get("matched_category") == "NEUTRAL" for _, j in group),
            "intensity_estimate": _summary([j.get("intensity_est") for _, j in group]),
        }

    per_condition = {}
    for c, j in judged:
        key = f"polar {c['polar_level']:+d}, {_category(c['category'])}, intensity {c['subtype']}"
        per_condition.setdefault(key, []).append((c, j))
    manipulation = {
        "spoken_replies": len(spoken), "judge_coverage_percent": _pct(len(judged), len(spoken)),
        "threshold": block_below, "overall": check(judged),
        "by_condition": [{"condition": k, **check(v)} for k, v in sorted(per_condition.items())],
        "detector": {"n": len(detector), "p_faithful": _summary([d["p_faithful"] for d in detector]),
                     "flagged_softened": sum(bool(d.get("softened")) for d in detector)},
    }

    # --- risk and wellbeing ---------------------------------------------------------------------
    def monitor_max(c):
        scores = _j(c.get("monitor_json"), {}).get("scores") or {}
        return max((scores.get(k) or 0 for k in MONITOR_DIMENSIONS), default=None) if scores else None

    monitored = [c for c in cands if monitor_max(c) is not None]
    risk = {
        "risk_rating_spoken": dict(Counter(c.get("risk_rating") for c in spoken)),
        "monitor_scored": len(monitored),
        "monitor_clear_risk_generated": sum(monitor_max(c) >= 2 for c in monitored),
        "monitor_clear_risk_spoken": sum(monitor_max(c) >= 2 for c in monitored if c in spoken),
        "monitor_errors": sum(1 for c in cands if _j(c.get("monitor_json"), {}).get("error")),
        "monitor_clear_risk_by_dimension": {
            k: sum((_j(c.get("monitor_json"), {}).get("scores") or {}).get(k, 0) >= 2 for c in monitored)
            for k in MONITOR_DIMENSIONS},
        "participant_distress_cues": event_counts.get("participant_distress", 0),
        "sessions_with_distress_cue": len({e["session_id"] for e in events if e["event"] == "participant_distress"}),
        "session_end_reasons": dict(Counter(r["end_reason"] for r in session_rows)),
    }

    latency = {
        "generation_ms": _summary([c.get("latency_llm_ms") for c in cands]),
        "speech_ms": _summary([t.get("latency_tts_ms") for t in turns]),
        "turn_total_ms": _summary([t.get("latency_total_ms") for t in turns]),
    }

    report = {
        "title": title, "generated_at": datetime.now().isoformat(timespec="seconds"),
        "database": str(Path(db_path).name),
        "study": {"sessions": len(sessions), "participants": len({s["participant_id"] for s in sessions}),
                  "first_session": sessions[0]["start_time"] if sessions else None,
                  "last_session": sessions[-1]["start_time"] if sessions else None,
                  "duration_min": _summary(durations)},
        "sessions": session_rows, "setup": setup, "conditions": conditions, "review": review,
        "manipulation_check": manipulation, "risk": risk, "latency": latency,
    }
    report["methods_paragraph"] = methods_paragraph(report)
    return report


# --- Markdown -----------------------------------------------------------------------------------

def _one(values) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "not recorded"
    return " / ".join({True: "on", False: "off"}.get(v, str(v)) if isinstance(v, bool) else str(v) for v in vals) + (" (varied across sessions)" if len(vals) > 1 else "")


def _n(value) -> str:
    return "n/a" if value is None else str(value)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _ms(s: dict, unit="s") -> str:
    """Median and IQR of a _summary(); values in ms shown in seconds, or already in minutes."""
    if not s.get("n"):
        return "n/a"
    f = (lambda x: f"{x / 1000:.1f} s") if unit == "s" else (lambda x: f"{x:.1f} min")
    iqr = f", IQR {f(s['q1'])}–{f(s['q3'])}" if "q1" in s else ""
    return f"median {f(s['median'])}{iqr} (n = {s['n']})"


def _table(header: list, body: list) -> str:
    if not body:
        return "_None._"
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join("" if v is None else str(v) for v in row) + " |" for row in body]
    return "\n".join(lines)


def methods_paragraph(r: dict) -> str:
    st, rv, mc = r["setup"], r["review"], r["manipulation_check"]
    mode = _one(st["review_mode"])
    if "timed" in st["review_mode"]:
        policy = (f"Replies were released automatically after a {_one(st['hold_seconds'])} s review window "
                  f"unless the operator held them, a background check flagged them, or they were rated "
                  f"{_one(st['auto_release_below_risk'])} or higher, in which case the operator had to send them")
    else:
        policy = "Every reply waited for the operator to send it"
    disp = {d["disposition"]: d for d in rv["dispositions"]}
    changed = sum(disp.get(k, {}).get("n", 0) for k in ("tempered", "intensified", "regenerated"))
    fid = mc["overall"]["fidelity"]
    check = (f"An LLM judge ({_one(st['judge_model'])}) rated how well each spoken reply showed the requested "
             f"behavior (0–10); spoken replies scored a mean of {fid['mean']} (SD {fid['sd']}), and "
             f"{mc['overall']['category_match_percent']}% showed the requested category."
             if fid.get("n") else "No judge ratings were recorded for the spoken replies.")
    return (
        f"The robot ({_one(st['robot'])}) spoke replies generated by {_one(st['llm_model'])} "
        f"(temperature {_one(st['llm_temperature'])}) through the RAWR operator console in {mode} review mode. "
        f"{policy}. Across {_plural(r['study']['sessions'], 'session')} with "
        f"{_plural(r['study']['participants'], 'participant')}, "
        f"{rv['candidates_generated']} replies were generated and {rv['spoken_turns']} were spoken; the operator "
        f"tempered, intensified, or regenerated {changed} ({_pct(changed, rv['candidates_generated'])}%), and "
        f"{rv['held_percent']}% waited for the operator's decision, which took {_ms(rv['operator_decision_ms'])}. "
        f"{check} The robot's ability to stop speech mid-sentence was {_one(st['stop_speech'])}."
    )


def render_markdown(r: dict, template_path: Optional[Path] = None) -> str:
    st, cd, rv, mc, rk, lt = (r["setup"], r["conditions"], r["review"], r["manipulation_check"], r["risk"],
                              r["latency"])
    fid = mc["overall"]["fidelity"]
    values = {
        "title": r["title"], "generated_at": r["generated_at"], "database": r["database"],
        "n_sessions": r["study"]["sessions"], "n_participants": r["study"]["participants"],
        "first_session": (r["study"]["first_session"] or "n/a")[:16].replace("T", " "),
        "last_session": (r["study"]["last_session"] or "n/a")[:16].replace("T", " "),
        "session_duration": _ms(r["study"]["duration_min"], unit="min"),
        "robot": _one(st["robot"]), "robot_backend": _one(st["robot_backend"]),
        "robot_speech": _one(st["robot_speech"]), "stop_speech": _one(st["stop_speech"]),
        "nonverbal_cues": _one(st["nonverbal_cues"]),
        "llm": f"{_one(st['llm_model'])} via {_one(st['llm_provider'])}",
        "llm_temperature": _one(st["llm_temperature"]), "llm_max_tokens": _one(st["llm_max_tokens"]),
        "monitor": _one(st["monitor_model"]) if True in st["monitor_enabled"] else "off",
        "judge": _one(st["judge_model"]) if True in st["judge_enabled"] else "off",
        "judge_threshold": _one(st["judge_blocks_below"]),
        "detector": "on" if True in st["detector_enabled"] else "off",
        "review_mode": _one(st["review_mode"]), "hold_seconds": _one(st["hold_seconds"]),
        "auto_release_below": _one(st["auto_release_below_risk"]),
        "model_can_end": _one(st["model_can_end_session"]),
        "policy_changes": len(st["review_policy_changes"]),
        "conditions_table": _table(["Condition (as spoken)", "Turns", "%"],
                                   [[c["condition"], c["turns"], c["percent"]]
                                    for c in cd["spoken_turns_by_condition"]]),
        "condition_changes": ", ".join(f"{n} by {k}" for k, n in cd["condition_changes"].items()) or "none",
        "below_requested": cd["turns_spoken_below_requested_level"],
        "n_candidates": rv["candidates_generated"], "n_spoken": rv["spoken_turns"],
        "per_turn": rv["candidates_per_spoken_turn"],
        "dispositions_table": _table(["Disposition", "n", "% of generated"],
                                     [[d["disposition"], d["n"], d["percent"]] for d in rv["dispositions"]]),
        "decided_by": ", ".join(f"{k} {n}" for k, n in rv["decided_by"].items()) or "n/a",
        "n_held": rv["candidates_held_for_operator"], "held_pct": rv["held_percent"],
        "hold_reasons_table": _table(["Why the reply waited for the operator", "n"],
                                     [[h["reason"], h["n"]] for h in rv["hold_reasons"]]),
        "decision_time": _ms(rv["operator_decision_ms"]), "review_wait": _ms(rv["review_wait_per_spoken_turn_ms"]),
        "stop_used": rv["stop_speech_used"], "interrupted": rv["speech_interrupted_turns"],
        "judge_coverage": mc["judge_coverage_percent"], "n_judged": mc["overall"]["n_judged"],
        "fidelity_mean": fid.get("mean", "n/a"), "fidelity_sd": fid.get("sd", "n/a"),
        "fidelity_median": fid.get("median", "n/a"),
        "category_match": mc["overall"]["category_match_percent"],
        "below_threshold": mc["overall"]["below_threshold"], "refusals": mc["overall"]["refusals"],
        "judged_neutral": mc["overall"]["judged_neutral"],
        "manipulation_table": _table(
            ["Condition", "n judged", "Fidelity mean (SD)", "Category match %", "Est. intensity", "Below threshold"],
            [[g["condition"], g["n_judged"], f"{_n(g['fidelity'].get('mean'))} ({_n(g['fidelity'].get('sd'))})",
              g["category_match_percent"], g["intensity_estimate"].get("mean"), g["below_threshold"]]
             for g in mc["by_condition"]]),
        "detector_summary": (f"mean P(faithful) {mc['detector']['p_faithful']['mean']}, "
                             f"{mc['detector']['flagged_softened']} of {mc['detector']['n']} flagged as softened"
                             if mc["detector"]["n"] else "not used"),
        "risk_spoken": ", ".join(f"{k} {n}" for k, n in rk["risk_rating_spoken"].items()) or "n/a",
        "monitor_scored": rk["monitor_scored"], "clear_risk_generated": rk["monitor_clear_risk_generated"],
        "clear_risk_spoken": rk["monitor_clear_risk_spoken"], "monitor_errors": rk["monitor_errors"],
        "clear_risk_dims": ", ".join(f"{k} {n}" for k, n in rk["monitor_clear_risk_by_dimension"].items() if n)
                           or "none",
        "distress_cues": rk["participant_distress_cues"], "distress_sessions": rk["sessions_with_distress_cue"],
        "end_reasons": ", ".join(f"{k} {n}" for k, n in rk["session_end_reasons"].items()) or "n/a",
        "latency_generation": _ms(lt["generation_ms"]), "latency_speech": _ms(lt["speech_ms"]),
        "latency_total": _ms(lt["turn_total_ms"]),
        "sessions_table": _table(["Session", "Participant", "Start", "Min", "Initial condition", "Spoken",
                                  "Generated", "Ended by"],
                                 [[s["session_id"], s["participant_id"], s["start_time"][:16], s["duration_min"],
                                   s["initial_condition"], s["spoken_turns"], s["candidates"], s["end_reason"]]
                                  for s in r["sessions"]]),
        "methods_paragraph": r["methods_paragraph"],
    }
    text = Path(template_path or TEMPLATE_PATH).read_text(encoding="utf-8")
    return Template(text).safe_substitute({k: "n/a" if v is None else v for k, v in values.items()})
