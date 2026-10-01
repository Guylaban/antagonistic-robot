"""Rehearsal scenarios: a scripted participant, a scripted operator, and captions for the recorded video.

A scenario is a YAML file (see scenarios/demo.yaml):

    participant:
      voice: am_michael              # Kokoro voice for the participant's lines in the video (optional)
      lines: [...]                   # what the participant says, in order
    robot_voice:                     # optional, per backend (Kokoro voice for Reachy Mini, Furhat voice name)
      reachy_mini: af_heart
    steps:                           # what the operator does, in order
      - matrix: {polar: 2, category: D, subtype: 2, modifiers: {M4: true}}
        moment: setup                # snapshot the console here (caption key)
      - start: {participant_id: DEMO}
      - reply: {on_arrival: t1_generated, on_review: t1_held, action: send}
      - reply: {hold: true, on_review: t2_review, action: temper,
                replacement: {on_arrival: t2_tempered, action: send}}
      - matrix: {category: C}
        apply: true
        moment: apply_sarcastic
      - end: {moment: end}
    captions:                        # keyed by moment; numbered in the order they appear
      setup: {title: ..., body: ..., highlight: [matrix], pause: 4.5, essential: true}

reply steps wait for the next reply in the review panel. hold presses Hold as soon as it appears.
If the reply is held (by the operator, the monitor, the fidelity judge, or the risk threshold), the
operator reads it for review_s seconds and then performs action (send, temper, intensify,
regenerate); a reply the timer releases needs no action. temper/intensify/regenerate produce a
replacement reply, handled by the nested replacement step (default: send).

A caption can switch on the reason a reply was held, e.g. for the fidelity judge's softening flag:

    t5_held:
      title: Back to confrontational
      body: ...
      if_reason:
        softening: {title: Too soft, body: "The judge rates it {exhibits} ({fidelity}/10) ...", pause: 5.5}
"""

import re
from typing import Optional

import yaml

ACTIONS = {"send": "#sendBtn", "temper": "#temperBtn", "intensify": "#intensifyBtn", "regenerate": "#regenBtn"}
HIGHLIGHTS = {"matrix", "policy", "monitor_log", "pending", "safety", "fidelity", "psych", "latency", "data",
              "temper", "send", "apply", "discarded", "last_robot"}
STEP_KINDS = {"matrix", "start", "reply", "end", "wait"}


class ScenarioError(ValueError):
    pass


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        sc = yaml.safe_load(f)
    validate(sc)
    return sc


def validate(sc: dict) -> None:
    if not isinstance(sc, dict):
        raise ScenarioError("scenario must be a mapping")
    lines = (sc.get("participant") or {}).get("lines")
    if not lines or not all(isinstance(x, str) and x.strip() for x in lines):
        raise ScenarioError("participant.lines must be a non-empty list of strings")
    steps = sc.get("steps") or []
    if not steps:
        raise ScenarioError("steps must be a non-empty list")
    replies = 0
    for i, step in enumerate(steps):
        kinds = STEP_KINDS & set(step)
        if len(kinds) != 1:
            raise ScenarioError(f"step {i + 1} must have exactly one of {sorted(STEP_KINDS)}: {step}")
        if "reply" in step:
            replies += 1
            _validate_reply(step["reply"] or {}, f"step {i + 1}")
    if replies > len(lines):
        raise ScenarioError(f"{replies} reply steps but only {len(lines)} participant lines")
    if "start" not in {k for s in steps for k in s}:
        raise ScenarioError("a scenario needs a start step")
    for key, cap in (sc.get("captions") or {}).items():
        for c in [cap] + list((cap.get("if_reason") or {}).values()):
            bad = set(c.get("highlight", [])) - HIGHLIGHTS
            if bad:
                raise ScenarioError(f"caption {key}: unknown highlight {sorted(bad)}; use {sorted(HIGHLIGHTS)}")


def _validate_reply(r: dict, where: str) -> None:
    action = r.get("action", "send")
    if action not in ACTIONS:
        raise ScenarioError(f"{where}: action must be one of {sorted(ACTIONS)}, got {action!r}")
    if action != "send":
        _validate_reply(r.get("replacement") or {}, where + " (replacement)")


def caption_for(captions: dict, key: str, reasons: Optional[list] = None) -> Optional[dict]:
    """The caption for a moment, switched on the reasons the reply was held (if_reason), placeholders filled."""
    cap = captions.get(key)
    if cap is None:
        return None
    text = " ".join(reasons or [])
    for needle, alt in (cap.get("if_reason") or {}).items():
        if needle.lower() in text.lower():
            cap = {**{k: v for k, v in cap.items() if k != "if_reason"}, **alt}
            break
    values = {"fidelity": "?", "exhibits": "off-condition"}
    m = re.search(r"fidelity (\d+)/10", text)
    if m:
        values["fidelity"] = m.group(1)
    m = re.search(r"exhibits (\w+)", text)
    if m:
        values["exhibits"] = m.group(1).lower()
    out = {"title": cap.get("title", key), "body": cap.get("body", ""), "highlight": list(cap.get("highlight", [])),
           "pause": float(cap.get("pause", 0)), "essential": bool(cap.get("essential", False))}
    for k in ("title", "body"):
        out[k] = out[k].format(**values) if "{" in out[k] else out[k]
    return out


def number_captions(keys_in_order: list) -> dict:
    """Step numbers for captions in display order (the intro is not numbered)."""
    n, out = 0, {}
    for key in keys_in_order:
        if key == "intro" or key in out:
            continue
        n += 1
        out[key] = n
    return out


def visible_box(rect, viewport=(1600, 900), min_visible=0.6):
    """Clip a page rectangle (x, y, w, h) to the viewport; None if mostly outside it."""
    x, y, w, h = rect
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(viewport[0], x + w), min(viewport[1], y + h)
    if w <= 0 or h <= 0 or (x1 - x0) * (y1 - y0) < min_visible * w * h:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def defer_to_silence(t: float, speech: list, margin: float = 0.15, after: float = 0.3) -> float:
    """First time >= t at which nobody is speaking (speech = [(start, end), ...]), so a pause cuts no word."""
    for _ in range(500):
        busy = [e for s, e in speech if s - margin <= t <= e + margin]
        if not busy:
            return t
        t = max(busy) + after
    return t
