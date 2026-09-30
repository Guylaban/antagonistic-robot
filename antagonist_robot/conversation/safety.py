"""Deterministic safety scanning for robot responses and participant speech.

SafetyChecker runs on every candidate response before it can reach the
robot, and on every participant utterance. It uses fixed regular
expressions only (no API call, microseconds per check), so its behavior is
identical across runs and models and can be audited from the source.

Risk levels, in increasing order: Green < Yellow < Orange < Red.

- Content risk: what the generated text contains.
- Configuration risk: what the operator asked for (polar level, category).
- The turn's risk is the higher of the two.

The checker never rewrites text. It rates it, and the operator gate
(operator.py) decides whether a rated response may be released without
an explicit operator action.
"""

import re
from dataclasses import dataclass, field
from typing import List

RISK_LEVELS = ["Green", "Yellow", "Orange", "Red"]


def risk_index(level: str) -> int:
    """Return the position of a risk level (unknown levels count as Red)."""
    try:
        return RISK_LEVELS.index(level)
    except ValueError:
        return len(RISK_LEVELS) - 1


def max_risk(*levels: str) -> str:
    """Return the highest of the given risk levels."""
    return max(levels, key=risk_index)


# Hard violations: the robot must never say these. Red.
_RED_PATTERNS = {
    "self_harm_encouragement": [
        r"\bkill (?:yo)?urself\b", r"\bkys\b", r"\bend (?:your|ur) (?:own )?life\b",
        r"\byou should (?:just )?die\b", r"\bgo die\b", r"\bhurt yourself\b",
        r"\b(?:cut|harm) yourself\b", r"\bbetter off dead\b",
        r"\bno one would (?:miss|care if) you\b",
    ],
    "explicit_threat": [
        r"\bi(?:'ll| will|'m going to| am going to) (?:hurt|kill|beat|hit|punch|stab|shoot|find) you\b",
        r"\bi(?:'ll| will) make you (?:bleed|suffer)\b",
        r"\byou(?:'re| are) (?:going to|gonna) (?:die|get hurt)\b",
    ],
    "slur": [
        r"\bn[i1!]gg(?:er|a|ah)s?\b", r"\bf[a@]gg?(?:ot)?s?\b", r"\bretard(?:ed|s)?\b",
        r"\btr[a@]nn(?:y|ies)\b", r"\bk[i1]kes?\b", r"\bsp[i1]cs?\b", r"\bch[i1]nks?\b",
        r"\bwetbacks?\b", r"\bdykes?\b",
    ],
    "sexual_content": [
        r"\bsex(?:ual(?:ly)?)?\b.*\byou\b", r"\bnaked\b", r"\bnudes?\b",
    ],
}

# Strong personal attacks and coercive framing. Orange.
_ORANGE_PATTERNS = {
    "strong_insult": [
        r"\bidiot(?:ic|s)?\b", r"\bstupid(?:ity)?\b", r"\bworthless\b", r"\bpathetic\b",
        r"\bmoron(?:ic|s)?\b", r"\bloser\b", r"\buseless\b", r"\bgarbage\b",
        r"\bdisgrace(?:ful)?\b", r"\bdumb\b", r"\bimbecile\b", r"\bincompeten(?:t|ce)\b",
        r"\bshut up\b",
    ],
    "profanity": [
        r"\bf+u+c+k\w*", r"\bs+h+i+t+\w*", r"\bbullshi\w*", r"\bass(?:hole|hat)s?\b",
        r"\bbitch\w*", r"\bbastards?\b", r"\bdamn(?:ed|it)?\b", r"\bcrap\b", r"\bpiss\w*",
    ],
    "coercive_warning": [
        r"\byou(?:'ll| will) regret\b", r"\bor else\b", r"\byou(?:'ll| will) be sorry\b",
        r"\bthere will be consequences\b", r"\bwatch yourself\b",
    ],
}

# Mild negative evaluation. Yellow.
_YELLOW_PATTERNS = {
    "negative_evaluation": [
        r"\bwrong\b", r"\bdisappoint(?:ed|ing)?\b", r"\blazy\b", r"\bweak\b",
        r"\bexcuses?\b", r"\bmediocre\b", r"\bsloppy\b", r"\bnot good enough\b",
        r"\bnaive\b", r"\bsimplistic\b", r"\bridiculous\b", r"\bclueless\b",
        r"\bshould know better\b", r"\bembarrass(?:ing)?\b",
    ],
}

# Participant speech that should stop automatic release and alert the operator.
_DISTRESS_PATTERNS = [
    r"\b(?:kill|hurt|harm) myself\b", r"\bsuicid", r"\bwant to die\b", r"\bend (?:it|my life)\b",
    r"\bi can'?t (?:take|do) (?:this|it) anymore\b", r"\bplease stop\b", r"\bstop (?:it|this)\b",
    r"\bi want to stop\b", r"\bi(?:'m| am) (?:scared|afraid|crying)\b", r"\bleave me alone\b",
    r"\bi (?:quit|withdraw)\b", r"\bpanic\b",
]


def _compile(groups: dict) -> list:
    return [(name, re.compile(p, re.IGNORECASE)) for name, pats in groups.items() for p in pats]


_COMPILED = [
    ("Red", _compile(_RED_PATTERNS)),
    ("Orange", _compile(_ORANGE_PATTERNS)),
    ("Yellow", _compile(_YELLOW_PATTERNS)),
]
_DISTRESS = [re.compile(p, re.IGNORECASE) for p in _DISTRESS_PATTERNS]


@dataclass
class SafetyFlag:
    """One matched pattern."""
    level: str
    category: str
    match: str


@dataclass
class SafetyResult:
    """Rating of one text."""
    level: str
    flags: List[SafetyFlag] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "flags": [{"level": f.level, "category": f.category, "match": f.match} for f in self.flags],
        }


class SafetyChecker:
    """Deterministic lexical scanner for responses and participant speech."""

    def check_response(self, text: str) -> SafetyResult:
        """Rate a candidate robot response Green, Yellow, Orange, or Red."""
        flags: List[SafetyFlag] = []
        for level, patterns in _COMPILED:
            for category, rx in patterns:
                m = rx.search(text or "")
                if m:
                    flags.append(SafetyFlag(level, category, m.group(0)))
        level = max_risk("Green", *[f.level for f in flags]) if flags else "Green"
        return SafetyResult(level=level, flags=flags)

    def check_participant(self, text: str) -> List[str]:
        """Return distress cues found in a participant utterance (empty if none)."""
        return [m.group(0) for rx in _DISTRESS for m in [rx.search(text or "")] if m]


def config_risk(polar_level: int, category: str) -> str:
    """Risk implied by the requested configuration alone.

    Supportive, neutral, and mild (+1) settings are Green. At +2, the
    mid-range categories (B-E) are Yellow and Aggressive (F) is Orange.
    At +3, B-E are Orange and F is Red. Extreme (G) is Red at any
    positive level.
    """
    if polar_level <= 0:
        return "Green"
    if category == "G":
        return "Red"
    if polar_level == 1:
        return "Green"
    if polar_level == 2:
        return "Orange" if category == "F" else "Yellow"
    return "Red" if category == "F" else "Orange"
