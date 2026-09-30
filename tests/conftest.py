"""Shared fixtures: fake pipeline components and a mock robot on a free port.

The tests never call a real LLM, microphone, or robot. The mock robot is
the real nao_speaker_server.py running with tools/fake_naoqi.
"""

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from antagonist_robot.config.settings import AvctConfig, OperatorConfig  # noqa: E402
from antagonist_robot.conversation.avct_manager import AvctManager  # noqa: E402
from antagonist_robot.conversation.manager import ConversationManager  # noqa: E402
from antagonist_robot.conversation.operator import OperatorGate  # noqa: E402
from antagonist_robot.logging.session_logger import SessionLogger  # noqa: E402
from antagonist_robot.nao.base import NAOAdapter  # noqa: E402
from antagonist_robot.pipeline.audio_output import NAOAudioOutput  # noqa: E402
from antagonist_robot.robots.nao import NaoBackend  # noqa: E402
from antagonist_robot.pipeline.scripted_input import ScriptedParticipant  # noqa: E402
from antagonist_robot.pipeline.types import LLMResult  # noqa: E402

_POLAR = re.compile(r"Operate at polar level (-?\d)")


class FakeLLM:
    """Returns a scripted response per call; the text reports the polar level it was asked for."""

    def __init__(self, texts=None):
        self.texts = list(texts or [])
        self.calls = []

    def generate(self, system_prompt, messages):
        polar = int(_POLAR.search(system_prompt).group(1))
        self.calls.append({"system_prompt": system_prompt, "messages": messages, "polar": polar})
        text = self.texts.pop(0) if self.texts else f"Reply at polar {polar}."
        return LLMResult(text=text, model="fake-llm", total_tokens=10, generation_time_seconds=0.0,
                         reasoning=f"reasoning for polar {polar}")


class NullNAO(NAOAdapter):
    def connect(self): pass
    def disconnect(self): pass
    def on_response(self, text, hostility_level): pass
    def on_listening(self): pass
    def on_idle(self): pass
    def is_connected(self): return True


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def mock_robot():
    """Start tools/mock_nao.py on a free port; yield (ip, port)."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "tools" / "mock_nao.py"), "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(ROOT),
    )
    out = NAOAudioOutput("127.0.0.1", port)
    for _ in range(100):
        if out.ping():
            break
        time.sleep(0.05)
    else:
        proc.kill()
        raise RuntimeError("mock robot did not start")
    yield "127.0.0.1", port
    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture
def make_manager(tmp_path, mock_robot):
    """Factory for a ConversationManager wired to fakes and the mock robot."""

    def _make(utterances, llm_texts=None, review_mode="timed", hold_seconds=0.2, block_at="Orange", monitor=None,
              fidelity=None, robot=None):
        ip, port = mock_robot
        logger = SessionLogger(str(tmp_path / "test.db"), str(tmp_path / "audio"))
        participant = ScriptedParticipant(utterances, delay_s=0.0)
        llm = FakeLLM(llm_texts)
        gate = OperatorGate(OperatorConfig(review_mode=review_mode, hold_seconds=hold_seconds,
                                           block_auto_send_at=block_at))
        manager = ConversationManager(
            audio_capture=participant, asr=participant, llm=llm,
            robot=robot or NaoBackend(ip, port), avct_manager=AvctManager(AvctConfig()),
            session_logger=logger, gate=gate, monitor=monitor, fidelity=fidelity,
        )
        events = []
        manager.on_event = events.append
        return manager, llm, logger, events

    return _make


def wait_for(predicate, timeout=5.0):
    """Poll until predicate() is truthy."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("condition not met in time")
