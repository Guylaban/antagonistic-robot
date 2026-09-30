# Antagonistic Robot — notes for Claude

Voice conversation system for an HRI study: laptop mic → Silero VAD → faster-whisper → LLM (Grok via OpenAI-compatible API) → text over TCP to `nao_speaker_server.py` on the NAO, which speaks it with NAO's built-in TTS. See README.md for setup and the troubleshooting table.

## Status (2026-09-30)

The NAO connection path (`deploy_nao.py`, `nao.local` resolution, the hardened `nao_speaker_server.py`) was tested only against a fake robot. **The first run on the real NAO has not happened yet.** If something fails there, it is most likely on the robot side; start from `python deploy_nao.py --log`.

## Running it

```
ping -4 nao.local          # robot on + cabled?
python deploy_nao.py       # upload + start speaker server on the robot (after every robot boot)
python main.py             # web UI on http://localhost:8000
```

`deploy_nao.py` does not need API keys; `main.py` needs `GROK_API_KEY` in `.env`.

## The lab robot

- NAO V6, NAOqi 2.8.5.10. The robot's own Python is 2.7; `nao_speaker_server.py` must stay Python 2.7 compatible. The laptop side is Python 3.10+.
- Connected by Ethernet cable from the robot's head to the laptop's USB Ethernet adapter. There is no DHCP, so both sides use link-local `169.254.x.x` addresses, and **the robot's address changes between sessions**. Never hardcode it: `nao.ip` is `nao.local` and is resolved on every connection (IPv4 only; the speaker server does not listen on IPv6).
- Don't scan `169.254.0.0/16` to find it (65k hosts). `ping -4 nao.local` answers immediately; pressing the chest button once makes the robot say its IP.
- SSH: user `nao`, password from `nao.password` in `config.yaml` (default `nao`).
- Ports: 22 SSH, 9559 NAOqi, 9600 our speaker server.

## Known traps

- **NAOqi port opens before the broker is ready.** Right after boot, `ALProxy(...)` can fail with `ALBroker::createBroker Cannot connect` on a healthy robot. It is a retry, not a dead robot. The speaker server retries 10 times, 2 s apart.
- **Autonomous Life fights manual motion.** After a fall NAO sits in `safeguard` and `goToPosture` won't take. The sequence that works: `ALAutonomousLife.setState("disabled")` → `ALMotion.wakeUp()` → `goToPosture(...)`. The speaker server does this at startup.
- **`naoqi` import over SSH.** `deploy_nao.py` starts the server through `bash -lc` and first checks `import naoqi`; if that fails it retries with `PYTHONPATH=/opt/aldebaran/lib/python2.7/site-packages`, then any `--pythonpath` given. If all fail, find the module on the robot with `find / -name naoqi.py 2>/dev/null`.
- **Check Point VPN on the lab laptop** hooks outbound connections on port 80, so port-80 scans report hosts that don't exist. Ports 22 / 9559 / 9600 are not affected.
- A robot reboot kills the speaker server; rerun `deploy_nao.py`.

## Lab laptop specifics

On the lab laptop the working copy is `Desktop\job\naoqi\NAO_LLM` with a ready `venv`. Its `.exe` console shims (`pip.exe`, `uvicorn.exe`) are broken because the folder was moved; call `venv\Scripts\python.exe -m pip ...` instead. The venv itself works. Don't rebuild it on the phone hotspot (torch is ~2 GB).

`webui/src/App.js` there may have uncommitted UI edits by the repo owner; don't discard them. Without `webui/build/`, the server serves the fallback UI in `antagonist_robot/ui/static/index.html`, which is fully functional.

## LLM model

`llm.model` must be a **non-reasoning** model: every second of generation is silence in front of the participant. x.ai retires and silently re-routes model names (`grok-4-fast` became a slow reasoning model in 2026), so if replies start taking several seconds, list models with `GET https://api.x.ai/v1/models` and time a few. `main.py` makes one test call at startup and exits if the key or model is bad.

## Conventions

- Safety boundaries in `avct_manager.py` are always included in every prompt; don't add a code path that skips them.
- Every turn is logged to SQLite (`data/`), which holds participant data and is gitignored. Never commit `data/`, `logs/`, or `.env`.
- Type hints and docstrings on public methods; dataclasses for data passed between pipeline stages.
