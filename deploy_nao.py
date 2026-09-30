"""Deploy and start nao_speaker_server.py on the NAO robot over SSH.

Reads nao.ip / nao.port / nao.password from config.yaml, uploads the
server script, restarts it in the background, and waits until it is
accepting connections.

Usage:
    python deploy_nao.py            # upload + (re)start + verify
    python deploy_nao.py --log      # show the server's log on the robot
    python deploy_nao.py --stop     # stop the server
"""

import argparse
import socket
import sys
import time
from pathlib import Path

import paramiko

from antagonist_robot.config.settings import NAOConfig, _build_dataclass
from antagonist_robot.nao.host import resolve_ipv4

import yaml

SSH_USER = "nao"
REMOTE_SCRIPT = "/home/nao/nao_speaker_server.py"
REMOTE_LOG = "/home/nao/nao_speaker_server.log"
LOCAL_SCRIPT = Path(__file__).parent / "nao_speaker_server.py"


def load_nao_config(path: str) -> NAOConfig:
    """Load only the nao section, so no API keys are needed to deploy."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _build_dataclass(NAOConfig, raw.get("nao", {}))


def run(ssh: paramiko.SSHClient, command: str) -> str:
    """Run a command on the robot and return its combined output."""
    _, stdout, stderr = ssh.exec_command(command)
    return (stdout.read() + stderr.read()).decode("utf-8", "replace").strip()


def wait_for_port(ip: str, port: int, timeout: float) -> bool:
    """Poll until the speaker server accepts TCP connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((ip, port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False


def main():
    parser = argparse.ArgumentParser(description="Deploy nao_speaker_server.py to the NAO")
    parser.add_argument("--config", default="config.yaml")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--log", action="store_true", help="print the server log")
    group.add_argument("--stop", action="store_true", help="stop the server")
    args = parser.parse_args()

    nao = load_nao_config(args.config)
    try:
        ip = resolve_ipv4(nao.ip, 22)
    except OSError as e:
        sys.exit(f"Cannot resolve {nao.ip}: {e}\n"
                 f"Is the robot on and cabled? Try: ping -4 {nao.ip}")
    print(f"Robot: {nao.ip} -> {ip}")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, username=SSH_USER, password=nao.password, timeout=10)
    try:
        if args.log:
            print(run(ssh, f"tail -n 50 {REMOTE_LOG}"))
            return
        # Kill any previous instance; [n] stops pkill matching its own shell.
        run(ssh, "pkill -f '[n]ao_speaker_server.py'")
        if args.stop:
            print("Speaker server stopped.")
            return

        sftp = ssh.open_sftp()
        sftp.put(str(LOCAL_SCRIPT), REMOTE_SCRIPT)
        sftp.close()
        print(f"Uploaded {LOCAL_SCRIPT.name}")

        # Login shell so PYTHONPATH includes the robot's naoqi module.
        run(ssh, f"bash -lc 'nohup python {REMOTE_SCRIPT} > {REMOTE_LOG} 2>&1 &'")
        print("Starting (standing up and connecting to NAOqi can take ~20s)...")

        if wait_for_port(ip, nao.port, timeout=45):
            print(f"Speaker server is up on {ip}:{nao.port}. Now run: python main.py")
        else:
            print("Speaker server did not come up. Robot log:\n")
            print(run(ssh, f"tail -n 30 {REMOTE_LOG}"))
            sys.exit(1)
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
