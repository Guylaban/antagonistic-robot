"""Run the real nao_speaker_server.py on this computer with a fake NAOqi.

For dry runs, operator training, and tests without a robot. The speaker
server code is the same file that deploy_nao.py uploads to the robot;
only the naoqi module is replaced (tools/fake_naoqi), so robot speech is
printed instead of spoken.

Usage:
    python tools/mock_nao.py [--port 9600]
    python main.py --nao-ip 127.0.0.1          # in a second terminal
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fake_naoqi"))
sys.argv = [os.path.join(HERE, "..", "nao_speaker_server.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
