"""Record what the computer's speakers play (WASAPI loopback, Windows; pip install pyaudiowpatch).

Used for the virtual Furhat, whose voice is played by the Furhat SDK and never reaches CRAB.
compose.py keeps this track only while the robot was speaking (from the session database), so other
sounds on the computer are dropped. Gaps (the loopback delivers nothing while the speakers are
silent) are filled with zeros on the wall clock.

Usage: python loopback.py out.wav stamp.json STOP_FILE
"""
import json
import os
import sys
import time
import wave


def main():
    import pyaudiowpatch as pyaudio
    out, stamp, stop = sys.argv[1:4]
    pa = pyaudio.PyAudio()
    dev = pa.get_default_wasapi_loopback()
    sr, ch = int(dev["defaultSampleRate"]), dev["maxInputChannels"]
    w = wave.open(out, "wb")
    w.setnchannels(ch)
    w.setsampwidth(2)
    w.setframerate(sr)
    state = {"t": time.time(), "n": 0}
    json.dump({"audio_start": state["t"], "sr": sr, "channels": ch}, open(stamp, "w"))

    def cb(data, frames, info, status):
        start = time.time() - frames / sr
        gap = int((start - state["t"]) * sr) - state["n"]
        if gap > sr // 20:
            w.writeframes(b"\0" * (gap * ch * 2))
            state["n"] += gap
        w.writeframes(data)
        state["n"] += frames
        return (None, pyaudio.paContinue)

    s = pa.open(format=pyaudio.paInt16, channels=ch, rate=sr, input=True, input_device_index=dev["index"],
                frames_per_buffer=1024, stream_callback=cb)
    s.start_stream()
    while not os.path.exists(stop):
        time.sleep(0.1)
    s.stop_stream()
    s.close()
    pa.terminate()
    w.close()
    print("recorded", out)


if __name__ == "__main__":
    main()
