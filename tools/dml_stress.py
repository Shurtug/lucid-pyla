"""Sustained DirectML inference test, run as its own process.

The previous failure wasn't a Python exception - it was a native driver crash
that killed the interpreter outright (exit 139, no traceback), and only ever
under concurrent GPU load from the emulator. So this has to run with BlueStacks
going, and it has to be isolated: if the driver dies it takes this process with
it and nothing else.

Prints progress unbuffered so the last line before a crash tells us how far it
got.
"""
import os
import sys
import time

os.environ["DML_DISABLE_METACOMMANDS"] = "1"

import numpy as np
import onnxruntime as ort

MODEL = "models/mainInGameModel.onnx"
DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0


def log(msg):
    print(msg, flush=True)


providers = ort.get_available_providers()
log("available providers: %s" % providers)
if "DmlExecutionProvider" not in providers:
    log("RESULT: DirectML provider not available at all")
    sys.exit(2)

so = ort.SessionOptions()
# graph fusion was the suspected trigger last time, so keep it disabled -
# matching exactly what detect.py would do if GPU were re-enabled
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
so.intra_op_num_threads = 4
so.inter_op_num_threads = 4

t0 = time.time()
sess = ort.InferenceSession(
    MODEL, sess_options=so,
    providers=[("DmlExecutionProvider", {"device_id": 0, "disable_metacommands": True})])
log("session created in %.1fs on %s" % (time.time() - t0, sess.get_providers()))

name = sess.get_inputs()[0].name
shape = sess.get_inputs()[0].shape
h = shape[2] if isinstance(shape[2], int) else 640
w = shape[3] if isinstance(shape[3], int) else 640
buf = np.random.rand(1, 3, h, w).astype(np.float32)

log("running sustained inference for %.0fs at %dx%d ..." % (DURATION, w, h))
start = time.time()
n = 0
times = []
while time.time() - start < DURATION:
    t = time.time()
    sess.run(None, {name: buf})
    times.append((time.time() - t) * 1000.0)
    n += 1
    if n % 50 == 0:
        el = time.time() - start
        log("  %5.1fs  %4d inferences  last50 avg %.1fms" % (el, n, np.mean(times[-50:])))

log("")
log("RESULT: SURVIVED %.0fs, %d inferences" % (DURATION, n))
log("  mean %.1fms  median %.1fms  p95 %.1fms  max %.1fms"
    % (np.mean(times), np.median(times), np.percentile(times, 95), max(times)))
