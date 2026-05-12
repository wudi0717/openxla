import os
import ctypes
import sys

SO_PATH = "/workspace/openxla/bazel-bin/pjrt_plugin/libmusa_pjrt_plugin_zy.so"

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "1"

# TF 2.15 的 NPD / PJRT 通道
os.environ["NEXT_PLUGGABLE_DEVICE_USE_C_API"] = "true"
os.environ["TF_PLUGGABLE_DEVICE_LIBRARY_PATH"] = SO_PATH

# XLA / PJRT 后端发现通道
os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = f"MUSA:{SO_PATH}"

import tensorflow as tf

print("==== load plugin hook ====")
try:
    lib = ctypes.CDLL(SO_PATH, mode=ctypes.RTLD_GLOBAL)
    lib.ForceRegisterMusa.restype = None
    lib.ForceRegisterMusa()
    print("ForceRegisterMusa ok")
except Exception as e:
    print("ForceRegisterMusa failed:", e)
    sys.exit(1)

print("all devices:", tf.config.list_physical_devices())
print("musa devices:", tf.config.list_physical_devices("MUSA"))

musa_devices = tf.config.list_physical_devices("MUSA")
if not musa_devices:
    raise RuntimeError("No MUSA device found by TensorFlow 2.15")

@tf.function(jit_compile=True)
def add_one(x):
    return x + 1.0

with tf.device("/device:MUSA:0"):
    x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    y = add_one(x)

print("result:", y.numpy())