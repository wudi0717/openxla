import os
import ctypes
import sys

so_path = "/workspace/openxla/bazel-bin/pjrt_plugin/libmusa_pjrt_plugin_zy.so"
os.environ["TF_PLUGGABLE_DEVICE_LIBRARY_PATH"] = so_path
os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = f"MUSA:{so_path}"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()  # 强行进入 TF1 静态图模式！

print(">>> [PYTHON] 强行调起 C++ 注册钩子...")
try:
    lib = ctypes.CDLL(so_path)
    lib.ForceRegisterMusa()
except Exception as e:
    print(f"!!! 钩子失败: {e}")
    sys.exit(1)

devices = tf.config.list_physical_devices('MUSA')
print(f">>> 检测到设备: {devices}")

# ==========================================
# 构建一个极简的 TF1 计算图 (没有 resource 变量)
# ==========================================
with tf.Graph().as_default() as graph:
    with tf.device('/device:MUSA:0'):
        # 纯粹的 Placeholder，预期正常的 float 数据
        a = tf.placeholder(tf.float32, shape=[2, 2], name="input_a")
        b = tf.placeholder(tf.float32, shape=[2, 2], name="input_b")
        c = tf.matmul(a, b)
        output = tf.add(c, tf.constant([[1.0, 1.0], [1.0, 1.0]]))

# ==========================================
# 开启 Session，连续运行 10 次！
# ==========================================
config = tf.ConfigProto()
config.allow_soft_placement = True

print("\n>>> 开始 Session 循环测试...")
with tf.Session(graph=graph, config=config) as sess:
    for i in range(10):
        # 喂入正常的数据
        feed_dict = {
            a: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            b: np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        }
        res = sess.run(output, feed_dict=feed_dict)
        print(f"--- 第 {i+1} 次运行成功! ---")
        
print(">>> 测试完美结束！没有任何卡死！")
