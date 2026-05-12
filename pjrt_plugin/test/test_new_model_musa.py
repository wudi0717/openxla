import os
import sys

# =========================================================================
# 🛠 关键修复：防止 Protobuf 符号冲突
# =========================================================================
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import ctypes
import numpy as np

# 插件路径
so_path = "/workspace/openxla/bazel-bin/pjrt_plugin/libmusa_pjrt_plugin_zy.so"

# 1. 配置环境变量
os.environ["TF_PLUGGABLE_DEVICE_LIBRARY_PATH"] = so_path
os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = f"MUSA:{so_path}"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 

import tensorflow as tf

# =========================================================================
# 🔥 核心补丁：手动按下 C++ 注册开关
# =========================================================================
print("\n" + "="*50)
print("[PYTHON] 正在初始化 MUSA PJRT 插件...")
try:
    lib = ctypes.CDLL(so_path)
    lib.ForceRegisterMusa()
    print("[PYTHON] ✅ C++ 注册钩子调用完毕！")
except Exception as e:
    print(f"[PYTHON] 💥 插件加载失败：{e}")
    sys.exit(1)

tf.debugging.set_log_device_placement(True)

# =========================================================================
# 🚀 MLP 模型测试 (只用 MatMul 和 Element-wise 算子)
# =========================================================================
devices = tf.config.list_physical_devices('MUSA')
if not devices:
    print("❌ 未找到 MUSA 设备！")
    sys.exit(1)

print(f"✅ 成功检测到设备: {devices}")

try:
    with tf.device('/device:MUSA:0'):
        print("\n[STEP] 启动 MLP 3层神经网络测试...")

        # 1. 模拟输入 (BatchSize=4, Features=128)
        # 避开随机数，直接用 constant 防止溢出
        x = tf.constant(np.ones((4, 128)).astype(np.float32) * 0.5)

        # --- 第一层: 128 -> 256 ---
        w1 = tf.constant(np.random.randn(128, 256).astype(np.float32) * 0.01)
        b1 = tf.constant(np.zeros(256).astype(np.float32))
        layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

        # --- 第二层: 256 -> 128 ---
        w2 = tf.constant(np.random.randn(256, 128).astype(np.float32) * 0.01)
        b2 = tf.constant(np.zeros(128).astype(np.float32))
        layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)

        # --- 第三层 (输出层): 128 -> 10 ---
        w3 = tf.constant(np.random.randn(128, 10).astype(np.float32) * 0.01)
        b3 = tf.constant(np.zeros(10).astype(np.float32))
        logits = tf.matmul(layer2, w3) + b3

        # 最终输出
        output = tf.nn.softmax(logits)

        print("\n✅ [MUSA] MLP 模型执行成功！")
        print(f"输出 Shape: {output.shape}")
        print("预测结果（前两行）:\n", output.numpy()[:2])

except Exception as e:
    print(f"\n❌ MLP 运行时发生错误：\n{e}")

print("\n" + "="*50)
