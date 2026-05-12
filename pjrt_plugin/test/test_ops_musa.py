import os
import ctypes
import sys

so_path = "/workspace/openxla/bazel-bin/pjrt_plugin/libmusa_pjrt_plugin_zy.so"

# 1. 喂给 TF：让它发现 MUSA 物理设备
os.environ["TF_PLUGGABLE_DEVICE_LIBRARY_PATH"] = so_path

# 2. 喂给 XLA：强行外挂纯 C 的 PJRT 算子执行引擎！
os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = f"MUSA:{so_path}"

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 💡 注意：为了能看到咱们 C++ 里的 printf 调试信息，建议运行时别把日志等级设太高。
# 先注释掉这行，或者保持原样，咱们主要看 stdout 的 printf 即可。
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 

import tensorflow as tf

# =========================================================================
# 🔥 核心补丁：用 Python 手动按下 C++ 的“注册开关”！
# =========================================================================
print("\n====================================================")
print("[PYTHON] 正在强行调起 C++ 注册钩子...")
try:
    # 加载我们刚刚编译出的带后门的动态库
    lib = ctypes.CDLL(so_path)
    
    # 执行我们在 pjrt_api.cc 写的暴露给 Python 的函数
    lib.ForceRegisterMusa()
    print("[PYTHON] C++ 注册钩子调用完毕！")
except Exception as e:
    print(f"[PYTHON] 💥 糟糕，强制注册钩子调起失败：{e}")
    sys.exit(1)
print("====================================================\n")

tf.debugging.set_log_device_placement(True)

print("\n=====================================")
print("开始测试 MUSA 算力链路...")
print("=====================================")

# 寻找 MUSA 设备
devices = tf.config.list_physical_devices('MUSA')
if not devices:
    print("❌ 未找到设备！")
    exit(1)

print(f"✅ 成功检测到设备: {devices}")

try:
    # 强制在 MUSA 上进行计算
    with tf.device('/device:MUSA:0'):
        a = tf.constant([[1.0, 2.0], 
                         [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], 
                         [7.0, 8.0]])
        c = tf.matmul(a, b)
        d = tf.add(c, tf.constant([[1.0, 1.0], [1.0, 1.0]]))

    print("\n✅ 计算执行成功！结果如下：")
    print("A * B + 1 = \n", d.numpy())

except Exception as e:
    print(f"\n❌ 发生运行时错误：\n{e}")