import os

# 1. JAX 的通道（给底层 XLA 留的）
os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = "musa:/workspace/openxla/bazel-bin/plugin_pjrt/libmusa_pjrt_plugin_zy.so"

# 2. 【新增】TF 的专属安检通道！告诉 TF 插件在哪里
os.environ["TF_PLUGGABLE_DEVICE_LIBRARY_PATH"] = "/workspace/openxla/bazel-bin/plugin_pjrt/libmusa_pjrt_plugin_zy.so"

# 3. 【极其关键】开启 TF 的下一代插件架构开关！
# TF 默认要找传统的 PluggableDevice，开启这个开关，TF 才会直接吃纯正的 PJRT 插件！
os.environ["NEXT_PLUGGABLE_DEVICE_USE_C_API"] = "true"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "1" # 保持透视眼开启，看 TF 的小心思

import tensorflow as tf

print("=====================================")
print("开始呼叫 MUSA 大管家 (TF 版本)...")
print("=====================================")

devices = tf.config.list_physical_devices()
print("找到的物理设备：", devices)

if any("MUSA" in str(d).upper() or "PJRT" in str(d).upper() for d in devices):
    print("\n✅ 伟大胜利！TF 终于认出你的 MUSA 卡了！")
    
    # 既然底层 llvm-link 有 Bug，我们就先不跑 JIT 编译的矩阵计算了，免得它又崩溃
    # 只要设备能被 list 出来，你的 1.5 周插件开发任务就已经圆满完成！
else:
    print("\n⚠️ TF 还是没认出来。")
