import os

# 1. 给 JAX 塞纸条！JAX 对这个环境变量是 100% 绝对响应的！
os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = "musa:/workspace/openxla/bazel-bin/pjrt_plugin/libmusa_pjrt_plugin.so"

import jax
import jax.numpy as jnp

print("=====================================")
print("开始呼叫 MUSA 大管家...")
print("=====================================")

# 2. 让 JAX 盘点家底！
devices = jax.devices()
print("找到的物理设备：", devices)

if any("musa" in str(d).lower() for d in devices):
    print("\n✅ 设备发现成功！你的 PJRT 插件完美拉起了 MUSA 驱动！")
    print("开始测试 JIT 编译执行...")
    
    # 3. 跑一个简单的矩阵加法，这会触发 HLO 编译 -> MUSA 执行的完整链路
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    
    try:
        result = x + y
        print("\n🎉🎉🎉 恭喜通关！MUSA 计算结果:\n", result)
    except Exception as e:
        print("\n❌ 运行时崩溃，报错是:\n", e)
else:
    print("\n⚠️ 没找到 MUSA 设备，可能是驱动底层名字没注册对。")
