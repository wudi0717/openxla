import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# 替换为你实际的模型路径
pb_path = "/workspace/openxla/pjrt_plugin/musa_test/musa_model/graph_def.pb"

print(f">>> 正在解析模型: {pb_path}")
with tf.io.gfile.GFile(pb_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

print("\n=========================================")
print("🕵️ 图中的 Placeholder (输入) 节点检查")
print("=========================================")

resource_count = 0
for node in graph_def.node:
    if node.op == "Placeholder":
        # 解析数据类型
        dtype_enum = node.attr["dtype"].type
        try:
            dtype_name = tf.as_dtype(dtype_enum).name
        except:
            dtype_name = f"Unknown Enum ({dtype_enum})"
            
        print(f"🔸 节点名: {node.name}")
        print(f"   数据类型: {dtype_name}")
        
        if dtype_name == "resource":
            resource_count += 1

print("\n=========================================")
print(f"总计找到 {resource_count} 个 resource 类型的输入！")
if resource_count > 0:
    print("⚠️ 警告：推理图中包含了 resource 节点。这意味着模型尚未被正确冻结 (Frozen)！")
