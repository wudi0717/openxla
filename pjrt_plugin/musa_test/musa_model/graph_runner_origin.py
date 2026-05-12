import tensorflow as tf
import numpy as np
import os
import sys
import time
import argparse
from tensorflow.core.framework import graph_pb2
import logging
from typing import Dict, List, Any, Optional

# ==========================================
# 配置区域
# ==========================================
model_path = "/workspace/openxla/pjrt_plugin/musa_test/musa_model/graph_def.pb"
output_node_name = "predicts"  # 输出节点名称
musa_plugin_path = "/workspace/openxla/bazel-bin/pjrt_plugin/libmusa_pjrt_plugin_zy.so"

# 禁用 V2 行为，确保 TF1 图能正常运行
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


# ==========================================
# 1. 加载 MUSA 插件，配置config
# ==========================================
def load_musa_plugin():
    if os.path.exists(musa_plugin_path):
        try:
            tf.load_op_library(musa_plugin_path)
            print(f">>>> [MUSA] Plugin loaded successfully from: {musa_plugin_path}")
        except Exception as e:
            print(f"!!!! [MUSA] Failed to load plugin: {e}")
    else:
        print(f"!!!! [MUSA] Plugin not found at {musa_plugin_path}, assuming built-in.")

def create_session_config(
    device_type: str = "cpu",
    xla: bool = False,
    log_device_placement: bool = False,
    logger: Optional[logging.Logger] = None,
) -> tf.ConfigProto:
    """Create a Session config with device and XLA settings."""
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = log_device_placement

    device_type_upper = (device_type or "cpu").upper()

    if device_type_upper == "CUDA":
        config.gpu_options.allow_growth = True
        # XLA 仅在 CUDA 设备下有效
        if xla:
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            if logger is not None:
                logger.info("Enabled XLA JIT compilation for CUDA")

    elif device_type_upper == "MUSA":
        rewrite_options = config.graph_options.rewrite_options
        rewrite_options.custom_optimizers.add().name = "musa_graph_optimizer"
        if logger is not None:
            logger.info("Enabled custom optimizer: musa_graph_optimizer")

    return config


# ==========================================
# 2. 辅助函数：推断 Placeholder 形状
# ==========================================
def infer_placeholder_shape_from_usage(graph_def, placeholder_name):
    """
    通过分析图中使用该 Placeholder 的节点来推断其形状
    """
    for node in graph_def.node:
        for input_name in node.input:
            clean_input = input_name.split(":")[0].lstrip("^")
            if clean_input == placeholder_name:
                if node.op == "MatMul" or node.op == "Tensordot":
                    if "_output_shapes" in node.attr:
                        output_shapes = node.attr["_output_shapes"].list.shape
                        if len(output_shapes) > 0:
                            output_shape = output_shapes[0]
                            if len(output_shape.dim) == 2:
                                # 简单推断逻辑
                                return [64, 32]  # 默认权重矩阵大小
                elif node.op == "BiasAdd":
                    if "_output_shapes" in node.attr:
                        output_shapes = node.attr["_output_shapes"].list.shape
                        if len(output_shapes) > 0:
                            output_shape = output_shapes[0]
                            if len(output_shape.dim) >= 1:
                                return [output_shape.dim[-1].size]
    return None


def load_graph_and_get_placeholders(pb_path):
    """
    加载图并获取所有 placeholder 节点信息
    """
    print(f"\n=== 加载图文件: {pb_path} ===")

    if not os.path.exists(pb_path):
        print(f"错误: 文件 {pb_path} 不存在!")
        sys.exit(1)

    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def = graph_pb2.GraphDef()
        graph_def.ParseFromString(f.read())

    print(f"图加载成功，总节点数: {len(graph_def.node)}")

    placeholders = {}
    for node in graph_def.node:
        if node.op == "Placeholder":
            # 获取数据类型
            dtype_enum = node.attr["dtype"].type
            dtype_map = {
                tf.float32.as_datatype_enum: np.float32,
                tf.int32.as_datatype_enum: np.int32,
                tf.int64.as_datatype_enum: np.int64,
                tf.bool.as_datatype_enum: np.bool_,
                tf.string.as_datatype_enum: np.str_,
            }
            dtype = dtype_map.get(dtype_enum, np.float32)

            # 获取形状
            shape = []
            shape_found = False

            if "shape" in node.attr:
                shape_proto = node.attr["shape"].shape
                if not shape_proto.unknown_rank:
                    for dim in shape_proto.dim:
                        shape.append(dim.size if dim.size != -1 else None)
                    shape_found = True

            if not shape_found and "_output_shapes" in node.attr:
                output_shapes = node.attr["_output_shapes"].list.shape
                if len(output_shapes) > 0:
                    shape_proto = output_shapes[0]
                    if not shape_proto.unknown_rank:
                        for dim in shape_proto.dim:
                            shape.append(dim.size if dim.size != -1 else None)
                        shape_found = True

            # 兜底推断
            if not shape_found:
                inferred = infer_placeholder_shape_from_usage(graph_def, node.name)
                shape = inferred if inferred else []

            placeholders[node.name] = {"dtype": dtype, "shape": shape}

    print(f"找到 {len(placeholders)} 个 Placeholder 节点")
    return graph_def, placeholders


# ==========================================
# 3. 创建 Mock 数据 (含关键修复)
# ==========================================
def create_mock_data(placeholders, batch_size):
    """
    根据 placeholder 信息创建 mock 数据
    """
    print("\n=== 创建 Mock 数据 ===")

    feed_dict = {}

    for name, info in placeholders.items():
        shape = info["shape"]
        dtype = info["dtype"]

        # 处理形状，将 None 替换为 batch_size
        mock_shape = []
        for dim in shape:
            if dim is None:
                mock_shape.append(batch_size)
            elif dim == 0:
                # 维度为0的情况，保持为0
                mock_shape.append(0)
            else:
                mock_shape.append(dim)

        # 如果形状为空列表，说明是标量，但某些 Placeholder 可能需要特定形状
        # 检查名称中是否包含特定模式来推断正确的形状
        if not mock_shape:
            # 对于 ReadVariableOp/resource 类型的 Placeholder，尝试从名称推断形状
            if "/ReadVariableOp/resource" in name:
                # 这些通常是权重或偏置参数，需要根据上下文推断形状
                # 暂时使用一个合理的默认形状
                if "BiasAdd" in name:
                    # 偏置通常是一维向量
                    mock_shape = [32]  # 默认偏置大小
                elif "MatMul" in name or "Tensordot" in name:
                    # 权重矩阵通常是二维
                    mock_shape = [64, 32]  # 默认权重矩阵大小
                else:
                    # 其他情况使用标量
                    mock_shape = []
            else:
                # 其他标量 Placeholder 保持标量
                mock_shape = []

        # 生成 mock 数据
        if dtype == np.float32:
            # 生成随机浮点数据
            mock_data = np.random.normal(0.0, 1.0, mock_shape).astype(dtype)
        elif dtype == np.int32:
            # 生成随机整数数据
            mock_data = np.random.randint(0, 100, mock_shape).astype(dtype)
        elif dtype == np.int64:
            # 生成随机长整数数据
            mock_data = np.random.randint(0, 100, mock_shape).astype(dtype)
        elif dtype == np.bool_:
            # 生成随机布尔数据
            mock_data = np.random.choice([True, False], mock_shape).astype(dtype)
        else:
            # 默认生成浮点数据
            mock_data = np.random.normal(0.0, 1.0, mock_shape).astype(np.float32)

        feed_dict[name + ":0"] = mock_data

        print(f"Mock 数据 - {name}:")
        print(f"  形状: {mock_shape}")
        print(f"  数据类型: {dtype}")
        print(f"  数据范围: [{np.min(mock_data):.4f}, {np.max(mock_data):.4f}]")

    return feed_dict


# ==========================================
# 4. 执行推理
# ==========================================
def run_inference(graph_def, feed_dict, output_node_name, device="cpu", xla=False, num_runs=100, warmup_runs=10):
    print(f"\n=== 执行图推理 ===")
    print(f"输出节点: {output_node_name}")
    print(f"设备: {device.upper()}")
    if device.lower() == "cuda":
        print(f"XLA: {xla}")
    print(f"预热次数: {warmup_runs}, 正式运行次数: {num_runs}")

    with tf.Graph().as_default() as graph:
        # 测量图导入时间
        t_import_start = time.time()
        tf.import_graph_def(graph_def, name="")
        t_import_end = time.time()
        print(f"[时间] 图导入耗时: {(t_import_end - t_import_start)*1000:.2f} ms")

        # 准备 Session Feed
        t_feed_start = time.time()
        session_feed_dict = {}
        for name, data in feed_dict.items():
            try:
                tensor = graph.get_tensor_by_name(name)
                session_feed_dict[tensor] = data
            except KeyError:
                pass  # 忽略图中不存在的 tensor

        # 获取输出 Tensor
        try:
            output_tensor = graph.get_tensor_by_name(f"{output_node_name}:0")
        except KeyError:
            print(f"错误: 找不到输出张量 {output_node_name}:0")
            # 尝试打印所有节点找名字
            # for n in graph.as_graph_def().node: print(n.name)
            return None
        t_feed_end = time.time()
        print(f"[时间] Feed Dict 准备耗时: {(t_feed_end - t_feed_start)*1000:.2f} ms")

        # 配置 Session
        config = create_session_config(device_type=device, xla=xla)

        # 测量 Session 创建时间
        t_sess_start = time.time()
        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            t_sess_end = time.time()
            print(f"[时间] Session 创建耗时: {(t_sess_end - t_sess_start)*1000:.2f} ms")

            try:
                # 预热运行
                print(f">>> 预热运行 {warmup_runs} 次...")
                for _ in range(warmup_runs):
                    _ = sess.run(output_tensor, feed_dict=session_feed_dict)
                print(">>> 预热完成")

                # 正式测量
                print(f">>> 正式运行 {num_runs} 次...")
                run_times = []
                for i in range(num_runs):
                    t_run_start = time.time()
                    result = sess.run(output_tensor, feed_dict=session_feed_dict)
                    t_run_end = time.time()
                    run_times.append((t_run_end - t_run_start) * 1000)  # ms

                # 性能统计
                total_time = sum(run_times)
                avg_time = total_time / num_runs
                min_time = min(run_times)
                max_time = max(run_times)
                p50 = np.percentile(run_times, 50)
                p95 = np.percentile(run_times, 95)
                p99 = np.percentile(run_times, 99)

                print("\n" + "="*50)
                print("[性能统计]")
                print("="*50)
                print(f"  运行次数: {num_runs}")
                print(f"  总耗时:   {total_time:.2f} ms")
                print(f"  平均:     {avg_time:.4f} ms")
                print(f"  最小:     {min_time:.4f} ms")
                print(f"  最大:     {max_time:.4f} ms")
                print(f"  P50:      {p50:.4f} ms")
                print(f"  P95:      {p95:.4f} ms")
                print(f"  P99:      {p99:.4f} ms")
                print(f"  吞吐量:   {1000/avg_time:.2f} 次/秒")
                print("="*50)

                print(f"\n[推理结果统计]")
                print(f"  Shape: {result.shape}")
                print(f"  Dtype: {result.dtype}")
                print(f"  Min:   {np.min(result):.4f}")
                print(f"  Max:   {np.max(result):.4f}")
                print(f"  Mean:  {np.mean(result):.4f}")

                if result.size <= 20:
                    print(f"  Data: {result}")

                return result

            except Exception as e:
                print(f"\n!!!! 推理失败 !!!!")
                print(f"错误信息: {e}")
                import traceback
                traceback.print_exc()
                return None


# ==========================================
# 解析命令行参数
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="TensorFlow 图推理性能测试工具")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "musa"],
        help="运行设备: cpu, cuda, musa (默认: cpu)"
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=100,
        help="输入数据的 batch size (默认: 100)"
    )
    parser.add_argument(
        "--xla",
        action="store_true",
        help="启用 XLA 加速 (仅当 device=cuda 时有效)"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1000,
        help="正式运行次数 (默认: 1000)"
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=10,
        help="预热运行次数 (默认: 10)"
    )
    return parser.parse_args()


# ==========================================
# 主函数
# ==========================================
def main():
    args = parse_args()

    print("="*50)
    print("参数配置")
    print("="*50)
    print(f"  设备:     {args.device.upper()}")
    print(f"  Batch Size: {args.batchsize}")
    print(f"  XLA:      {args.xla if args.device == 'cuda' else 'N/A'}")
    print(f"  运行次数: {args.num_runs}")
    print(f"  预热次数: {args.warmup_runs}")
    print("="*50)

    total_start = time.time()

    # 仅当 device=musa 时加载 MUSA 插件
    if args.device.lower() == "musa":
        load_musa_plugin()

    # 1. 分析图
    t0 = time.time()
    graph_def, placeholders = load_graph_and_get_placeholders(model_path)
    print(f"[时间] 图加载与分析耗时: {(time.time() - t0)*1000:.2f} ms")
    if not placeholders:
        print("错误: 未找到 Placeholder")
        return

    # 2. 造数据 (含自动修复)
    t1 = time.time()
    feed_dict = create_mock_data(placeholders, args.batchsize)
    print(f"[时间] Mock 数据创建耗时: {(time.time() - t1)*1000:.2f} ms")

    # 3. 跑推理
    run_inference(
        graph_def,
        feed_dict,
        output_node_name,
        device=args.device,
        xla=args.xla,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs
    )

    total_end = time.time()
    print(f"\n[总耗时] {(total_end - total_start)*1000:.2f} ms")


if __name__ == "__main__":
    main()