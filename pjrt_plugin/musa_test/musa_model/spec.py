import os
import sys
import time
import argparse
import ctypes
import logging
from typing import Dict, List, Any, Optional, Tuple

# ==========================================
# 配置区域
# ==========================================
model_dir = "/workspace/openxla/pjrt_plugin/musa_test/musa_model"
model_path = os.path.join(model_dir, "graph_def.pb")
spec_model_paths = {
    "1": os.path.join(model_dir, "meta_graph_1.spec"),
    "2": os.path.join(model_dir, "meta_graph_2.spec"),
    "3": os.path.join(model_dir, "meta_graph_3.spec"),
}
output_node_name = "predicts"  # 默认输出节点名称；如果 spec 里有 SignatureDef，会自动优先使用签名输出
musa_plugin_path = "/workspace/openxla/bazel-bin/pjrt_plugin/libmusa_pjrt_plugin_zy.so"
musa_bridge_path = "/workspace/openxla/bazel-bin/pjrt_plugin/libmusa_tf215_registry_bridge.so"

# =========================================================================
# 🔥 核心补丁 1：注入 PJRT 环境变量 (必须在 import tensorflow 之前！)
# =========================================================================
os.environ["TF_PLUGGABLE_DEVICE_LIBRARY_PATH"] = musa_plugin_path
os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = f"MUSA:{musa_plugin_path}"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def _preimport_arg_value(name, default=None):
    prefix = f"{name}="
    argv = sys.argv[1:]
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(prefix):
            return value[len(prefix):]
    return default

def _append_env_flag(env_name, flag):
    current = os.environ.get(env_name, "").strip()
    flag_key = flag.split("=", 1)[0]
    for existing in current.split():
        existing_key = existing.split("=", 1)[0]
        if existing == flag or existing_key == flag_key:
            return
    if flag in current:
        return
    os.environ[env_name] = f"{current} {flag}".strip() if current else flag

_preimport_device = (_preimport_arg_value("--device", "cpu") or "cpu").lower()
if _preimport_device == "musa":
    os.environ.setdefault("MUSA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("MUSA_PJRT_PREALLOCATE", "false")

    if "--xla" in sys.argv:
        _append_env_flag("TF_XLA_FLAGS", "--tf_xla_use_device_api=true")
        _append_env_flag("TF_XLA_FLAGS", "--tf_xla_use_device_api_for_auto_jit=true")
        _append_env_flag("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")
        _xla_min_cluster_size = _preimport_arg_value(
            "--xla_min_cluster_size",
            os.environ.get("MUSA_TF_XLA_MIN_CLUSTER_SIZE", "1"),
        )
        _append_env_flag(
            "TF_XLA_FLAGS",
            f"--tf_xla_min_cluster_size={_xla_min_cluster_size}",
        )
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' # 如果需要看 C++ 底层 printf，取消注释此行

# 环境变量设置完毕后，再导入 TensorFlow
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
tf_v1 = tf

# 禁用 V2 行为，确保 TF1 图能正常运行
tf.disable_v2_behavior()
tf.disable_eager_execution()

# ==========================================
# 1. 加载 MUSA 插件 (使用 ctypes 强行挂载 PJRT)
# ==========================================
def load_musa_plugin():
    print("\n====================================================")
    print("[PYTHON] 正在强行调起 MUSA C++ PJRT 注册钩子...")
    if os.path.exists(musa_plugin_path):
        try:
            # 加载我们编译出的动态库
            mode = getattr(ctypes, "RTLD_GLOBAL", None)
            lib = ctypes.CDLL(musa_plugin_path, mode=mode) if mode is not None else ctypes.CDLL(musa_plugin_path)
            # 执行我们在 C++ 里写的暴露给 Python 的函数
            lib.ForceRegisterMusa()
            print(f">>>> [MUSA] PJRT Plugin hook executed successfully from: {musa_plugin_path}")
            if os.path.exists(musa_bridge_path):
                bridge = ctypes.CDLL(musa_bridge_path, mode=mode) if mode is not None else ctypes.CDLL(musa_bridge_path)
                register_bridge = getattr(bridge, "MusaTf215_RegisterRuntimeFactory", None)
                if register_bridge is not None:
                    register_bridge.restype = ctypes.c_int
                    register_bridge.argtypes = []
                    print(f">>>> [MUSA] MusaTf215_RegisterRuntimeFactory 返回: {register_bridge()}")
            
            # 顺便验证一下物理设备是否被 TF 识别
            devices = tf.config.list_logical_devices('MUSA')
            if devices:
                print(f">>>> [MUSA] ✅ 成功检测到物理设备: {devices}")
            else:
                print(f">>>> [MUSA] ⚠️ 钩子已执行，但 TF 暂未列出设备。")
                
        except Exception as e:
            print(f"!!!! [MUSA] 💥 强制注册钩子调起失败：{e}")
            sys.exit(1)
    else:
        print(f"!!!! [MUSA] Plugin not found at {musa_plugin_path}, assuming built-in.")
    print("====================================================\n")

def create_session_config(
    device_type: str = "cpu",
    xla: bool = False,
    log_device_placement: bool = False,
    logger: Optional[logging.Logger] = None,
) -> tf_v1.ConfigProto:
    """Create a Session config with device and XLA settings."""
    config = tf_v1.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = log_device_placement

    device_type_upper = (device_type or "cpu").upper()

    if device_type_upper == "CUDA":
        config.gpu_options.allow_growth = True
        # XLA 仅在 CUDA 设备下有效
        if xla:
            config.graph_options.optimizer_options.global_jit_level = tf_v1.OptimizerOptions.ON_1
            if logger is not None:
                logger.info("Enabled XLA JIT compilation for CUDA")

    elif device_type_upper == "MUSA":
        # 保留你原本针对 MUSA 的 custom optimizer 逻辑
        from tensorflow.core.protobuf import rewriter_config_pb2

        rewrite_options = config.graph_options.rewrite_options
        rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
        rewrite_options.remapping = rewriter_config_pb2.RewriterConfig.OFF
        rewrite_options.custom_optimizers.add().name = "musa_graph_optimizer"
        if xla:
            config.graph_options.optimizer_options.global_jit_level = tf_v1.OptimizerOptions.ON_2
        if logger is not None:
            logger.info("Enabled custom optimizer: musa_graph_optimizer")

    return config

# ==========================================
# 2. 模型/spec 加载与 Placeholder 输入解析
# ==========================================

def _dtype_enum_to_numpy_dtype(dtype_enum):
    """把 TensorFlow dtype enum 转成 numpy dtype，用于构造 feed 输入。"""
    dtype_map = {
        tf.float16.as_datatype_enum: np.float16,
        tf.float32.as_datatype_enum: np.float32,
        tf.float64.as_datatype_enum: np.float64,
        tf.int8.as_datatype_enum: np.int8,
        tf.int16.as_datatype_enum: np.int16,
        tf.int32.as_datatype_enum: np.int32,
        tf.int64.as_datatype_enum: np.int64,
        tf.uint8.as_datatype_enum: np.uint8,
        tf.uint16.as_datatype_enum: np.uint16,
        tf.uint32.as_datatype_enum: np.uint32,
        tf.uint64.as_datatype_enum: np.uint64,
        tf.bool.as_datatype_enum: np.bool_,
        # TF string placeholder feed 推荐使用 bytes/object，避免 unicode 转换问题
        tf.string.as_datatype_enum: np.object_,
    }
    return dtype_map.get(dtype_enum, np.float32)


def _shape_proto_to_list(shape_proto):
    """把 TensorShapeProto 转成 Python list；动态维度统一记为 None；unknown_rank 返回 None。"""
    if shape_proto.unknown_rank:
        return None

    shape = []
    for dim in shape_proto.dim:
        shape.append(dim.size if dim.size not in (-1,) else None)
    return shape


def _clean_tensor_name(tensor_name: str) -> str:
    """把 name:0 转成 name，兼容 ^control_dependency。"""
    return (tensor_name or "").split(":", 1)[0].lstrip("^")


def _normalize_tensor_name(tensor_name: str) -> str:
    """确保 tensor 名包含 :0，方便 graph.get_tensor_by_name 查找。"""
    if not tensor_name:
        return tensor_name
    return tensor_name if ":" in tensor_name else f"{tensor_name}:0"


def infer_placeholder_shape_from_usage(graph_def, placeholder_name):
    for node in graph_def.node:
        for input_name in node.input:
            clean_input = _clean_tensor_name(input_name)
            if clean_input == placeholder_name:
                if node.op == "MatMul" or node.op == "Tensordot":
                    if "_output_shapes" in node.attr:
                        output_shapes = node.attr["_output_shapes"].list.shape
                        if len(output_shapes) > 0:
                            output_shape = output_shapes[0]
                            if len(output_shape.dim) == 2:
                                return [64, 32]
                elif node.op == "BiasAdd":
                    if "_output_shapes" in node.attr:
                        output_shapes = node.attr["_output_shapes"].list.shape
                        if len(output_shapes) > 0:
                            output_shape = output_shapes[0]
                            if len(output_shape.dim) >= 1:
                                return [output_shape.dim[-1].size]
    return None


def _extract_signature_io(meta_graph, signature_key: Optional[str] = None):
    """
    从 MetaGraphDef 的 SignatureDef 里提取真实输入/输出 tensor。
    如果指定 signature_key，则优先使用该签名；否则按 serving_default -> predict -> 第一个签名 的顺序选择。
    """
    if not getattr(meta_graph, "signature_def", None):
        return None, {}, {}

    signature_def = meta_graph.signature_def
    selected_key = None

    candidate_keys = []
    if signature_key:
        candidate_keys.append(signature_key)
    candidate_keys.extend(["serving_default", "predict", "classification", "regression"])

    for key in candidate_keys:
        if key in signature_def:
            selected_key = key
            break

    if selected_key is None and len(signature_def) > 0:
        selected_key = next(iter(signature_def.keys()))

    if selected_key is None:
        return None, {}, {}

    sig = signature_def[selected_key]
    inputs = {
        logical_name: tensor_info.name
        for logical_name, tensor_info in sig.inputs.items()
        if tensor_info.name
    }
    outputs = {
        logical_name: tensor_info.name
        for logical_name, tensor_info in sig.outputs.items()
        if tensor_info.name
    }
    return selected_key, inputs, outputs



def _extract_init_op_names_from_meta_graph(meta_graph):
    """从 MetaGraphDef 的 collection_def 中提取可能的初始化 op 名称。"""
    if meta_graph is None:
        return []

    init_keys = [
        "saved_model_main_op",
        "legacy_init_op",
        "init_op",
        "table_initializer",
        "table_initializers",
    ]
    init_op_names = []
    collection_def = getattr(meta_graph, "collection_def", {})

    for key in init_keys:
        if key not in collection_def:
            continue
        collection = collection_def[key]
        kind = collection.WhichOneof("kind")
        if kind == "node_list":
            init_op_names.extend(list(collection.node_list.value))
        # 其他 kind 例如 bytes_list/any_list 暂不解析，spec 里常见的是 node_list。

    # 去重且保持顺序
    deduped = []
    seen = set()
    for name in init_op_names:
        clean_name = _clean_tensor_name(name)
        if clean_name and clean_name not in seen:
            deduped.append(clean_name)
            seen.add(clean_name)
    return deduped


def _parse_model_proto(raw_data: bytes, pb_path: str):
    """
    支持两类文件：
      1) GraphDef: 常见 graph_def.pb
      2) MetaGraphDef: 常见 .meta/.spec，里面包含 graph_def/signature_def/stripped_op_list
    """
    ext = os.path.splitext(pb_path)[1].lower()

    # .spec/.meta 优先按 MetaGraphDef 解析；普通 .pb 优先按 GraphDef 解析。
    parse_order = ["meta_graph", "graph_def"] if ext in {".spec", ".meta"} else ["graph_def", "meta_graph"]
    last_error = None

    for proto_type in parse_order:
        try:
            if proto_type == "meta_graph":
                meta_graph = meta_graph_pb2.MetaGraphDef()
                meta_graph.ParseFromString(raw_data)
                if len(meta_graph.graph_def.node) > 0:
                    return meta_graph.graph_def, meta_graph, "MetaGraphDef"
            else:
                graph_def = graph_pb2.GraphDef()
                graph_def.ParseFromString(raw_data)
                if len(graph_def.node) > 0:
                    return graph_def, None, "GraphDef"
        except Exception as e:
            last_error = e

    raise ValueError(f"无法从 {pb_path} 解析出有效的 GraphDef/MetaGraphDef: {last_error}")


def _extract_placeholders_from_graph(graph_def, signature_inputs: Optional[Dict[str, str]] = None):
    """
    从 graph_def 中提取 Placeholder。
    如果 signature_inputs 不为空，则只保留签名里声明的输入，避免把内部占位节点也 feed 进去。
    """
    placeholders = {}
    signature_node_names = None
    signature_logical_names = {}

    if signature_inputs:
        signature_node_names = set()
        for logical_name, tensor_name in signature_inputs.items():
            node_name = _clean_tensor_name(tensor_name)
            signature_node_names.add(node_name)
            signature_logical_names[node_name] = logical_name

    for node in graph_def.node:
        if node.op != "Placeholder":
            continue

        if signature_node_names is not None and node.name not in signature_node_names:
            continue

        dtype_enum = node.attr["dtype"].type
        dtype = _dtype_enum_to_numpy_dtype(dtype_enum)

        shape = []
        shape_found = False

        if "shape" in node.attr:
            parsed_shape = _shape_proto_to_list(node.attr["shape"].shape)
            if parsed_shape is not None:
                shape = parsed_shape
                shape_found = True

        if not shape_found and "_output_shapes" in node.attr:
            output_shapes = node.attr["_output_shapes"].list.shape
            if len(output_shapes) > 0:
                parsed_shape = _shape_proto_to_list(output_shapes[0])
                if parsed_shape is not None:
                    shape = parsed_shape
                    shape_found = True

        if not shape_found:
            inferred = infer_placeholder_shape_from_usage(graph_def, node.name)
            shape = inferred if inferred else []

        logical_name = signature_logical_names.get(node.name, node.name)
        placeholders[node.name] = {
            "dtype": dtype,
            "shape": shape,
            "tensor_name": f"{node.name}:0",
            "logical_name": logical_name,
        }

    return placeholders


def _resolve_output_node_name(graph_def, signature_outputs: Optional[Dict[str, str]], output_override: Optional[str]):
    if output_override:
        return _clean_tensor_name(output_override)

    if signature_outputs:
        # 优先使用 SignatureDef 第一个输出，通常是真实 serving output
        _, tensor_name = next(iter(signature_outputs.items()))
        return _clean_tensor_name(tensor_name)

    # 保留老逻辑：默认取 predicts
    for node in graph_def.node:
        if node.name == output_node_name:
            return output_node_name

    # 如果 spec 没有 SignatureDef 且没有 predicts，给一个兜底输出，避免直接 KeyError。
    # 这里只跳过明显不适合作 fetch 的节点。
    skip_ops = {"NoOp", "Const", "Placeholder", "Assign"}
    for node in reversed(graph_def.node):
        if node.op not in skip_ops:
            print(f">>>> [模型] 未找到默认输出 {output_node_name}，兜底使用最后一个可 fetch 节点: {node.name}")
            return node.name

    return output_node_name


def load_model_and_get_placeholders(
    pb_path: str,
    signature_key: Optional[str] = None,
    output_override: Optional[str] = None,
):
    print(f"\n=== 加载模型文件: {pb_path} ===")
    if not os.path.exists(pb_path):
        print(f"错误: 文件 {pb_path} 不存在!")
        sys.exit(1)

    with tf.io.gfile.GFile(pb_path, "rb") as f:
        raw_data = f.read()

    graph_def, meta_graph, proto_type = _parse_model_proto(raw_data, pb_path)
    print(f"模型解析类型: {proto_type}")
    print(f"图加载成功，总节点数: {len(graph_def.node)}")

    selected_signature = None
    signature_inputs = {}
    signature_outputs = {}
    init_op_names = []

    if meta_graph is not None:
        selected_signature, signature_inputs, signature_outputs = _extract_signature_io(
            meta_graph,
            signature_key=signature_key,
        )
        if selected_signature:
            print(f"使用 SignatureDef: {selected_signature}")
            print(f"  签名输入数: {len(signature_inputs)}")
            print(f"  签名输出数: {len(signature_outputs)}")
        init_op_names = _extract_init_op_names_from_meta_graph(meta_graph)
        if init_op_names:
            print(f"  MetaGraph 初始化 op: {init_op_names}")

    placeholders = _extract_placeholders_from_graph(graph_def, signature_inputs=signature_inputs)
    if signature_inputs and not placeholders:
        print(">>>> [模型] SignatureDef 中没有匹配到 Placeholder，回退为扫描全部 Placeholder。")
        placeholders = _extract_placeholders_from_graph(graph_def, signature_inputs=None)

    resolved_output = _resolve_output_node_name(graph_def, signature_outputs, output_override)

    print(f"找到 {len(placeholders)} 个 Placeholder 输入")
    print(f"输出节点: {resolved_output}")

    return graph_def, placeholders, resolved_output, init_op_names


def load_graph_and_get_placeholders(pb_path):
    """兼容旧调用：返回 graph_def/placeholders。"""
    graph_def, placeholders, _, _ = load_model_and_get_placeholders(pb_path)
    return graph_def, placeholders


def resolve_model_paths(args):
    if args.run_all_specs:
        return [spec_model_paths[str(i)] for i in range(1, 4)]

    if args.model_path:
        return [args.model_path]

    if args.model_id:
        return [spec_model_paths[args.model_id]]

    return [model_path]


# ==========================================
# 3. 创建 Mock 数据 / 输入 Feed 逻辑
# ==========================================


def _materialize_mock_shape(shape, batch_size):
    """
    把 Placeholder shape 转成实际 mock shape。
    None/-1 视为动态维度；默认都用 batch_size 填充。
    """
    mock_shape = []
    for dim in shape:
        if dim is None or dim == -1:
            mock_shape.append(batch_size)
        else:
            mock_shape.append(dim)
    return mock_shape


def _format_mock_range(mock_data):
    if not hasattr(mock_data, "size") or mock_data.size == 0:
        return "EMPTY"
    if mock_data.dtype == np.object_:
        sample = mock_data.flat[0] if mock_data.shape else mock_data.item()
        return f"sample={sample!r}"
    return f"[{np.min(mock_data):.4f}, {np.max(mock_data):.4f}]"


def _is_shape_compatible(data_shape, tensor_shape):
    """判断 numpy feed 的 shape 是否兼容 graph 中 Tensor 的静态 shape。"""
    if tensor_shape is None:
        return True
    if len(data_shape) != len(tensor_shape):
        return False
    for actual_dim, expected_dim in zip(data_shape, tensor_shape):
        if expected_dim is not None and actual_dim != expected_dim:
            return False
    return True


def _target_shape_from_tensor_shape(tensor_shape, data_shape):
    """用图里真实 Tensor shape 修正 mock shape；动态维度优先沿用原 feed 的 batch。"""
    if tensor_shape is None:
        return None

    target_shape = []
    fallback_batch = data_shape[0] if len(data_shape) > 0 else 1
    for index, dim in enumerate(tensor_shape):
        if dim is None:
            if index < len(data_shape):
                target_shape.append(data_shape[index])
            elif index == 0:
                target_shape.append(fallback_batch)
            else:
                target_shape.append(1)
        else:
            target_shape.append(dim)
    return target_shape


def _make_mock_array_for_shape(dtype, shape, int_max_id: int = 5):
    """按照 dtype/shape 重新生成一份安全 mock 数据。"""
    np_dtype = np.dtype(dtype)

    if np.issubdtype(np_dtype, np.floating):
        return np.random.normal(0.0, 1.0, shape).astype(np_dtype)
    if np.issubdtype(np_dtype, np.integer):
        high = max(1, int_max_id)
        return np.random.randint(0, high, shape).astype(np_dtype)
    if np_dtype == np.dtype(np.bool_):
        return np.random.choice([True, False], shape).astype(np_dtype)
    if np_dtype == np.dtype(object):
        return np.full(shape, b"mock", dtype=object)
    return np.zeros(shape, dtype=np_dtype)


def _coerce_feed_value_to_tensor_shape(tensor_name, data, tensor):
    """
    最后一层防线：以 import 后 graph.get_tensor_by_name() 拿到的真实 Tensor shape 为准。
    这样即使 SignatureDef 里的 shape 和 GraphDef Placeholder shape 不一致，也不会 feed 错形状。
    """
    try:
        tensor_shape = tensor.shape.as_list()
    except Exception:
        return data

    data_shape = list(getattr(data, "shape", []))
    if _is_shape_compatible(data_shape, tensor_shape):
        return data

    target_shape = _target_shape_from_tensor_shape(tensor_shape, data_shape)
    if target_shape is None:
        return data

    fixed_data = _make_mock_array_for_shape(data.dtype, target_shape)
    print(
        f">>>> [输入修正] {tensor_name}: feed shape {tuple(data_shape)} 与图中 shape "
        f"{tensor_shape} 不匹配，已按真实 Placeholder shape 重建为 {tuple(target_shape)}"
    )
    return fixed_data


def create_mock_data(
    placeholders,
    batch_size,
    int_max_id: int = 5,
    seed: Optional[int] = None,
):
    """
    根据 Placeholder 信息创建 mock 输入数据。

    输入策略：
      - float: 正态分布随机值
      - int: 小范围 ID，默认 [0, 5)，避免 Gather/Embedding 越界
      - bool: 随机 True/False
      - string: bytes/object，占位填充 b"mock"
      - 动态维度 None/-1: 替换成 batch_size
    """
    print("\n=== 创建 Mock 输入数据 ===")

    if seed is not None:
        np.random.seed(seed)

    feed_dict = {}

    for name, info in placeholders.items():
        shape = info.get("shape", [])
        dtype = info.get("dtype", np.float32)
        tensor_name = info.get("tensor_name", f"{name}:0")
        logical_name = info.get("logical_name", name)

        mock_shape = _materialize_mock_shape(shape, batch_size)

        if dtype in (np.float16, np.float32, np.float64):
            mock_data = np.random.normal(0.0, 1.0, mock_shape).astype(dtype)
        elif dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64):
            # 对 embedding/sparse id 输入更安全；如果需要更大 ID，可用 --int_max_id 调整
            high = max(1, int_max_id)
            mock_data = np.random.randint(0, high, mock_shape).astype(dtype)
        elif dtype == np.bool_:
            mock_data = np.random.choice([True, False], mock_shape).astype(dtype)
        elif dtype == np.object_:
            mock_data = np.full(mock_shape, b"mock", dtype=object)
        else:
            mock_data = np.zeros(mock_shape, dtype=dtype)

        feed_dict[_normalize_tensor_name(tensor_name)] = mock_data

        print(f"Mock 输入 - {logical_name} ({tensor_name}):")
        print(f"  形状: {mock_shape}")
        print(f"  数据类型: {dtype}")
        print(f"  数据范围: {_format_mock_range(mock_data)}")

    return feed_dict


# ==========================================
# 4. 变量初始化逻辑
# ==========================================


def _get_graph_ops_by_names(graph, op_names):
    ops = []
    for name in op_names or []:
        clean_name = _clean_tensor_name(name)
        try:
            ops.append(graph.get_operation_by_name(clean_name))
            continue
        except KeyError:
            pass
        try:
            ops.append(graph.get_tensor_by_name(_normalize_tensor_name(clean_name)).op)
        except Exception:
            print(f">>>> [变量初始化] 图中找不到 init op: {name}")
    return ops


def _collect_assign_ops(graph):
    assign_types = {"Assign", "AssignAdd", "AssignSub", "AssignVariableOp", "AssignAddVariableOp", "AssignSubVariableOp"}
    return [op for op in graph.get_operations() if op.type in assign_types]


def _collect_variable_ops(graph):
    return [op for op in graph.get_operations() if op.type in {"VariableV2", "VarHandleOp"}]


def _shape_to_zero_shape(shape):
    if shape is None:
        return []
    out = []
    for dim in shape:
        out.append(1 if dim is None or dim < 0 else dim)
    return out


def _build_variable_initialized_status_ops(graph):
    status = []
    variables = []
    for op in _collect_variable_ops(graph):
        try:
            if op.type == "VariableV2":
                tensor = op.outputs[0]
                status_op = tf.is_variable_initialized(tensor, name=f"{op.name.replace('/', '_')}_is_initialized")
            else:
                # Resource variable handle
                tensor = op.outputs[0]
                status_op = tf.raw_ops.VarIsInitializedOp(resource=tensor, name=f"{op.name.replace('/', '_')}_is_initialized")
            status.append(status_op)
            variables.append(op)
        except Exception as e:
            print(f">>>> [变量初始化] 无法构造初始化状态检查: {op.name} ({op.type}): {e}")
    return variables, status


def _build_zero_init_ops_for_variables(variable_ops, only_indices=None):
    zero_init_ops = []
    selected = set(only_indices) if only_indices is not None else None
    for index, op in enumerate(variable_ops):
        if selected is not None and index not in selected:
            continue
        try:
            safe_name = op.name.replace('/', '_').replace(':', '_')
            if op.type == "VariableV2":
                var_tensor = op.outputs[0]
                shape = _shape_to_zero_shape(var_tensor.shape.as_list())
                zero_value = tf.zeros(shape, dtype=var_tensor.dtype)
                zero_init_ops.append(
                    tf.assign(var_tensor, zero_value, validate_shape=False, name=f"{safe_name}_zero_init")
                )
            elif op.type == "VarHandleOp":
                handle = op.outputs[0]
                dtype = op.get_attr("dtype")
                shape_proto = op.get_attr("shape")
                try:
                    shape = _shape_to_zero_shape(shape_proto.as_list())
                except Exception:
                    shape = []
                zero_value = tf.zeros(shape, dtype=dtype)
                zero_init_ops.append(
                    tf.raw_ops.AssignVariableOp(resource=handle, value=zero_value, name=f"{safe_name}_zero_init")
                )
        except Exception as e:
            print(f">>>> [变量初始化] 无法构造 0 初始化 op: {op.name} ({op.type}): {e}")
    return zero_init_ops


def initialize_imported_graph_variables(
    sess,
    graph,
    meta_init_op_names=None,
    skip_variable_init=False,
    zero_init_uninitialized_vars=True,
):
    """在 warmup 前初始化 import_graph_def 导入的 VariableV2/VarHandleOp。"""
    if skip_variable_init:
        print(">>>> [变量初始化] 已按参数跳过。")
        return

    init_targets = []
    meta_init_ops = _get_graph_ops_by_names(graph, meta_init_op_names)
    if meta_init_ops:
        print(f">>>> [变量初始化] 运行 MetaGraph init/main ops: {[op.name for op in meta_init_ops]}")
        init_targets.extend(meta_init_ops)

    assign_ops = _collect_assign_ops(graph)
    if assign_ops:
        print(f">>>> [变量初始化] 检测到 {len(assign_ops)} 个 Assign/AssignVariableOp，尝试运行一次。")
        init_targets.extend(assign_ops)

    if init_targets:
        try:
            sess.run(init_targets)
            print(">>>> [变量初始化] 原图初始化 op 执行完成。")
        except Exception as e:
            print(f">>>> [变量初始化] 原图初始化 op 执行失败，继续检查未初始化变量: {e}")
    else:
        print(">>>> [变量初始化] 未发现 MetaGraph init op 或 Assign op，将检查 VariableV2 状态。")

    variable_ops, status_ops = _build_variable_initialized_status_ops(graph)
    if not variable_ops:
        print(">>>> [变量初始化] 图中没有 VariableV2/VarHandleOp。")
        return

    uninitialized_indices = list(range(len(variable_ops)))
    if status_ops:
        try:
            status_values = sess.run(status_ops)
            uninitialized_indices = [i for i, ok in enumerate(status_values) if not bool(ok)]
            if not uninitialized_indices:
                print(f">>>> [变量初始化] {len(variable_ops)} 个变量均已初始化。")
                return
            names = [variable_ops[i].name for i in uninitialized_indices[:20]]
            suffix = " ..." if len(uninitialized_indices) > 20 else ""
            print(f">>>> [变量初始化] 仍有 {len(uninitialized_indices)} 个变量未初始化: {names}{suffix}")
        except Exception as e:
            print(f">>>> [变量初始化] 变量状态检查失败，将按需执行 0 初始化兜底: {e}")

    if zero_init_uninitialized_vars:
        zero_init_ops = _build_zero_init_ops_for_variables(variable_ops, only_indices=uninitialized_indices)
        if zero_init_ops:
            print(f">>>> [变量初始化] 对 {len(zero_init_ops)} 个未初始化变量执行 0 初始化兜底。")
            sess.run(zero_init_ops)
            print(">>>> [变量初始化] 0 初始化兜底完成。")
    else:
        print(">>>> [变量初始化] 未启用 0 初始化兜底；如只是为了打通性能测试，可不要加 --no_zero_init_uninitialized_vars。")



# ==========================================
# 5. Graph3 崩溃定位辅助：图摘要 / 设备标注
# ==========================================

HIGH_RISK_OPS = {
    "Conv2D", "Einsum", "Switch", "Merge", "BitwiseAnd", "Pad",
    "Round", "Softplus", "Log1p", "Max", "GreaterEqual", "Abs",
}


def _parse_op_list(value: Optional[str]) -> set:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def dump_graph_summary(graph_def, title="GraphDef", top_k=40):
    """打印图结构摘要，用于对比 meta_graph_1/2/3 的 op 差异。"""
    from collections import Counter

    counter = Counter(node.op for node in graph_def.node)
    total = sum(counter.values())
    print("\n" + "=" * 50)
    print(f"[{title} 摘要]")
    print("=" * 50)
    print(f"  总节点数: {total}")
    print(f"  Op 类型数: {len(counter)}")
    print("  Top Ops:")
    for op, count in counter.most_common(top_k):
        print(f"    {op:<28} {count}")

    hit = sorted(op for op in HIGH_RISK_OPS if op in counter)
    if hit:
        print("  高风险/需重点验证 Ops:")
        for op in hit:
            print(f"    {op:<28} {counter[op]}")
    print("=" * 50 + "\n")


def clone_graph_def_with_device_overrides(
    graph_def,
    force_cpu_ops=None,
    clear_existing_devices=False,
):
    """
    复制 GraphDef，并按 op 类型覆写 device。

    主要用途是定位 graph3 的 native segfault：
      - clear_existing_devices=True: 清掉 spec 自带 device 约束，让 TF placer 重新放置；
      - force_cpu_ops=Conv2D,Einsum...: 把指定 op 类型强制放到 CPU，避免进入 MUSA/XLA cluster。
    """
    force_cpu_ops = set(force_cpu_ops or [])
    if not force_cpu_ops and not clear_existing_devices:
        return graph_def

    cloned = graph_pb2.GraphDef()
    cloned.CopyFrom(graph_def)

    forced_count = 0
    cleared_count = 0
    for node in cloned.node:
        if clear_existing_devices and node.device:
            node.device = ""
            cleared_count += 1
        if node.op in force_cpu_ops:
            node.device = "/device:CPU:0"
            forced_count += 1

    if clear_existing_devices:
        print(f">>>> [设备标注] 已清除 {cleared_count} 个节点的原始 device 标注。")
    if force_cpu_ops:
        print(
            f">>>> [设备标注] 已将 {forced_count} 个节点强制标注到 CPU；"
            f"op 类型: {sorted(force_cpu_ops)}"
        )
    return cloned

# ==========================================
# 5. 执行推理
# ==========================================
def run_inference(
    graph_def,
    feed_dict,
    output_node_name,
    device="cpu",
    xla=False,
    num_runs=100,
    warmup_runs=10,
    fetch_output=False,
    meta_init_op_names=None,
    skip_variable_init=False,
    zero_init_uninitialized_vars=True,
    no_device_scope=False,
    force_cpu_ops=None,
    clear_existing_devices=False,
):
    print(f"\n=== 执行图推理 ===")
    print(f"输出节点: {output_node_name}")
    print(f"设备: {device.upper()}")
    if device.lower() in {"cuda", "musa"}:
        print(f"XLA: {xla}")
    print(f"预热次数: {warmup_runs}, 正式运行次数: {num_runs}")

    # 默认使用 device scope 强制全图落入指定设备；graph3 调试时可用 --no_device_scope 放松该约束。
    device_name = f"/device:{device.upper()}:0" if device.lower() != "cpu" else "/device:CPU:0"
    force_cpu_ops = set(force_cpu_ops or [])
    import_graph_def_obj = clone_graph_def_with_device_overrides(
        graph_def,
        force_cpu_ops=force_cpu_ops,
        clear_existing_devices=clear_existing_devices,
    )

    with tf.Graph().as_default() as graph:
        t_import_start = time.time()
        if no_device_scope:
            print(">>>> [设备放置] 未使用全图 device scope，交由 TF placer + GraphDef device 标注决定。")
            tf.import_graph_def(import_graph_def_obj, name="")
        else:
            print(f">>>> [设备放置] 使用全图 device scope: {device_name}")
            with tf.device(device_name):
                tf.import_graph_def(import_graph_def_obj, name="")
        t_import_end = time.time()
        print(f"[时间] 图导入耗时: {(t_import_end - t_import_start)*1000:.2f} ms")

        t_feed_start = time.time()
        session_feed_dict = {}
        for name, data in feed_dict.items():
            try:
                tensor = graph.get_tensor_by_name(name)
                data = _coerce_feed_value_to_tensor_shape(name, data, tensor)
                session_feed_dict[tensor] = data
            except KeyError:
                print(f">>>> [输入跳过] 图中找不到 feed tensor: {name}")
                pass

        try:
            output_tensor = graph.get_tensor_by_name(f"{output_node_name}:0")
        except KeyError:
            print(f"错误: 找不到输出张量 {output_node_name}:0")
            return None
        materialize_output = fetch_output or not (device.lower() == "musa" and xla)
        run_fetch = output_tensor if materialize_output else output_tensor.op
        print(f"Fetch Mode: {'tensor' if materialize_output else 'op-only'}")
        if not materialize_output:
            print(">>>> [MUSA] Benchmark mode skips host-side tensor materialization")
        t_feed_end = time.time()
        print(f"[时间] Feed Dict 准备耗时: {(t_feed_end - t_feed_start)*1000:.2f} ms")

        config = create_session_config(device_type=device, xla=xla)

        t_sess_start = time.time()
        with tf_v1.Session(graph=graph, config=config) as sess:
            t_sess_end = time.time()
            print(f"[时间] Session 创建耗时: {(t_sess_end - t_sess_start)*1000:.2f} ms")

            initialize_imported_graph_variables(
                sess,
                graph,
                meta_init_op_names=meta_init_op_names,
                skip_variable_init=skip_variable_init,
                zero_init_uninitialized_vars=zero_init_uninitialized_vars,
            )

            try:
                print(f">>> 预热运行 {warmup_runs} 次...")
                for _ in range(warmup_runs):
                    _ = sess.run(run_fetch, feed_dict=session_feed_dict)
                    # time.sleep(1)
                print(">>> 预热完成")

                print(f">>> 正式运行 {num_runs} 次...")
                run_times = []
                result = None
                for i in range(num_runs):
                    t_run_start = time.time()
                    current = sess.run(run_fetch, feed_dict=session_feed_dict)
                    t_run_end = time.time()
                    run_times.append((t_run_end - t_run_start) * 1000)
                    if materialize_output:
                        result = current

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

                if not materialize_output:
                    print("\n[推理结果] Output tensor materialization was skipped.")
                    return None

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
        "--model_id",
        type=str,
        default=None,
        choices=["1", "2", "3"],
        help="读取内置 spec 模型: 1=meta_graph_1.spec, 2=meta_graph_2.spec, 3=meta_graph_3.spec"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="手动指定模型文件路径，支持 GraphDef(.pb) 或 MetaGraphDef(.spec/.meta)"
    )
    parser.add_argument(
        "--run_all_specs",
        action="store_true",
        help="依次读取并运行 meta_graph_1.spec、meta_graph_2.spec、meta_graph_3.spec"
    )
    parser.add_argument(
        "--signature_key",
        type=str,
        default=None,
        help="MetaGraphDef/Spec 中要使用的 SignatureDef key；不指定时自动选择 serving_default/predict/第一个签名"
    )
    parser.add_argument(
        "--output_node",
        type=str,
        default=None,
        help="手动指定输出节点名；不指定时优先使用 spec 的 SignatureDef 输出，其次使用默认 predicts"
    )
    parser.add_argument(
        "--int_max_id",
        type=int,
        default=5,
        help="int 输入 mock 数据的最大 ID 上界，生成范围为 [0, int_max_id)，默认 5，可避免 embedding/gather 越界"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子；指定后 mock 输入可复现"
    )
    parser.add_argument(
        "--xla",
        action="store_true",
        help="启用 XLA 加速 (device=cuda/musa)"
    )
    parser.add_argument(
        "--xla_min_cluster_size",
        type=int,
        default=int(os.environ.get("MUSA_TF_XLA_MIN_CLUSTER_SIZE", "1")),
        help="TF XLA auto-jit minimum cluster size; larger values reduce PJRT launch fragmentation",
    )
    parser.add_argument(
        "--fetch_output",
        action="store_true",
        help="MUSA+XLA benchmark 默认只执行输出 op；设置该参数才拉回输出 tensor"
    )
    parser.add_argument(
        "--skip_variable_init",
        action="store_true",
        help="跳过变量初始化；默认会先运行原图 Assign/init op，再对仍未初始化的变量做 0 初始化兜底"
    )
    parser.add_argument(
        "--no_zero_init_uninitialized_vars",
        action="store_true",
        help="关闭未初始化变量的 0 初始化兜底；只建议需要严格验证真实权重初始化时使用"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=100,
        help="正式运行次数 (默认: 1000)"
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=10,
        help="预热运行次数 (默认: 10)"
    )
    parser.add_argument(
        "--dump_graph_summary",
        action="store_true",
        help="打印 GraphDef 的 op 统计摘要，便于对比 graph1/2/3 的结构差异"
    )
    parser.add_argument(
        "--no_device_scope",
        action="store_true",
        help="导入图时不再用 tf.device 强制整图到目标设备；用于定位 MUSA/XLA native 崩溃"
    )
    parser.add_argument(
        "--force_cpu_ops",
        type=str,
        default="",
        help="逗号分隔的 op 类型列表，导入前将这些节点强制标注到 CPU，例如 Conv2D,Einsum,Switch,Merge"
    )
    parser.add_argument(
        "--clear_existing_devices",
        action="store_true",
        help="导入前清除 GraphDef 中已有 device 标注，让 TF 重新放置节点"
    )
    parser.add_argument(
        "--no_hard_exit",
        action="store_true",
        help="MUSA 模式默认在打印完结果后直接 os._exit(0)，跳过 TF/PJRT/Python 析构链；设置该参数保留普通退出流程"
    )
    return parser.parse_args()


# ==========================================
# 主函数
# ==========================================
def main():
    args = parse_args()
    selected_model_paths = resolve_model_paths(args)

    print("="*50)
    print("参数配置")
    print("="*50)
    print(f"  设备:       {args.device.upper()}")
    print(f"  Batch Size: {args.batchsize}")
    print(f"  模型文件:   {', '.join(selected_model_paths)}")
    print(f"  XLA:        {args.xla if args.device in {'cuda', 'musa'} else 'N/A'}")
    if args.xla:
        print(f"  XLA min cluster size: {args.xla_min_cluster_size}")
    print(f"  Fetch Output: {args.fetch_output}")
    print(f"  Skip Var Init: {args.skip_variable_init}")
    print(f"  Zero Init Uninitialized Vars: {not args.no_zero_init_uninitialized_vars}")
    print(f"  Int Max ID:  {args.int_max_id}")
    print(f"  No Device Scope: {args.no_device_scope}")
    print(f"  Force CPU Ops: {args.force_cpu_ops or 'N/A'}")
    print(f"  Clear Existing Devices: {args.clear_existing_devices}")
    print(f"  Hard Exit On MUSA: {args.device.lower() == 'musa' and not args.no_hard_exit}")
    if args.seed is not None:
        print(f"  Seed:        {args.seed}")
    if args.device == "musa":
        print(f"  MUSA_PJRT_PREALLOCATE: {os.environ.get('MUSA_PJRT_PREALLOCATE', '')}")
    print(f"  运行次数:   {args.num_runs}")
    print(f"  预热次数:   {args.warmup_runs}")
    print("="*50)

    total_start = time.time()

    # 1. 如果指定了 musa，调用强力外挂钩子
    if args.device.lower() == "musa":
        load_musa_plugin()

    for index, current_model_path in enumerate(selected_model_paths, start=1):
        if len(selected_model_paths) > 1:
            print("\n" + "#"*70)
            print(f"# 开始运行第 {index}/{len(selected_model_paths)} 个 spec: {current_model_path}")
            print("#"*70)

        # 2. 分析图 / spec
        t0 = time.time()
        graph_def, placeholders, resolved_output_node, init_op_names = load_model_and_get_placeholders(
            current_model_path,
            signature_key=args.signature_key,
            output_override=args.output_node,
        )
        print(f"[时间] 模型加载与输入分析耗时: {(time.time() - t0)*1000:.2f} ms")
        if args.dump_graph_summary:
            dump_graph_summary(graph_def, title=os.path.basename(current_model_path))
        if not placeholders:
            print("错误: 未找到 Placeholder")
            continue

        # 3. 造输入数据
        t1 = time.time()
        feed_dict = create_mock_data(
            placeholders,
            args.batchsize,
            int_max_id=args.int_max_id,
            seed=args.seed,
        )
        print(f"[时间] Mock 输入创建耗时: {(time.time() - t1)*1000:.2f} ms")

        # 4. 跑推理
        run_inference(
            graph_def,
            feed_dict,
            resolved_output_node,
            device=args.device,
            xla=args.xla,
            num_runs=args.num_runs,
            warmup_runs=args.warmup_runs,
            fetch_output=args.fetch_output,
            meta_init_op_names=init_op_names,
            skip_variable_init=args.skip_variable_init,
            zero_init_uninitialized_vars=not args.no_zero_init_uninitialized_vars,
            no_device_scope=args.no_device_scope,
            force_cpu_ops=_parse_op_list(args.force_cpu_ops),
            clear_existing_devices=args.clear_existing_devices,
        )

    total_end = time.time()
    print(f"\n[总耗时] {(total_end - total_start)*1000:.2f} ms")

    # MUSA/PJRT 插件通过 ctypes 加载后，普通 Python 退出阶段可能触发
    # TF/PJRT/C++ runtime 析构链里的后台线程/TLS 清理问题，表现为：
    #   Fatal Python error: _PyGILState_NoteThreadState: Couldn't create autoTSSkey mapping
    # Benchmark 结果已经打印完后，直接硬退出可以避免退出阶段 core dump，
    # 不影响已完成的 sess.run 性能统计。
    if args.device.lower() == "musa" and not args.no_hard_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
