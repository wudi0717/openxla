import argparse
import ctypes
import os
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
LINUX_REPO_ROOT = Path("/workspace/openxla")
LINUX_GRAPH_PATH = LINUX_REPO_ROOT / "pjrt_plugin/musa_test/musa_model/graph_def.pb"
LINUX_PLUGIN_PATH = LINUX_REPO_ROOT / "bazel-bin/pjrt_plugin/libmusa_pjrt_plugin_zy.so"
LINUX_BRIDGE_PATH = LINUX_REPO_ROOT / "bazel-bin/pjrt_plugin/libmusa_tf215_registry_bridge.so"

DEFAULT_OUTPUT_NODE = "predicts"
DEFAULT_GRAPH_PATH = LINUX_GRAPH_PATH if LINUX_GRAPH_PATH.exists() else SCRIPT_DIR / "graph_def.pb"
DEFAULT_MUSA_PLUGIN_PATH = LINUX_PLUGIN_PATH
DEFAULT_MUSA_BRIDGE_PATH = LINUX_BRIDGE_PATH

DEVICE_TYPE_MAP = {
    "cpu": "CPU",
    "cuda": "GPU",
    "musa": "MUSA",
}


def find_repo_root(path: Path) -> Path:
    for parent in path.parents:
        if (parent / "WORKSPACE").exists() or (parent / "WORKSPACE.bazel").exists():
            return parent
    return path.parent


REPO_ROOT = LINUX_REPO_ROOT if LINUX_REPO_ROOT.exists() else find_repo_root(SCRIPT_PATH)


def normalize_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def append_env_flag(env_name: str, flag: str) -> None:
    current = os.environ.get(env_name, "").strip()
    if flag in current:
        return
    os.environ[env_name] = f"{current} {flag}".strip() if current else flag


def append_env_entry(env_name: str, entry: str) -> None:
    current = os.environ.get(env_name, "").strip()
    if not current:
        os.environ[env_name] = entry
        return

    entries = [item.strip() for item in current.split(",") if item.strip()]
    if entry in entries:
        return
    os.environ[env_name] = ",".join(entries + [entry])


def runtime_device_index(args: argparse.Namespace) -> int:
    if args.device.lower() == "musa":
        return 0
    return args.device_index


def runtime_device_name(args: argparse.Namespace) -> str:
    return target_device_name(args.device, runtime_device_index(args))


def load_dynamic_library(path: Path) -> ctypes.CDLL:
    mode = getattr(ctypes, "RTLD_GLOBAL", None)
    if mode is None:
        return ctypes.CDLL(str(path))
    return ctypes.CDLL(str(path), mode=mode)


def maybe_call_symbol(
    library: ctypes.CDLL,
    symbol_name: str,
    restype=None,
    argtypes: Optional[Sequence[object]] = None,
):
    symbol = getattr(library, symbol_name, None)
    if symbol is None:
        return None
    if argtypes is not None:
        symbol.argtypes = list(argtypes)
    if restype is not None:
        symbol.restype = restype
    return symbol()


def load_tensorflow_musa_plugin(tf, plugin_path: Path) -> None:
    env_plugin_paths = os.environ.get("TF_PLUGGABLE_DEVICE_LIBRARY_PATH", "")
    if str(plugin_path) in env_plugin_paths:
        print(">>>> [MUSA] 插件已通过环境变量预注册，跳过显式 load_pluggable_device_library")
        return

    try:
        from tensorflow.python.framework import load_library as tf_load_library
    except Exception:
        tf_load_library = None

    if tf_load_library is not None:
        load_pluggable_device_library = getattr(
            tf_load_library,
            "load_pluggable_device_library",
            None,
        )
        if load_pluggable_device_library is not None:
            load_pluggable_device_library(str(plugin_path))
            print(">>>> [MUSA] load_pluggable_device_library 成功")
            return

    tf.load_op_library(str(plugin_path))


def prepare_runtime_env(args: argparse.Namespace) -> None:
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    if args.device.lower() == "musa":
        os.environ["MUSA_VISIBLE_DEVICES"] = str(args.device_index)
        plugin_path = normalize_path(args.musa_plugin_path)
        append_env_entry("TF_PLUGGABLE_DEVICE_LIBRARY_PATH", str(plugin_path))
        append_env_entry("PJRT_NAMES_AND_LIBRARY_PATHS", f"MUSA:{plugin_path}")
        if args.xla:
            os.environ["TF_NUM_INTEROP_THREADS"] = str(args.tf_interop_threads)
            os.environ["TF_NUM_INTRAOP_THREADS"] = str(args.tf_intraop_threads)
            os.environ["MUSA_PJRT_MAX_INFLIGHT_COMPILES"] = str(
                args.musa_pjrt_max_inflight_compiles
            )
            os.environ["MUSA_PJRT_MAX_INFLIGHT_TRANSFERS"] = str(
                args.musa_pjrt_max_inflight_transfers
            )
            os.environ["MUSA_PJRT_MAX_INFLIGHT_EXECUTES"] = str(
                args.musa_pjrt_max_inflight_executes
            )
    if args.xla:
        if args.device.lower() == "musa":
            append_env_flag("TF_XLA_FLAGS", "--tf_xla_use_device_api=true")
            append_env_flag("TF_XLA_FLAGS", "--tf_xla_use_device_api_for_auto_jit=true")
        append_env_flag("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")
        append_env_flag("TF_XLA_FLAGS", "--tf_xla_min_cluster_size=1")

    print("\n=== Runtime Env ===")
    for env_name in [
        "TF_ENABLE_ONEDNN_OPTS",
        "MUSA_VISIBLE_DEVICES",
        "TF_PLUGGABLE_DEVICE_LIBRARY_PATH",
        "PJRT_NAMES_AND_LIBRARY_PATHS",
        "TF_NUM_INTEROP_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "MUSA_PJRT_MAX_INFLIGHT_COMPILES",
        "MUSA_PJRT_MAX_INFLIGHT_TRANSFERS",
        "MUSA_PJRT_MAX_INFLIGHT_EXECUTES",
        "TF_XLA_FLAGS",
    ]:
        value = os.environ.get(env_name)
        if value:
            print(f"  {env_name}={value}")


def import_tensorflow_v1():
    try:
        import tensorflow.compat.v1 as tf
        from tensorflow.core.framework import graph_pb2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "未找到 TensorFlow 运行环境。请在包含 TensorFlow 2.15 的 Python 环境中执行该脚本，"
            "或先激活你们用于 pjrt_plugin 验证的虚拟环境。"
        ) from exc

    tf.disable_v2_behavior()
    tf.disable_eager_execution()
    return tf, graph_pb2


def canonical_device_type(device: str) -> str:
    device_key = (device or "cpu").lower()
    if device_key not in DEVICE_TYPE_MAP:
        raise ValueError(f"不支持的设备类型: {device}")
    return DEVICE_TYPE_MAP[device_key]


def target_device_name(device: str, device_index: int) -> str:
    return f"/device:{canonical_device_type(device)}:{device_index}"


def print_visible_devices(tf) -> None:
    logical_devices = tf.config.list_logical_devices()
    if not logical_devices:
        print("!!!! [TF] 当前进程没有发现任何逻辑设备")
        return

    print("\n=== TensorFlow 逻辑设备 ===")
    for device in logical_devices:
        print(f"  - {device.name} [{device.device_type}]")


def ensure_target_device_visible(tf, device: str, device_index: int) -> None:
    canonical = canonical_device_type(device)
    visible = [
        logical_device
        for logical_device in tf.config.list_logical_devices()
        if logical_device.device_type.upper() == canonical
    ]
    visible_index = 0 if canonical == "MUSA" else device_index

    if not visible:
        raise RuntimeError(
            f"TensorFlow 当前没有发现 {canonical} 逻辑设备。"
            "请先确认 pluggable device 插件、PJRT 插件和 bridge 已正确加载。"
        )

    if device_index >= len(visible) and canonical != "MUSA":
        raise RuntimeError(
            f"请求使用 {canonical}:{device_index}，但当前只发现了 {len(visible)} 个 {canonical} 设备。"
        )

    if visible_index >= len(visible):
        raise RuntimeError(
            f"Requested {canonical}:{visible_index}, but only found {len(visible)} visible {canonical} devices."
        )

    print(f">>>> [TF] Target device: {visible[visible_index].name}")
    if canonical == "MUSA":
        print(f">>>> [MUSA] Physical device {device_index} mapped to logical device 0 in current process")


def bootstrap_musa_runtime(tf, args: argparse.Namespace) -> Tuple[Path, Optional[Path]]:
    plugin_path = normalize_path(args.musa_plugin_path)
    bridge_path = normalize_path(args.musa_bridge_path)
    loaded_bridge_path: Optional[Path] = None

    if not plugin_path.exists():
        raise RuntimeError(
            f"MUSA PJRT 插件不存在: {plugin_path}\n"
            "请先执行 `bazel build //pjrt_plugin:libmusa_pjrt_plugin_zy.so`。"
        )

    print("\n=== MUSA Runtime Bootstrap ===")
    print(f"  Plugin: {plugin_path}")
    print(f"  Bridge: {bridge_path if bridge_path.exists() else '<not found>'}")

    plugin_lib = load_dynamic_library(plugin_path)

    try:
        load_tensorflow_musa_plugin(tf, plugin_path)
        print(">>>> [MUSA] TensorFlow 插件加载成功")
    except Exception as exc:
        raise RuntimeError(f"MUSA 插件加载失败: {exc}") from exc

    try:
        maybe_call_symbol(plugin_lib, "ForceRegisterMusa")
        print(">>>> [MUSA] ForceRegisterMusa 已执行")
    except Exception as exc:
        print(f"!!!! [MUSA] ForceRegisterMusa 调用失败: {exc}")

    if not bridge_path.exists():
        print("!!!! [MUSA] 未发现 runtime bridge，继续仅使用 pluggable device + PJRT。")
        return plugin_path, None

    try:
        bridge_lib = load_dynamic_library(bridge_path)
        result = maybe_call_symbol(
            bridge_lib,
            "MusaTf215_RegisterRuntimeFactory",
            restype=ctypes.c_int,
            argtypes=[],
        )
        print(f">>>> [MUSA] MusaTf215_RegisterRuntimeFactory 返回: {result}")
        loaded_bridge_path = bridge_path
    except OSError as exc:
        print(
            "!!!! [MUSA] runtime bridge 加载失败，继续仅使用 pluggable device + PJRT: "
            f"{exc}"
        )
    except Exception as exc:
        print(
            "!!!! [MUSA] runtime bridge 注册失败，继续仅使用 pluggable device + PJRT: "
            f"{exc}"
        )

    return plugin_path, loaded_bridge_path


def create_session_config(
    tf,
    device_type: str,
    xla: bool,
    log_device_placement: bool,
    allow_soft_placement: bool,
    tf_interop_threads: int,
    tf_intraop_threads: int,
):
    from tensorflow.core.protobuf import rewriter_config_pb2

    config = tf.ConfigProto()
    config.allow_soft_placement = allow_soft_placement
    config.log_device_placement = log_device_placement

    device_type_upper = canonical_device_type(device_type)
    rewrite_options = config.graph_options.rewrite_options

    if device_type_upper == "GPU":
        config.gpu_options.allow_growth = True

    if device_type_upper == "MUSA":
        config.inter_op_parallelism_threads = tf_interop_threads
        config.intra_op_parallelism_threads = tf_intraop_threads
        rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
        rewrite_options.remapping = rewriter_config_pb2.RewriterConfig.OFF
        rewrite_options.custom_optimizers.add().name = "musa_graph_optimizer"
        print(">>>> [MUSA] 已关闭 oneDNN/MKL 相关 Grappler rewrite")
        print(">>>> [MUSA] 已启用 musa_graph_optimizer")
        print(
            ">>>> [MUSA] TensorFlow inter/intra op parallelism: "
            f"{tf_interop_threads}/{tf_intraop_threads}"
        )

    if xla:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_2
        print(f">>>> [XLA] 已为 {device_type_upper} 打开 XLA JIT")

    return config


def infer_placeholder_shape_from_usage(graph_def, placeholder_name: str) -> Optional[List[int]]:
    for node in graph_def.node:
        for input_name in node.input:
            clean_input = input_name.split(":")[0].lstrip("^")
            if clean_input != placeholder_name:
                continue

            if node.op in {"MatMul", "Tensordot"} and "_output_shapes" in node.attr:
                output_shapes = node.attr["_output_shapes"].list.shape
                if output_shapes and len(output_shapes[0].dim) == 2:
                    return [64, 32]

            if node.op == "BiasAdd" and "_output_shapes" in node.attr:
                output_shapes = node.attr["_output_shapes"].list.shape
                if output_shapes and output_shapes[0].dim:
                    return [output_shapes[0].dim[-1].size]

    return None


def load_graph_and_get_placeholders(tf, graph_pb2, pb_path: Path):
    print(f"\n=== 加载图文件: {pb_path} ===")

    if not pb_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {pb_path}")

    with tf.io.gfile.GFile(str(pb_path), "rb") as file_handle:
        graph_def = graph_pb2.GraphDef()
        graph_def.ParseFromString(file_handle.read())

    print(f"图加载成功，总节点数: {len(graph_def.node)}")

    placeholders = {}
    for node in graph_def.node:
        if node.op != "Placeholder":
            continue

        dtype_enum = node.attr["dtype"].type
        dtype_map = {
            tf.float32.as_datatype_enum: np.float32,
            tf.int32.as_datatype_enum: np.int32,
            tf.int64.as_datatype_enum: np.int64,
            tf.bool.as_datatype_enum: np.bool_,
            tf.string.as_datatype_enum: np.str_,
        }
        dtype = dtype_map.get(dtype_enum, np.float32)

        shape: List[Optional[int]] = []
        shape_found = False

        if "shape" in node.attr:
            shape_proto = node.attr["shape"].shape
            if not shape_proto.unknown_rank:
                for dim in shape_proto.dim:
                    shape.append(dim.size if dim.size != -1 else None)
                shape_found = True

        if not shape_found and "_output_shapes" in node.attr:
            output_shapes = node.attr["_output_shapes"].list.shape
            if output_shapes:
                shape_proto = output_shapes[0]
                if not shape_proto.unknown_rank:
                    for dim in shape_proto.dim:
                        shape.append(dim.size if dim.size != -1 else None)
                    shape_found = True

        if not shape_found:
            inferred = infer_placeholder_shape_from_usage(graph_def, node.name)
            shape = inferred if inferred else []

        placeholders[node.name] = {"dtype": dtype, "shape": shape}

    print(f"找到 {len(placeholders)} 个 Placeholder 节点")
    return graph_def, placeholders


def summarize_array(name: str, array: np.ndarray) -> None:
    print(f"Mock 数据 - {name}:")
    print(f"  形状: {list(array.shape)}")
    print(f"  数据类型: {array.dtype}")

    if array.size == 0:
        print("  数据范围: <empty>")
        return

    if np.issubdtype(array.dtype, np.number):
        print(f"  数据范围: [{np.min(array):.4f}, {np.max(array):.4f}]")
    else:
        sample = array.flat[0]
        print(f"  示例值: {sample}")


def create_mock_data(placeholders, batch_size: int, seed: int):
    print("\n=== 创建 Mock 数据 ===")

    rng = np.random.default_rng(seed)
    feed_dict = {}

    for name, info in placeholders.items():
        shape = info["shape"]
        dtype = info["dtype"]

        mock_shape = []
        for dim in shape:
            if dim is None or dim == -1:
                mock_shape.append(batch_size)
            else:
                mock_shape.append(dim)

        if not mock_shape and "/ReadVariableOp/resource" in name:
            if "BiasAdd" in name:
                mock_shape = [32]
            elif "MatMul" in name or "Tensordot" in name:
                mock_shape = [64, 32]

        if dtype == np.float32:
            mock_data = rng.normal(0.0, 1.0, size=mock_shape).astype(dtype)
        elif dtype == np.int32:
            mock_data = rng.integers(0, 100, size=mock_shape, dtype=np.int32)
        elif dtype == np.int64:
            mock_data = rng.integers(0, 100, size=mock_shape, dtype=np.int64)
        elif dtype == np.bool_:
            mock_data = rng.choice(np.array([False, True], dtype=np.bool_), size=mock_shape)
        elif dtype == np.str_:
            mock_data = np.full(mock_shape if mock_shape else (), "mock", dtype=np.str_)
        else:
            mock_data = rng.normal(0.0, 1.0, size=mock_shape).astype(np.float32)

        feed_dict[f"{name}:0"] = mock_data
        summarize_array(name, mock_data)

    return feed_dict


def import_graph_with_runtime_scope(
    tf,
    graph_def,
    device_name: str,
    enable_xla: bool,
    device_type: str,
) -> None:
    with ExitStack() as stack:
        stack.enter_context(tf.device(device_name))
        if enable_xla and canonical_device_type(device_type) == "GPU":
            stack.enter_context(tf.xla.experimental.jit_scope(compile_ops=True))
        tf.import_graph_def(graph_def, name="")


def format_output_tensor_name(output_node_name: str) -> str:
    return output_node_name if ":" in output_node_name else f"{output_node_name}:0"


def optional_output_path(path_like: Optional[str]) -> Optional[Path]:
    if not path_like:
        return None
    return normalize_path(path_like)


def should_materialize_output(args: argparse.Namespace) -> bool:
    if getattr(args, "fetch_output", False):
        return True
    if getattr(args, "save_output_path", None):
        return True
    if getattr(args, "compare_output_path", None):
        return True
    if args.device.lower() == "musa" and args.xla:
        return False
    return True


def maybe_save_output(result: Optional[np.ndarray], args: argparse.Namespace) -> Optional[Path]:
    save_path = optional_output_path(getattr(args, "save_output_path", None))
    if result is None or save_path is None:
        return None

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as file_handle:
        np.save(file_handle, result)
    print(f"[Output] Saved tensor to: {save_path}")
    return save_path


def maybe_compare_output(result: Optional[np.ndarray], args: argparse.Namespace) -> None:
    compare_path = optional_output_path(getattr(args, "compare_output_path", None))
    if result is None or compare_path is None:
        return

    if not compare_path.exists():
        raise FileNotFoundError(f"Compare output file not found: {compare_path}")

    with open(compare_path, "rb") as file_handle:
        reference = np.load(file_handle, allow_pickle=False)

    if reference.shape != result.shape:
        raise RuntimeError(
            f"Output shape mismatch: current={result.shape}, reference={reference.shape}"
        )
    if reference.dtype != result.dtype:
        print(f"[Compare] Dtype differs: current={result.dtype}, reference={reference.dtype}")

    print("\n[Output Compare]")
    print(f"  Reference: {compare_path}")
    print(f"  Current:   shape={result.shape}, dtype={result.dtype}")
    print(f"  Baseline:  shape={reference.shape}, dtype={reference.dtype}")

    if np.issubdtype(result.dtype, np.number) and np.issubdtype(reference.dtype, np.number):
        current_fp64 = result.astype(np.float64, copy=False)
        reference_fp64 = reference.astype(np.float64, copy=False)
        abs_diff = np.abs(current_fp64 - reference_fp64)
        allowed_diff = args.compare_atol + args.compare_rtol * np.abs(reference_fp64)
        mismatch_mask = abs_diff > allowed_diff
        max_abs_diff = float(np.max(abs_diff)) if abs_diff.size else 0.0
        mean_abs_diff = float(np.mean(abs_diff)) if abs_diff.size else 0.0
        allclose = not bool(np.any(mismatch_mask))
        print(f"  atol:      {args.compare_atol}")
        print(f"  rtol:      {args.compare_rtol}")
        print(f"  max_abs:   {max_abs_diff:.8e}")
        print(f"  mean_abs:  {mean_abs_diff:.8e}")
        print(f"  allclose:  {allclose}")
        if not allclose:
            diff_excess = abs_diff - allowed_diff
            max_index = int(np.argmax(diff_excess)) if diff_excess.size else 0
            mismatch_index = np.unravel_index(max_index, diff_excess.shape) if diff_excess.size else ()
            mismatch_count = int(np.count_nonzero(mismatch_mask))
            raise RuntimeError(
                "Output mismatch detected: "
                f"mismatch_count={mismatch_count}, index={mismatch_index}, "
                f"abs_diff={abs_diff[mismatch_index]:.8e}, "
                f"allowed_diff={allowed_diff[mismatch_index]:.8e}, "
                f"current={result[mismatch_index]}, reference={reference[mismatch_index]}"
            )
        return

    equal = np.array_equal(result, reference)
    print(f"  exact_equal: {equal}")
    if not equal:
        raise RuntimeError("Output mismatch detected for non-numeric tensor")


def run_inference(tf, graph_def, feed_dict, args: argparse.Namespace):
    tensor_name = format_output_tensor_name(args.output_node)
    device_name = runtime_device_name(args)
    allow_soft_placement = args.allow_soft_placement or args.device.lower() in {"cpu", "cuda", "musa"}

    print("\n=== 执行图推理 ===")
    print(f"输出张量: {tensor_name}")
    print(f"目标设备: {device_name}")
    print(f"XLA: {args.xla}")
    print(f"Allow Soft Placement: {allow_soft_placement}")
    print(f"预热次数: {args.warmup_runs}, 正式运行次数: {args.num_runs}")

    with tf.Graph().as_default() as graph:
        import_start = time.time()
        import_graph_with_runtime_scope(
            tf,
            graph_def,
            device_name,
            args.xla,
            args.device,
        )
        import_end = time.time()
        print(f"[时间] 图导入耗时: {(import_end - import_start) * 1000:.2f} ms")

        feed_start = time.time()
        session_feed_dict = {}
        for name, data in feed_dict.items():
            try:
                tensor = graph.get_tensor_by_name(name)
            except KeyError:
                continue
            session_feed_dict[tensor] = data

        try:
            output_tensor = graph.get_tensor_by_name(tensor_name)
        except KeyError as exc:
            raise RuntimeError(f"找不到输出张量 {tensor_name}") from exc

        feed_end = time.time()
        print(f"[时间] Feed Dict 准备耗时: {(feed_end - feed_start) * 1000:.2f} ms")

        materialize_output = should_materialize_output(args)
        run_fetch = output_tensor if materialize_output else output_tensor.op
        print(f"Fetch Mode: {'tensor' if materialize_output else 'op-only'}")
        if not materialize_output:
            print(">>>> [MUSA] Skip host-side tensor materialization during benchmark")

        config = create_session_config(
            tf=tf,
            device_type=args.device,
            xla=args.xla,
            log_device_placement=args.log_device_placement,
            allow_soft_placement=allow_soft_placement,
            tf_interop_threads=args.tf_interop_threads,
            tf_intraop_threads=args.tf_intraop_threads,
        )

        session_start = time.time()
        with tf.Session(graph=graph, config=config) as sess:
            session_end = time.time()
            print(f"[时间] Session 创建耗时: {(session_end - session_start) * 1000:.2f} ms")

            print(">>>> [Session] 可见设备:")
            for device in sess.list_devices():
                print(f"  - {device.name} [{device.device_type}]")

            print(f">>> 预热运行 {args.warmup_runs} 次...")
            for warmup_idx in range(args.warmup_runs):
                iter_start = time.time()
                sess.run(run_fetch, feed_dict=session_feed_dict)
                iter_end = time.time()
            print(">>> 预热完成")

            print(f">>> 正式运行 {args.num_runs} 次...")
            run_times = []
            result = None
            for run_idx in range(args.num_runs):
                run_start = time.time()
                current = sess.run(run_fetch, feed_dict=session_feed_dict)
                run_end = time.time()
                elapsed_ms = (run_end - run_start) * 1000.0
                run_times.append(elapsed_ms)
                if materialize_output:
                    result = current

    total_time = sum(run_times)
    avg_time = total_time / args.num_runs
    min_time = min(run_times)
    max_time = max(run_times)
    p50 = np.percentile(run_times, 50)
    p95 = np.percentile(run_times, 95)
    p99 = np.percentile(run_times, 99)
    runs_per_second = 1000.0 / avg_time
    samples_per_second = (args.batchsize * 1000.0) / avg_time
    avg_time_per_sample_us = (avg_time * 1000.0) / args.batchsize

    print("\n" + "=" * 50)
    print("[性能统计]")
    print("=" * 50)
    print(f"  运行次数: {args.num_runs}")
    print(f"  总耗时:   {total_time:.2f} ms")
    print(f"  平均:     {avg_time:.4f} ms")
    print(f"  最小:     {min_time:.4f} ms")
    print(f"  最大:     {max_time:.4f} ms")
    print(f"  P50:      {p50:.4f} ms")
    print(f"  P95:      {p95:.4f} ms")
    print(f"  P99:      {p99:.4f} ms")
    print(f"  吞吐量:   {runs_per_second:.2f} 次/秒")
    print(f"  样本吞吐: {samples_per_second:.2f} 样本/秒")
    print(f"  单样本:   {avg_time_per_sample_us:.2f} us/样本")
    print("=" * 50)

    if result is None:
        print("[推理结果] Output tensor materialization was skipped for this run.")
        return None

    print("\n[推理结果统计]")
    print(f"  Shape: {result.shape}")
    print(f"  Dtype: {result.dtype}")
    if result.size > 0 and np.issubdtype(result.dtype, np.number):
        print(f"  Min:   {np.min(result):.4f}")
        print(f"  Max:   {np.max(result):.4f}")
        print(f"  Mean:  {np.mean(result):.4f}")
    else:
        print("  数据摘要: <empty or non-numeric>")

    if result.size <= 20:
        print(f"  Data: {result}")

    maybe_save_output(result, args)
    maybe_compare_output(result, args)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TensorFlow 图推理性能测试工具")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "musa"],
        help="运行设备: cpu, cuda, musa (默认: cpu)",
    )
    parser.add_argument(
        "--device_index",
        type=int,
        default=0,
        help="目标设备编号 (默认: 0)",
    )
    parser.add_argument(
        "--graph_path",
        type=str,
        default=str(DEFAULT_GRAPH_PATH),
        help=f"GraphDef 路径 (默认: {DEFAULT_GRAPH_PATH})",
    )
    parser.add_argument(
        "--output_node",
        type=str,
        default=DEFAULT_OUTPUT_NODE,
        help=f"输出节点名称 (默认: {DEFAULT_OUTPUT_NODE})",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=100,
        help="输入数据的 batch size (默认: 100)",
    )
    parser.add_argument(
        "--xla",
        action="store_true",
        help="启用 XLA JIT，并在导图时附加 jit_scope",
    )
    parser.add_argument(
        "--fetch_output",
        action="store_true",
        help="在跑分时真正取回输出 tensor，而不只是执行 op",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1000,
        help="正式运行次数 (默认: 1000)",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=10,
        help="预热运行次数 (默认: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260421,
        help="Mock 数据随机种子 (默认: 20260421)",
    )
    parser.add_argument(
        "--musa_plugin_path",
        type=str,
        default=str(DEFAULT_MUSA_PLUGIN_PATH),
        help=f"MUSA PJRT 插件路径 (默认: {DEFAULT_MUSA_PLUGIN_PATH})",
    )
    parser.add_argument(
        "--musa_bridge_path",
        type=str,
        default=str(DEFAULT_MUSA_BRIDGE_PATH),
        help=f"MUSA TF2.15 runtime bridge 路径 (默认: {DEFAULT_MUSA_BRIDGE_PATH})",
    )
    parser.add_argument(
        "--allow_soft_placement",
        action="store_true",
        help="允许 TensorFlow 在目标设备不可用时回退到其他设备",
    )
    parser.add_argument(
        "--log_device_placement",
        action="store_true",
        help="打印 TensorFlow 的 op 放置日志",
    )
    parser.add_argument(
        "--list_devices",
        action="store_true",
        help="完成 runtime bootstrap 后仅列出当前 TensorFlow 可见设备并退出",
    )
    parser.add_argument(
        "--save_output_path",
        type=str,
        default=None,
        help="Save fetched output tensor to a .npy file",
    )
    parser.add_argument(
        "--compare_output_path",
        type=str,
        default=None,
        help="Compare fetched output tensor with a baseline .npy file",
    )
    parser.add_argument(
        "--compare_atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for output comparison",
    )
    parser.add_argument(
        "--compare_rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for output comparison",
    )
    parser.add_argument(
        "--tf_interop_threads",
        type=int,
        default=1,
        help="TensorFlow inter-op thread count for MUSA+XLA",
    )
    parser.add_argument(
        "--tf_intraop_threads",
        type=int,
        default=1,
        help="TensorFlow intra-op thread count for MUSA+XLA",
    )
    parser.add_argument(
        "--musa_pjrt_max_inflight_compiles",
        type=int,
        default=1,
        help="Plugin-side max in-flight compile calls",
    )
    parser.add_argument(
        "--musa_pjrt_max_inflight_transfers",
        type=int,
        default=1,
        help="Plugin-side max in-flight host-to-device transfers",
    )
    parser.add_argument(
        "--musa_pjrt_max_inflight_executes",
        type=int,
        default=1,
        help="Plugin-side max in-flight execute calls",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_path = normalize_path(args.graph_path)

    print("=" * 50)
    print("参数配置")
    print("=" * 50)
    print(f"  设备:               {args.device.upper()}")
    print(f"  设备编号:           {args.device_index}")
    print(f"  Graph 路径:         {graph_path}")
    print(f"  输出节点:           {args.output_node}")
    print(f"  Batch Size:         {args.batchsize}")
    print(f"  XLA:                {args.xla}")
    print(f"  Fetch Output:       {args.fetch_output}")
    print(f"  TF InterOp:         {args.tf_interop_threads}")
    print(f"  TF IntraOp:         {args.tf_intraop_threads}")
    print(f"  Max Compiles:       {args.musa_pjrt_max_inflight_compiles}")
    print(f"  Max Transfers:      {args.musa_pjrt_max_inflight_transfers}")
    print(f"  Max Executes:       {args.musa_pjrt_max_inflight_executes}")
    print(f"  Save Output Path:   {args.save_output_path}")
    print(f"  Compare Output:     {args.compare_output_path}")
    print(f"  运行次数:           {args.num_runs}")
    print(f"  预热次数:           {args.warmup_runs}")
    print(f"  Random Seed:        {args.seed}")
    print(f"  Repo Root:          {REPO_ROOT}")
    print("=" * 50)

    total_start = time.time()

    prepare_runtime_env(args)
    tf, graph_pb2 = import_tensorflow_v1()

    if args.device.lower() == "musa":
        bootstrap_musa_runtime(tf, args)

    print_visible_devices(tf)
    if args.device.lower() != "cpu":
        ensure_target_device_visible(tf, args.device, args.device_index)

    if args.list_devices:
        return

    graph_load_start = time.time()
    graph_def, placeholders = load_graph_and_get_placeholders(tf, graph_pb2, graph_path)
    print(f"[时间] 图加载与分析耗时: {(time.time() - graph_load_start) * 1000:.2f} ms")
    if not placeholders:
        raise RuntimeError("未找到 Placeholder，无法构造 feed_dict")

    mock_start = time.time()
    feed_dict = create_mock_data(placeholders, args.batchsize, args.seed)
    print(f"[时间] Mock 数据创建耗时: {(time.time() - mock_start) * 1000:.2f} ms")

    run_inference(tf, graph_def, feed_dict, args)

    total_end = time.time()
    print(f"\n[总耗时] {(total_end - total_start) * 1000:.2f} ms")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n!!!! 脚本执行失败 !!!!")
        print(f"错误信息: {exc}")
        traceback.print_exc()
        sys.exit(1)
