import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
LINUX_REPO_ROOT = Path("/workspace/openxla")
LINUX_GRAPH_PATH = LINUX_REPO_ROOT / "pjrt_plugin/musa_test/musa_model/graph_def.pb"
LINUX_PLUGIN_PATH = LINUX_REPO_ROOT / "bazel-bin/pjrt_plugin/libmusa_pjrt_plugin_zy.so"
LINUX_BRIDGE_PATH = LINUX_REPO_ROOT / "bazel-bin/pjrt_plugin/libmusa_tf215_registry_bridge.so"


def find_repo_root(path: Path) -> Path:
    for parent in path.parents:
        if (parent / "WORKSPACE").exists() or (parent / "WORKSPACE.bazel").exists():
            return parent
    return path.parent


REPO_ROOT = LINUX_REPO_ROOT if LINUX_REPO_ROOT.exists() else find_repo_root(SCRIPT_PATH)
LOCAL_PLUGIN_PATH = REPO_ROOT / "bazel-bin/pjrt_plugin/libmusa_pjrt_plugin_zy.so"
LOCAL_BRIDGE_PATH = REPO_ROOT / "bazel-bin/pjrt_plugin/libmusa_tf215_registry_bridge.so"

DEFAULT_GRAPH_PATH = LINUX_GRAPH_PATH if LINUX_GRAPH_PATH.exists() else SCRIPT_DIR / "graph_def.pb"
DEFAULT_MUSA_PLUGIN_PATH = LINUX_PLUGIN_PATH if LINUX_PLUGIN_PATH.exists() else LOCAL_PLUGIN_PATH
DEFAULT_MUSA_BRIDGE_PATH = LINUX_BRIDGE_PATH if LINUX_BRIDGE_PATH.exists() else LOCAL_BRIDGE_PATH

PROFILES = [
    "legacy",
    "legacy_xla_env",
    "legacy_tf215_config_no_limits",
    "legacy_inflight_only",
    "legacy_tf_threads_only",
    "legacy_limits",
    "legacy_global",
    "legacy_tf215_config",
    "tf215_nobridge",
    "tf215",
]


CHILD_CODE = r'''
import argparse
import ctypes
import faulthandler
import os
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

import numpy as np


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def normalize_path(path_like):
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def append_env_flag(env_name, flag):
    current = os.environ.get(env_name, "").strip()
    if flag in current:
        return
    os.environ[env_name] = f"{current} {flag}".strip() if current else flag


def load_dynamic_library(path, global_symbols):
    mode = getattr(ctypes, "RTLD_GLOBAL", None) if global_symbols else None
    if mode is None:
        return ctypes.CDLL(str(path))
    return ctypes.CDLL(str(path), mode=mode)


def use_tf215_import(profile):
    return profile in {"tf215", "tf215_nobridge"}


def use_xla_env(profile):
    return profile in {
        "legacy_xla_env",
        "legacy_tf215_config_no_limits",
        "legacy_inflight_only",
        "legacy_tf_threads_only",
        "legacy_limits",
        "legacy_global",
        "legacy_tf215_config",
        "tf215_nobridge",
        "tf215",
    }


def use_tf_thread_limits(profile):
    return profile in {
        "legacy_tf_threads_only",
        "legacy_limits",
        "legacy_global",
        "legacy_tf215_config",
        "tf215_nobridge",
        "tf215",
    }


def use_pjrt_inflight_limits(profile):
    return profile in {
        "legacy_inflight_only",
        "legacy_limits",
        "legacy_global",
        "legacy_tf215_config",
        "tf215_nobridge",
        "tf215",
    }


def use_rtld_global(profile):
    return profile in {"legacy_global", "legacy_tf215_config", "tf215_nobridge", "tf215"}


def use_tf215_config(profile):
    return profile in {
        "legacy_tf215_config_no_limits",
        "legacy_tf215_config",
        "tf215_nobridge",
        "tf215",
    }


def use_bridge(profile):
    return profile == "tf215"


def prepare_env(args):
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    if args.device == "musa":
        plugin_path = normalize_path(args.musa_plugin_path)
        os.environ["MUSA_VISIBLE_DEVICES"] = str(args.device_index)
        os.environ["TF_PLUGGABLE_DEVICE_LIBRARY_PATH"] = str(plugin_path)
        os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = f"MUSA:{plugin_path}"

        if use_tf_thread_limits(args.profile):
            os.environ["TF_NUM_INTEROP_THREADS"] = str(args.tf_interop_threads)
            os.environ["TF_NUM_INTRAOP_THREADS"] = str(args.tf_intraop_threads)

        if use_pjrt_inflight_limits(args.profile):
            os.environ["MUSA_PJRT_MAX_INFLIGHT_COMPILES"] = str(args.musa_pjrt_max_inflight_compiles)
            os.environ["MUSA_PJRT_MAX_INFLIGHT_TRANSFERS"] = str(args.musa_pjrt_max_inflight_transfers)
            os.environ["MUSA_PJRT_MAX_INFLIGHT_EXECUTES"] = str(args.musa_pjrt_max_inflight_executes)

        if args.musa_pjrt_preallocate:
            os.environ["MUSA_PJRT_PREALLOCATE"] = args.musa_pjrt_preallocate
        if args.musa_pjrt_memory_fraction:
            os.environ["MUSA_PJRT_MEMORY_FRACTION"] = args.musa_pjrt_memory_fraction
        if args.musa_pjrt_gpu_system_memory_size_mb:
            os.environ["MUSA_PJRT_GPU_SYSTEM_MEMORY_SIZE_MB"] = args.musa_pjrt_gpu_system_memory_size_mb

    if args.xla and use_xla_env(args.profile):
        append_env_flag("TF_XLA_FLAGS", "--tf_xla_use_device_api=true")
        append_env_flag("TF_XLA_FLAGS", "--tf_xla_use_device_api_for_auto_jit=true")
        append_env_flag("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")
        append_env_flag("TF_XLA_FLAGS", "--tf_xla_min_cluster_size=1")


def print_env(args):
    log(f"PROFILE {args.profile}")
    for name in [
        "TF_ENABLE_ONEDNN_OPTS",
        "MUSA_VISIBLE_DEVICES",
        "TF_PLUGGABLE_DEVICE_LIBRARY_PATH",
        "PJRT_NAMES_AND_LIBRARY_PATHS",
        "TF_NUM_INTEROP_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "MUSA_PJRT_MAX_INFLIGHT_COMPILES",
        "MUSA_PJRT_MAX_INFLIGHT_TRANSFERS",
        "MUSA_PJRT_MAX_INFLIGHT_EXECUTES",
        "MUSA_PJRT_PREALLOCATE",
        "MUSA_PJRT_MEMORY_FRACTION",
        "MUSA_PJRT_GPU_SYSTEM_MEMORY_SIZE_MB",
        "TF_XLA_FLAGS",
    ]:
        value = os.environ.get(name)
        if value:
            log(f"ENV {name}={value}")


def import_tensorflow(args):
    log("STEP_BEGIN import_tensorflow")
    if use_tf215_import(args.profile):
        import tensorflow.compat.v1 as tf
        from tensorflow.core.framework import graph_pb2

        tf.disable_v2_behavior()
        tf.disable_eager_execution()
        tf_v1 = tf
        log("IMPORT_MODE tf.compat.v1 disable_v2_behavior")
    else:
        import tensorflow as tf
        import tensorflow.compat.v1 as tf_v1
        from tensorflow.core.framework import graph_pb2

        tf_v1.disable_eager_execution()
        log("IMPORT_MODE legacy tensorflow + compat.v1 disable_eager_only")

    log(f"STEP_END import_tensorflow version={tf.__version__}")
    return tf, tf_v1, graph_pb2


def maybe_call_symbol(library, symbol_name, restype=None, argtypes=None):
    symbol = getattr(library, symbol_name, None)
    if symbol is None:
        log(f"SYMBOL_MISSING {symbol_name}")
        return None
    if argtypes is not None:
        symbol.argtypes = list(argtypes)
    if restype is not None:
        symbol.restype = restype
    log(f"STEP_BEGIN call_symbol {symbol_name}")
    result = symbol()
    log(f"STEP_END call_symbol {symbol_name} result={result}")
    return result


def bootstrap_musa(tf, args):
    if args.device != "musa":
        return

    plugin_path = normalize_path(args.musa_plugin_path)
    if not plugin_path.exists():
        raise FileNotFoundError(f"MUSA plugin not found: {plugin_path}")

    log(f"STEP_BEGIN load_plugin_cdll global={use_rtld_global(args.profile)}")
    plugin_lib = load_dynamic_library(plugin_path, use_rtld_global(args.profile))
    log("STEP_END load_plugin_cdll")

    maybe_call_symbol(plugin_lib, "ForceRegisterMusa")

    if use_bridge(args.profile):
        bridge_path = normalize_path(args.musa_bridge_path)
        if not bridge_path.exists():
            log(f"WARN bridge not found: {bridge_path}")
        else:
            log("STEP_BEGIN load_bridge_cdll")
            bridge_lib = load_dynamic_library(bridge_path, True)
            log("STEP_END load_bridge_cdll")
            maybe_call_symbol(
                bridge_lib,
                "MusaTf215_RegisterRuntimeFactory",
                restype=ctypes.c_int,
                argtypes=[],
            )

    log("STEP_BEGIN list_devices_after_bootstrap")
    try:
        if use_tf215_import(args.profile):
            devices = tf.config.list_logical_devices()
        else:
            devices = tf.config.list_physical_devices("MUSA")
        for device in devices:
            log(f"DEVICE {device}")
        log(f"STEP_END list_devices_after_bootstrap count={len(devices)}")
    except Exception as exc:
        log(f"WARN list devices failed: {exc}")


def infer_placeholder_shape_from_usage(graph_def, placeholder_name):
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


def load_graph_and_get_placeholders(tf, graph_pb2, args):
    graph_path = normalize_path(args.graph_path)
    log(f"STEP_BEGIN read_graph {graph_path}")
    with tf.io.gfile.GFile(str(graph_path), "rb") as file_handle:
        graph_bytes = file_handle.read()
    log(f"STEP_END read_graph bytes={len(graph_bytes)}")

    log("STEP_BEGIN parse_graphdef")
    graph_def = graph_pb2.GraphDef()
    graph_def.ParseFromString(graph_bytes)
    log(f"STEP_END parse_graphdef nodes={len(graph_def.node)}")

    dtype_map = {
        tf.float32.as_datatype_enum: np.float32,
        tf.int32.as_datatype_enum: np.int32,
        tf.int64.as_datatype_enum: np.int64,
        tf.bool.as_datatype_enum: np.bool_,
        tf.string.as_datatype_enum: np.str_,
    }

    placeholders = {}
    for node in graph_def.node:
        if node.op != "Placeholder":
            continue

        dtype = dtype_map.get(node.attr["dtype"].type, np.float32)
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

    log(f"PLACEHOLDERS count={len(placeholders)}")
    return graph_def, placeholders


def create_mock_data(placeholders, args):
    log("STEP_BEGIN create_mock_data")
    if args.seed >= 0:
        np.random.seed(args.seed)

    feed_dict = {}
    for name, info in placeholders.items():
        dtype = info["dtype"]
        mock_shape = []
        for dim in info["shape"]:
            if dim is None or dim == -1:
                mock_shape.append(args.batchsize)
            elif dim == 0 and args.legacy_zero_dim:
                mock_shape.append(0)
            elif dim == 0:
                mock_shape.append(args.batchsize)
            else:
                mock_shape.append(dim)

        if not mock_shape and "/ReadVariableOp/resource" in name:
            if "BiasAdd" in name:
                mock_shape = [32]
            elif "MatMul" in name or "Tensordot" in name:
                mock_shape = [64, 32]

        if dtype == np.float32:
            data = np.random.normal(0.0, 1.0, mock_shape).astype(dtype)
        elif dtype == np.int32:
            data = np.random.randint(0, 100, mock_shape).astype(dtype)
        elif dtype == np.int64:
            data = np.random.randint(0, 100, mock_shape).astype(dtype)
        elif dtype == np.bool_:
            data = np.random.choice([True, False], mock_shape).astype(dtype)
        elif dtype == np.str_:
            data = np.full(mock_shape if mock_shape else (), "mock", dtype=np.str_)
        else:
            data = np.random.normal(0.0, 1.0, mock_shape).astype(np.float32)

        feed_dict[f"{name}:0"] = data

    log(f"STEP_END create_mock_data feeds={len(feed_dict)}")
    return feed_dict


def device_name(args):
    if args.device == "cpu":
        return "/device:CPU:0"
    if args.device == "cuda":
        return f"/device:GPU:{args.device_index}"
    return "/device:MUSA:0"


def import_graph(tf, graph_def, args):
    graph = tf.Graph()
    with graph.as_default():
        log(f"STEP_BEGIN import_graph device={device_name(args)}")
        with ExitStack() as stack:
            stack.enter_context(tf.device(device_name(args)))
            if args.xla and args.device == "cuda":
                stack.enter_context(tf.xla.experimental.jit_scope(compile_ops=True))
            tf.import_graph_def(graph_def, name="")
        log("STEP_END import_graph")
    return graph


def make_session_feed_dict(graph, feed_dict):
    result = {}
    for name, data in feed_dict.items():
        try:
            result[graph.get_tensor_by_name(name)] = data
        except KeyError:
            log(f"WARN feed tensor not found: {name}")
    return result


def output_tensor(tf_graph, args):
    name = args.output_node if ":" in args.output_node else f"{args.output_node}:0"
    log(f"STEP_BEGIN resolve_output {name}")
    tensor = tf_graph.get_tensor_by_name(name)
    log(f"STEP_END resolve_output {name}")
    return tensor


def create_session_config(tf, tf_v1, args):
    config = tf_v1.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = args.log_device_placement

    if args.device == "cuda":
        config.gpu_options.allow_growth = True
        if args.xla:
            config.graph_options.optimizer_options.global_jit_level = tf_v1.OptimizerOptions.ON_1

    if args.device == "musa":
        rewrite_options = config.graph_options.rewrite_options
        if use_tf215_config(args.profile):
            from tensorflow.core.protobuf import rewriter_config_pb2

            if use_tf_thread_limits(args.profile):
                config.inter_op_parallelism_threads = args.tf_interop_threads
                config.intra_op_parallelism_threads = args.tf_intraop_threads
                log("CONFIG_THREAD_LIMITS on")
            else:
                log("CONFIG_THREAD_LIMITS off")
            rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
            rewrite_options.remapping = rewriter_config_pb2.RewriterConfig.OFF
            if args.xla:
                config.graph_options.optimizer_options.global_jit_level = tf_v1.OptimizerOptions.ON_2
            log("CONFIG_MODE tf215")
        else:
            log("CONFIG_MODE legacy")

        rewrite_options.custom_optimizers.add().name = "musa_graph_optimizer"

    return config


def summarize_times(label, times):
    if not times:
        return
    arr = np.array(times, dtype=np.float64)
    log(
        f"SUMMARY {label} count={len(times)} avg_ms={np.mean(arr):.3f} "
        f"min_ms={np.min(arr):.3f} max_ms={np.max(arr):.3f} "
        f"p50_ms={np.percentile(arr, 50):.3f} p95_ms={np.percentile(arr, 95):.3f}"
    )
    if len(times) > 1:
        tail = arr[1:]
        log(
            f"SUMMARY {label}_skip_first count={len(tail)} avg_ms={np.mean(tail):.3f} "
            f"min_ms={np.min(tail):.3f} max_ms={np.max(tail):.3f} "
            f"p50_ms={np.percentile(tail, 50):.3f} p95_ms={np.percentile(tail, 95):.3f}"
        )


def run_loop(sess, fetch, feed_dict, count, label):
    times = []
    result = None
    for idx in range(count):
        log(f"ITER_BEGIN {label} {idx}")
        started = time.perf_counter()
        result = sess.run(fetch, feed_dict=feed_dict)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        times.append(elapsed_ms)
        log(f"ITER_END {label} {idx} elapsed_ms={elapsed_ms:.3f}")
    summarize_times(label, times)
    return result


def run(args):
    faulthandler.enable()
    if args.stack_dump_after > 0:
        faulthandler.dump_traceback_later(args.stack_dump_after, repeat=True)

    log("CHILD_BEGIN")
    prepare_env(args)
    print_env(args)
    tf, tf_v1, graph_pb2 = import_tensorflow(args)
    bootstrap_musa(tf, args)

    graph_def, placeholders = load_graph_and_get_placeholders(tf, graph_pb2, args)
    feeds_by_name = create_mock_data(placeholders, args)
    graph = import_graph(tf, graph_def, args)

    with graph.as_default():
        session_feed_dict = make_session_feed_dict(graph, feeds_by_name)
        tensor = output_tensor(graph, args)
        fetch = tensor.op if args.fetch_mode == "op" else tensor

    config = create_session_config(tf, tf_v1, args)
    log("STEP_BEGIN session_create")
    sess = tf_v1.Session(graph=graph, config=config)
    log("STEP_END session_create")

    try:
        log("STEP_BEGIN session_list_devices")
        for dev in sess.list_devices():
            log(f"SESSION_DEVICE {dev.name} [{dev.device_type}]")
        log("STEP_END session_list_devices")

        run_loop(sess, fetch, session_feed_dict, args.warmup_runs, "warmup")
        result = run_loop(sess, fetch, session_feed_dict, args.num_runs, "run")

        if args.fetch_mode == "tensor" and result is not None:
            log(f"RESULT shape={getattr(result, 'shape', None)} dtype={getattr(result, 'dtype', None)}")
            if hasattr(result, "size") and result.size and np.issubdtype(result.dtype, np.number):
                log(f"RESULT min={np.min(result):.6f} max={np.max(result):.6f} mean={np.mean(result):.6f}")
    finally:
        log("STEP_BEGIN session_close")
        sess.close()
        log("STEP_END session_close")

    log("CHILD_END")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "musa"], default="musa")
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--graph_path", required=True)
    parser.add_argument("--output_node", default="predicts")
    parser.add_argument("--batchsize", type=int, default=100)
    parser.add_argument("--xla", action="store_true")
    parser.add_argument("--fetch_mode", choices=["tensor", "op"], default="tensor")
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--legacy_zero_dim", action="store_true")
    parser.add_argument("--musa_plugin_path", required=True)
    parser.add_argument("--musa_bridge_path", required=True)
    parser.add_argument("--log_device_placement", action="store_true")
    parser.add_argument("--tf_interop_threads", type=int, default=1)
    parser.add_argument("--tf_intraop_threads", type=int, default=1)
    parser.add_argument("--musa_pjrt_max_inflight_compiles", type=int, default=1)
    parser.add_argument("--musa_pjrt_max_inflight_transfers", type=int, default=1)
    parser.add_argument("--musa_pjrt_max_inflight_executes", type=int, default=1)
    parser.add_argument("--musa_pjrt_preallocate", default="")
    parser.add_argument("--musa_pjrt_memory_fraction", default="")
    parser.add_argument("--musa_pjrt_gpu_system_memory_size_mb", default="")
    parser.add_argument("--stack_dump_after", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as exc:
        log(f"CHILD_ERROR {exc}")
        traceback.print_exc()
        sys.exit(1)
'''


def split_profiles(value: str) -> List[str]:
    profiles = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in profiles if item not in PROFILES]
    if unknown:
        raise SystemExit(f"Unknown profile(s): {', '.join(unknown)}")
    return profiles


def add_child_args(command: List[str], args: argparse.Namespace, profile: str) -> None:
    command.extend(
        [
            "--profile",
            profile,
            "--device",
            args.device,
            "--device_index",
            str(args.device_index),
            "--graph_path",
            str(Path(args.graph_path).expanduser()),
            "--output_node",
            args.output_node,
            "--batchsize",
            str(args.batchsize),
            "--fetch_mode",
            args.fetch_mode,
            "--warmup_runs",
            str(args.warmup_runs),
            "--num_runs",
            str(args.num_runs),
            "--seed",
            str(args.seed),
            "--musa_plugin_path",
            str(Path(args.musa_plugin_path).expanduser()),
            "--musa_bridge_path",
            str(Path(args.musa_bridge_path).expanduser()),
            "--tf_interop_threads",
            str(args.tf_interop_threads),
            "--tf_intraop_threads",
            str(args.tf_intraop_threads),
            "--musa_pjrt_max_inflight_compiles",
            str(args.musa_pjrt_max_inflight_compiles),
            "--musa_pjrt_max_inflight_transfers",
            str(args.musa_pjrt_max_inflight_transfers),
            "--musa_pjrt_max_inflight_executes",
            str(args.musa_pjrt_max_inflight_executes),
            "--musa_pjrt_preallocate",
            str(args.musa_pjrt_preallocate),
            "--musa_pjrt_memory_fraction",
            str(args.musa_pjrt_memory_fraction),
            "--musa_pjrt_gpu_system_memory_size_mb",
            str(args.musa_pjrt_gpu_system_memory_size_mb),
            "--stack_dump_after",
            str(args.stack_dump_after),
        ]
    )
    if args.xla:
        command.append("--xla")
    if args.legacy_zero_dim:
        command.append("--legacy_zero_dim")
    if args.log_device_placement:
        command.append("--log_device_placement")


def stream_output(process: subprocess.Popen, lines: List[str]) -> threading.Thread:
    def reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            print(line, end="")

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    return thread


def run_profile(profile: str, args: argparse.Namespace) -> bool:
    command = [sys.executable, "-u", "-c", CHILD_CODE]
    add_child_args(command, args, profile)

    print("\n" + "=" * 80)
    print(f"RUN_PROFILE {profile}")
    print("=" * 80, flush=True)

    started = time.perf_counter()
    output_lines: List[str] = []
    process = subprocess.Popen(
        command,
        cwd=str(SCRIPT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    reader_thread = stream_output(process, output_lines)

    timed_out = False
    while process.poll() is None:
        if time.perf_counter() - started > args.timeout:
            timed_out = True
            print(f"\nPARENT_TIMEOUT profile={profile} after {args.timeout:.1f}s", flush=True)
            process.kill()
            break
        time.sleep(0.2)

    return_code = process.wait()
    reader_thread.join(timeout=2.0)
    elapsed = time.perf_counter() - started

    if timed_out:
        print(f"PROFILE_RESULT {profile} TIMEOUT elapsed_s={elapsed:.2f}")
        print("Last child lines before timeout:")
        for line in output_lines[-8:]:
            print(line, end="")
        return False

    if return_code != 0:
        print(f"PROFILE_RESULT {profile} FAILED rc={return_code} elapsed_s={elapsed:.2f}")
        return False

    print(f"PROFILE_RESULT {profile} OK elapsed_s={elapsed:.2f}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A/B probe for old graph_runner.py versus graph_runner_tf215.py behavior."
    )
    parser.add_argument(
        "--profiles",
        default=(
            "legacy,legacy_xla_env,legacy_tf215_config_no_limits,"
            "legacy_inflight_only,legacy_tf_threads_only,legacy_limits,tf215"
        ),
        help=f"Comma-separated profiles. Available: {', '.join(PROFILES)}",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "musa"], default="musa")
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--graph_path", default=str(DEFAULT_GRAPH_PATH))
    parser.add_argument("--output_node", default="predicts")
    parser.add_argument("--batchsize", type=int, default=100)
    parser.add_argument("--xla", action="store_true")
    parser.add_argument("--fetch_mode", choices=["tensor", "op"], default="tensor")
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument(
        "--legacy_zero_dim",
        action="store_true",
        help="Keep graph_runner.py's dim==0 behavior when creating mock data.",
    )
    parser.add_argument("--musa_plugin_path", default=str(DEFAULT_MUSA_PLUGIN_PATH))
    parser.add_argument("--musa_bridge_path", default=str(DEFAULT_MUSA_BRIDGE_PATH))
    parser.add_argument("--log_device_placement", action="store_true")
    parser.add_argument("--tf_interop_threads", type=int, default=1)
    parser.add_argument("--tf_intraop_threads", type=int, default=1)
    parser.add_argument("--musa_pjrt_max_inflight_compiles", type=int, default=1)
    parser.add_argument("--musa_pjrt_max_inflight_transfers", type=int, default=1)
    parser.add_argument("--musa_pjrt_max_inflight_executes", type=int, default=1)
    parser.add_argument("--musa_pjrt_preallocate", default="")
    parser.add_argument("--musa_pjrt_memory_fraction", default="")
    parser.add_argument("--musa_pjrt_gpu_system_memory_size_mb", default="")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--stack_dump_after", type=int, default=30)
    parser.add_argument("--keep_going", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profiles = split_profiles(args.profiles)

    print("Diff probe configuration")
    print(f"  script_dir: {SCRIPT_DIR}")
    print(f"  profiles: {', '.join(profiles)}")
    print(f"  device: {args.device}:{args.device_index}")
    print(f"  xla: {args.xla}")
    print(f"  fetch_mode: {args.fetch_mode}")
    print(f"  warmup_runs: {args.warmup_runs}")
    print(f"  num_runs: {args.num_runs}")
    print(f"  timeout_per_profile_s: {args.timeout}")
    print(f"  graph_path: {args.graph_path}")
    print(f"  musa_plugin_path: {args.musa_plugin_path}")
    print(f"  musa_bridge_path: {args.musa_bridge_path}")

    failed: List[str] = []
    for profile in profiles:
        ok = run_profile(profile, args)
        if not ok:
            failed.append(profile)
            if not args.keep_going:
                break

    print("\n" + "=" * 80)
    if failed:
        print(f"OVERALL FAILED first_bad_profile={failed[0]} failed_profiles={', '.join(failed)}")
        return 1

    print("OVERALL OK all selected profiles completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
