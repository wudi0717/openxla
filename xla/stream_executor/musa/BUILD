"""MUSa-platform specific StreamExecutor support code."""

load(
    "@local_config_musa//musa:build_defs.bzl",
    "musa_library",
)
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/service/gpu:build_defs.bzl", "get_cub_sort_kernel_types")
load(
    "//xla/stream_executor:build_defs.bzl",
    "stream_executor_friends",
)
load("//xla/tests:build_defs.bzl", "xla_test")
load(
    "//xla/tsl:tsl.bzl",
    "internal_visibility",
)
load("//xla/tsl/platform:build_config_root.bzl", "if_static")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

cc_library(
    name = "musa_diagnostics",
    srcs = ["musa_diagnostics.cc"],
    hdrs = ["musa_diagnostics.h"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "musa_context",
    srcs = ["musa_context.cc"],
    hdrs = ["musa_context.h"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        ":musa_driver_wrapper",
        ":musa_status",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/gpu:context",
        "//xla/stream_executor/gpu:context_map",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_config_musa//musa:musa_headers",
    ],
)

cc_library(
    name = "musa_driver_wrapper",
    hdrs = ["musa_driver_wrapper.h"],
    defines = {"__MUSA_DISABLE_CPP_FUNCTIONS__": "1"},
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        "//xla/tsl/platform:env",
        "@local_config_musa//musa:musa",  # buildcleaner: keep
        "@local_config_musa//musa:musa_headers",
        "@tsl//tsl/platform:dso_loader",
    ],
)

cc_library(
    name = "musa_event",
    srcs = ["musa_event.cc"],
    hdrs = ["musa_event.h"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        ":musa_driver_wrapper",
        ":musa_status",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:event",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_musa//musa:musa_headers",
    ],
)

cc_library(
    name = "musa_executor",
    srcs = ["musa_executor.cc"],
    hdrs = ["musa_executor.h"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        ":musa_command_buffer",
        ":musa_context",
        ":musa_driver_wrapper",
        ":musa_event",
        ":musa_kernel",
        ":musa_platform_id",
        ":musa_status",
        ":musa_stream",
        ":musa_timer",
        ":musa_version_parser",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:command_buffer",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:event",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:fft",
        "//xla/stream_executor:generic_memory_allocation",
        "//xla/stream_executor:generic_memory_allocator",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:memory_allocator",
        "//xla/stream_executor:module_spec",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:plugin_registry",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:context",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor/gpu:read_numa_node",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "//xla/stream_executor/platform:initialize",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_config_musa//musa:musa_headers",
        "@tsl//tsl/platform:casts",
        "@tsl//tsl/platform:fingerprint",
        "@tsl//tsl/platform:numbers",
    ],
    alwayslink = True,
)

cc_library(
    name = "musa_kernel",
    srcs = ["musa_kernel.cc"],
    hdrs = ["musa_kernel.h"],
    tags = [
        "gpu",
        "musa-only",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":musa_driver_wrapper",
        ":musa_status",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_config_musa//musa:musa_headers",
    ],
)

cc_library(
    name = "musa_platform",
    srcs = ["musa_platform.cc"],
    hdrs = ["musa_platform.h"],
    tags = [
        "gpu",
        "musa-only",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":musa_diagnostics",
        ":musa_driver_wrapper",
        ":musa_executor",
        ":musa_platform_id",
        ":musa_status",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:executor_cache",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/platform:initialize",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_config_musa//musa:musa_headers",
    ],
    alwayslink = True,  # Registers itself with the PlatformManager.
)

cc_library(
    name = "musa_platform_id",
    srcs = ["musa_platform_id.cc"],
    hdrs = ["musa_platform_id.h"],
    deps = ["//xla/stream_executor:platform"],
)

cc_library(
    name = "all_runtime",
    tags = [
        "gpu",
        "musa-only",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":musa_platform",
    ],
    alwayslink = 1,
)

cc_library(
    name = "stream_executor_musa",
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        ":musa_platform_id",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/stream_executor/host:host_platform_id",
        "@local_config_musa//musa:musa_rpath",
    ] + if_static(
        [":all_runtime"],
    ),
)

cc_library(
    name = "musa_stream",
    srcs = ["musa_stream.cc"],
    hdrs = ["musa_stream.h"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        ":musa_driver_wrapper",
        ":musa_event",
        ":musa_kernel",
        ":musa_status",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:event",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_common",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@local_config_musa//musa:musa_headers",
    ],
)

cc_library(
    name = "musa_status",
    srcs = ["musa_status.cc"],
    hdrs = ["musa_status.h"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_config_musa//musa:musa_headers",
    ],
)

cc_library(
    name = "musa_command_buffer",
    srcs = [
        "musa_command_buffer.cc",
    ],
    hdrs = ["musa_command_buffer.h"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        ":musa_driver_wrapper",
        ":musa_kernel",
        ":musa_status",
        "//xla/stream_executor:bit_pattern",
        "//xla/stream_executor:command_buffer",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_command_buffer",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_config_musa//musa:musa_headers",
        "@tsl//tsl/platform:casts",
    ],
)

cc_library(
    name = "musa_timer",
    srcs = ["musa_timer.cc"],
    hdrs = ["musa_timer.h"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        ":musa_driver_wrapper",
        ":musa_event",
        ":musa_status",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@local_config_musa//musa:musa_headers",
    ],
)

cc_library(
    name = "musa_version_parser",
    srcs = ["musa_version_parser.cc"],
    hdrs = ["musa_version_parser.h"],
    deps = [
        "//xla/stream_executor:semantic_version",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)
