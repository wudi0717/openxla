load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@local_config_musa//musa:build_defs.bzl", "musa_version_number", "select_threshold")

licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

package(default_visibility = ["//visibility:private"])

string_flag(
    name = "musa_path_type",
    build_setting_default = "system",
    values = [
        "hermetic",
        "multiple",
        "system",
    ],
)

config_setting(
    name = "build_hermetic",
    flag_values = {
        ":musa_path_type": "hermetic",
    },
)

config_setting(
    name = "multiple_musa_paths",
    flag_values = {
        ":musa_path_type": "multiple",
    },
)

config_setting(
    name = "using_mcc",
    values = {
        "define": "using_musa_mcc=true",
    },
)

cc_library(
    name = "config",
    hdrs = [
        "musa_config/musa_config.h",
    ],
    include_prefix = "musa",
    strip_include_prefix = "musa_config",
)

cc_library(
    name = "config_hermetic",
    hdrs = [
        "musa_config_hermetic/musa_config.h",
    ],
    include_prefix = "musa",
    strip_include_prefix = "musa_config_hermetic",
)

cc_library(
    name = "musa_config",
    visibility = ["//visibility:public"],
    deps = select({
        ":build_hermetic": [
            ":config_hermetic",
        ],
        "//conditions:default": [
            "config",
        ],
    }),
)

# This target is required to
# add includes that are used by musa headers themself
# through the virtual includes
# cleaner solution would be to adjust the xla code
# and remove include prefix that is used to include musa headers.
cc_library(
    name = "musa_headers_includes",
    hdrs = glob([
        "%{musa_root}/include/**",
    ]),
    strip_include_prefix = "%{musa_root}/include",
)

cc_library(
    name = "musa_headers",
    hdrs = glob([
        "%{musa_root}/include/**",
    ]),
    strip_include_prefix = "%{musa_root}/include",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "musa",
    visibility = ["//visibility:public"],
    deps = [
        ":mublas",
        ":mublaslt",
        ":murand",
        ":musolver",
        ":musparse",
        ":musa_config",
        ":mufft",
    ]
)

cc_library(
    name = "musa_rpath",
    linkopts = select({
        ":build_hermetic": [
            "-Wl,-rpath,%{musa_toolkit_path}/lib",
        ],
        ":multiple_musa_paths": [
            "-Wl,-rpath=%{musa_lib_paths}",
        ],
        "//conditions:default": [
            "-Wl,-rpath,/usr/local/musa/lib",
        ],
    }),
    visibility = ["//visibility:public"],
)

# Used by jax_musa_plugin to minimally link to hip runtime.
cc_library(
    name = "musa_runtime",
    srcs = glob(["%{musa_root}/lib/libmusa*.so"]),
    hdrs = glob(["%{musa_root}/include/musa/**"]),
    include_prefix = "musa",
    includes = [
        "%{musa_root}/include",
    ],
    strip_include_prefix = "%{musa_root}",
    visibility = ["//visibility:public"],
    deps = [
        ":musa_config",
        ":system_libs",
    ],
)

cc_library(
    name = "mublas",
    hdrs = glob(["%{musa_root}/include/mublas*.h"]),
    data = glob([
        "%{musa_root}/lib/libmublas*.so*",
    ]),
    include_prefix = "musa",
    includes = [
        "%{musa_root}/include",
    ],
    # workaround to  bring tensile files to the same fs layout as expected in the lib
    # rocblas assumes that tensile files are located in ../roblas/libraries directory
    linkopts = ["-Wl,-rpath,local_config_musa/musa/musa_dis/lib"],
    strip_include_prefix = "%{musa_root}",
    visibility = ["//visibility:public"],
    deps = [":musa_config"],
)

cc_library(
    name = "mufft",
    srcs = glob(["%{musa_root}/lib/libmufft*.so*"]),
    include_prefix = "musa",
    includes = [
        "%{musa_root}/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [":musa_config"],
)

cc_library(
    name = "murand",
    srcs = glob(["%{musa_root}/lib/libmurand*.so*"]),
    hdrs = glob(["%{musa_root}/include/murand*.h"]),
    include_prefix = "musa",
    includes = [
        "%{musa_root}/include",
    ],
    linkstatic = 1,
    strip_include_prefix = "%{musa_root}",
    visibility = ["//visibility:public"],
    deps = [":musa_config"],
)


bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "musparse",
    srcs = glob(["%{musa_root}/lib/libmusparse*.so*"]),
    hdrs = glob(["%{musa_root}/include/musparse*.h"]),
    data = glob(["%{musa_root}/lib/libmusparse*.so*"]),
    include_prefix = "musa",
    includes = [
        "%{musa_root}/include/",
    ],
    strip_include_prefix = "%{musa_root}",
    visibility = ["//visibility:public"],
    deps = [":musa_config"],
)

cc_library(
    name = "musolver",
    srcs = glob(["%{musa_root}/lib/libmusolver*.so*"]),
    hdrs = glob(["%{musa_root}/include/musolver*.h"]),
    include_prefix = "musa",
    includes = [
        "%{musa_root}/include/",
    ],
    strip_include_prefix = "%{musa_root}",
    visibility = ["//visibility:public"],
    deps = [":musa_config"],
)

cc_library(
    name = "system_libs",
    srcs = glob([
        "musa_dist/usr/lib/**/libelf.so*",
        "musa_dist/usr/lib/**/libdrm.so*",
        "musa_dist/usr/lib/**/libnuma.so*",
        "musa_dist/usr/lib/**/libdrm_mtgpu.so*",
    ]),
    data = glob([
        "musa_dist/usr/**",
    ]),
)

filegroup(
    name = "musa_root",
    srcs = [
        "%{musa_root}/bin/clang-offload-bundler",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "all_files",
    srcs = glob(["%{musa_root}/**"]),
    visibility = ["//visibility:public"],
)
