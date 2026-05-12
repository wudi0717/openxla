"""Repository rule for MUSa autoconfiguration.

"""

load(
    "//third_party/remote_config:common.bzl",
    "config_repo_label",
    "err_out",
    "execute",
    "files_exist",
    "get_bash_bin",
    "get_cpu_value",
    "get_host_environ",
    "get_python_bin",
    "realpath",
    "relative_to",
    "which",
)
load(
    ":compiler_common_tools.bzl",
    "get_cxx_inc_directories",
    "to_list_of_strings",
)
_DUMMY_CROSSTOOL_BUILD_FILE = """
"""
def _musa_include_path(repository_ctx):
    """Generates the entries for rocm inc dirs based on rocm_config.

    Args:
      repository_ctx: The repository context.
      rocm_config: The path to the gcc host compiler.
      bash_bin: path to the bash interpreter.

    Returns:
      A string containing the Starlark string for each of the hipcc
      compiler include directories, which can be added to the CROSSTOOL
      file.
    """
    inc_dirs = []

    inc_dirs.append("/home/mccxadmin/moon/workspace/install/musa/include")

    return inc_dirs
def _tf_sysroot(repository_ctx):
    return ""

def find_cc(repository_ctx):
    """Find the C++ compiler."""
    cc_name_from_env = "/usr/bin/gcc" 
    if cc_name_from_env:
        cc_name = cc_name_from_env
    if cc_name.startswith("/"):
        # Absolute path, maybe we should make this supported by our which function.
        return cc_name
    cc = which(repository_ctx, cc_name)
    return cc

def _crosstool_verbose(repository_ctx):
    """Returns the environment variable value CROSSTOOL_VERBOSE.

    Args:
        repository_ctx: The repository context.

    Returns:
        A string containing value of environment variable CROSSTOOL_VERBOSE.
    """
    return get_host_environ(repository_ctx, "CROSSTOOL_VERBOSE", "0")

def _lib_name(lib, version = "", static = False):
    """Constructs the name of a library on Linux.

    Args:
      lib: The name of the library, such as "hip"
      version: The version of the library.
      static: True the library is static or False if it is a shared object.

    Returns:
      The platform-specific name of the library.
    """
    if static:
        return "lib%s.a" % lib
    else:
        if version:
            version = ".%s" % version
        return "lib%s.so%s" % (lib, version)

def _tpl_path(repository_ctx, labelname):
    return repository_ctx.path(Label("//third_party/gpus/%s.tpl" % labelname))

def _tpl(repository_ctx, tpl, substitutions = {}, out = None):
    if not out:
        out = tpl.replace(":", "/")
    repository_ctx.template(
        out,
        _tpl_path(repository_ctx, tpl),
        substitutions,
    )

def _mcc_env(repository_ctx):
    """Returns the environment variable string for mcc.

    Args:
        repository_ctx: The repository context.

    Returns:
        A string containing environment variables for mcc.
    """
    mcc_env = ""
    for name in [
        "MUSA_CLANG_PATH",
        "DEVICE_LIB_PATH",
        "MUSA_VDI_HOME",
        "MCC_VERBOSE",
        "MCC_COMPILE_FLAGS_APPEND",
        "MCC_LINK_FLAGS_APPEND",
        "MCC_PLATFORM",
    ]:
        env_value = get_host_environ(repository_ctx, name)
        if env_value:
            mcc_env = (mcc_env + " " + name + "=\"" + env_value + "\";")
    return mcc_env.strip()

def _musa_autoconf_impl(repository_ctx):
    """Implementation of the musa_autoconf repository rule."""
    root = repository_ctx.os.environ.get("MUSA_HOME", "/home/mccxadmin/moon/workspace/install/musa")
    if not repository_ctx.path(root).exists:
        fail("MUSA not found at " + root)
    repository_ctx.symlink(root, "musa/env")

    # Set up BUILD file for musa/.
    _tpl(
        repository_ctx,
        "musa:BUILD",
        {
            "%{musa_root}": "env",
            "%{musa_toolkit_path}":"env",
            "%{musa_lib_paths}":"env",
        },
    )

    repository_ctx.template(
        "musa/build_defs.bzl",
        _tpl_path(repository_ctx, "musa:build_defs.bzl"),
        {
            "%{musa_is_configured}": "True",
            "%{gpu_is_configured}": "if_true",
            "%{cuda_or_musa}": "if_true",
            "%{musa_extra_copts}": "[]",
            "%{musa_gpu_architectures}": "[]",
            "%{musa_version_number}": "3.1",
        },
    )

    repository_ctx.template(
        "crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc",
        _tpl_path(repository_ctx, "crosstool:clang/bin/crosstool_wrapper_driver_musa"),
        {
            "%{cpu_compiler}": "gcc",
            "%{compiler_is_clang}": "False",
            "%{mcc_path}": "/usr/local/musa/bin/mcc",
            # "%{mcc_path}": "/data/lmw/musa/bin/mcc",
            "%{mcc_env}":  _mcc_env(repository_ctx),
            "%{musa_path}": "/usr/local/musa",
            # "%{musa_path}": "/data/lmw/musa",
            # "%{musa_runtime_path}": "/usr/local/musa/lib",
            "%{musa_runtime_path}": "/data/lmw/musa/lib",
            "%{musa_runtime_library}": "musart",
            "%{crosstool_verbose}": _crosstool_verbose(repository_ctx),
        },
    )

    tf_sysroot = _tf_sysroot(repository_ctx)
    # Set up crosstool/
    cc = find_cc(repository_ctx)
    host_compiler_includes = get_cxx_inc_directories(
        repository_ctx,
        cc,
        tf_sysroot,
    )
    # TODO(perfxlab): get path from env
    host_compiler_includes.append("/usr/local/musa/lib/clang/14.0.0/include/")
    host_compiler_includes.append("/usr/local/musa/lib/clang/20/include/")
    # host_compiler_includes.append("/data/lmw/musa/lib/14.0.0/include/")
    # host_compiler_includes.append("/data/lmw/musa/lib/20/include/")

    musa_defines = {}
    musa_defines["%{builtin_sysroot}"] = "/home/mccxadmin/moon/workspace/install/musa"
    musa_defines["%{compiler}"] = "mcc"
    host_compiler_prefix = "/usr/bin"
    musa_defines["%{host_compiler_prefix}"] = host_compiler_prefix
    musa_defines["%{linker_bin_path}"] = ""
    musa_defines["%{extra_no_canonical_prefixes_flags}"] = ""
    musa_defines["%{unfiltered_compile_flags}"] = to_list_of_strings([
        "-DTENSORFLOW_USE_MUSA=1",
        "-D__MUSA_PLATFORM_MT__",
        "-DUSE_MUSA",
    ])
    musa_defines["%{extra_no_canonical_prefixes_flags}"] = "\"-no-canonical-prefixes\""

    musa_defines["%{host_compiler_path}"] = "clang/bin/crosstool_wrapper_driver_is_not_gcc"

    musa_defines["%{cxx_builtin_include_directories}"] = to_list_of_strings(
        host_compiler_includes
    )

    repository_ctx.template(
        "crosstool/BUILD",
        _tpl_path(repository_ctx, "crosstool:BUILD.musa"),
        musa_defines,
    )
    repository_ctx.template(
        "crosstool/cc_toolchain_config.bzl",
        _tpl_path(repository_ctx, "crosstool:mcc_cc_toolchain_config.bzl"),
        musa_defines,
    )
    """
    musa_root = "/usr/local/musa"
    repository_ctx.symlink(musa_root, "musa_env")
    """

musa_configure = repository_rule(
    implementation = _musa_autoconf_impl,
    environ = ["MUSA_HOME"],
    local = True, 
)

"""Detects and configures the local MUSa toolchain.

Add the following to your WORKSPACE FILE:

```python
musa_configure(name = "local_config_musa")
```

Args:
  name: A unique name for this workspace rule.
"""
