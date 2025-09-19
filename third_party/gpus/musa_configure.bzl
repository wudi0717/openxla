"""Repository rule for MUSa autoconfiguration.

`musa_configure` depends on the following environment variables:

  * `TF_NEED_MUSA`: Whether to enable building with MUSa.
  * `TF_MUSA_CLANG`: Whether to use clang for C++ and MCC for MUSa compilation.
  * `TF_SYSROOT`: The sysroot to use when compiling.
  * `CLANG_COMPILER_PATH`: The clang compiler path that will be used for
    host code compilation if TF_MUSA_CLANG is 1.
  * `MUSA_PATH`: The path to the MUSa toolkit. Default is `/opt/musa`.
  * `TF_MUSA_AMDGPU_TARGETS`: The AMDGPU targets.
"""

load(
    "//third_party/gpus/musa:musa_redist.bzl",
    "musa_redist",
)
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
load(
    ":cuda_configure.bzl",
    "enable_cuda",
)
load(
    ":sycl_configure.bzl",
    "enable_sycl",
)

_CLANG_COMPILER_PATH = "CLANG_COMPILER_PATH"
_TF_SYSROOT = "TF_SYSROOT"
_MUSA_TOOLKIT_PATH = "MUSA_PATH"
_TF_MUSA_AMDGPU_TARGETS = "TF_MUSA_AMDGPU_TARGETS"
_TF_MUSA_CONFIG_REPO = "TF_MUSA_CONFIG_REPO"
_DISTRIBUTION_PATH = "musa/musa_dist"
_OS = "OS"
_MUSA_VERSION = "MUSA_VERSION"

_DEFAULT_MUSA_TOOLKIT_PATH = "/opt/musa"
_TF_MUSA_MULTIPLE_PATHS = "TF_MUSA_MULTIPLE_PATHS"
_LLVM_PATH = "LLVM_PATH"

def verify_build_defines(params):
    """Verify all variables that crosstool/BUILD.musa.tpl expects are substituted.

    Args:
      params: dict of variables that will be passed to the BUILD.tpl template.
    """
    missing = []
    for param in [
        "cxx_builtin_include_directories",
        "extra_no_canonical_prefixes_flags",
        "host_compiler_path",
        "host_compiler_prefix",
        "linker_bin_path",
        "unfiltered_compile_flags",
    ]:
        if ("%{" + param + "}") not in params:
            missing.append(param)

    if missing:
        auto_configure_fail(
            "BUILD.musa.tpl template is missing these variables: " +
            str(missing) +
            ".\nWe only got: " +
            str(params) +
            ".",
        )

def find_cc(repository_ctx, use_musa_clang):
    """Find the C++ compiler."""

    target_cc_name = "clang"
    cc_path_envvar = _CLANG_COMPILER_PATH
    cc_name = target_cc_name

    cc_name_from_env = get_host_environ(repository_ctx, cc_path_envvar)
    if cc_name_from_env:
        cc_name = cc_name_from_env
    if cc_name.startswith("/"):
        # Absolute path, maybe we should make this supported by our which function.
        return cc_name
    cc = which(repository_ctx, cc_name)
    if cc == None:
        fail(("Cannot find {}, either correct your path or set the {}" +
              " environment variable").format(target_cc_name, cc_path_envvar))
    return cc

def auto_configure_fail(msg):
    """Output failure message when musa configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("\n%sMUSa Configuration Error:%s %s\n" % (red, no_color, msg))

def auto_configure_warning(msg):
    """Output warning message during auto configuration."""
    yellow = "\033[1;33m"
    no_color = "\033[0m"
    print("\n%sAuto-Configuration Warning:%s %s\n" % (yellow, no_color, msg))

# END cc_configure common functions (see TODO above).

def _musa_include_path(repository_ctx, musa_config, bash_bin):
    """Generates the entries for musa inc dirs based on musa_config.

    Args:
      repository_ctx: The repository context.
      musa_config: The path to the gcc host compiler.
      bash_bin: path to the bash interpreter.

    Returns:
      A string containing the Starlark string for each of the mcc
      compiler include directories, which can be added to the CROSSTOOL
      file.
    """
    inc_dirs = []

    # Add HIP-Clang headers (relative to musa root)
    musa_path = repository_ctx.path(musa_config.musa_toolkit_path)
    clang_path = musa_path.get_child("llvm/bin/clang")
    resource_dir_result = execute(repository_ctx, [str(clang_path), "-print-resource-dir"])

    if resource_dir_result.return_code:
        auto_configure_fail("Failed to run mcc -print-resource-dir: %s" % err_out(resource_dir_result))

    resource_dir_abs = resource_dir_result.stdout.strip()

    resource_dir_rel = relative_to(repository_ctx, str(musa_path.realpath), resource_dir_abs, bash_bin)

    resource_dir = str(musa_path.get_child(resource_dir_rel))

    inc_dirs.append(resource_dir + "/include")
    inc_dirs.append(resource_dir + "/share")

    return inc_dirs

def _enable_musa(repository_ctx):
    enable_musa = get_host_environ(repository_ctx, "TF_NEED_MUSA")
    if enable_musa == "1":
        if get_cpu_value(repository_ctx) != "Linux":
            auto_configure_warning("MUSa configure is only supported on Linux")
            return False
        return True
    return False

def _amdgpu_targets(repository_ctx, musa_toolkit_path, bash_bin):
    """Returns a list of strings representing AMDGPU targets."""
    amdgpu_targets_str = get_host_environ(repository_ctx, _TF_MUSA_AMDGPU_TARGETS)
    if not amdgpu_targets_str:
        cmd = "%s/bin/musa_agent_enumerator" % musa_toolkit_path
        result = execute(repository_ctx, [bash_bin, "-c", cmd])
        targets = [target for target in result.stdout.strip().split("\n") if target != "gfx000"]
        targets = {x: None for x in targets}
        targets = list(targets.keys())
        amdgpu_targets_str = ",".join(targets)
    amdgpu_targets = [amdgpu for amdgpu in amdgpu_targets_str.split(",") if amdgpu]
    for amdgpu_target in amdgpu_targets:
        if amdgpu_target[:3] != "gfx":
            auto_configure_fail("Invalid AMDGPU target: %s" % amdgpu_target)
    return amdgpu_targets

def _mcc_env(repository_ctx):
    """Returns the environment variable string for mcc.

    Args:
        repository_ctx: The repository context.

    Returns:
        A string containing environment variables for mcc.
    """
    mcc_env = ""
    for name in [
        "HIP_CLANG_PATH",
        "DEVICE_LIB_PATH",
        "HIP_VDI_HOME",
        "HIPCC_VERBOSE",
        "HIPCC_COMPILE_FLAGS_APPEND",
        "HIPPCC_LINK_FLAGS_APPEND",
        "HCC_AMDGPU_TARGET",
        "HIP_PLATFORM",
    ]:
        env_value = get_host_environ(repository_ctx, name)
        if env_value:
            mcc_env = (mcc_env + " " + name + "=\"" + env_value + "\";")
    return mcc_env.strip()

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

def _musa_lib_paths(repository_ctx, lib, basedir):
    file_name = _lib_name(lib, version = "", static = False)
    return [
        repository_ctx.path("%s/lib64/%s" % (basedir, file_name)),
        repository_ctx.path("%s/lib64/stubs/%s" % (basedir, file_name)),
        repository_ctx.path("%s/lib/x86_64-linux-gnu/%s" % (basedir, file_name)),
        repository_ctx.path("%s/lib/%s" % (basedir, file_name)),
        repository_ctx.path("%s/%s" % (basedir, file_name)),
    ]

def _batch_files_exist(repository_ctx, libs_paths, bash_bin):
    all_paths = []
    for row in libs_paths:
        lib_paths = row[1]
        for lib_path in lib_paths:
            all_paths.append(lib_path)
    return files_exist(repository_ctx, all_paths, bash_bin)

def _select_musa_lib_paths(repository_ctx, libs_paths, bash_bin):
    test_results = _batch_files_exist(repository_ctx, libs_paths, bash_bin)

    libs = {}
    i = 0
    for row in libs_paths:
        name = row[0]
        lib_paths = row[1]
        optional = (len(row) > 2 and row[2] == True)
        selected_path = None
        for path in lib_paths:
            if test_results[i] and selected_path == None:
                # For each lib select the first path that exists.
                selected_path = path
            i = i + 1
        if selected_path == None:
            if optional:
                libs[name] = None
                continue
            else:
                auto_configure_fail("Cannot find musa library %s" % name)

        libs[name] = struct(file_name = selected_path.basename, path = realpath(repository_ctx, selected_path, bash_bin))

    return libs

def _find_libs(repository_ctx, musa_config, miopen_path, rccl_path, bash_bin):
    """Returns the MUSa libraries on the system.

    Args:
      repository_ctx: The repository context.
      musa_config: The MUSa config as returned by _get_musa_config
      bash_bin: the path to the bash interpreter

    Returns:
      Map of library names to structs of filename and path
    """
    libs_paths = [
        (name, _musa_lib_paths(repository_ctx, name, path))
        for name, path in [
            ("amdhip64", musa_config.musa_toolkit_path),
            ("rocblas", musa_config.musa_toolkit_path),
            ("hiprand", musa_config.musa_toolkit_path),
            ("MIOpen", miopen_path),
            ("rccl", rccl_path),
            ("hipsparse", musa_config.musa_toolkit_path),
            ("roctracer64", musa_config.musa_toolkit_path),
            ("rocsolver", musa_config.musa_toolkit_path),
        ]
    ]
    if int(musa_config.musa_version_number) >= 40500:
        libs_paths.append(("hipsolver", _musa_lib_paths(repository_ctx, "hipsolver", musa_config.musa_toolkit_path)))
        libs_paths.append(("hipblas", _musa_lib_paths(repository_ctx, "hipblas", musa_config.musa_toolkit_path)))

    # hipblaslt may be absent even in versions of MUSa where it exists
    # (it is not installed by default in some containers). Autodetect.
    libs_paths.append(("hipblaslt", _musa_lib_paths(repository_ctx, "hipblaslt", musa_config.musa_toolkit_path), True))
    return _select_musa_lib_paths(repository_ctx, libs_paths, bash_bin)

def find_musa_config(repository_ctx, musa_path):
    """Returns MUSa config dictionary from running find_musa_config.py"""
    python_bin = get_python_bin(repository_ctx)
    exec_result = execute(repository_ctx, [python_bin, repository_ctx.attr._find_musa_config], env_vars = {"MUSA_PATH": musa_path})
    if exec_result.return_code:
        auto_configure_fail("Failed to run find_musa_config.py: %s" % err_out(exec_result))

    # Parse the dict from stdout.
    return dict([tuple(x.split(": ")) for x in exec_result.stdout.splitlines()])

def _get_musa_config(repository_ctx, bash_bin, musa_path, install_path):
    """Detects and returns information about the MUSa installation on the system.

    Args:
      repository_ctx: The repository context.
      bash_bin: the path to the path interpreter

    Returns:
      A struct containing the following fields:
        musa_toolkit_path: The MUSa toolkit installation directory.
        amdgpu_targets: A list of the system's AMDGPU targets.
        musa_version_number: The version of MUSa on the system.
        miopen_version_number: The version of MIOpen on the system.
        hipruntime_version_number: The version of HIP Runtime on the system.
    """
    config = find_musa_config(repository_ctx, musa_path)
    musa_toolkit_path = config["musa_toolkit_path"]
    musa_version_number = config["musa_version_number"]
    miopen_version_number = config["miopen_version_number"]
    hipruntime_version_number = config["hipruntime_version_number"]
    return struct(
        amdgpu_targets = _amdgpu_targets(repository_ctx, musa_toolkit_path, bash_bin),
        musa_toolkit_path = musa_toolkit_path,
        musa_version_number = musa_version_number,
        miopen_version_number = miopen_version_number,
        hipruntime_version_number = hipruntime_version_number,
        install_path = install_path,
    )

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

_DUMMY_CROSSTOOL_BZL_FILE = """
def error_gpu_disabled():
  fail("ERROR: Building with --config=musa but TensorFlow is not configured " +
       "to build with GPU support. Please re-run ./configure and enter 'Y' " +
       "at the prompt to build with GPU support.")

  native.genrule(
      name = "error_gen_crosstool",
      outs = ["CROSSTOOL"],
      cmd = "echo 'Should not be run.' && exit 1",
  )

  native.filegroup(
      name = "crosstool",
      srcs = [":CROSSTOOL"],
      output_licenses = ["unencumbered"],
  )
"""

_DUMMY_CROSSTOOL_BUILD_FILE = """
load("//crosstool:error_gpu_disabled.bzl", "error_gpu_disabled")

error_gpu_disabled()
"""

def _create_dummy_repository(repository_ctx):
    # Set up BUILD file for musa/.
    _tpl(
        repository_ctx,
        "musa:build_defs.bzl",
        {
            "%{musa_is_configured}": "False",
            "%{gpu_is_configured}": "if_true" if enable_cuda(repository_ctx) or enable_sycl(repository_ctx) else "if_false",
            "%{cuda_or_musa}": "if_true" if enable_cuda(repository_ctx) else "if_false",
            "%{musa_extra_copts}": "[]",
            "%{musa_gpu_architectures}": "[]",
            "%{musa_version_number}": "0",
            "%{musa_hipblaslt}": "False",
        },
    )
    _tpl(
        repository_ctx,
        "musa:BUILD",
        {
            "%{hip_lib}": _lib_name("hip"),
            "%{rocblas_lib}": _lib_name("rocblas"),
            "%{hipblas_lib}": _lib_name("hipblas"),
            "%{miopen_lib}": _lib_name("miopen"),
            "%{rccl_lib}": _lib_name("rccl"),
            "%{hiprand_lib}": _lib_name("hiprand"),
            "%{hipsparse_lib}": _lib_name("hipsparse"),
            "%{roctracer_lib}": _lib_name("roctracer64"),
            "%{rocsolver_lib}": _lib_name("rocsolver"),
            "%{hipsolver_lib}": _lib_name("hipsolver"),
            "%{hipblaslt_lib}": _lib_name("hipblaslt"),
            "%{musa_headers}": "",
        },
    )

    # Create dummy files for the MUSa toolkit since they are still required by
    # tensorflow/compiler/xla/stream_executor/musa:musa_rpath
    repository_ctx.file("musa/hip/include/hip/hip_runtime.h", "")

    # Set up musa_config.h, which is used by
    # tensorflow/compiler/xla/stream_executor/dso_loader.cc.
    _tpl(
        repository_ctx,
        "musa:musa_config.h",
        {
            "%{musa_toolkit_path}": _DEFAULT_MUSA_TOOLKIT_PATH,
            "%{hipblaslt_flag}": "0",
        },
        "musa/musa_config/musa_config.h",
    )

    # If musa_configure is not configured to build with GPU support, and the user
    # attempts to build with --config=musa, add a dummy build rule to intercept
    # this and fail with an actionable error message.
    repository_ctx.file(
        "crosstool/error_gpu_disabled.bzl",
        _DUMMY_CROSSTOOL_BZL_FILE,
    )
    repository_ctx.file("crosstool/BUILD", _DUMMY_CROSSTOOL_BUILD_FILE)

def _norm_path(path):
    """Returns a path with '/' and remove the trailing slash."""
    path = path.replace("\\", "/")
    if path[-1] == "/":
        path = path[:-1]
    return path

def _genrule(src_dir, genrule_name, command, outs):
    """Returns a string with a genrule.

    Genrule executes the given command and produces the given outputs.
    """
    return (
        "genrule(\n" +
        '    name = "' +
        genrule_name + '",\n' +
        "    outs = [\n" +
        outs +
        "\n    ],\n" +
        '    cmd = """\n' +
        command +
        '\n   """,\n' +
        ")\n"
    )

def _flag_enabled(repository_ctx, flag_name):
    return get_host_environ(repository_ctx, flag_name) == "1"

def _use_musa_clang(repository_ctx):
    # Returns the flag if we need to use clang for the host.
    return _flag_enabled(repository_ctx, "TF_MUSA_CLANG")

def _tf_sysroot(repository_ctx):
    return get_host_environ(repository_ctx, _TF_SYSROOT, "")

def _compute_musa_extra_copts(repository_ctx, amdgpu_targets):
    amdgpu_target_flags = ["--offload-arch=" +
                           amdgpu_target for amdgpu_target in amdgpu_targets]
    return str(amdgpu_target_flags)

def _get_file_name(url):
    last_slash_index = url.rfind("/")
    return url[last_slash_index + 1:]

def _download_package(repository_ctx, archive):
    file_name = _get_file_name(archive.url)
    tmp_dir = "tmp"
    repository_ctx.file(tmp_dir + "/.idx")  # create tmp dir

    repository_ctx.report_progress("Downloading and extracting {}, expected hash is {}".format(archive.url, archive.sha256))  # buildifier: disable=print
    repository_ctx.download_and_extract(
        url = archive.url,
        output = tmp_dir if archive.url.endswith(".deb") else _DISTRIBUTION_PATH,
        sha256 = archive.sha256,
    )

    all_files = repository_ctx.path(tmp_dir).readdir()

    matched_files = [f for f in all_files if _get_file_name(str(f)).startswith("data.")]
    for f in matched_files:
        repository_ctx.extract(f, _DISTRIBUTION_PATH)

    repository_ctx.delete(tmp_dir)
    repository_ctx.delete(file_name)

def _remove_root_dir(path, root_dir):
    if path.startswith(root_dir + "/"):
        return path[len(root_dir) + 1:]
    return path

def _setup_musa_distro_dir(repository_ctx):
    """Sets up the musa hermetic installation directory to be used in hermetic build"""
    bash_bin = get_bash_bin(repository_ctx)
    os = repository_ctx.os.environ.get(_OS)
    musa_version = repository_ctx.os.environ.get(_MUSA_VERSION)
    multiple_paths = repository_ctx.os.environ.get(_TF_MUSA_MULTIPLE_PATHS)
    if os and musa_version:
        redist = musa_redist[os][musa_version]
        repository_ctx.file("musa/.index")
        for archive in redist["archives"]:
            _download_package(repository_ctx, archive)
        return _get_musa_config(repository_ctx, bash_bin, "{}/{}".format(_DISTRIBUTION_PATH, redist["musa_root"]), "/{}".format(redist["musa_root"]))
    elif multiple_paths:
        paths_list = multiple_paths.split(":")
        for musa_custom_path in paths_list:
            cmd = "find " + musa_custom_path + "/* \\( -type f -o -type l \\)"
            result = execute(repository_ctx, [bash_bin, "-c", cmd]).stdout.strip().split("\n")
            for file_path in result:
                relative_path = file_path[len(musa_custom_path):]
                symlink_path = _DISTRIBUTION_PATH + relative_path
                if files_exist(repository_ctx, [symlink_path], bash_bin)[0]:
                    fail("File already present: " + relative_path)
                else:
                    repository_ctx.symlink(file_path, symlink_path)
        llvm_path = repository_ctx.os.environ.get(_LLVM_PATH)
        if llvm_path:
            repository_ctx.symlink(llvm_path, _DISTRIBUTION_PATH + "/llvm")
            repository_ctx.symlink(llvm_path, _DISTRIBUTION_PATH + "/lib/llvm")
            repository_ctx.symlink(llvm_path + "/amdgcn", _DISTRIBUTION_PATH + "/amdgcn")
        return _get_musa_config(repository_ctx, bash_bin, _DISTRIBUTION_PATH, _DISTRIBUTION_PATH)
    else:
        musa_path = repository_ctx.os.environ.get(_MUSA_TOOLKIT_PATH, _DEFAULT_MUSA_TOOLKIT_PATH)
        repository_ctx.report_progress("Using local musa installation {}".format(musa_path))  # buildifier: disable=print
        repository_ctx.symlink(musa_path, _DISTRIBUTION_PATH)
        return _get_musa_config(repository_ctx, bash_bin, _DISTRIBUTION_PATH, _DEFAULT_MUSA_TOOLKIT_PATH)

def _create_local_musa_repository(repository_ctx):
    """Creates the repository containing files set up to build with MUSa."""

    tpl_paths = {labelname: _tpl_path(repository_ctx, labelname) for labelname in [
        "musa:build_defs.bzl",
        "musa:BUILD",
        "crosstool:BUILD.musa",
        "crosstool:mcc_cc_toolchain_config.bzl",
        "crosstool:clang/bin/crosstool_wrapper_driver_musa",
        "musa:musa_config.h",
    ]}

    musa_config = _setup_musa_distro_dir(repository_ctx)
    musa_version_number = int(musa_config.musa_version_number)

    # For MUSa 5.2 and above, find MIOpen and RCCL in the main musa lib path
    miopen_path = musa_config.musa_toolkit_path + "/miopen" if musa_version_number < 50200 else musa_config.musa_toolkit_path
    rccl_path = musa_config.musa_toolkit_path + "/rccl" if musa_version_number < 50200 else musa_config.musa_toolkit_path

    # Copy header and library files to execroot.
    # musa_toolkit_path
    musa_toolkit_path = _remove_root_dir(musa_config.musa_toolkit_path, "musa")

    bash_bin = get_bash_bin(repository_ctx)
    musa_libs = _find_libs(repository_ctx, musa_config, miopen_path, rccl_path, bash_bin)
    musa_lib_srcs = []
    musa_lib_outs = []
    for lib in musa_libs.values():
        if lib:
            musa_lib_srcs.append(lib.path)
            musa_lib_outs.append("musa/lib/" + lib.file_name)

    clang_offload_bundler_path = musa_toolkit_path + "/llvm/bin/clang-offload-bundler"

    have_hipblaslt = "1" if musa_libs["hipblaslt"] != None else "0"

    # Set up BUILD file for musa/
    repository_ctx.template(
        "musa/build_defs.bzl",
        tpl_paths["musa:build_defs.bzl"],
        {
            "%{musa_is_configured}": "True",
            "%{gpu_is_configured}": "if_true",
            "%{cuda_or_musa}": "if_true",
            "%{musa_extra_copts}": _compute_musa_extra_copts(
                repository_ctx,
                musa_config.amdgpu_targets,
            ),
            "%{musa_gpu_architectures}": str(musa_config.amdgpu_targets),
            "%{musa_version_number}": str(musa_version_number),
            "%{musa_hipblaslt}": "True" if musa_libs["hipblaslt"] != None else "False",
        },
    )

    repository_dict = {
        "%{musa_root}": musa_toolkit_path,
        "%{musa_toolkit_path}": str(repository_ctx.path(musa_config.musa_toolkit_path)),
    }

    is_musa_clang = _use_musa_clang(repository_ctx)
    tf_sysroot = _tf_sysroot(repository_ctx)

    if musa_libs["hipblaslt"] != None:
        repository_dict["%{hipblaslt_lib}"] = musa_libs["hipblaslt"].file_name

    if musa_version_number >= 40500:
        repository_dict["%{hipsolver_lib}"] = musa_libs["hipsolver"].file_name
        repository_dict["%{hipblas_lib}"] = musa_libs["hipblas"].file_name

    multiple_paths = repository_ctx.os.environ.get(_TF_MUSA_MULTIPLE_PATHS)
    if multiple_paths:
        paths_list = multiple_paths.split(":")
        musa_lib_paths = []
        for musa_custom_path in paths_list:
            lib_path = musa_custom_path + "/lib/"
            if files_exist(repository_ctx, [lib_path], bash_bin)[0] and not lib_path in musa_lib_paths:
                musa_lib_paths.append(lib_path)
        repository_dict["%{musa_lib_paths}"] = ":".join(musa_lib_paths)

    repository_ctx.template(
        "musa/BUILD",
        tpl_paths["musa:BUILD"],
        repository_dict,
    )

    # Set up crosstool/
    cc = find_cc(repository_ctx, is_musa_clang)
    host_compiler_includes = get_cxx_inc_directories(
        repository_ctx,
        cc,
        tf_sysroot,
    )

    # host_compiler_includes = get_cxx_inc_directories(repository_ctx, cc)

    musa_defines = {}
    musa_defines["%{builtin_sysroot}"] = tf_sysroot
    musa_defines["%{compiler}"] = "clang"
    host_compiler_prefix = "/usr/bin"
    musa_defines["%{host_compiler_prefix}"] = host_compiler_prefix
    musa_defines["%{linker_bin_path}"] = musa_config.musa_toolkit_path + host_compiler_prefix
    musa_defines["%{extra_no_canonical_prefixes_flags}"] = ""
    musa_defines["%{unfiltered_compile_flags}"] = ""
    musa_defines["%{musa_mcc_files}"] = "[]"
    musa_defines["%{extra_no_canonical_prefixes_flags}"] = "\"-no-canonical-prefixes\""

    musa_defines["%{unfiltered_compile_flags}"] = to_list_of_strings([
        "-DTENSORFLOW_USE_MUSA=1",
        "-D__HIP_PLATFORM_AMD__",
        "-DEIGEN_USE_HIP",
        "-DUSE_MUSA",
    ])

    musa_defines["%{host_compiler_path}"] = "clang/bin/crosstool_wrapper_driver_is_not_gcc"

    musa_defines["%{cxx_builtin_include_directories}"] = to_list_of_strings(
        host_compiler_includes + _musa_include_path(repository_ctx, musa_config, bash_bin),
    )

    verify_build_defines(musa_defines)

    # Only expand template variables in the BUILD file
    repository_ctx.template(
        "crosstool/BUILD",
        tpl_paths["crosstool:BUILD.musa"],
        musa_defines,
    )

    # No templating of cc_toolchain_config - use attributes and templatize the
    # BUILD file.
    repository_ctx.template(
        "crosstool/cc_toolchain_config.bzl",
        tpl_paths["crosstool:mcc_cc_toolchain_config.bzl"],
        musa_defines,
    )

    repository_ctx.template(
        "crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc",
        tpl_paths["crosstool:clang/bin/crosstool_wrapper_driver_musa"],
        {
            "%{cpu_compiler}": str(cc),
            "%{compiler_is_clang}": "True" if is_musa_clang else "False",
            "%{mcc_path}": str(repository_ctx.path(musa_config.musa_toolkit_path + "/bin/mcc")),
            "%{mcc_env}": _mcc_env(repository_ctx),
            "%{musa_path}": str(repository_ctx.path(musa_config.musa_toolkit_path)),
            "%{rocr_runtime_path}": str(repository_ctx.path(musa_config.musa_toolkit_path + "/lib")),
            "%{rocr_runtime_library}": "hsa-runtime64",
            "%{hip_runtime_path}": str(repository_ctx.path(musa_config.musa_toolkit_path + "/lib")),
            "%{hip_runtime_library}": "amdhip64",
            "%{crosstool_verbose}": _crosstool_verbose(repository_ctx),
            "%{gcc_host_compiler_path}": str(cc),
            "%{musa_amdgpu_targets}": ",".join(
                ["\"%s\"" % c for c in musa_config.amdgpu_targets],
            ),
        },
    )

    # Set up musa_config.h, which is used by
    # tensorflow/compiler/xla/stream_executor/dso_loader.cc.
    repository_ctx.template(
        "musa/musa_config/musa_config.h",
        tpl_paths["musa:musa_config.h"],
        {
            "%{musa_amdgpu_targets}": ",".join(
                ["\"%s\"" % c for c in musa_config.amdgpu_targets],
            ),
            "%{musa_toolkit_path}": musa_config.install_path,
            "%{musa_version_number}": musa_config.musa_version_number,
            "%{miopen_version_number}": musa_config.miopen_version_number,
            "%{hipruntime_version_number}": musa_config.hipruntime_version_number,
            "%{hipblaslt_flag}": have_hipblaslt,
            "%{hip_soversion_number}": "6" if int(musa_config.musa_version_number) >= 60000 else "5",
            "%{rocblas_soversion_number}": "5" if int(musa_config.musa_version_number) >= 70000 else "4",
        },
    )

    # Set up musa_config.h, which is used by
    # tensorflow/compiler/xla/stream_executor/dso_loader.cc.
    repository_ctx.template(
        "musa/musa_config_hermetic/musa_config.h",
        tpl_paths["musa:musa_config.h"],
        {
            "%{musa_amdgpu_targets}": ",".join(
                ["\"%s\"" % c for c in musa_config.amdgpu_targets],
            ),
            "%{musa_toolkit_path}": str(repository_ctx.path(musa_config.musa_toolkit_path)),
            "%{musa_version_number}": musa_config.musa_version_number,
            "%{miopen_version_number}": musa_config.miopen_version_number,
            "%{hipruntime_version_number}": musa_config.hipruntime_version_number,
            "%{hipblaslt_flag}": have_hipblaslt,
            "%{hip_soversion_number}": "6" if int(musa_config.musa_version_number) >= 60000 else "5",
            "%{rocblas_soversion_number}": "5" if int(musa_config.musa_version_number) >= 70000 else "4",
        },
    )

def _create_remote_musa_repository(repository_ctx, remote_config_repo):
    """Creates pointers to a remotely configured repo set up to build with MUSa."""
    _tpl(
        repository_ctx,
        "musa:build_defs.bzl",
        {
            "%{musa_is_configured}": "True",
            "%{gpu_is_configured}": "if_true",
            "%{cuda_or_musa}": "if_true",
            "%{musa_extra_copts}": _compute_musa_extra_copts(
                repository_ctx,
                [],  #_compute_capabilities(repository_ctx)
            ),
        },
    )
    repository_ctx.template(
        "musa/BUILD",
        config_repo_label(remote_config_repo, "musa:BUILD"),
        {},
    )
    repository_ctx.template(
        "musa/build_defs.bzl",
        config_repo_label(remote_config_repo, "musa:build_defs.bzl"),
        {},
    )
    repository_ctx.template(
        "musa/musa/musa_config.h",
        config_repo_label(remote_config_repo, "musa:musa/musa_config.h"),
        {},
    )
    repository_ctx.template(
        "crosstool/BUILD",
        config_repo_label(remote_config_repo, "crosstool:BUILD"),
        {},
    )
    repository_ctx.template(
        "crosstool/cc_toolchain_config.bzl",
        config_repo_label(remote_config_repo, "crosstool:cc_toolchain_config.bzl"),
        {},
    )
    repository_ctx.template(
        "crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc",
        config_repo_label(remote_config_repo, "crosstool:clang/bin/crosstool_wrapper_driver_is_not_gcc"),
        {},
    )

def _musa_autoconf_impl(repository_ctx):
    """Implementation of the musa_autoconf repository rule."""
    if not _enable_musa(repository_ctx):
        _create_dummy_repository(repository_ctx)
    elif get_host_environ(repository_ctx, _TF_MUSA_CONFIG_REPO) != None:
        _create_remote_musa_repository(
            repository_ctx,
            get_host_environ(repository_ctx, _TF_MUSA_CONFIG_REPO),
        )
    else:
        _create_local_musa_repository(repository_ctx)

_ENVIRONS = [
    "TF_NEED_MUSA",
    "TF_MUSA_CLANG",
    "TF_NEED_CUDA",  # Needed by the `if_gpu_is_configured` macro
    _MUSA_TOOLKIT_PATH,
    _TF_MUSA_AMDGPU_TARGETS,
    _OS,
    _MUSA_VERSION,
]

remote_musa_configure = repository_rule(
    implementation = _create_local_musa_repository,
    environ = _ENVIRONS,
    remotable = True,
    attrs = {
        "environ": attr.string_dict(),
        "_find_musa_config": attr.label(
            default = Label("@xla//third_party/gpus:find_musa_config.py"),
        ),
    },
)

musa_configure = repository_rule(
    implementation = _musa_autoconf_impl,
    environ = _ENVIRONS + [_TF_MUSA_CONFIG_REPO],
    attrs = {
        "_find_musa_config": attr.label(
            default = Label("@xla//third_party/gpus:find_musa_config.py"),
        ),
    },
)
"""Detects and configures the local MUSa toolchain.

Add the following to your WORKSPACE FILE:

```python
musa_configure(name = "local_config_musa")
```

Args:
  name: A unique name for this workspace rule.
"""
