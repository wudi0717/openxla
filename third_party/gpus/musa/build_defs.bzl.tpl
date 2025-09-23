# Macros for building MUSa code.
def if_musa(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with MUSa.

    Returns a select statement which evaluates to if_true if we're building
    with MUSa enabled.  Otherwise, the select statement evaluates to if_false.

    """
    return select({
        "@local_config_musa//musa:using_mcc": if_true,
        "//conditions:default": if_false
    })

def select_threshold(value, above_or_eq, threshold, below):
    return below if value < threshold else above_or_eq

def musa_default_copts():
    """Default options for all MUSa compilations."""
    return if_musa(["-x", "musa"] + %{musa_extra_copts})

def musa_copts(opts = []):
    """Gets the appropriate set of copts for (maybe) MUSa compilation.

      If we're doing MUSa compilation, returns copts for our particular MUSa
      compiler.  If we're not doing MUSa compilation, returns an empty list.

      """
    return musa_default_copts() + select({
        "//conditions:default": [],
        "@local_config_musa//musa:using_mcc": ([
            "",
        ]),
    }) + if_musa_is_configured(opts)

def musa_gpu_architectures():
    """Returns a list of supported GPU architectures."""
    return %{musa_gpu_architectures}

def musa_version_number():
    """Returns a list of supported GPU architectures."""
    return %{musa_version_number}

def if_gpu_is_configured(if_true, if_false = []):
    """Tests if MUSa or CUDA or SYCL was enabled during the configure process."""
    return select({"//conditions:default": %{gpu_is_configured}})

def if_cuda_or_musa(if_true, if_false = []):
    """Tests if MUSa or CUDA was enabled during the configure process.

    Unlike if_musa() or if_cuda(), this does not require that we are building
    with --config=musa or --config=cuda, respectively. Used to allow non-GPU
    code to depend on MUSa or CUDA libraries.

    """
    return select({"//conditions:default": %{cuda_or_musa}})

def if_musa_is_configured(if_true, if_false = []):
    """Tests if the MUSa was enabled during the configure process.

    Unlike if_musa(), this does not require that we are building with
    --config=musa. Used to allow non-MUSa code to depend on MUSa libraries.
    """
    if %{musa_is_configured}:
      return select({"//conditions:default": if_true})
    return select({"//conditions:default": if_false})

def is_musa_configured():
    """
    Returns True if MUSa is configured. False otherwise.
    """
    return %{musa_is_configured}

def musa_library(copts = [], deps = [], **kwargs):
    """Wrapper over cc_library which adds default MUSa options."""
    if "@local_config_musa//musa:musa_headers" not in deps:
      deps.append("@local_config_musa//musa:musa_headers")
    native.cc_library(copts = musa_default_copts() + copts, deps = deps, **kwargs)
