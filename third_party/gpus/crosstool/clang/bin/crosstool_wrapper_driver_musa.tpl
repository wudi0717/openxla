#!/usr/bin/env python3
"""Crosstool wrapper for compiling MUSa programs.

SYNOPSIS:
  crosstool_wrapper_driver_musa [options passed in by cc_library()
                                or cc_binary() rule]

DESCRIPTION:
  This script is expected to be called by the cc_library() or cc_binary() bazel
  rules. When the option "-x musa" is present in the list of arguments passed
  to this script, it invokes the mcc compiler. Most arguments are passed
  as is as a string to --compiler-options of mcc. When "-x musa" is not
  present, this wrapper invokes gcc with the input arguments as is.
"""

__author__ = 'pengzhang_nudt@sina.com (Peng Zhang)'

from argparse import ArgumentParser
import os
import subprocess
import re
import sys
import shlex

# Template values set by musa_configure.bzl.
CPU_COMPILER = ('%{cpu_compiler}')
HOST_COMPILER_PATH = ('%{host_compiler_path}')

MCC_PATH = '%{mcc_path}'
PREFIX_DIR = os.path.dirname(HOST_COMPILER_PATH)
MCC_ENV = '%{mcc_env}'
MUSA_RUNTIME_PATH = '%{musa_runtime_path}'
MUSA_RUNTIME_LIBRARY = '%{musa_runtime_library}'
VERBOSE = '%{crosstool_verbose}'=='1'

def Log(s):
  print('gpus/crosstool: {0}'.format(s))


def GetOptionValue(argv, option):
  """Extract the list of values for option from the argv list.

  Args:
    argv: A list of strings, possibly the argv passed to main().
    option: The option whose value to extract, without the leading '-'.

  Returns:
    A list of values, either directly following the option,
    (eg., -opt val1 val2) or values collected from multiple occurrences of
    the option (eg., -opt val1 -opt val2).
  """

  parser = ArgumentParser()
  parser.add_argument('-' + option, nargs='*', action='append')
  args, _ = parser.parse_known_args(argv)
  if not args or not vars(args)[option]:
    return []
  else:
    return sum(vars(args)[option], [])


def GetHostCompilerOptions(argv):
  """Collect the -isystem, -iquote, and --sysroot option values from argv.

  Args:
    argv: A list of strings, possibly the argv passed to main().

  Returns:
    The string that can be used as the --compiler-options to mcc.
  """

  parser = ArgumentParser()
  parser.add_argument('-isystem', nargs='*', action='append')
  parser.add_argument('-iquote', nargs='*', action='append')
  parser.add_argument('--sysroot', nargs=1)
  parser.add_argument('-g', nargs='*', action='append')
  parser.add_argument('-no-canonical-prefixes', action='store_true')
  parser.add_argument('--genco', action='store_true')

  args, _ = parser.parse_known_args(argv)

  opts = ''

  if args.isystem:
    opts += ' -isystem ' + ' -isystem '.join(sum(args.isystem, []))
  if args.iquote:
    opts += ' -iquote ' + ' -iquote '.join(sum(args.iquote, []))
  if args.g:
    opts += ' -g' + ' -g'.join(sum(args.g, []))
  if args.no_canonical_prefixes:
    opts += ' -no-canonical-prefixes'
  if args.sysroot:
    opts += ' --sysroot ' + args.sysroot[0]
  if args.genco:
    opts += ' --genco'

  return opts

def GetMccOptions(argv):
  """Collect the -mcc_options values from argv.
  Args:
    argv: A list of strings, possibly the argv passed to main().
  Returns:
    The string that can be passed directly to mcc.
  """

  parser = ArgumentParser()
  parser.add_argument('--offload-arch', nargs='*', action='append')
  # TODO find a better place for this
  parser.add_argument('-gline-tables-only', action='store_true')

  args, _ = parser.parse_known_args(argv)

  mcc_opts = ' -gline-tables-only ' if args.gline_tables_only else ''
  if args.offload_arch:
    mcc_opts = mcc_opts + ' '.join(['--offload-arch=' + a for a in sum(args.offload_arch, [])])

  return mcc_opts

def system(cmd):
  """Invokes cmd with os.system().

  Args:
    cmd: The command.

  Returns:
    The exit code if the process exited with exit() or -signal
    if the process was terminated by a signal.
  """
  retv = os.system(cmd)
  if os.WIFEXITED(retv):
    return os.WEXITSTATUS(retv)
  else:
    return -os.WTERMSIG(retv)


def InvokeMcc(argv, log=False):
  """Call mcc with arguments assembled from argv.

  Args:
    argv: A list of strings, possibly the argv passed to main().
    log: True if logging is requested.

  Returns:
    The return value of calling os.system('mcc ' + args)
  """

  host_compiler_options = GetHostCompilerOptions(argv)
  mcc_compiler_options = GetMccOptions(argv)
  opt_option = GetOptionValue(argv, 'O')
  m_options = GetOptionValue(argv, 'm')
  m_options = ''.join([' -m' + m for m in m_options if m in ['32', '64']])
  include_options = GetOptionValue(argv, 'I')
  out_file = GetOptionValue(argv, 'o')
  depfiles = GetOptionValue(argv, 'MF')
  defines = GetOptionValue(argv, 'D')
  defines = ''.join([' -D' + define for define in defines])
  undefines = GetOptionValue(argv, 'U')
  undefines = ''.join([' -U' + define for define in undefines])
  std_options = GetOptionValue(argv, 'std')
  mcc_allowed_std_options = ["c++11", "c++14", "c++17"]
  std_options = ''.join([' -std=' + define
      for define in std_options if define in mcc_allowed_std_options])

  # The list of source files get passed after the -c option. I don't know of
  # any other reliable way to just get the list of source files to be compiled.
  src_files = GetOptionValue(argv, 'c')

  if len(src_files) == 0:
    return 1
  if len(out_file) != 1:
    return 1

  opt = (' -O2' if (len(opt_option) > 0 and int(opt_option[0]) > 0)
         else ' -g')

  includes = (' -I ' + ' -I '.join(include_options)
              if len(include_options) > 0
              else '')

  # Unfortunately, there are other options that have -c prefix too.
  # So allowing only those look like C/C++ files.
  src_files = [f for f in src_files if
               re.search(r'\.cpp$|\.cc$|\.c$|\.cxx$|\.C$', f)]
  srcs = ' '.join(src_files)
  out = ' -o ' + out_file[0]

  mccopts = mcc_compiler_options + ' -x musa  --offload-arch=mp_31' #TODO(perfxlab): need to adjust
  # In mcc-clang environment, we need to make sure that musa header is included
  # before some standard math header like <complex> is included in any source.
  # Otherwise, we get build error.
  # Also we need to retain warning about uninitialised shared variable as
  # warning only, even when -Werror option is specified.
  mccopts += ' --include=musa_runtime.h '
  # Force C++17 dialect (note, everything in just one string!)
  mccopts += ' --std=c++17 '
  # Use -fno-gpu-rdc by default for early GPU kernel finalization
  # This flag would trigger GPU kernels be generated at compile time, instead
  # of link time. This allows the default host compiler (gcc) be used as the
  # linker for TensorFlow on MUSa platform.
  mccopts += ' -fno-gpu-rdc '
  mccopts += ' -fcuda-flush-denormals-to-zero '
  mccopts += undefines
  mccopts += defines
  mccopts += std_options
  mccopts += m_options
  mccopts += ' --musa-path="%{musa_path}" '

  if depfiles:
    # Generate the dependency file
    depfile = depfiles[0]
    cmd = (MCC_PATH + ' ' + mccopts +
           host_compiler_options +
           ' -I .' + includes + ' ' + srcs + ' -M -o ' + depfile)
    cmd = MCC_ENV.replace(';', ' ') + ' ' + cmd
    if log: Log(cmd)
    if VERBOSE: print(cmd)
    exit_status = os.system(cmd)
    if exit_status != 0:
      return exit_status

  cmd = (MCC_PATH + ' ' + mccopts +
         host_compiler_options + ' -fPIC' +
         ' -I .' + opt + includes + ' -c ' + srcs + out)

  cmd = MCC_ENV.replace(';', ' ') + ' '\
        + cmd
  if log: Log(cmd)
  if VERBOSE: print(cmd)
  return system(cmd)


def main():
  # ignore PWD env var
  os.environ['PWD']=''

  parser = ArgumentParser(fromfile_prefix_chars='@')
  parser.add_argument('-x', nargs=1)
  parser.add_argument('--musa_log', action='store_true')
  parser.add_argument('-pass-exit-codes', action='store_true')
  args, leftover = parser.parse_known_args(sys.argv[1:])

  if VERBOSE: print('PWD=' + os.getcwd())
  if VERBOSE: print('MCC_ENV=' + MCC_ENV)

  if args.x and args.x[0] == 'musa':
    # compilation for GPU objects
    if args.musa_log: Log('-x musa')
    leftover = [shlex.quote(s) for s in leftover]
    if args.musa_log: Log('using mcc')
    return InvokeMcc(leftover, log=args.musa_log)

  elif args.pass_exit_codes:
    # link
    # with mcc compiler invoked with -fno-gpu-rdc by default now, it's ok to
    # use host compiler as linker, but we have to link with MUSA runtime.
    # Such restriction would be revised further as the bazel script get
    # improved to fine tune dependencies to MUSa libraries.
    gpu_linker_flags = [flag for flag in sys.argv[1:]
                               if not flag.startswith(('--musa_log'))]

    gpu_linker_flags.append('-L' + MUSA_RUNTIME_PATH)
    gpu_linker_flags.append('-Wl,-rpath=' + MUSA_RUNTIME_PATH)
    gpu_linker_flags.append('-l' + MUSA_RUNTIME_LIBRARY)
    gpu_linker_flags.append("-lrt")
    gpu_linker_flags.append("-lstdc++")

    if VERBOSE: print(' '.join([CPU_COMPILER] + gpu_linker_flags))
    return subprocess.call([CPU_COMPILER] + gpu_linker_flags)

  else:
    # compilation for host objects

    # Strip our flags before passing through to the CPU compiler for files which
    # are not -x musa. We can't just pass 'leftover' because it also strips -x.
    # We not only want to pass -x to the CPU compiler, but also keep it in its
    # relative location in the argv list (the compiler is actually sensitive to
    # this).
    cpu_compiler_flags = [flag for flag in sys.argv[1:]
                               if not flag.startswith(('--musa_log'))]

    # XXX: SE codes need to be built with gcc, but need this macro defined
    cpu_compiler_flags.append("-D__MUSA_PLATFORM_HCC__")
    if VERBOSE: print(' '.join([CPU_COMPILER] + cpu_compiler_flags))
    return subprocess.call([CPU_COMPILER] + cpu_compiler_flags)

if __name__ == '__main__':
  sys.exit(main())
