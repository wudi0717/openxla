# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Prints MUSa library and header directories and versions found on the system.

The script searches for MUSa library and header files on the system, inspects
them to determine their version and prints the configuration to stdout.
The path to inspect is specified through an environment variable (MUSA_PATH).
If no valid configuration is found, the script prints to stderr and
returns an error code.

The script takes the directory specified by the MUSA_PATH environment variable.
The script looks for headers and library files in a hard-coded set of
subdirectories from base path of the specified directory. If MUSA_PATH is not
specified, then "/opt/musa" is used as it default value

"""

import io
import os
import re
import sys


class ConfigError(Exception):
  pass


def _get_default_musa_path():
  return "/usr/local/musa"


def _get_musa_install_path():
  """Determines and returns the MUSa installation path."""
  musa_install_path = _get_default_musa_path()
  if "MUSA_PATH" in os.environ:
    musa_install_path = os.environ["MUSA_PATH"]
  # musa_install_path = os.path.realpath(musa_install_path)
  return musa_install_path


def _get_composite_version_number(major, minor, patch):
  return 10000 * major + 100 * minor + patch


def _get_header_version(path, name):
  """Returns preprocessor defines in C header file."""
  for line in io.open(path, "r", encoding="utf-8"):
    match = re.match(r"#define %s +(\d+)" % name, line)
    if match:
      value = match.group(1)
      return int(value)

  raise ConfigError('#define "{}" is either\n'.format(name) +
                    "  not present in file {} OR\n".format(path) +
                    "  its value is not an integer literal")


def _find_musa_config(musa_install_path):

  def musa_version_numbers(path):
    possible_version_files = [
        "include/musa.h",  # MUSa 5.1 and prior
    ]
    version_file = None
    for f in possible_version_files:
      version_file_path = os.path.join(path, f)
      if os.path.exists(version_file_path):
        version_file = version_file_path
        break
    if not version_file:
      raise ConfigError(
          "MUSa version file not found in {}".format(possible_version_files))

    version = _get_header_version(version_file, "MUSA_VERSION")
    return version

  musa_config = {
      "musa_version_number": musa_version_numbers(musa_install_path)
  }

  return musa_config



def _find_mublas_config(musa_install_path):

  def mublas_version_numbers(path):
    possible_version_files = [
        "include/internal/mublas-version.h",  # MUSa 5.2
    ]
    version_file = None
    for f in possible_version_files:
      version_file_path = os.path.join(path, f)
      if os.path.exists(version_file_path):
        version_file = version_file_path
        break
    if not version_file:
      raise ConfigError(
          "mublas version file not found in {}".format(
              possible_version_files))
    major = _get_header_version(version_file, "MUBLAS_VERSION_MAJOR")
    minor = _get_header_version(version_file, "MUBLAS_VERSION_MINOR")
    patch = _get_header_version(version_file, "MUBLAS_VERSION_PATCH")
    return major, minor, patch

  major, minor, patch = mublas_version_numbers(musa_install_path)

  mublas_config = {
      "mublas_version_number":
          _get_composite_version_number(major, minor, patch)
  }

  return mublas_config


def _find_murand_config(musa_install_path):

  def murand_version_number(path):
    possible_version_files = [
        "include/murand_version.h",  # MUSa 5.1
    ]
    version_file = None
    for f in possible_version_files:
      version_file_path = os.path.join(path, f)
      if os.path.exists(version_file_path):
        version_file = version_file_path
        break
    if not version_file:
      raise ConfigError(
          "murand version file not found in {}".format(possible_version_files))
    version_number = _get_header_version(version_file, "MURAND_VERSION")
    return version_number

  murand_config = {
      "murand_version_number": murand_version_number(musa_install_path)
  }

  return murand_config

def _find_musparse_config(musa_install_path):

  def musparse_version_numbers(path):
    possible_version_files = [
        "include/musparse-version.h",  # MUSa 5.2
    ]
    version_file = None
    for f in possible_version_files:
      version_file_path = os.path.join(path, f)
      if os.path.exists(version_file_path):
        version_file = version_file_path
        break
    if not version_file:
      raise ConfigError("musparse version file not found in {}".format(
          possible_version_files))
    major = _get_header_version(version_file, "MUSPARSE_VERSION_MAJOR")
    minor = _get_header_version(version_file, "MUSPARSE_VERSION_MINOR")
    patch = _get_header_version(version_file, "MUSPARSE_VERSION_PATCH")
    return major, minor, patch

  major, minor, patch = musparse_version_numbers(musa_install_path)

  musparse_config = {
      "musparse_version_number":
          _get_composite_version_number(major, minor, patch)
  }

  return musparse_config

def _find_musolver_config(musa_install_path):

  def musolver_version_numbers(path):
    possible_version_files = [
        "include/musolver_version.h",  # MUSa 5.2
    ]
    version_file = None
    for f in possible_version_files:
      version_file_path = os.path.join(path, f)
      if os.path.exists(version_file_path):
        version_file = version_file_path
        break
    if not version_file:
      raise ConfigError("musolver version file not found in {}".format(
          possible_version_files))
    major = _get_header_version(version_file, "MUSOLVER_VERSION_MAJOR")
    minor = _get_header_version(version_file, "MUSOLVER_VERSION_MINOR")
    patch = _get_header_version(version_file, "MUSOLVER_VERSION_PATCH")
    return major, minor, patch

  major, minor, patch = musolver_version_numbers(musa_install_path)

  musolver_config = {
      "musolver_version_number":
          _get_composite_version_number(major, minor, patch)
  }

  return musolver_config


def find_musa_config():
  """Returns a dictionary of MUSa components config info."""
  musa_install_path = _get_musa_install_path()
  if not os.path.exists(musa_install_path):
    raise ConfigError(
        'Specified MUSA_PATH "{}" does not exist'.format(musa_install_path))

  result = {}

  result["musa_toolkit_path"] = musa_install_path
  result.update(_find_musa_config(musa_install_path))
  result.update(_find_mublas_config(musa_install_path))
  result.update(_find_murand_config(musa_install_path))
  result.update(_find_musparse_config(musa_install_path))
  result.update(_find_musolver_config(musa_install_path))

  return result


def main():
  try:
    for key, value in sorted(find_musa_config().items()):
      print("%s: %s" % (key, value))
  except ConfigError as e:
    sys.stderr.write("\nERROR: {}\n\n".format(str(e)))
    sys.exit(1)


if __name__ == "__main__":
  main()
