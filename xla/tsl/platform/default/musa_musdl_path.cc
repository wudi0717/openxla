/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/tsl/platform/musa_musdl_path.h"

#include <stdlib.h>

#include "tsl/platform/path.h"

#include "xla/tsl/platform/logging.h"

namespace tsl {

std::string MusaRoot() {
#if TENSORFLOW_USE_MUSA
  if (const char* musa_path_env = std::getenv("MUSA_PATH")) {
    VLOG(3) << "MUSA root = " << musa_path_env;
    return musa_path_env;
  }
#else
  return "";
#endif
}

std::string MusdlRoot() {
  if (const char* device_lib_path_env = std::getenv("MUSA_DEVICE_LIB_PATH")) {
    return device_lib_path_env;
  } else {
    return io::JoinPath(MusaRoot(), "mtgpu/bitcode");
  }
}

}  // namespace tsl
