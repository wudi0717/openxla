/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/musa/musa_status.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "driver_types.h"
#include "musa_runtime.h"

namespace stream_executor::musa::internal {

absl::Status ToStatusSlow(MUresult result, absl::string_view detail) {
  const char* error_name;
  std::string error_detail;
  if (muGetErrorName(result, &error_name)) {
    error_detail = absl::StrCat(detail, ": UNKNOWN ERROR (",
                                static_cast<int>(result), ")");
  } else {
    const char* error_string;
    if (muGetErrorString(result, &error_string)) {
      error_detail = absl::StrCat(detail, ": ", error_name);
    } else {
      error_detail = absl::StrCat(detail, ": ", error_name, ": ", error_string);
    }
  }

  if (result == MUSA_ERROR_OUT_OF_MEMORY) {
    return absl::ResourceExhaustedError(error_detail);
  } else if (result == MUSA_ERROR_NOT_FOUND) {
    return absl::NotFoundError(error_detail);
  } else {
    return absl::InternalError(absl::StrCat("MUSA error: ", error_detail));
  }
}

absl::Status ToStatusSlow(musaError_t result, absl::string_view detail) {
  std::string error_detail(detail);
  const char* error_name = musaGetErrorName(result);
  const char* error_string = musaGetErrorString(result);
  if (error_name == nullptr) {
    absl::StrAppend(&error_detail, ": UNKNOWN ERROR (",
                    static_cast<int>(result), ")");
  } else {
    absl::StrAppend(&error_detail, ": ", error_name);
  }

  if (error_string != nullptr) {
    absl::StrAppend(&error_detail, ": ", error_string);
  }

  return absl::InternalError(
      absl::StrCat("MUSA Runtime error: ", error_detail));
}

}  // namespace stream_executor::musa::internal
