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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_STATUS_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_STATUS_H_

#include <string>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "musa.h"
#include "musa_runtime.h"

namespace stream_executor::musa {

namespace internal {
// Helper method to handle the slow path of ToStatus.  Assumes a non-successful
// result code.
absl::Status ToStatusSlow(MUresult result, absl::string_view detail);
absl::Status ToStatusSlow(musaError_t result, absl::string_view detail);
}  // namespace internal

// Returns an absl::Status corresponding to the MUresult.
inline absl::Status ToStatus(MUresult result, absl::string_view detail = "") {
  if (ABSL_PREDICT_TRUE(result == MUSA_SUCCESS)) {
    return absl::OkStatus();
  }
  return internal::ToStatusSlow(result, detail);
}

// Returns an absl::Status corresponding to the musaError_t (MUSA runtime API
// error type). The string `detail` will be included in the error message.
inline absl::Status ToStatus(musaError_t result,
                             absl::string_view detail = "") {
  if (ABSL_PREDICT_TRUE(result == musaSuccess)) {
    return absl::OkStatus();
  }
  return internal::ToStatusSlow(result, detail);
}

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_STATUS_H_
