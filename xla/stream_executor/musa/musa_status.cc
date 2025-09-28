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

namespace stream_executor::gpu {

// Formats musaError_t to output prettified values into a log stream.
// Error summaries taken from:
std::string ToString(musaError_t result) {
#define OSTREAM_MUSA_ERROR(__name) \
  case musaError##__name:           \
    return "MUSA_ERROR_" #__name;

  switch (result) {
    OSTREAM_MUSA_ERROR(InvalidValue)
    OSTREAM_MUSA_ERROR(OutOfMemory)
    OSTREAM_MUSA_ERROR(NotInitialized)
    OSTREAM_MUSA_ERROR(Deinitialized)
    OSTREAM_MUSA_ERROR(NoDevice)
    OSTREAM_MUSA_ERROR(InvalidDevice)
    OSTREAM_MUSA_ERROR(InvalidImage)
    OSTREAM_MUSA_ERROR(InvalidContext)
    OSTREAM_MUSA_ERROR(InvalidHandle)
    OSTREAM_MUSA_ERROR(NotFound)
    OSTREAM_MUSA_ERROR(NotReady)
    OSTREAM_MUSA_ERROR(NoBinaryForGpu)

    // Encountered an uncorrectable ECC error during execution.
    OSTREAM_MUSA_ERROR(ECCNotCorrectable)

    // Load/store on an invalid address. Must reboot all context.
    case 700:
      return "MUSA_ERROR_ILLEGAL_ADDRESS";
    // Passed too many / wrong arguments, too many threads for register count.
    case 701:
      return "MUSA_ERROR_LAUNCH_OUT_OF_RESOURCES";
      OSTREAM_MUSA_ERROR(ContextAlreadyInUse)
      OSTREAM_MUSA_ERROR(PeerAccessUnsupported)
      OSTREAM_MUSA_ERROR(Unknown)  // Unknown internal error to MUSA.

    default:
      return absl::StrCat("musaError_t(", static_cast<int>(result), ")");
  }
#undef OSTREAM_MUSA_ERROR
}

std::string ToString(MUresult result) {
#define OSTREAM_MUSA_ERROR(__name) \
  case MUSA_ERROR_##__name:           \
    return "MUSA_ERROR_" #__name;

  switch (result) {
    OSTREAM_MUSA_ERROR(INVALID_VALUE)
    OSTREAM_MUSA_ERROR(OUT_OF_MEMORY)
    OSTREAM_MUSA_ERROR(NOT_INITIALIZED)
    OSTREAM_MUSA_ERROR(DEINITIALIZED)
    OSTREAM_MUSA_ERROR(NO_DEVICE)
    OSTREAM_MUSA_ERROR(INVALID_DEVICE)
    OSTREAM_MUSA_ERROR(INVALID_IMAGE)
    OSTREAM_MUSA_ERROR(INVALID_CONTEXT)
    OSTREAM_MUSA_ERROR(INVALID_HANDLE)
    OSTREAM_MUSA_ERROR(NOT_FOUND)
    OSTREAM_MUSA_ERROR(NOT_READY)
    OSTREAM_MUSA_ERROR(NO_BINARY_FOR_GPU)

    // Encountered an uncorrectable ECC error during execution.
    OSTREAM_MUSA_ERROR(ECC_UNCORRECTABLE)

    // Load/store on an invalid address. Must reboot all context.
    case 700:
      return "MUSA_ERROR_ILLEGAL_ADDRESS";
    // Passed too many / wrong arguments, too many threads for register count.
    case 701:
      return "MUSA_ERROR_LAUNCH_OUT_OF_RESOURCES";
      OSTREAM_MUSA_ERROR(CONTEXT_ALREADY_IN_USE)
      OSTREAM_MUSA_ERROR(PEER_ACCESS_UNSUPPORTED)
      OSTREAM_MUSA_ERROR(UNKNOWN)  // Unknown internal error to MUSA.

    default:
      return absl::StrCat("musaError_t(", static_cast<int>(result), ")");
  }
#undef OSTREAM_MUSA_ERROR
}

namespace internal {
absl::Status ToStatusSlow(musaError_t result, absl::string_view detail) {
  std::string error_message = absl::StrCat(detail, ": ", ToString(result));
  if (result == musaErrorOutOfMemory) {
    return absl::ResourceExhaustedError(error_message);
  }
  return absl::InternalError(error_message);
}
absl::Status ToStatusSlow(MUresult result, absl::string_view detail) {
  std::string error_message = absl::StrCat(detail, ": ", ToString(result));
  if (result == MUSA_ERROR_OUT_OF_MEMORY) {
    return absl::ResourceExhaustedError(error_message);
  }
  return absl::InternalError(error_message);
}
}  // namespace internal

}  // namespace stream_executor::gpu
