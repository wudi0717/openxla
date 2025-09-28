/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/stream_executor/musa/musa_event.h"

#include <cstdint>
#include <memory>

#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "musa_runtime.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/musa/musa_driver_wrapper.h"
#include "xla/stream_executor/musa/musa_status.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {
namespace gpu {
namespace {
absl::Status WaitStreamOnEvent(StreamExecutor *executor, musaStream_t stream,
                               musaEvent_t event) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  TF_RETURN_IF_ERROR(
      ToStatus(musaStreamWaitEvent(stream, event, 0 /* = flags */),
               "could not wait stream on event"));
  return absl::OkStatus();
}

enum class EventFlags { kDefault, kDisableTiming };
absl::StatusOr<musaEvent_t> InitEvent(StreamExecutor *executor,
                                     EventFlags flags) {
  int hipflags;
  switch (flags) {
    case EventFlags::kDefault:
      hipflags = musaEventDefault;
      break;
    case EventFlags::kDisableTiming:
      hipflags = musaEventDisableTiming;
      break;
    default:
      LOG(FATAL) << "impossible event flags: " << int(hipflags);
  }

  std::unique_ptr<ActivateContext> activation = executor->Activate();
  musaEvent_t event;
  musaError_t res = musaEventCreateWithFlags(&event, hipflags);

  if (res == musaSuccess) {
    return event;
  }
  if (res == musaErrorMemoryAllocation) {
    return absl::ResourceExhaustedError(
        "could not create MUSA event: out of device memory");
  }
  return absl::FailedPreconditionError(
      absl::StrCat("could not create MUSA event: ", ToString(res)));
}

void DestroyEvent(StreamExecutor *executor, musaEvent_t event) {
  if (event == nullptr) {
    return;
  }

  std::unique_ptr<ActivateContext> activation = executor->Activate();
  musaError_t res = musaEventDestroy(event);

  if (res != musaSuccess) {
    LOG(ERROR) << absl::StrFormat(
        "error destroying MUSA event in device %d: %s",
        executor->device_ordinal(), ToString(res));
  }
}

}  // namespace

Event::Status MusaEvent::PollForStatus() {
  std::unique_ptr<ActivateContext> activated = executor_->Activate();
  musaError_t res = musaEventQuery(handle_);

  if (res == musaSuccess) {
    return Event::Status::kComplete;
  } else if (res == musaErrorNotReady) {
    return Event::Status::kPending;
  }

  return Event::Status::kError;
}

absl::Status MusaEvent::WaitForEventOnExternalStream(std::intptr_t stream) {
  return WaitStreamOnEvent(executor_, absl::bit_cast<musaStream_t>(stream),
                           handle_);
}

absl::StatusOr<MusaEvent> MusaEvent::Create(StreamExecutor *executor,
                                            bool allow_timing) {
  TF_ASSIGN_OR_RETURN(
      musaEvent_t event_handle,
      InitEvent(executor, allow_timing ? EventFlags::kDefault
                                       : EventFlags::kDisableTiming));

  return MusaEvent(executor, event_handle);
}

MusaEvent::~MusaEvent() { DestroyEvent(executor_, handle_); }

MusaEvent::MusaEvent(MusaEvent &&other)
    : executor_(other.executor_), handle_(other.handle_) {
  other.executor_ = nullptr;
  other.handle_ = nullptr;
}

MusaEvent& MusaEvent::operator=(MusaEvent&& other) {
  if (this == &other) {
    return *this;
  }

  DestroyEvent(executor_, handle_);

  executor_ = other.executor_;
  handle_ = other.handle_;
  other.executor_ = nullptr;
  other.handle_ = nullptr;
  return *this;
}
}  // namespace gpu
}  // namespace stream_executor
