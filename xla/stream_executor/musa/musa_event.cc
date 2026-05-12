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
absl::Status WaitStreamOnEvent(StreamExecutor *executor, MUstream stream,
                               MUevent event) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  return musa::ToStatus(muStreamWaitEvent(stream, event, 0 /* = flags */));
}

void DestroyEvent(StreamExecutor *executor, MUevent event) {
  if (event == nullptr) {
    return;
  }

  std::unique_ptr<ActivateContext> activation = executor->Activate();
  auto result =
      musa::ToStatus(muEventDestroy(event), "Error destroying MUSA event");
  if (!result.ok()) {
    LOG(ERROR) << result.message();
  }
}

enum class EventFlags { kDefault, kDisableTiming };
absl::StatusOr<MUevent> InitEvent(StreamExecutor *executor, EventFlags flags) {
  int muflags;
  switch (flags) {
    case EventFlags::kDefault:
      muflags = MU_EVENT_DEFAULT;
      break;
    case EventFlags::kDisableTiming:
      muflags = MU_EVENT_DISABLE_TIMING;
      break;
    default:
      LOG(FATAL) << "impossible event flags: " << int(flags);
  }

  std::unique_ptr<ActivateContext> activation = executor->Activate();
  MUevent event_handle;
  TF_RETURN_IF_ERROR(musa::ToStatus(muEventCreate(&event_handle, muflags)));
  return event_handle;
}

}  // namespace

Event::Status MusaEvent::PollForStatus() {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  MUresult res = muEventQuery(handle_);
  if (res == MUSA_SUCCESS) {
    return Event::Status::kComplete;
  } else if (res == MUSA_ERROR_NOT_READY) {
    return Event::Status::kPending;
  }
  return Event::Status::kError;
}

absl::Status MusaEvent::WaitForEventOnExternalStream(std::intptr_t stream) {
  return WaitStreamOnEvent(executor_, absl::bit_cast<MUstream>(stream),
                           handle_);
}

absl::StatusOr<MusaEvent> MusaEvent::Create(StreamExecutor *executor,
                                            bool allow_timing) {
  TF_ASSIGN_OR_RETURN(
      MUevent event_handle,
      InitEvent(executor, allow_timing ? EventFlags::kDefault
                                       : EventFlags::kDisableTiming));

  return MusaEvent(executor, event_handle);
}

MusaEvent::~MusaEvent() { DestroyEvent(executor_, handle_); }

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

MusaEvent::MusaEvent(MusaEvent &&other)
    : executor_(other.executor_), handle_(other.handle_) {
  other.executor_ = nullptr;
  other.handle_ = nullptr;
}

}  // namespace gpu
}  // namespace stream_executor
