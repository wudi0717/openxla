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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_EVENT_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_EVENT_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "musa_runtime.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// This class implements Event for ROCm devices.
class MusaEvent : public Event {
 public:
  Event::Status PollForStatus() override;
  absl::Status WaitForEventOnExternalStream(std::intptr_t stream) override;

  // Creates a new MusaEvent. If allow_timing is false, the event will not
  // support timing, which is cheaper to create.
  static absl::StatusOr<MusaEvent> Create(StreamExecutor* executor,
                                          bool allow_timing);

  musaEvent_t GetHandle() const { return handle_; }

  ~MusaEvent() override;
  MusaEvent(const MusaEvent&) = delete;
  MusaEvent& operator=(const MusaEvent&) = delete;
  MusaEvent(MusaEvent&& other);
  MusaEvent& operator=(MusaEvent&& other);

 private:
  explicit MusaEvent(StreamExecutor* executor, musaEvent_t handle)
      : executor_(executor), handle_(handle) {}

  // The Executor used to which this object and musaEvent_t are bound.
  StreamExecutor* executor_;

  // The underlying CUDA event handle.
  musaEvent_t handle_;
};
}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_EVENT_H_
