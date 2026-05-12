#include "absl/status/statusor.h"
/* Copyright 2023 The OpenXLA Authors.

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

// The MUSA-specific Driver library support, implementing the general Driver
// interface.

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_CONTEXT_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_CONTEXT_H_

#include <cstdint>

#include "absl/status/status.h"
#include "musa.h"
#include "musa_runtime.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/context_map.h"

namespace stream_executor::gpu {

// MusaContext implements the Context class for MT GPUs.
class MusaContext : public Context {
 public:
  MusaContext(MUcontext context, int device_ordinal)
      : context_(context), device_ordinal_(device_ordinal) {}
  ~MusaContext() override;

  void SetActive() override;
  bool IsActive() const override;
  MUcontext context() const { return context_; }
  int device_ordinal() const override { return device_ordinal_; }
  absl::Status Synchronize() override;

  // Disallow copying and moving.
  MusaContext(MusaContext&&) = delete;
  MusaContext(const MusaContext&) = delete;
  MusaContext& operator=(MusaContext&&) = delete;
  MusaContext& operator=(const MusaContext&) = delete;

  // Returns a new context for the given device.
  static absl::StatusOr<MusaContext*> Create(int device_ordinal,
                                             MUdevice device);

  // Returns the context map for all XLA-known CUDA contexts.
  static ContextMap<MUcontext, MusaContext>* GetContextMap();

 private:
  MUcontext const context_;
  const int device_ordinal_;
};
}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_CONTEXT_H_
