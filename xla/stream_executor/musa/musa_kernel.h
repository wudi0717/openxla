/* Copyright 2019 The OpenXLA Authors.

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

// The MUSA implementation of the StreamExecutor functionality.
// MUSA inclusions are ideally confined to this implementation file.
//
// The notions from the StreamExecutor basically correspond to the MUSA streams
// programming model provided by the libmusa.so driver APIs, so we don't have
// to do much more than wrap the calls to the libraries appropriately.
#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_KERNEL_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_KERNEL_H_

#include <cstddef>
#include <cstdint>

#include "absl/status/statusor.h"
#include "musa.h"
#include "musa_runtime.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/logging.h"

namespace stream_executor::gpu {

class MusaKernel : public Kernel {
 public:
  explicit MusaKernel(StreamExecutor* executor) : executor_(executor) {}

  // Note that the function is unloaded when the module is unloaded, and the
  // module that the function is contained in is owned by the StreamExecutor.
  ~MusaKernel() override { executor_->UnloadKernel(this); }

  // As arity cannot be reflected upon using the MUSA API, the arity is
  // explicitly set during the StreamExecutor::GetKernel initialization process.
  void set_arity(unsigned arity) { arity_ = arity; }
  unsigned Arity() const override { return arity_; }

  absl::StatusOr<int32_t> GetMaxOccupiedBlocksPerCore(
      ThreadDim threads, size_t dynamic_shared_memory_bytes) const override;

  // Simple accessor methods.
  MUfunction gpu_function() const { return gpu_function_; }
  void set_gpu_function(MUfunction gpu_function) {
    gpu_function_ = gpu_function;
  }

  // Collects metadata for the specified kernel.
  absl::StatusOr<KernelMetadata> GetKernelMetadata();

 private:
  absl::Status Launch(const ThreadDim &thread_dims, const BlockDim &block_dims,
                      const std::optional<ClusterDim> &cluster_dims,
                      Stream *stream, const KernelArgs &args) override;

  StreamExecutor* executor_ = nullptr;

  MUfunction gpu_function_ = nullptr;  // wrapped MUSA kernel handle
  unsigned arity_ = 0;  // number of formal parameters the kernel takes
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_KERNEL_H_
