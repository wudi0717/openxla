/* Copyright 2015 The OpenXLA Authors.

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

// The MUSA-specific DNN library support, implementing the general DnnSupport
// interface.

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_DNN_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_DNN_H_

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/numeric_options.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
namespace gpu {

// miopen-library based DNN support. For details on overridden interface
// functions, see dnn.h.
class MudnnSupport : public dnn::DnnSupport {
 public:
  explicit MudnnSupport(StreamExecutor* parent);

  absl::Status Init() override;
  absl::StatusOr<stream_executor::dnn::VersionInfo> GetVersion() override;

  absl::StatusOr<std::unique_ptr<dnn::RnnDescriptor>> CreateRnnDescriptor(
      int num_layers, int hidden_size, int input_size, int cell_size,
      int batch_size, dnn::RnnInputMode input_mode,
      dnn::RnnDirectionMode direction_mode, dnn::RnnMode rnn_mode,
      dnn::DataType data_type, const dnn::AlgorithmConfig& algorithm_config,
      const NumericOptions& numeric_options, float dropout, uint64_t seed,
      ScratchAllocator* state_allocator, bool use_padded_io) override;

  absl::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
  CreateRnnSequenceTensorDescriptor(int seq_length, int batch_size,
                                    int data_size,
                                    dnn::DataType data_type) override;

  absl::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>
  CreateRnnStateTensorDescriptor(int num_layer, int batch_size, int data_size,
                                 dnn::DataType data_type) override;

  absl::Status DoConvolve(
      dnn::ConvolutionKind kind, dnn::DataType element_type,
      dnn::DataType output_type, Stream* stream,
      const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceMemoryBase filter_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemoryBase output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      dnn::AlgorithmDesc algorithm_desc, DeviceMemory<uint8_t> scratch_memory,
      dnn::ProfileResult* output_profile_result) override;

  absl::Status DoPoolForward(dnn::DataType element_type, Stream* stream,
                             const dnn::PoolingDescriptor& pooling_dimensions,
                             const dnn::BatchDescriptor& input_dimensions,
                             DeviceMemoryBase input_data,
                             const dnn::BatchDescriptor& output_dimensions,
                             DeviceMemoryBase output_data,
                             ScratchAllocator* workspace_allocator) override;

  absl::Status DoPoolBackward(dnn::DataType element_type, Stream* stream,
                              const dnn::PoolingDescriptor& pooling_dimensions,
                              const dnn::BatchDescriptor& input_dimensions,
                              DeviceMemoryBase input_data,
                              const dnn::BatchDescriptor& output_dimensions,
                              DeviceMemoryBase output_data,
                              DeviceMemoryBase input_diff_data,
                              DeviceMemoryBase output_diff_data,
                              ScratchAllocator* workspace_allocator) override;

 private:
  StreamExecutor* parent_;  // Parent executor object. Not owned.

  // Flag to indicate whether Get*Algorithm routines should only return
  // the best algorithm (as opposed to a list of all applicable ones)
  bool return_best_algo_only_;

  // Flag to indicate whether to use Immediate (or Find) mode for Convolutions
  bool use_immediate_mode_;

  bool m_pooling_cache_allowed = false;
  bool m_pooling_cache_enabled = false;

  MudnnSupport(const MudnnSupport&) = delete;
  void operator=(const MudnnSupport&) = delete;
};
}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_DNN_H_
