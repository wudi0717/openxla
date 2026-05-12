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

#include "xla/stream_executor/musa/musa_dnn.h"

#include <musa_bf16.h>
#include <musa_fp16.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "mudnn.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/numeric_options.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/musa/musa_diagnostics.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/determinism.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/platform/hash.h"

#ifndef PLATFORM_GOOGLE
#include "xla/tsl/platform/env.h"
#include "tsl/platform/dso_loader.h"
#endif

namespace stream_executor {
namespace gpu {

MudnnSupport::MudnnSupport(StreamExecutor* parent) : parent_(parent) {
  // by default, the Get*Algorithm API will return the list of all applicable
  // algorithms
  return_best_algo_only_ = false;
  // by default, use Find Mode APIs for convolution
  use_immediate_mode_ = false;
  // swich to Find Mode if env var TF_ROCM_USE_IMMEDIATE_MODE is set

}

absl::Status MudnnSupport::Init() {
  return absl::OkStatus();
}

absl::StatusOr<stream_executor::dnn::VersionInfo> MudnnSupport::GetVersion() {
  return stream_executor::dnn::VersionInfo(1, 3, 0);
}

absl::StatusOr<std::unique_ptr<dnn::RnnDescriptor>>
MudnnSupport::CreateRnnDescriptor(
    int num_layers, int hidden_size, int input_size, int cell_size,
    int batch_size, dnn::RnnInputMode input_mode,
    dnn::RnnDirectionMode direction_mode, dnn::RnnMode rnn_mode,
    dnn::DataType data_type, const dnn::AlgorithmConfig& algorithm_config,
    const NumericOptions& numeric_options, float dropout, uint64_t seed,
    ScratchAllocator* state_allocator, bool use_padded_io) {
  return absl::UnimplementedError("CreateRnnDescriptor not implemented yet");
}
absl::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
MudnnSupport::CreateRnnSequenceTensorDescriptor(int seq_length, int batch_size,
                                                 int data_size,
                                                 dnn::DataType data_type) {
  return absl::UnimplementedError("CreateRnnSequenceTensorDescriptor not implemented yet");
}
absl::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>
MudnnSupport::CreateRnnStateTensorDescriptor(int num_layer, int batch_size,
                                              int data_size,
                                              dnn::DataType data_type) {
  return absl::UnimplementedError("CreateRnnStateTensorDescriptor not implemented yet");
}
absl::Status MudnnSupport::DoConvolve(
    dnn::ConvolutionKind kind, dnn::DataType element_type,
    dnn::DataType output_type, Stream* stream,
    const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemoryBase filter_data, const dnn::BatchDescriptor& output_descriptor,
    DeviceMemoryBase output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    dnn::AlgorithmDesc algorithm_desc, DeviceMemory<uint8_t> scratch_memory,
    dnn::ProfileResult* output_profile_result) {
  return absl::UnimplementedError("DoConvolve not implemented yet");
}
absl::Status MudnnSupport::DoPoolForward(
    dnn::DataType element_type, Stream* stream,
    const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions, DeviceMemoryBase input_data,
    const dnn::BatchDescriptor& output_dimensions, DeviceMemoryBase output_data,
    ScratchAllocator* workspace_allocator) {
  return absl::UnimplementedError("DoPoolForward not implemented yet");
}
absl::Status MudnnSupport::DoPoolBackward(
    dnn::DataType element_type, Stream* stream,
    const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions, DeviceMemoryBase input_data,
    const dnn::BatchDescriptor& output_dimensions, DeviceMemoryBase output_data,
    DeviceMemoryBase input_diff_data, DeviceMemoryBase output_diff_data,
    ScratchAllocator* workspace_allocator) {
  return absl::UnimplementedError("DoPoolBackward not implemented yet");
}

}  // namespace gpu

void initialize_mudnn() {
  auto mudnnAlreadyRegistered = PluginRegistry::Instance()->HasFactory(
      musa::kMUSaPlatformId, PluginKind::kDnn);

  if (!mudnnAlreadyRegistered) {
    absl::Status status =
        PluginRegistry::Instance()->RegisterFactory<PluginRegistry::DnnFactory>(
            musa::kMUSaPlatformId, "muDNN",
            [](StreamExecutor* parent) -> dnn::DnnSupport* {
              gpu::MudnnSupport* dnn = new gpu::MudnnSupport(parent);
              if (!dnn->Init().ok()) {
                // Note: Init() will log a more specific error.
                delete dnn;
                return nullptr;
              }
              return dnn;
            });

    if (!status.ok()) {
      LOG(ERROR) << "Unable to register Mudnn factory: " << status.message();
    }
  }
}

}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(register_mudnn, {
  stream_executor::initialize_mudnn();
});
