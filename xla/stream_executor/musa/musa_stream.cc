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

#include "xla/stream_executor/musa/musa_stream.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/base/casts.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "driver_types.h"
#include "musa_runtime.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/musa/musa_driver_wrapper.h"
#include "xla/stream_executor/musa/musa_context.h"
#include "xla/stream_executor/musa/musa_event.h"
#include "xla/stream_executor/musa/musa_kernel.h"
#include "xla/stream_executor/musa/musa_status.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {
int GetGpuStreamPriority(stream_executor::StreamPriority stream_priority) {
  if (stream_priority == stream_executor::StreamPriority::Default) {
    return 0;
  }
  int lowest, highest;
  auto status = musa::ToStatus(muCtxGetStreamPriorityRange(&lowest, &highest));
  if (!status.ok()) {
    LOG(ERROR)
        << "Could not query stream priority range. Returning default priority.";
    return 0;
  }
  return stream_priority == stream_executor::StreamPriority::Highest ? highest
                                                                     : lowest;
}

absl::StatusOr<MUstream> CreateStream(StreamExecutor* executor, int priority) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  MUstream stream;
  // If the priority is 0, then use the previous api to create the stream with
  // the default priority for backward compatibility. Probably there is no
  // difference in using the new api call but leaving it as is for now.
  if (priority == 0) {
    TF_RETURN_IF_ERROR(
        musa::ToStatus(muStreamCreate(&stream, MU_STREAM_NON_BLOCKING)));
  } else {
    TF_RETURN_IF_ERROR(musa::ToStatus(
        muStreamCreateWithPriority(&stream, MU_STREAM_NON_BLOCKING, priority)));
  }

  VLOG(2) << "successfully created stream " << stream << " for executor "
          << executor << " on thread";
  return stream;
}

absl::StatusOr<bool> StreamIsCapturing(MUstream stream) {
  VLOG(2) << "Checking if stream " << stream << " is capturing";

  MUstreamCaptureStatus status;
  TF_RETURN_IF_ERROR(musa::ToStatus(muStreamIsCapturing(stream, &status),
                                    "Failed to check stream capturing status"));

  return status == MU_STREAM_CAPTURE_STATUS_ACTIVE;
}


absl::Status WaitStreamOnEvent(StreamExecutor* executor, MUstream stream,
                               MUevent event) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  TF_RETURN_IF_ERROR(
      musa::ToStatus(muStreamWaitEvent(stream, event, 0 /* = flags */),
               "could not wait stream on event"));
  return absl::OkStatus();
}

absl::Status RecordGpuEvent(StreamExecutor* executor, MUevent event,
                            MUstream stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  return musa::ToStatus(muEventRecord(event, stream),
                        "Error recording MUSA event");
}

absl::Status AsynchronousMemcpyD2H(StreamExecutor* executor, void* host_dst,
                                   MUdeviceptr gpu_src, uint64_t size,
                                   MUstream stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  TF_RETURN_IF_ERROR(
      musa::ToStatus(muMemcpyDtoHAsync(host_dst, gpu_src, size, stream)));

  VLOG(2) << "successfully enqueued async memcpy d2h of " << size
          << " bytes from " << absl::bit_cast<void*>(gpu_src) << " to "
          << host_dst << " on stream " << stream;
  return absl::OkStatus();
}

absl::Status AsynchronousMemcpyH2D(StreamExecutor* executor,
                                   MUdeviceptr gpu_dst, const void* host_src,
                                   uint64_t size, MUstream stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  TF_RETURN_IF_ERROR(
      musa::ToStatus(muMemcpyHtoDAsync(gpu_dst, host_src, size, stream)));

  VLOG(2) << "successfully enqueued async memcpy h2d of " << size << " bytes"
          << " from " << host_src << " to " << absl::bit_cast<void*>(gpu_dst)
          << " on stream " << stream;
  return absl::OkStatus();
}

absl::Status AsynchronousMemcpyD2D(StreamExecutor* executor,
                                   MUdeviceptr gpu_dst, MUdeviceptr gpu_src,
                                   uint64_t size, MUstream stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  // In graph capture mode we never have operations that access peer memory, so
  // we can always make a call to muMemcpyDtoDAsync.
  TF_ASSIGN_OR_RETURN(bool is_capturing, StreamIsCapturing(stream));

  if ((gpu_dst == 0 || gpu_src == 0) || is_capturing) {
    // GetContextMap()->GetAnyContext() doesn't work when ptr == 0.
    // This happens when the size is 0.
    TF_RETURN_IF_ERROR(
        musa::ToStatus(muMemcpyDtoDAsync(gpu_dst, gpu_src, size, stream)));
  } else {
    // Any context work here.
    MUcontext dst_context = MusaContext::GetContextMap()->GetAnyContext(
        absl::bit_cast<void*>(gpu_dst));
    MUcontext src_context = MusaContext::GetContextMap()->GetAnyContext(
        absl::bit_cast<void*>(gpu_src));

    if (dst_context == src_context) {
      // Since the CUDA context is the same, the src and dst are within the same
      // GPU. So we can use cuMemcpyDtoD.
      TF_RETURN_IF_ERROR(
          musa::ToStatus(muMemcpyDtoDAsync(gpu_dst, gpu_src, size, stream)));
    } else {
      TF_RETURN_IF_ERROR(musa::ToStatus(muMemcpyPeerAsync(
          gpu_dst, dst_context, gpu_src, src_context, size, stream)));
    }
  }

  VLOG(2) << "successfully enqueued async memcpy d2d of " << size << " bytes"
          << " from " << absl::bit_cast<void*>(gpu_src) << " to "
          << absl::bit_cast<void*>(gpu_dst) << " on stream " << stream;
  return absl::OkStatus();
}

absl::Status SynchronizeStream(StreamExecutor* executor, MUstream stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  CHECK(stream != nullptr);
  TF_RETURN_IF_ERROR(musa::ToStatus(muStreamSynchronize(stream),
                              "Could not synchronize on MUSA stream"));
  VLOG(2) << "successfully synchronized stream " << stream << " on device "
          << executor->device_ordinal();
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<MusaStream>> MusaStream::Create(
    StreamExecutor* executor,
    std::optional<std::variant<StreamPriority, int>> priority) {
  int stream_priority = [&]() {
    if (priority.has_value() && std::holds_alternative<int>(priority.value())) {
      return std::get<int>(priority.value());
    }
    std::unique_ptr<ActivateContext> activation = executor->Activate();
    return GetGpuStreamPriority(
        std::get<StreamPriority>(priority.value_or(StreamPriority::Default)));
  }();
  TF_ASSIGN_OR_RETURN(auto stream_handle,
                      CreateStream(executor, stream_priority));

  TF_ASSIGN_OR_RETURN(auto completed_event,
                      MusaEvent::Create(executor,
                                        /*allow_timing=*/false));

  return std::unique_ptr<MusaStream>(new MusaStream(
      executor, std::move(completed_event), priority, stream_handle));
}

absl::Status MusaStream::WaitFor(Stream* other) {
  MusaStream* other_stream = static_cast<MusaStream*>(other);

  TF_RETURN_IF_ERROR(other_stream->RecordCompletedEvent());
  return WaitStreamOnEvent(executor_, stream_handle_,
                           other_stream->completed_event_.GetHandle());
}

absl::Status MusaStream::RecordEvent(Event* event) {
  return RecordGpuEvent(executor_, static_cast<MusaEvent*>(event)->GetHandle(),
                        stream_handle_);
}

absl::Status MusaStream::WaitFor(Event* event) {
  return WaitStreamOnEvent(executor_, stream_handle_,
                           static_cast<MusaEvent*>(event)->GetHandle());
}

absl::Status MusaStream::RecordCompletedEvent() {
  return RecordEvent(&completed_event_);
}

namespace {
void DestroyStream(StreamExecutor* executor, MUstream stream) {
  if (stream == nullptr) {
    return;
  }

  std::unique_ptr<ActivateContext> activation = executor->Activate();
  MUresult res = muStreamQuery(stream);
  if (res != MUSA_SUCCESS) {
    LOG(ERROR) << "stream not idle on destroy: " << musa::ToStatus(res);
  }

  auto status = musa::ToStatus(muStreamDestroy(stream));
  if (!status.ok()) {
    LOG(ERROR) << "failed to destroy MUSA stream for executor " << executor
               << ": " << status;
  } else {
    VLOG(2) << "successfully destroyed stream " << stream << " for executor "
            << executor;
  }
}
}  // namespace

MusaStream::~MusaStream() {
  BlockHostUntilDone().IgnoreError();
  executor_->DeallocateStream(this);

  DestroyStream(executor_, stream_handle_);
}

absl::Status MusaStream::Memset32(DeviceMemoryBase* location, uint32_t pattern,
                                  uint64_t size) {
  if (absl::bit_cast<uintptr_t>(location->opaque()) % alignof(uint32_t) != 0) {
    return absl::InvalidArgumentError("location must be 4 byte aligned.");
  }
  if (size % sizeof(uint32_t) != 0) {
    return absl::InvalidArgumentError("size must be a multiple of 4 bytes.");
  }
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  return musa::ToStatus(
      muMemsetD32Async(absl::bit_cast<MUdeviceptr>(location->opaque()), pattern,
                       size / 4, stream_handle_),
      "Failed to enqueue async memset operation");
}

absl::Status MusaStream::MemZero(DeviceMemoryBase* location, uint64_t size) {
  if (reinterpret_cast<uintptr_t>(location->opaque()) % alignof(uint32_t) ==
          0 &&
      size % sizeof(uint32_t) == 0) {
    return Memset32(location, 0x0, size);
  } else {
    std::unique_ptr<ActivateContext> activation = executor_->Activate();
    return musa::ToStatus(
        muMemsetD8Async(absl::bit_cast<MUdeviceptr>(location->opaque()), 0x0,
                        size, stream_handle_),
        "Failed to enqueue async memset operation");
  }
}

absl::Status MusaStream::Memcpy(DeviceMemoryBase* gpu_dst,
                                const DeviceMemoryBase& gpu_src,
                                uint64_t size) {
  return AsynchronousMemcpyD2D(
      executor_, absl::bit_cast<MUdeviceptr>(gpu_dst->opaque()),
      absl::bit_cast<MUdeviceptr>(gpu_src.opaque()), size, stream_handle_);
}

absl::Status MusaStream::Memcpy(DeviceMemoryBase* gpu_dst, const void* host_src,
                                uint64_t size) {
  return AsynchronousMemcpyH2D(
      executor_, absl::bit_cast<MUdeviceptr>(gpu_dst->opaque()), host_src,
      size, stream_handle_);
}

absl::Status MusaStream::Memcpy(void* host_dst, const DeviceMemoryBase& gpu_src,
                                uint64_t size) {
  return AsynchronousMemcpyD2H(executor_, host_dst,
                               absl::bit_cast<MUdeviceptr>(gpu_src.opaque()),
                               size, stream_handle_);
}

namespace {
void InternalHostCallback(void* data) {
  auto* callback = reinterpret_cast<absl::AnyInvocable<void() &&>*>(data);
  std::move (*callback)();
  delete callback;
}
}  // namespace

absl::Status MusaStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback) {
  auto callback_ptr = new absl::AnyInvocable<void() &&>(
      [cb = std::move(callback), this]() mutable {
        absl::Status s = (std::move(cb))();
        if (!s.ok()) {
          LOG(ERROR) << "Host callback failed: " << s;
        }
        int num_pending_host_callbacks = num_pending_host_callbacks_.fetch_sub(
                                             1, std::memory_order_acq_rel) -
                                         1;
        // num_pending_host_callbacks_ can theoretically reach -1 if this
        // callback gets executed before we increase the counter on the main
        // thread.
        if (num_pending_host_callbacks == 0) {
          absl::MutexLock lock(&mutex_);
          no_pending_host_callbacks_ = num_pending_host_callbacks_ <= 0;
        }
      });
  TF_RETURN_IF_ERROR(musa::ToStatus(
      muLaunchHostFunc(stream_handle_, InternalHostCallback, callback_ptr)));
  int num_pending_host_callbacks =
      num_pending_host_callbacks_.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (num_pending_host_callbacks == 1) {
    // num_pending_host_callbacks == 1 means we had no pending host callbacks
    // before this one.
    absl::MutexLock lock(&mutex_);
    no_pending_host_callbacks_ = num_pending_host_callbacks_ <= 0;
  }
  return absl::OkStatus();
}

namespace {
absl::Status LaunchMusaKernel(
    StreamExecutor* executor, absl::string_view kernel_name,
    MUfunction function, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_mem_bytes, MUstream stream,
    void** kernel_params, void** extra) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  VLOG(2) << "launching kernel: " << kernel_name << "; gdx: " << grid_dim_x
          << " gdy: " << grid_dim_y << " gdz: " << grid_dim_z
          << " bdx: " << block_dim_x << " bdy: " << block_dim_y
          << " bdz: " << block_dim_z
          << "; shared_mem_bytes: " << shared_mem_bytes;

  // TODO(ezhulenev): Why do we do it on every call to launch kernel? This
  // should be moved one level up to se::Kernel level, and done just once (or
  // updated once we get a new larger shared memory request).
  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(musa::ToStatus(
        muFuncSetAttribute(function,
                           MU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_mem_bytes),
        "Failed to set shared memory size"));
    TF_RETURN_IF_ERROR(musa::ToStatus(
        muFuncSetCacheConfig(function, MU_FUNC_CACHE_PREFER_SHARED)));
  }

  return musa::ToStatus(
      muLaunchKernel(function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x,
                     block_dim_y, block_dim_z, shared_mem_bytes, stream,
                     kernel_params, extra),
      absl::StrCat("Failed to launch MUSA kernel: ", kernel_name,
                   "; block dims: ", block_dim_x, "x", block_dim_y, "x",
                   block_dim_z, "; grid dims: ", grid_dim_x, "x", grid_dim_y,
                   "x", grid_dim_z,
                   "; shared memory size: ", shared_mem_bytes));
}

absl::Status LaunchMusaKernel(
    StreamExecutor* executor, absl::string_view kernel_name,
    MUfunction function, unsigned int cluster_dim_x, unsigned int cluster_dim_y,
    unsigned int cluster_dim_z, unsigned int grid_dim_x,
    unsigned int grid_dim_y, unsigned int grid_dim_z, unsigned int block_dim_x,
    unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, MUstream stream, void** kernel_params,
    void** extra) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  VLOG(2) << "launching kernel: " << kernel_name << "; cdx: " << cluster_dim_x
          << " cdy: " << cluster_dim_y << " cdz: " << cluster_dim_z
          << " gdx: " << grid_dim_x << " gdy: " << grid_dim_y
          << " gdz: " << grid_dim_z << " bdx: " << block_dim_x
          << " bdy: " << block_dim_y << " bdz: " << block_dim_z
          << "; shared_mem_bytes: " << shared_mem_bytes;

  // TODO(ezhulenev): Why do we do it on every call to launch kernel? This
  // should be moved one level up to se::Kernel level, and done just once (or
  // updated once we get a new larger shared memory request).
  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(musa::ToStatus(
        muFuncSetAttribute(function,
                           MU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_mem_bytes),
        "Failed to set shared memory size"));
    TF_RETURN_IF_ERROR(musa::ToStatus(
        muFuncSetCacheConfig(function, MU_FUNC_CACHE_PREFER_SHARED)));
  }

  MUlaunchConfig launch_config;
  memset(&launch_config, 0, sizeof(launch_config));
  launch_config.blockDimX = block_dim_x;
  launch_config.blockDimY = block_dim_y;
  launch_config.blockDimZ = block_dim_z;
  launch_config.gridDimX = grid_dim_x;
  launch_config.gridDimY = grid_dim_y;
  launch_config.gridDimZ = grid_dim_z;
  launch_config.hStream = stream;
  launch_config.sharedMemBytes = shared_mem_bytes;

  MUlaunchAttribute cluster_dims;
  memset(&cluster_dims, 0, sizeof(cluster_dims));
  cluster_dims.id = MU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  cluster_dims.value.clusterDim.x = cluster_dim_x;
  cluster_dims.value.clusterDim.y = cluster_dim_y;
  cluster_dims.value.clusterDim.z = cluster_dim_z;

  launch_config.attrs = &cluster_dims;
  launch_config.numAttrs = 1;

  return musa::ToStatus(
      muLaunchKernelEx(&launch_config, function, kernel_params, extra),
      absl::StrCat("Failed to launch MUSA kernel: ", kernel_name,
                   "; cluster dims: ", cluster_dim_x, "x", cluster_dim_y, "x",
                   cluster_dim_z, "; block dims: ", block_dim_x, "x",
                   block_dim_y, "x", block_dim_z, "; grid dims: ", grid_dim_x,
                   "x", grid_dim_y, "x", grid_dim_z,
                   "; shared memory size: ", shared_mem_bytes));
}
}  // namespace

absl::Status MusaStream::BlockHostUntilDone() {
  TF_RETURN_IF_ERROR(SynchronizeStream(executor_, stream_handle_));
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(&no_pending_host_callbacks_));
  return absl::OkStatus();
}

absl::Status MusaStream::LaunchKernel(
    const ThreadDim& thread_dims, const BlockDim& block_dims,
    const std::optional<ClusterDim>& cluster_dims, void* function,
    absl::string_view name, void** args, int64_t shmem_bytes) {
  if (cluster_dims.has_value()) {
    return LaunchMusaKernel(executor_, name, static_cast<MUfunction>(function),
                            cluster_dims->x, cluster_dims->y, cluster_dims->z,
                            block_dims.x, block_dims.y, block_dims.z,
                            thread_dims.x, thread_dims.y, thread_dims.z,
                            shmem_bytes, stream_handle_, args,
                            /*extra=*/nullptr);
  } else {
    return LaunchMusaKernel(executor_, name, static_cast<MUfunction>(function),
                            block_dims.x, block_dims.y, block_dims.z,
                            thread_dims.x, thread_dims.y, thread_dims.z,
                            shmem_bytes, stream_handle_, args,
                            /*extra=*/nullptr);
  }
}

}  // namespace stream_executor::gpu
