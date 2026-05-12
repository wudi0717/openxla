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

#include "xla/stream_executor/musa/musa_executor.h"

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstdlib>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "driver_types.h"
#include "musa_runtime.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/generic_memory_allocation.h"
#include "xla/stream_executor/generic_memory_allocator.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/read_numa_node.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/musa/musa_command_buffer.h"
#include "xla/stream_executor/musa/musa_context.h"
#include "xla/stream_executor/musa/musa_driver_wrapper.h"
#include "xla/stream_executor/musa/musa_event.h"
#include "xla/stream_executor/musa/musa_kernel.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/musa/musa_status.h"
#include "xla/stream_executor/musa/musa_stream.h"
#include "xla/stream_executor/musa/musa_timer.h"
#include "xla/stream_executor/musa/musa_version_parser.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/numbers.h"

namespace stream_executor {
namespace gpu {

class AlignedGPUAllocator {
public:
    static void* alloc(size_t size) {
        // 分配额外空间用于对齐
	if (size == 0) return nullptr;

        MUdeviceptr raw;
        auto status = musa::ToStatus(muMemAlloc(&raw, size + 256));
        if (!status.ok()) {
          LOG(INFO) << "failed to allocate "
              << tsl::strings::HumanReadableNumBytes(size) << " (" <<size 
              << " bytes) from device: " << status;
          return nullptr;
        }

        // 计算对齐地址
        long long unsigned int addr = reinterpret_cast<long long unsigned int>(raw);
        VLOG(2) << " muMemAlloc alloc " << size << " bytes, return " << addr;
	if ((addr & 0xff) == 0) return reinterpret_cast<void*>(addr);

	//Handle non-aligned
        long long unsigned int aligned_addr = (addr + 255) & ~255;
        void* aligned_ptr = reinterpret_cast<void*>(aligned_addr);

        // 加锁保护映射表
        std::lock_guard<std::mutex> lock(mutex_);
        g_addr_map[aligned_ptr] = reinterpret_cast<void*>(raw);
        return aligned_ptr;
    }

    static void free(void* aligned_ptr) {
        if (!aligned_ptr) return;

        std::lock_guard<std::mutex> lock(mutex_);
        auto it = g_addr_map.find(aligned_ptr);
        if (it != g_addr_map.end()) {
            void* raw = it->second;
            g_addr_map.erase(it);
            MUdeviceptr pointer = absl::bit_cast<MUdeviceptr>(raw);
            auto status = musa::ToStatus(muMemFree(pointer));
            if (!status.ok()) {
              LOG(ERROR) << " failed to free device memory at " << raw 
                << "; result: " << status;
            } else {
              VLOG(2) << " deallocated " << raw;
          }
        } else {
            MUdeviceptr pointer = absl::bit_cast<MUdeviceptr>(aligned_ptr);
            auto status = musa::ToStatus(muMemFree(pointer));
            if (!status.ok()) {
              LOG(ERROR) << " failed to free device memory at " << aligned_ptr
                << "; result: " << status;
            } else {
              VLOG(2) << " deallocated " << aligned_ptr;
          }
        }
    }

private:
    static std::unordered_map<void*, void*> g_addr_map;
    static std::mutex mutex_;
};

std::unordered_map<void*, void*> AlignedGPUAllocator::g_addr_map;
std::mutex AlignedGPUAllocator::mutex_;

namespace {
bool ShouldLaunchDelayKernel() {
  // Only launch the delay kernel if MUSA_LAUNCH_BLOCKING is not set to 1.
  static bool value = [] {
    const char* blocking = std::getenv("MUSA_LAUNCH_BLOCKING");
    return !blocking || absl::string_view{blocking} != "1";
  }();
  return value;
}

// MUSA driver routines may require a large amount of stack (particularly
// muModuleLoadDataEx, in our experience). To avoid stack overflow when using
// stack-limited threads (such as those spawned by a default-argument
// thread::ThreadPool on some platforms), we run certain routines in this pool
// and wait for completion.
tsl::thread::ThreadPool* GetDriverExecutor() {
  static tsl::thread::ThreadPool* const thread_pool =
      new tsl::thread::ThreadPool(tsl::Env::Default(), tsl::ThreadOptions(),
                                  "musa_driver", 1);
  return thread_pool;
}

absl::StatusOr<MUmodule> LoadPtx(Context* context, const char* ptx_contents) {
  absl::Notification notification;
  absl::Status returned_status = absl::OkStatus();
  MUmodule module;
  GetDriverExecutor()->Schedule(
      [context, ptx_contents, &module, &returned_status, &notification]() {
        ScopedActivateContext activation(context);
        void* ptx_data = const_cast<char*>(ptx_contents);
        static const unsigned int kLogBufferBytesLimit = 1024;
        unsigned int error_log_buffer_bytes = kLogBufferBytesLimit;
        unsigned int info_log_buffer_bytes = kLogBufferBytesLimit;
        absl::InlinedVector<char, 4> error_log_buffer(error_log_buffer_bytes);
        absl::InlinedVector<char, 4> info_log_buffer(info_log_buffer_bytes);
        bool log_verbose = true;
        MUjit_option options[] = {MU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                                  MU_JIT_ERROR_LOG_BUFFER,
                                  MU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                                  MU_JIT_INFO_LOG_BUFFER, MU_JIT_LOG_VERBOSE};
        // Note that the driver API wants the contents of this values to be
        // stored in an array of void*s, so we coerce them accordingly.
        void* option_values[] = {
            absl::bit_cast<void*>(uintptr_t(error_log_buffer_bytes)),
            absl::bit_cast<void*>(error_log_buffer.data()),
            absl::bit_cast<void*>(uintptr_t(info_log_buffer_bytes)),
            absl::bit_cast<void*>(info_log_buffer.data()),
            absl::bit_cast<void*>(uintptr_t(log_verbose))};
        CHECK(TF_ARRAYSIZE(options) == TF_ARRAYSIZE(option_values));

        absl::Status status;
        status = musa::ToStatus(muModuleLoadDataEx(
            &module, ptx_data, TF_ARRAYSIZE(options), options, option_values));

        // The PTX JIT mutates the values in the option values array to reflect
        // the size of the logs it output; now that we've made the call, read
        // the values back out.
        error_log_buffer_bytes = reinterpret_cast<uintptr_t>(option_values[0]);
        info_log_buffer_bytes = reinterpret_cast<uintptr_t>(option_values[2]);
        CHECK_LE(error_log_buffer_bytes, kLogBufferBytesLimit);
        CHECK_LE(info_log_buffer_bytes, kLogBufferBytesLimit);

        if (!status.ok()) {
          LOG(ERROR) << "[" << context->device_ordinal()
                     << "] failed to load PTX text as a module: " << status;
          // As a precaution for null termination of the API-provided value,
          // ensure that at least the last byte is null.
          error_log_buffer[error_log_buffer_bytes ? error_log_buffer_bytes - 1
                                                  : 0] = '\0';
          LOG(ERROR) << "[" << context->device_ordinal()
                     << "] error log buffer (" << error_log_buffer_bytes
                     << " bytes): " << error_log_buffer.data();
          if (absl::StrContains(error_log_buffer.data(),
                                "Register allocation failed")) {
            returned_status = absl::ResourceExhaustedError(absl::StrFormat(
                "[%d] Failed to load PTX text as a module (register "
                "allocation failed): %s",
                context->device_ordinal(), status.ToString()));
          } else {
            returned_status = status;
          }
          notification.Notify();
          return;
        }

        VLOG(3) << "[" << context->device_ordinal()
                << "] PTX compilation info log (" << info_log_buffer_bytes
                << " bytes): " << info_log_buffer.data();
        VLOG(3) << "[" << context->device_ordinal()
                << "] PTX compilation error log (" << error_log_buffer_bytes
                << " bytes): " << error_log_buffer.data();
        CHECK(module != nullptr);
        notification.Notify();
      });
  notification.WaitForNotification();

  TF_RETURN_IF_ERROR(returned_status);
  return module;
}

// Loads mubin_bytes with the MUSA driver's blob loading interface and stores
// the resulting handle in "module".
absl::StatusOr<MUmodule> LoadMubin(Context* context, const char* mubin_bytes) {
  ScopedActivateContext activation(context);
  MUmodule module;
  TF_RETURN_IF_ERROR(
      musa::ToStatus(muModuleLoadFatBinary(&module, mubin_bytes),
                     absl::StrFormat("[%d] Failed to load in-memory MUBIN "
                                     "(compiled for a different GPU?).",
                                     context->device_ordinal())));
  return module;
}

// Retrieves a named kernel from a loaded module, and return the MUfunction
// handle on success. Neither kernel_name nor function may be null. No ownership
// is taken of kernel_name.
absl::StatusOr<MUfunction> GetModuleFunction(Context* context, MUmodule module,
                                             const char* kernel_name) {
  ScopedActivateContext activated{context};
  CHECK(module != nullptr && kernel_name != nullptr);
  musaError_t musa_error = musaPeekAtLastError();
  if (musa_error != musaSuccess) {
    return absl::InternalError(absl::StrCat(
        "[", context->device_ordinal(),
        "] There was an error before calling muModuleGetFunction (", musa_error,
        "): ", musaGetErrorName(musa_error), " : ",
        musaGetErrorString(musa_error)));
  }
  MUfunction function;
  TF_RETURN_IF_ERROR(
      musa::ToStatus(muModuleGetFunction(&function, module, kernel_name),
                     absl::StrFormat("[%d] Failed to get module function",
                                     context->device_ordinal())));
  return function;
}

// Retrieves a named global/constant symbol from a loaded module, and returns
// a device pointer and size of the symbol on success. symbol_name may not be
// null. At least one of dptr or bytes should not be null. No ownership is
// taken of symbol_name.
absl::Status GetModuleSymbol(Context* context, MUmodule module,
                             const char* symbol_name, MUdeviceptr* dptr,
                             size_t* bytes) {
  ScopedActivateContext activated{context};
  CHECK(module != nullptr && symbol_name != nullptr &&
        (dptr != nullptr || bytes != nullptr));
  return musa::ToStatus(
      muModuleGetGlobal(dptr, bytes, module, symbol_name),
      absl::StrCat("Failed to get symbol '", symbol_name, "'"));
}

// Unloads module from the current context via muModuleUnload.
void UnloadMusaModule(Context* context, MUmodule module) {
  ScopedActivateContext activated{context};
  auto status = musa::ToStatus(muModuleUnload(module));
  if (!status.ok()) {
    LOG(ERROR) << "failed to unload module " << module
               << "; leaking: " << status;
  }
}

// Returns the integer output of muDeviceGetAttribute.
absl::StatusOr<int> GetDeviceAttribute(MUdevice_attribute attribute,
                                       MUdevice device) {
  int val;
  TF_RETURN_IF_ERROR(
      musa::ToStatus(muDeviceGetAttribute(&val, attribute, device)));
  return val;
}

// Returns the name of the device.
absl::StatusOr<std::string> GetDeviceName(MUdevice device) {
  static const size_t kCharLimit = 64;
  absl::InlinedVector<char, 4> chars(kCharLimit);
  TF_RETURN_IF_ERROR(
      musa::ToStatus(muDeviceGetName(chars.begin(), kCharLimit - 1, device),
                     "Failed to get device name"));
  chars[kCharLimit - 1] = '\0';
  return chars.begin();
}

// Returns the compute capability for the device; i.e (3, 5).
absl::StatusOr<MusaComputeCapability> GetComputeCapability(MUdevice device) {
  int cc_major = 0;
  TF_RETURN_IF_ERROR(musa::ToStatus(muDeviceGetAttribute(
      &cc_major, MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)));

  int cc_minor = 0;
  TF_RETURN_IF_ERROR(musa::ToStatus(muDeviceGetAttribute(
      &cc_minor, MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)));

  return MusaComputeCapability(
      cc_major, cc_minor);
}

// Helper function that turns the integer output of muDeviceGetAttribute to type
// T and wraps it in a absl::StatusOr.
template <typename T>
static absl::StatusOr<T> GetSimpleAttribute(MUdevice device,
                                            MUdevice_attribute attribute) {
  int value = -1;
  TF_RETURN_IF_ERROR(musa::ToStatus(
      muDeviceGetAttribute(&value, attribute, device),
      absl::StrCat("Could not retrieve MUSA device attribute (", attribute)));
  T converted = value;
  return converted;
}

// Returns the number of multiprocessors on the device (note that the device
// may be multi-GPU-per-board).
absl::StatusOr<int> GetMultiprocessorCount(MUdevice device) {
  return GetSimpleAttribute<int>(device,
                                 MU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
}

absl::StatusOr<int64_t> GetMaxSharedMemoryPerCore(MUdevice device) {
  return GetSimpleAttribute<int64_t>(
      device, MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
}

absl::StatusOr<int64_t> GetMaxSharedMemoryPerBlock(MUdevice device) {
  return GetSimpleAttribute<int64_t>(
      device, MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
}

absl::StatusOr<int64_t> GetMaxSharedMemoryPerBlockOptin(MUdevice device) {
  return GetSimpleAttribute<int64_t>(
      device, MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
}

absl::StatusOr<int64_t> GetMaxThreadsPerMultiprocessor(MUdevice device) {
  return GetSimpleAttribute<int64_t>(
      device, MU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
}

absl::StatusOr<int64_t> GetMaxRegistersPerBlock(MUdevice device) {
  return GetSimpleAttribute<int64_t>(
      device, MU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
}

absl::StatusOr<int64_t> GetThreadsPerWarp(MUdevice device) {
  return GetSimpleAttribute<int64_t>(device, MU_DEVICE_ATTRIBUTE_WARP_SIZE);
}

absl::Status GetGridLimits(int* x, int* y, int* z, MUdevice device) {
  int value;
  TF_RETURN_IF_ERROR(musa::ToStatus(
      muDeviceGetAttribute(&value, MU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device),
      "Could not get device attribute"));
  *x = value;

  TF_RETURN_IF_ERROR(musa::ToStatus(
      muDeviceGetAttribute(&value, MU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device),
      "Could not get device attribute"));
  *y = value;

  TF_RETURN_IF_ERROR(musa::ToStatus(
      muDeviceGetAttribute(&value, MU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device),
      "Could not get device attribute"));
  *z = value;
  return absl::OkStatus();
}

// Returns the device associated with the given device_ordinal.
absl::StatusOr<MUdevice> GetDevice(int device_ordinal) {
  MUdevice device;
  TF_RETURN_IF_ERROR(musa::ToStatus(muDeviceGet(&device, device_ordinal),
                                    "Failed call to muDeviceGet"));
  return device;
}

// Returns the device associated with the given context.
absl::StatusOr<MUdevice> DeviceFromContext(Context* context) {
  ScopedActivateContext activated{context};
  MUdevice device = -1;
  auto status = musa::ToStatus(muCtxGetDevice(&device));
  if (status.ok()) {
    return device;
  }

  return status;
}

bool CanEnablePeerAccess(MUdevice from, MUdevice to) {
  int can_access_peer = -1;
  auto status =
      musa::ToStatus(muDeviceCanAccessPeer(&can_access_peer, from, to));
  if (!status.ok()) {
    LOG(ERROR) << "failed to detect peer access capability: " << status;
    return false;
  }
  return can_access_peer;
}

bool CanEnablePeerAccess(Context* from, Context* to) {
  if (from == to) {
    return true;  // A context can always access its own memory.
  }

  auto from_device = DeviceFromContext(from);
  if (!from_device.ok()) {
    LOG(ERROR) << "failed to resolve 'from' peer access context to a device: "
               << from_device.status();
    return false;
  }
  auto to_device = DeviceFromContext(to);
  if (!to_device.ok()) {
    LOG(ERROR) << "failed to resolve 'to' peer access context to a device: "
               << to_device.status();
    return false;
  }
  return CanEnablePeerAccess(from_device.value(), to_device.value());
}

absl::Status EnablePeerAccess(Context* from, Context* to) {
  if (from == to) {
    return absl::OkStatus();  // A context can always access its own
                              // memory.
  }

  ScopedActivateContext activated{from};
  MUresult result = muCtxEnablePeerAccess(
      tensorflow::down_cast<MusaContext*>(to)->context(), 0 /* = flags */);
  if (result != MUSA_SUCCESS &&
      result != MUSA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
    return absl::InternalError(
        absl::StrFormat("failed to enable peer access from %p to %p: %s", from,
                        to, musa::ToStatus(result).ToString()));
  }

  return absl::OkStatus();
}

// Returns the total amount of memory available on the device.
bool GetDeviceTotalMemory(MUdevice device, uint64_t* result) {
  size_t value{};
  auto status = musa::ToStatus(muDeviceTotalMem(&value, device));
  if (!status.ok()) {
    LOG(ERROR) << "failed to query total available memory: " << status;
    return false;
  }

  *result = value;
  return true;
}

bool IsEccEnabled(MUdevice device, bool* result) {
  int value = -1;
  auto status = musa::ToStatus(
      muDeviceGetAttribute(&value, MU_DEVICE_ATTRIBUTE_ECC_ENABLED, device));
  if (!status.ok()) {
    LOG(ERROR) << "failed to query ECC status: " << status;
    return false;
  }

  *result = value;
  return true;
}

std::string GetPCIBusID(MUdevice device) {
  // PCI bus id is of the format [domain]:[bus]:[device].[function], and is 13
  // characters long in practice.
  constexpr int kBufferSize = 64;
  std::array<char, kBufferSize> raw_pci_bus_id{};
  absl::Status status = musa::ToStatus(
      muDeviceGetPCIBusId(raw_pci_bus_id.data(), kBufferSize, device));
  if (!status.ok()) {
    LOG(ERROR) << "failed to query PCI bus id for device: " << status;
    return "";
  }
  if (!absl::c_linear_search(raw_pci_bus_id, '\0')) {
    LOG(ERROR) << "PCI bus id is not null terminated.";
    return "";
  }
  // Lower the hex characters to match sysfs.
  return absl::AsciiStrToLower(absl::string_view(raw_pci_bus_id.data()));
}

bool HostRegister(Context* context, void* location, uint64_t size) {
  ScopedActivateContext activation(context);
  // "Portable" memory is visible to all MUSA contexts. Safe for our use model.
  auto status = musa::ToStatus(
      muMemHostRegister(location, size, MU_MEMHOSTREGISTER_PORTABLE));
  if (!status.ok()) {
    LOG(ERROR) << "error registering host memory at " << location << ": "
               << status;
    return false;
  }
  return true;
}

bool HostUnregister(Context* context, void* location) {
  ScopedActivateContext activation(context);
  auto status = musa::ToStatus(muMemHostUnregister(location));
  if (!status.ok()) {
    LOG(ERROR) << "error unregistering host memory at " << location << ": "
               << status;
    return false;
  }
  return true;
}

// Allocates memory on the GPU device.
void* DeviceAllocate(Context* context, uint64_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }

  ScopedActivateContext activated{context};
  void* ptr = AlignedGPUAllocator::alloc(bytes);;
  VLOG(2) << "[" << context->device_ordinal() << "] allocated " << ptr
          << " for context " << context << " of " << bytes << " bytes";
  return ptr;
}

// Deallocates memory on the GPU device that was previously allocated via
// DeviceAllocate.
void DeviceDeallocate(Context* context, void* location) {
  ScopedActivateContext activation(context);
  AlignedGPUAllocator::free(location);
}

// Allocates memory on the host.
absl::StatusOr<void*> HostAllocate(Context* context, int numa_node,
                                   uint64_t size) {
  if (numa_node != tsl::port::kNUMANoAffinity) {
    // MUSA programming guide: "Any address of a variable ... returned by one
    // of the memory allocation routines from the driver ... API is always
    // aligned to at least 256 bytes."
    auto* buffer =
        tsl::port::NUMAMalloc(numa_node, size, /* minimum_alignment=*/256);
    if (buffer == nullptr && size > 0) {
      return absl::InternalError(
          absl::StrFormat("[%d] Failed to allocate host memory of size %d "
                          "pinned to NUMA node %d",
                          context->device_ordinal(), size, numa_node));
    }
    if (size > 0 && !HostRegister(context, buffer, size)) {
      tsl::port::NUMAFree(buffer, size);
      return absl::InternalError(absl::StrFormat(
          "[%d] Failed to register host memory of size %d pinned to "
          "NUMA node %d with the GPU driver",
          context->device_ordinal(), size, numa_node));
    }
    return buffer;
  } else {
    ScopedActivateContext activation(context);
    void* buffer = nullptr;
    // "Portable" memory is visible to all MUSA contexts. Safe for our use
    // model.
    TF_RETURN_IF_ERROR(musa::ToStatus(
        muMemHostAlloc(&buffer, size, MU_MEMHOSTALLOC_PORTABLE)));
    if (!buffer && size > 0) {
      return absl::InternalError(absl::StrFormat(
          "[%d] Failed to allocate pinned host memory of size %d",
          context->device_ordinal(), size));
    }
    return buffer;
  }
}

// Deallocates memory allocated via HostAllocate.
void HostDeallocate(Context* context, int numa_node, void* location,
                    uint64_t size) {
  if (numa_node != tsl::port::kNUMANoAffinity) {
    if (size > 0) {
      HostUnregister(context, location);
    }
    tsl::port::NUMAFree(location, size);
  } else {
    ScopedActivateContext activation(context);
    auto status = musa::ToStatus(muMemFreeHost(location));
    if (!status.ok()) {
      LOG(ERROR) << "[" << context->device_ordinal()
                 << "] error deallocating host memory at " << location << ": "
                 << status;
    }
  }
}

// Creates a MemoryAllocation wrapping the given host buffer.
absl::StatusOr<std::unique_ptr<MemoryAllocation>> AllocateHostMemory(
    MusaContext* musa_context, int numa_node, uint64_t size) {
  TF_ASSIGN_OR_RETURN(void* ptr, HostAllocate(musa_context, numa_node, size));
  VLOG(2) << "[" << musa_context->device_ordinal() << "] allocated " << ptr
          << " for context " << musa_context << " of " << size
          << " bytes of host memory";
  return std::make_unique<GenericMemoryAllocation>(
      ptr, size, [musa_context, numa_node](void* location, uint64_t size) {
        HostDeallocate(musa_context, numa_node, location, size);
        VLOG(2) << "[" << musa_context->device_ordinal()
                << "] deallocated collective memory at " << location
                << " for context " << musa_context;
      });
}

}  // namespace


// Given const GPU memory, returns a libmusa device pointer datatype, suitable
// for passing directly to libmusa APIs.
//
// N.B. we must lose constness in order to pass a suitable type to the existing
// libmusa APIs, so the caller should take care to only pass the result of const
// GPU memory conversions to libmusa functions which will honor constness.
static MUdeviceptr AsMusaDevicePtr(const DeviceMemoryBase& gpu_mem) {
  return reinterpret_cast<MUdeviceptr>(gpu_mem.opaque());
}

// See description on const version above.
static MUdeviceptr AsMusaDevicePtr(DeviceMemoryBase* gpu_mem) {
  return AsMusaDevicePtr(*gpu_mem);
}

absl::StatusOr<DeviceMemoryBase> MusaExecutor::GetMemoryRange(
    const DeviceMemoryBase& location) {
  MUdeviceptr device_pointer;
  size_t size;
  TF_RETURN_IF_ERROR(musa::ToStatus(
      muMemGetAddressRange(&device_pointer, &size, AsMusaDevicePtr(location))));
  return DeviceMemoryBase(reinterpret_cast<void*>(device_pointer), size);
}

std::unique_ptr<ActivateContext> MusaExecutor::Activate() {
  return std::make_unique<ScopedActivateContext>(musa_context_);
}

MusaExecutor::~MusaExecutor() {
  CHECK(kernel_to_gpu_binary_.empty()) << "MusaExecutor has live kernels.";
  CHECK(gpu_binary_to_module_.empty()) << "MusaExecutor has loaded modules.";
}

absl::StatusOr<xla::gpu::GpuCollectives*> GetGpuCollectives(
    StreamExecutor* executor) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  TF_ASSIGN_OR_RETURN(xla::Collectives * collectives,
                      xla::CollectivesRegistry::Default("gpu"));
  return tsl::down_cast<xla::gpu::GpuCollectives*>(collectives);
}

absl::StatusOr<void*> CollectiveMemoryAllocate(StreamExecutor* executor,
                                               uint64_t bytes) {
  if (bytes == 0) return nullptr;

  std::unique_ptr<ActivateContext> activation = executor->Activate();
  TF_ASSIGN_OR_RETURN(xla::gpu::GpuCollectives * gpu_collectives,
                      GetGpuCollectives(executor));
  return gpu_collectives->Allocate(bytes);
}

absl::Status CollectiveMemoryDeallocate(StreamExecutor* executor,
                                        void* location) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  TF_ASSIGN_OR_RETURN(xla::gpu::GpuCollectives * gpu_collectives,
                      GetGpuCollectives(executor));
  return gpu_collectives->Deallocate(location);
}

absl::StatusOr<std::unique_ptr<MemoryAllocator>>
MusaExecutor::CreateMemoryAllocator(MemoryType type) {
  if (type == MemoryType::kUnified) {
    return std::make_unique<GenericMemoryAllocator>(
        [this](uint64_t size)
            -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
          std::unique_ptr<ActivateContext> activation = Activate();
          MUdeviceptr result = 0;
          // "Portable" memory is visible to all MUSA contexts. Safe for our use
          // model.
          TF_RETURN_IF_ERROR(musa::ToStatus(
              muMemAllocManaged(&result, size, MU_MEM_ATTACH_GLOBAL)));
          void* ptr = reinterpret_cast<void*>(result);
          VLOG(2) << "[" << device_ordinal() << "] allocated " << ptr
                  << " for context " << musa_context_ << " of " << size
                  << " bytes in unified memory";
          return std::make_unique<GenericMemoryAllocation>(
              ptr, size, [this](void* location, uint64_t size) {
                std::unique_ptr<ActivateContext> activation = Activate();
		AlignedGPUAllocator::free(location);
              });
        });
  } else if (type == MemoryType::kCollective) {
    return std::make_unique<GenericMemoryAllocator>(
        [this](uint64_t size)
            -> absl::StatusOr<std::unique_ptr<MemoryAllocation>> {
          TF_ASSIGN_OR_RETURN(void* ptr, CollectiveMemoryAllocate(this, size));
          VLOG(2) << "[" << device_ordinal() << "] allocated " << ptr
                  << " for context " << musa_context_ << " of " << size
                  << " bytes of collective memory";
          return std::make_unique<GenericMemoryAllocation>(
              ptr, size, [this](void* location, uint64_t size) {
                auto status = CollectiveMemoryDeallocate(this, location);
                if (!status.ok()) {
                  LOG(ERROR) << "[" << device_ordinal()
                             << "] failed to free collective memory at "
                             << location << "; result: " << status;
                } else {
                  VLOG(2) << "[" << device_ordinal()
                          << "] deallocated collective memory at " << location
                          << " for context " << musa_context_;
                }
              });
        });
  } else if (type == MemoryType::kHost) {
    return std::make_unique<GenericMemoryAllocator>([this](uint64_t size) {
      return AllocateHostMemory(musa_context_, numa_node_, size);
    });
  }
  return absl::UnimplementedError(
      absl::StrFormat("Unsupported memory type %d", type));
}

absl::Status MusaExecutor::Init() {
  TF_ASSIGN_OR_RETURN(device_, GetDevice(device_ordinal()));
  TF_ASSIGN_OR_RETURN(MusaContext * context,
                      MusaContext::Create(device_ordinal(), device_));
  musa_context_ = context;
  TF_ASSIGN_OR_RETURN(delay_kernels_supported_, DelayKernelIsSupported());
  numa_node_ = ReadNumaNode(GetPCIBusID(device_), device_ordinal())
                   .value_or(tsl::port::kNUMANoAffinity);
  if (numa_node_ == tsl::port::kNUMANoAffinity) {
    VLOG(2) << "[" << device_ordinal() << "] Could not determine NUMA node";
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> MusaExecutor::DelayKernelIsSupported() {
  // Check the assumption that this device supports unified addressing,
  // otherwise skip the delay kernel
  TF_ASSIGN_OR_RETURN(
      int status,
      GetDeviceAttribute(MU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device_));

  return static_cast<bool>(status);
}

absl::StatusOr<ModuleHandle> MusaExecutor::LoadModuleFromMuBin(
    const char* mubin) {
  ModuleHandle module_handle{mubin};
  uint64_t module_refcount;
  MUmodule module;
  std::tie(module, module_refcount) = gpu_binary_to_module_[module_handle];

  if (module == nullptr) {
    TF_ASSIGN_OR_RETURN(module, LoadMubin(musa_context_, mubin));
    module_refcount = 1;
    VLOG(3) << "[" << device_ordinal() << "] Loaded MUBIN "
            << static_cast<const void*>(mubin) << " as module " << module;
  } else {
    ++module_refcount;
    VLOG(3) << "[" << device_ordinal() << "] MUBIN "
            << static_cast<const void*>(mubin)
            << " is already loaded as module " << module;
  }
  gpu_binary_to_module_[module_handle] = {module, module_refcount};
  return module_handle;
}

absl::StatusOr<ModuleHandle> MusaExecutor::LoadModuleFromPtx(const char* ptx) {
  ModuleHandle module_handle{ptx};
  uint64_t module_refcount;
  MUmodule module;
  std::tie(module, module_refcount) = gpu_binary_to_module_[module_handle];

  if (module == nullptr) {
    TF_ASSIGN_OR_RETURN(module, LoadPtx(musa_context_, ptx));
    VLOG(3) << "[" << device_ordinal() << "] Loaded PTX "
            << static_cast<const void*>(ptx) << " as module " << module;
    module_refcount = 1;
  } else {
    ++module_refcount;
    VLOG(3) << "[" << device_ordinal() << "] PTX "
            << static_cast<const void*>(ptx) << " is already loaded as module "
            << module;
  }
  gpu_binary_to_module_[module_handle] = {module, module_refcount};
  return module_handle;
}

absl::StatusOr<std::unique_ptr<Kernel>> MusaExecutor::LoadKernel(
    const KernelLoaderSpec& spec) {
  auto musa_kernel = std::make_unique<MusaKernel>(this);
  const std::string& kernel_name = spec.kernel_name();

  if (spec.has_cuda_cubin_in_memory()) {
    absl::MutexLock lock{&in_memory_modules_mu_};
    const char* mubin = reinterpret_cast<const char*>(
        spec.cuda_cubin_in_memory()->cubin_bytes.data());
    TF_ASSIGN_OR_RETURN(ModuleHandle module_handle, LoadModuleFromMuBin(mubin));
    kernel_to_gpu_binary_[musa_kernel.get()] = module_handle;

    MUmodule module = gpu_binary_to_module_.at(module_handle).first;
    VLOG(2) << "[" << device_ordinal() << "] getting function " << kernel_name
            << " from module " << module;
    TF_ASSIGN_OR_RETURN(
        MUfunction function,
        GetModuleFunction(musa_context_, module, kernel_name.c_str()));
    musa_kernel->set_gpu_function(function);

  } else if (spec.has_cuda_ptx_in_memory()) {
    const char* ptx = spec.cuda_ptx_in_memory()->ptx.data();
    if (ptx == nullptr) {
      LOG(FATAL) << "[" << device_ordinal()
                 << "] Loader spec has no ptx for kernel " << kernel_name;
    }

    absl::MutexLock lock{&in_memory_modules_mu_};
    TF_ASSIGN_OR_RETURN(ModuleHandle module_handle, LoadModuleFromPtx(ptx));
    kernel_to_gpu_binary_[musa_kernel.get()] = module_handle;

    MUmodule module = gpu_binary_to_module_.at(module_handle).first;
    VLOG(2) << "[" << device_ordinal() << "] getting function " << kernel_name
            << " from module " << module;
    TF_ASSIGN_OR_RETURN(
        MUfunction function,
        GetModuleFunction(musa_context_, module, kernel_name.c_str()));
    musa_kernel->set_gpu_function(function);

  } else if (spec.has_in_process_symbol()) {
    void* symbol = spec.in_process_symbol()->symbol;

    VLOG(2) << "[" << device_ordinal() << "] Resolve MUSA kernel "
            << kernel_name << " from symbol pointer: " << symbol;
    musaFunction_t func;
    TF_RETURN_IF_ERROR(musa::ToStatus(
        musaGetFuncBySymbol(&func, symbol),
        absl::StrFormat("[%d] Failed call to musaGetFuncBySymbol",
                        device_ordinal())));
    musa_kernel->set_gpu_function(func);

  } else {
    return absl::InternalError("No method of loading MUSA kernel provided");
  }
  VLOG(3) << "[" << device_ordinal()
          << "] LoadKernel on kernel : " << kernel_name;

  {
    // Keep track of loaded kernels.
    absl::MutexLock lock{&in_memory_modules_mu_};
    loaded_kernels_.insert(musa_kernel.get());
  }

  // Update MUSA kernel properties after it was loaded in the MUSA context.
  musa_kernel->set_name(kernel_name);

  // We have to trust the kernel loader spec arity because there doesn't appear
  // to be a way to reflect on the number of expected arguments w/the MUSA API.
  musa_kernel->set_arity(spec.arity());

  TF_ASSIGN_OR_RETURN(KernelMetadata kernel_metadata,
                      musa_kernel->GetKernelMetadata());
  musa_kernel->set_metadata(kernel_metadata);
  musa_kernel->set_args_packing(spec.kernel_args_packing());
  return std::move(musa_kernel);
}

absl::StatusOr<std::unique_ptr<EventBasedTimer>>
MusaExecutor::CreateEventBasedTimer(Stream* stream, bool use_delay_kernel) {
  const MusaTimer::TimerType timer_type =
      (use_delay_kernel && ShouldLaunchDelayKernel() &&
       delay_kernels_supported_)
          ? MusaTimer::TimerType::kDelayKernel
          : MusaTimer::TimerType::kEventBased;

  TF_ASSIGN_OR_RETURN(MusaTimer timer,
                      MusaTimer::Create(this, stream, timer_type));
  return std::make_unique<MusaTimer>(std::move(timer));
}

bool MusaExecutor::UnloadGpuBinary(ModuleHandle gpu_binary) {
  auto module_it = gpu_binary_to_module_.find(gpu_binary);
  if (gpu_binary_to_module_.end() == module_it) {
    VLOG(3) << "[" << device_ordinal() << "] No loaded MUSA module for "
            << gpu_binary;
    return false;
  }
  auto& module = module_it->second.first;
  auto& refcount = module_it->second.second;
  VLOG(3) << "[" << device_ordinal() << "] Found MUSA module " << module
          << " with refcount " << refcount;
  if (--refcount == 0) {
    VLOG(3) << "[" << device_ordinal() << "] Unloading MUSA module " << module;
    UnloadMusaModule(musa_context_, module);
    gpu_binary_to_module_.erase(module_it);
  }
  return true;
}

void MusaExecutor::UnloadKernel(const Kernel* kernel) {
  VLOG(3) << "[" << device_ordinal() << "] Unloading kernel " << kernel << " : "
          << kernel->name();

  absl::MutexLock lock{&in_memory_modules_mu_};
  loaded_kernels_.erase(kernel);

  auto gpu_binary_it = kernel_to_gpu_binary_.find(kernel);
  if (kernel_to_gpu_binary_.end() == gpu_binary_it) {
    // We might never see kernel being explicitly loaded if it was resolved from
    // in process symbol pointer (MUSA C++ device function pointer).
    VLOG(3) << "[" << device_ordinal() << "] Kernel " << kernel << " : "
            << kernel->name() << " has never been loaded.";
    return;
  }
  VLOG(3) << "[" << device_ordinal() << "] Kernel " << kernel << " : "
          << kernel->name() << " has loaded GPU code " << gpu_binary_it->second;
  UnloadGpuBinary(gpu_binary_it->second);
  kernel_to_gpu_binary_.erase(gpu_binary_it);
}

absl::StatusOr<ModuleHandle> MusaExecutor::LoadModule(
    const MultiModuleLoaderSpec& spec) {
  // We store the pointer to the GPU binary (PTX or MUBIN) as
  // ModuleHandle::id().
  if (spec.has_cuda_cubin_in_memory()) {
    absl::MutexLock lock{&in_memory_modules_mu_};
    return LoadModuleFromMuBin(
        reinterpret_cast<const char*>(spec.cuda_cubin_in_memory().data()));
  } else if (spec.has_cuda_ptx_in_memory()) {
    if (!spec.cuda_ptx_in_memory()) {
      return absl::InternalError("PTX not found in spec");
    }

    absl::MutexLock lock{&in_memory_modules_mu_};
    return LoadModuleFromPtx(spec.cuda_ptx_in_memory());
  }
  return absl::InternalError("No method of loading MUSA module provided");
}

bool MusaExecutor::UnloadModule(ModuleHandle module_handle) {
  absl::MutexLock lock{&in_memory_modules_mu_};
  return UnloadGpuBinary(module_handle);
}

namespace {
absl::uint128 Fingerprint128(const absl::string_view s) {
  auto fp = tsl::Fingerprint128(s);
  return absl::MakeUint128(fp.high64, fp.low64);
}

int fpus_per_core(int cc_major, int cc_minor) {
  // Source:
  // https://docs.nvidia.com/musa/musa-c-programming-guide/index.html#arithmetic-instructions
  int n = 128;          // 5.x, 6.1, 6.2, 8.6, 9.0 -> 128.
  if (cc_major == 3) {  // 3.x -> 192.
    n = 192;
  } else if ((cc_major == 6 && cc_minor == 0) || (cc_major == 7) ||
             (cc_major == 8 && cc_minor == 0)) {
    n = 64;  // 6.0, 7.x, 8.0 -> 64.
  }
  return n;
}

}  // namespace

absl::StatusOr<std::shared_ptr<DeviceMemoryBase>>
MusaExecutor::CreateOrShareConstant(Stream* stream,
                                    absl::Span<const uint8_t> content) {
  absl::MutexLock lock{&shared_constants_mu_};
  // We assume all constants are uniquely identified by this hash. In the
  // (highly unlikely) event of a hash collision, the program will likely crash
  // (because the cached constant that will be returned by mistake is unlikely
  // to have the correct size).
  absl::uint128 fingerprint = Fingerprint128(absl::string_view(
      reinterpret_cast<const char*>(content.data()), content.size()));
  // Must insert nullptr first to get an iterator to the insertion point.
  auto insert_result = shared_constants_.insert(
      {fingerprint, std::weak_ptr<DeviceMemoryBase>()});
  auto it = insert_result.first;
  bool was_already_in_cache = !insert_result.second;
  std::shared_ptr<DeviceMemoryBase> shared_constant;

  if (was_already_in_cache) {
    shared_constant = it->second.lock();
  }

  if (shared_constant == nullptr) {
    // Either the constant wasn't found in the cache, or it was but its
    // weak_ptr had expired.
    auto new_constant = std::make_unique<DeviceMemoryBase>(
        Allocate(content.size(), /*memory_space=*/0));
    if (new_constant->opaque() == nullptr) {
      return absl::InternalError(absl::StrFormat(
          "Failed to allocate %d bytes for new constant", content.size()));
    }

    TF_RETURN_IF_ERROR(
        stream->Memcpy(new_constant.get(), content.data(), content.size()));
    absl::Status status = stream->BlockHostUntilDone();
    if (!status.ok()) {
      Deallocate(new_constant.get());
      status.Update(absl::InternalError(absl::StrFormat(
          "Memcpy to device address %p failed", new_constant->opaque())));
      return status;
    }

    // Capturing 'this' in the custom deleter means this executor must
    // outlive all shared uses of this constant.
    shared_constant = std::shared_ptr<DeviceMemoryBase>(
        new_constant.release(), [this](DeviceMemoryBase* p) {
          Deallocate(p);
          delete p;
        });
    it->second = std::weak_ptr<DeviceMemoryBase>(shared_constant);
  }

  return shared_constant;
}

DeviceMemoryBase MusaExecutor::Allocate(uint64_t size, int64_t memory_space) {
  VLOG(1) << "[" << device_ordinal()
          << "] MusaExecutor::Allocate size: " << size
          << " memory_space: " << memory_space;

  if (memory_space == static_cast<int64_t>(MemoryType::kCollective)) {
    auto result = CollectiveMemoryAllocate(this, size);
    if (!result.ok()) {
      LOG(ERROR) << "Failed to allocate collective memory: " << result.status();
      return DeviceMemoryBase(nullptr, 0);
    }
    VLOG(1) << "[" << device_ordinal() << "] MusaExecutor::Allocate returns "
            << result.value();
    return DeviceMemoryBase(result.value(), size);
  } else if (memory_space ==
             static_cast<int64_t>(stream_executor::MemoryType::kHost)) {
    auto result = HostAllocate(musa_context_, numa_node_, size);
    if (!result.ok()) {
      LOG(ERROR) << "[" << device_ordinal()
                 << "] Failed to allocate host memory: " << result.status();
      return DeviceMemoryBase(nullptr, 0);
    }
    VLOG(1) << "[" << device_ordinal() << "] MusaExecutor::Allocate returns "
            << result.value();
    return DeviceMemoryBase(result.value(), size);
  }
  CHECK_EQ(memory_space, 0);
  auto device_buf_base = DeviceAllocate(musa_context_, size);
  VLOG(1) << "[" << device_ordinal() << "] MusaExecutor::Allocate returns "
          << device_buf_base;
  return DeviceMemoryBase(device_buf_base, size);
}

absl::StatusOr<std::unique_ptr<MemoryAllocation>>
MusaExecutor::HostMemoryAllocate(uint64_t size) {
  return AllocateHostMemory(musa_context_, numa_node_, size);
}

void MusaExecutor::Deallocate(DeviceMemoryBase* mem) {
  VLOG(1) << "[" << device_ordinal()
          << "] MusaExecutor::Deallocate mem: " << mem->opaque();

  auto status_or_memory_space = GetPointerMemorySpace(mem->opaque());
  if (!status_or_memory_space.ok()) {
    LOG(ERROR) << status_or_memory_space.status();
    return;
  }
  auto memory_space = status_or_memory_space.value();
  if (memory_space == MemoryType::kHost) {
    HostDeallocate(musa_context_, numa_node_, mem->opaque(), mem->size());
  } else {
    DeviceDeallocate(musa_context_, mem->opaque());
  }
}

bool MusaExecutor::SynchronizeAllActivity() {
  return musa_context_->Synchronize().ok();
}

bool MusaExecutor::HostMemoryRegister(void* location, uint64_t size) {
  VLOG(1) << "[" << device_ordinal()
          << "] Called StreamExecutor::HostMemoryRegister(data=" << location
          << ")";
  return HostRegister(musa_context_, location, size);
}

bool MusaExecutor::HostMemoryUnregister(void* location) {
  VLOG(1) << "[" << device_ordinal()
          << "] Called StreamExecutor::HostUnregister(data=" << location << ")";
  return HostUnregister(musa_context_, location);
}

absl::Status MusaExecutor::SynchronousMemZero(DeviceMemoryBase* location,
                                              uint64_t size) {
  std::unique_ptr<ActivateContext> activation = Activate();
  MUdeviceptr musa_location = AsMusaDevicePtr(location);
  if (reinterpret_cast<uintptr_t>(location->opaque()) % sizeof(uint32_t) == 0 &&
      size % sizeof(uint32_t) == 0) {
    return musa::ToStatus(
        muMemsetD32(musa_location, 0x0, size / sizeof(uint32_t)),
        "Failed to memset memory");
  }
  return musa::ToStatus(muMemsetD8(musa_location, 0x0, size),
                        "Failed to memset memory");
}

absl::Status MusaExecutor::SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                             const void* host_src,
                                             uint64_t size) {
  std::unique_ptr<ActivateContext> activation = Activate();
  TF_RETURN_IF_ERROR(
      musa::ToStatus(muMemcpyHtoD(AsMusaDevicePtr(gpu_dst), host_src, size),
                     absl::StrFormat("[%d] failed to synchronous memcpy from "
                                     "host to device: GPU dst: %llx;"
                                     " host src: %p; size: %u=0x%x",
                                     device_ordinal(), AsMusaDevicePtr(gpu_dst),
                                     host_src, size, size)));
  VLOG(2) << "[" << device_ordinal()
          << "] successfully enqueued sync memcpy h2d of " << size << " bytes";
  return absl::OkStatus();
}

absl::Status MusaExecutor::SynchronousMemcpy(void* host_dst,
                                             const DeviceMemoryBase& gpu_src,
                                             uint64_t size) {
  std::unique_ptr<ActivateContext> activation = Activate();
  TF_RETURN_IF_ERROR(musa::ToStatus(
      muMemcpyDtoH(host_dst, AsMusaDevicePtr(gpu_src), size),
      absl::StrFormat("[%d] failed to synchronous memcpy from device to host "
                      "host dst: %p; GPU src: %llx; size: %u=0x%x",
                      device_ordinal(), host_dst, AsMusaDevicePtr(gpu_src),
                      size, size)));
  VLOG(2) << "[" << device_ordinal() << "] successfully sync memcpy'd d2h of "
          << size << " bytes to " << host_dst;
  return absl::OkStatus();
}

void MusaExecutor::DeallocateStream(Stream* stream) {
  {
    absl::MutexLock lock(&mu_);
    if (dnn_ != nullptr) {
      dnn_->NotifyStreamDestroyed(stream);
    }
  }
  absl::MutexLock l(&alive_gpu_streams_mu_);
  alive_gpu_streams_.erase(stream->platform_specific_handle().stream);
}

blas::BlasSupport* MusaExecutor::AsBlas() {
  absl::MutexLock lock(&mu_);
  if (blas_ != nullptr) {
    return blas_.get();
  }

  PluginRegistry* registry = PluginRegistry::Instance();
  absl::StatusOr<PluginRegistry::BlasFactory> status =
      registry->GetFactory<PluginRegistry::BlasFactory>(musa::kMUSaPlatformId);
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve BLAS factory: "
               << status.status().message();
    return nullptr;
  }

  auto blas = status.value()(this);
  blas_.reset(blas);
  return blas_.get();
}

dnn::DnnSupport* MusaExecutor::AsDnn() {
  absl::MutexLock lock(&mu_);
  if (dnn_ != nullptr) {
    return dnn_.get();
  }
  PluginRegistry* registry = PluginRegistry::Instance();
  absl::StatusOr<PluginRegistry::DnnFactory> status =
      registry->GetFactory<PluginRegistry::DnnFactory>(musa::kMUSaPlatformId);
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve DNN factory: "
               << status.status().message();
    return nullptr;
  }

  auto dnn = status.value()(this);

  dnn_.reset(dnn);

  return dnn_.get();
}

fft::FftSupport* MusaExecutor::AsFft() {
  absl::MutexLock lock(&mu_);
  if (fft_ != nullptr) {
    return fft_.get();
  }
  PluginRegistry* registry = PluginRegistry::Instance();
  absl::StatusOr<PluginRegistry::FftFactory> status =
      registry->GetFactory<PluginRegistry::FftFactory>(musa::kMUSaPlatformId);
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve FFT factory: "
               << status.status().message();
    return nullptr;
  }

  auto fft = status.value()(this);

  fft_.reset(fft);
  return fft_.get();
}

bool MusaExecutor::CanEnablePeerAccessTo(StreamExecutor* other) {
  MusaExecutor* musa_other = static_cast<MusaExecutor*>(other);
  return CanEnablePeerAccess(musa_context_, musa_other->musa_context_);
}

absl::Status MusaExecutor::EnablePeerAccessTo(StreamExecutor* other) {
  MusaExecutor* musa_other = static_cast<MusaExecutor*>(other);
  return EnablePeerAccess(musa_context_, musa_other->musa_context_);
}

bool MusaExecutor::DeviceMemoryUsage(int64_t* free_out,
                                     int64_t* total_out) const {
  ScopedActivateContext activation(musa_context_);
  size_t free = 0;
  size_t total = 0;
  auto status = musa::ToStatus(muMemGetInfo(&free, &total));
  if (!status.ok()) {
    LOG(ERROR) << "failed to query device memory info: " << status;
    return false;
  }

  *free_out = free;
  *total_out = total;
  return true;
}

absl::StatusOr<DeviceMemoryBase> MusaExecutor::GetSymbol(
    const std::string& symbol_name, ModuleHandle module_handle) {
  void* mem = nullptr;
  size_t bytes = 0;
  CHECK(static_cast<bool>(module_handle));

  {  // give limited scope to MutexLock
    absl::MutexLock lock{&in_memory_modules_mu_};
    auto it = gpu_binary_to_module_.find(module_handle);
    CHECK(it != gpu_binary_to_module_.end());

    MUmodule gpu_module_handle = it->second.first;
    CHECK(gpu_module_handle != nullptr);
    TF_RETURN_IF_ERROR(
        GetModuleSymbol(musa_context_, gpu_module_handle, symbol_name.c_str(),
                        reinterpret_cast<MUdeviceptr*>(&mem), &bytes));
    return DeviceMemoryBase(mem, bytes);
  }

  return absl::NotFoundError(
      absl::StrCat("Check if module containing symbol ", symbol_name,
                   " is loaded (module_handle = ",
                   reinterpret_cast<uintptr_t>(module_handle.id()), ")"));
}

namespace {
absl::Status FillBlockDimLimit(MUdevice device, BlockDim* block_dim_limit) {
  // The BlockDim name is a mismatch against these GRID_DIM_* queries because
  // we use BlockDims to express the dimensions of blocks within a grid
  // (as opposed to ThreadDim which expresses the dimensions of threads
  // within a block).
  int x, y, z;
  TF_RETURN_IF_ERROR(GetGridLimits(&x, &y, &z, device));
  block_dim_limit->x = x;
  block_dim_limit->y = y;
  block_dim_limit->z = z;
  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<std::unique_ptr<Event>> MusaExecutor::CreateEvent() {
  TF_ASSIGN_OR_RETURN(auto event, MusaEvent::Create(this, false));
  return std::make_unique<MusaEvent>(std::move(event));
}

absl::StatusOr<std::unique_ptr<Stream>> MusaExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  TF_ASSIGN_OR_RETURN(auto stream, MusaStream::Create(this, priority));
  absl::MutexLock l(&alive_gpu_streams_mu_);
  alive_gpu_streams_[stream->stream_handle()] = stream.get();
  return std::move(stream);
}

absl::StatusOr<std::unique_ptr<CommandBuffer>>
MusaExecutor::CreateCommandBuffer(CommandBuffer::Mode mode) {
  VLOG(2) << "[" << device_ordinal()
          << "] Create MUSA command buffer (MUSA graph)";
  return MusaCommandBuffer::Create(mode, this, musa_context_);
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
MusaExecutor::CreateDeviceDescription(int device_ordinal) {
  TF_ASSIGN_OR_RETURN(MUdevice device, GetDevice(device_ordinal));
  TF_ASSIGN_OR_RETURN(MusaComputeCapability cc, GetComputeCapability(device));

  DeviceDescription desc;
  int32_t driver_version{};
  {
    // TODO(b/381052076): Return an error instead of silent failure once TF can
    // accommodate that.
    absl::Status result = musa::ToStatus(muDriverGetVersion(&driver_version),
                                         "Could not get driver version");
    if (!result.ok()) {
      LOG(ERROR) << result;
    }
  }
  desc.set_driver_version(
      ParseMusaVersion(driver_version).value_or(SemanticVersion{0, 0, 0}));

  int32_t runtime_version{};
  {
    // TODO(b/381052076): Return an error instead of silent failure once TF can
    // accommodate that.
    absl::Status result =
        musa::ToStatus(musaRuntimeGetVersion(&runtime_version),
                       "Failed call to musaGetRuntimeVersion");
    if (!result.ok()) {
      LOG(ERROR) << result;
    }
  }
  desc.set_runtime_version(
      ParseMusaVersion(runtime_version).value_or(SemanticVersion{0, 0, 0}));
  desc.set_compile_time_toolkit_version(
      ParseMusaVersion(MUSA_VERSION).value_or(SemanticVersion{0, 0, 0}));

  {
    std::string pci_bus_id = GetPCIBusID(device);
    desc.set_pci_bus_id(pci_bus_id);

    // Read the NUMA node corresponding to the PCI bus ID out of sysfs.
    std::optional<int> numa_node = ReadNumaNode(pci_bus_id, device_ordinal);
    // If the kernel reports -1, adjust to 0; leave as -1 if no value could be
    // obtained.
    desc.set_numa_node(numa_node.has_value() ? std::max(0, *numa_node)
                                             : tsl::port::kNUMANoAffinity);
  }

  {
    desc.set_threads_per_block_limit(
        GetDeviceAttribute(MU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device)
            .value());

    ThreadDim thread_dim_limit;
    thread_dim_limit.x =
        GetDeviceAttribute(MU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device).value();
    thread_dim_limit.y =
        GetDeviceAttribute(MU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device).value();
    thread_dim_limit.z =
        GetDeviceAttribute(MU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device).value();
    desc.set_thread_dim_limit(thread_dim_limit);
  }

  int sm_clock_khz =
      GetDeviceAttribute(MU_DEVICE_ATTRIBUTE_CLOCK_RATE, device).value();
  desc.set_clock_rate_ghz(static_cast<float>(sm_clock_khz) / 1e6);

  {
    bool ecc_enabled = false;
    IsEccEnabled(device, &ecc_enabled);
    desc.set_ecc_enabled(ecc_enabled);
  }

  uint64_t device_memory_size = static_cast<uint64_t>(-1);
  GetDeviceTotalMemory(device, &device_memory_size);
  desc.set_device_memory_size(device_memory_size);

  int64_t l2_cache_bytes =
      GetDeviceAttribute(MU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device).value();
  desc.set_l2_cache_size(l2_cache_bytes);

  absl::StatusOr<int> mem_clock_khz =
      GetDeviceAttribute(MU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device_ordinal);
  absl::StatusOr<int> mem_bus_width_bits = GetDeviceAttribute(
      MU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device_ordinal);
  if (mem_clock_khz.ok() && mem_bus_width_bits.ok()) {
    // Times 2 because HBM is DDR memory; it gets two data bits per each data
    // lane.
    desc.set_memory_bandwidth(2 * int64_t{mem_clock_khz.value()} * 1000 *
                              int64_t{mem_bus_width_bits.value()} / 8);
  }

  {
    BlockDim block_dim_limit;
    TF_RETURN_IF_ERROR(FillBlockDimLimit(device, &block_dim_limit));
    desc.set_block_dim_limit(block_dim_limit);
  }

  {
    TF_ASSIGN_OR_RETURN(std::string device_name, GetDeviceName(device));
    desc.set_name(device_name);
  }

  desc.set_platform_version(absl::StrCat("Compute Capability ", cc.ToString()));

  // TODO(leary) should be a way to query this from the driver, but this is
  // unlikely to change for us any time soon.
  desc.set_device_address_bits(64);

  desc.set_device_vendor("MT Corporation");
  desc.set_musa_compute_capability(cc);
  desc.set_shared_memory_per_core(GetMaxSharedMemoryPerCore(device).value());
  desc.set_shared_memory_per_block(GetMaxSharedMemoryPerBlock(device).value());
  desc.set_shared_memory_per_block_optin(
      GetMaxSharedMemoryPerBlockOptin(device).value());
  int core_count = GetMultiprocessorCount(device).value();
  desc.set_core_count(core_count);
  desc.set_fpus_per_core(fpus_per_core(cc.major, cc.minor));
  desc.set_threads_per_core_limit(
      GetMaxThreadsPerMultiprocessor(device).value());
  desc.set_registers_per_block_limit(GetMaxRegistersPerBlock(device).value());
  desc.set_threads_per_warp(GetThreadsPerWarp(device).value());
  desc.set_registers_per_core_limit(
      GetDeviceAttribute(MU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
                         device)
          .value());

  auto value_or = [](const auto& status_or, auto default_val) {
    if (status_or.ok()) return *status_or;
    return default_val;
  };

  // It would be better to use the PCI device ID or some other truly unique
  // identifier for the GPU model.  But getting this requires using NVML or
  // other hacks, which we don't have access to in OSS TensorFlow.
  //
  // Alternatively you might be tempted to use GetDeviceName as a
  // unique identifier, but this is not stable across GPU VBIOS versions.
  //
  // For now, this identifier is good enough.
  desc.set_model_str(absl::StrFormat(
      "sm_%s with %dB RAM, %d cores, %dKHz clock, %dKHz mem clock, %dB L2$",
      cc.ToString(), device_memory_size, core_count, sm_clock_khz,
      value_or(mem_clock_khz, 0), l2_cache_bytes));

  return std::make_unique<DeviceDescription>(std::move(desc));
}

absl::StatusOr<MemoryType> MusaExecutor::GetPointerMemorySpace(
    const void* ptr) {
  MUdeviceptr pointer = reinterpret_cast<MUdeviceptr>(const_cast<void*>(ptr));
  unsigned int value;
  TF_RETURN_IF_ERROR(musa::ToStatus(muPointerGetAttribute(
      &value, MU_POINTER_ATTRIBUTE_MEMORY_TYPE, pointer)));
  switch (value) {
    case MU_MEMORYTYPE_DEVICE:
      return MemoryType::kDevice;
    case MU_MEMORYTYPE_HOST:
      return MemoryType::kHost;
    default:
      return absl::InternalError(
          absl::StrCat("unknown memory space provided by MUSA API: ", value));
  }
}

absl::StatusOr<const MusaKernel*> MusaExecutor::GetMusaKernel(
    const Kernel* kernel) {
  absl::MutexLock lock{&in_memory_modules_mu_};
  auto it = loaded_kernels_.find(kernel);
  if (it == loaded_kernels_.end()) {
    return absl::NotFoundError("Kernel not loaded in this executor.");
  }
  return static_cast<const MusaKernel*>(*it);
}

absl::StatusOr<TensorMap> MusaExecutor::CreateTensorMap(
    const TmaDescriptor& tma_desc, void* global_address) {
  return absl::UnimplementedError(
      absl::StrFormat("Unsupported CreateTensorMap"));
}

}  // namespace gpu

}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(musa_executor, {});
