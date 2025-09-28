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

#include "xla/stream_executor/musa/musa_context.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "musa_runtime.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/context_map.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "xla/stream_executor/musa/musa_driver_wrapper.h"
#include "xla/stream_executor/musa/musa_status.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"

namespace stream_executor::gpu {

namespace {

// Returns the current context or dies if it fails.
MUcontext CurrentContextOrDie() {
  MUcontext current = nullptr;
  TF_CHECK_OK(
      ToStatus(muCtxGetCurrent(&current), "Failed to query current context"));
  return current;
}

// Returns the current context and checks that it is in the set of HIP contexts
// created by StreamExecutor (to ensure that the HIP runtime didn't create a
// context behind our backs).
MUcontext CurrentContext() {
  MUcontext current = CurrentContextOrDie();
  if (current != nullptr && !MusaContext::GetContextMap()->Has(current)) {
    LOG(FATAL) << "current context was not created by the StreamExecutor "
                  "musa_driver API: "
               << current
               << "; a HIP runtime call "
                  "was likely performed without using a StreamExecutor context";
  }
  return current;
}

// Returns the amount of memory reserved by ROCm libraries.
bool GetReservedMemory(uint64_t* reserve) {
  musaDeviceProp props;
  MUdevice dev;
  musaError_t res = musaGetDevice(&dev);

  if (res != musaSuccess) {
    LOG(FATAL) << "failed to query current device: " << ToString(res);
    return false;
  }
  res = musaGetDeviceProperties(&props, dev);
  if (res != musaSuccess) {
    LOG(ERROR) << "failed to query device properties: " << ToString(res);
    return false;
  }

  std::string gcnArchName = props.name;
  // On gfx90a, we hide 1 GB of GPU memory (512MB for gfx908) from TF,
  // to allow for late allocations by internal ROCm libraries
  // (e.g. rocBLAS alone needs~200 MB to put its kernels as of ROCm 4.1)
  const uint64_t RESERVED_GFX908 = 1048576 * 512;
  *reserve = RESERVED_GFX908;

  return true;
}

}  // namespace

// Returns the singleton ContextMap.
ContextMap<MUcontext, MusaContext>* MusaContext::GetContextMap() {
  static ContextMap<MUcontext, MusaContext>* context_map =
      new ContextMap<MUcontext, MusaContext>([](void* ptr) {
        int device_ordinal;
        MUresult result =
            muPointerGetAttribute(static_cast<void*>(&device_ordinal),
                                   MU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                   reinterpret_cast<MUdeviceptr>(ptr));
        if (result != MUSA_SUCCESS) {
          LOG(FATAL) << "Not able to get the device_ordinal for ptr: " << ptr
                     << ". Error: " << ToString(result);
        }
        return device_ordinal;
      });
  return context_map;
}

bool MusaContext::GetDeviceTotalMemory(MUdevice device, uint64_t* result) {
  size_t value = -1;
  MUresult res = muDeviceTotalMem(&value, device);
  if (res != MUSA_SUCCESS) {
    LOG(ERROR) << "failed to query total available memory: " << ToString(res);
    return false;
  }
  uint64_t reserve = 0;
  if (!GetReservedMemory(&reserve)) {
    LOG(ERROR) << "failed to reserved device memory for MUSA libraries";
    return false;
  }
  *result = value - reserve;
  return true;
}

bool MusaContext::GetDeviceMemoryUsage(int64_t* free_out, int64_t* total_out) {
  ScopedActivateContext activation(this);
  size_t free = 0;
  size_t total = 0;
  musaError_t res = musaMemGetInfo(&free, &total);
  if (res != musaSuccess) {
    LOG(ERROR) << "failed to query device memory info: " << ToString(res);
    return false;
  }

  uint64_t reserve = 0;
  if (!GetReservedMemory(&reserve)) {
    LOG(ERROR) << "failed to reserved device memory for ROCm libraries";
    return false;
  }

  VLOG(1) << "Device memory: " << total / 1048576 << " MB total, "
          << free / 1048576 << " MB free, reserving " << reserve / 1048576
          << " MB";

  // overflow check
  if (free > std::numeric_limits<int64_t>::max()) {
    LOG(ERROR) << "free memory (" << free << ") is overflow int64_t";
    return false;
  }

  *free_out = free >= reserve ? free - reserve : 0;
  *total_out = total - reserve;
  return true;
}

MusaContext::~MusaContext() {
  MUcontext former_context = CurrentContext();
  // Explicitly call MusaContext::SetActive() to silence clang-tidy warnings
  // about calling a virtual method in the destructor.
  MusaContext::SetActive();
  MUdevice device;
  CHECK_EQ(MUSA_SUCCESS, muCtxGetDevice(&device));
  CHECK_EQ(MUSA_SUCCESS, muCtxSetCurrent(former_context));

  auto res = muDevicePrimaryCtxRelease(device);

  if (res != MUSA_SUCCESS) {
    LOG(ERROR) << "failed to release HIP context; leaking: " << ToString(res);
  }

  GetContextMap()->Remove(context());
}

void MusaContext::SetActive() {
  TF_CHECK_OK(
      ToStatus(muCtxSetCurrent(context_), "Failed setting context"));
}

bool MusaContext::IsActive() const { return CurrentContext() == context_; }

absl::Status MusaContext::Synchronize() {
  ScopedActivateContext activation(this);
  TF_RETURN_IF_ERROR(ToStatus(musaDeviceSynchronize(),
                              "could not synchronize on MUSA device"));
  return absl::OkStatus();
}

absl::StatusOr<MusaContext*> MusaContext::Create(int device_ordinal,
                                                 MUdevice device) {
  MusaContext* context = nullptr;

  int flags = 0;

  MUresult res;
  MUcontext former_context;
  MUcontext new_context;

  unsigned int former_primary_context_flags;
  int former_primary_context_is_active;
  CHECK_EQ(musaSuccess, muDevicePrimaryCtxGetState(
                           device, &former_primary_context_flags,
                           &former_primary_context_is_active));
  if (former_primary_context_flags != flags) {
    if (former_primary_context_is_active) {
      LOG(ERROR)
          << "The primary context is active and has a different flag set ("
          << former_primary_context_flags << ") than the desired flag set ("
          << flags << ").";
    } else {
      CHECK_EQ(musaSuccess, muDevicePrimaryCtxSetFlags(device, flags));
    }
  }

  former_context = CurrentContextOrDie();
  res = muDevicePrimaryCtxRetain(&new_context, device);
  if (former_context != nullptr) {
    MUdevice former_device;
    if (muCtxGetDevice(&former_device) == musaSuccess) {
      if (former_device == device) {
        if (former_context == new_context) {
          VLOG(2) << "The primary context " << former_context << " for device "
                  << device
                  << " exists before initializing the StreamExecutor.";
        } else {
          LOG(WARNING) << "A non-primary context " << former_context
                       << " for device " << device
                       << " exists before initializing the StreamExecutor. The "
                       << "primary context is now " << new_context << ". We "
                       << "haven't verified StreamExecutor works with that.";
        }
      }
    } else {
      LOG(ERROR) << "Failed to get the device of the current context "
                 << former_context;
    }
  }
  CHECK_EQ(MUSA_SUCCESS, muCtxSetCurrent(former_context));

  if (res == MUSA_SUCCESS) {
    context = GetContextMap()->Add(new_context, device_ordinal);
    CHECK(context != nullptr)
        << "success in this call must entail non-null result";
    VLOG(2) << "created or reused context " << new_context
            << " for this thread";
    return context;
  }

  std::string message =
      "failed call to hipDevicePrimaryCtxRetain: " + ToString(res);
  if (res == MUSA_ERROR_OUT_OF_MEMORY) {
    uint64_t total_memory;
    if (GetDeviceTotalMemory(device, &total_memory)) {
      absl::StrAppend(&message, "; total memory reported: ", total_memory);
    } else {
      absl::StrAppend(&message, "; could not query total memory");
    }
  }

  return absl::InternalError(message);
}

}  // namespace stream_executor::gpu
