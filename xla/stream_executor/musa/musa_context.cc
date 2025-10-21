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

// Synchronize with spinlocks.
const char kScheduleSpinString[] = "spin";
// Synchronize with spinlocks that also call CPU yield instructions.
const char kScheduleYieldString[] = "yield";
// Synchronize with a "synchronization primitive" (e.g. mutex).
const char kScheduleBlockingSyncString[] = "blocking_sync";

int GetFlagsFromEnv() {
  const char* gpu_schedule_string =
      std::getenv("TF_MUSA_PLATFORM_GPU_DEVICE_SCHEDULE");

  if (gpu_schedule_string == nullptr) {
    return 0;
  }

  unsigned device_flags = 0;
  if (strcmp(kScheduleSpinString, gpu_schedule_string) == 0) {
    device_flags = MU_CTX_SCHED_SPIN;
  } else if (strcmp(kScheduleYieldString, gpu_schedule_string) == 0) {
    device_flags = MU_CTX_SCHED_YIELD;
  } else if (strcmp(kScheduleBlockingSyncString, gpu_schedule_string) == 0) {
    device_flags = MU_CTX_SCHED_BLOCKING_SYNC;
  } else {
    LOG(QFATAL) << "Unknown option for environment variable "
                   "TF_MUSA_PLATFORM_GPU_DEVICE_SCHEDULE "
                << gpu_schedule_string << " should be one of {"
                << kScheduleBlockingSyncString << ", " << kScheduleSpinString
                << ", " << kScheduleYieldString << "}";
  }

  return device_flags;
}

// Returns the current context or dies if it fails.
MUcontext CurrentContextOrDie() {
  MUcontext current = nullptr;
  TF_CHECK_OK(musa::ToStatus(muCtxGetCurrent(&current),
                             "Failed to query current context"));
  return current;
}

// Returns the current context and checks that it is in the set of MUSA contexts
// created by StreamExecutor (to ensure that the MUSA runtime didn't create a
// context behind our backs).
MUcontext CurrentContext() {
  MUcontext current = CurrentContextOrDie();
  if (current != nullptr && !MusaContext::GetContextMap()->Has(current)) {
    LOG(FATAL) << "current context was not created by the StreamExecutor "
                  "musa_driver API: "
               << current
               << "; a MUSA runtime call "
                  "was likely performed without using a StreamExecutor context";
  }
  return current;
}

}  // namespace

// Returns the singleton ContextMap.
ContextMap<MUcontext, MusaContext>* MusaContext::GetContextMap() {
  static ContextMap<MUcontext, MusaContext>* context_map =
      new ContextMap<MUcontext, MusaContext>([](void* ptr) {
        int device_ordinal;
        absl::Status status = musa::ToStatus(
            muPointerGetAttribute(static_cast<void*>(&device_ordinal),
                                  MU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                  reinterpret_cast<MUdeviceptr>(ptr)));
        if (!status.ok()) {
          LOG(FATAL) << "Not able to get the device_ordinal for ptr: " << ptr
                     << ". Error: " << status;
        }
        return device_ordinal;
      });
  return context_map;
}

MusaContext::~MusaContext() {
  auto status = musa::ToStatus(muCtxPushCurrent(context()));
  if (!status.ok()) {
    LOG(ERROR) << "failed to Push MUSA context; leaking: " << status;
  }
  MUdevice device;
  muCtxGetDevice(&device);
  muCtxPopCurrent(nullptr);

  status = musa::ToStatus(muDevicePrimaryCtxRelease(device));

  if (!status.ok()) {
    LOG(ERROR) << "failed to release MUSA context; leaking: " << status;
  }

  GetContextMap()->Remove(context());
}

absl::StatusOr<MusaContext*> MusaContext::Create(int device_ordinal,
                                                 MUdevice device) {
  MusaContext* context = nullptr;

  int flags = GetFlagsFromEnv();

  unsigned int former_primary_context_flags;
  int former_primary_context_is_active;
  TF_RETURN_IF_ERROR(musa::ToStatus(
      muDevicePrimaryCtxGetState(device, &former_primary_context_flags,
                                 &former_primary_context_is_active)));
  if (former_primary_context_flags != flags) {
    if (former_primary_context_is_active) {
      LOG(ERROR)
          << "The primary context is active and has a different flag set ("
          << former_primary_context_flags << ") than the desired flag set ("
          << flags << ").";
    } else {
      TF_RETURN_IF_ERROR(
          musa::ToStatus(muDevicePrimaryCtxSetFlags(device, flags)));
    }
  }

  MUcontext former_context = CurrentContextOrDie();
  MUcontext new_context;
  TF_RETURN_IF_ERROR(
      musa::ToStatus(muDevicePrimaryCtxRetain(&new_context, device)));
  if (former_context != nullptr) {
    MUdevice former_device;
    if (muCtxGetDevice(&former_device) == MUSA_SUCCESS) {
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
  TF_RETURN_IF_ERROR(musa::ToStatus(muCtxSetCurrent(former_context)));

  context = GetContextMap()->Add(new_context, device_ordinal);
  CHECK(context != nullptr)
      << "success in this call must entail non-null result";
  VLOG(2) << "created or reused context " << new_context << " for this thread";
  return context;
}

void MusaContext::SetActive() {
  TF_CHECK_OK(
      musa::ToStatus(muCtxSetCurrent(context_), "Failed setting context"));
}

bool MusaContext::IsActive() const { return CurrentContext() == context_; }

absl::Status MusaContext::Synchronize() {
  ScopedActivateContext activation(this);
  return musa::ToStatus(muCtxSynchronize());
}

}  // namespace stream_executor::gpu
