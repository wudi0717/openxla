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

#include "xla/stream_executor/musa/musa_platform.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "musa_runtime.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/musa/musa_diagnostics.h"
#include "xla/stream_executor/musa/musa_driver_wrapper.h"
#include "xla/stream_executor/musa/musa_executor.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/musa/musa_status.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"

namespace stream_executor {
namespace gpu {
namespace {

// Actually performs the work of MUSA initialization. Wrapped up in one-time
// execution guard.
static absl::Status InternalInit() {
  absl::Status status =
      musa::ToStatus(muInit(0 /* = flags */), "Failed call to muInit");
  if (status.ok()) {
    return status;
  }

  LOG(ERROR) << "failed call to muInit: " << status;

  musa::Diagnostician::LogDiagnosticInformation();
  return status;
}

static absl::Status PlatformInitialize() {
  // Cached return value from calling InternalInit(), as muInit need only be
  // called once, but PlatformInitialize may be called many times.
  static absl::Status* initialization_status = [] {
    return new absl::Status(InternalInit());
  }();
  return *initialization_status;
}

}  // namespace

MusaPlatform::MusaPlatform() : name_("MUSA") {}

Platform::Id MusaPlatform::id() const { return musa::kMUSaPlatformId; }

int MusaPlatform::VisibleDeviceCount() const {
  // Initialized in a thread-safe manner the first time this is run.
  static const int num_devices = [] {
    if (!PlatformInitialize().ok()) {
      return -1;
    }
    int device_count = 0;
    auto status = musa::ToStatus(muDeviceGetCount(&device_count));
    if (!status.ok()) {
      LOG(ERROR) << "could not retrieve MUSA device count: " << status;
      return 0;
    }

    return device_count;
  }();
  return num_devices;
}

const std::string& MusaPlatform::Name() const { return name_; }

absl::StatusOr<std::unique_ptr<DeviceDescription>>
MusaPlatform::DescriptionForDevice(int ordinal) const {
  TF_RETURN_IF_ERROR(PlatformInitialize());
  return MusaExecutor::CreateDeviceDescription(ordinal);
}

absl::StatusOr<StreamExecutor*> MusaPlatform::ExecutorForDevice(int ordinal) {
  TF_RETURN_IF_ERROR(PlatformInitialize());
  return executor_cache_.GetOrCreate(
      ordinal, [this, ordinal]() { return GetUncachedExecutor(ordinal); });
}

absl::StatusOr<StreamExecutor*> MusaPlatform::FindExisting(int ordinal) {
  return executor_cache_.Get(ordinal);
}

absl::StatusOr<std::unique_ptr<StreamExecutor>>
MusaPlatform::GetUncachedExecutor(int ordinal) {
  auto executor = std::make_unique<MusaExecutor>(this, ordinal);
  TF_RETURN_IF_ERROR(executor->Init());
  return std::move(executor);
}

}  // namespace gpu

static void InitializeMusaPlatform() {
  TF_CHECK_OK(
      PlatformManager::RegisterPlatform(std::make_unique<gpu::MusaPlatform>()));
}

}  // namespace stream_executor
STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    musa_platform, stream_executor::InitializeMusaPlatform());
