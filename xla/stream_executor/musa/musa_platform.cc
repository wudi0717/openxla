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
static absl::Status InternalInitialize() {
  musaError_t res = wrap::muInit(0 /* = flags */);

  if (res == musaSuccess) {
    return absl::OkStatus();
  }

  LOG(ERROR) << "failed call to muInit: " << ToString(res);
  musa::Diagnostician::LogDiagnosticInformation();
  return absl::AbortedError(
      absl::StrCat("failed call to muInit: ", ToString(res)));
}

static absl::Status PlatformInitialize() {
  // Cached return value from calling InternalInitialize(), as hipInit need only
  // be called once, but PlatformInitialize may be called many times.
  static absl::Status* init_retval = [] {
    return new absl::Status(InternalInitialize());
  }();
  return *init_retval;
}
}  // namespace

MUSaPlatform::MUSaPlatform() : name_("MUSA") {}

Platform::Id MUSaPlatform::id() const { return musa::kMUSaPlatformId; }

int MUSaPlatform::VisibleDeviceCount() const {
  // Throw away the result - it logs internally, and this [containing] function
  // isn't in the path of user control. It's safe to call this > 1x.

  if (!PlatformInitialize().ok()) {
    return -1;
  }

  int device_count = 0;
  musaError_t res = wrap::musaGetDeviceCount(&device_count);
  if (res != musaSuccess) {
    LOG(ERROR) << "could not retrieve MUSA device count: " << ToString(res);
    return 0;
  }

  return device_count;
}

const std::string& MUSaPlatform::Name() const { return name_; }

absl::StatusOr<std::unique_ptr<DeviceDescription>>
MUSaPlatform::DescriptionForDevice(int ordinal) const {
  TF_RETURN_IF_ERROR(PlatformInitialize());
  return MusaExecutor::CreateDeviceDescription(ordinal);
}

absl::StatusOr<StreamExecutor*> MUSaPlatform::ExecutorForDevice(int ordinal) {
  TF_RETURN_IF_ERROR(PlatformInitialize());
  return executor_cache_.GetOrCreate(
      ordinal, [this, ordinal]() { return GetUncachedExecutor(ordinal); });
}

absl::StatusOr<StreamExecutor*> MUSaPlatform::FindExisting(int ordinal) {
  return executor_cache_.Get(ordinal);
}

absl::StatusOr<std::unique_ptr<StreamExecutor>>
MUSaPlatform::GetUncachedExecutor(int ordinal) {
  auto executor = std::make_unique<MusaExecutor>(this, ordinal);
  TF_RETURN_IF_ERROR(executor->Init());
  return std::move(executor);
}

}  // namespace gpu

static void InitializeMUSaPlatform() {
  auto status = PlatformManager::PlatformWithName("MUSA");
  if (!status.ok()) {
    TF_CHECK_OK(PlatformManager::RegisterPlatform(
        std::make_unique<gpu::MUSaPlatform>()));
  }
}

}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    musa_platform, stream_executor::InitializeMUSaPlatform());
