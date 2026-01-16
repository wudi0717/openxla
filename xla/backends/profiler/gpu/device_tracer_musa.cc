/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
// #include "musa/include/musa/amd_detail/musa_prof_str.h"
// #include "musa/include/mutracer/ext/prof_protocol.h"
#include "mupti.h"
#include "mupti_callbacks.h"      // 定义了 MUPTI_CB_DOMAIN_RUNTIME_API
#include "mupti_runtime_cbid.h"   // 定义了具体的函数 ID
#include "xla/backends/profiler/gpu/musa_collector.h"
#include "xla/backends/profiler/gpu/musa_tracer.h"
#include "xla/tsl/platform/env_time.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"

namespace xla {
namespace profiler {

using tensorflow::ProfileOptions;
using tsl::profiler::AnnotationStack;
using tsl::profiler::ProfilerInterface;
using tsl::profiler::RegisterProfilerFactory;
using tsl::profiler::XSpace;

// GpuTracer for MUSa GPU.
class GpuTracer : public profiler::ProfilerInterface {
 public:
  explicit GpuTracer(MusaTracer* musa_tracer) : musa_tracer_(musa_tracer) {
    LOG(INFO) << "GpuTracer created.";
  }
  ~GpuTracer() override {}

  // GpuTracer interface:
  absl::Status Start() override;
  absl::Status Stop() override;
  absl::Status CollectData(XSpace* space) override;

 private:
  absl::Status DoStart();
  absl::Status DoStop();

  MusaTracerOptions GetMusaTracerOptions();

  MusaTraceCollectorOptions GetMusaTraceCollectorOptions(uint32_t num_gpus);

  enum State {
    kNotStarted,
    kStartedOk,
    kStartedError,
    kStoppedOk,
    kStoppedError
  };
  State profiling_state_ = State::kNotStarted;

  MusaTracer* musa_tracer_;
  std::unique_ptr<MusaTraceCollector> musa_trace_collector_;
};

MusaTracerOptions GpuTracer::GetMusaTracerOptions() {
  MusaTracerOptions options;
  std::vector<uint32_t> empty_vec;

  // 定义需要追踪的 Runtime API 列表
  std::vector<uint32_t> musa_api_domain_ops{
      // --- KERNEL LAUNCH ---
      MUPTI_RUNTIME_TRACE_CBID_musaLaunchKernel_v7000,
      MUPTI_RUNTIME_TRACE_CBID_musaLaunchKernel_ptsz_v7000,
      MUPTI_RUNTIME_TRACE_CBID_musaLaunchKernelExC_v11060,
      MUPTI_RUNTIME_TRACE_CBID_musaLaunchKernelExC_ptsz_v11060,

      // --- MEMCPY ---
      MUPTI_RUNTIME_TRACE_CBID_musaMemcpy_v3020,
      MUPTI_RUNTIME_TRACE_CBID_musaMemcpy_ptds_v7000,
      MUPTI_RUNTIME_TRACE_CBID_musaMemcpyAsync_v3020,
      MUPTI_RUNTIME_TRACE_CBID_musaMemcpyAsync_ptsz_v7000,
      MUPTI_RUNTIME_TRACE_CBID_musaMemcpyPeer_v4000,
      MUPTI_RUNTIME_TRACE_CBID_musaMemcpyPeerAsync_v4000,

      // --- MEMSET ---
      MUPTI_RUNTIME_TRACE_CBID_musaMemset_v3020,
      MUPTI_RUNTIME_TRACE_CBID_musaMemset_ptds_v7000,
      MUPTI_RUNTIME_TRACE_CBID_musaMemsetAsync_v3020,
      MUPTI_RUNTIME_TRACE_CBID_musaMemsetAsync_ptsz_v7000,

      // --- MALLOC/FREE ---
      MUPTI_RUNTIME_TRACE_CBID_musaMalloc_v3020,
      MUPTI_RUNTIME_TRACE_CBID_musaMallocAsync_v11020,
      MUPTI_RUNTIME_TRACE_CBID_musaMallocAsync_ptsz_v11020,
      MUPTI_RUNTIME_TRACE_CBID_musaFree_v3020,
      MUPTI_RUNTIME_TRACE_CBID_musaFreeAsync_v11020,
      MUPTI_RUNTIME_TRACE_CBID_musaFreeAsync_ptsz_v11020,
      
      // --- STREAM SYNC ---
      MUPTI_RUNTIME_TRACE_CBID_musaStreamSynchronize_v3020,
      MUPTI_RUNTIME_TRACE_CBID_musaStreamSynchronize_ptsz_v7000,
  };

  // 将列表转换为 Set 供快速查找
  options.api_tracking_set =
      std::set<uint32_t>(musa_api_domain_ops.begin(), musa_api_domain_ops.end());

  // 注册 API Callback
  // 使用 grep 查到的正确 Domain ID: MUPTI_CB_DOMAIN_RUNTIME_API
  options.api_callbacks.emplace(MUPTI_CB_DOMAIN_RUNTIME_API, musa_api_domain_ops);

  // 注册 Activity Tracing (暂时留空或者根据需求开启)
  // 如果你需要追踪 Activity (类似 Nsight Systems)，需要找到 MUPTI_ACTIVITY_KIND_...
  // 目前先保持为空即可编译通过
  // options.activity_tracing.emplace(MUPTI_CB_DOMAIN_RUNTIME_API, empty_vec);

  return options;
}

MusaTraceCollectorOptions GpuTracer::GetMusaTraceCollectorOptions(
    uint32_t num_gpus) {
  MusaTraceCollectorOptions options;
  options.max_callback_api_events = 2 * 1024 * 1024;
  options.max_activity_api_events = 2 * 1024 * 1024;
  options.max_annotation_strings = 1024 * 1024;
  options.num_gpus = num_gpus;
  return options;
}

absl::Status GpuTracer::DoStart() {
  if (!musa_tracer_->IsAvailable()) {
    return tsl::errors::Unavailable("Another profile session running.");
  }

  AnnotationStack::Enable(true);

  MusaTraceCollectorOptions trace_collector_options =
      GetMusaTraceCollectorOptions(musa_tracer_->NumGpus());
  uint64_t start_gputime_ns = MusaTracer::GetTimestamp();
  uint64_t start_walltime_ns = tsl::EnvTime::NowNanos();
  musa_trace_collector_ = CreateMusaCollector(
      trace_collector_options, start_walltime_ns, start_gputime_ns);

  MusaTracerOptions tracer_options = GetMusaTracerOptions();
  musa_tracer_->Enable(tracer_options, musa_trace_collector_.get());

  return absl::OkStatus();
}

absl::Status GpuTracer::Start() {
  absl::Status status = DoStart();
  if (status.ok()) {
    profiling_state_ = State::kStartedOk;
    return absl::OkStatus();
  }
  profiling_state_ = State::kStartedError;
  return status;
}

absl::Status GpuTracer::DoStop() {
  musa_tracer_->Disable();
  AnnotationStack::Enable(false);
  return absl::OkStatus();
}

absl::Status GpuTracer::Stop() {
  if (profiling_state_ == State::kStartedOk) {
    absl::Status status = DoStop();
    profiling_state_ = status.ok() ? State::kStoppedOk : State::kStoppedError;
  }
  return absl::OkStatus();
}

absl::Status GpuTracer::CollectData(XSpace* space) {
  switch (profiling_state_) {
    case State::kNotStarted:
      VLOG(3) << "No trace data collected, session wasn't started";
      return absl::OkStatus();
    case State::kStartedOk:
      return tsl::errors::FailedPrecondition(
          "Cannot collect trace before stopping");
    case State::kStartedError:
      LOG(ERROR) << "Cannot collect, mutracer failed to start";
      return absl::OkStatus();
    case State::kStoppedError:
      VLOG(3) << "No trace data collected";
      return absl::OkStatus();
    case State::kStoppedOk: {
      if (musa_trace_collector_) {
        musa_trace_collector_->Export(space);
      }
      return absl::OkStatus();
    }
  }
  return absl::InternalError(
      absl::StrCat("Invalid profiling state: ", profiling_state_));
}

// Not in anonymous namespace for testing purposes.
std::unique_ptr<profiler::ProfilerInterface> CreateGpuTracer(
    const ProfileOptions& options) {
  if (options.device_type() != ProfileOptions::GPU &&
      options.device_type() != ProfileOptions::UNSPECIFIED) {
    return nullptr;
  }

  profiler::MusaTracer* musa_tracer =
      profiler::MusaTracer::GetMusaTracerSingleton();
  if (!musa_tracer->IsAvailable()) {
    return nullptr;
  }

  return std::make_unique<profiler::GpuTracer>(musa_tracer);
}

auto register_musa_gpu_tracer_factory = [] {
  RegisterProfilerFactory(&CreateGpuTracer);
  return 0;
}();

}  // namespace profiler
}  // namespace xla
