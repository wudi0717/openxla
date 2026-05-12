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

#include "xla/backends/profiler/gpu/musa_tracer.h"

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"
#include "xla/tsl/profiler/utils/time_utils.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"

namespace stream_executor { }
namespace se = stream_executor;

namespace xla {
namespace profiler {

static absl::Status ToStatus(MUptiResult result, const char* msg) {
  if (result == MUPTI_SUCCESS) return absl::OkStatus();
  return absl::InternalError(absl::StrCat(msg, ": ", result));
}

MusaTracer* MusaTracer::GetMusaTracerSingleton() {
  static auto* singleton = new MusaTracer();
  return singleton;
}

bool MusaTracer::IsAvailable() const {
  return !api_tracing_enabled_;
}

int MusaTracer::NumGpus() {
  return 1; 
}

/*static*/ uint64_t MusaTracer::GetTimestamp() {
  uint64_t ts = 0;
  se::wrap::muptiGetTimestamp(&ts);
  return ts;
}

// ============================================================================
// API Callback Implementation
// ============================================================================

absl::Status MusaApiCallbackImpl::operator()(uint32_t domain, uint32_t cbid,
                                             const void* cbdata) {
  if (domain != MUPTI_CB_DOMAIN_RUNTIME_API) {
    return absl::OkStatus();
  }

  const auto* data = static_cast<const MUpti_CallbackData*>(cbdata);

  if (data->callbackSite != MUPTI_API_EXIT) {
    absl::MutexLock lock(&api_call_start_mutex_);
    api_call_start_time_.emplace(data->correlationId, MusaTracer::GetTimestamp());
    return absl::OkStatus();
  }

  // --- API EXIT ---
  
  uint64_t enter_time = 0;
  uint64_t exit_time = MusaTracer::GetTimestamp();

  {
    absl::MutexLock lock(&api_call_start_mutex_);
    auto it = api_call_start_time_.find(data->correlationId);
    if (it != api_call_start_time_.end()) {
      enter_time = it->second;
      api_call_start_time_.erase(it);
    } else {
      enter_time = exit_time; 
    }
  }

  switch (cbid) {
    case MUPTI_RUNTIME_TRACE_CBID_musaLaunchKernel_v7000:
    case MUPTI_RUNTIME_TRACE_CBID_musaLaunchKernel_ptsz_v7000:
    case MUPTI_RUNTIME_TRACE_CBID_musaLaunchKernelExC_v11060:
    case MUPTI_RUNTIME_TRACE_CBID_musaLaunchKernelExC_ptsz_v11060:
      AddKernelEventUponApiExit(cbid, data, enter_time, exit_time);
      break;

    case MUPTI_RUNTIME_TRACE_CBID_musaMemcpy_v3020:
    case MUPTI_RUNTIME_TRACE_CBID_musaMemcpy_ptds_v7000:
    case MUPTI_RUNTIME_TRACE_CBID_musaMemcpyAsync_v3020:
    case MUPTI_RUNTIME_TRACE_CBID_musaMemcpyAsync_ptsz_v7000:
    case MUPTI_RUNTIME_TRACE_CBID_musaMemcpyPeer_v4000:
    case MUPTI_RUNTIME_TRACE_CBID_musaMemcpyPeerAsync_v4000:
      AddNormalMemcpyEventUponApiExit(cbid, data, enter_time, exit_time);
      break;

    case MUPTI_RUNTIME_TRACE_CBID_musaMemset_v3020:
    case MUPTI_RUNTIME_TRACE_CBID_musaMemset_ptds_v7000:
    case MUPTI_RUNTIME_TRACE_CBID_musaMemsetAsync_v3020:
    case MUPTI_RUNTIME_TRACE_CBID_musaMemsetAsync_ptsz_v7000:
      AddMemsetEventUponApiExit(cbid, data, enter_time, exit_time);
      break;

    case MUPTI_RUNTIME_TRACE_CBID_musaMalloc_v3020:
    case MUPTI_RUNTIME_TRACE_CBID_musaMallocAsync_v11020:
    case MUPTI_RUNTIME_TRACE_CBID_musaMallocAsync_ptsz_v11020:
    case MUPTI_RUNTIME_TRACE_CBID_musaFree_v3020:
    case MUPTI_RUNTIME_TRACE_CBID_musaFreeAsync_v11020:
    case MUPTI_RUNTIME_TRACE_CBID_musaFreeAsync_ptsz_v11020:
      AddMallocFreeEventUponApiExit(cbid, data, 0, enter_time, exit_time);
      break;

    case MUPTI_RUNTIME_TRACE_CBID_musaStreamSynchronize_v3020:
    case MUPTI_RUNTIME_TRACE_CBID_musaStreamSynchronize_ptsz_v7000:
      AddStreamSynchronizeEventUponApiExit(cbid, data, enter_time, exit_time);
      break;

    default:
      break;
  }

  return absl::OkStatus();
}

void MusaApiCallbackImpl::AddKernelEventUponApiExit(
    uint32_t cbid, const MUpti_CallbackData* data, uint64_t enter_time,
    uint64_t exit_time) {
  
  MusaTracerEvent event;
  event.type = MusaTracerEventType::Kernel;
  event.source = MusaTracerEventSource::ApiCallback;
  event.domain = MusaTracerEventDomain::MUSA_API;
  event.start_time_ns = enter_time;
  event.end_time_ns = exit_time;
  event.thread_id = tsl::Env::Default()->GetCurrentThreadId();
  event.correlation_id = data->correlationId;
  
  if (data->symbolName) {
    event.name = data->symbolName;
  } else {
    event.name = "musaLaunchKernel";
  }

  event.kernel_info.grid_x = 0;
  event.kernel_info.block_x = 0;
  event.kernel_info.dynamic_shared_memory_usage = 0;

  collector_->AddEvent(std::move(event));
}

void MusaApiCallbackImpl::AddNormalMemcpyEventUponApiExit(
    uint32_t cbid, const MUpti_CallbackData* data, uint64_t enter_time,
    uint64_t exit_time) {
  
  MusaTracerEvent event;
  event.type = MusaTracerEventType::MemcpyOther;
  event.source = MusaTracerEventSource::ApiCallback;
  event.domain = MusaTracerEventDomain::MUSA_API;
  event.start_time_ns = enter_time;
  event.end_time_ns = exit_time;
  event.thread_id = tsl::Env::Default()->GetCurrentThreadId();
  event.correlation_id = data->correlationId;
  event.name = "musaMemcpy";

  event.memcpy_info.num_bytes = 0;
  event.memcpy_info.async = true;

  collector_->AddEvent(std::move(event));
}

void MusaApiCallbackImpl::AddMemcpyPeerEventUponApiExit(
    uint32_t cbid, const MUpti_CallbackData* data, uint64_t enter_time,
    uint64_t exit_time) {
  
  MusaTracerEvent event;
  event.type = MusaTracerEventType::MemcpyP2P;
  event.source = MusaTracerEventSource::ApiCallback;
  event.domain = MusaTracerEventDomain::MUSA_API;
  event.start_time_ns = enter_time;
  event.end_time_ns = exit_time;
  event.thread_id = tsl::Env::Default()->GetCurrentThreadId();
  event.correlation_id = data->correlationId;
  event.name = "musaMemcpyPeer";

  collector_->AddEvent(std::move(event));
}

void MusaApiCallbackImpl::AddMemsetEventUponApiExit(
    uint32_t cbid, const MUpti_CallbackData* data, uint64_t enter_time,
    uint64_t exit_time) {
  
  MusaTracerEvent event;
  event.type = MusaTracerEventType::Memset;
  event.source = MusaTracerEventSource::ApiCallback;
  event.domain = MusaTracerEventDomain::MUSA_API;
  event.start_time_ns = enter_time;
  event.end_time_ns = exit_time;
  event.thread_id = tsl::Env::Default()->GetCurrentThreadId();
  event.correlation_id = data->correlationId;
  event.name = "musaMemset";
  
  event.memset_info.num_bytes = 0;

  collector_->AddEvent(std::move(event));
}

void MusaApiCallbackImpl::AddMallocFreeEventUponApiExit(
    uint32_t cbid, const MUpti_CallbackData* data, uint32_t device_id,
    uint64_t enter_time, uint64_t exit_time) {
  
  MusaTracerEvent event;
  if (cbid == MUPTI_RUNTIME_TRACE_CBID_musaFree_v3020 || 
      cbid == MUPTI_RUNTIME_TRACE_CBID_musaFreeAsync_v11020 ||
      cbid == MUPTI_RUNTIME_TRACE_CBID_musaFreeAsync_ptsz_v11020) {
      event.type = MusaTracerEventType::MemoryFree;
      event.name = "musaFree";
  } else {
      event.type = MusaTracerEventType::MemoryAlloc;
      event.name = "musaMalloc";
  }

  event.source = MusaTracerEventSource::ApiCallback;
  event.domain = MusaTracerEventDomain::MUSA_API;
  event.start_time_ns = enter_time;
  event.end_time_ns = exit_time;
  event.thread_id = tsl::Env::Default()->GetCurrentThreadId();
  event.correlation_id = data->correlationId;
  event.device_id = device_id;

  event.memalloc_info.num_bytes = 0;

  collector_->AddEvent(std::move(event));
}

void MusaApiCallbackImpl::AddStreamSynchronizeEventUponApiExit(
    uint32_t cbid, const MUpti_CallbackData* data, uint64_t enter_time,
    uint64_t exit_time) {
  
  MusaTracerEvent event;
  event.type = MusaTracerEventType::Synchronization;
  event.source = MusaTracerEventSource::ApiCallback;
  event.domain = MusaTracerEventDomain::MUSA_API;
  event.start_time_ns = enter_time;
  event.end_time_ns = exit_time;
  event.thread_id = tsl::Env::Default()->GetCurrentThreadId();
  event.correlation_id = data->correlationId;
  event.name = "musaStreamSynchronize";

  event.synchronization_info.sync_type = MusaTracerSyncTypes::StreamSynchronize;

  collector_->AddEvent(std::move(event));
}

void MusaApiCallbackImpl::AddSynchronizeEventUponApiExit(
    uint32_t cbid, const MUpti_CallbackData* data, uint64_t enter_time,
    uint64_t exit_time) {
}

// ============================================================================
// Tracer Control (Enable/Disable)
// ============================================================================

// [重要修改] 将参数类型修改为 MUSA 枚举类型，匹配 muptiSubscribe 的要求
static void MUPTIAPI ApiCallbackThunk(void *userdata, 
                                      MUpti_CallbackDomain domain,
                                      MUpti_CallbackId cbid, 
                                      const void *cbdata) {
  MusaTracer* tracer = reinterpret_cast<MusaTracer*>(userdata);
  // 这里将枚举强转回 uint32_t 传递给内部逻辑处理，这是安全的
  tracer->ApiCallbackHandler((uint32_t)domain, (uint32_t)cbid, cbdata);
}

absl::Status MusaTracer::ApiCallbackHandler(uint32_t domain, uint32_t cbid,
                                            const void* cbdata) {
  if (api_tracing_enabled_ && api_cb_impl_) {
    return (*api_cb_impl_)(domain, cbid, cbdata);
  }
  return absl::OkStatus();
}

void MusaTracer::Enable(const MusaTracerOptions& options,
                        MusaTraceCollector* collector) {
  options_ = options;
  collector_ = collector;

  if (!options_->api_callbacks.empty()) {
    EnableApiTracing();
  }
}

void MusaTracer::Disable() {
  if (api_tracing_enabled_) {
    DisableApiTracing();
  }
  options_.reset();
  collector_ = nullptr;
}

absl::Status MusaTracer::EnableApiTracing() {
  if (api_tracing_enabled_) return absl::OkStatus();

  api_cb_impl_ = new MusaApiCallbackImpl(*options_, this, collector_);

  // 1. Subscribe
  MUptiResult res = se::wrap::muptiSubscribe(
      &subscriber_handle_,
      ApiCallbackThunk,        // 现在签名匹配了
      (void*)this);
  
  if (res != MUPTI_SUCCESS) return ToStatus(res, "muptiSubscribe failed");

  // 2. Enable Domain (Runtime API)
  res = se::wrap::muptiEnableDomain(
      1,
      subscriber_handle_,
      // [重要修改] 显式转换枚举类型
      static_cast<MUpti_CallbackDomain>(MUPTI_CB_DOMAIN_RUNTIME_API));
  
  if (res != MUPTI_SUCCESS) return ToStatus(res, "muptiEnableDomain failed");

  // 3. Enable Callbacks
  for (const auto& pair : options_->api_callbacks) {
    uint32_t domain = pair.first;
    const std::vector<uint32_t>& ops = pair.second;
    for (uint32_t cbid : ops) {
      res = se::wrap::muptiEnableCallback(
          1,
          subscriber_handle_,
          // [重要修改] 显式转换枚举类型
          static_cast<MUpti_CallbackDomain>(domain), 
          static_cast<MUpti_CallbackId>(cbid));
      
      if (res != MUPTI_SUCCESS) {
         LOG(WARNING) << "Failed to enable callback for cbid: " << cbid;
      }
    }
  }

  api_tracing_enabled_ = true;
  return absl::OkStatus();
}

absl::Status MusaTracer::DisableApiTracing() {
  if (!api_tracing_enabled_) return absl::OkStatus();

  // 1. Disable Domain
  se::wrap::muptiEnableDomain(
      0,
      subscriber_handle_,
      static_cast<MUpti_CallbackDomain>(MUPTI_CB_DOMAIN_RUNTIME_API));

  // 2. Unsubscribe
  if (subscriber_handle_) {
    se::wrap::muptiUnsubscribe(subscriber_handle_);
    subscriber_handle_ = nullptr;
  }

  delete api_cb_impl_;
  api_cb_impl_ = nullptr;
  api_tracing_enabled_ = false;
  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace xla