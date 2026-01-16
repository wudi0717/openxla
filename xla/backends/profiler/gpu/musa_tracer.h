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

#ifndef XLA_BACKENDS_PROFILER_GPU_MUSA_TRACER_H_
#define XLA_BACKENDS_PROFILER_GPU_MUSA_TRACER_H_

#include <optional>
#include <vector>
#include <set>
#include <map>

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/profiler/gpu/musa_collector.h"
#include "xla/stream_executor/musa/mupti_wrapper.h"
// [MUSA_MOD] 新增：为了使用 MUpti_CallbackData 结构体
#include "mupti_callbacks.h" 

#include "tsl/platform/errors.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/status.h"
#include "tsl/platform/thread_annotations.h"
#include "tsl/platform/types.h"

namespace xla {
namespace profiler {

// [MUSA_MOD] 定义 Domain 类型，MUSA/CUDA 通常使用 uint32_t
using activity_domain_t = uint32_t;

enum class MusaTracerSyncTypes {
  InvalidSync = 0,
  StreamSynchronize,  // caller thread wait stream to become empty
  EventSynchronize,   // caller thread will block until event happens
  StreamWait          // compute stream will wait for event to happen
};

struct MusaTracerOptions {
  std::set<uint32_t> api_tracking_set;  // actual api set we want to profile

  // map of domain --> ops for which we need to enable the API callbacks
  // If the ops vector is empty, then enable API callbacks for entire domain
  absl::flat_hash_map<activity_domain_t, std::vector<uint32_t> > api_callbacks;

  // map of domain --> ops for which we need to enable the Activity records
  // If the ops vector is empty, then enable Activity records for entire domain
  absl::flat_hash_map<activity_domain_t, std::vector<uint32_t> >
      activity_tracing;
};

class MusaTracer;

class MusaApiCallbackImpl {
 public:
  MusaApiCallbackImpl(const MusaTracerOptions& options, MusaTracer* tracer,
                      MusaTraceCollector* collector)
      : options_(options), tracer_(tracer), collector_(collector) {}

  absl::Status operator()(uint32_t domain, uint32_t cbid, const void* cbdata);

 private:
  // [MUSA_MOD] 将 hip_api_data_t 替换为 MUpti_CallbackData
  // MUSA 的 CallbackData 结构体定义在 mupti_callbacks.h 中
  void AddKernelEventUponApiExit(uint32_t cbid, const MUpti_CallbackData* data,
                                 uint64_t enter_time, uint64_t exit_time);
  void AddNormalMemcpyEventUponApiExit(uint32_t cbid,
                                       const MUpti_CallbackData* data,
                                       uint64_t enter_time, uint64_t exit_time);
  void AddMemcpyPeerEventUponApiExit(uint32_t cbid, const MUpti_CallbackData* data,
                                     uint64_t enter_time, uint64_t exit_time);
  void AddMemsetEventUponApiExit(uint32_t cbid, const MUpti_CallbackData* data,
                                 uint64_t enter_time, uint64_t exit_time);
  void AddMallocFreeEventUponApiExit(uint32_t cbid, const MUpti_CallbackData* data,
                                     uint32_t device_id, uint64_t enter_time,
                                     uint64_t exit_time);
  void AddStreamSynchronizeEventUponApiExit(uint32_t cbid,
                                            const MUpti_CallbackData* data,
                                            uint64_t enter_time,
                                            uint64_t exit_time);
  
  // [MUSA_MOD] 保留此函数声明，暂时可能用不上
  void AddSynchronizeEventUponApiExit(uint32_t cbid, const MUpti_CallbackData* data,
                                      uint64_t enter_time, uint64_t exit_time);

  MusaTracerOptions options_;
  MusaTracer* tracer_ = nullptr;
  MusaTraceCollector* collector_ = nullptr;
  absl::Mutex api_call_start_mutex_;
  // TODO(musa-profiler): replace this with absl hashmap
  // keep a map from the corr. id to enter time for API callbacks.
  std::map<uint32_t, uint64_t> api_call_start_time_
      TF_GUARDED_BY(api_call_start_mutex_);
};

// [MUSA_TODO] Activity Tracing 暂时注释掉
// 原来的实现依赖于 roctracer_record_t (AMD specific)。
// MUSA 应该使用 MUpti_Activity 结构，实现逻辑会完全不同。
// 暂时保留代码结构供参考。
/*
class MusaActivityCallbackImpl {
 public:
  MusaActivityCallbackImpl(const MusaTracerOptions& options, MusaTracer* tracer,
                           MusaTraceCollector* collector)
      : options_(options), tracer_(tracer), collector_(collector) {}

  absl::Status operator()(const char* begin, const char* end);

 private:
  void AddHipKernelActivityEvent(const roctracer_record_t* record);
  void AddNormalHipMemcpyActivityEvent(const roctracer_record_t* record);
  void AddHipMemsetActivityEvent(const roctracer_record_t* record);
  void AddHipMallocActivityEvent(const roctracer_record_t* record);
  void AddHipStreamSynchronizeActivityEvent(const roctracer_record_t* record);
  void AddHccKernelActivityEvent(const roctracer_record_t* record);
  void AddNormalHipOpsMemcpyActivityEvent(const roctracer_record_t* record);
  void AddHipOpsMemsetActivityEvent(const roctracer_record_t* record);
  MusaTracerOptions options_;
  MusaTracer* tracer_ = nullptr;
  MusaTraceCollector* collector_ = nullptr;
};
*/

// The class uses roctracer callback/activity API and forward the collected
// trace events to MusaTraceCollector. There should be only one MusaTracer
// per process.
class MusaTracer {
 public:
  // Returns a pointer to singleton MusaTracer.
  static MusaTracer* GetMusaTracerSingleton();

  // Only one profile session can be live in the same time.
  bool IsAvailable() const;

  void Enable(const MusaTracerOptions& options, MusaTraceCollector* collector);
  void Disable();

  absl::Status ApiCallbackHandler(uint32_t domain, uint32_t cbid,
                                  const void* cbdata);
  
  // [MUSA_TODO] 暂时注释掉 Activity Handler
  // absl::Status ActivityCallbackHandler(const char* begin, const char* end);

  static uint64_t GetTimestamp();
  static int NumGpus();

  // [MUSA_TODO] 以下 PendingActivity 逻辑是为了关联 API 和 Activity 事件的
  // 既然暂时移除了 Activity，这部分也需要注释掉
  /*
  void AddToPendingActivityRecords(uint32_t correlation_id) {
    pending_activity_records_.Add(correlation_id);
  }

  void RemoveFromPendingActivityRecords(uint32_t correlation_id) {
    pending_activity_records_.Remove(correlation_id);
  }

  void ClearPendingActivityRecordsCount() { pending_activity_records_.Clear(); }

  size_t GetPendingActivityRecordsCount() {
    return pending_activity_records_.Count();
  }
  */

 protected:
  // protected constructor for injecting mock cupti interface for testing.
  explicit MusaTracer() : num_gpus_(NumGpus()) {}

 private:
  absl::Status EnableApiTracing();
  absl::Status DisableApiTracing();

  // [MUSA_TODO] Activity Tracing 控制函数
  // absl::Status EnableActivityTracing();
  // absl::Status DisableActivityTracing();

  int num_gpus_;
  std::optional<MusaTracerOptions> options_;
  MusaTraceCollector* collector_ = nullptr;

  bool api_tracing_enabled_ = false;
  
  // [MUSA_MOD] 暂时不需要 Activity tracing 标记
  // bool activity_tracing_enabled_ = false;

  MusaApiCallbackImpl* api_cb_impl_ = nullptr;
  // [新增] 用于存储 MUPTI 返回的订阅句柄
  MUpti_SubscriberHandle subscriber_handle_ = nullptr;
  
  // [MUSA_TODO] 
  // MusaActivityCallbackImpl* activity_cb_impl_;

  // [MUSA_TODO] 原始的 PendingActivityRecords 类定义
  /*
  class PendingActivityRecords {
   public:
    // add a correlation id to the pending set
    void Add(uint32_t correlation_id) {
      absl::MutexLock lock(&mutex);
      pending_set.insert(correlation_id);
    }
    // remove a correlation id from the pending set
    void Remove(uint32_t correlation_id) {
      absl::MutexLock lock(&mutex);
      pending_set.erase(correlation_id);
    }
    // clear the pending set
    void Clear() {
      absl::MutexLock lock(&mutex);
      pending_set.clear();
    }
    // count the number of correlation ids in the pending set
    size_t Count() {
      absl::MutexLock lock(&mutex);
      return pending_set.size();
    }

   private:
    // set of co-relation ids for which the hcc activity record is pending
    absl::flat_hash_set<uint32_t> pending_set;
    // the callback which processes the activity records (and consequently
    // removes items from the pending set) is called in a separate thread
    // from the one that adds item to the list.
    absl::Mutex mutex;
  };
  PendingActivityRecords pending_activity_records_;
  */

 public:
  // Disable copy and move.
  MusaTracer(const MusaTracer&) = delete;
  MusaTracer& operator=(const MusaTracer&) = delete;
};

}  // namespace profiler
}  // namespace xla
#endif  // XLA_BACKENDS_PROFILER_GPU_MUSA_TRACER_H_