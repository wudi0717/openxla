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

#include "xla/backends/profiler/gpu/musa_collector.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h" // 修复 TF_GUARDED_BY 问题
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tsl/platform/logging.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"

namespace xla {
namespace profiler {

using tsl::profiler::XEventBuilder;
using tsl::profiler::XLineBuilder;
using tsl::profiler::XPlaneBuilder;
using tsl::profiler::XSpace;

// =============================================================================
// Helper Functions Implementation
// 注意：这些必须定义在 xla::profiler 命名空间下，不能放在 namespace { ... } 中
// =============================================================================

const char* GetMusaTracerEventTypeName(const MusaTracerEventType& type) {
  switch (type) {
    case MusaTracerEventType::Kernel:
      return "Kernel";
    case MusaTracerEventType::MemcpyH2D:
      return "MemcpyH2D";
    case MusaTracerEventType::MemcpyD2H:
      return "MemcpyD2H";
    case MusaTracerEventType::MemcpyD2D:
      return "MemcpyD2D";
    case MusaTracerEventType::MemcpyP2P:
      return "MemcpyP2P";
    case MusaTracerEventType::MemcpyOther:
      return "MemcpyOther";
    case MusaTracerEventType::MemoryAlloc:
      return "MemoryAlloc";
    case MusaTracerEventType::MemoryFree:
      return "MemoryFree";
    case MusaTracerEventType::Memset:
      return "Memset";
    case MusaTracerEventType::Synchronization:
      return "Synchronization";
    case MusaTracerEventType::Generic:
      return "Generic";
    default:
      return "Unknown";
  }
}

const char* GetMusaTracerEventSourceName(const MusaTracerEventSource& source) {
  switch (source) {
    case MusaTracerEventSource::ApiCallback:
      return "ApiCallback";
    case MusaTracerEventSource::Activity:
      return "Activity";
    default:
      return "Invalid";
  }
}

const char* GetMusaTracerEventDomainName(const MusaTracerEventDomain& domain) {
  switch (domain) {
    case MusaTracerEventDomain::MUSA_API:
      return "MUSA_API";
    case MusaTracerEventDomain::MUSA_OPS:
      return "MUSA_OPS";
    default:
      return "InvalidDomain";
  }
}

// =============================================================================
// AnnotationMap Implementation
// 注意：必须定义在 xla::profiler 命名空间下
// =============================================================================

void AnnotationMap::Add(uint32_t correlation_id, const std::string& annotation) {
  absl::MutexLock lock(&map_.mutex);
  if (map_.correlation_map.size() < max_size_) {
    auto it = map_.annotations.insert(annotation);
    map_.correlation_map.emplace(correlation_id, *it.first);
  }
}

absl::string_view AnnotationMap::LookUp(uint32_t correlation_id) {
  absl::MutexLock lock(&map_.mutex);
  auto it = map_.correlation_map.find(correlation_id);
  return it != map_.correlation_map.end() ? it->second : absl::string_view();
}

// =============================================================================
// Internal Implementation (MusaTraceCollectorImpl)
// 这个类只在内部使用，所以放在匿名命名空间是正确的
// =============================================================================
namespace {

class MusaTraceCollectorImpl : public MusaTraceCollector {
 public:
  MusaTraceCollectorImpl(const MusaTraceCollectorOptions& options,
                         uint64_t start_walltime_ns, uint64_t start_gputime_ns)
      : MusaTraceCollector(options),
        start_walltime_ns_(start_walltime_ns),
        start_gputime_ns_(start_gputime_ns) {}

  void AddEvent(MusaTracerEvent&& event, bool is_auxiliary) override {
    absl::MutexLock lock(&mutex_);
    events_.push_back(std::move(event));
  }

  void OnEventsDropped(const std::string& reason, uint32_t num_events) override {
    LOG(WARNING) << "Dropped " << num_events << " events. Reason: " << reason;
  }

  void Flush() override {
    absl::MutexLock lock(&mutex_);
    events_.clear();
  }

  void Export(XSpace* space) override {
    absl::MutexLock lock(&mutex_);
    
    if (events_.empty()) return;

    // 创建 XPlane，名字必须为 "/device:GPU"
    XPlaneBuilder plane = XPlaneBuilder(space->add_planes());
    plane.SetName(tsl::profiler::kGpuPlanePrefix);
    plane.SetId(0);

    // 分组准备
    absl::flat_hash_map<uint64_t, XLineBuilder> lines;

    // 排序
    std::sort(events_.begin(), events_.end(),
              [](const MusaTracerEvent& a, const MusaTracerEvent& b) {
                return a.start_time_ns < b.start_time_ns;
              });

    // 填充数据
    for (const auto& event : events_) {
      auto it = lines.find(event.thread_id);
      if (it == lines.end()) {
        XLineBuilder line = plane.GetOrCreateLine(event.thread_id);
        line.SetName(absl::StrCat("Thread ", event.thread_id));
        line.SetTimestampNs(start_walltime_ns_); 
        it = lines.emplace(event.thread_id, std::move(line)).first;
      }
      XLineBuilder& line = it->second;

      XEventBuilder xevent = line.AddEvent(*plane.GetOrCreateEventMetadata(event.name));
      
      xevent.SetTimestampNs(event.start_time_ns);
      
      if (event.end_time_ns >= event.start_time_ns) {
        xevent.SetDurationNs(event.end_time_ns - event.start_time_ns);
      } else {
        xevent.SetDurationNs(1);
      }

      if (event.device_id != MusaTracerEvent::kInvalidDeviceId) {
        xevent.AddStatValue(*plane.GetOrCreateStatMetadata("device_id"),
                            event.device_id);
      }
      if (event.correlation_id != MusaTracerEvent::kInvalidCorrelationId) {
        xevent.AddStatValue(*plane.GetOrCreateStatMetadata("correlation_id"),
                            event.correlation_id);
      }
      
      switch (event.type) {
        case MusaTracerEventType::MemcpyH2D:
        case MusaTracerEventType::MemcpyD2H:
        case MusaTracerEventType::MemcpyD2D:
           xevent.AddStatValue(*plane.GetOrCreateStatMetadata("bytes"),
                               static_cast<int64_t>(event.memcpy_info.num_bytes));
           xevent.AddStatValue(*plane.GetOrCreateStatMetadata("async"),
                               event.memcpy_info.async ? 1 : 0);
           break;
        case MusaTracerEventType::MemoryAlloc:
           xevent.AddStatValue(*plane.GetOrCreateStatMetadata("bytes"),
                               static_cast<int64_t>(event.memalloc_info.num_bytes));
           break;
        default:
           break;
      }
    }
  }

 private:
  MusaTraceCollectorOptions options_;
  uint64_t start_walltime_ns_;
  uint64_t start_gputime_ns_;
  
  absl::Mutex mutex_;
  // 修改：使用 ABSL_GUARDED_BY 替代 TF_GUARDED_BY，并引入对应头文件
  std::vector<MusaTracerEvent> events_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace

// =============================================================================
// Factory Function
// =============================================================================

std::unique_ptr<MusaTraceCollector> CreateMusaCollector(
    const MusaTraceCollectorOptions& options, uint64_t start_walltime_ns,
    uint64_t start_gputime_ns) {
  return std::make_unique<MusaTraceCollectorImpl>(options, start_walltime_ns,
                                                  start_gputime_ns);
}

}  // namespace profiler
}  // namespace xla