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

#include "xla/stream_executor/musa/musa_command_buffer.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "driver_types.h"
#include "musa_runtime.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/musa/musa_driver_wrapper.h"
#include "xla/stream_executor/musa/musa_kernel.h"
#include "xla/stream_executor/musa/musa_status.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"

namespace stream_executor::gpu {
namespace {
absl::StatusOr<musaGraph_t> CreateGraph() {
  VLOG(2) << "Create new MUSA graph";
  musaGraph_t graph;
  TF_RETURN_IF_ERROR(ToStatus(musaGraphCreate(&graph, /*flags=*/0),
                              "Failed to create MUSA graph"));
  VLOG(2) << "Created MUSA graph " << graph;
  return graph;
}

void *AsDevicePtr(const DeviceMemoryBase& mem) {
  return absl::bit_cast<void *>(mem.opaque());
}

using GraphNodeHandle = GpuCommandBuffer::GraphNodeHandle;

// Converts a platform independent GraphNodeHandle into a HIP specific
// musaGraphNode_t.
musaGraphNode_t ToMusaGraphHandle(GpuCommandBuffer::GraphNodeHandle handle) {
  return absl::bit_cast<musaGraphNode_t>(handle);
}

// Converts a list of platform independent GraphNodeHandles into a list of
// HIP specific musaGraphNode_t.
std::vector<musaGraphNode_t> ToMusaGraphHandles(
    absl::Span<const GraphNodeHandle> opaque_handles) {
  std::vector<musaGraphNode_t> handles;
  handles.reserve(opaque_handles.size());
  for (const GraphNodeHandle opaque_handle : opaque_handles) {
    handles.push_back(ToMusaGraphHandle(opaque_handle));
  }
  return handles;
}

// Converts a HIP specific musaGraphNode_t into a platform independent
// GraphNodeHandle. This function will be removed once all Node factory
// functions have been migrated into the subclasses.
GraphNodeHandle FromMusaGraphHandle(musaGraphNode_t handle) {
  return absl::bit_cast<GpuCommandBuffer::GraphNodeHandle>(handle);
}
}  // namespace

absl::StatusOr<std::unique_ptr<MusaCommandBuffer>> MusaCommandBuffer::Create(
    Mode mode, StreamExecutor* executor) {
  TF_ASSIGN_OR_RETURN(musaGraph_t graph, CreateGraph());
  return std::unique_ptr<MusaCommandBuffer>(
      new MusaCommandBuffer(mode, executor, graph, /*is_owned_graph=*/true));
}

absl::StatusOr<GpuCommandBuffer::GraphConditionalNodeHandle>
MusaCommandBuffer::CreateConditionalNode(
    absl::Span<const GraphNodeHandle> dependencies,
    GraphConditionalHandle conditional, ConditionType type) {
  return absl::UnimplementedError("Conditionals are not supported on MUSA.");
}

absl::StatusOr<GraphNodeHandle> MusaCommandBuffer::CreateSetCaseConditionNode(
    absl::Span<const GraphConditionalHandle> conditionals,
    DeviceMemory<uint8_t> index, bool index_is_bool, int32_t batch_offset,
    bool enable_conditional_default,
    absl::Span<const GraphNodeHandle> dependencies) {
  return absl::UnimplementedError("Conditionals are not supported on MUSA.");
}

absl::Status MusaCommandBuffer::UpdateSetCaseConditionNode(
    GraphNodeHandle handle,
    absl::Span<const GraphConditionalHandle> conditionals,
    DeviceMemory<uint8_t> index, bool index_is_bool, int32_t batch_offset,
    bool enable_conditional_default) {
  return absl::UnimplementedError("Conditionals are not supported on MUSA.");
}

absl::StatusOr<GraphNodeHandle> MusaCommandBuffer::CreateSetWhileConditionNode(
    GraphConditionalHandle conditional, DeviceMemory<bool> predicate,
    absl::Span<const GraphNodeHandle> dependencies) {
  return absl::UnimplementedError("Conditionals are not supported on MUSA.");
}

absl::Status MusaCommandBuffer::UpdateSetWhileConditionNode(
    GraphNodeHandle handle, GraphConditionalHandle conditional,
    DeviceMemory<bool> predicate) {
  return absl::UnimplementedError("Conditionals are not supported on MUSA.");
}

absl::StatusOr<GraphNodeHandle> MusaCommandBuffer::CreateMemsetNode(
    absl::Span<const GraphNodeHandle> dependencies,
    DeviceMemoryBase destination, BitPattern bit_pattern, size_t num_elements) {
  VLOG(2) << "Add memset node to a graph " << graph_
          << "; dst: " << destination.opaque()
          << "; bit_pattern: " << bit_pattern.ToString()
          << "; num_elements: " << num_elements
          << "; deps: " << dependencies.size();

  musaMemsetParams params{};
  params.dst = AsDevicePtr(destination);
  params.elementSize = bit_pattern.GetElementSize();
  params.height = 1;
  params.pitch = 0;  // unused if height is 1
  params.value = bit_pattern.GetPatternBroadcastedToUint32();
  params.width = num_elements;

  std::vector<musaGraphNode_t> deps = ToMusaGraphHandles(dependencies);

  musaGraphNode_t node_handle = nullptr;
  TF_RETURN_IF_ERROR(
      ToStatus(musaGraphAddMemsetNode(&node_handle, graph_, deps.data(),
                                           deps.size(), &params),
               "Failed to add memset node to a HIP graph"));
  return FromMusaGraphHandle(node_handle);
}

absl::Status MusaCommandBuffer::UpdateMemsetNode(GraphNodeHandle node_handle,
                                                 DeviceMemoryBase destination,
                                                 BitPattern bit_pattern,
                                                 size_t num_elements) {
  VLOG(2) << "Set memset node params " << node_handle << " in graph executable "
          << exec_ << "; dst: " << destination.opaque()
          << "; bit_pattern: " << bit_pattern.ToString()
          << "; num_elements: " << num_elements;

  return absl::UnimplementedError("Empty nodes are not supported on MUSA.");
}

absl::StatusOr<GraphNodeHandle> MusaCommandBuffer::CreateMemcpyD2DNode(
    absl::Span<const GraphNodeHandle> dependencies,
    DeviceMemoryBase destination, DeviceMemoryBase source, uint64_t size) {
  VLOG(2) << "Add memcpy d2d node to a graph " << graph_
          << "; dst: " << destination.opaque() << "; src: " << source.opaque()
          << "; size: " << size << "; deps: " << dependencies.size();

  std::vector<musaGraphNode_t> deps = ToMusaGraphHandles(dependencies);

  musaGraphNode_t node_handle = nullptr;
  TF_RETURN_IF_ERROR(ToStatus(
      musaGraphAddMemcpyNode1D(&node_handle, graph_, deps.data(),
                                    deps.size(), AsDevicePtr(destination),
                                    AsDevicePtr(source), size,
                                    musaMemcpyDeviceToDevice),
      "Failed to add memcpy d2d node to a HIP graph"));
  return FromMusaGraphHandle(node_handle);
}

absl::Status MusaCommandBuffer::UpdateMemcpyD2DNode(
    GraphNodeHandle node_handle, DeviceMemoryBase destination,
    DeviceMemoryBase source, uint64_t size) {
  VLOG(2) << "Set memcpy d2d node params " << node_handle
          << " in graph executable " << exec_
          << "; dst: " << destination.opaque() << "; src: " << source.opaque()
          << "; size: " << size;

  return absl::InternalError("Not Support");
}

absl::StatusOr<GraphNodeHandle> MusaCommandBuffer::CreateChildNode(
    ChildCommandType type, absl::Span<const GraphNodeHandle> dependencies,
    CommandBuffer& nested) {
  CHECK(type == ChildCommandType::kCloned)
      << "Moved child node creation is not supported";
  auto& child_command_buffer =
      tensorflow::down_cast<MusaCommandBuffer&>(nested);
  CHECK(child_command_buffer.parent_ == nullptr)
      << "Nested command buffer's parent is not null";
  child_command_buffer.parent_ = this;
  musaGraph_t child_graph = child_command_buffer.graph_;
  VLOG(2) << "Create a new node by cloning the child graph " << child_graph
          << " and add it to " << graph_ << "; deps: " << dependencies.size();

  std::vector<musaGraphNode_t> deps = ToMusaGraphHandles(dependencies);

  musaGraphNode_t node_handle = nullptr;
  TF_RETURN_IF_ERROR(ToStatus(
      musaGraphAddChildGraphNode(&node_handle, graph_, deps.data(),
                                      deps.size(), child_graph),
      "Failed to create a child graph node and add it to a HIP graph"));
  return FromMusaGraphHandle(node_handle);
}

absl::Status MusaCommandBuffer::UpdateChildNode(ChildCommandType type,
                                                GraphNodeHandle node_handle,
                                                const CommandBuffer& nested) {
  CHECK(type == ChildCommandType::kCloned)
      << "Moved child node update is not supported";
  musaGraph_t child_graph =
      tensorflow::down_cast<const MusaCommandBuffer&>(nested).graph_;

  VLOG(2) << "Set child node params " << node_handle << " in graph executable "
          << exec_ << "to params contained in " << child_graph;

    return absl::InternalError(
        "Not support");
}

absl::StatusOr<GraphNodeHandle> MusaCommandBuffer::CreateKernelNode(
    absl::Span<const GraphNodeHandle> dependencies, StreamPriority priority,
    const ThreadDim& threads, const BlockDim& blocks, const Kernel& kernel,
    const KernelArgsPackedArrayBase& args) {
  const uint64_t shared_mem_bytes = args.number_of_shared_bytes();

  VLOG(2) << "Add kernel node to a graph " << graph_
          << "; kernel: " << kernel.name() << "; gdx: " << blocks.x
          << " gdy: " << blocks.y << " gdz: " << blocks.z
          << " bdx: " << threads.x << " bdy: " << threads.y
          << " bdz: " << threads.z << "; shmem: " << shared_mem_bytes
          << "; deps: " << dependencies.size();

  musaKernelNodeParams params{};

  MUfunction function =
      static_cast<const MusaKernel&>(kernel).gpu_function();
  params.func = function;
  params.gridDim.x = blocks.x;
  params.gridDim.y = blocks.y;
  params.gridDim.z = blocks.z;
  params.blockDim.x = threads.x;
  params.blockDim.y = threads.y;
  params.blockDim.z = threads.z;
  params.sharedMemBytes = shared_mem_bytes;
  params.kernelParams = const_cast<void**>(args.argument_addresses().data());
  params.extra = nullptr;

  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(ToStatus(
        musaFuncSetAttribute(function,
                                  musaFuncAttributeMaxDynamicSharedMemorySize,
                                  shared_mem_bytes),
        "Failed to set shared memory size"));
  }

  std::vector<musaGraphNode_t> deps = ToMusaGraphHandles(dependencies);

  musaGraphNode_t node_handle = nullptr;
  TF_RETURN_IF_ERROR(
      ToStatus(musaGraphAddKernelNode(&node_handle, graph_, deps.data(),
                                           deps.size(), &params),
               "Failed to add kernel node to a HIP graph"));

  return FromMusaGraphHandle(node_handle);
}

absl::Status MusaCommandBuffer::UpdateKernelNode(
    GraphNodeHandle node_handle, const ThreadDim& threads,
    const BlockDim& blocks, const Kernel& kernel,
    const KernelArgsPackedArrayBase& args) {
  const uint64_t shared_mem_bytes = args.number_of_shared_bytes();

  VLOG(2) << "Set kernel node params " << node_handle << " in graph executable "
          << exec_ << "; kernel: " << kernel.name() << "; gdx: " << blocks.x
          << " gdy: " << blocks.y << " gdz: " << blocks.z
          << " bdx: " << threads.x << " bdy: " << threads.y
          << " bdz: " << threads.z << "; shmem: " << shared_mem_bytes;

  musaKernelNodeParams params{};

  MUfunction function =
      static_cast<const MusaKernel&>(kernel).gpu_function();
  params.func = function;
  params.gridDim.x = blocks.x;
  params.gridDim.y = blocks.y;
  params.gridDim.z = blocks.z;
  params.blockDim.x = threads.x;
  params.blockDim.y = threads.y;
  params.blockDim.z = threads.z;
  params.sharedMemBytes = shared_mem_bytes;
  params.kernelParams = const_cast<void**>(args.argument_addresses().data());
  params.extra = nullptr;

  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(ToStatus(
        musaFuncSetAttribute(function,
                                  musaFuncAttributeMaxDynamicSharedMemorySize,
                                  shared_mem_bytes),
        "Failed to set shared memory size"));
  }

  return ToStatus(musaGraphExecKernelNodeSetParams(
                      exec_, ToMusaGraphHandle(node_handle), &params),
                  "Failed to set HIP graph kernel node params");
}

absl::StatusOr<GraphNodeHandle> MusaCommandBuffer::CreateEmptyNode(
    absl::Span<const GraphNodeHandle> dependencies) {
  return absl::UnimplementedError("Empty nodes are not supported on MUSA.");
}

absl::Status MusaCommandBuffer::Trace(
    Stream* stream, absl::AnyInvocable<absl::Status()> function) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());
  TF_ASSIGN_OR_RETURN(size_t count, GetNodeCount());
  if (count != 0 || !is_owned_graph_)
    return absl::InternalError(
        "Stream can't be traced on non empty command buffer");

  VLOG(5) << "Trace into GPU command buffer graph " << graph_
          << " on a stream: " << stream;

  musaStream_t stream_handle =
      static_cast<musaStream_t>(stream->platform_specific_handle().stream);

  // Switch stream into the capture mode.
  uint64_t start_nanos = tsl::Env::Default()->NowNanos();
  TF_RETURN_IF_ERROR(
      ToStatus(musaStreamBeginCapture(stream_handle,
                                           musaStreamCaptureModeThreadLocal),
               "Failed to begin stream capture"));
  auto traced = function();

  // Always stop capturing the stream before checking `traced` result.
  VLOG(5) << "End stream " << stream << " capture";
  musaGraph_t captured_graph;
  TF_RETURN_IF_ERROR(
      ToStatus(musaStreamEndCapture(stream_handle, &captured_graph),
               "Failed to end stream capture"));
  TF_RETURN_IF_ERROR(
      ToStatus(musaGraphDestroy(std::exchange(graph_, captured_graph)),
               "Failed to destroy HIP graph"));
  uint64_t end_nanos = tsl::Env::Default()->NowNanos();

  if (!traced.ok())
    return absl::InternalError(
        absl::StrCat("Failed to capture gpu graph: ", traced.message()));

  VLOG(5) << "Traced into the GPU command buffer graph " << graph_ << " (took "
          << (end_nanos - start_nanos) / 1000 << " μs)";

  return absl::OkStatus();
}

absl::Status MusaCommandBuffer::LaunchGraph(Stream* stream) {
  VLOG(3) << "Launch command buffer executable graph " << exec_
          << " on a stream: " << stream;
  return ToStatus(musaGraphLaunch(
                      exec_, static_cast<musaStream_t>(
                                 stream->platform_specific_handle().stream)),
                  "Failed to launch HIP graph");
}
absl::StatusOr<size_t> MusaCommandBuffer::GetNodeCount() const {
  size_t numNodes;
  TF_RETURN_IF_ERROR(
      ToStatus(musaGraphGetNodes(graph_, /*nodes=*/nullptr, &numNodes),
               "Failed to get HIP graph node count"));

  return numNodes;
}

absl::Status MusaCommandBuffer::PrepareFinalization() {
  return absl::OkStatus();
}

absl::StatusOr<GpuCommandBuffer::GraphConditionalHandle>
MusaCommandBuffer::CreateConditionalHandle() {
  return absl::UnimplementedError(
      "Graph conditionals are not yet supported on HIP graphs.");
}

absl::Status MusaCommandBuffer::WriteGraphToDotFile(absl::string_view path) {
  VLOG(2) << "Print HIP graph " << graph_ << " debug dot file to " << path;

  int flags = musaGraphDebugDotFlagsVerbose;
  return absl::UnimplementedError(
      "Graph conditionals are not yet supported on HIP graphs.");
}

absl::Status MusaCommandBuffer::InstantiateGraph() {
  VLOG(2) << "Instantiate HIP executable graph from graph " << graph_;
  return ToStatus(
      musaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0),
      "Failed to instantiate HIP graph");
}

MusaCommandBuffer::~MusaCommandBuffer() {
  if (exec_ != nullptr) {
    auto exec_num = NotifyExecDestroyed();
    VLOG(5) << "Destroy GPU command buffer executable graph " << exec_ << " "
            << "(remaining alive executable graphs: " << exec_num << ")";
    if (auto status = ToStatus(musaGraphExecDestroy(exec_),
                               "Failed to destroy HIP executable graph");
        !status.ok()) {
      LOG(ERROR) << status.message();
    }
  }
  if (graph_ != nullptr && is_owned_graph_) {
    if (auto status =
            ToStatus(musaGraphDestroy(graph_), "Failed to destroy HIP graph");
        !status.ok()) {
      LOG(ERROR) << status.message();
    }
  }
}
absl::Status MusaCommandBuffer::CheckCanBeUpdated() {
  if (exec_ == nullptr) {
    return absl::InternalError(
        "Command buffer has to have a graph executable to be updated.");
  }
  return absl::OkStatus();
}

}  // namespace stream_executor::gpu
