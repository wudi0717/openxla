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
#include "musa.h"
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

extern "C" {
MUresult MUSAAPI muStreamBeginCaptureToGraph(MUstream hStream, MUgraph hGraph, const MUgraphNode *dependencies, const MUgraphEdgeData *dependencyData, size_t numDependencies, MUstreamCaptureMode mode);
}

namespace stream_executor::gpu {
namespace {
absl::StatusOr<musaGraph_t> CreateGraph() {
  VLOG(2) << "Create new MUSA graph";
  MUgraph graph = nullptr;
  TF_RETURN_IF_ERROR(musa::ToStatus(muGraphCreate(&graph, /*flags=*/0),
                                    "Failed to create MUSA graph"));
  VLOG(2) << "Created MUSA graph " << graph;
  return graph;
}

MUdeviceptr AsDevicePtr(const DeviceMemoryBase& mem) {
  return absl::bit_cast<MUdeviceptr>(mem.opaque());
}

using GraphNodeHandle = GpuCommandBuffer::GraphNodeHandle;
using GraphConditionalHandle = GpuCommandBuffer::GraphConditionalHandle;

// Converts a platform independent GraphNodeHandle into a HIP specific
// musaGraphNode_t.
MUgraphNode ToMusaGraphHandle(GraphNodeHandle handle) {
  return absl::bit_cast<MUgraphNode>(handle);
}

int ToMusaGraphKernelNodePriority(StreamPriority priority) {
  switch (priority) {
    case StreamPriority::Default:
      return 0;
    case StreamPriority::Lowest:
      return -1;
    case StreamPriority::Highest:
      return 1;
    default:
      return 0;
  }
}

MUgraphConditionalHandle ToMusaGraphHandle(GraphConditionalHandle handle) {
  return absl::bit_cast<MUgraphConditionalHandle>(handle);
}

std::vector<MUgraphNode> ToMusaGraphHandles(
    absl::Span<const GraphNodeHandle> opaque_handles) {
  std::vector<MUgraphNode> handles;
  handles.reserve(opaque_handles.size());
  for (const GraphNodeHandle opaque_handle : opaque_handles) {
    handles.push_back(ToMusaGraphHandle(opaque_handle));
  }
  return handles;
}

// Converts a MUSA specific MUgraphNode into a platform independent
// GraphNodeHandle.
GraphNodeHandle FromMusaGraphHandle(MUgraphNode handle) {
  return absl::bit_cast<GraphNodeHandle>(handle);
}

// Converts a MUSA specific MUgraphConditionalHandle into a platform
// independent GraphConditionalHandle.
GraphConditionalHandle FromMusaGraphHandle(MUgraphConditionalHandle handle) {
  return absl::bit_cast<GraphConditionalHandle>(handle);
}

std::string ConditionalTypeToString(GpuCommandBuffer::ConditionType type) {
  switch (type) {
    case GpuCommandBuffer::ConditionType::kIf:
      return "IF";
    case GpuCommandBuffer::ConditionType::kWhile:
      return "WHILE";
  }
}

absl::Status GraphInstantiate(MUgraphExec* exec, MUgraph graph) {
  VLOG(2) << "Instantiate MUSA executable graph from graph " << graph;

  uint64_t mu_flags = MUSA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY;
  return musa::ToStatus(muGraphInstantiate(exec, graph, mu_flags),
                        "Failed to instantiate MUSA graph");
}

}  // namespace

absl::StatusOr<std::unique_ptr<MusaCommandBuffer>> MusaCommandBuffer::Create(
    Mode mode, StreamExecutor* executor, MusaContext* musa_context) {
  TF_ASSIGN_OR_RETURN(MUgraph graph, CreateGraph());
  return std::unique_ptr<MusaCommandBuffer>(new MusaCommandBuffer(
      mode, executor, musa_context, graph, /*is_owned_graph=*/true));
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

absl::StatusOr<MusaCommandBuffer::NoOpKernel*>
MusaCommandBuffer::GetNoOpKernel() {
  return absl::UnimplementedError("NoOpKernel are not supported on MUSA.");
}

absl::StatusOr<GpuCommandBuffer::GraphConditionalNodeHandle>
MusaCommandBuffer::CreateConditionalNode(
    absl::Span<const GraphNodeHandle> dependencies,
    GraphConditionalHandle conditional, ConditionType type) {
  return absl::UnimplementedError("Conditionals are not supported on MUSA.");
}

absl::StatusOr<GraphNodeHandle> MusaCommandBuffer::CreateMemsetNode(
    absl::Span<const GraphNodeHandle> dependencies,
    DeviceMemoryBase destination, BitPattern bit_pattern, size_t num_elements) {
  VLOG(2) << "Add memset node to a graph " << graph_
          << "; dst: " << destination.opaque()
          << "; bit_pattern: " << bit_pattern.ToString()
          << "; num_elements: " << num_elements
          << "; context: " << musa_context_->context()
          << "; deps: " << dependencies.size();

  MUSA_MEMSET_NODE_PARAMS params{};
  params.dst = AsDevicePtr(destination);
  params.elementSize = bit_pattern.GetElementSize();
  params.height = 1;
  params.pitch = 0;  // unused if height is 1
  params.value = bit_pattern.GetPatternBroadcastedToUint32();
  params.width = num_elements;

  std::vector<MUgraphNode> deps = ToMusaGraphHandles(dependencies);

  MUgraphNode node_handle = nullptr;
  TF_RETURN_IF_ERROR(musa::ToStatus(
      muGraphAddMemsetNode(&node_handle, graph_, deps.data(), deps.size(),
                           &params, musa_context_->context()),
      "Failed to add memset node to a MUSA graph"));

  return FromMusaGraphHandle(node_handle);
}

absl::Status MusaCommandBuffer::UpdateMemsetNode(GraphNodeHandle node_handle,
                                                 DeviceMemoryBase destination,
                                                 BitPattern bit_pattern,
                                                 size_t num_elements) {
  VLOG(2) << "Set memset node params " << node_handle << " in graph executable "
          << graph_exec() << "; dst: " << destination.opaque()
          << "; bit_pattern: " << bit_pattern.ToString()
          << "; num_elements: " << num_elements
          << "; context: " << musa_context_->context();

  MUSA_MEMSET_NODE_PARAMS params{};
  params.dst = AsDevicePtr(destination);
  params.elementSize = bit_pattern.GetElementSize();
  params.height = 1;
  params.pitch = 0;  // unused if height is 1
  params.value = bit_pattern.GetPatternBroadcastedToUint32();
  params.width = num_elements;

  return musa::ToStatus(muGraphExecMemsetNodeSetParams(
                            graph_exec(), ToMusaGraphHandle(node_handle),
                            &params, musa_context_->context()),
                        "Failed to set memset node params");
}

absl::StatusOr<GraphNodeHandle> MusaCommandBuffer::CreateMemcpyD2DNode(
    absl::Span<const GraphNodeHandle> dependencies,
    DeviceMemoryBase destination, DeviceMemoryBase source, uint64_t size) {
  VLOG(2) << "Add memcpy d2d node to a graph " << graph_
          << "; dst: " << destination.opaque() << "; src: " << source.opaque()
          << "; size: " << size << "; context: " << musa_context_->context()
          << "; deps: " << dependencies.size();

  MUSA_MEMCPY3D params{};
  params.srcMemoryType = MU_MEMORYTYPE_DEVICE;
  params.srcDevice = AsDevicePtr(source);
  params.dstMemoryType = MU_MEMORYTYPE_DEVICE;
  params.dstDevice = AsDevicePtr(destination);
  params.WidthInBytes = size;
  params.Height = 1;
  params.Depth = 1;

  std::vector<MUgraphNode> deps = ToMusaGraphHandles(dependencies);

  MUgraphNode node_handle = nullptr;
  TF_RETURN_IF_ERROR(musa::ToStatus(
      muGraphAddMemcpyNode(&node_handle, graph_, deps.data(), deps.size(),
                           &params, musa_context_->context()),
      "Failed to add memcpy d2d node to a MUSA graph"));
  return FromMusaGraphHandle(node_handle);
}

absl::Status MusaCommandBuffer::UpdateMemcpyD2DNode(
    GraphNodeHandle node_handle, DeviceMemoryBase destination,
    DeviceMemoryBase source, uint64_t size) {
  VLOG(2) << "Set memcpy d2d node params " << node_handle
          << " in graph executable " << graph_exec()
          << "; dst: " << destination.opaque() << "; src: " << source.opaque()
          << "; size: " << size << "; context: " << musa_context_->context();

  MUSA_MEMCPY3D params{};
  params.srcMemoryType = MU_MEMORYTYPE_DEVICE;
  params.srcDevice = AsDevicePtr(source);
  params.dstMemoryType = MU_MEMORYTYPE_DEVICE;
  params.dstDevice = AsDevicePtr(destination);
  params.WidthInBytes = size;
  params.Height = 1;
  params.Depth = 1;
  return musa::ToStatus(muGraphExecMemcpyNodeSetParams(
                            graph_exec(), ToMusaGraphHandle(node_handle),
                            &params, musa_context_->context()),
                        "Failed to set memcpy d2d node params");
}

absl::Status MusaCommandBuffer::PopulateDnnGraphNode(
    dnn::DnnGraph& dnn_graph, Stream& stream,
    absl::Span<DeviceMemoryBase> operands) {
  return dnn_graph.PopulateOrUpdateRawCommandBuffer(stream, operands, graph_,
                                                    false);
}

absl::Status MusaCommandBuffer::UpdateDnnGraphNode(
    dnn::DnnGraph& dnn_graph, Stream& stream,
    absl::Span<DeviceMemoryBase> operands, GraphNodeHandle node_handle) {
  MUgraph child_graph;
  TF_RETURN_IF_ERROR(musa::ToStatus(muGraphChildGraphNodeGetGraph(
      ToMusaGraphHandle(node_handle), &child_graph)));
  TF_RETURN_IF_ERROR(dnn_graph.PopulateOrUpdateRawCommandBuffer(
      stream, operands, child_graph, true));
  return musa::ToStatus(
      muGraphExecChildGraphNodeSetParams(
          graph_exec(), ToMusaGraphHandle(node_handle), child_graph),
      "Failed to set MUSA graph child node params");
}

absl::StatusOr<GraphNodeHandle> MusaCommandBuffer::CreateChildNode(
    ChildCommandType type, absl::Span<const GraphNodeHandle> dependencies,
    CommandBuffer& nested) {
  auto& child_command_buffer =
      tensorflow::down_cast<MusaCommandBuffer&>(nested);
  CHECK(child_command_buffer.parent_ == nullptr)
      << "Nested command buffer's parent is not null";
  child_command_buffer.parent_ = this;
  MUgraph child_graph = child_command_buffer.graph_;
  VLOG(2) << "Create a new node by cloning the child graph " << child_graph
          << " and add it to " << graph_ << "; deps: " << dependencies.size();

  std::vector<MUgraphNode> deps = ToMusaGraphHandles(dependencies);

  MUgraphNode node_handle;
  if (type == ChildCommandType::kCloned) {
    TF_RETURN_IF_ERROR(musa::ToStatus(
        muGraphAddChildGraphNode(&node_handle, graph_, deps.data(), deps.size(),
                                 child_graph),
        "Failed to create a child graph node and add it to a MUSA graph"));
    VLOG(5) << "CreateClonedChildNode: "
            << reinterpret_cast<const void*>(&node_handle);

    return FromMusaGraphHandle(node_handle);

  } else if (type == ChildCommandType::kMoved) {
#if 0
    child_command_buffer.is_owned_graph_ = false;
    MUgraphNodeParams nodeParams;
    std::memset(&nodeParams, 0, sizeof(nodeParams));
    nodeParams.type = MU_GRAPH_NODE_TYPE_GRAPH;
    nodeParams.graph.graph = child_graph;
    nodeParams.graph.ownership = MU_GRAPH_CHILD_GRAPH_OWNERSHIP_MOVE;
    VLOG(2) << "Create a new node by moving the child graph " << child_graph
            << " and add it to " << graph_ << "; deps: " << dependencies.size();

    std::vector<MUgraphNode> deps = ToMusaGraphHandles(dependencies);

    MUgraphNode node_handle;
    TF_RETURN_IF_ERROR(musa::ToStatus(
        muGraphAddNode_v2(&node_handle, graph_, deps.data(),
                          /*dependencyData=*/nullptr, deps.size(), &nodeParams),
        "Failed to create a child graph node and add it to a MUSA graph"));
    return FromMusaGraphHandle(node_handle);
#else
    return absl::UnimplementedError(
        "Moved child node is not supported for MUSA < 12.9");
#endif
  } else {
    return absl::InternalError("Unsupported child command type");
  }
}

absl::Status MusaCommandBuffer::UpdateChildNode(ChildCommandType type,
                                                GraphNodeHandle node_handle,
                                                const CommandBuffer& nested) {
  CHECK(type == ChildCommandType::kCloned)
      << "Moved child node update is not supported";
  MUgraph child_graph =
      tensorflow::down_cast<const MusaCommandBuffer&>(nested).graph_;
  VLOG(2) << "Set child node params " << node_handle << " in graph executable "
          << graph_exec() << " to params contained in " << child_graph;

  MUgraphExec exec_update = graph_exec();
  CHECK(exec_update != nullptr) << "graph executor for update is nullptr";
  return musa::ToStatus(
      muGraphExecChildGraphNodeSetParams(
          exec_update, ToMusaGraphHandle(node_handle), child_graph),
      "Failed to set MUSA graph child node params");
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

  MUSA_KERNEL_NODE_PARAMS params{};

  MUfunction function = static_cast<const MusaKernel&>(kernel).gpu_function();
  params.func = function;
  params.gridDimX = blocks.x;
  params.gridDimY = blocks.y;
  params.gridDimZ = blocks.z;
  params.blockDimX = threads.x;
  params.blockDimY = threads.y;
  params.blockDimZ = threads.z;
  params.sharedMemBytes = shared_mem_bytes;
  params.kernelParams = const_cast<void**>(args.argument_addresses().data());
  params.extra = nullptr;

  // TODO(ezhulenev): Why do we do it on every call to launch kernel? This
  // should be moved one level up to se::Kernel level, and done just once (or
  // updated once we get a new larger shared memory request).
  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(musa::ToStatus(
        muFuncSetAttribute(function,
                           MU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_mem_bytes),
        "Failed to set shared memory size"));
  }

  std::vector<MUgraphNode> deps = ToMusaGraphHandles(dependencies);

  MUgraphNode node_handle = nullptr;
  TF_RETURN_IF_ERROR(
      musa::ToStatus(muGraphAddKernelNode(&node_handle, graph_, deps.data(),
                                          deps.size(), &params),
                     "Failed to add kernel node to a MUSA graph"));

  if (priority != StreamPriority::Default) {
    MUlaunchAttributeValue value;
    value.priority = ToMusaGraphKernelNodePriority(priority);
    TF_RETURN_IF_ERROR(
        musa::ToStatus(muGraphKernelNodeSetAttribute(
                           node_handle, MU_LAUNCH_ATTRIBUTE_PRIORITY, &value),
                       "Failed to set kernel node priority"));
  }
  return FromMusaGraphHandle(node_handle);
}

absl::Status MusaCommandBuffer::UpdateKernelNode(
    GraphNodeHandle node_handle, const ThreadDim& threads,
    const BlockDim& blocks, const Kernel& kernel,
    const KernelArgsPackedArrayBase& args) {
  const uint64_t shared_mem_bytes = args.number_of_shared_bytes();

  VLOG(2) << "Set kernel node params " << node_handle << " in graph executable "
          << graph_exec() << "; kernel: " << kernel.name()
          << "; gdx: " << blocks.x << " gdy: " << blocks.y
          << " gdz: " << blocks.z << " bdx: " << threads.x
          << " bdy: " << threads.y << " bdz: " << threads.z
          << "; shmem: " << shared_mem_bytes;

  MUSA_KERNEL_NODE_PARAMS params{};
  MUfunction function = static_cast<const MusaKernel&>(kernel).gpu_function();
  params.func = function;
  params.gridDimX = blocks.x;
  params.gridDimY = blocks.y;
  params.gridDimZ = blocks.z;
  params.blockDimX = threads.x;
  params.blockDimY = threads.y;
  params.blockDimZ = threads.z;
  params.sharedMemBytes = shared_mem_bytes;
  params.kernelParams = const_cast<void**>(args.argument_addresses().data());
  params.extra = nullptr;

  // TODO(ezhulenev): Why do we do it on every call to launch kernel? This
  // should be moved one level up to se::Kernel level, and done just once (or
  // updated once we get a new larger shared memory request).
  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(musa::ToStatus(
        muFuncSetAttribute(function,
                           MU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_mem_bytes),
        "Failed to set shared memory size"));
  }
  return musa::ToStatus(
      muGraphExecKernelNodeSetParams(graph_exec(),
                                     ToMusaGraphHandle(node_handle), &params),
      "Failed to set MUSA graph kernel node params");
}

absl::StatusOr<GraphNodeHandle> MusaCommandBuffer::CreateEmptyNode(
    absl::Span<const GraphNodeHandle> dependencies) {
  VLOG(2) << "Add empty node to a graph " << graph_
          << "; deps: " << dependencies.size();

  std::vector<MUgraphNode> deps = ToMusaGraphHandles(dependencies);

  MUgraphNode node_handle = nullptr;
  TF_RETURN_IF_ERROR(musa::ToStatus(
      muGraphAddEmptyNode(&node_handle, graph_, deps.data(), deps.size()),
      "Failed to add empty node to a MUSA graph"));

  return FromMusaGraphHandle(node_handle);
}

absl::Status MusaCommandBuffer::Trace(
    Stream* stream, absl::AnyInvocable<absl::Status()> function) {
#if 0
  return absl::UnimplementedError(
      "StreamBeginCaptureToGraph is not implemented for MUSA below version "
      "12.3. Therefore tracing is not supported.");
#else

  TF_RETURN_IF_ERROR(CheckNotFinalized());

  VLOG(5) << "Trace into GPU command buffer graph " << graph_
          << " on a stream: " << stream;

  MUstream stream_handle =
      absl::bit_cast<MUstream>(stream->platform_specific_handle().stream);

  // Switch stream into the capture mode.
  uint64_t start_nanos = tsl::Env::Default()->NowNanos();

  TF_RETURN_IF_ERROR(musa::ToStatus(
      muStreamBeginCaptureToGraph(stream_handle, graph_,
                                  /*dependencies=*/nullptr,
                                  /*dependencyData=*/nullptr,
                                  /*numDependencies=*/0,
                                  MU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
      "Failed to begin stream capture to graph"));
  auto traced = function();

  // Always stop capturing the stream before checking `traced` result.
  VLOG(5) << "End stream " << stream << " capture";
  MUgraph captured_graph;
  TF_RETURN_IF_ERROR(
      musa::ToStatus(muStreamEndCapture(stream_handle, &captured_graph),
                     "Failed to end stream capture"));
  DCHECK(captured_graph == graph_) << "Stream capture should update graph_";
  uint64_t end_nanos = tsl::Env::Default()->NowNanos();

  if (!traced.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to capture gpu graph: ", traced.message()));
  }

  VLOG(5) << "Traced into the GPU command buffer graph " << graph_ << " (took "
          << (end_nanos - start_nanos) / 1000 << " μs)";

  // Check that traced graph is not empty. Trying to instantiate a MUSA graph
  // with empty child node leads to a crash.
  size_t num_root_nodes = 0;
  TF_RETURN_IF_ERROR(musa::ToStatus(
      muGraphGetRootNodes(captured_graph, nullptr, &num_root_nodes)));

  if (num_root_nodes == 0) {
    return absl::InternalError(
        "Traced MUSA graph is empty. Traced function (custom call) did not "
        "launch any MUSA operations on the captured MUSA stream. Instantiating "
        "empty child nodes leads to MUSA crashes.");
  }

  return absl::OkStatus();
#endif
}

absl::Status MusaCommandBuffer::LaunchGraph(Stream* stream) {
  VLOG(3) << "Launch command buffer executable graph " << graph_exec()
          << " on a stream: " << stream;
  return musa::ToStatus(
      muGraphLaunch(
          graph_exec(),
          absl::bit_cast<MUstream>(stream->platform_specific_handle().stream)),
      "Failed to launch MUSA graph");
}

absl::StatusOr<size_t> MusaCommandBuffer::GetNodeCount() const {
  size_t num_nodes;
  TF_RETURN_IF_ERROR(
      musa::ToStatus(muGraphGetNodes(graph_, /*nodes=*/nullptr, &num_nodes)));
  return num_nodes;
}

absl::Status MusaCommandBuffer::SetPriority(StreamPriority priority) {
  size_t num_nodes;
  TF_RETURN_IF_ERROR(
      musa::ToStatus(muGraphGetNodes(graph_, /*nodes=*/nullptr, &num_nodes)));

  std::vector<MUgraphNode> nodes(num_nodes);
  TF_RETURN_IF_ERROR(
      musa::ToStatus(muGraphGetNodes(graph_, nodes.data(), &num_nodes)));

  for (size_t i = 0; i < num_nodes; i++) {
    MUgraphNodeType type;
    TF_RETURN_IF_ERROR(musa::ToStatus(muGraphNodeGetType(nodes[i], &type),
                                      "Failed to get kernel node type"));

    if (type == MU_GRAPH_NODE_TYPE_KERNEL) {
      MUlaunchAttributeValue value;
      value.priority = ToMusaGraphKernelNodePriority(priority);
      TF_RETURN_IF_ERROR(
          musa::ToStatus(muGraphKernelNodeSetAttribute(
                             nodes[i], MU_LAUNCH_ATTRIBUTE_PRIORITY, &value),
                         "Failed to set kernel node priority"));
    }
  }
  return absl::OkStatus();
}

absl::Status MusaCommandBuffer::PrepareFinalization() {
  if (stream_exec_->GetDeviceDescription().driver_version() <
      SemanticVersion{12, 8, 0}) {
    // For MUSA < 12080, musa graph conditional node does not support
    // empty body graph.
    TF_ASSIGN_OR_RETURN(auto node_count, GetNodeCount());
    if (node_count > 0) {
      return absl::OkStatus();
    }

    TF_ASSIGN_OR_RETURN(NoOpKernel * noop, GetNoOpKernel());
    TF_RETURN_IF_ERROR(
        CreateLaunch(*noop, ThreadDim(), BlockDim(), {}).status());
  }
  return absl::OkStatus();
}

absl::StatusOr<GraphConditionalHandle>
MusaCommandBuffer::CreateConditionalHandle() {
  constexpr int kDefaultLaunchValue = 0;
  constexpr int kNoFlags = 0;
  VLOG(2) << "Create conditional handle for a graph " << graph_
          << "; context: " << musa_context_
          << "; default_launch_value: " << kDefaultLaunchValue
          << "; flags: " << kNoFlags;

#if 0
  MUgraphConditionalHandle handle;
  TF_RETURN_IF_ERROR(musa::ToStatus(
      muGraphConditionalHandleCreate(&handle, graph_, musa_context_->context(),
                                     kDefaultLaunchValue, kNoFlags),
      "Failed to create conditional handle for a MUSA graph"));
  return FromMusaGraphHandle(handle);
#else
  return absl::UnimplementedError(
      "MUSA graph conditional nodes are not implemented");
#endif  // MUSA_VERSION >= 12030
}

absl::Status MusaCommandBuffer::WriteGraphToDotFile(absl::string_view path) {
#if 1
  VLOG(2) << "Print MUSA graph " << graph_ << " debug dot file to " << path;

  int flags = MU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE;
  return musa::ToStatus(
      muGraphDebugDotPrint(graph_, std::string{path}.c_str(), flags),
      "Failed to print gpu graph debug file");
#endif  // MUSA_VERSION >= 12000

  return absl::UnimplementedError(
      "MUSA graph debug dot print is not supported.");
}

absl::Status MusaCommandBuffer::InstantiateGraph() {
  // If we get a "resource exhausted error" we retry instantiating Gpu graph
  // one more time after releasing unused device memory allocated for graphs.
  auto instantiated = GraphInstantiate(&graph_exec_, graph_);
  if (instantiated.code() == absl::StatusCode::kResourceExhausted) {
    LOG(WARNING) << "Retry MUSA graph instantiation after OOM error";
    MUdevice device;
    TF_RETURN_IF_ERROR(
        musa::ToStatus(muDeviceGet(&device, stream_exec_->device_ordinal()),
                       "Failed call to cuDeviceGet"));
    TF_RETURN_IF_ERROR(musa::ToStatus(muDeviceGraphMemTrim(device),
                                      "Failed to trim device graph memory"));
    TF_RETURN_IF_ERROR(GraphInstantiate(&graph_exec_, graph_));
  } else {
    TF_RETURN_IF_ERROR(instantiated);
  }

  return absl::OkStatus();
}

MUgraphExec MusaCommandBuffer::graph_exec() const {
  const MusaCommandBuffer* current = this;
  while (current->parent_ != nullptr) {
    current = current->parent_;
  }
  CHECK(current->graph_exec_ != nullptr)
      << "graph_exec_ is nullptr for top level musa command buffer";
  return current->graph_exec_;
}

MusaCommandBuffer::~MusaCommandBuffer() {
  if (graph_exec_ != nullptr) {
    auto exec_num = NotifyExecDestroyed();
    VLOG(5) << "Destroy GPU command buffer executable graph " << graph_exec_
            << " "
            << "(remaining alive executable graphs: " << exec_num << ")";
    if (auto status = musa::ToStatus(muGraphExecDestroy(graph_exec_),
                                     "Failed to destroy MUSA executable graph");
        !status.ok()) {
      LOG(ERROR) << status.message();
    }
  }
  if (graph_ != nullptr && is_owned_graph_) {
    if (auto status = musa::ToStatus(muGraphDestroy(graph_),
                                     "Failed to destroy MUSA graph");
        !status.ok()) {
      LOG(ERROR) << status.message();
    }
  }
}

absl::Status MusaCommandBuffer::CheckCanBeUpdated() {
  if (graph_exec() == nullptr) {
    return absl::InternalError(
        "Command buffer has to have a graph executable to be updated.");
  }
  return absl::OkStatus();
}
}  // namespace stream_executor::gpu
