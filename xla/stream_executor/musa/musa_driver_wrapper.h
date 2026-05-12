/* Copyright 2019 The OpenXLA Authors.

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

// This file wraps musa driver calls with dso loader so that we don't need to
// have explicit linking to libmusa. All TF musa driver usage should route
// through this wrapper.

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_DRIVER_WRAPPER_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_DRIVER_WRAPPER_H_

#include "musa.h"
#include "musa_runtime.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/dso_loader.h"

namespace stream_executor {
namespace wrap {
	/*
// Use static linked library
#define STREAM_EXECUTOR_MUSA_WRAP(musaSymbolName)                            \
  template <typename... Args>                                              \
  auto musaSymbolName(Args... args) -> decltype(::musaSymbolName(args...)) { \
    return ::musaSymbolName(args...);                                       \
  }

// clang-format off
// IMPORTANT: if you add a new MUSA API to this list, please notify
// the musa-profiler developers to track the API traces.
#define MUSA_ROUTINE_EACH(__macro)                   \
  __macro(muCtxGetDevice)                          \
  __macro(muCtxSetCurrent)                         \
  __macro(muCtxGetDevice)                         \
  __macro(musaCtxEnablePeerAccess)                   \
  __macro(muDeviceCanAccessPeer)                   \
  __macro(musaDeviceEnablePeerAccess)                \
  __macro(muDeviceGet)                             \
  __macro(musaDeviceGetAttribute)                    \
  __macro(muDeviceGetName)                         \
  __macro(musaDeviceGetPCIBusId)                     \
  __macro(musaDeviceGetSharedMemConfig)              \
  __macro(musaDeviceGetStreamPriorityRange)          \
  __macro(musaDeviceGraphMemTrim)                    \
  __macro(muDevicePrimaryCtxGetState)              \
  __macro(muDevicePrimaryCtxSetFlags)              \
  __macro(muDevicePrimaryCtxRetain)                \
  __macro(muDevicePrimaryCtxRelease)               \
  __macro(musaDeviceSetSharedMemConfig)              \
  __macro(musaDeviceSynchronize)                     \
  __macro(muDeviceTotalMem)                        \
  __macro(musaDriverGetVersion)                      \
  __macro(musaEventCreateWithFlags)                  \
  __macro(musaEventDestroy)                          \
  __macro(musaEventElapsedTime)                      \
  __macro(musaEventQuery)                            \
  __macro(musaEventRecord)                           \
  __macro(musaEventSynchronize)                      \
  __macro(musaFree)                                  \
  __macro(musaFuncSetCacheConfig)                    \
  __macro(muFuncGetAttribute)                      \
  __macro(muFuncSetAttribute)                      \
  __macro(musaGetDevice)                             \
  __macro(musaGetDeviceCount)                        \
  __macro(musaGetDeviceProperties)                   \
  __macro(musaGetErrorString)                        \
  __macro(musaGraphAddKernelNode)                    \
  __macro(musaGraphAddChildGraphNode)                \
  __macro(musaGraphAddEmptyNode)                     \
  __macro(musaGraphAddMemAllocNode)                  \
  __macro(musaGraphAddMemcpyNode1D)                  \
  __macro(musaGraphAddMemsetNode)                    \
  __macro(musaGraphAddMemFreeNode)                   \
  __macro(musaGraphCreate)                           \
  __macro(musaGraphDebugDotPrint)                    \
  __macro(musaGraphDestroy)                          \
  __macro(musaGraphGetNodes)                         \
  __macro(musaGraphExecChildGraphNodeSetParams)      \
  __macro(musaGraphExecDestroy)                      \
  __macro(musaGraphExecKernelNodeSetParams)          \
  __macro(musaGraphExecMemcpyNodeSetParams1D)        \
  __macro(musaGraphExecMemsetNodeSetParams)          \
  __macro(musaGraphExecUpdate)                       \
  __macro(musaGraphInstantiate)                      \
  __macro(musaGraphMemAllocNodeGetParams)            \
  __macro(musaGraphLaunch)                           \
  __macro(musaGraphNodeGetType)                      \
  __macro(musaGraphNodeSetEnabled)                   \
  __macro(musaFreeHost)                              \
  __macro(musaHostAlloc)                            \
  __macro(musaHostRegister)                          \
  __macro(musaHostUnregister)                        \
  __macro(muInit)                                  \
  __macro(musaKernelNameRefByPtr)                    \
  __macro(musaLaunchHostFunc)                        \
  __macro(muLaunchKernel)                          \
  __macro(musaMalloc)                                \
  __macro(musaMallocManaged)                         \
  __macro(muMemGetAddressRange)                    \
  __macro(musaMemGetInfo)                            \
  __macro(musaMemcpyDtoD)                            \
  __macro(muMemcpyDtoDAsync)                       \
  __macro(musaMemcpyDtoH)                            \
  __macro(muMemcpyDtoHAsync)                       \
  __macro(musaMemcpyHtoD)                            \
  __macro(muMemcpyHtoDAsync)                       \
  __macro(musaMemset)                                \
  __macro(musaMemsetD8)                              \
  __macro(musaMemsetD16)                             \
  __macro(musaMemsetD32)                             \
  __macro(musaMemsetAsync)                           \
  __macro(muMemsetD8Async)                         \
  __macro(muMemsetD16Async)                        \
  __macro(muMemsetD32Async)                        \
  __macro(muModuleGetFunction)                     \
  __macro(muModuleGetGlobal)                       \
  __macro(musaModuleLaunchKernel)                    \
  __macro(muModuleLoadData)                        \
  __macro(muModuleUnload)                          \
  __macro(musaOccupancyMaxActiveBlocksPerMultiprocessor) \
  __macro(musaOccupancyMaxPotentialBlockSize)  \
  __macro(muPointerGetAttribute)                   \
  __macro(musaPointerGetAttributes)                  \
  __macro(musaRuntimeGetVersion)                     \
  __macro(musaSetDevice)                             \
  __macro(musaStreamAddCallback)                     \
  __macro(musaStreamBeginCapture)                    \
  __macro(musaStreamCreateWithFlags)                 \
  __macro(musaStreamCreateWithPriority)              \
  __macro(musaStreamDestroy)                         \
  __macro(musaStreamEndCapture)                      \
  __macro(musaStreamIsCapturing)                     \
  __macro(musaStreamQuery)                           \
  __macro(musaStreamSynchronize)                     \
  __macro(musaStreamWaitEvent)  // clang-format on

MUSA_ROUTINE_EACH(STREAM_EXECUTOR_MUSA_WRAP)

#undef MUSA_ROUTINE_EACH
#undef STREAM_EXECUTOR_MUSA_WRAP
*/
}  // namespace wrap
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_DRIVER_WRAPPER_H_
