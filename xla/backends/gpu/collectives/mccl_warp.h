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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_RCCL_WARP_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_RCCL_WARP_H_

#if TENSORFLOW_USE_MUSA
#define MARCH_TYPE 220
#define NCCL_VERSION_CODE 21000
#include <climits>
#include <stdio.h>
#include "mccl.h"
#include "musa.h"
// #define ncclConfig_t mcclConfig_t
// #define NCCL_CONFIG_INITIALIZER MCCL_CONFIG_INITIALIZER
typedef struct ncclConfig_v22800 {
  /* attributes that users should never touch. */
  size_t size;
  unsigned int magic;
  unsigned int version;
  /* attributes that users are able to customize. */
  int blocking;
  int cgaClusterSize;
  int minCTAs;
  int maxCTAs;
  const char *netName;
  int splitShare;
  int trafficClass;
  const char *commName;
  int collnetEnable;
  int CTAPolicy;
  int shrinkShare;
  int nvlsCTAs;
  int nChannelsPerNetPeer;
  int nvlinkCentricSched;
} ncclConfig_t;
#define NCCL_CONFIG_UNDEF_INT INT_MIN
#define NCCL_CONFIG_UNDEF_PTR NULL
#define NCCL_CONFIG_INITIALIZER {                                       \
  sizeof(ncclConfig_t), /* size */                                      \
  0xcafebeef,           /* magic */                                     \
  NCCL_VERSION_CODE,    /* version */       \
  NCCL_CONFIG_UNDEF_INT,                    /* blocking */              \
  NCCL_CONFIG_UNDEF_INT,                    /* cgaClusterSize */        \
  NCCL_CONFIG_UNDEF_INT,                    /* minCTAs */               \
  NCCL_CONFIG_UNDEF_INT,                    /* maxCTAs */               \
  NCCL_CONFIG_UNDEF_PTR,                    /* netName */               \
  NCCL_CONFIG_UNDEF_INT,                    /* splitShare */            \
  NCCL_CONFIG_UNDEF_INT,                    /* trafficClass */          \
  NCCL_CONFIG_UNDEF_PTR,                    /* commName */              \
  NCCL_CONFIG_UNDEF_INT,                    /* collnetEnable */         \
  NCCL_CONFIG_UNDEF_INT,                    /* CTAPolicy */             \
  NCCL_CONFIG_UNDEF_INT,                    /* shrinkShare */           \
  NCCL_CONFIG_UNDEF_INT,                    /* nvlsCTAs */              \
  NCCL_CONFIG_UNDEF_INT,                    /* nChannelsPerNetPeer */   \
  NCCL_CONFIG_UNDEF_INT,                    /* nvlinkCentricSched */    \
}
#define ncclComm_t mcclComm_t
#define ncclUniqueId mcclUniqueId
#define ncclGetUniqueId mcclGetUniqueId
#define NCCL_UNIQUE_ID_BYTES MCCL_UNIQUE_ID_BYTES
// #define ncclCommSplit mcclCommSplit
#define ncclMemAlloc muMemAlloc
#define ncclMemFree muMemFree
#define ncclResult_t mcclResult_t
#define ncclSuccess mcclSuccess
#define ncclGetErrorString mcclGetErrorString
// #define ncclGetLastError mcclGetLastError
__inline__ const char* ncclGetLastError(ncclComm_t comm) {
  return "";
}
// #define ncclInProgress mcclSuccess // TODO(perfxlab), not impl
#ifdef ncclInProgress
#undef ncclInProgress
#endif

// 确保 != ncclSuccess（
#define ncclInProgress ((ncclResult_t)999)

#define ncclCommGetAsyncError mcclCommGetAsyncError
#define ncclInt8 mcclInt8
#define ncclUint8 mcclUint8
#define ncclInt32 mcclInt32
#define ncclUint32 mcclUint32
#define ncclInt64 mcclInt64
#define ncclUint64 mcclUint64
#define ncclFloat16 mcclFloat16
#define ncclFloat32 mcclFloat32
#define ncclFloat64 mcclFloat64
#define ncclBfloat16 mcclBfloat16
#define ncclRedOp_t mcclRedOp_t
#define ncclDataType_t mcclDataType_t
#define ncclCommCount mcclCommCount
#define ncclSum mcclSum
#define ncclProd mcclProd
#define ncclMin mcclMin
#define ncclMax mcclMax
#define ncclCommDestroy mcclCommDestroy
#define ncclCommAbort mcclCommAbort
// #define ncclGroupStart mcclGroupStart
// #define ncclGroupEnd mcclGroupEnd
#define ncclAllReduce mcclAllReduce
#define ncclBroadcast mcclBroadcast
#define ncclReduceScatter mcclReduceScatter
#define ncclAllGather mcclAllGather
// #define ncclSend mcclSend
// #define ncclRecv mcclRecv
// #define ncclCommInitRankConfig mcclCommInitRankConfig
static inline ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks,
                                                 ncclUniqueId commId, int rank,
                                                 ncclConfig_t* /*config*/) {
  fprintf(stderr, "[MCCL SHIM] nranks=%d rank=%d\n", nranks, rank);
  return mcclCommInitRank(comm, nranks, commId, rank);
}

static inline ncclResult_t ncclGroupStart_wrap() {
  fprintf(stderr, "[MCCL SHIM] GroupStart\n");
  return mcclGroupStart();
}
static inline ncclResult_t ncclGroupEnd_wrap() {
  fprintf(stderr, "[MCCL SHIM] GroupEnd\n");
  return mcclGroupEnd();
}
static inline ncclResult_t ncclSend_wrap(const void* sendbuff, size_t count,
                                         ncclDataType_t datatype, int peer,
                                         ncclComm_t comm, musaStream_t stream) {
  fprintf(stderr, "[MCCL SHIM] Send peer=%d count=%zu\n", peer, count);
  return mcclSend(sendbuff, count, datatype, peer, comm, stream);
}
static inline ncclResult_t ncclRecv_wrap(void* recvbuff, size_t count,
                                         ncclDataType_t datatype, int peer,
                                         ncclComm_t comm, musaStream_t stream) {
  fprintf(stderr, "[MCCL SHIM] Recv peer=%d count=%zu\n", peer, count);
  return mcclRecv(recvbuff, count, datatype, peer, comm, stream);
}

#define ncclGroupStart ncclGroupStart_wrap
#define ncclGroupEnd   ncclGroupEnd_wrap
#define ncclSend       ncclSend_wrap
#define ncclRecv       ncclRecv_wrap


#define ncclCommInitRank mcclCommInitRank
#endif

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_RCCL_WARP_H_
