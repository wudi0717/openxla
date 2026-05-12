/* Copyright 2023 The OpenXLA Authors.

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

// #define EIGEN_USE_HIP_BF16 1
#define EIGEN_HAS_HIP_BF16 1
// #define hip_bfloat16 __mt_bfloat16 
#include <cstdint>
#include <musa_bf16.h>

#include "xla/types.h"
#include "xla/stream_executor/musa/topk_kernel_musa_common.cu.h"

namespace stream_executor::musa {

using xla::bfloat16;

#define KERNEL_TRAIT(K_VAL, TYPE, VT) \
  stream_executor::gpu::TopKKernel<K_VAL, TYPE, VT>
#define REGISTER_TOPK_KERNEL_BF16(K_VAL, TYPE, VT)                                 \
  GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(                             \
      TopKKernelMusa_K##K_VAL##_bfloat16_##VT, KERNEL_TRAIT(K_VAL, bfloat16, VT), \
      stream_executor::musa::kMUSaPlatformId, ([](size_t arity) {             \
        return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(  \
            absl::bit_cast<void*>(&Run<K_VAL, TYPE, VT>),                     \
            "topk_k" #K_VAL "_bfloat16_" #VT, arity);                        \
      }));

REGISTER_TOPK_KERNEL_BF16(1, __mt_bfloat16, uint16_t);
REGISTER_TOPK_KERNEL_BF16(2, __mt_bfloat16, uint16_t);
REGISTER_TOPK_KERNEL_BF16(4, __mt_bfloat16, uint16_t);
REGISTER_TOPK_KERNEL_BF16(8, __mt_bfloat16, uint16_t);
REGISTER_TOPK_KERNEL_BF16(16, __mt_bfloat16, uint16_t);

REGISTER_TOPK_KERNEL_BF16(1, __mt_bfloat16, uint32_t);
REGISTER_TOPK_KERNEL_BF16(2, __mt_bfloat16, uint32_t);
REGISTER_TOPK_KERNEL_BF16(4, __mt_bfloat16, uint32_t);
REGISTER_TOPK_KERNEL_BF16(8, __mt_bfloat16, uint32_t);
REGISTER_TOPK_KERNEL_BF16(16, __mt_bfloat16, uint32_t);

}  // namespace stream_executor::musa
