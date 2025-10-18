/* Copyright 2020 The OpenXLA Authors.

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

// This file wraps mublas API calls with dso loader so that we don't need to
// have explicit linking to libmublas.

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUBLAS_WRAPPER_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUBLAS_WRAPPER_H_

// needed for mublas_gemm_ex_get_solutions* functionality
#define MUBLAS_BETA_FEATURES_API

#define __MUSACC__  1
#include "mublas.h"
#include "musa_runtime.h"
#include "internal/mublas_functions.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/dso_loader.h"
#include "tsl/platform/platform.h"

namespace stream_executor {
namespace wrap {

#if 1 //def PLATFORM_GOOGLE
#define MUBLAS_API_WRAPPER(__name)               \
  struct WrapperShim__##__name {                  \
    constexpr static const char* kName = #__name; \
    template <typename... Args>                   \
    mublasStatus operator()(Args... args) {     \
      return (::__name)(args...);                 \
    }                                             \
  } __name;

#else
using tsl::internal::CachedDsoLoader::GetMublasDsoHandle;

#define MUBLAS_API_WRAPPER(__name)                                      \
  static struct DynLoadShim__##__name {                                  \
    constexpr static const char* kName = #__name;                        \
    using FuncPtrT = std::add_pointer<decltype(::__name)>::type;         \
    static void* GetDsoHandle() {                                        \
      auto s = GetMublasDsoHandle();                                    \
      return s.value();                                                  \
    }                                                                    \
    static FuncPtrT LoadOrDie() {                                        \
      void* f;                                                           \
      auto s = tsl::Env::Default()->GetSymbolFromLibrary(GetDsoHandle(), \
                                                         kName, &f);     \
      CHECK(s.ok()) << "could not find " << kName                        \
                    << " in mublas DSO; dlerror: " << s.message();      \
      return reinterpret_cast<FuncPtrT>(f);                              \
    }                                                                    \
    static FuncPtrT DynLoad() {                                          \
      static FuncPtrT f = LoadOrDie();                                   \
      return f;                                                          \
    }                                                                    \
    template <typename... Args>                                          \
    auto operator()(Args... args) {                                      \
      return DynLoad()(args...);                                         \
    }                                                                    \
  } __name;

#endif

// clang-format off
#define FOREACH_MUBLAS_API(__macro)            \
  __macro(mublasSnrm2)                        \
  __macro(mublasDnrm2)                        \
  __macro(mublasScnrm2)                       \
  __macro(mublasDznrm2)                       \
  __macro(mublasSdot)                         \
  __macro(mublasDdot)                         \
  __macro(mublasCdotu)                        \
  __macro(mublasCdotc)                        \
  __macro(mublasZdotu)                        \
  __macro(mublasZdotc)                        \
  __macro(mublasSscal)                        \
  __macro(mublasDscal)                        \
  __macro(mublasCscal)                        \
  __macro(mublasCsscal)                       \
  __macro(mublasZscal)                        \
  __macro(mublasZdscal)                       \
  __macro(mublasSaxpy)                        \
  __macro(mublasDaxpy)                        \
  __macro(mublasCaxpy)                        \
  __macro(mublasZaxpy)                        \
  __macro(mublasScopy)                        \
  __macro(mublasDcopy)                        \
  __macro(mublasCcopy)                        \
  __macro(mublasZcopy)                        \
  __macro(mublasSswap)                        \
  __macro(mublasDswap)                        \
  __macro(mublasCswap)                        \
  __macro(mublasZswap)                        \
  __macro(mublasIsamax)                       \
  __macro(mublasIdamax)                       \
  __macro(mublasIcamax)                       \
  __macro(mublasIzamax)                       \
  __macro(mublasIsamin)                       \
  __macro(mublasIdamin)                       \
  __macro(mublasIcamin)                       \
  __macro(mublasIzamin)                       \
  __macro(mublasSasum)                        \
  __macro(mublasDasum)                        \
  __macro(mublasScasum)                       \
  __macro(mublasDzasum)                       \
  __macro(mublasSrot)                         \
  __macro(mublasDrot)                         \
  __macro(mublasCrot)                         \
  __macro(mublasCsrot)                        \
  __macro(mublasZrot)                         \
  __macro(mublasZdrot)                        \
  __macro(mublasSrotg)                        \
  __macro(mublasDrotg)                        \
  __macro(mublasCrotg)                        \
  __macro(mublasZrotg)                        \
  __macro(mublasSrotm)                        \
  __macro(mublasDrotm)                        \
  __macro(mublasSrotmg)                       \
  __macro(mublasDrotmg)                       \
  __macro(mublasSgemv)                        \
  __macro(mublasDgemv)                        \
  __macro(mublasCgemv)                        \
  __macro(mublasZgemv)                        \
  __macro(mublasSgbmv)                        \
  __macro(mublasDgbmv)                        \
  __macro(mublasCgbmv)                        \
  __macro(mublasZgbmv)                        \
  __macro(mublasStrmv)                        \
  __macro(mublasDtrmv)                        \
  __macro(mublasCtrmv)                        \
  __macro(mublasZtrmv)                        \
  __macro(mublasStbmv)                        \
  __macro(mublasDtbmv)                        \
  __macro(mublasCtbmv)                        \
  __macro(mublasZtbmv)                        \
  __macro(mublasStpmv)                        \
  __macro(mublasDtpmv)                        \
  __macro(mublasCtpmv)                        \
  __macro(mublasZtpmv)                        \
  __macro(mublasStrsv)                        \
  __macro(mublasDtrsv)                        \
  __macro(mublasCtrsv)                        \
  __macro(mublasZtrsv)                        \
  __macro(mublasStpsv)                        \
  __macro(mublasDtpsv)                        \
  __macro(mublasCtpsv)                        \
  __macro(mublasZtpsv)                        \
  __macro(mublasStbsv)                        \
  __macro(mublasDtbsv)                        \
  __macro(mublasCtbsv)                        \
  __macro(mublasZtbsv)                        \
  __macro(mublasSsymv)                        \
  __macro(mublasDsymv)                        \
  __macro(mublasCsymv)                        \
  __macro(mublasZsymv)                        \
  __macro(mublasChemv)                        \
  __macro(mublasZhemv)                        \
  __macro(mublasSsbmv)                        \
  __macro(mublasDsbmv)                        \
  __macro(mublasChbmv)                        \
  __macro(mublasZhbmv)                        \
  __macro(mublasSspmv)                        \
  __macro(mublasDspmv)                        \
  __macro(mublasChpmv)                        \
  __macro(mublasZhpmv)                        \
  __macro(mublasSger)                         \
  __macro(mublasDger)                         \
  __macro(mublasCgeru)                        \
  __macro(mublasCgerc)                        \
  __macro(mublasZgeru)                        \
  __macro(mublasZgerc)                        \
  __macro(mublasSsyr)                         \
  __macro(mublasDsyr)                         \
  __macro(mublasCsyr)                         \
  __macro(mublasZsyr)                         \
  __macro(mublasCher)                         \
  __macro(mublasZher)                         \
  __macro(mublasSspr)                         \
  __macro(mublasDspr)                         \
  __macro(mublasChpr)                         \
  __macro(mublasZhpr)                         \
  __macro(mublasSsyr2)                        \
  __macro(mublasDsyr2)                        \
  __macro(mublasCsyr2)                        \
  __macro(mublasZsyr2)                        \
  __macro(mublasCher2)                        \
  __macro(mublasZher2)                        \
  __macro(mublasSspr2)                        \
  __macro(mublasDspr2)                        \
  __macro(mublasChpr2)                        \
  __macro(mublasZhpr2)                        \
  __macro(mublasSgemm)                        \
  __macro(mublasDgemm)                        \
  __macro(mublasHgemm)                        \
  __macro(mublasCgemm)                        \
  __macro(mublasZgemm)                        \
  __macro(mublasSsyrk)                        \
  __macro(mublasDsyrk)                        \
  __macro(mublasCsyrk)                        \
  __macro(mublasZsyrk)                        \
  __macro(mublasCherk)                        \
  __macro(mublasZherk)                        \
  __macro(mublasSsyr2k)                       \
  __macro(mublasDsyr2k)                       \
  __macro(mublasCsyr2k)                       \
  __macro(mublasZsyr2k)                       \
  __macro(mublasCher2k)                       \
  __macro(mublasZher2k)                       \
  __macro(mublasSsyrkx)                       \
  __macro(mublasDsyrkx)                       \
  __macro(mublasCsyrkx)                       \
  __macro(mublasZsyrkx)                       \
  __macro(mublasCherkx)                       \
  __macro(mublasZherkx)                       \
  __macro(mublasSsymm)                        \
  __macro(mublasDsymm)                        \
  __macro(mublasCsymm)                        \
  __macro(mublasZsymm)                        \
  __macro(mublasChemm)                        \
  __macro(mublasZhemm)                        \
  __macro(mublasStrsm)                        \
  __macro(mublasDtrsm)                        \
  __macro(mublasCtrsm)                        \
  __macro(mublasZtrsm)                        \
  __macro(mublasStrmm)                        \
  __macro(mublasDtrmm)                        \
  __macro(mublasCtrmm)                        \
  __macro(mublasZtrmm)                        \
  __macro(mublasSgeam)                        \
  __macro(mublasDgeam)                        \
  __macro(mublasCgeam)                        \
  __macro(mublasZgeam)                        \
  __macro(mublasSdgmm)                        \
  __macro(mublasDdgmm)                        \
  __macro(mublasCdgmm)                        \
  __macro(mublasZdgmm)                        \
  __macro(mublasSgemmBatched)                \
  __macro(mublasDgemmBatched)                \
  __macro(mublasCgemmBatched)                \
  __macro(mublasZgemmBatched)                \
  __macro(mublasHgemmStridedBatched)        \
  __macro(mublasSgemmStridedBatched)        \
  __macro(mublasDgemmStridedBatched)        \
  __macro(mublasCgemmStridedBatched)        \
  __macro(mublasGemmEx)                      \
  __macro(mublasCreate)                         \
  __macro(mublasDestroy)                        \
  __macro(mublasSetStream)                            \
  __macro(mublasGetStream)                            \
  __macro(mublasSetAtomicsMode)                      \
  __macro(mublasGemmStridedBatchedEx)      \
  __macro(mublasStrsmBatched)                         \
  __macro(mublasDtrsmBatched)                         \
  __macro(mublasCtrsmBatched)                         \
  __macro(mublasZtrsmBatched)                         \
  __macro(mublasZgemmStridedBatched)/*        \
  __macro(mublas_gemm_ex_get_solutions)                 \
  __macro(mublas_gemm_ex_get_solutions_by_type)         \
  __macro(mublas_gemmBatched_ex_get_solutions)         \
  __macro(mublas_gemmBatched_ex_get_solutions_by_type) \
  __macro(mublas_gemmStridedBatched_ex_get_solutions) \
  __macro(mublasIs_managing_device_memory)             \
  __macro(mublasSet_workspace)                         \
  __macro(mublasStrsmBatched)                         \
  __macro(mublasDtrsmBatched)                         \
  __macro(mublasCtrsmBatched)                         \
  __macro(mublasZtrsmBatched)                         \
  __macro(mublas_get_version_string_size)               \
  __macro(mublasGetVersionString)*/

// clang-format on

FOREACH_MUBLAS_API(MUBLAS_API_WRAPPER)

}  // namespace wrap
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUBLAS_WRAPPER_H_
