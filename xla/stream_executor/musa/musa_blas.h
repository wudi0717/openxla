/* Copyright 2015 The OpenXLA Authors.

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

// MUSA-specific support for BLAS functionality -- this wraps the muBlas
// library capabilities, and is only included into MUSA implementation code --
// it will not introduce musa headers into other code.

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_BLAS_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_BLAS_H_
#include "musa_fp16.h"
#include "mublas.h"
#include "musa_bf16.h"
#include <muComplex.h>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

#define MUBLAS_BETA_FEATURES_API
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/plugin_registry.h"
#if TF_MUBLASLT
#include "xla/stream_executor/musa/musa_blas_lt.h"
#endif
#include "xla/stream_executor/stream_executor.h"

#if 1
// struct __half;
typedef struct
{
    uint16_t data;
} mublas_bfloat16;
typedef struct
{
    uint16_t data;
} __half;
MUBLAS_EXPORT mublasStatus mublasHgemm(mublasHandle_t    handle,
                                       mublasOperation_t transA,
                                       mublasOperation_t transB,
                                       int               m,
                                       int               n,
                                       int               k,
                                       const __half*     alpha,
                                       const __half*     A,
                                       int               lda,
                                       const __half*     B,
                                       int               ldb,
                                       const __half*     beta,
                                       __half*           C,
                                       int               ldc);
MUBLAS_EXPORT mublasStatus mublasHgemmBatched(mublasHandle_t      handle,
                                              mublasOperation_t   transA,
                                              mublasOperation_t   transB,
                                              int                 m,
                                              int                 n,
                                              int                 k,
                                              const __half*       alpha,
                                              const __half* const A[],
                                              int                 lda,
                                              const __half* const B[],
                                              int                 ldb,
                                              const __half*       beta,
                                              __half* const       C[],
                                              int                 ldc,
                                              int                 batch_count);
MUBLAS_EXPORT mublasStatus mublasHgemmStridedBatched(mublasHandle_t    handle,
                                                     mublasOperation_t transA,
                                                     mublasOperation_t transB,
                                                     int               m,
                                                     int               n,
                                                     int               k,
                                                     const __half*     alpha,
                                                     const __half*     A,
                                                     int               lda,
                                                     long long int     stride_a,
                                                     const __half*     B,
                                                     int               ldb,
                                                     long long int     stride_b,
                                                     const __half*     beta,
                                                     __half*           C,
                                                     int               ldc,
                                                     long long int     stride_c,
                                                     int               batch_count);
#endif

typedef long long int mublasStride;

namespace stream_executor {

class Stream;

namespace gpu {

template <bool ErrorIfMissing, class Target, class A, class B, class... T>
struct ChooseType {
  using type = std::conditional_t<
      std::is_same_v<Target, A>, B,
      typename ChooseType<ErrorIfMissing, Target, T...>::type>;
};

template <class Target, class A, class B>
struct ChooseType<false, Target, A, B> {
  // default case: return the same type Target if there is no recursive match
  using type = std::conditional_t<std::is_same_v<Target, A>, B, Target>;
};

template <class Target, class A, class B>
struct ChooseType<true, Target, A, B> {
  // default case: return compile error if type is not found
  static_assert(std::is_same_v<Target, A>,
                "ChooseType: the target type is not found!");
  using type = B;
};

// Type conversion helper that helps to map non-mublas types to mublas types
template <typename T>
using MuBlasType_t =
    typename ChooseType<false, T, Eigen::half, __half, Eigen::bfloat16,
                        __mt_bfloat16, std::complex<float>,
                        muComplex, std::complex<double>,
                        muDoubleComplex>::type;

// BLAS plugin for MUSA platform via muBlas library.
//
// This satisfies the platform-agnostic BlasSupport interface.
//
// Note that the muBlas handle that this encapsulates is implicitly tied to the
// context (and, as a result, the device) that the parent StreamExecutor is tied
// to. This simply happens as an artifact of creating the muBlas handle when a
// MUSA context is active.
//
// Thread-safe post-initialization.
class MUSABlas : public blas::BlasSupport {
 public:
  explicit MUSABlas(StreamExecutor *parent);

  // Allocates a muBlas handle.
  bool Init();

  // Releases the muBlas handle, if present.
  ~MUSABlas() override;

  TENSORFLOW_STREAM_EXECUTOR_GPU_BLAS_SUPPORT_OVERRIDES

  gpu::BlasLt *GetBlasLt() override {
#if TF_MUBLASLT
    return &blas_lt_;
#else
    return nullptr;
#endif
  }

 private:
  // Tells muBlas to enqueue the BLAS operation onto a particular Stream.
  //
  // muBlas is stateful, and only be associated with one stream (in order to
  // enqueue dispatch) at a given time. As a result, this generally must be
  // invoked before calling into muBlas.
  bool SetStream(Stream *stream) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // A helper function that calls the real muBlas function together with error
  // handling.
  //
  // mublas_func:       muBlas function pointer.
  // mublas_name:       muBlas function name.
  // stream:             Stream to enqueue the BLAS operation onto.
  // pointer_mode_host:  Indicate if the pointer to a scalar value is from host
  //                     (true) or device (false).
  // err_on_failure:     Whether to print an error if the muBlas function
  // fails. args:               Arguments of muBlas function.
  template <typename FuncT, typename... Args>
  absl::Status DoBlasInternalImpl(FuncT mublas_func, Stream *stream,
                                  bool pointer_mode_host, bool err_on_failure,
                                  Args &&...args);

  // Convenience functions that call DoBlasInternalImpl with different values
  // for err_on_failure.
  template <typename FuncT, typename... Args>
  bool DoBlasInternal(FuncT mublas_func, Stream *stream,
                      bool pointer_mode_host, Args &&...args) {
    auto ret = DoBlasInternalImpl(mublas_func, stream, pointer_mode_host,
                                  /*err_on_failure=*/true,
                                  std::forward<Args>(args)...);
    return ret.ok();
  }

  // Same as above, but returns absl::Status.
  template <typename FuncT, typename... Args>
  absl::Status DoBlasInternalStatus(FuncT mublas_func, Stream *stream,
                                    bool pointer_mode_host, Args &&...args) {
    return DoBlasInternalImpl(mublas_func, stream, pointer_mode_host,
                              /*err_on_failure=*/true,
                              std::forward<Args>(args)...);
  }

  template <typename FuncT, typename... Args>
  bool DoBlasInternalFailureOK(FuncT mublas_func, Stream *stream,
                               bool pointer_mode_host, Args &&...args) {
    auto ret = DoBlasInternalImpl(mublas_func, stream, pointer_mode_host,
                                  /*err_on_failure=*/false,
                                  std::forward<Args>(args)...);
    return ret.ok();
  }

  // A helper function to implement DoBlasGemmBatched interfaces for generic
  // types.
  //
  // Note: This function is implemented using gemm_strided_batched interface,
  // NOT gemm_batched interface, because mublas do not support it. As a
  // result, if the passed in batch matrix are not allocated in strided batched
  // format, it might end up in non-trivial amount of memory allocation and
  // copy. To avoid this, always prioritize to use DoBlasGemmStridedBatched
  // interface.
  //
  // In most use cases, batch matrix do get allocated in strided manner, making
  // calling this interface equivalent with DoBlasGemmStridedBatched. The only
  // use case we see so far that violates this observation is when batch
  // matrix is created by broadcasting from a smaller matrix. When it happens,
  // It will take advantage of the AllocateStridedBuffer subroutine to
  // reallocate the memory layout to be strided batched.
  template <typename T, typename FuncT>
  absl::Status DoBlasGemmBatchedInternal(
      FuncT mublas_func, Stream *stream, blas::Transpose transa,
      blas::Transpose transb, uint64_t m, uint64_t n, uint64_t k, T alpha,
      DeviceMemorySlice<T> a_ptrs_to_wrappers, int lda,
      DeviceMemorySlice<T> b_ptrs_to_wrappers, int ldb, T beta,
      DeviceMemorySlice<T> c_ptrs_to_wrappers, int ldc, int batch_count,
      ScratchAllocator *scratch_allocator);

  // mutex that guards the muBlas handle for this device.
  mutable absl::Mutex mu_;

  // StreamExecutor which instantiated this MUSABlas.
  // Immutable post-initialization.
  StreamExecutor *parent_;

  // muBlas library handle on the device.
  mublasHandle_t blas_ ABSL_GUARDED_BY(mu_);

  // container holding solutions vector (to avoid reallocating it each time)
  std::vector<int> solutions_;

  void MaybeLogGemmOp(StreamExecutor::GemmCallTrace::GemmType op,
                      blas::CallContext context, uint64_t size1,
                      uint64_t size2);

#if TF_MUBLASLT
  musa::BlasLt blas_lt_;
#endif

  MUSABlas(const MUSABlas &) = delete;
  void operator=(const MUSABlas &) = delete;

  bool has_mfma_ = false;
  bool use_hgemm_alt_impl_ = false;
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_BLAS_H_
