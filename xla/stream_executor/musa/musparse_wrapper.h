#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSPARSE_WRAPPER_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSPARSE_WRAPPER_H_

#include "musa_config.h"


#include "musparse.h"

#include "xla/tsl/platform/env.h"
#include "tsl/platform/dso_loader.h"
#include "tsl/platform/platform.h"

namespace stream_executor {
namespace wrap {

#ifdef PLATFORM_GOOGLE

#define MUSPARSE_API_WRAPPER(__name)               \
  struct WrapperShim__##__name {                    \
    template <typename... Args>                     \
    musparseStatus_t operator()(Args... args) {    \
      musparseStatus_t retval = ::__name(args...); \
      return retval;                                \
    }                                               \
  } __name;

#else

#define MUSPARSE_API_WRAPPER(__name)                                    \
  static struct DynLoadShim__##__name {                                  \
    constexpr static const char* kName = #__name;                        \
    using FuncPtrT = std::add_pointer<decltype(::__name)>::type;         \
    static void* GetDsoHandle() {                                        \
      auto s = tsl::internal::CachedDsoLoader::GetMusparseDsoHandle();  \
      return s.value();                                                  \
    }                                                                    \
    static FuncPtrT LoadOrDie() {                                        \
      void* f;                                                           \
      auto s = tsl::Env::Default()->GetSymbolFromLibrary(GetDsoHandle(), \
                                                         kName, &f);     \
      CHECK(s.ok()) << "could not find " << kName                        \
                    << " in miopen DSO; dlerror: " << s.message();       \
      return reinterpret_cast<FuncPtrT>(f);                              \
    }                                                                    \
    static FuncPtrT DynLoad() {                                          \
      static FuncPtrT f = LoadOrDie();                                   \
      return f;                                                          \
    }                                                                    \
    template <typename... Args>                                          \
    musparseStatus_t operator()(Args... args) {                         \
      return DynLoad()(args...);                                         \
    }                                                                    \
  } __name;

#endif

// clang-format off
#define FOREACH_MUSPARSE_API(__macro)          \
  __macro(musparseCreate)                      \
  __macro(musparseCreateMatDescr)              \
  __macro(musparseCcsr2csc)                    \
  __macro(musparseCcsrgeam2)                   \
  __macro(musparseCcsrgeam2_bufferSizeExt)     \
  __macro(musparseCcsrgemm)                    \
  __macro(musparseCcsrmm)                      \
  __macro(musparseCcsrmm2)                     \
  __macro(musparseCcsrmv)                      \
  __macro(musparseDcsr2csc)                    \
  __macro(musparseDcsrgeam2)                   \
  __macro(musparseDcsrgeam2_bufferSizeExt)     \
  __macro(musparseDcsrgemm)                    \
  __macro(musparseDcsrmm)                      \
  __macro(musparseDcsrmm2)                     \
  __macro(musparseDcsrmv)                      \
  __macro(musparseDestroy)                     \
  __macro(musparseDestroyMatDescr)             \
  __macro(musparseScsr2csc)                    \
  __macro(musparseScsrgeam2)                   \
  __macro(musparseScsrgeam2_bufferSizeExt)     \
  __macro(musparseScsrgemm)                    \
  __macro(musparseScsrmm)                      \
  __macro(musparseScsrmm2)                     \
  __macro(musparseScsrmv)                      \
  __macro(musparseSetStream)                   \
  __macro(musparseSetMatIndexBase)             \
  __macro(musparseSetMatType)                  \
  __macro(musparseXcoo2csr)                    \
  __macro(musparseXcsr2coo)                    \
  __macro(musparseXcsrgeam2Nnz)                \
  __macro(musparseXcsrgemmNnz)                 \
  __macro(musparseZcsr2csc)                    \
  __macro(musparseZcsrgeam2)                   \
  __macro(musparseZcsrgeam2_bufferSizeExt)     \
  __macro(musparseZcsrgemm)                    \
  __macro(musparseZcsrmm)                      \
  __macro(musparseZcsrmm2)                     \
  __macro(musparseZcsrmv)

#if TF_MUSA_VERSION >= 40200
#define FOREACH_MUSPARSE_MUSA42_API(__macro)   \
  __macro(musparseCcsru2csr_bufferSizeExt)     \
  __macro(musparseCcsru2csr)                   \
  __macro(musparseCreateCsr)                   \
  __macro(musparseCreateDnMat)                 \
  __macro(musparseDestroyDnMat)                \
  __macro(musparseDestroySpMat)                \
  __macro(musparseDcsru2csr_bufferSizeExt)     \
  __macro(musparseDcsru2csr)                   \
  __macro(musparseScsru2csr_bufferSizeExt)     \
  __macro(musparseScsru2csr)                   \
  __macro(musparseSpMM_bufferSize)             \
  __macro(musparseSpMM)                        \
  __macro(musparseZcsru2csr_bufferSizeExt)     \
  __macro(musparseZcsru2csr)


FOREACH_MUSPARSE_MUSA42_API(MUSPARSE_API_WRAPPER)

#undef FOREACH_MUSPARSE_MUSA42_API
#endif

// clang-format on

FOREACH_MUSPARSE_API(MUSPARSE_API_WRAPPER)

#undef FOREACH_MUSPARSE_API
#undef MUSPARSE_API_WRAPPER

}  // namespace wrap
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSPARSE_WRAPPER_H_