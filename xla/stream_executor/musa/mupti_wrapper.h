/* Copyright 2021 The OpenXLA Authors.

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

// This file wraps mupti API calls with dso loader so that we don't need to
// have explicit linking to libmupti. All TF hipsarse API usage should route
// through this wrapper.

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUPTI_WRAPPER_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUPTI_WRAPPER_H_

#include "mupti.h"
#include "musa_runtime.h"
#include "mupti_callbacks.h"  // 包含 Subscribe/Enable 等定义
// #include "rocm/include/mupti/mupti.h"
// #include "rocm/include/mupti/mupti_hip.h"
// #include "rocm/rocm_config.h"
// #if TF_MUSA_VERSION >= 50300
// #include "rocm/include/mupti/mupti_roctx.h"
// #else
// #include "rocm/include/mupti/mupti_hcc.h"
// #endif
#include "xla/tsl/platform/env.h"
#include "tsl/platform/dso_loader.h"
#include "tsl/platform/platform.h"

namespace stream_executor {
namespace wrap {

#if 1 //def PLATFORM_GOOGLE
#define MUPTI_API_WRAPPER(API_NAME)                            \
  template <typename... Args>                                      \
  auto API_NAME(Args... args) -> decltype((::API_NAME)(args...)) { \
    return (::API_NAME)(args...);                                  \
  }

#else

#define MUPTI_API_WRAPPER(API_NAME)                                    \
  template <typename... Args>                                              \
  auto API_NAME(Args... args) -> decltype(::API_NAME(args...)) {           \
    using FuncPtrT = std::add_pointer<decltype(::API_NAME)>::type;         \
    static FuncPtrT loaded = []() -> FuncPtrT {                            \
      static const char* kName = #API_NAME;                                \
      void* f;                                                             \
      auto s = tsl::Env::Default()->GetSymbolFromLibrary(                  \
          tsl::internal::CachedDsoLoader::GetRoctracerDsoHandle().value(), \
          kName, &f);                                                      \
      CHECK(s.ok()) << "could not find " << kName                          \
                    << " in mupti DSO; dlerror: " << s.message();      \
      return reinterpret_cast<FuncPtrT>(f);                                \
    }();                                                                   \
    return loaded(args...);                                                \
  }

#endif  // PLATFORM_GOOGLE

#define FOREACH_MUPTI_API(DO_FUNC)    \
  DO_FUNC(muptiSubscribe)             \
  DO_FUNC(muptiUnsubscribe)           \
  DO_FUNC(muptiEnableDomain)          \
  DO_FUNC(muptiEnableCallback)        \
  DO_FUNC(muptiGetResultString)       \
  DO_FUNC(muptiGetTimestamp)

FOREACH_MUPTI_API(MUPTI_API_WRAPPER)

}  // namespace wrap
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUPTI_WRAPPER_H_
