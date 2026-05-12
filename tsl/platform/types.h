#ifndef TSL_PLATFORM_TYPES_H_
#define TSL_PLATFORM_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <string>

namespace tsl {

using std::string;

using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;
using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;

}  // namespace tsl

namespace stream_executor {}
namespace tensorflow {
namespace se = ::stream_executor;
}  // namespace tensorflow

#if defined(PLATFORM_WINDOWS)
typedef std::ptrdiff_t ssize_t;
#endif

#endif  // TSL_PLATFORM_TYPES_H_
