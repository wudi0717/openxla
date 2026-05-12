// TensorFlow 2.15 wheel on this setup does not export the
// xla::PjrtClientFactoryRegistry C++ symbol that the original bridge used.
// Keep this bridge as a loadable compatibility stub so the Python runner can
// continue with the pluggable-device + PJRT path.

#include <cstdio>

extern "C" {

__attribute__((visibility("default"))) int MusaTf215_RegisterRuntimeFactory() {
  std::fprintf(
      stderr,
      "[musa_tf215_bridge] compatibility stub active; relying on "
      "pluggable-device + PJRT registration.\n");
  std::fflush(stderr);
  return 1;
}

}  // extern "C"

__attribute__((constructor)) static void MusaTf215BridgeCtor() {
  std::fprintf(stderr,
               "[musa_tf215_bridge] compatibility stub loaded.\n");
  std::fflush(stderr);
}
