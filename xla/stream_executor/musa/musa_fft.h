  #ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_FFT_H_
  #define XLA_STREAM_EXECUTOR_MUSA_MUSA_FFT_H_

  #include <cstddef>
  #include <cstdint>

  #include "absl/log/log.h"
  #include "absl/status/status.h"
  #include "mufft.h"
  #include "xla/stream_executor/fft.h"
  #include "xla/stream_executor/scratch_allocator.h"
  #include "xla/stream_executor/stream.h"
  #include "xla/stream_executor/stream_executor.h"

  namespace stream_executor {
  namespace gpu {

  // MusaFftPlan uses deferred initialization. Only a single call of
  // Initialize() is allowed to properly create mufft plan and set member
  // variable is_initialized_ to true.
  class MusaFftPlan : public fft::Plan {
   public:
    MusaFftPlan()
        : parent_(nullptr),
          plan_(nullptr),
          fft_type_(fft::Type::kInvalid),
          scratch_(nullptr),
          scratch_size_bytes_(0),
          is_initialized_(false),
          scratch_allocator_(nullptr) {}
    ~MusaFftPlan() override;

    // Get FFT direction in muFFT based on FFT type.
    int GetFftDirection() const;
    mufftHandle GetPlan() const {
      if (IsInitialized()) {
        return plan_;
      } else {
        LOG(FATAL) << "Try to get mufftHandle value before initialization.";
      }
    }

    // Initialize function for batched plan
    absl::Status Initialize(StreamExecutor* parent, Stream* stream, int rank,
                            uint64_t* elem_count, uint64_t* input_embed,
                            uint64_t input_stride, uint64_t input_distance,
                            uint64_t* output_embed, uint64_t output_stride,
                            uint64_t output_distance, fft::Type type,
                            int batch_count, ScratchAllocator* scratch_allocator);

    absl::Status UpdateScratchAllocator(Stream* stream,
                                        ScratchAllocator* scratch_allocator);

    ScratchAllocator* GetScratchAllocator() const { return scratch_allocator_; }

   protected:
    bool IsInitialized() const { return is_initialized_; }

   private:
    StreamExecutor* parent_;
    mufftHandle plan_;
    fft::Type fft_type_;
    DeviceMemory<uint8_t> scratch_;
    size_t scratch_size_bytes_;
    bool is_initialized_;
    ScratchAllocator* scratch_allocator_;
  };

  // FFT support for MUSA platform via muFFT library.
  //
  // This satisfies the platform-agnostic FftSupport interface.
  //
  // Thread-safe. The MUSA context associated with all operations is the MUSA
  // context of parent_, so all context is explicit.
  class MusaFft : public fft::FftSupport {
   public:
    explicit MusaFft(StreamExecutor* parent) : parent_(parent) {}
    ~MusaFft() override {}

    TENSORFLOW_STREAM_EXECUTOR_GPU_FFT_SUPPORT_OVERRIDES

   private:
    StreamExecutor* parent_;

    // Two helper functions that execute mufftExec functions.

    // This is for complex to complex FFT, when the direction is required.
    template <typename FuncT, typename InputT, typename OutputT>
    bool DoFftWithDirectionInternal(Stream* stream, fft::Plan* plan,
                                    FuncT mufft_exec,
                                    const DeviceMemory<InputT>& input,
                                    DeviceMemory<OutputT>* output);

    // This is for complex to real or real to complex FFT, when the direction
    // is implied.
    template <typename FuncT, typename InputT, typename OutputT>
    bool DoFftInternal(Stream* stream, fft::Plan* plan, FuncT mufft_exec,
                       const DeviceMemory<InputT>& input,
                       DeviceMemory<OutputT>* output);

    MusaFft(const MusaFft&) = delete;
    void operator=(const MusaFft&) = delete;
  };

  }  // namespace gpu
  }  // namespace stream_executor

  #endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_FFT_H_