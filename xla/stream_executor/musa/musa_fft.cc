
  #include "xla/stream_executor/musa/musa_fft.h"

  #include <array>
  #include <complex>
  #include <cstdint>
  #include <limits>
  #include <memory>
  #include <type_traits>
  #include <utility>

  #include "absl/base/casts.h"
  #include "absl/log/log.h"
  #include "absl/status/status.h"
  #include "absl/status/statusor.h"
  #include "absl/strings/str_cat.h"
  #include "musa_runtime.h"
  #include "mufft.h"
  #include "xla/stream_executor/activate_context.h"
  #include "xla/stream_executor/device_memory.h"
  #include "xla/stream_executor/fft.h"
  #include "xla/stream_executor/gpu/gpu_helpers.h"
  #include "xla/stream_executor/musa/musa_platform_id.h"
  #include "xla/stream_executor/musa/musa_complex_converters.h"
  #include "xla/stream_executor/platform/initialize.h"
  #include "xla/stream_executor/plugin_registry.h"
  #include "xla/stream_executor/scratch_allocator.h"
  #include "xla/stream_executor/stream.h"
  #include "xla/stream_executor/stream_executor.h"
  #include "xla/tsl/platform/statusor.h"

  namespace stream_executor {
  namespace gpu {

  namespace {

  // A helper function transforming gpu_fft arguments into muFFT arguments.
  mufftType MusaFftType(fft::Type type) {
    switch (type) {
      case fft::Type::kC2CForward:
      case fft::Type::kC2CInverse:
        return MUFFT_C2C;
      case fft::Type::kC2R:
        return MUFFT_C2R;
      case fft::Type::kR2C:
        return MUFFT_R2C;
      case fft::Type::kZ2ZForward:
      case fft::Type::kZ2ZInverse:
        return MUFFT_Z2Z;
      case fft::Type::kZ2D:
        return MUFFT_Z2D;
      case fft::Type::kD2Z:
        return MUFFT_D2Z;
      default:
        LOG(FATAL) << "Invalid value of fft::Type.";
    }
  }

  // Associates the given stream with the given muFFT plan.
  bool SetStream(StreamExecutor *parent, mufftHandle plan, Stream *stream) {
    std::unique_ptr<ActivateContext> activation = parent->Activate();
    auto ret = mufftSetStream(
        plan,
        absl::bit_cast<musaStream_t>((stream->platform_specific_handle().stream)));
    if (ret != MUFFT_SUCCESS) {
      LOG(ERROR) << "Failed to run muFFT routine mufftSetStream: " << ret;
      return false;
    }
    return true;
  }

  // Populates array of 32b integers from 64b integers, or an error if the
  // numbers don't fit in 32b (signed).
  absl::StatusOr<std::array<int32_t, 3>> Downsize64bArray(
      std::array<long long, 3> source, int32_t rank) {  // NOLINT
    std::array<int32_t, 3> downsized = {0};
    for (int32_t i = 0; i < rank; ++i) {
      if (source[i] > std::numeric_limits<int32_t>::max()) {
        return absl::InvalidArgumentError(absl::StrCat(
            source[i], " exceeds max 32b signed integer. Conversion failed."));
      }
      downsized[i] = static_cast<int32_t>(source[i]);
    }
    return downsized;
  }

  }  // namespace

  absl::Status MusaFftPlan::Initialize(
      StreamExecutor *parent, Stream *stream, int rank, uint64_t *elem_count,
      uint64_t *input_embed, uint64_t input_stride, uint64_t input_distance,
      uint64_t *output_embed, uint64_t output_stride, uint64_t output_distance,
      fft::Type type, int batch_count, ScratchAllocator *scratch_allocator) {
    if (IsInitialized()) {
      return absl::InternalError("muFFT is already initialized.");
    }
    is_initialized_ = true;
    scratch_allocator_ = scratch_allocator;
    std::unique_ptr<ActivateContext> activation = parent->Activate();
    // NOLINTBEGIN
    std::array<long long, 3> elem_count_ = {0};
    std::array<long long, 3> input_embed_ = {0};
    std::array<long long, 3> output_embed_ = {0};
    // NOLINTEND
    for (int32_t i = 0; i < rank; ++i) {
      elem_count_[i] = elem_count[i];
      if (input_embed) {
        input_embed_[i] = input_embed[i];
      }
      if (output_embed) {
        output_embed_[i] = output_embed[i];
      }
    }
    parent_ = parent;
    fft_type_ = type;
    if (batch_count == 1 && input_embed == nullptr && output_embed == nullptr) {
      mufftResult ret;
      if (scratch_allocator == nullptr) {
        switch (rank) {
          case 1:
            // mufftPlan1d
            ret = mufftPlan1d(&plan_, elem_count_[0], MusaFftType(type),
                              1 /* = batch */);
            if (ret != MUFFT_SUCCESS) {
              LOG(ERROR) << "Failed to create muFFT 1d plan: " << ret;
              return absl::InternalError("Failed to create muFFT 1d plan.");
            }
            return absl::OkStatus();
          case 2:
            // mufftPlan2d
            ret = mufftPlan2d(&plan_, elem_count_[0], elem_count_[1],
                              MusaFftType(type));
            if (ret != MUFFT_SUCCESS) {
              LOG(ERROR) << "Failed to create muFFT 2d plan: " << ret;
              return absl::InternalError("Failed to create muFFT 2d plan.");
            }
            return absl::OkStatus();
          case 3:
            // mufftPlan3d
            ret = mufftPlan3d(&plan_, elem_count_[0], elem_count_[1],
                              elem_count_[2], MusaFftType(type));
            if (ret != MUFFT_SUCCESS) {
              LOG(ERROR) << "Failed to create muFFT 3d plan: " << ret;
              return absl::InternalError("Failed to create muFFT 3d plan.");
            }
            return absl::OkStatus();
          default:
            LOG(ERROR) << "Invalid rank value for mufftPlan. "
                          "Requested 1, 2, or 3, given: "
                       << rank;
            return absl::InvalidArgumentError(
                "mufftPlan only takes rank 1, 2, or 3.");
        }
      } else {
        ret = mufftCreate(&plan_);
        if (ret != MUFFT_SUCCESS) {
          LOG(ERROR) << "Failed to create muFFT plan: " << ret;
          return absl::InternalError("Failed to create muFFT plan.");
        }
        ret = mufftSetAutoAllocation(plan_, 0);
        if (ret != MUFFT_SUCCESS) {
          LOG(ERROR) << "Failed to set auto allocation for muFFT plan: " << ret;
          return absl::InternalError(
              "Failed to set auto allocation for muFFT plan.");
        }
        switch (rank) {
          case 1:
            ret = mufftMakePlan1d(plan_, elem_count_[0], MusaFftType(type),
                                  /*batch=*/1, &scratch_size_bytes_);
            if (ret != MUFFT_SUCCESS) {
              LOG(ERROR) << "Failed to make muFFT 1d plan: " << ret;
              return absl::InternalError("Failed to make muFFT 1d plan.");
            }
            break;
          case 2:
            ret = mufftMakePlan2d(plan_, elem_count_[0], elem_count_[1],
                                  MusaFftType(type), &scratch_size_bytes_);
            if (ret != MUFFT_SUCCESS) {
              LOG(ERROR) << "Failed to make muFFT 2d plan: " << ret;
              return absl::InternalError("Failed to make muFFT 2d plan.");
            }
            break;
          case 3:
            ret = mufftMakePlan3d(plan_, elem_count_[0], elem_count_[1],
                                  elem_count_[2], MusaFftType(type),
                                  &scratch_size_bytes_);
            if (ret != MUFFT_SUCCESS) {
              LOG(ERROR) << "Failed to make muFFT 3d plan: " << ret;
              return absl::InternalError("Failed to make muFFT 3d plan.");
            }
            break;
          default:
            LOG(ERROR) << "Invalid rank value for mufftPlan. "
                          "Requested 1, 2, or 3, given: "
                       << rank;
            return absl::InvalidArgumentError(
                "mufftPlan only takes rank 1, 2, or 3.");
        }
        return UpdateScratchAllocator(stream, scratch_allocator);
      }
    } else {
      // For either multiple batches or rank higher than 3, use mufft*PlanMany*().
      if (scratch_allocator == nullptr) {
        // Downsize 64b arrays to 32b as there's no 64b version of mufftPlanMany
        TF_ASSIGN_OR_RETURN(auto elem_count_32b_,
                            Downsize64bArray(elem_count_, rank));
        TF_ASSIGN_OR_RETURN(auto input_embed_32b_,
                            Downsize64bArray(input_embed_, rank));
        TF_ASSIGN_OR_RETURN(auto output_embed_32b_,
                            Downsize64bArray(output_embed_, rank));
        auto ret = mufftPlanMany(
            &plan_, rank, elem_count_32b_.data(),
            input_embed ? input_embed_32b_.data() : nullptr, input_stride,
            input_distance, output_embed ? output_embed_32b_.data() : nullptr,
            output_stride, output_distance, MusaFftType(type), batch_count);
        if (ret != MUFFT_SUCCESS) {
          LOG(ERROR) << "Failed to create muFFT batched plan: " << ret;
          return absl::InternalError("Failed to create muFFT batched plan.");
        }
      } else {
        // Downsize 64b arrays to 32b as there's no 64b version of mufftMakePlanMany
        TF_ASSIGN_OR_RETURN(auto elem_count_32b_,
                            Downsize64bArray(elem_count_, rank));
        TF_ASSIGN_OR_RETURN(auto input_embed_32b_,
                            Downsize64bArray(input_embed_, rank));
        TF_ASSIGN_OR_RETURN(auto output_embed_32b_,
                            Downsize64bArray(output_embed_, rank));
        auto ret = mufftCreate(&plan_);
        if (ret != MUFFT_SUCCESS) {
          LOG(ERROR) << "Failed to create muFFT batched plan: " << ret;
          return absl::InternalError("Failed to create muFFT batched plan.");
        }
        ret = mufftSetAutoAllocation(plan_, 0);
        if (ret != MUFFT_SUCCESS) {
          LOG(ERROR) << "Failed to set auto allocation for muFFT batched plan: "
                     << ret;
          return absl::InternalError(
              "Failed to set auto allocation for muFFT batched plan.");
        }
        ret = mufftMakePlanMany(
            plan_, rank, elem_count_32b_.data(),
            input_embed ? input_embed_32b_.data() : nullptr, input_stride,
            input_distance, output_embed ? output_embed_32b_.data() : nullptr,
            output_stride, output_distance, MusaFftType(type), batch_count,
            &scratch_size_bytes_);
        if (ret != MUFFT_SUCCESS) {
          LOG(ERROR) << "Failed to make muFFT batched plan: " << ret;
          return absl::InternalError("Failed to make muFFT batched plan.");
        }
        return UpdateScratchAllocator(stream, scratch_allocator);
      }
    }
    return absl::OkStatus();
  }

  absl::Status MusaFftPlan::UpdateScratchAllocator(
      Stream *stream, ScratchAllocator *scratch_allocator) {
    scratch_allocator_ = scratch_allocator;

    if (scratch_size_bytes_ != 0) {
      auto allocated = scratch_allocator->AllocateBytes(scratch_size_bytes_);
      if (!allocated.ok() || (scratch_ = allocated.value()) == nullptr) {
        LOG(ERROR) << "Failed to allocate work area.";
        return allocated.status();
      }
    }
    // Connect work area with allocated space.
    std::unique_ptr<ActivateContext> activation = parent_->Activate();
    mufftResult ret = mufftSetWorkArea(plan_, scratch_.opaque());
    if (ret != MUFFT_SUCCESS) {
      LOG(ERROR) << "Failed to set work area for muFFT plan: " << ret;
      return absl::InternalError("Failed to set work area for muFFT plan.");
    }
    return absl::OkStatus();
  }

  MusaFftPlan::~MusaFftPlan() {
    std::unique_ptr<ActivateContext> activation = parent_->Activate();
    mufftDestroy(plan_);
  }

  int MusaFftPlan::GetFftDirection() const {
    if (!IsInitialized()) {
      LOG(FATAL) << "Try to get fft direction before initialization.";
    } else {
      switch (fft_type_) {
        case fft::Type::kC2CForward:
        case fft::Type::kZ2ZForward:
        case fft::Type::kR2C:
        case fft::Type::kD2Z:
          return MUFFT_FORWARD;
        case fft::Type::kC2CInverse:
        case fft::Type::kZ2ZInverse:
        case fft::Type::kC2R:
        case fft::Type::kZ2D:
          return MUFFT_INVERSE;
        default:
          LOG(FATAL) << "Invalid value of fft::Type.";
      }
    }
  }

  std::unique_ptr<fft::Plan> MusaFft::CreateBatchedPlanWithScratchAllocator(
      Stream *stream, int rank, uint64_t *elem_count, uint64_t *input_embed,
      uint64_t input_stride, uint64_t input_distance, uint64_t *output_embed,
      uint64_t output_stride, uint64_t output_distance, fft::Type type,
      bool in_place_fft, int batch_count, ScratchAllocator *scratch_allocator) {
    std::unique_ptr<MusaFftPlan> fft_plan_ptr{new MusaFftPlan()};
    absl::Status status = fft_plan_ptr->Initialize(
        parent_, stream, rank, elem_count, input_embed, input_stride,
        input_distance, output_embed, output_stride, output_distance, type,
        batch_count, scratch_allocator);
    if (!status.ok()) {
      LOG(ERROR) << "Initialize Params: rank: " << rank
                 << " elem_count: " << *elem_count
                 << " input_embed: " << *input_embed
                 << " input_stride: " << input_stride
                 << " input_distance: " << input_distance
                 << " output_embed: " << *output_embed
                 << " output_stride: " << output_stride
                 << " output_distance: " << output_distance
                 << " batch_count: " << batch_count;
      LOG(ERROR)
          << "Failed to initialize batched mufft plan with customized allocator: "
          << status.message();
      return nullptr;
    }
    return std::move(fft_plan_ptr);
  }

  void MusaFft::UpdatePlanWithScratchAllocator(
      Stream *stream, fft::Plan *plan, ScratchAllocator *scratch_allocator) {
    MusaFftPlan *musa_fft_plan = dynamic_cast<MusaFftPlan *>(plan);
    absl::Status status =
        musa_fft_plan->UpdateScratchAllocator(stream, scratch_allocator);
    if (!status.ok()) {
      LOG(FATAL) << "Failed to update custom allocator for mufft plan: "
                 << status.message();
    }
  }

  template <typename FuncT, typename InputT, typename OutputT>
  bool MusaFft::DoFftInternal(Stream *stream, fft::Plan *plan, FuncT mufftExec,
                              const DeviceMemory<InputT> &input,
                              DeviceMemory<OutputT> *output) {
    MusaFftPlan *musa_fft_plan = dynamic_cast<MusaFftPlan *>(plan);

    if (musa_fft_plan == nullptr) {
      LOG(ERROR) << "The passed-in plan is not a MusaFftPlan object.";
      return false;
    }

    if (!SetStream(parent_, musa_fft_plan->GetPlan(), stream)) {
      return false;
    }

    std::unique_ptr<ActivateContext> activation = parent_->Activate();
    auto ret =
      mufftExec(musa_fft_plan->GetPlan(),
                musa::MUSAComplex(const_cast<InputT *>(GpuMemory(input))),
                musa::MUSAComplex(GpuMemoryMutable(output)));

    if (ret != MUFFT_SUCCESS) {
      LOG(ERROR) << "Failed to run muFFT routine: " << ret;
      return false;
    }

    return true;
  }

  template <typename FuncT, typename InputT, typename OutputT>
  bool MusaFft::DoFftWithDirectionInternal(Stream *stream, fft::Plan *plan,
                                           FuncT mufftExec,
                                           const DeviceMemory<InputT> &input,
                                           DeviceMemory<OutputT> *output) {
    MusaFftPlan *musa_fft_plan = dynamic_cast<MusaFftPlan *>(plan);
    if (musa_fft_plan == nullptr) {
      LOG(ERROR) << "The passed-in plan is not a MusaFftPlan object.";
      return false;
    }

    if (!SetStream(parent_, musa_fft_plan->GetPlan(), stream)) {
      return false;
    }

    std::unique_ptr<ActivateContext> activation = parent_->Activate();
    auto ret = mufftExec(musa_fft_plan->GetPlan(),
                        musa::MUSAComplex(const_cast<InputT *>(GpuMemory(input))),
                        musa::MUSAComplex(GpuMemoryMutable(output)),
                        musa_fft_plan->GetFftDirection());

    if (ret != MUFFT_SUCCESS) {
      LOG(ERROR) << "Failed to run muFFT routine: " << ret;
      return false;
    }

    return true;
  }

  #define STREAM_EXECUTOR_MUSA_DEFINE_FFT(__type, __fft_type1, __fft_type2,      \
                                          __fft_type3)                           \
    bool MusaFft::DoFft(Stream *stream, fft::Plan *plan,                         \
                        const DeviceMemory<std::complex<__type>> &input,         \
                        DeviceMemory<std::complex<__type>> *output) {            \
      return DoFftWithDirectionInternal(stream, plan, mufftExec##__fft_type1,    \
                                        input, output);                          \
    }                                                                            \
    bool MusaFft::DoFft(Stream *stream, fft::Plan *plan,                         \
                        const DeviceMemory<__type> &input,                       \
                        DeviceMemory<std::complex<__type>> *output) {            \
      return DoFftInternal(stream, plan, mufftExec##__fft_type2, input, output); \
    }                                                                            \
    bool MusaFft::DoFft(Stream *stream, fft::Plan *plan,                         \
                        const DeviceMemory<std::complex<__type>> &input,         \
                        DeviceMemory<__type> *output) {                          \
      return DoFftInternal(stream, plan, mufftExec##__fft_type3, input, output); \
    }

  STREAM_EXECUTOR_MUSA_DEFINE_FFT(float, C2C, R2C, C2R)
  STREAM_EXECUTOR_MUSA_DEFINE_FFT(double, Z2Z, D2Z, Z2D)

  #undef STREAM_EXECUTOR_MUSA_DEFINE_FFT

  }  // namespace gpu

  void initialize_mufft() {
    absl::Status status =
        PluginRegistry::Instance()->RegisterFactory<PluginRegistry::FftFactory>(
            musa::kMUSaPlatformId, "muFFT",
            [](StreamExecutor *parent) -> fft::FftSupport * {
              return new gpu::MusaFft(parent);
            });
    if (!status.ok()) {
      LOG(INFO) << "Unable to register muFFT factory: " << status.message();
    }
  }

  }  // namespace stream_executor

  STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(register_mufft, {
    stream_executor::initialize_mufft();
  });