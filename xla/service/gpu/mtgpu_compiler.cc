/* Copyright 2017 The OpenXLA Authors.

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
#include "xla/service/gpu/mtgpu_compiler.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/IR/Module.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/cublas.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/convert_mover.h"
#include "xla/hlo/transforms/simplifiers/dot_dimension_merger.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/hlo/transforms/simplifiers/hlo_constant_folding.h"
#include "xla/hlo/transforms/simplifiers/reshape_mover.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/call_inliner.h"
#include "xla/service/float_support.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/autotuning/autotuner_pass.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/autotuning/conv_algorithm_picker.h"
#include "xla/service/gpu/autotuning/gemm_fusion_autotuner.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/cublas_padding_requirements.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/llvm_gpu_backend/mtgpu_backend.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/service/gpu/transforms/algebraic_simplifier.h"
#include "xla/service/gpu/transforms/conv_padding_legalization.h"
#include "xla/service/gpu/transforms/conv_rewriter.h"
#include "xla/service/gpu/transforms/cublas_pad_for_gemms.h"
#include "xla/service/gpu/transforms/cudnn_fused_conv_rewriter.h"
#include "xla/service/gpu/transforms/gpusolver_rewriter.h"
#include "xla/service/gpu/transforms/triangular_solve_rewriter.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {
class ConvBfloat16Support : public FloatSupport {
 public:
  explicit ConvBfloat16Support()
      : FloatSupport(BF16),
        is_conv_bf16_supported_(true) {}

  bool SupportsLowPrecisionOperand(const HloInstruction& hlo,
                                   int64_t operand_index) const override {
    return is_conv_bf16_supported_;
  }

  bool SupportsLowPrecisionOutput(const HloInstruction& hlo) const override {
    return is_conv_bf16_supported_;
  }

  bool SupportsMixedPrecisions(const HloInstruction& hlo) const override {
    // Skip all HLOs other than convolutions.
    return is_conv_bf16_supported_;
  }

 private:
  bool is_conv_bf16_supported_;
};

class MatmulBfloat16Support : public FloatSupport {
 public:
  explicit MatmulBfloat16Support()
      : FloatSupport(BF16),
        is_matmul_bf16_supported_(false) {}

  bool SupportsLowPrecisionOperand(const HloInstruction& hlo,
                                   int64_t operand_index) const override {
    return false;
  }

  bool SupportsLowPrecisionOutput(const HloInstruction& hlo) const override {
    return false;
  }

  bool SupportsMixedPrecisions(const HloInstruction& hlo) const override {
    return false;
  }

 private:
  bool is_matmul_bf16_supported_;
};

}  // namespace

absl::Status MTGPUCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, se::GpuComputeCapability gpu_version,
    se::dnn::VersionInfo dnn_version,
    const se::SemanticVersion& toolkit_version) {
  // Convert convolutions into CustomCalls to MIOpen, then canonicalize them
  // (PadInsertion).
  HloPassPipeline pipeline("conv_canonicalization");
  pipeline.AddInvariantCheckerDebug<HloVerifier>(
      /*layout_sensitive=*/false,
      /*allow_mixed_precision=*/false);

  //const std::pair<PrimitiveType, PrimitiveType> ar_promoted_types[] = {
  //    {BF16, F32}};
  //pipeline.AddPass<AllReducePromotion>(ar_promoted_types);

  // Convert unsupported bf16 convolutions to f32.
  ConvBfloat16Support conv_bf16_support;

   //Just for test bf16 by zp 20251222.
  pipeline.AddPass<FloatNormalization>(&conv_bf16_support);

  //MatmulBfloat16Support matmul_bf16_support;
  //pipeline.AddPass<FloatNormalization>(&matmul_bf16_support);

  //pipeline.AddPass<GpusolverRewriter>(
  //    stream_executor::RocmSolverContext::Create);
  //pipeline.AddPass<ConvRewriter>(gpu_version);
  //pipeline.AddPass<ConvPaddingLegalization>();
  //auto rcc = std::get<se::RocmComputeCapability>(gpu_version);
  //pipeline.AddPass<CudnnFusedConvRewriter>(rcc, dnn_version, toolkit_version);

  // The conv padding/vectorization passes which we need to get rid of.  They
  // also leave behind unnecessary tuple/get-tuple-element pairs that
  // TupleSimplifier fixes.
  pipeline.AddPass<CallInliner>();
  pipeline.AddPass<TupleSimplifier>();

  // tf2xla bridge, DepthwiseConvolutionConverter and ConvRewriter
  // introduces reshapes and transposes that can be eliminated using
  // AlgebraicSimplifier  We run algsimp to a fixed point.
  AlgebraicSimplifierOptions algsimp_options = GetAlgebraicSimplifierOptions(
      AlgebraicSimplifierMode::kGpuConvoluationCanonicalization,
      hlo_module->config().debug_options(),
      /*is_musa=*/true);
  pipeline.AddPass<HloPassFix<GpuAlgebraicSimplifier>>(algsimp_options,
                                                       gpu_version);

  pipeline.AddPass<ConvertMover>();
  pipeline.AddPass<GpuAlgebraicSimplifier>(algsimp_options, gpu_version);

  // ConvRewriter, ConvPaddingLegalization and
  // CudnnConvPadForTensorCores may add instructions which can be simplified
  // by constant folding.
  pipeline.AddPass<HloConstantFolding>();
  TF_RETURN_IF_ERROR(
      pipeline
          .Run(hlo_module,
               /*execution_threads=*/{HloInstruction::kMainExecutionThread})
          .status());

  return absl::OkStatus();
}

absl::Status MTGPUCompiler::OptimizeHloPostLayoutAssignment(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& options, const TargetConfig& gpu_target_config,
    const GpuAliasInfo* alias_info, tsl::thread::ThreadPool* thread_pool) {
  HloPassPipeline pre_pipeline("MTGPU post-layout_assignment part 1");

  //auto musa_compute_capability = std::get<se::RocmComputeCapability>(
  //    gpu_target_config.device_description.gpu_compute_capability());

  //pre_pipeline.AddPass<DotDimensionMerger>();

  //for (const auto& req : HipblasPaddingRequirements) {
  //  pre_pipeline.AddPass<CublasPadForGemms>(musa_compute_capability,
  //                                          req.data_type, req.multiple_of);
  //}
  // Padding a gemm operand that's a constant results in pad(constant).  Run
  // constant-folding to simplify this into a new constant.
  pre_pipeline.AddPass<HloConstantFolding>();
  TF_RETURN_IF_ERROR(
      pre_pipeline
          .Run(hlo_module,
               /*execution_threads=*/{HloInstruction::kMainExecutionThread})
          .status());

  TF_RETURN_IF_ERROR(GpuCompiler::OptimizeHloPostLayoutAssignment(
      hlo_module, stream_exec, options, gpu_target_config, alias_info,
      thread_pool));

  HloPassPipeline post_pipeline("MTGPU post-layout_assignment part 2");

  // Transform TriangularSolve ops into custom-calls, so we can add temp
  // memory.
  //post_pipeline.AddPass<TriangularSolveRewriter>();

  TF_RETURN_IF_ERROR(
      post_pipeline
          .Run(hlo_module,
               /*execution_threads=*/{HloInstruction::kMainExecutionThread})
          .status());

  return absl::OkStatus();
}

// Linearize collective schedule under if online autotuning of convolutions is
// enabled.
bool MTGPUCompiler::RequiresCollectiveScheduleLinearizer(
    const HloModule* module, se::StreamExecutor* stream_exec) {
  if (stream_exec == nullptr || !GpuConvAlgorithmPicker::IsEnabled(module)) {
    return false;
  }
  for (const HloComputation* comp : module->MakeNonfusionComputations()) {
    for (const HloInstruction* inst : comp->instructions()) {
      if (GpuConvAlgorithmPicker::IsCandidate(inst)) {
        return true;
      }
    }
  }
  // No convolution auto-tuning candidates found in the module.
  return false;
}

absl::Status MTGPUCompiler::AddConvAndGemmAutotuningPasses(
    HloPassPipeline* pipeline, const se::GpuComputeCapability& gpu_version,
    const CompileOptions& options, HloModule* hlo_module,
    AutotuneConfig& autotune_config, tsl::thread::ThreadPool* thread_pool,
    se::StreamExecutor* stream_exec) {
  const DebugOptions& debug_options = hlo_module->config().debug_options();
  if (hlo_module->config()
          .debug_options()
          .xla_gpu_experimental_disable_binary_libraries() ||
      debug_options.xla_gpu_autotune_level() == 0 ||
      debug_options.xla_gpu_exclude_nondeterministic_ops() ||
      stream_exec == nullptr) {
    return absl::OkStatus();
  }

  return absl::OkStatus();
}

MTGPUCompiler::MTGPUCompiler()
    : GpuCompiler(stream_executor::musa::kMUSaPlatformId,
                  mtgpu::TargetTriple(), mtgpu::DataLayout()) {}

absl::StatusOr<GpuCompiler::BackendCompileResult>
MTGPUCompiler::CompileTargetBinary(
    const HloModuleConfig& module_config, llvm::Module* llvm_module,
    const se::DeviceDescription& device_description, bool relocatable,
    const HloModule* debug_module, const CompileOptions& options,
    std::optional<int> shard_number) {
  if (relocatable) {
    return Unimplemented("relocatable target binary is not implemented");
  }

  std::vector<uint8_t> hsaco;
  {
    // This may print multiple lines per HLO compilation because of the
    // parallelized compilation of LLVM modules.
    XLA_SCOPED_LOGGING_TIMER_IF(
        "MTGPUCompiler::CompileTargetBinary - CompileToHsaco",
        !options.is_autotuning_compilation);
    TF_ASSIGN_OR_RETURN(
        hsaco, mtgpu::CompileToHsaco(
                   llvm_module, device_description.gpu_compute_capability(),
                   module_config.debug_options(),
                   module_config.compilation_cache_key()));
  }

  return BackendCompileResult{"", std::move(hsaco)};
}

absl::Status MTGPUCompiler::AddGemmFusionAutotuningPasses(
    HloPassPipeline* pipeline, HloModule* hlo_module,
    AutotuneConfig& autotune_config, tsl::thread::ThreadPool* thread_pool,
    const MultiProcessKeyValueStore& key_value_store,
    const se::SemanticVersion& toolkit_version,
    se::StreamExecutor* stream_executor) {
  //TODO
  return absl::OkStatus();
}

std::vector<std::string> MTGPUCompiler::GetLLVMCommandLineOptions(
    const DebugOptions& debug_options) const {
  return mtgpu::GetMTGPUBackendOptions(debug_options);
}
}  // namespace gpu
}  // namespace xla
