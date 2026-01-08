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

#include "xla/service/gpu/llvm_gpu_backend/mtgpu_backend.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <random>
#include <functional>
#include <ios>
#include <memory>
#include <mutex>  // NOLINT
#include <optional>
#include <string>
#include <system_error>  // NOLINT
#include <utility>
#include <variant>
#include <vector>
#include <regex>

#include "absl/base/call_once.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/PassRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Scalar.h"
#include "xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "xla/service/gpu/llvm_gpu_backend/load_ir_module.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/musa_musdl_path.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/env_var.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"
#include "tsl/platform/random.h"
#include "tsl/profiler/lib/traceme.h"

#ifdef HAS_SUPPORT_FOR_LLD_AS_A_LIBRARY
#include <array>

#include "absl/base/const_init.h"
#include "absl/synchronization/mutex.h"
#include "lld/Common/Driver.h"
LLD_HAS_DRIVER(elf)
#endif

namespace xla {
namespace gpu {
namespace {

// Inline threshold value to use in LLVM MTGPU backend.
const int kMTGPUInlineThreshold = 0x100000;

// Gets the MUSa-Device-Libs filenames for a particular MTGPU version.
std::vector<std::string> GetMUSDLPaths(std::string gcn_arch_name,
                                       const std::string& musdl_dir_path){
  // MTGPU version-neutral bitcodes.
  static std::vector<std::string>* musdl_filenames =
      new std::vector<std::string>(
          {"libdevice.31.bc", "libdevice.bc", "libdevice.mthg.bc"});

  // Construct full path to MUSDL bitcode libraries.
  std::vector<std::string> result;
  result.reserve(musdl_filenames->size() + 1);
  for (auto& filename : *musdl_filenames) {
    result.push_back(tsl::io::JoinPath(musdl_dir_path, filename));
  }

  // Add MTGPU version-specific bitcodes.
  std::vector<std::string> tokens = absl::StrSplit(gcn_arch_name, ':');
  std::string mtgpu_version = gcn_arch_name;
  if (!tokens.empty() && tokens[0].size() >= 3) {
    mtgpu_version = tokens[0].substr(3);
  }
  result.push_back(tsl::io::JoinPath(
      musdl_dir_path,
      absl::StrCat("oclc_isa_version_", mtgpu_version, ".bc")));
  return result;
}

struct HsacoCacheEntry {
  uint64_t hash;
  std::string ir;
  std::string gfx;
  std::vector<uint8_t> hsaco;
};

struct HsacoCache {
 protected:
  std::vector<HsacoCacheEntry> cache;
  std::mutex m_mutex;
  int request_count = 0;
  int hit_count = 0;

 public:
  static bool Find(const std::string& ir, uint64_t& hash,
                   const std::string& gfx, std::vector<uint8_t>& hsaco);
  static void Add(const std::string& ir, uint64_t hash, const std::string& gfx,
                  const std::vector<uint8_t>& hsaco);
};

static HsacoCache g_hsacoCache;  // NOLINT: static/global vars forbidden

bool HsacoCache::Find(const std::string& ir, uint64_t& hash,
                      const std::string& gfx, std::vector<uint8_t>& hsaco) {
  std::lock_guard<std::mutex> lg(g_hsacoCache.m_mutex);
  hash = std::hash<std::string>{}(ir);
  bool hit = false;
  for (auto& x : g_hsacoCache.cache) {
    if (x.hash != hash) continue;
    if (x.gfx != gfx) continue;
    if (x.ir != ir) continue;
    hsaco = x.hsaco;
    hit = true;
    break;
  }
  g_hsacoCache.request_count++;
  if (hit) g_hsacoCache.hit_count++;
  if (!(g_hsacoCache.request_count % 50))
    VLOG(1) << "HSACO cache: " << g_hsacoCache.request_count << " requests, "
            << g_hsacoCache.hit_count << " hits";
  return hit;
}

void HsacoCache::Add(const std::string& ir, uint64_t hash,
                     const std::string& gfx,
                     const std::vector<uint8_t>& hsaco) {
  std::lock_guard<std::mutex> lg(g_hsacoCache.m_mutex);
  g_hsacoCache.cache.resize(g_hsacoCache.cache.size() + 1);
  g_hsacoCache.cache.back().ir = ir;
  g_hsacoCache.cache.back().hash = hash;
  g_hsacoCache.cache.back().gfx = gfx;
  g_hsacoCache.cache.back().hsaco = hsaco;
}

// Emits the given module to HSA Code Object. target_machine is an initialized
// TargetMachine for the MTGPU target.
absl::StatusOr<std::vector<uint8_t>> EmitModuleToHsaco(
    llvm::Module* module, llvm::TargetMachine* target_machine,
    const DebugOptions& debug_options) {
  auto* env = tsl::Env::Default();
  std::vector<std::string> tempdir_vector;
  env->GetLocalTempDirectories(&tempdir_vector);
  if (tempdir_vector.empty()) {
    return xla::Internal(
        "Unable to locate a temporary directory for compile-time artifacts.");
  }
  std::string tempdir_name = tempdir_vector.front();
  VLOG(1) << "Compile-time artifacts located at: " << tempdir_name;

  bool keep_tempfiles = false;
  TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_MUSA_KEEP_XLA_TEMPFILES",
                                      /*default_val=*/false, &keep_tempfiles));
  // Prepare filenames for all stages of compilation:
  // IR, binary ISA, and HSACO.
  std::string random_number = std::to_string(tsl::random::New64());
  std::string ir_filename =
      absl::StrCat(module->getModuleIdentifier(), random_number + ".ll");
  std::string ir_path = tsl::io::JoinPath(tempdir_name, ir_filename);

  std::string ir_opt_filename =
      absl::StrCat(module->getModuleIdentifier(), random_number + "_opt.ll");
  std::string ir_opt_path = tsl::io::JoinPath(tempdir_name, ir_opt_filename);

  std::string isabin_filename =
      absl::StrCat(module->getModuleIdentifier(), random_number + ".o");
  std::string isabin_path = tsl::io::JoinPath(tempdir_name, isabin_filename);

  std::string hsaco_filename =
      absl::StrCat(module->getModuleIdentifier(), random_number + ".hsaco");
  std::string hsaco_path = tsl::io::JoinPath(tempdir_name, hsaco_filename);

  std::error_code ec;

  // Dump LLVM IR.
  std::unique_ptr<llvm::raw_fd_ostream> ir_fs(
      new llvm::raw_fd_ostream(ir_path, ec, llvm::sys::fs::OF_None));
  module->print(*ir_fs, nullptr);
  ir_fs->flush();

  // Emit GCN ISA binary.
  llvm::legacy::PassManager pm;
  pm.add(new llvm::TargetLibraryInfoWrapperPass(
      llvm::Triple(module->getTargetTriple())));
  llvm::SmallVector<char, 0> stream;
  llvm::raw_svector_ostream pstream(stream);
  std::unique_ptr<llvm::raw_fd_ostream> isabin_fs(
      new llvm::raw_fd_ostream(isabin_path, ec, llvm::sys::fs::OF_Text));
  module->setDataLayout(target_machine->createDataLayout());
  target_machine->addPassesToEmitFile(pm, *isabin_fs, nullptr,
                                      llvm::CodeGenFileType::ObjectFile);
  pm.run(*module);
  isabin_fs->flush();

  if (keep_tempfiles) {
    std::unique_ptr<llvm::raw_fd_ostream> ir_fs(
        new llvm::raw_fd_ostream(ir_opt_path, ec, llvm::sys::fs::OF_None));
    module->print(*ir_fs, nullptr);
    ir_fs->flush();
  }

  if (debug_options.xla_gpu_use_inprocess_lld()) {
#ifdef HAS_SUPPORT_FOR_LLD_AS_A_LIBRARY
    static absl::Mutex lld_mu(absl::kConstInit);

    std::array<const char*, 7> args{
        "ld.lld",           "--threads=1",       "-shared",
        "--no-undefined",   isabin_path.c_str(), "-o",
        hsaco_path.c_str(),
    };

    std::string error_message;
    llvm::raw_string_ostream os(error_message);
    lld::Result result;
    {
      absl::MutexLock lock(&lld_mu);
      result =
          lld::lldMain(args, llvm::nulls(), os, {{lld::Gnu, &lld::elf::link}});
    }
    CHECK(result.canRunAgain)
        << "ld.lld (in-process) failed with fatal error " << error_message;
    if (result.retCode) {
      return xla::Internal(
          "ld.lld (in-process) execute fail: %s, error code %d", error_message,
          result.retCode);
    }
#else
    CHECK(false) << "Inprocess LLD is not supported.";
#endif
  } else {
    // Locate lld.
    std::string lld_path;
    if (std::getenv("LLVM_PATH")) {
      lld_path = tsl::io::JoinPath(std::getenv("LLVM_PATH"), "bin");
    } else {
      lld_path = tsl::io::JoinPath(tsl::MusaRoot(), "llvm/bin");
    }
    auto lld_program = llvm::sys::findProgramByName("ld.lld", {lld_path});
    if (!lld_program) {
      return xla::Internal("unable to find ld.lld in PATH: %s",
                           lld_program.getError().message());
    }
    std::vector<llvm::StringRef> lld_args{
        llvm_ir::AsStringRef("ld.lld"),
        llvm_ir::AsStringRef("-flavor"),
        llvm_ir::AsStringRef("gnu"),
        llvm_ir::AsStringRef("-shared"),
        llvm_ir::AsStringRef("--no-undefined"),
        llvm_ir::AsStringRef(isabin_path),
        llvm_ir::AsStringRef("-o"),
        llvm_ir::AsStringRef(hsaco_path),
    };

    std::string error_message;
    int lld_result =
        llvm::sys::ExecuteAndWait(*lld_program, llvm_ir::AsArrayRef(lld_args),
                                  std::nullopt, {}, 0, 0, &error_message);
    if (lld_result) {
      return xla::Internal("ld.lld execute fail: %s, error code %d",
                           error_message, lld_result);
    }
  }

  // Read HSACO.
  std::ifstream hsaco_file(hsaco_path, std::ios::binary | std::ios::ate);
  std::ifstream::pos_type hsaco_file_size = hsaco_file.tellg();

  std::vector<uint8_t> hsaco(hsaco_file_size);
  hsaco_file.seekg(0, std::ios::beg);
  hsaco_file.read(reinterpret_cast<char*>(hsaco.data()), hsaco_file_size);
  hsaco_file.close();
  if (!keep_tempfiles) {
    remove(ir_path.c_str());
    remove(isabin_path.c_str());
    remove(hsaco_path.c_str());
  }
  return hsaco;
}

// Links MUSa-Device-Libs into the given module if the module needs it.
absl::Status LinkMUSDLIfNecessary(llvm::Module* module,
                                  std::string gcn_arch_name,
                                  const std::string& musdl_dir_path) {
  if (!CouldNeedDeviceBitcode(*module)) {
    return absl::OkStatus();
  }

  return LinkWithBitcodeVector(module,
                               GetMUSDLPaths(gcn_arch_name, musdl_dir_path));
}

absl::Status MTGPUTargetModuleLinker(
    llvm::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options,
    const std::string& device_bitcode_dir_path) {
  // Link the input module with MUSDL.

  std::string gcn_arch_name = "mtgpu";
  TF_RETURN_IF_ERROR(
      LinkMUSDLIfNecessary(module, gcn_arch_name, device_bitcode_dir_path));

  // If ftz is enabled, set it as an attribute on every function in the module.
  if (debug_options.xla_gpu_ftz()) {
    for (llvm::Function& fn : *module) {
      fn.addFnAttr("denormal-fp-math-f32", "preserve-sign");
    }
  }
  const int32_t kAbiVersion = 500;
  module->addModuleFlag(llvm::Module::Error, "amdhsa_code_object_version",
                        kAbiVersion);

  return absl::OkStatus();
}

// The following routine maps a feature token extracted from the
// hipDeviceProp_t::gcnArchName string, and maps it to a valid feature_str
// to be used for creating the MTGPUTarget.
// This mapping is currently in a state of flux because TF XLA uses its
// own copy of LLVM, which is different from the LLVM version used by
// hipcc/runtime in the MUSa install. Ordinarily this is not a problem,
// but right now, the LLVM version used by hipcc/runtime has "targetID"
// related changes which have not yet been upstreamed (to the LLVM repo)
// When that upstreaming happens (and TF LLVM pointer moves past the
// upstream commit), the following mapping will need to change
std::string MapGCNArchNameTokenToFeatureStr(const std::string& token,
                                            const std::string& gfx) {
  return "";
}

std::pair<std::string, std::string> GetFeatureStrFromGCNArchName(
    const std::string& gcn_arch_name) {
  std::string feature_str;

  std::string gfx = gcn_arch_name;
  // For MUSa versions 4.0 and greater, we need to specify the correct
  // feature str, based on the underlying GPU HW to get max performance.
  std::vector<std::string> tokens = absl::StrSplit(gcn_arch_name, ':');
  std::vector<std::string> mapped_tokens;
  if (!tokens.empty()) gfx = tokens[0];
  for (auto it = tokens.begin(); it != tokens.end(); it++) {
    // Skip the first token, that is the gfxNNN str
    // The rest of the tokens are the feature/targetid strings
    if (it != tokens.begin()) {
      std::string token(*it);
      std::string mapped_token = MapGCNArchNameTokenToFeatureStr(token, gfx);
      mapped_tokens.push_back(mapped_token);
    }
  }
  feature_str = absl::StrJoin(mapped_tokens, ",");

  return std::make_pair(gfx, feature_str);
}

std::unique_ptr<llvm::TargetMachine> MTGPUGetTargetMachine(
    llvm::Triple target_triple, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options) {
  return GetTargetMachine(std::move(target_triple), "sm1.6", debug_options,
                          "");
}

void MTGPUBackendInit(const DebugOptions& debug_options,
                       std::string& musdl_dir_path) {
  // Initialize the MTGPU target; it's the only target we link with, so call
  // its specific initialization functions instead of the catch-all
  // InitializeAll*.
  /*
  LLVMInitializeMTGPUTarget();
  LLVMInitializeMTGPUTargetInfo();
  LLVMInitializeMTGPUTargetMC();
  LLVMInitializeMTGPUAsmParser();
  LLVMInitializeMTGPUAsmPrinter();

  musdl_dir_path = GetMUSDLDir(debug_options);
  */
  llvm::PassRegistry* registry = llvm::PassRegistry::getPassRegistry();
  gpu::InitializePasses(registry);
}

}  // namespace

namespace mtgpu {

std::vector<std::string> GetMTGPUBackendOptions(
    const DebugOptions& debug_options) {
  std::vector<std::string> backend_llvm_opts;

  // Extra backend options must go after regular backend options in order to be
  // able for the later to override the former.
  auto backend_extra_llvm_opts = llvm_ir::ExtractXlaBackendExtraOptions(
      debug_options.xla_backend_extra_options());
  backend_llvm_opts.insert(backend_llvm_opts.end(),
                           backend_extra_llvm_opts.cbegin(),
                           backend_extra_llvm_opts.cend());

  return backend_llvm_opts;
}

std::string LibDevicePath(std::string gcn_arch_name,
                          const std::string& musdl_dir_path) {
  auto libdevice_dir_paths = GetMUSDLPaths(gcn_arch_name, musdl_dir_path);
  for (auto libdevice_dir_path : libdevice_dir_paths) {
    if (libdevice_dir_path.find("libdevice.bc")) {
      return libdevice_dir_path;
    }
  }
  return "";
}

static std::string randomTmpPath() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    std::string random_name = "temp_";
    for (int i = 0; i < 16; ++i) {
      random_name += "0123456789abcdef"[dis(gen)];
    }
    return random_name;
}

#include "musa_intrinsic.def"
using llvm::CallInst;
static void preserveGlobalVars(llvm::Module &M)
{
    llvm::SmallPtrSet<llvm::Constant*, 16> Keep;   // 自动去重

    /* 1. 已存在于 @llvm.used 里的元素 */
    if (llvm::GlobalVariable *LLVMUsed = M.getGlobalVariable("llvm.used"))
      if (auto *CA = llvm::dyn_cast<llvm::ConstantArray>(LLVMUsed->getInitializer()))
        for (llvm::Value *Op : CA->operands())
	{
	  llvm::Value *V = llvm::cast<llvm::Constant>(Op)->getOperand(0);  // 先取 User 的 operand
    	  Keep.insert(llvm::cast<llvm::Constant>(V)); 
	}

    /* 2. 本模块所有定义 */
    for (llvm::GlobalVariable &GV : M.globals())
      if (!GV.isDeclaration())
        Keep.insert(&GV);

    if (Keep.empty()) return ;
    /* 3. 生成新的 @llvm.used */
    llvm::LLVMContext &Ctx = M.getContext();
    llvm::ArrayType *ATy = llvm::ArrayType::get(llvm::PointerType::get(M.getContext(), 0),
                                      Keep.size());

    llvm::SmallVector<llvm::Constant*, 64> Elements;
    for (llvm::Constant *C : Keep)
      Elements.push_back(llvm::ConstantExpr::getPointerCast(C, llvm::PointerType::get(M.getContext(), 0)));

    llvm::Constant *NewInit = llvm::ConstantArray::get(ATy, Elements);

    llvm::GlobalVariable *LLVMUsed = M.getGlobalVariable("llvm.used");
    if (!LLVMUsed) {
      LLVMUsed = new llvm::GlobalVariable(M, ATy, false,
                                    llvm::GlobalValue::AppendingLinkage,
                                    NewInit, "llvm.used");
      LLVMUsed->setSection("llvm.metadata");
    } else {
      LLVMUsed->setInitializer(NewInit);
    }
}
static void convertBF16BinOpToMusaIntrinsics(llvm::Module &M) {
    llvm::LLVMContext &Ctx = M.getContext();
    llvm::Type *BFloatTy = llvm::Type::getBFloatTy(Ctx);

    // 1. 一次性声明两个内建
    llvm::Function *HSub = M.getFunction("llvm.musa.hsub.bf16");
    llvm::Function *HMul = M.getFunction("llvm.musa.hmul.bf16");
    llvm::Function *HAdd = M.getFunction("llvm.musa.hadd.bf16");
    if (!HSub) {
      llvm::FunctionType *FTy = llvm::FunctionType::get(BFloatTy, {BFloatTy, BFloatTy}, false);
      HSub = llvm::Function::Create(FTy, llvm::GlobalValue::ExternalLinkage,
                              "llvm.musa.hsub.bf16", &M);
      HSub->setOnlyReadsMemory();
      HSub->setDoesNotThrow();
      HSub->setWillReturn();
    }
    if (!HMul) {
      llvm::FunctionType *FTy = llvm::FunctionType::get(BFloatTy, {BFloatTy, BFloatTy}, false);
      HMul = llvm::Function::Create(FTy, llvm::GlobalValue::ExternalLinkage,
                              "llvm.musa.hmul.bf16", &M);
      HMul->setOnlyReadsMemory();
      HMul->setDoesNotThrow();
      HMul->setWillReturn();
    }
    if (!HAdd) {
      llvm::FunctionType *FTy = llvm::FunctionType::get(BFloatTy, {BFloatTy, BFloatTy}, false);
      HAdd = llvm::Function::Create(FTy, llvm::GlobalValue::ExternalLinkage,
                          "llvm.musa.hadd.bf16", &M);
      HAdd->setOnlyReadsMemory(); HAdd->setDoesNotThrow(); HAdd->setWillReturn();
    }    

    // 2. 收集所有 fsub/fmul bfloat
    llvm::SmallVector<llvm::BinaryOperator*, 8> WorkList;
    for (llvm::Function &F : M)
      for (llvm::BasicBlock &BB : F)
        for (llvm::Instruction &I : BB)
          if (auto *Bin = llvm::dyn_cast<llvm::BinaryOperator>(&I))
            if (Bin->getType()->isBFloatTy() &&
                (Bin->getOpcode() == llvm::Instruction::FSub ||
                 Bin->getOpcode() == llvm::Instruction::FMul ||
                 Bin->getOpcode() == llvm::Instruction::FAdd
	       ))
            WorkList.push_back(Bin);

    // 3. 逐个替换
    llvm::IRBuilder<> B(Ctx);
    for (llvm::BinaryOperator *Bin : WorkList) {
      B.SetInsertPoint(Bin);
      llvm::Value *LHS = Bin->getOperand(0);
      llvm::Value *RHS = Bin->getOperand(1);
      llvm::Function *Callee = nullptr;
      switch (Bin->getOpcode()) {
        case llvm::Instruction::FAdd: Callee = HAdd; break;
        case llvm::Instruction::FSub: Callee = HSub; break;
        case llvm::Instruction::FMul: Callee = HMul; break;
        default: break;          // 其他指令不管
      }

      llvm::CallInst *Call = B.CreateCall(Callee->getFunctionType(), Callee, {LHS, RHS}, Bin->getName());
      Call->setTailCall();

      Bin->replaceAllUsesWith(Call);
      Bin->eraseFromParent();
    }
}

static void convertIRToMusaIntrinsics3(llvm::Module &M) {
    llvm::LLVMContext &Ctx = M.getContext();
    llvm::Type *BFloatTy = llvm::Type::getBFloatTy(Ctx);
    llvm::Type *F32Ty    = llvm::Type::getFloatTy(Ctx);

    // 1. 声明 musa 库函数（只一次）
    llvm::Function *Callee = M.getFunction("llvm.musa.bfloat162float");
    if (!Callee) {
      llvm::FunctionType *FTy = llvm::FunctionType::get(F32Ty, {BFloatTy}, false);
      Callee = llvm::Function::Create(FTy, llvm::GlobalValue::ExternalLinkage,
                                "llvm.musa.bfloat162float", &M);
      Callee->setOnlyReadsMemory();
      Callee->setDoesNotThrow();
      Callee->setWillReturn();
    }

    // 2. 扫描所有 fptosint (bfloat, bfloat)
    llvm::SmallVector<llvm::FPToSIInst *, 8> WorkList;
    for (llvm::Function &F : M)
      for (llvm::BasicBlock &BB : F)
        for (llvm::Instruction &I : BB)
          if (auto *CI = llvm::dyn_cast<llvm::FPToSIInst>(&I))
            if (CI->getSrcTy()->isBFloatTy())
              WorkList.push_back(CI);

    // 3. 逐个替换
    llvm::IRBuilder<> B(Ctx);
    for (llvm::FPToSIInst *CI : WorkList) {
      llvm::Value *Op0 = CI->getOperand(0);

      B.SetInsertPoint(CI);

      // 3.1 转 float
      llvm::CallInst *TempX = B.CreateCall(Callee->getFunctionType(), Callee, {Op0});
      TempX->setTailCall();

      // 3.2 新的 fcmp
      llvm::Value *NewCmp = B.CreateFPToSI(TempX, CI->getType(), CI->getName());

      CI->replaceAllUsesWith(NewCmp);
      CI->eraseFromParent();
    }
}

static void convertIRToMusaIntrinsics2(llvm::Module &M) {
    llvm::LLVMContext &Ctx = M.getContext();
    llvm::Type *BFloatTy = llvm::Type::getBFloatTy(Ctx);
    llvm::Type *F32Ty    = llvm::Type::getFloatTy(Ctx);

    // 1. 声明 musa 库函数（只一次）
    llvm::Function *Callee = M.getFunction("llvm.musa.bfloat162float");
    if (!Callee) {
      llvm::FunctionType *FTy = llvm::FunctionType::get(F32Ty, {BFloatTy}, false);
      Callee = llvm::Function::Create(FTy, llvm::GlobalValue::ExternalLinkage,
                                "llvm.musa.bfloat162float", &M);
      Callee->setOnlyReadsMemory();
      Callee->setDoesNotThrow();
      Callee->setWillReturn();
    }

    // 2. 扫描所有 fcmp une (bfloat, bfloat)
    llvm::SmallVector<llvm::FCmpInst *, 8> WorkList;
    for (llvm::Function &F : M)
      for (llvm::BasicBlock &BB : F)
        for (llvm::Instruction &I : BB)
          if (auto *CI = llvm::dyn_cast<llvm::FCmpInst>(&I))
            if (CI->getOperand(0)->getType()->isBFloatTy() || 
		CI->getOperand(1)->getType()->isBFloatTy() )
              WorkList.push_back(CI);

    // 3. 逐个替换
    llvm::IRBuilder<> B(Ctx);
    for (llvm::FCmpInst *CI : WorkList) {
      llvm::Value *Op0 = CI->getOperand(0);
      llvm::Value *Op1 = CI->getOperand(1);

      B.SetInsertPoint(CI);

      // 3.1 转 float
      llvm::CallInst *TempX = B.CreateCall(Callee->getFunctionType(), Callee, {Op0});
      TempX->setTailCall();
      llvm::CallInst *TempImm = B.CreateCall(Callee->getFunctionType(), Callee, {Op1});
      TempImm->setTailCall();

      // 3.2 新的 fcmp
      llvm::Value *NewCmp = B.CreateFCmp(CI->getPredicate(), TempX, TempImm, CI->getName());

      CI->replaceAllUsesWith(NewCmp);
      CI->eraseFromParent();
    }
}

static void convertIRToMusaIntrinsics1(llvm::Module &M) {
// 0. 提前准备好的变量
  llvm::LLVMContext &Ctx = M.getContext();
  llvm::Type *F32Ty    = llvm::Type::getFloatTy(Ctx);
  llvm::Type *BFloatTy = llvm::Type::getBFloatTy(Ctx);

// 1. 声明 musa 库函数（只一次）
  llvm::Function *Callee = M.getFunction("llvm.musa.float2bfloat16");
  if (!Callee) {
    llvm::FunctionType *FTy = llvm::FunctionType::get(BFloatTy, {F32Ty}, false);
    Callee = llvm::Function::Create(FTy, llvm::GlobalValue::ExternalLinkage,
                            "llvm.musa.float2bfloat16", &M);
    Callee->setOnlyReadsMemory();   // readnone
    Callee->setDoesNotThrow();      // nounwind
    Callee->setWillReturn();        // willreturn
  }

// 2. 收集所有 fptrunc float→bfloat
  llvm::SmallVector<llvm::FPTruncInst*, 8> WorkList;
  for (llvm::Function &F : M)
    for (llvm::BasicBlock &BB : F)
      for (llvm::Instruction &I : BB)
        if (auto *Trunc = llvm::dyn_cast<llvm::FPTruncInst>(&I))
          if (Trunc->getSrcTy()->isFloatTy() &&
              Trunc->getDestTy()->isBFloatTy())
            WorkList.push_back(Trunc);

// 3. 逐个替换
  llvm::IRBuilder<> B(Ctx);
  for (llvm::FPTruncInst *Trunc : WorkList) {
    B.SetInsertPoint(Trunc);
    llvm::CallInst *Call = B.CreateCall(Callee->getFunctionType(),
                                Callee,
                                {Trunc->getOperand(0)});
    Call->setTailCall();                                    // tail
    Call->setAttributes(Callee->getAttributes());

    Trunc->replaceAllUsesWith(Call);
    Trunc->eraseFromParent();
  }
}
static void convertIRToMusaIntrinsics(llvm::Module &M) {
// 1. 先声明 musa 库函数（只声明一次）
  llvm::Function *F32FromBF16 = M.getFunction("llvm.musa.bfloat162float");
  if (!F32FromBF16) {
    llvm::Type *BFloatTy = llvm::Type::getBFloatTy(M.getContext());
    llvm::Type *F32Ty    = llvm::Type::getFloatTy(M.getContext());
    llvm::FunctionType *FTy =
      llvm::FunctionType::get(F32Ty, {BFloatTy}, /*isVarArg=*/false);

    F32FromBF16 = llvm::Function::Create(FTy, llvm::GlobalValue::ExternalLinkage,
                                 "llvm.musa.bfloat162float", &M);
  // 加属性
    F32FromBF16->setOnlyReadsMemory();          // readnone
    F32FromBF16->setDoesNotThrow();             // nounwind
    F32FromBF16->setWillReturn();               // willreturn
  }

  // 2. 收集所有 fpext bfloat → float
  llvm::SmallVector<llvm::FPExtInst*, 8> WorkList;
  for (llvm::Function &F : M)
    for (llvm::BasicBlock &BB : F)
      for (llvm::Instruction &I : BB)
        if (auto *Ext = llvm::dyn_cast<llvm::FPExtInst>(&I))
          if (Ext->getSrcTy()->isBFloatTy() &&
            Ext->getDestTy()->isFloatTy())
            WorkList.push_back(Ext);

// 3. 逐个替换
  llvm::IRBuilder<> B(M.getContext());
  for (llvm::FPExtInst *Ext : WorkList) {
    B.SetInsertPoint(Ext);

    llvm::CallInst *Call = B.CreateCall(F32FromBF16->getFunctionType(),
                                F32FromBF16,
                                {Ext->getOperand(0)});
    Call->setTailCall();                // tail
    Call->getFastMathFlags().setAllowContract(true);
    Call->setAttributes(F32FromBF16->getAttributes());

    Ext->replaceAllUsesWith(Call);
    Ext->eraseFromParent();
  }
}
static void convertNvvmToMusaIntrinsics(llvm::Module &M) {
    // 1. 收集所有匹配规则的调用
    struct Rec { llvm::CallInst *CI; const Rule *R; };
    std::vector<Rec> worklist;

    for (llvm::Function &F : M) {
        for (llvm::BasicBlock &BB : F) {
            for (llvm::Instruction &I : BB) {
                if (auto *CI = llvm::dyn_cast<llvm::CallInst>(&I)) {
                    llvm::Function *Callee = CI->getCalledFunction();
                    if (!Callee) continue;
                    llvm::StringRef old = Callee->getName();
                    for (const Rule &R : rules) {
                        if (old != R.oldName) continue;
                        if (R.type == 1) {
                            if (CI->arg_size() != 1) continue;
                            //auto *C = dyn_cast<ConstantInt>(CI->getArgOperand(0));
                            //if (!C || C->getZExtValue() != 0) continue;
                        }
                        worklist.push_back({CI, &R});
                        break; // 一条规则已命中，无需再试
                    }
                }
            }
        }
    }

    // 2. 统一处理：创建新声明 + 替换调用
    for (const Rec &rec : worklist) {
        llvm::CallInst *CI   = rec.CI;
        const Rule *R  = rec.R;
        llvm::Function *oldF = CI->getCalledFunction();
        llvm::Function *newF = M.getFunction(R->newName);
        if (!newF) {
	  switch (R->type) {
	    case 3:
	    case 2:
	    {
	      llvm::LLVMContext &Ctx = M.getContext();
	      llvm::Type *Int32Ty = llvm::Type::getInt32Ty(Ctx);
      	      llvm::PointerType *PtrAS5Ty = llvm::PointerType::get(Ctx, 5);
      	      llvm::FunctionType *NewFTy = llvm::FunctionType::get(Int32Ty, {Int32Ty, Int32Ty, Int32Ty, PtrAS5Ty}, false);
	      newF = llvm::Function::Create(NewFTy, llvm::GlobalValue::ExternalLinkage, R->newName, &M);

	      llvm::AttributeList NewAttrs;
	      llvm::AttrBuilder FnAttr(Ctx);
	      FnAttr.addAttribute(llvm::Attribute::Convergent);
	      FnAttr.addAttribute(llvm::Attribute::NoUnwind);
	      FnAttr.addMemoryAttr(llvm::MemoryEffects::writeOnly());   // LLVM 15+ 写法
	      NewAttrs = NewAttrs.addFnAttributes(Ctx, FnAttr);

	      newF->setAttributes(NewAttrs);
              newF->setCallingConv(oldF->getCallingConv());
	      break;
	    }
	    case 1:
	    {
	      llvm::Type *voidTy = llvm::Type::getVoidTy(M.getContext());
  	      llvm::FunctionType *noArgTy = llvm::FunctionType::get(voidTy, /*isVarArg=*/false);

  	      newF = llvm::Function::Create(noArgTy,
                                llvm::GlobalValue::ExternalLinkage,
                                R->newName, &M);
              newF->setAttributes(oldF->getAttributes());
              newF->setCallingConv(oldF->getCallingConv());
	      break;
	    }
	    default:
	    {
              newF = llvm::Function::Create(oldF->getFunctionType(),
                                    llvm::GlobalValue::ExternalLinkage,
                                    R->newName, &M);
              newF->setAttributes(oldF->getAttributes());
              newF->setCallingConv(oldF->getCallingConv());
	      break;
	    }
	  }
        }
	switch (R->type) {
	  case 3:
	  {
	    llvm::LLVMContext &Ctx = M.getContext();
	    llvm::Type *Int32Ty = llvm::Type::getInt32Ty(Ctx);
	    llvm::Type *F32Ty = llvm::Type::getFloatTy(Ctx);
      	    llvm::PointerType *PtrAS5Ty = llvm::PointerType::get(Ctx, 5);
      	    llvm::FunctionType *NewFTy = llvm::FunctionType::get(Int32Ty, {Int32Ty, Int32Ty, Int32Ty, PtrAS5Ty}, false);
	    llvm::IRBuilder<> Builder(CI);
            llvm::Value *Val = CI->getArgOperand(1);
            llvm::Value *Offset = CI->getArgOperand(2);
            llvm::Value *Mask = CI->getArgOperand(3);
	    llvm::Value *NullPtr = llvm::ConstantPointerNull::get(PtrAS5Ty);

	    llvm::Value *TempX = Builder.CreateBitCast(Val, Int32Ty);   // 不指定名字
	    llvm::CallInst *NewCall = llvm::CallInst::Create(NewFTy, newF, {TempX, Offset, Mask, NullPtr}, "", CI);
	    NewCall->setCallingConv(CI->getCallingConv());
	    NewCall->setAttributes(CI->getAttributes());
	    llvm::Value *Result = Builder.CreateBitCast(NewCall, F32Ty, CI->getName()); // 继承原调用名

            CI->replaceAllUsesWith(Result);
            CI->eraseFromParent();
	    break;
	  }
	  case 2:
	  {
	    llvm::LLVMContext &Ctx = M.getContext();
	    llvm::Type *Int32Ty = llvm::Type::getInt32Ty(Ctx);
      	    llvm::PointerType *PtrAS5Ty = llvm::PointerType::get(Ctx, 5);
      	    llvm::FunctionType *NewFTy = llvm::FunctionType::get(Int32Ty, {Int32Ty, Int32Ty, Int32Ty, PtrAS5Ty}, false);
	    llvm::IRBuilder<> Builder(CI);
            llvm::Value *Val = CI->getArgOperand(1);
            llvm::Value *Offset = CI->getArgOperand(2);
            llvm::Value *Mask = CI->getArgOperand(3);
	    llvm::Value *NullPtr = llvm::ConstantPointerNull::get(PtrAS5Ty);
	    llvm::CallInst *NewCall = Builder.CreateCall(NewFTy, newF, {Val, Offset, Mask, NullPtr});

	    NewCall->takeName(CI);                      // 保留原名字
	    NewCall->setCallingConv(CI->getCallingConv());
	    NewCall->setAttributes(CI->getAttributes());
	    if (auto DL = CI->getDebugLoc())
    	      NewCall->setDebugLoc(DL);

            CI->replaceAllUsesWith(NewCall);
            CI->eraseFromParent();
	    break;
	  }
	  case 1:
	  {
	    llvm::Type *voidTy = llvm::Type::getVoidTy(M.getContext());
  	    llvm::FunctionType *noArgTy = llvm::FunctionType::get(voidTy, /*isVarArg=*/false);
	    llvm::IRBuilder<> Builder(CI);
	    llvm::CallInst *NewCall = Builder.CreateCall(noArgTy, newF, {});
	    NewCall->takeName(CI);                      // 保留原名字
	    NewCall->setCallingConv(CI->getCallingConv());
	    NewCall->setAttributes(CI->getAttributes());
	    if (auto DL = CI->getDebugLoc())
    	      NewCall->setDebugLoc(DL);
	    CI->replaceAllUsesWith(NewCall);
	    CI->eraseFromParent();
	    //CI->mutateFunctionType(noArgTy); // 确保类型一致
	    break;
	  }
	  default:
	  {
            CI->setCalledFunction(newF);
	    break;
	  }
        }
    }

    // 3. 清理无人使用的旧声明
    for (const Rule &R : rules) {
        if (llvm::Function *F = M.getFunction(R.oldName))
            if (F->use_empty()) F->eraseFromParent();
    }
    assert(!llvm::verifyModule(M, &llvm::errs()));
}

absl::StatusOr<std::vector<uint8_t>> CompileToHsaco(
    llvm::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options,
    const std::string& module_config_cache_key) {
  std::vector<uint8_t> hsaco;
  std::string str;
  llvm::raw_string_ostream stream(str);
  stream << *module;
  // Delete the first two lines, since they usually vary even when the rest of
  // the code is the same (but verify that they are what we expect).
  if (str.size() >= 13 && str.substr(0, 13) == "; ModuleID = ") {
    auto pos = str.find('\n');
    if (pos != std::string::npos) str = str.substr(pos + 1);
  }
  if (str.size() >= 18 && str.substr(0, 18) == "source_filename = ") {
    auto pos = str.find('\n');
    if (pos != std::string::npos) str = str.substr(pos + 1);
  }
  str += module_config_cache_key;
  tsl::profiler::TraceMe activity(
    [&] { return absl::StrCat("Compiling IR", module->getName().str()); },
    tsl::profiler::TraceMeLevel::kInfo);
  XLA_SCOPED_LOGGING_TIMER("Compile module " + module->getName().str());

  std::string gcn_arch_name = "mtgpu";

  uint64_t hash;
  if (HsacoCache::Find(str, hash, gcn_arch_name, hsaco)) {
    VLOG(1) << "HSACO cache hit";
    return hsaco;
  }
  VLOG(1) << "HSACO cache miss";
  
  std::string temp_name = randomTmpPath();
  std::string objFile = temp_name+".o";
  std::string mubinFile = temp_name+".mubin";
  std::string optFile = temp_name+"_opt.ll";
  std::string orgFile = temp_name+"_org.ll";
  std::string linkFile;
  std::string llFile;
  //Just for debug
  const std::filesystem::path file = "xla_test_debug.ll";
  const std::filesystem::path mufile = "xla_test_debug.mubin";
  if (std::filesystem::exists(mufile) && std::filesystem::is_regular_file(mufile)) {
    std::ifstream bin(mufile, std::ios::binary | std::ios::ate);
    if (!bin) throw std::runtime_error("cannot open mubin");
    size_t sz = bin.tellg();
    bin.seekg(0, std::ios::beg);
    hsaco.resize(sz);
    bin.read(reinterpret_cast<char*>(hsaco.data()), sz);
    bin.close();

    HsacoCache::Add(str, hash, gcn_arch_name, hsaco);

    return hsaco;
  }
  if (std::filesystem::exists(file) && std::filesystem::is_regular_file(file)) {
    linkFile = "xla_test_debug.ll";
    llFile = "xla_test_debug.ll";
  }
  else
  {
    //Dump org .ll
    std::error_code EC;
    llvm::raw_fd_ostream f_os(orgFile, EC, llvm::sys::fs::OF_Text);
    if (EC) throw std::runtime_error("cannot open " + orgFile);
    module->print(f_os, nullptr);
    f_os.close();

    //Change func call convension.

    convertNvvmToMusaIntrinsics(*module);
    convertIRToMusaIntrinsics(*module);
    convertIRToMusaIntrinsics1(*module);
    convertIRToMusaIntrinsics2(*module);
    convertIRToMusaIntrinsics3(*module);
    convertBF16BinOpToMusaIntrinsics(*module);
    preserveGlobalVars(*module);

    llFile = temp_name+".ll";
    linkFile = temp_name+"_link.ll";
    std::string bcFile = "/data/moon/github/openxla/third_party/gpus/musa/libdevice.31.ll";

    std::string ir_content;
    llvm::raw_string_ostream ir_stream(ir_content);
    module->print(ir_stream, nullptr);
    ir_stream.flush();

    //Replace ptx_kernel to mtgpu_kernel
    std::regex ptx_kernel_regex("ptx_kernel");
    ir_content = std::regex_replace(ir_content, ptx_kernel_regex, "mtgpu_kernel");

    llvm::raw_fd_ostream os(llFile, EC, llvm::sys::fs::OF_Text);
    if (EC) throw std::runtime_error("cannot open " + llFile);
    os << ir_content;
    os.close();

    std::string llvmlinkCmd =
    "llvm-link \"" + llFile + "\" \"" + bcFile + "\" --only-needed -S -o \"" + linkFile + "\"";
    if (std::system(llvmlinkCmd.c_str()) != 0)
      throw std::runtime_error("llvm-link failed");
  }

    std::string optCmd = "opt -O2 \"" + linkFile + "\" -S -o \"" + optFile + "\"";
    if (std::system(optCmd.c_str()) != 0)
      throw std::runtime_error("opt failed");

    std::string llcCmd = "llc \"" + optFile + "\" -march=mtgpu -mcpu=mp_31 "
                         "-filetype=obj -o \"" + objFile + "\"";
    if (std::system(llcCmd.c_str()) != 0)
      throw std::runtime_error("llc failed");
    // 5. lld
  std::string lldCmd = "lld -flavor gnu -shared \"" + objFile + "\" -o \"" + mubinFile + "\"";
  if (std::system(lldCmd.c_str()) != 0)
    throw std::runtime_error("lld failed");

    // 6. 读二进制
  std::ifstream bin(mubinFile, std::ios::binary | std::ios::ate);
  if (!bin) throw std::runtime_error("cannot open mubin");
  size_t sz = bin.tellg();
  bin.seekg(0, std::ios::beg);
  hsaco.resize(sz);
  bin.read(reinterpret_cast<char*>(hsaco.data()), sz);
  bin.close();

    // 7. 清理
  //std::remove(llFile.c_str());
  //std::remove(objFile.c_str());
  //std::remove(mubinFile.c_str());
  //std::cout << llFile;


  //Cache hsaco
  HsacoCache::Add(str, hash, gcn_arch_name, hsaco);
  //module->dropAllReferences();  // 清空 use-list

  return hsaco;
}

}  // namespace mtgpu
}  // namespace gpu
}  // namespace xla
