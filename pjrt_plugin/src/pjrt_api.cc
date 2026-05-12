#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/stream_executor/platform_manager.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/plugin_c_api.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/c_api.h"

#include <iostream>
#include <memory>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <string>
#include <dlfcn.h>

extern "C" {

// =========================================================================
// 🏛️ TF 2.15 官方真实 C API 布局还原 (对齐 tpu/c_api_decl.h)
// =========================================================================
#define TPU_C_API_MAX_INLINED 6

struct TF215_IntList { union { int* heap; int inlined[TPU_C_API_MAX_INLINED]; }; int64_t size; };
struct TF215_Int64List { union { int64_t* heap; int64_t inlined[TPU_C_API_MAX_INLINED]; }; int64_t size; };
struct TF215_BoolList { union { bool* heap; bool inlined[TPU_C_API_MAX_INLINED]; }; int64_t size; };
struct TF215_Tile { TF215_Int64List dimensions; };
struct TF215_TileList { union { TF215_Tile* heap; TF215_Tile inlined[TPU_C_API_MAX_INLINED]; }; int64_t size; };

struct TF215_Layout {
    TF215_Int64List minor_to_major;
    TF215_IntList dim_level_types;
    TF215_IntList dim_unique;
    TF215_IntList dim_ordered;
    TF215_TileList tiles;
    int index_primitive_type;
    int pointer_primitive_type;
    int64_t element_size_in_bits;
    int64_t memory_space;
    int64_t dynamic_shape_metadata_prefix_bytes;
};

struct TF215_Shape {
    int element_type;
    TF215_Int64List dimensions;
    TF215_BoolList dynamic_dimensions;
    struct TF215_Shape* tuple_shapes; // owned
    int ntuple_shapes;
    bool has_layout;
    TF215_Layout layout;
};

// 深拷贝拦截器实现
static void DeepCopyIntList(TF215_IntList* dst, const TF215_IntList* src) {
    dst->size = src->size;
    if (src->size > TPU_C_API_MAX_INLINED) {
        dst->heap = new int[src->size]; memcpy(dst->heap, src->heap, src->size * sizeof(int));
    } else { memcpy(dst->inlined, src->inlined, src->size * sizeof(int)); }
}
static void DeepCopyInt64List(TF215_Int64List* dst, const TF215_Int64List* src) {
    dst->size = src->size;
    if (src->size > TPU_C_API_MAX_INLINED) {
        dst->heap = new int64_t[src->size]; memcpy(dst->heap, src->heap, src->size * sizeof(int64_t));
    } else { memcpy(dst->inlined, src->inlined, src->size * sizeof(int64_t)); }
}
static void DeepCopyBoolList(TF215_BoolList* dst, const TF215_BoolList* src) {
    dst->size = src->size;
    if (src->size > TPU_C_API_MAX_INLINED) {
        dst->heap = new bool[src->size]; memcpy(dst->heap, src->heap, src->size * sizeof(bool));
    } else { memcpy(dst->inlined, src->inlined, src->size * sizeof(bool)); }
}
static void DeepCopyTile(TF215_Tile* dst, const TF215_Tile* src) {
    DeepCopyInt64List(&dst->dimensions, &src->dimensions);
}
static void DeepCopyTileList(TF215_TileList* dst, const TF215_TileList* src) {
    dst->size = src->size;
    if (src->size > TPU_C_API_MAX_INLINED) {
        dst->heap = new TF215_Tile[src->size];
        for (int64_t i = 0; i < src->size; ++i) DeepCopyTile(&dst->heap[i], &src->heap[i]);
    } else {
        for (int64_t i = 0; i < src->size; ++i) DeepCopyTile(&dst->inlined[i], &src->inlined[i]);
    }
}
static void DeepCopyLayout(TF215_Layout* dst, const TF215_Layout* src) {
    DeepCopyInt64List(&dst->minor_to_major, &src->minor_to_major);
    DeepCopyIntList(&dst->dim_level_types, &src->dim_level_types);
    DeepCopyIntList(&dst->dim_unique, &src->dim_unique);
    DeepCopyIntList(&dst->dim_ordered, &src->dim_ordered);
    DeepCopyTileList(&dst->tiles, &src->tiles);
    dst->index_primitive_type = src->index_primitive_type;
    dst->pointer_primitive_type = src->pointer_primitive_type;
    dst->element_size_in_bits = src->element_size_in_bits;
    dst->memory_space = src->memory_space;
    dst->dynamic_shape_metadata_prefix_bytes = src->dynamic_shape_metadata_prefix_bytes;
}
static void DeepCopyShape(TF215_Shape* dst, const TF215_Shape* src) {
    dst->element_type = src->element_type;
    DeepCopyInt64List(&dst->dimensions, &src->dimensions);
    DeepCopyBoolList(&dst->dynamic_dimensions, &src->dynamic_dimensions);
    dst->ntuple_shapes = src->ntuple_shapes;
    if (src->ntuple_shapes > 0 && src->tuple_shapes != nullptr) {
        dst->tuple_shapes = new TF215_Shape[src->ntuple_shapes];
        for (int i = 0; i < src->ntuple_shapes; ++i) DeepCopyShape(&dst->tuple_shapes[i], &src->tuple_shapes[i]);
    } else { dst->tuple_shapes = nullptr; }
    dst->has_layout = src->has_layout;
    if (src->has_layout) DeepCopyLayout(&dst->layout, &src->layout);
}

// =========================================================================
// 🚀 核心垫片层与实现
// =========================================================================

static PJRT_Api base_api;
static bool base_api_initialized = false;
static std::mutex g_musa_runtime_mu;
static std::mutex g_execute_submit_mu;
static bool g_musa_runtime_registered = false;
static std::atomic<unsigned long long> g_event_destroy_bypass_count{0};
static std::atomic<unsigned long long> g_buffer_destroy_bypass_count{0};
static std::atomic<unsigned long long> g_client_compile_log_count{0};
static std::atomic<unsigned long long> g_buffer_from_host_log_count{0};
static std::atomic<unsigned long long> g_execute_log_count{0};

struct InflightGate {
    std::mutex mu;
    std::condition_variable cv;
    size_t inflight = 0;
};

static InflightGate g_compile_gate;
static InflightGate g_transfer_gate;
static InflightGate g_execute_gate;

static PJRT_Error* WaitForEventViaCallback(PJRT_Event* event);

static int GetPositiveEnvInt(const char* env_name) {
    const char* env = std::getenv(env_name);
    if (env == nullptr || env[0] == '\0') return 0;

    char* end = nullptr;
    long value = std::strtol(env, &end, 10);
    if (end == env || (end != nullptr && *end != '\0') || value <= 0) {
        return 0;
    }
    if (value > std::numeric_limits<int>::max()) {
        return std::numeric_limits<int>::max();
    }
    return static_cast<int>(value);
}

static int GetCompileMaxInflight() {
    // MTGPU compilation uses process-global LLVM options and external toolchain
    // state. Keep runtime transfer/execute paths unconstrained, but avoid
    // concurrent compile requests entering the MUSA backend by default.
    const char* env = std::getenv("MUSA_PJRT_MAX_INFLIGHT_COMPILES");
    if (env == nullptr || env[0] == '\0') return 1;
    return GetPositiveEnvInt("MUSA_PJRT_MAX_INFLIGHT_COMPILES");
}

static double MsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

class ScopedInflightGate {
   public:
    ScopedInflightGate(InflightGate* gate, int max_inflight)
        : gate_(gate), active_(gate != nullptr && max_inflight > 0) {
        if (!active_) return;
        std::unique_lock<std::mutex> lock(gate_->mu);
        gate_->cv.wait(lock, [this, max_inflight]() {
            return gate_->inflight < static_cast<size_t>(max_inflight);
        });
        ++gate_->inflight;
    }

    ~ScopedInflightGate() {
        if (!active_) return;
        {
            std::lock_guard<std::mutex> lock(gate_->mu);
            --gate_->inflight;
        }
        gate_->cv.notify_one();
    }

   private:
    InflightGate* gate_;
    bool active_;
};

using TF_CreateAndSetPjRtCApiClient_Fn =
    void (*)(const char*, TF_Status*, void*, int);

TF_CreateAndSetPjRtCApiClient_Fn ResolveTfCreateAndSetPjRtCApiClient() {
    void* symbol = dlsym(RTLD_DEFAULT, "TF_CreateAndSetPjRtCApiClient");
    if (symbol != nullptr) {
        return reinterpret_cast<TF_CreateAndSetPjRtCApiClient_Fn>(symbol);
    }

#ifdef RTLD_NOLOAD
    constexpr const char* kTensorFlowRuntimeLibs[] = {
        "libtensorflow_framework.so.2",
        "libtensorflow_cc.so.2",
    };
    for (const char* lib_name : kTensorFlowRuntimeLibs) {
        void* runtime_handle =
            dlopen(lib_name, RTLD_LAZY | RTLD_GLOBAL | RTLD_NOLOAD);
        if (runtime_handle == nullptr) {
            continue;
        }
        symbol = dlsym(runtime_handle, "TF_CreateAndSetPjRtCApiClient");
        if (symbol != nullptr) {
            return reinterpret_cast<TF_CreateAndSetPjRtCApiClient_Fn>(symbol);
        }
    }
#endif

    void* process_handle = dlopen(nullptr, RTLD_LAZY | RTLD_GLOBAL);
    if (process_handle == nullptr) {
        return nullptr;
    }

    symbol = dlsym(process_handle, "TF_CreateAndSetPjRtCApiClient");
    return reinterpret_cast<TF_CreateAndSetPjRtCApiClient_Fn>(symbol);
}

bool EnsureMusaRuntimeRegistered(TF_Status* tf_status, bool verbose) {
    std::lock_guard<std::mutex> lock(g_musa_runtime_mu);
    if (g_musa_runtime_registered) {
        if (tf_status) TF_SetStatus(tf_status, TF_OK, "");
        return true;
    }

    auto platform_or = stream_executor::PlatformManager::PlatformWithName("MUSA");
    if (!platform_or.ok()) {
        const std::string msg = std::string("failed to initialize MUSA platform: ") +
                                platform_or.status().ToString();
        if (verbose) fprintf(stderr, "!!!! [MUSA] %s\n", msg.c_str());
        if (tf_status) TF_SetStatus(tf_status, TF_INTERNAL, msg.c_str());
        return false;
    }

    auto create_pjrt_client = ResolveTfCreateAndSetPjRtCApiClient();
    if (!create_pjrt_client) {
        const std::string msg =
            "symbol TF_CreateAndSetPjRtCApiClient not found in current TensorFlow runtime";
        if (verbose) fprintf(stderr, "!!!! [MUSA] %s\n", msg.c_str());
        if (tf_status) TF_SetStatus(tf_status, TF_INTERNAL, msg.c_str());
        return false;
    }

    TF_Status* create_status = TF_NewStatus();
    create_pjrt_client("MUSA", create_status, nullptr, 0);
    if (TF_GetCode(create_status) != TF_OK) {
        const std::string msg = std::string("TF_CreateAndSetPjRtCApiClient failed: ") +
                                TF_Message(create_status);
        if (verbose) fprintf(stderr, "!!!! [MUSA] %s\n", msg.c_str());
        if (tf_status) TF_SetStatus(tf_status, TF_GetCode(create_status), msg.c_str());
        TF_DeleteStatus(create_status);
        return false;
    }
    TF_DeleteStatus(create_status);

    g_musa_runtime_registered = true;
    if (tf_status) TF_SetStatus(tf_status, TF_OK, "");
    if (verbose) {
        printf("✅ [MUSA] Plugin internal device states initialized.\n");
        fflush(stdout);
    }
    return true;
}

// 1. 修正 AddressableMemories 大小错误 (32 -> 40)
PJRT_Error* Proxy_Device_AddressableMemories(PJRT_Device_AddressableMemories_Args* args) {
    if (!args) return nullptr;
    if (args->struct_size == 32) args->struct_size = 40; 
    return base_api.PJRT_Device_AddressableMemories(args);
}

static bool ShouldBypassEventDestroy() {
    const char* env = std::getenv("MUSA_PJRT_BYPASS_EVENT_DESTROY");
    if (env == nullptr || env[0] == '\0') return false;
    return strcmp(env, "0") != 0 &&
           strcmp(env, "false") != 0 &&
           strcmp(env, "False") != 0 &&
           strcmp(env, "FALSE") != 0;
}

static bool ShouldBypassBufferDestroy() {
    const char* env = std::getenv("MUSA_PJRT_BYPASS_BUFFER_DESTROY");
    if (env == nullptr || env[0] == '\0') return false;
    return strcmp(env, "0") != 0 &&
           strcmp(env, "false") != 0 &&
           strcmp(env, "False") != 0 &&
           strcmp(env, "FALSE") != 0;
}

static bool ShouldLogProxyDebug() {
    const char* env = std::getenv("MUSA_PJRT_DEBUG_LOG");
    if (env == nullptr || env[0] == '\0') return false;
    return strcmp(env, "0") != 0 &&
           strcmp(env, "false") != 0 &&
           strcmp(env, "False") != 0 &&
           strcmp(env, "FALSE") != 0;
}

static bool ShouldSerializeExecuteSubmit() {
    const char* env = std::getenv("MUSA_PJRT_SERIALIZE_EXECUTE_SUBMIT");
    if (env == nullptr || env[0] == '\0') return false;
    return strcmp(env, "0") != 0 &&
           strcmp(env, "false") != 0 &&
           strcmp(env, "False") != 0 &&
           strcmp(env, "FALSE") != 0;
}

static bool ShouldWaitEventBeforeDestroy() {
    const char* env = std::getenv("MUSA_PJRT_WAIT_EVENT_BEFORE_DESTROY");
    if (env == nullptr || env[0] == '\0') return false;
    return strcmp(env, "0") != 0 &&
           strcmp(env, "false") != 0 &&
           strcmp(env, "False") != 0 &&
           strcmp(env, "FALSE") != 0;
}

static bool ShouldWaitBufferReadyBeforeDestroy() {
    const char* env = std::getenv("MUSA_PJRT_WAIT_BUFFER_READY_BEFORE_DESTROY");
    if (env == nullptr || env[0] == '\0') return false;
    return strcmp(env, "0") != 0 &&
           strcmp(env, "false") != 0 &&
           strcmp(env, "False") != 0 &&
           strcmp(env, "FALSE") != 0;
}

static bool ShouldForceHostBufferCopy() {
    const char* env = std::getenv("MUSA_PJRT_FORCE_HOST_BUFFER_COPY");
    if (env == nullptr || env[0] == '\0') return true;
    return strcmp(env, "0") != 0 &&
           strcmp(env, "false") != 0 &&
           strcmp(env, "False") != 0 &&
           strcmp(env, "FALSE") != 0;
}

static bool ShouldWaitTransferDoneBeforeReturn() {
    const char* env = std::getenv("MUSA_PJRT_WAIT_TRANSFER_DONE");
    if (env == nullptr || env[0] == '\0') return false;
    return strcmp(env, "0") != 0 &&
           strcmp(env, "false") != 0 &&
           strcmp(env, "False") != 0 &&
           strcmp(env, "FALSE") != 0;
}

static bool ShouldWaitExecuteDoneBeforeReturn() {
    const char* env = std::getenv("MUSA_PJRT_WAIT_EXECUTE_DONE");
    if (env == nullptr || env[0] == '\0') return false;
    return strcmp(env, "0") != 0 &&
           strcmp(env, "false") != 0 &&
           strcmp(env, "False") != 0 &&
           strcmp(env, "FALSE") != 0;
}

PJRT_Error* Proxy_Client_Compile(PJRT_Client_Compile_Args* args) {
    const int max_inflight_compiles = GetCompileMaxInflight();
    unsigned long long log_count =
        g_client_compile_log_count.fetch_add(1, std::memory_order_relaxed) + 1;
    bool should_log = ShouldLogProxyDebug();
    const auto wait_start = std::chrono::steady_clock::now();
    if (should_log) {
        fprintf(stderr,
                "[musa_pjrt] client compile wait: count=%llu max_inflight=%d struct_size=%zu program=%p code_size=%zu options_size=%zu\n",
                log_count,
                max_inflight_compiles,
                args ? static_cast<size_t>(args->struct_size) : 0,
                args ? static_cast<const void*>(args->program) : nullptr,
                (args && args->program) ? static_cast<size_t>(args->program->code_size) : 0,
                args ? static_cast<size_t>(args->compile_options_size) : 0);
        fflush(stderr);
    }
    ScopedInflightGate gate(&g_compile_gate, max_inflight_compiles);
    const double wait_ms = MsSince(wait_start);
    const auto compile_start = std::chrono::steady_clock::now();
    if (should_log) {
        fprintf(stderr,
                "[musa_pjrt] client compile begin: count=%llu max_inflight=%d wait_ms=%.3f struct_size=%zu program=%p code_size=%zu options_size=%zu\n",
                log_count,
                max_inflight_compiles,
                wait_ms,
                args ? static_cast<size_t>(args->struct_size) : 0,
                args ? static_cast<const void*>(args->program) : nullptr,
                (args && args->program) ? static_cast<size_t>(args->program->code_size) : 0,
                args ? static_cast<size_t>(args->compile_options_size) : 0);
        fflush(stderr);
    }
    PJRT_Error* err = base_api.PJRT_Client_Compile(args);
    if (should_log || err != nullptr) {
        fprintf(stderr,
                "[musa_pjrt] client compile returned: count=%llu err=%p executable=%p compile_ms=%.3f total_ms=%.3f\n",
                log_count,
                static_cast<void*>(err),
                (args ? static_cast<void*>(args->executable) : nullptr),
                MsSince(compile_start),
                MsSince(wait_start));
        fflush(stderr);
    }
    return err;
}

PJRT_Error* Proxy_Client_BufferFromHostBuffer(PJRT_Client_BufferFromHostBuffer_Args* args) {
    ScopedInflightGate gate(
        &g_transfer_gate,
        GetPositiveEnvInt("MUSA_PJRT_MAX_INFLIGHT_TRANSFERS"));
    unsigned long long log_count =
        g_buffer_from_host_log_count.fetch_add(1, std::memory_order_relaxed) + 1;
    bool should_log = ShouldLogProxyDebug();
    if (should_log) {
        fprintf(stderr,
                "[musa_pjrt] buffer-from-host begin: count=%llu struct_size=%zu data=%p type=%d num_dims=%zu semantics=%d device=%p\n",
                log_count,
                args ? static_cast<size_t>(args->struct_size) : 0,
                args ? args->data : nullptr,
                args ? static_cast<int>(args->type) : -1,
                args ? static_cast<size_t>(args->num_dims) : 0,
                args ? static_cast<int>(args->host_buffer_semantics) : -1,
                args ? static_cast<void*>(args->device) : nullptr);
        fflush(stderr);
    }

    PJRT_HostBufferSemantics original_semantics =
        args ? args->host_buffer_semantics
             : PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
    if (args != nullptr && ShouldForceHostBufferCopy()) {
        args->host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    }

    const auto transfer_start = std::chrono::steady_clock::now();
    PJRT_Error* err = base_api.PJRT_Client_BufferFromHostBuffer(args);
    if (args != nullptr) {
        args->host_buffer_semantics = original_semantics;
    }

    if (err == nullptr && args != nullptr && args->done_with_host_buffer != nullptr &&
        ShouldWaitTransferDoneBeforeReturn()) {
        PJRT_Error* wait_err = WaitForEventViaCallback(args->done_with_host_buffer);
        if (wait_err != nullptr && ShouldLogProxyDebug()) {
            fprintf(stderr,
                    "[musa_pjrt] buffer-from-host wait returned error: count=%llu err=%p done_event=%p\n",
                    log_count,
                    static_cast<void*>(wait_err),
                    static_cast<void*>(args->done_with_host_buffer));
            fflush(stderr);
        }
    }

    if (should_log || err != nullptr) {
        fprintf(stderr,
                "[musa_pjrt] buffer-from-host returned: count=%llu err=%p done_event=%p buffer=%p force_copy=%d wait_done=%d transfer_ms=%.3f\n",
                log_count,
                static_cast<void*>(err),
                args ? static_cast<void*>(args->done_with_host_buffer) : nullptr,
                args ? static_cast<void*>(args->buffer) : nullptr,
                ShouldForceHostBufferCopy() ? 1 : 0,
                ShouldWaitTransferDoneBeforeReturn() ? 1 : 0,
                MsSince(transfer_start));
        fflush(stderr);
    }
    return err;
}

PJRT_Error* Proxy_Event_Destroy(PJRT_Event_Destroy_Args* args) {
    if (args == nullptr || args->event == nullptr) {
        return nullptr;
    }
    if (ShouldBypassEventDestroy()) {
        const char* bypass_env = std::getenv("MUSA_PJRT_BYPASS_EVENT_DESTROY");
        const char* bypass_env_text =
            (bypass_env != nullptr && bypass_env[0] != '\0') ? bypass_env : "<unset>";
        unsigned long long bypass_count =
            g_event_destroy_bypass_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (ShouldLogProxyDebug() &&
            (bypass_count <= 4 || (bypass_count % 100000) == 0)) {
            fprintf(stderr,
                    "[musa_pjrt] event destroy bypassed: count=%llu event=%p env=%s\n",
                    bypass_count,
                    static_cast<void*>(args->event),
                    bypass_env_text);
            fflush(stderr);
        }
        return nullptr;
    }
    if (ShouldWaitEventBeforeDestroy()) {
        PJRT_Error* wait_err = WaitForEventViaCallback(args->event);
        if (wait_err != nullptr && ShouldLogProxyDebug()) {
            fprintf(stderr,
                    "[musa_pjrt] event destroy wait returned error: err=%p event=%p\n",
                    static_cast<void*>(wait_err),
                    static_cast<void*>(args->event));
            fflush(stderr);
        }
    }
    return base_api.PJRT_Event_Destroy(args);
}

PJRT_Error* Proxy_Buffer_Destroy(PJRT_Buffer_Destroy_Args* args) {
    if (args == nullptr || args->buffer == nullptr) {
        return nullptr;
    }
    if (ShouldBypassBufferDestroy()) {
        unsigned long long bypass_count =
            g_buffer_destroy_bypass_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (ShouldLogProxyDebug() &&
            (bypass_count <= 4 || (bypass_count % 100000) == 0)) {
            fprintf(stderr,
                    "[musa_pjrt] buffer destroy bypassed: count=%llu buffer=%p\n",
                    bypass_count,
                    static_cast<void*>(args->buffer));
            fflush(stderr);
        }
        return nullptr;
    }

    if (ShouldWaitBufferReadyBeforeDestroy()) {
        PJRT_Buffer_ReadyEvent_Args ready_args;
        memset(&ready_args, 0, sizeof(ready_args));
        ready_args.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
        ready_args.buffer = args->buffer;
        PJRT_Error* ready_err = base_api.PJRT_Buffer_ReadyEvent(&ready_args);
        if (ready_err == nullptr && ready_args.event != nullptr) {
            PJRT_Error* wait_err = WaitForEventViaCallback(ready_args.event);
            if (wait_err != nullptr && ShouldLogProxyDebug()) {
                fprintf(stderr,
                        "[musa_pjrt] buffer destroy wait returned error: err=%p buffer=%p ready_event=%p\n",
                        static_cast<void*>(wait_err),
                        static_cast<void*>(args->buffer),
                        static_cast<void*>(ready_args.event));
                fflush(stderr);
            }

            PJRT_Event_Destroy_Args destroy_ready_event_args;
            memset(&destroy_ready_event_args, 0, sizeof(destroy_ready_event_args));
            destroy_ready_event_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
            destroy_ready_event_args.event = ready_args.event;
            PJRT_Error* destroy_ready_event_err =
                base_api.PJRT_Event_Destroy(&destroy_ready_event_args);
            if (destroy_ready_event_err != nullptr && ShouldLogProxyDebug()) {
                fprintf(stderr,
                        "[musa_pjrt] buffer destroy ready-event cleanup error: err=%p buffer=%p ready_event=%p\n",
                        static_cast<void*>(destroy_ready_event_err),
                        static_cast<void*>(args->buffer),
                        static_cast<void*>(ready_args.event));
                fflush(stderr);
            }
        } else if (ready_err != nullptr && ShouldLogProxyDebug()) {
            fprintf(stderr,
                    "[musa_pjrt] buffer ready-event query error: err=%p buffer=%p\n",
                    static_cast<void*>(ready_err),
                    static_cast<void*>(args->buffer));
            fflush(stderr);
        }
    }

    return base_api.PJRT_Buffer_Destroy(args);
}

struct BlockingEventState {
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    PJRT_Error* error = nullptr;
};

static void OnReadyBlockCallback(PJRT_Error* error, void* user_arg) {
    auto* state = static_cast<BlockingEventState*>(user_arg);
    {
        std::lock_guard<std::mutex> lock(state->mu);
        state->done = true;
        state->error = error;
    }
    state->cv.notify_one();
}

static PJRT_Error* WaitForEventViaCallback(PJRT_Event* event) {
    if (event == nullptr) return nullptr;

    BlockingEventState state;
    PJRT_Event_OnReady_Args onready_args;
    memset(&onready_args, 0, sizeof(onready_args));
    onready_args.struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE;
    onready_args.event = event;
    onready_args.callback = OnReadyBlockCallback;
    onready_args.user_arg = &state;

    PJRT_Error* onready_err = base_api.PJRT_Event_OnReady(&onready_args);
    if (onready_err != nullptr) {
        return onready_err;
    }

    std::unique_lock<std::mutex> lock(state.mu);
    state.cv.wait(lock, [&state]() { return state.done; });
    return state.error;
}

static PJRT_Error* GetLoadedExecutableNumOutputs(
    PJRT_LoadedExecutable* loaded_executable,
    size_t* num_outputs) {
    if (loaded_executable == nullptr || num_outputs == nullptr) return nullptr;

    PJRT_LoadedExecutable_GetExecutable_Args get_exec_args;
    memset(&get_exec_args, 0, sizeof(get_exec_args));
    get_exec_args.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
    get_exec_args.loaded_executable = loaded_executable;
    PJRT_Error* get_exec_err = base_api.PJRT_LoadedExecutable_GetExecutable(&get_exec_args);
    if (get_exec_err != nullptr) {
        return get_exec_err;
    }

    PJRT_Executable_NumOutputs_Args num_outputs_args;
    memset(&num_outputs_args, 0, sizeof(num_outputs_args));
    num_outputs_args.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
    num_outputs_args.executable = get_exec_args.executable;
    PJRT_Error* num_outputs_err = base_api.PJRT_Executable_NumOutputs(&num_outputs_args);

    PJRT_Executable_Destroy_Args destroy_exec_args;
    memset(&destroy_exec_args, 0, sizeof(destroy_exec_args));
    destroy_exec_args.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
    destroy_exec_args.executable = get_exec_args.executable;
    PJRT_Error* destroy_exec_err = base_api.PJRT_Executable_Destroy(&destroy_exec_args);

    if (num_outputs_err != nullptr) {
        return num_outputs_err;
    }
    if (destroy_exec_err != nullptr) {
        return destroy_exec_err;
    }

    *num_outputs = num_outputs_args.num_outputs;
    return nullptr;
}

// 2. 修正 ExecuteOptions 大小错误 (52 -> 88)
// 2. 修正 ExecuteOptions 大小错误 (52 -> 88)
PJRT_Error* Proxy_LoadedExecutable_Execute(PJRT_LoadedExecutable_Execute_Args* args) {
    if (!args || !args->options) return base_api.PJRT_LoadedExecutable_Execute(args);

    ScopedInflightGate gate(
        &g_execute_gate,
        GetPositiveEnvInt("MUSA_PJRT_MAX_INFLIGHT_EXECUTES"));

    PJRT_ExecuteOptions* original_options = args->options;
    unsigned long long log_count =
        g_execute_log_count.fetch_add(1, std::memory_order_relaxed) + 1;
    bool should_log = ShouldLogProxyDebug();

    if (should_log) {
        fprintf(stderr,
                "[musa_pjrt] execute begin: count=%llu args_size=%zu options_size=%zu num_devices=%zu events=%p execute_device=%p serialize_submit=%d wait_done=%d\n",
                log_count,
                static_cast<size_t>(args->struct_size),
                static_cast<size_t>(original_options->struct_size),
                static_cast<size_t>(args->num_devices),
                static_cast<void*>(args->device_complete_events),
                static_cast<void*>(args->execute_device),
                ShouldSerializeExecuteSubmit() ? 1 : 0,
                ShouldWaitExecuteDoneBeforeReturn() ? 1 : 0);
        fflush(stderr);
    }

    PJRT_Error* err = nullptr;
    std::unique_ptr<char[]> proxy_options_buf;
    const auto execute_start = std::chrono::steady_clock::now();

    if (original_options->struct_size == 52) {
        // 1. 修复：堆分配保命，防止 options 被毁
        proxy_options_buf.reset(new char[88]);
        memset(proxy_options_buf.get(), 0, 88);
        memcpy(proxy_options_buf.get(), original_options, 52);
        reinterpret_cast<PJRT_ExecuteOptions*>(proxy_options_buf.get())->struct_size = 88;
        args->options = reinterpret_cast<PJRT_ExecuteOptions*>(proxy_options_buf.get());

        if (ShouldSerializeExecuteSubmit()) {
            std::lock_guard<std::mutex> submit_lock(g_execute_submit_mu);
            err = base_api.PJRT_LoadedExecutable_Execute(args);
        } else {
            err = base_api.PJRT_LoadedExecutable_Execute(args);
        }
    }

    if (original_options->struct_size != 52) {
        if (ShouldSerializeExecuteSubmit()) {
            std::lock_guard<std::mutex> submit_lock(g_execute_submit_mu);
            err = base_api.PJRT_LoadedExecutable_Execute(args);
        } else {
            err = base_api.PJRT_LoadedExecutable_Execute(args);
        }
    }

    if (err == nullptr && args->device_complete_events != nullptr &&
        ShouldWaitExecuteDoneBeforeReturn()) {
        for (size_t device_index = 0; device_index < args->num_devices; ++device_index) {
            PJRT_Event* event = args->device_complete_events[device_index];
            if (event == nullptr) {
                continue;
            }
            PJRT_Error* wait_err = WaitForEventViaCallback(event);
            if (wait_err != nullptr) {
                err = wait_err;
                if (ShouldLogProxyDebug()) {
                    fprintf(stderr,
                            "[musa_pjrt] execute wait returned error: count=%llu err=%p device_index=%zu event=%p\n",
                            log_count,
                            static_cast<void*>(wait_err),
                            device_index,
                            static_cast<void*>(event));
                    fflush(stderr);
                }
                break;
            }
        }
    }

    args->options = original_options;

    if (should_log || err != nullptr) {
        fprintf(stderr,
                "[musa_pjrt] execute returned: count=%llu err=%p events=%p first_event=%p execute_ms=%.3f\n",
                log_count,
                static_cast<void*>(err),
                static_cast<void*>(args->device_complete_events),
                (args->device_complete_events != nullptr && args->num_devices > 0)
                    ? static_cast<void*>(args->device_complete_events[0])
                    : nullptr,
                MsSince(execute_start));
        fflush(stderr);
    }

    return err;
}

void Musa_XlaShapeToDeviceShapeRepresentation(
    XLA_Shape* c_xla_shape, int data_type, bool use_fast_memory,
    XLA_LayoutPreference layout_preference, XLA_Shape* c_device_shape, TF_Status* status) {
    if (c_xla_shape && c_device_shape) {
        auto* src = reinterpret_cast<const TF215_Shape*>(c_xla_shape);
        auto* dst = reinterpret_cast<TF215_Shape*>(c_device_shape);
        DeepCopyShape(dst, src);
    }
    if (status) TF_SetStatus(status, TF_OK, "");
}

int32_t Musa_GetDeviceCount(TF_Status* status) {
    if (status) TF_SetStatus(status, TF_OK, "");
    return 8;
}

void Musa_InitPluginInternalDeviceStates(TF_Status* status) {
    EnsureMusaRuntimeRegistered(status, true);
}

static bool ReadBoolEnv(const char* name, bool* value) {
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') return false;

    std::string text(env);
    if (text == "1" || text == "true" || text == "TRUE" ||
        text == "True" || text == "yes" || text == "YES" ||
        text == "on" || text == "ON") {
        *value = true;
        return true;
    }
    if (text == "0" || text == "false" || text == "FALSE" ||
        text == "False" || text == "no" || text == "NO" ||
        text == "off" || text == "OFF") {
        *value = false;
        return true;
    }

    fprintf(stderr, "[MUSA PJRT] ignoring invalid %s=%s\n", name, env);
    return false;
}

static bool ReadDoubleEnv(const char* name, double* value) {
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') return false;

    char* end = nullptr;
    double parsed = std::strtod(env, &end);
    if (end == env || *end != '\0' || parsed <= 0.0) {
        fprintf(stderr, "[MUSA PJRT] ignoring invalid %s=%s\n", name, env);
        return false;
    }

    *value = parsed;
    return true;
}

static bool ReadInt64Env(const char* name, int64_t* value) {
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') return false;

    char* end = nullptr;
    long long parsed = std::strtoll(env, &end, 10);
    if (end == env || *end != '\0' || parsed <= 0) {
        fprintf(stderr, "[MUSA PJRT] ignoring invalid %s=%s\n", name, env);
        return false;
    }

    *value = static_cast<int64_t>(parsed);
    return true;
}

static void ApplyMusaAllocatorEnv(xla::GpuClientOptions* options) {
    // Keep execution concurrency intact. These knobs only control the device
    // memory pool reservation policy used by the PJRT GPU client.
    options->allocator_config.preallocate = false;

    bool preallocate = false;
    if (ReadBoolEnv("MUSA_PJRT_PREALLOCATE", &preallocate)) {
        options->allocator_config.preallocate = preallocate;
    }

    double memory_fraction = 0.0;
    if (ReadDoubleEnv("MUSA_PJRT_MEMORY_FRACTION", &memory_fraction)) {
        options->allocator_config.memory_fraction = memory_fraction;
    }

    int64_t system_memory_mb = 0;
    if (ReadInt64Env("MUSA_PJRT_GPU_SYSTEM_MEMORY_SIZE_MB",
                     &system_memory_mb)) {
        constexpr int64_t kMiB = 1024LL * 1024LL;
        if (system_memory_mb <= std::numeric_limits<int64_t>::max() / kMiB) {
            options->allocator_config.gpu_system_memory_size =
                system_memory_mb * kMiB;
        } else {
            fprintf(stderr,
                    "[MUSA PJRT] ignoring too large "
                    "MUSA_PJRT_GPU_SYSTEM_MEMORY_SIZE_MB=%lld\n",
                    static_cast<long long>(system_memory_mb));
        }
    }

    fprintf(stderr,
            "[MUSA PJRT] allocator preallocate=%s memory_fraction=%.3f",
            options->allocator_config.preallocate ? "true" : "false",
            options->allocator_config.memory_fraction);
    if (options->allocator_config.gpu_system_memory_size.has_value()) {
        fprintf(stderr, " gpu_system_memory_size=%lld",
                static_cast<long long>(
                    options->allocator_config.gpu_system_memory_size.value()));
    }
    fprintf(stderr, "\n");
    fflush(stderr);
}

PJRT_Error* Musa_Client_Create(PJRT_Client_Create_Args* args) {
    xla::GpuClientOptions options;
    options.platform_name = "MUSA"; 
    ApplyMusaAllocatorEnv(&options);
    auto client_or = xla::GetStreamExecutorGpuClient(options);
    if (!client_or.ok()) {
        fprintf(stderr, "🚨 MUSA Init Failed: %s\n", client_or.status().ToString().c_str());
        abort();
    }
    args->client = pjrt::CreateWrapperClient(std::move(client_or.value()));
    printf("✅ MUSA Client Created.\n"); fflush(stdout);
    return nullptr;
}

// =========================================================================
// 🎯 发现钩子 (API Mounting)
// =========================================================================

__attribute__((visibility("default"))) const PJRT_Api* GetPjrtApi() {
    if (!base_api_initialized) {
        base_api = pjrt::CreatePjrtApi(Musa_Client_Create, nullptr, nullptr, pjrt::PJRT_Plugin_Initialize_NoOp);
        base_api_initialized = true;
    }
    
    static PJRT_Api* truncated_api = nullptr;
    if (!truncated_api) {
        truncated_api = (PJRT_Api*)malloc(792);
        memset(truncated_api, 0, 792);
        size_t local_api_size = sizeof(PJRT_Api);
        size_t copy_size = (local_api_size < 792) ? local_api_size : 792;
        memcpy(truncated_api, &base_api, copy_size);
        truncated_api->struct_size = 792; 

        // 挂载补丁代理函数
        truncated_api->PJRT_Device_AddressableMemories = Proxy_Device_AddressableMemories;
        truncated_api->PJRT_Client_Compile = Proxy_Client_Compile;
        truncated_api->PJRT_Client_BufferFromHostBuffer = Proxy_Client_BufferFromHostBuffer;
        truncated_api->PJRT_LoadedExecutable_Execute = Proxy_LoadedExecutable_Execute;
        truncated_api->PJRT_Event_Destroy = Proxy_Event_Destroy;
        truncated_api->PJRT_Buffer_Destroy = Proxy_Buffer_Destroy;
    }
    return truncated_api;
}

__attribute__((visibility("default"))) const TFNPD_Api* TFNPD_InitPlugin(TFNPD_PluginParams* params, TF_Status* tf_status) {
    char* mem = reinterpret_cast<char*>(params);
    void* ext_val = nullptr; memcpy(mem + 8, &ext_val, sizeof(void*));
    const char* dev_type = "MUSA"; memcpy(mem + 16, &dev_type, sizeof(const char*));
    const char* comp_dev = "XLA_GPU_JIT"; memcpy(mem + 24, &comp_dev, sizeof(const char*));
    int priority_val = 1000; memcpy(mem + 32, &priority_val, sizeof(int));
    bool is_pluggable = false; memcpy(mem + 36, &is_pluggable, sizeof(bool));
    bool use_pjrt = true; memcpy(mem + 37, &use_pjrt, sizeof(bool));

    static TFNPD_Api npd_api = {};
    npd_api.struct_size = TFNPD_Api_STRUCT_SIZE;
    npd_api.TFNPD_XlaShapeToDeviceShapeRepresentation = Musa_XlaShapeToDeviceShapeRepresentation;
    npd_api.TFNPD_GetDeviceCount = Musa_GetDeviceCount;
    npd_api.TFNPD_InitPluginInternalDeviceStates = Musa_InitPluginInternalDeviceStates;

    if (tf_status) TF_SetStatus(tf_status, TF_OK, "");
    return &npd_api;
}

__attribute__((visibility("default"))) void ForceRegisterMusa() {
    TF_Status* status = TF_NewStatus();
    if (EnsureMusaRuntimeRegistered(status, true) && TF_GetCode(status) == TF_OK) {
        printf("✅ [PYTHON HOOK] MUSA PJRT Factory Registered!\n");
    }
    TF_DeleteStatus(status);
    fflush(stdout);
}

} // extern "C"
