#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/plugin_c_api.h"
#include <iostream>
#include <memory>
#include <cstring>
#include <dlfcn.h>

extern "C" {

int32_t Musa_GetDeviceCount(TF_Status* status) {
    printf("[DEBUG-1] Musa_GetDeviceCount called.\n"); fflush(stdout);
    if (status) TF_SetStatus(status, TF_OK, "");
    return 8;
}

void Musa_InitPluginInternalDeviceStates(TF_Status* status) {
    printf("[DEBUG-2] Musa_InitPluginInternalDeviceStates called.\n"); fflush(stdout);
    if (status) TF_SetStatus(status, TF_OK, "");
}

void Musa_XlaShapeToDeviceShapeRepresentation(
    XLA_Shape* c_xla_shape, int data_type, bool use_fast_memory,
    XLA_LayoutPreference layout_preference, XLA_Shape* c_device_shape, TF_Status* status) {
    if (c_xla_shape && c_device_shape) *c_device_shape = *c_xla_shape;
    if (status) TF_SetStatus(status, TF_OK, "");
}

// 核心创建函数：之前这个函数从未被 TF 真正调用过！
PJRT_Error* Musa_Client_Create(PJRT_Client_Create_Args* args) {
    printf("\n[DEBUG-3] ---> Musa_Client_Create INVOKED BY TF! <----------\n"); fflush(stdout);
    
    xla::GpuClientOptions options;
    options.platform_name = "MUSA";
    auto client_or = xla::GetStreamExecutorGpuClient(options);
    
    if (!client_or.ok()) {
        printf("[DEBUG-3.1] StreamExecutor INIT FAILED: %s\n", client_or.status().ToString().c_str());
        fflush(stdout);
        abort();
    }
    
    args->client = pjrt::CreateWrapperClient(std::move(client_or.value()));
    printf("[DEBUG-3.2] Client successfully created and wrapped for TF!\n\n"); fflush(stdout);
    return nullptr;
}

__attribute__((visibility("default"))) const PJRT_Api* GetPjrtApi() {
    printf("[DEBUG-4] GetPjrtApi called by TF.\n"); fflush(stdout);
    static PJRT_Api base_api = pjrt::CreatePjrtApi(Musa_Client_Create, nullptr, nullptr, pjrt::PJRT_Plugin_Initialize_NoOp);
    static PJRT_Api* truncated_api = nullptr;
    if (!truncated_api) {
        truncated_api = (PJRT_Api*)malloc(792);
        memcpy(truncated_api, &base_api, 792);
        truncated_api->struct_size = 792;
        printf("[DEBUG-4.1] GetPjrtApi: struct size forced to 792.\n"); fflush(stdout);
    }
    return truncated_api;
}

__attribute__((visibility("default"))) const TFNPD_Api* TFNPD_InitPlugin(TFNPD_PluginParams* params, TF_Status* tf_status) {
    printf("\n[DEBUG-5] TFNPD_InitPlugin called by TF.\n"); fflush(stdout);
    
    char* mem = reinterpret_cast<char*>(params);
    void* ext_val = nullptr;
    memcpy(mem + 8, &ext_val, sizeof(void*));
    const char* dev_type = "MUSA";
    memcpy(mem + 16, &dev_type, sizeof(const char*));
    const char* comp_dev = "XLA_GPU_JIT";
    memcpy(mem + 24, &comp_dev, sizeof(const char*));
    int priority_val = 1000;
    memcpy(mem + 32, &priority_val, sizeof(int));
    bool is_pluggable = true;
    memcpy(mem + 36, &is_pluggable, sizeof(bool));
    bool use_pjrt = true;
    memcpy(mem + 37, &use_pjrt, sizeof(bool));

    static TFNPD_Api npd_api = {};
    npd_api.struct_size = TFNPD_Api_STRUCT_SIZE;
    npd_api.TFNPD_XlaShapeToDeviceShapeRepresentation = Musa_XlaShapeToDeviceShapeRepresentation;
    npd_api.TFNPD_GetDeviceCount = Musa_GetDeviceCount;
    npd_api.TFNPD_InitPluginInternalDeviceStates = Musa_InitPluginInternalDeviceStates;

    if (tf_status) TF_SetStatus(tf_status, TF_OK, "");
    printf("[DEBUG-5.1] TFNPD_InitPlugin returning OK.\n"); fflush(stdout);
    return &npd_api;
}

// =========================================================================
// 🔥 终极神兵：暴露给 Python 的手动注册钩子（消除所有多线程时序问题）
// =========================================================================
// =========================================================================
// 🔥 终极神兵：暴露给 Python 的手动注册钩子（修复了 4 参数签名！）
// =========================================================================
__attribute__((visibility("default"))) void ForceRegisterMusa() {
    printf("\n[DEBUG-6] ForceRegisterMusa explicitly called from Python!\n"); fflush(stdout);
    
    void* handle = dlopen("libtensorflow_cc.so.2", RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) handle = dlopen(NULL, RTLD_LAZY);
    
    if (handle) {
        // 🔥 修复点：恢复为官方的 4 参数签名！
        typedef void (*TF_CreateAndSetPjRtCApiClient_Fn)(const char*, TF_Status*, void*, int);
        auto create_pjrt_client = (TF_CreateAndSetPjRtCApiClient_Fn)dlsym(handle, "TF_CreateAndSetPjRtCApiClient");
        
        if (create_pjrt_client) {
            TF_Status* status = TF_NewStatus();
            printf("[DEBUG-6.1] Calling TF_CreateAndSetPjRtCApiClient(\"MUSA\", nullptr, 0)...\n"); fflush(stdout);
            
            // 传入 MUSA, status, 空的 options 指针, 选项数量为 0
            create_pjrt_client("MUSA", status, nullptr, 0);
            
            if (TF_GetCode(status) != TF_OK) {
                printf("[DEBUG-6.2] 💥 FATAL: TF_CreateAndSetPjRtCApiClient FAILED: %s\n", TF_Message(status));
            } else {
                printf("[DEBUG-6.3] ✅ SUCCESS: TF_CreateAndSetPjRtCApiClient returned OK!\n");
            }
            TF_DeleteStatus(status);
        } else {
            printf("[DEBUG-6.4] FAILED to dlsym TF_CreateAndSetPjRtCApiClient\n");
        }
    } else {
        printf("[DEBUG-6.5] FAILED to dlopen libtensorflow_cc.so.2\n");
    }
    fflush(stdout);
}

} // extern "C"