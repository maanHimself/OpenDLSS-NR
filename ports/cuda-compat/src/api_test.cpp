#include "nr_cuda.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
void checked(int status) {
    if (status)
        throw std::runtime_error(nr_error());
}
void cuda_checked(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
struct Context {
    void *value = nullptr;
    ~Context() {
        if (value)
            nr_destroy(value);
    }
    void close() {
        void *p = value;
        value = nullptr;
        checked(nr_destroy(p));
    }
};
} // namespace

int main(int argc, char **argv) {
    try {
        const bool host_only = argc == 2 && std::string(argv[1]) == "--host-only";
        require(argc == 1 || host_only, "Usage: nr-api-test [--host-only]");
        require(nr_abi_version() == NR_CUDA_ABI_VERSION, "ABI version");
        require(std::strlen(nr_source_sha256()) == 64, "Source identity");
        require(std::strstr(nr_build_info(), "browser-compat-v1"), "Build identity");
        void *invalid = reinterpret_cast<void *>(1);
        require(nr_create(nullptr, 6, 1) != 0, "Null context output accepted");
        for (int policy : {0, 1, 2, 3, 4, 5, 7, -1, 8}) {
            require(nr_create(&invalid, policy, 1) != 0 && invalid == nullptr,
                    "Non-public policy accepted");
        }
        require(nr_create(&invalid, 6, 2) != 0 && invalid == nullptr,
                "Invalid window choice accepted");
        require(nr_sync(nullptr) != 0 && nr_check_context(nullptr) != 0, "Null context accepted");
        checked(nr_destroy(nullptr));
        if (host_only) {
            std::cout << "{\"status\":\"pass\",\"gpu_executed\":false,\"abi_version\":1}\n";
            return 0;
        }
        Context context;
        checked(nr_create(&context.value, 6, 1));
        int ordinal = -1, major = 0, minor = 0, policy = 0, fused = 0;
        checked(nr_device_info(context.value, &ordinal, &major, &minor));
        checked(nr_context_options(context.value, &policy, &fused));
        require(major == 7 && minor == 5 && policy == 6 && fused == 1,
                "Creation identity mismatch");
        size_t before = 0, after = 0;
        checked(nr_allocated(context.value, &before));
        void *input = nullptr, *half = nullptr, *output = nullptr;
        checked(nr_alloc(context.value, &input, 16 * sizeof(float)));
        checked(nr_alloc(context.value, &half, 16 * sizeof(nr_half)));
        checked(nr_alloc(context.value, &output, 16 * sizeof(float)));
        checked(nr_allocated(context.value, &after));
        require(after - before == 16 * (2 * sizeof(float) + sizeof(nr_half)),
                "Allocation accounting");
        float values[16], result[16];
        for (int i = 0; i < 16; ++i)
            values[i] = (i - 8) * 0.125f;
        checked(nr_upload(context.value, input, values, sizeof(values)));
        checked(nr_capture_begin(context.value));
        checked(nr_convert(context.value, static_cast<float *>(input), static_cast<nr_half *>(half),
                           16));
        checked(nr_expand(context.value, static_cast<nr_half *>(half), static_cast<float *>(output),
                          16));
        checked(nr_capture_end(context.value));
        float elapsed = -1;
        for (int i = 0; i < 3; ++i) {
            checked(nr_replay(context.value, &elapsed));
            require(std::isfinite(elapsed) && elapsed >= 0, "Invalid CUDA timing");
            checked(nr_download(context.value, result, output, sizeof(result)));
            require(std::memcmp(result, values, sizeof(values)) == 0,
                    "Captured conversion mismatch");
        }
        int rejected_sync = 0, rejected_destroy = 0;
        std::thread other([&] {
            rejected_sync = nr_sync(context.value);
            rejected_destroy = nr_destroy(context.value);
        });
        other.join();
        require(rejected_sync != 0 && rejected_destroy != 0, "Cross-thread context accepted");
        checked(nr_sync(context.value)); // Failed ownership checks did not consume it.
        int devices = 0;
        cuda_checked(cudaGetDeviceCount(&devices));
        if (devices > 1) {
            cuda_checked(cudaSetDevice((ordinal + 1) % devices));
            const int rejected = nr_sync(context.value);
            const int retained = nr_destroy(context.value);
            cuda_checked(cudaSetDevice(ordinal));
            require(rejected != 0 && retained != 0, "Wrong current device accepted");
            checked(nr_sync(context.value));
        }
        context.close();
        // Closing an active capture must abort it and permit a clean new context.
        checked(nr_create(&context.value, 6, 0));
        checked(nr_capture_begin(context.value));
        require(nr_vit(context.value, nullptr, nullptr, nullptr, 0, 1) != 0,
                "Invalid ViT shape accepted");
        context.close();
        checked(nr_create(&context.value, 6, 1));
        checked(nr_sync(context.value));
        context.close();
        // Synchronizing a capturing stream is prohibited by CUDA and invalidates
        // capture. Cover both explicit end and destruction of that invalid state.
        for (bool end_first : {false, true}) {
            checked(nr_create(&context.value, 6, 1));
            checked(nr_capture_begin(context.value));
            require(nr_sync(context.value) != 0 &&
                        std::strcmp(nr_error(), cudaGetErrorString(cudaErrorStreamCaptureUnsupported)) == 0,
                    "Captured synchronization did not fail through CUDA");
            if (end_first)
                require(nr_capture_end(context.value) != 0 &&
                            std::strcmp(nr_error(), cudaGetErrorString(cudaErrorStreamCaptureInvalidated)) == 0,
                        "Capture end did not report invalidation");
            context.close();
            checked(nr_create(&context.value, 6, 1));
            checked(nr_sync(context.value));
            context.close();
        }
        std::cout << "{\"status\":\"pass\",\"gpu_executed\":true,\"abi_version\":1,\"thread_"
                     "rejection\":true,\"device_switch_tested\":"
                  << (devices > 1 ? "true" : "false") << ",\"capture_abort_and_reopen\":true}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "API TEST FAILED: " << error.what() << '\n';
        return 1;
    }
}
