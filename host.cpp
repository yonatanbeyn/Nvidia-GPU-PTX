//
// host.cpp — minimal CUDA Driver API host program.
//
// Loads hello.ptx at runtime, launches the `hello` kernel with 32 threads,
// and prints the array each thread wrote to.
//
// We use the *Driver API* (cuModuleLoadData) rather than the Runtime API
// (cudaLaunchKernel) because the driver API can load raw PTX from disk —
// exactly what we want when writing PTX by hand.
//

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(call)                                                          \
    do {                                                                     \
        CUresult _r = (call);                                                \
        if (_r != CUDA_SUCCESS) {                                            \
            const char* _s = "?";                                            \
            cuGetErrorString(_r, &_s);                                       \
            fprintf(stderr, "CUDA error: %s -> %s (line %d)\n",              \
                    #call, _s, __LINE__);                                    \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

int main(void)
{
    // ---- 1. Initialize driver, pick device 0, attach to the primary context ----
    //         (cuDevicePrimaryCtxRetain is the modern path — same context the
    //          Runtime API uses, and stable across CUDA driver-API revisions.)
    CHECK(cuInit(0));
    CUdevice  dev;
    CUcontext ctx;
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK(cuCtxSetCurrent(ctx));

    // ---- 2. Load the cubin (already produced offline by `ptxas`),
    //         get the kernel function handle. ----
    CUmodule   mod;
    CUfunction fn;
    CHECK(cuModuleLoad(&mod, "hello.cubin"));
    CHECK(cuModuleGetFunction(&fn, mod, "hello"));

    // ---- 4. Allocate the output array on the GPU ----
    const int N = 32;
    CUdeviceptr d_out;
    CHECK(cuMemAlloc(&d_out, N * sizeof(int)));

    // ---- 5. Launch: 1 block, N threads. Pass &d_out as the only kernel arg. ----
    void* args[] = { &d_out };
    CHECK(cuLaunchKernel(fn,
                         /* grid  */ 1, 1, 1,
                         /* block */ N, 1, 1,
                         /* shared bytes */ 0,
                         /* stream       */ 0,
                         args, NULL));
    CHECK(cuCtxSynchronize());

    // ---- 6. Copy back, print, cleanup ----
    int h_out[N];
    CHECK(cuMemcpyDtoH(h_out, d_out, N * sizeof(int)));

    printf("Output from PTX kernel running on sm_89:\n");
    for (int i = 0; i < N; ++i) printf("%d ", h_out[i]);
    printf("\n");

    cuMemFree(d_out);
    cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);
    return 0;
}