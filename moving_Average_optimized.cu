// moving_Average_optimized.cu
//
// Optimized version of moving_Average.cu. The original file is left untouched;
// see README.md for a side-by-side explanation of what changed and why.
//
// Build:  nvcc -O3 -std=c++11 -o moving_average_optimized moving_Average_optimized.cu
// Run:    ./moving_average_optimized

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

using std::vector;

#define SIZE 10000          // number of input samples
#define N 10                // moving-average window length
#define BLOCK_SIZE 256      // threads per block (multiple of the 32-wide warp)

// Report and exit on any CUDA runtime error. Unlike the original macros this
// also prints the error string and covers *every* runtime call, including the
// kernel launch (via cudaGetLastError).
#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                                \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Sum a value across the 32 threads of a warp using register shuffles.
// After the loop lane 0 holds the warp total.
__device__ __forceinline__ float warpReduceSum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

// Sum a value across the whole block. Thread 0 receives the block total.
// `scratch` must hold at least BLOCK_SIZE / 32 floats.
__device__ __forceinline__ float blockReduceSum(float v, float *scratch) {
    const int lane   = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;

    v = warpReduceSum(v);
    if (lane == 0) scratch[warpId] = v;
    __syncthreads();

    // First warp reduces the per-warp partials.
    const int numWarps = BLOCK_SIZE / 32;
    v = (threadIdx.x < numWarps) ? scratch[lane] : 0.0f;
    if (warpId == 0) v = warpReduceSum(v);
    return v;
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
//
// One thread computes one window average. The window length is a compile-time
// template parameter so the inner loop is fully unrolled.
//
// Each block first stages the BLOCK_SIZE + WINDOW - 1 input elements it needs
// into shared memory with coalesced loads. Every input element is then read
// from global memory roughly once per block instead of WINDOW times, and all
// WINDOW reads per thread hit fast on-chip shared memory.
//
// The kernel also reduces its block's window averages to a single partial sum
// so the host only has to add `gridDim.x` floats to get the final answer,
// instead of copying back and summing every window average.
template <int WINDOW>
__global__ void computeMovingAverageOpt(const float *__restrict__ in,
                                        float *__restrict__ windowAvg,
                                        float *__restrict__ blockPartialSum,
                                        int size, int numWindows) {
    __shared__ float tile[BLOCK_SIZE + WINDOW - 1];
    __shared__ float reduceScratch[BLOCK_SIZE / 32];

    const int blockStart = blockIdx.x * BLOCK_SIZE;
    const int idx        = blockStart + threadIdx.x;

    // Cooperative, coalesced load of this block's tile (with halo of WINDOW-1).
    for (int i = threadIdx.x; i < BLOCK_SIZE + WINDOW - 1; i += BLOCK_SIZE) {
        const int g = blockStart + i;
        tile[i] = (g < size) ? in[g] : 0.0f;
    }
    __syncthreads();

    // Accumulate the window, then scale once by 1/WINDOW. This replaces the
    // original's WINDOW separate divisions (and WINDOW read-modify-writes to
    // global memory) with WINDOW adds and one multiply, all in registers.
    float avg = 0.0f;
    if (idx < numWindows) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < WINDOW; ++i)
            sum += tile[threadIdx.x + i];
        avg = sum * (1.0f / WINDOW);
        windowAvg[idx] = avg;          // single plain store, no += on global memory
    }

    // Block-level reduction of the window averages (threads past numWindows
    // contribute 0). One store per block instead of one per thread.
    const float blockSum = blockReduceSum(avg, reduceScratch);
    if (threadIdx.x == 0) blockPartialSum[blockIdx.x] = blockSum;
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

// Direct O(size * n) computation, same arithmetic as the GPU kernel so the two
// results can be compared tightly. Bounds are correct: there are exactly
// size - n + 1 full windows in an array of length size.
void computeMovingAverageOnCPU(const vector<float> &a, vector<float> &windowAvg,
                               double &meanOfAverages, int size, int n) {
    const int   numWindows = size - n + 1;
    const float invN       = 1.0f / n;
    windowAvg.assign(numWindows, 0.0f);

    for (int i = 0; i < numWindows; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) sum += a[i + j];
        windowAvg[i] = sum * invN;
    }

    double acc = 0.0;
    for (int i = 0; i < numWindows; ++i) acc += windowAvg[i];
    meanOfAverages = acc / numWindows;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(void) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    const int n          = N;
    const int size       = SIZE;
    const int numWindows = size - n + 1;         // correct count of full windows
    const int numBlocks  = (numWindows + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("Array Size: %d  Sample Size: %d  Windows: %d\n", size, n, numWindows);

    const size_t inBytes      = size * sizeof(float);
    const size_t outBytes     = numWindows * sizeof(float);
    const size_t partialBytes = numBlocks * sizeof(float);

    // initialize vector with random integers in [0, 5)
    srand((unsigned)time(0));
    vector<float> host_a(size);
    std::generate(host_a.begin(), host_a.end(), []() { return (float)(rand() % 5); });

    vector<float> host_b(numWindows);            // sized to what is actually copied back
    vector<float> host_partials(numBlocks);

    float *dev_a = nullptr, *dev_b = nullptr, *dev_partials = nullptr;
    CHECK(cudaMalloc(&dev_a, inBytes));
    CHECK(cudaMalloc(&dev_b, outBytes));
    CHECK(cudaMalloc(&dev_partials, partialBytes));
    CHECK(cudaMemcpy(dev_a, host_a.data(), inBytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE);
    dim3 grid(numBlocks);

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Warm-up launch. The very first kernel launch in a process pays one-time
    // context / module-load costs that have nothing to do with the algorithm,
    // so it is excluded from the timed run.
    computeMovingAverageOpt<N><<<grid, block>>>(dev_a, dev_b, dev_partials, size, numWindows);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // ---- Timed GPU run: kernel + copy of the per-block partial sums + tiny host sum
    CHECK(cudaEventRecord(start, 0));

    computeMovingAverageOpt<N><<<grid, block>>>(dev_a, dev_b, dev_partials, size, numWindows);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(host_partials.data(), dev_partials, partialBytes, cudaMemcpyDeviceToHost));

    double gpuRef = 0.0;
    for (int b = 0; b < numBlocks; ++b) gpuRef += host_partials[b];
    gpuRef /= numWindows;

    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    float GPUtime = 0.0f;
    CHECK(cudaEventElapsedTime(&GPUtime, start, stop));

    // ---- Timed CPU run (host-side clock; CUDA events measure the GPU stream)
    vector<float> cpu_b;
    double cpuRef = 0.0;
    auto cpuStart = std::chrono::high_resolution_clock::now();
    computeMovingAverageOnCPU(host_a, cpu_b, cpuRef, size, n);
    auto cpuStop = std::chrono::high_resolution_clock::now();
    float CPUtime = std::chrono::duration<float, std::milli>(cpuStop - cpuStart).count();

    // ---- Verification (outside the timed region): compare every window average
    CHECK(cudaMemcpy(host_b.data(), dev_b, outBytes, cudaMemcpyDeviceToHost));
    float maxAbsErr = 0.0f;
    for (int i = 0; i < numWindows; ++i)
        maxAbsErr = std::max(maxAbsErr, std::fabs(host_b[i] - cpu_b[i]));

    printf("Kernel: computeMovingAverageOpt<%d> <<<gridDim: %d, blockDim: %d>>>\n",
           n, grid.x, block.x);
    printf("Compute time on GPU: %3.6f ms \n", GPUtime);
    printf("Compute time on CPU: %3.6f ms \n", CPUtime);
    printf("Moving Average computed on CPU: %3.6f\n", cpuRef);
    printf("Moving Average computed on GPU: %3.6f\n", gpuRef);
    printf("Max |GPU - CPU| over all %d window averages: %g  (%s)\n",
           numWindows, maxAbsErr, maxAbsErr < 1e-4f ? "PASS" : "FAIL");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(dev_a));
    CHECK(cudaFree(dev_b));
    CHECK(cudaFree(dev_partials));

    return 0;
}
