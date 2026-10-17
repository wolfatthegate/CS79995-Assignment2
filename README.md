# CS79995 Assignment 2: Moving Average in CUDA

A CUDA program that computes a simple moving average over an array of random values on the GPU. It then runs the same computation on the CPU and compares the results and run times.

This repository contains two versions of the program:

| File | Description |
|------|-------------|
| [moving_Average.cu](moving_Average.cu) | The original implementation. Left unchanged for reference. |
| [moving_Average_optimized.cu](moving_Average_optimized.cu) | An optimized rewrite that fixes the known bugs in the original and uses shared-memory tiling, a fully unrolled window loop, and an on-GPU reduction. |

## How it works

Both programs follow the same outline:

1. Fill an array of `SIZE` floats with random integers from 0 to 4.
2. Launch a kernel on the GPU. Each thread computes the mean of one window of `N` consecutive elements.
3. Reduce the window averages to one value: the mean of all the moving averages.
4. Repeat the computation on the CPU.
5. Print the time each version took and the two results, which should match.

| Parameter | Macro | Default |
|-----------|-------|---------|
| Array size | `SIZE` | 10000 |
| Window (sample) size | `N` | 10 |
| Launch configuration (original) | `grid` / `block` in `main` | 10 blocks × 1000 threads |
| Launch configuration (optimized) | `BLOCK_SIZE`, grid derived from `SIZE` | 40 blocks × 256 threads |

To change the array or window size, edit the `#define`s at the top of the file you are building. In the original, if `SIZE` grows past 10,000 you must also increase `grid` or `block` so that every window still gets a thread. The optimized version computes the grid size from `SIZE` automatically.

## Requirements

- An NVIDIA GPU with CUDA support
- The CUDA Toolkit (`nvcc`), with C++11 or later

## Build and run

Original:

```bash
nvcc -std=c++11 -o moving_average moving_Average.cu
./moving_average
```

Optimized:

```bash
nvcc -O3 -std=c++11 -o moving_average_optimized moving_Average_optimized.cu
./moving_average_optimized
```

## Example output

Original:

```
Using Device 0: <GPU name>
Array Size: 10000  Sample Size: 10
Kernel: computeMovingAverage <<<gridDim: 10, blockDim: 1000>>>
Compute time on GPU: x.xxxxxx ms
Compute time on CPU: x.xxxxxx ms
Moving Average computed on CPU: x.xxxxxx
Moving Average computed on GPU: x.xxxxxx
```

Optimized:

```
Using Device 0: <GPU name>
Array Size: 10000  Sample Size: 10  Windows: 9991
Kernel: computeMovingAverageOpt<10> <<<gridDim: 40, blockDim: 256>>>
Compute time on GPU: x.xxxxxx ms
Compute time on CPU: x.xxxxxx ms
Moving Average computed on CPU: x.xxxxxx
Moving Average computed on GPU: x.xxxxxx
Max |GPU - CPU| over all 9991 window averages: 0  (PASS)
```

The values differ between runs because the input is seeded from the current time.

## What changed in the optimized version

### Correctness fixes

The original has several bugs (listed under *Known issues in the original* below). The optimized version fixes all of them:

| Issue in the original | Fix in the optimized version |
|-----------------------|------------------------------|
| Loops run `size - n + 2` times, one past the last full window, so the final thread reads one element past the end of the input. | Uses `numWindows = size - n + 1`, the exact number of full windows. |
| `host_b` is allocated with `size - n` elements but `size` elements are copied into it (heap overflow). | Every buffer is sized to exactly what is written to or read from it. |
| `dev_b` is never zeroed, and the kernel accumulates into it with `+=`, so stale device memory leaks into the result. | Each thread accumulates in a register and writes its result once with a plain store. No dependence on the initial contents of device memory. |
| Return codes of `cudaMalloc`, `cudaMemcpy`, and the kernel launch are ignored. | Every runtime call is wrapped in `CHECK`, which prints the CUDA error string and the source line. The launch is checked with `cudaGetLastError`. |
| A `__syncthreads()` at the end of the kernel does nothing useful. | Removed. Synchronization is used only where threads actually share data. |

### Performance changes

| Aspect | Original | Optimized | Why it matters |
|--------|----------|-----------|----------------|
| Global memory reads per thread | `n` reads of `dev_a`, each from global memory. Across a block, each input element is fetched `n` times. | Each block cooperatively loads its `BLOCK_SIZE + n - 1` elements into shared memory once with coalesced loads. The `n` reads per thread then come from on-chip shared memory. | Global memory traffic drops by roughly a factor of `n`. Shared memory has far higher bandwidth and lower latency than global memory. |
| Global memory writes per thread | `n` read-modify-write operations (`dev_b[idx] += ...`) to global memory. | One store per thread. | Removes `2n - 1` global memory operations per thread. |
| Arithmetic | `n` floating-point divisions (`/n`) per thread. | `n` additions, then a single multiply by the compile-time constant `1/n`. | Division is many times slower than multiplication on a GPU. |
| Inner loop | Runtime loop bound `n`. | `n` is a template parameter, so `#pragma unroll` fully unrolls the loop and shared-memory offsets become immediates. | Eliminates loop overhead and lets the compiler schedule all loads up front. |
| Final reduction | All `size` floats are copied to the host, then summed serially on the CPU. | Each block reduces its window averages with warp shuffles and shared memory, writing one partial sum per block. The host copies back only `gridDim.x` floats (40 by default) and sums those. | The device-to-host copy shrinks from 40 KB to 160 bytes, and the serial host loop shrinks from thousands of iterations to a few dozen. |
| Block size | 1000 threads per block. | 256 threads per block. | 1000 is not a multiple of the 32-thread warp, so the last warp in each block is 25% idle. 256 is a common choice that keeps occupancy high and leaves register room. |
| Grid size | Hard-coded `grid(10)`, which silently breaks if `SIZE` changes. | Computed as `ceil(numWindows / BLOCK_SIZE)`. | Works for any `SIZE` without editing the launch configuration. |
| Pointer qualifiers | None. | `const ... __restrict__` on the input, `__restrict__` on outputs. | Tells the compiler the buffers do not alias, which permits more aggressive load scheduling and read-only cache use. |
| Timing | The very first kernel launch is timed, which includes one-time CUDA context and module-load costs. | A warm-up launch runs before the timed region. | The reported GPU time reflects the algorithm, not driver start-up. |
| CPU timing | Measured with CUDA events. | Measured with `std::chrono`. | CUDA events time the GPU stream. A host clock is the right tool for host code. |

### Verification

The optimized program does not just compare the two final means. It also copies every window average back after the timed region and reports the maximum absolute difference between the GPU and CPU results. With the default integer inputs the difference is exactly 0.

## Notes on timing

- In the original, the GPU time covers the kernel, the device-to-host copy of the full output, and the host-side reduction. It does not include the host-to-device copy of the input.
- In the optimized version, the GPU time covers the kernel, the device-to-host copy of the per-block partial sums, and the summation of those partials on the host. It does not include the host-to-device copy of the input or the verification copy.
- At the default `SIZE` of 10,000 the whole problem is tiny, so on real hardware both GPU times are dominated by launch and copy latency rather than compute. Increase `SIZE` (for example to 10,000,000) to see the bandwidth savings of the optimized kernel show up clearly.

## Known issues in the original

These apply only to [moving_Average.cu](moving_Average.cu). They are fixed in [moving_Average_optimized.cu](moving_Average_optimized.cu).

- The loops run `size - n + 2` times, but an array of length `size` has only `size - n + 1` full windows. The last iteration reads one element past the end of the input.
- `host_b` holds `size - n` elements, but the program copies `size` elements from the device into it.
- `dev_b` is never zeroed before the kernel adds to it with `+=`, so the GPU result can pick up leftover values in device memory.
- The return values of `cudaMalloc`, `cudaMemcpy`, and the kernel launch are not checked.
