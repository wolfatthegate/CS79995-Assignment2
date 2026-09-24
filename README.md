# CS79995 Assignment 2: Moving Average in CUDA

A CUDA program that computes a simple moving average over an array of random values on the GPU. It then runs the same computation on the CPU and compares the results and run times.

## How it works

The program:

1. Fills an array of `SIZE` floats with random integers from 0 to 4.
2. Launches `computeMovingAverage` on the GPU. Each thread computes the mean of one window of `N` consecutive elements.
3. Copies the window averages back to the host and reduces them to one value: the mean of all the moving averages.
4. Repeats the computation on the CPU with `computeMovingAverageOnCPU`.
5. Prints the time each version took and the two results, which should match.

| Parameter | Macro | Default |
|-----------|-------|---------|
| Array size | `SIZE` | 10000 |
| Window (sample) size | `N` | 10 |
| Launch configuration | `grid` / `block` in `main` | 10 blocks × 1000 threads |

To change the array or window size, edit the `#define`s at the top of [moving_Average.cu](moving_Average.cu). If `SIZE` grows past 10,000, increase `grid` or `block` so that every window still gets a thread.

## Requirements

- An NVIDIA GPU with CUDA support
- The CUDA Toolkit (`nvcc`), with C++11 or later

## Build and run

```bash
nvcc -std=c++11 -o moving_average moving_Average.cu
./moving_average
```

## Example output

```
Using Device 0: <GPU name>
Array Size: 10000  Sample Size: 10
Kernel: computeMovingAverage <<<gridDim: 10, blockDim: 1000>>>
Compute time on GPU: x.xxxxxx ms
Compute time on CPU: x.xxxxxx ms
Moving Average computed on CPU: x.xxxxxx
Moving Average computed on GPU: x.xxxxxx
```

The values differ between runs because the input is seeded from the current time.

## Notes on timing

- The GPU time covers the kernel, the device-to-host copy, and the host-side reduction. It does not include the host-to-device copy of the input.
- Both versions are timed with CUDA events.

## Known issues

- The loops run `size - n + 2` times, but an array of length `size` has only `size - n + 1` full windows. The last iteration reads one element past the end of the input.
- `host_b` holds `size - n` elements, but the program copies `size` elements from the device into it.
- `dev_b` is never zeroed before the kernel adds to it with `+=`, so the GPU result can pick up leftover values in device memory.
- The return values of `cudaMalloc`, `cudaMemcpy`, and the kernel launch are not checked.
