#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <algorithm>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

#define SIZE 10000
#define N 10

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CHECK(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

__global__ void computeMovingAverage(const float *dev_a, float *dev_b, int size, int n) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x; 
	
	if(idx < (size - n + 2)){
		for(int i = 0; i< n; i++){
			dev_b[idx] += dev_a[idx + i]/n; 
		} 
	}
	__syncthreads(); 
}

void computeMovingAverageOnCPU(vector<float> &host_a, float &cpuRef, int size, int n) {	
	vector<float> temp_vec(size);

	for(int i = 0; i < (size - n + 2); i++)
		for(int j = 0; j < n; j++) 
			temp_vec[i] += host_a[i+j]/n; 
		
	for(int i = 0; i < (size - n + 2); i++)
		cpuRef += (float)(temp_vec[i]/(size - n + 1)); 
	
}

int main(void){

	// set up device
	int dev = 0; 
	cudaDeviceProp deviceProp; 
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev)); 

	int n = N; 
	int size = SIZE; 

	printf("Array Size: %d  Sample Size: %d\n", size, N);
	size_t nBytes = size * sizeof(float); 
	float cpuRef = 0.0f; 
	float gpuRef = 0.0f; 

	// initialize random number
	srand ((int)time(0));
 
	// initialize vector and generate random indices between 0 and 5. 
	vector<float> host_a(size);
	vector<float> host_b(size-n); 
	generate(host_a.begin(), host_a.end(), []() { return rand() % 5; }); 

	float *dev_a, *dev_b; 
	cudaMalloc(&dev_a, nBytes); 
	cudaMalloc(&dev_b, nBytes); 
	cudaMemcpy(dev_a, host_a.data(), nBytes, cudaMemcpyHostToDevice); 

	// declare block and grid dimension. 

	dim3 block (1000); 
	dim3 grid (10); 

	// Timer starts 
	float GPUtime, CPUtime; 
	cudaEvent_t start, stop; 

	cudaEventCreate(&start); 
	cudaEventCreate(&stop); 
	cudaEventRecord(start, 0); 

	computeMovingAverage <<< grid, block >>> (dev_a, dev_b, size, n); 
	cudaMemcpy(host_b.data(), dev_b, nBytes, cudaMemcpyDeviceToHost); 
	for(int x = 0; x< (size-n+2); x++){
		gpuRef += (float)(host_b[x]/(size - n + 1)); 
	}
	// timer stops
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&GPUtime, start, stop); 

	cudaEventCreate(&start); 
	cudaEventCreate(&stop); 
	cudaEventRecord(start, 0); 

	computeMovingAverageOnCPU(host_a, cpuRef, size, n);

	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&CPUtime, start, stop); 

    printf("Kernel: computeMovingAverage <<<gridDim: %d, blockDim: %d>>>\n", grid.x, block.x); 

	printf("Compute time on GPU: %3.6f ms \n", GPUtime); 
	printf("Compute time on CPU: %3.6f ms \n", CPUtime); 
	printf("Moving Average computed on CPU: %3.6f\n", cpuRef); 
	printf("Moving Average computed on GPU: %3.6f\n", gpuRef); 

	cudaFree(dev_a);
	cudaFree(dev_b); 

	return (0); 
}