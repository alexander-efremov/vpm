#include "common.h"
#include "utils.h"
#include "compute_density_cuda.cuh"
#include <algorithm>
#include <cuda.h>
#include <hemi.h>

__global__ void kernel(double* prev_result, double* result)
{	
int C_XY_LEN=10;
int C_OX_LEN=10;
int C_OY_LEN=10;
	 for (int opt = blockIdx.x * blockDim.x + threadIdx.x; opt < C_XY_LEN; opt += blockDim.x * gridDim.x)
	 {		
	 	int i = opt % (C_OX_LEN + 1);
	 	int j = opt / (C_OY_LEN + 1);
	 	
	 }
}

float solve_cuda(double* density)
{
	const int gridSize = 256;
	const int blockSize =  512; 
	double *result = NULL, *prev_result = NULL, *ox = NULL, *oy=NULL;
	float time;
	/*int size = sizeof(double)*XY_LEN;
	double *prev_result_h = new double[XY_LEN];
	for (int j = 0; j < OY_LEN + 1; j++)
	{
		for (int i = 0; i < OX_LEN_1; i++)
		{
			prev_result_h[OX_LEN_1 * j + i] = analytical_solution(0, OX[i], OY[j]);
		}
	}

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	checkCuda(cudaMemcpyToSymbol(C_OY_LEN, &OY_LEN, sizeof(int)));
	
	checkCuda(cudaMalloc((void**)&(result), size) );
	checkCuda(cudaMemset(result, 0, size) );
	checkCuda(cudaMalloc((void**)&(prev_result), size) );
	checkCuda(cudaMalloc((void**)&(ox), sizeof(ox)*(OX_LEN+1)));
	checkCuda(cudaMalloc((void**)&(oy), sizeof(oy)*(OY_LEN+1)));
	checkCuda(cudaMemcpy(prev_result, prev_result_h, size, cudaMemcpyHostToDevice));	

	cudaEventRecord(start, 0);   

	TIME = 0;
	int tl = 0;
	int tempTl  = TIME_STEP_CNT -1;
        while(tl < tempTl)
	{
	    checkCuda(cudaMemcpyToSymbol(C_PREV_TIME, &TIME, sizeof(double)));
            TIME = TAU * (tl+1);
	    checkCuda(cudaMemcpyToSymbol(C_TIME, &TIME, sizeof(double)));	
	    kernel<<<gridSize, blockSize>>>(prev_result, result);

	    checkCuda(cudaMemcpyToSymbol(C_PREV_TIME, &TIME, sizeof(double)));
            TIME = TAU * (tl+2);
	    checkCuda(cudaMemcpyToSymbol(C_TIME, &TIME, sizeof(double)));	
	    kernel<<<gridSize, blockSize>>>(result, prev_result);		 		 
	    tl+=2;            
	}
	
	if (TIME_STEP_CNT%2==0)
		checkCuda(cudaMemcpy(density, prev_result, size, cudaMemcpyDeviceToHost));
	else
		checkCuda(cudaMemcpy(density, result, size, cudaMemcpyDeviceToHost));
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	time /= 1000; // to seconds
	//printf("Computation Time %f\n", time);
	cudaFree(result);
	cudaFree(prev_result);
	cudaFree(ox);
	cudaFree(oy);
	cudaDeviceReset();
	
	delete[] prev_result_h;
	*/
	return time;
}

double* compute_density_cuda_internal(double b, double lb, double rb, double bb, double ub,
                        double tau, int time_step_count, int ox_length, int oy_length, double& norm, float& time)
{
#ifdef __NVCC__
int XY_LEN=10;	
	double* density = new double[XY_LEN];
	time = solve_cuda(density);
	return density;
#else
        return NULL;
#endif
}