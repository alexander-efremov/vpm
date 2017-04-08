#ifndef COMPUTE_DENSITY_CUDA_CUH
#define	COMPUTE_DENSITY_CUDA_CUH

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

extern double* compute_density_cuda_internal(double b, double lb, double rb, 						 double bb, double ub,
                        double tau, int time_step_count, int ox_length, int oy_length, double& norm, float& time);

#endif /* COMPUTE_DENSITY_CUDA_CUH */