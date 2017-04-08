#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include "common.h"
#include "test_utils.h"
#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)
#ifdef _OPENMP
  #include <omp.h>
#else
  #define omp_get_thread_num() 1
  #define omp_get_num_threads() 0
  #define OMP_NUM_THREADS -1
#endif

class cpu : public testing::Test
{
public:
	cpu()
	{
	    #ifdef VER
//		 printf("%s\n", STRINGIZE_VALUE_OF(VER));
	    #endif
	}
	
protected:
	double* solve_internal( double& time)
	{
	return NULL;
	}
	
	double* solve_internal_cuda( float& time)
	{
	return NULL;
	}
};

TEST_F(cpu, test_to_model_cuda)
{
	int first = 0, last = 1;
	double norm_test, norm_model;
	float time = 0;
	for (int lvl = first; lvl < last; ++lvl)
	{
	}
}