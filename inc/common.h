#ifndef COMMON_H_INCLUDED
#define COMMON_H_INCLUDED
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>
#include <float.h>
#include "timer.h"

extern double* compute_density(
	double b,
	double lb,
	double rb,
	double bb,
	double ub,
	double time_step,
	int time_step_count,
	int ox_length,
	int oy_length, double &norm, double& time);

extern double* compute_density_cuda(
	double b,
	double lb,
	double rb,
	double bb,
	double ub,
	double time_step,
	int time_step_count,
	int ox_length,
	int oy_length, double &norm, float& time);

#endif