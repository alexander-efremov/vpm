#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <fstream>
#include <iostream>
using namespace std;

#define size (int)4096
int numProc;

void NullVector(double *Matrix)
{
	int i;
	for (i = 0; i < size; i++)
		Matrix[i] = 0;
}

void NullMatrix(double **Matrix)
{
	int i,j;
	for (i = 0; i < size; i++)
		for (j = 0; j < size; j++)
			Matrix[i][j] = 0;
}

bool DiagDominationCheck(double **Matrix)
{
	int i,j; double k=0; bool answer;
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			if(i!=j)
			{
				k = k + fabs(Matrix[i][j]);
			}
		}
		if(fabs(Matrix[i][i])>fabs(k))
			answer = true;
		else
			answer = false;
		k=0;
	}
	return answer;
}

void DiagDomination(double **Matrix)
{
	bool check = false;
	while(check!=true)
	{
		int i;
		for (i = 0; i < size; i++)
		{
			Matrix[i][i] = Matrix[i][i] * 100;
		}
		check = DiagDominationCheck(Matrix);
	}
	printf("\nMatrix was rebuilt to diagonal dominance!\n");
}

int main(int argc, char* argv[])
{
	double startTime0, endTime0;
	static int numTh = 0;
    numProc = omp_get_num_procs();
	#pragma omp threadprivate(numTh)
	omp_set_num_threads(numProc);
	#ifdef __MIC__
		{	
			printf("Available number of MIC processors: %d\n", numProc);
			printf("Wait until the end of computation on MIC...\n");
		}
	#else
		{	
			printf("Available number of HOST processors: %d\n", numProc);
			printf("Wait until the end of computation on HOST...\n");
		}
	#endif
	
	int i,j,k;
	double EPS = 1e-8; //Эпсилон
	//-----------------Выделение памяти---------------------------
	double **A = (double**)_mm_malloc(size * sizeof(double*),64);
	for (i = 0; i < size; i++)
	{
		A[i] = (double*)_mm_malloc(size * sizeof(double),64);
	}
	double *B = (double*)_mm_malloc(size * sizeof(double),64);
	double *X = (double*)_mm_malloc(size * sizeof(double),64);
	double *X_NEW = (double*)_mm_malloc(size * sizeof(double),64);
	//------------------------------------------------------------
	NullMatrix(A);
	NullVector(B);
	//---Чтение матрицы из файла---
	ifstream A_F("Matrix.txt");
	for(i = 0; i < size; i++)
	{
		for(j = 0; j < size; j++)
		{
			A_F >> A[i][j];
		}
	}
	A_F.close();
	//---Чтение вектора из файла---
	ifstream B_F("Vector.txt");
	for(i = 0; i < size; i++)
	{
		B_F >> B[i];
	}
	B_F.close();
	//-----------------------------
	//Проверка матрицы на наличие диагонального преобладания
	bool Matrixcheck = DiagDominationCheck(A);
	if(Matrixcheck == false)
	{
		printf("\nMatrix has NOT diagonal dominance!\n");
		DiagDomination(A);
	}
	else
		printf("\nMatrix has diagonal dominance!\n");
	//----------------------CLEAR CODE-----------------
	double norm = 1; //Норма
	int N_iter = 0; //Счетчик итераций
	//----Нач.Приближение-----
	NullVector(X);
	for (i = 0; i < size; i++)
		X[i] = B[i]/A[i][i];
	//------------------------
	double diff = 0;
	startTime0 = omp_get_wtime();
	while(norm > EPS && N_iter < 10000)
	{
		norm = DBL_MIN;
		NullVector(X_NEW);
		for (i = 0; i < size; i++)
		{
			X_NEW[i] = B[i];
			for (j = 0; j < size; j++)
			{
					X_NEW[i] -= A[i][j]*X[j];
			}
			X_NEW[i] += A[i][i]*X[i];
			X_NEW[i] /= A[i][i];
			diff = fabs(X_NEW[i]-X[i]);
			if(diff > norm)
				norm = diff;
		}
		memcpy(X,X_NEW,size*sizeof(double));
		N_iter++;
	}
	endTime0 = omp_get_wtime();
	//-------------------------------------------------
	printf("\nNumber of iterations(K):%d\n",N_iter);
	printf("Matrix Size: %d*%d\n", size,size);
	//--print block--
	printf("It's done!\n");
	printf("\nTotal time(CLEAR) = %10.8f [sec]\n", endTime0 - startTime0);
	//Чистка памяти
	for (i = 0; i < size; i++)
	{
		_mm_free(A[i]);
	}
	_mm_free(A);
	_mm_free(B);
	_mm_free(X);
	_mm_free(X_NEW);
	//-------------
	return 0;
}