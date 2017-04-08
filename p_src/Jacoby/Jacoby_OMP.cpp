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
	#pragma omp parallel for
	for (i = 0; i < size; i++)
		Matrix[i] = 0;
}

void NullMatrix(double **Matrix)
{
	int i,j;
	#pragma omp parallel for
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

double FindMaxElement(double *Matrix,int Msize)
{
	int i;
	double max = fabs(Matrix[0]);
    for (i = 1; i < Msize; i++) 
	{
        if (fabs(Matrix[i]) > max)
		{
            max = fabs(Matrix[i]);
        }
    }
	return max;
}

int main(int argc, char* argv[])
{
	double startTime0, endTime0,startTime1, endTime1;
	static int numTh = 0;
    numProc = omp_get_num_procs();
	#pragma omp threadprivate(numTh)
	omp_set_num_threads(64);
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
	//---------------CODE FOR PARALLEL-----------------
	double diff = 0;
	double Norm[numProc];
	double normGL = 1;
	int N_iter = 0; //Счетчик итераций
	#pragma omp parallel for
	for (i = 0; i < numProc; i++)
	{
		numTh = omp_get_thread_num();
	}
	//----Нач.Приближение-----
	NullVector(X);
	#pragma omp parallel for
	for (i = 0; i < size; i++)
		X[i] = B[i]/A[i][i];
	//------------------------
	diff = 0;
	startTime1 = omp_get_wtime();
	while(normGL > EPS && N_iter < 10000)
	{
		for (i = 0; i < numProc; i++)
		Norm[i] = DBL_MIN;
		NullVector(X_NEW);
		#pragma omp parallel for private(i,j,diff) shared(X,X_NEW,B,EPS) schedule(static)
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
			if(diff > Norm[numTh])
				Norm[numTh] = diff;
		}
		normGL = FindMaxElement(Norm,numProc);
		memcpy(X,X_NEW,size*sizeof(double));
		N_iter++;
	}
	endTime1 = omp_get_wtime(); 
	//-------------------------------------------------
	printf("\nNumber of iterations(K):%d\n",N_iter);
	printf("Matrix Size: %d*%d\n", size,size);
	//--print block--
	printf("It's done!\n");
	printf("Total time(PARALLEL) = %10.8f [sec]\n", endTime1 - startTime1);
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