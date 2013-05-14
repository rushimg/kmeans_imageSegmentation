// gcc -O3 -fopenmp -o main main.c -lrt

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>     
#include "kmeans_omp.c"
#include "bmpfuncs.h"
#include "bmpfuncs.c"      
#include "kmeans_ser.c"   
#define GIG               1000000000
#define ITERS 			 100
#define TOLERANCE 		 0.5
#define DIM 		       	 512
#define CLUSTERS 		  4

int main(void){


	float* readImage(const char *filename, int* widthOut, int* heightOut);
	void storeImage(float *imageOut, const char *filename, int rows, int cols, const char* refFilename);
        float* k_means_omp(float *imageIn, int clusters, int dimension, int iterations);
 	float* k_means_serial(float *imageIn, int clusters, int dimension, int iterations);
	float* k_means_serial_optimized(float *imageIn, int clusters, int dimension, int iterations);
        struct timespec diff(struct timespec start, struct timespec end);
        struct timespec time1, time2;

        int w = DIM;
        int* width = &w;
	int h = DIM;
        int* height = &h;
	const char *imageInFile =  "testImageGrey.bmp";
        const char *imageOutFile =  "testImage512OMP_out.bmp";
	const char* refFilename = "testImageGrey.bmp";

	float* imageIn = readImage(imageInFile, width, height);

	printf("iters = %d\n", ITERS);
	printf("clusters = %d\n", CLUSTERS);

	printf("CPU CODE RUNNING \n");
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);   
        float *imageOut2 = k_means_serial(imageIn, CLUSTERS, DIM, ITERS);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
        printf("CPU TIME: %ld \n", (long int)(((double)GIG *diff(time1,time2).tv_sec + diff(time1,time2).tv_nsec)));


	printf("OMP CODE RUNNING \n");
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);   
        float *imageOut = k_means_omp(imageIn, CLUSTERS, DIM, ITERS);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
        printf("OMP TIME: %ld \n", (long int)(((double)GIG *diff(time1,time2).tv_sec + diff(time1,time2).tv_nsec)));


        storeImage(imageOut,imageOutFile, w, h, refFilename);
}


/******TimerStuff*****************/

struct timespec diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}

double fRand(double fMin, double fMax)
{
    double f = (double)random() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

/************************************/
