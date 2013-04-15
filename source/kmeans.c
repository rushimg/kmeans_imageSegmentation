/*
 * gcc -01 -o kmeans kmeans.c 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

#define ITERS              2000
#define TOLERANCE           0.5
#define DIM                 512
#define CLUSTERS             16
#define BIGNUM        100000000
/**
 * rough version of K-means algorithm to highlight the algorithm
 * need to impliment a random creation of centroids
 * also need to add distance calculations based on rgb values
 **/

void k_means(float *imageIn, float *centroids)
{
    // the output cluster vector
    int numElements = DIM;
    size_t size = numElements*numElements* sizeof(float);
    float *cluster = (float*) malloc(size);
                
    int iters = 0;
    int i,j;
    float distance;
    int min_temp = BIGNUM;
    
    while (iters<ITERS) {
        for ( i=0; i < DIM-1; i++){
            for ( j = 0; j < DIM-1; j++) {
                distance = imageIn[i]-centroids[j];/*compare image to centroids*/
                if (distance < min_temp){
                    min_temp = distance;
                    cluster[i] = j; /* assign this point to current cluster*/
                }
            }
        }
        iters++;/* currently using iters rather than a convergence*/
    }
}


