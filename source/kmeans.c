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
 **/

void k_means(float *imageIn)
{
    // the cluster vector
    int numElements = DIM*DIM;
    size_t size = numElements * sizeof(float);
    float *cluster = (float*) malloc(size);
    
    // the centroids or means
    int means = CLUSTERS;
    float *centroids = (float*) malloc(means);
    float *accumulator = (float*) malloc(means);/*needed for average step */
    float *numPixelsCentroid  = (float*) malloc(means);/*needed for the update average step*/
                
    int iters = 0;
    int i,j,temp;
    float distance;
    int min_temp = BIGNUM;
    
    while (iters<ITERS) {
        //assignment step-> assign each point to cluster of closest centroid
        for ( i=0; i < numElements-1; i++){
            for ( j = 0; j < means-1; j++) {
                distance = abs(imageIn[i]-centroids[j]);/*compare image to centroids*/
                if (distance < min_temp){
                    min_temp = distance;
                    cluster[i] = j; /* assign this point to current cluster*/
                }
            }
        }
        
        //update step-> set centroid of each cluster as mean
        for ( i=0; i < numElements-1; i++){
                temp = cluster[i];
                accumulator[temp] += imageIn[i];
                /* for cluster
                 * get its centroid
                 * and accumulate the pixel values matching this cluster
                 */
                numPixelsCentroid[i]+=1;
        }
        for ( j = 0; j < means-1; j++) {
            centroids[j] = accumulator[j]/numPixelsCentroid[j];
            //reset
            accumulator[j] =0;
            numPixelsCentroid[j] =0;
        }
        
        iters++;/* currently using iters rather than a convergence*/
    }
}

