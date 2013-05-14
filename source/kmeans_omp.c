#include <stdio.h>
#include <stdlib.h>
#define BIGNUM 		    1000

float* k_means_omp(float *imageIn, int clusters, int dimension, int iterations)
{
     // the cluster vector
    int numElements = (dimension)*(dimension);
    size_t size = numElements * sizeof(float);
    float *cluster = (float*) malloc(size);// which centroid does each cluster belong to?
    float *imageOut = (float*) malloc(size);//output image

    // the centroids or means
    int means = clusters;
    size_t size2 = means * sizeof(float);
    float *centroids = (float*) malloc(size2);// list of centroids(means)
    float *accumulator = (float*) malloc(size2);/*needed for average step */
    float *numPixelsCentroid = (float*) malloc(size2);/*needed for the update average step*/
                
    int iters = 0;
    int i,j,h,m,n,temp1,temp2;
    float distance;
    float min_temp = BIGNUM;
    float range = 255/(means-1);
 
    //initialize step to set everything to zero
    for ( m = 0; m < means; m++) {
        centroids[m] = range*m;
        accumulator[m] =0;
        numPixelsCentroid[m] =0;
    }
omp_set_num_threads(8);
  while (iters<iterations) {
#pragma omp parallel shared(imageIn,means,numElements,cluster,centroids,numPixelsCentroid,accumulator) private(i,j,distance,min_temp)
 {
#pragma omp for
        //assignment step-> assign each point to cluster of closest centroid
        for ( i=0; i < numElements-1; i++){
            for ( j = 0; j < means; j++) {
                distance = fabs(imageIn[i]-centroids[j]);// compare image to centroids
                if (distance < min_temp){
                    min_temp = distance;
                    cluster[i] = j; // assign this point to current cluster
                }
            }

 	   // set variables used to find average
	   temp1 = (int)cluster[i];
	   accumulator[temp1] += imageIn[i];
	   numPixelsCentroid[temp1]+=1;
	   min_temp = BIGNUM;  //reset mintemp
        }
}
#pragma omp parallel shared(numPixelsCentroid,accumulator,means,centroids,) private(h)
{
#pragma omp for
	//update centroids
        for ( h = 0; h < means; h++) {
	    if (numPixelsCentroid[h] != 0){
            centroids[h] = accumulator[h]/numPixelsCentroid[h];
            //reset
	    }
            accumulator[h] = 0;
            numPixelsCentroid[h] =0;
	   
        }
}
        iters++; // currently using iters rather than a convergence
    }

    // set output
#pragma omp parallel shared(numElements,cluster,imageOut,centroids) private(n)
{
#pragma omp for
    for ( n=0; n < numElements-1; n++){
        temp2 = (int)cluster[n];
        imageOut[n] = centroids[temp2];
    }
}
    for ( m = 0; m < means; m++) {
        printf("%f \n", centroids[m]);
    }

    return imageOut;
}
