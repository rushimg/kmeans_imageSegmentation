#include <stdio.h>
#include <stdlib.h>
#include "constants.h"

float* k_means_serial(float *imageIn, int clusters, int dimension, int iterations)
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

    while (iters<iterations) {
        //assignment step-> assign each point to cluster of closest centroid
        for ( i=0; i < numElements-1; i++){
            for ( j = 0; j < means; j++) {
                distance = fabs(imageIn[i]-centroids[j]);// compare image to centroids
                if (distance < min_temp){
                    min_temp = distance;
                    cluster[i] = j; // assign this point to current cluster
                }
            }
	   min_temp = BIGNUM;  //reset mintemp
        }

	//update centroids
        for ( h = 0; h < means; h++) {
	    for ( i=0; i < numElements-1; i++){
	    if (cluster[i] == h){
	    temp1 = (int)cluster[i];
	    accumulator[temp1] += imageIn[i];
	    numPixelsCentroid[temp1]+=1;
	    }
	   }
 	if (numPixelsCentroid[h] != 0){
            centroids[h] = accumulator[h]/numPixelsCentroid[h];
            //reset
	    }
            accumulator[h] = 0;
            numPixelsCentroid[h] =0;

        }
        iters++; // currently using iters rather than a convergence
    }
    
    // set output
    for ( n=0; n < numElements-1; n++){
        temp2 = (int)cluster[n];
        imageOut[n] = centroids[temp2];
    }

    for ( m = 0; m < means; m++) {
        printf("%f \n", centroids[m]);
    }

    return imageOut;
}
// loop unrolling of 4 for major loop, 2 for update and 8 for writeback

float* k_means_serial_optimized(float *imageIn, int clusters, int dimension, int iterations)
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
    int i,j,h,m,n;
    int temp1,temp2,temp3,temp4,temp;//,temp5,temp6,temp7,temp8,temp9,temp10,temp11;
    float distance,distance2,distance3,distance4;
    float min_temp= BIGNUM;
    float min_temp2= BIGNUM;
    float min_temp3= BIGNUM;
    float min_temp4= BIGNUM;

    float range = 255/(means-1);
 
    //initialize step to set everything to zero
    for ( m = 0; m < means; m++) {
        centroids[m] = range*m;
        accumulator[m] =0;
        numPixelsCentroid[m] =0;
    }

    while (iters<iterations) {
       	for ( i=0; i < numElements-1; i+=4){
	for ( j = 0; j < means; j++) {
                distance = fabs(imageIn[i]-centroids[j]);// compare image to centroids
                if (distance < min_temp){
                    min_temp = distance;
                    cluster[i] = j; // assign this point to current cluster
                }
		distance2 = abs(imageIn[i+1]-centroids[j]);// compare image to centroids
                if (distance2 < min_temp2){
                    min_temp2 = distance2;
                    cluster[i+1] = j; // assign this point to current cluster
                }
		distance3 = abs(imageIn[i+2]-centroids[j]);// compare image to centroids
                if (distance3 < min_temp3){
                    min_temp3 = distance3;
                    cluster[i+2] = j; // assign this point to current cluster
                }
		distance4 = abs(imageIn[i+3]-centroids[j]);// compare image to centroids
                if (distance4 < min_temp4){
                    min_temp4 = distance4;
                    cluster[i+3] = j; // assign this point to current cluster
                }
	   
         }

// set variables used to find average
	   temp1 = (int)cluster[i];
	   accumulator[temp1] += imageIn[i];
	   numPixelsCentroid[temp1]+=1;

	   temp2 = (int)cluster[i+1];
	   accumulator[temp2] += imageIn[i+1];
	   numPixelsCentroid[temp2]+=1;

	   temp3 = (int)cluster[i+2];
	   accumulator[temp3] += imageIn[i+2];
	   numPixelsCentroid[temp3]+=1;

	   temp4 = (int)cluster[i+3];
	   accumulator[temp4] += imageIn[i+3];
	   numPixelsCentroid[temp4]+=1;
	 min_temp=BIGNUM;
         min_temp2=BIGNUM; //reset mintemp
	 min_temp3=BIGNUM;
	 min_temp4=BIGNUM;
        }
      
	//update centroids
        for ( h = 0; h < means; h+=2) {
            if (numPixelsCentroid[h] != 0){
            	centroids[h] = accumulator[h]/numPixelsCentroid[h];
	    }
            //reset
            accumulator[h] = 0;
            numPixelsCentroid[h] =0;
	   if (numPixelsCentroid[h+1] != 0){
            	centroids[h+1] = accumulator[h+1]/numPixelsCentroid[h+1];
	    }
            //reset
            accumulator[h+1] = 0;
            numPixelsCentroid[h+1] =0;

        }
        iters++; // currently using iters rather than a convergence
    }
    
    // set output
    for ( n=0; n < numElements-1; n+=1){
        temp = (int)cluster[n];
        imageOut[n] = centroids[temp];
	/*temp5 = (int)cluster[n+1];
        imageOut[n+1] = centroids[temp5];
	temp6 = (int)cluster[n+2];
        imageOut[n+2] = centroids[temp6];
	temp7 = (int)cluster[n+3];
        imageOut[n+3] = centroids[temp7];
        temp8 = (int)cluster[n+4];
        imageOut[n+4] = centroids[temp8];
	temp9 = (int)cluster[n+5];
        imageOut[n+5] = centroids[temp9];
	temp10 = (int)cluster[n+6];
        imageOut[n+6] = centroids[temp10];
	temp11 = (int)cluster[n+7];
        imageOut[n+7] = centroids[temp11];*/
    }
   for ( m = 0; m < means; m++) {
        printf("%f \n", centroids[m]);
    }
    
    return imageOut;
}
