#include <stdio.h>
#include <stdlib.h>

#include "constants.h"
__global__ void k_means_kernel_assignment_split(const float *imageIn, float *cluster, float *centroids, const int means)
{
    // do this for each individual pixel
    float min_temp = BIGNUM;
    float distance;
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int j;
    for (j = 0; j <means;j++) {
                distance = fabs(centroids[j]-imageIn[index]);// compare image to centroids
                if (distance<min_temp){
                    min_temp = distance;
		    cluster[index]= j;		    
		}
   }
}



__global__ void k_means_kernel_writeBack_split(float *imageOut, const float *imageIn, const float *centroids, const float *cluster)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int temp2 = (int)cluster[index];
	imageOut[index] = centroids[temp2];
}

float* k_means_split(float *imageIn, int clusters, int dimension, int iterations){
   struct timespec diff(struct timespec start, struct timespec end);
   struct timespec timeStart, timeEnd;
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    
    // the cluster vector
    int numElements = (dimension)*(dimension);
    size_t size = numElements * sizeof(float);
    float *cluster = (float*) malloc(size);// which centroid does each cluster belong to?
    float *imageOut = (float*) malloc(size);//output image

    // the centroids or means
    int means = clusters;
    size_t size2= means * sizeof(float);
    float *centroids = (float*) malloc(size2);// list of centroids(means)
	float *accumulator = (float*) malloc(size2);
	float *numPixelsCentroid = (float*) malloc(size2);/*needed for the update average step*/
     
    float range = 255/(means-1);
    //initialize step to set everything to zero
    for (int m = 0; m < means; m++) {
        centroids[m] = range*m;
        accumulator[m] =0;
        numPixelsCentroid[m] =0;
    }

    // Allocat DEVICE vectors
    float *d_imageIn = NULL;
    err = cudaMalloc((void**)&d_imageIn,size);
    if(err != cudaSuccess)
    {
	fprintf(stderr, "Failed to allocate device vector imageIn (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }
    float *d_imageOut = NULL;
    err = cudaMalloc((void**)&d_imageOut,size);
    if(err != cudaSuccess)
    {
	fprintf(stderr, "Failed to allocate device vector imageOut (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }
    float *d_cluster = NULL;
    err = cudaMalloc((void**)&d_cluster,size);
    if(err != cudaSuccess)
    {
	fprintf(stderr, "Failed to allocate device vector cluster (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }
    float *d_centroids = NULL;
    err = cudaMalloc((void**)&d_centroids,size2);
    if(err != cudaSuccess)
    {
	fprintf(stderr, "Failed to allocate device vector centroids (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }


    // Copy vectors to DEVICE
    printf("Copy input data from the host memory to the CUDA device \n");
      
    err = cudaMemcpy(d_imageIn, imageIn, size , cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
	fprintf(stderr, "Failed to copy Vector imagein from host to device (error code %s)! \n", cudaGetErrorString(err));
    } 
    err = cudaMemcpy(d_imageOut, imageOut, size , cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
	fprintf(stderr, "Failed to copy Vector imageOut from host to device (error code %s)! \n", cudaGetErrorString(err));
    } 
    err = cudaMemcpy(d_cluster, cluster, size , cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
	fprintf(stderr, "Failed to copy Vector cluster from host to device (error code %s)! \n", cudaGetErrorString(err));
    }   

    err = cudaMemcpy(d_centroids, centroids, size2 , cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
	fprintf(stderr, "Failed to copy Vector centroids from host to device (error code %s)! \n", cudaGetErrorString(err));
    }

    // Launch the kmeans CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + 256 - 1)/256;

    dim3 dimBlock(THREADS_PER_BLOCK);
    dim3 dimGrid(BLOCKS_PER_GRID,CLUSTERS,1);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &timeStart);

for (int iters =0; iters<iterations; iters++){
    	k_means_kernel_assignment_split<<<dimGrid,dimBlock>>>(d_imageIn, d_cluster, d_centroids, means);
        cudaThreadSynchronize();

    err = cudaGetLastError();

    if( err != cudaSuccess)
    {
	fprintf(stderr, "Failed to launch kmeans kernel (error code %s)! \n",cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }
	err = cudaMemcpy(centroids,d_centroids,size2,cudaMemcpyDeviceToHost);
	err = cudaMemcpy(cluster, d_cluster, size , cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
	fprintf(stderr, "Failed to copy from host to device (error code %s)! \n", cudaGetErrorString(err));
    }   
  	 for ( int h = 0; h < means; h++) {
	    for ( int i=0; i < numElements-1; i++){
	    if (cluster[i] == h){
	    int temp1 = (int)cluster[i];
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
    err = cudaMemcpy(d_centroids, centroids, size2 , cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
	fprintf(stderr, "Failed to copy Vector centroids from host to device (error code %s)! \n", cudaGetErrorString(err));
    }
	iters++;

}
 k_means_kernel_writeBack_split<<<blocksPerGrid,threadsPerBlock>>>(d_imageOut, d_imageIn, d_centroids, d_cluster);
 cudaThreadSynchronize();  
 clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &timeEnd);
printf("Internal GPU time: %ld \n", (long int)(((double)GIG *diff(timeStart,timeEnd).tv_sec + diff(timeStart,timeEnd).tv_nsec)));
  err = cudaMemcpy(centroids,d_centroids,size2,cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
	fprintf(stderr, "Failed to copy vector centroids from device to host (error code %s)! \n",cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(imageOut,d_imageOut,size,cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
	fprintf(stderr, "Failed to copy vector imageOut from device to host (error code %s)! \n",cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }
    // Free divice global memory
    err = cudaFree(d_imageOut);
    if(err != cudaSuccess)
    {
	fprintf(stderr,"Failed to free device vector centroids (error code %s)! \n",cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }
    err = cudaFree(d_centroids);
    if(err != cudaSuccess)
    {
	fprintf(stderr,"Failed to free device vector centroids (error code %s)! \n",cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }

    
    err = cudaFree(d_imageIn);
    if(err != cudaSuccess)
    {
	fprintf(stderr,"Failed to free device vector imageIn (error code %s)! \n",cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }

    err = cudaFree(d_cluster);
    if(err != cudaSuccess)
    {
	fprintf(stderr,"Failed to free device vector cluster (error code %s)! \n",cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }

    // Reset the device and exit
    err = cudaDeviceReset();
    if(err != cudaSuccess)
    {
	fprintf(stderr,"Failed to deinitialize the device! (error code %s)! \n",cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }
   // set output
   for (int m = 0; m < means; m++) {
	printf("%f \n",centroids[m]);
}
   return imageOut;
}
