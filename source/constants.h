#include <stdio.h>
#include <stdlib.h>

#define GIG               1000000000
#define ITERS 			200
#define TOL    		      0.0005
#define DIM 		       	 512
#define CLUSTERS 		 4
#define BIGNUM			1000
#define THREADS_PER_BLOCK	256
#define BLOCKS_PER_GRID	((DIM*DIM + (256) - 1)/(256))
