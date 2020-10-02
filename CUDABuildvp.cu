#include "CUDABuildvp.h"

#include "euclidean_distance.h"
#include "my_select.h"
#include <sys/timeb.h>

#include <stdio.h>

#define THREADS_PER_BLOCK 1024
#define D(i,j,d) ( i*d + j )

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
static void HandleError( cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
    exit( EXIT_FAILURE );
  }
}



    vptree * CUDABuildvp(double *X,const int n,const int d){


      vptree *tree = NULL;
      tree = (vptree *)malloc(n*sizeof(vptree));   //vptree array

      int *X_indexes = (int *)malloc(n*sizeof(int));
      for (int i = 0, j=n-1; i < n; i++,j--)
        X_indexes[i]=j;


      int *X_indexes_dev; // Data and Results Array to Device
      HANDLE_ERROR( cudaMalloc((void**) &X_indexes_dev,  n*sizeof(int))  );
      HANDLE_ERROR( cudaMemcpy( X_indexes_dev, X_indexes, n*sizeof(int), cudaMemcpyHostToDevice)  ); // Copy Data from dataArr to dataArr_dev


      int *array_n = (int *)malloc(n*sizeof(int));
      for (int i = 0; i < n; i++)
        array_n[i]=n;

      int *array_n_dev;
      HANDLE_ERROR( cudaMalloc((void**) &array_n_dev,  n*sizeof(int))  );
      HANDLE_ERROR( cudaMemcpy( array_n_dev, array_n, n*sizeof(int), cudaMemcpyHostToDevice)  ); // Copy Data from dataArr to dataArr_dev
      //  free(array_n);

      int *X_heads = (int *)malloc(n*sizeof(int));
      for (int i = 0; i < n; i++)
        X_heads[i]=0;

      X_heads[X_indexes[0]]=-1;

      int *X_heads_dev; // Data and Results Array to Device
      HANDLE_ERROR( cudaMalloc((void**) &X_heads_dev,  n*sizeof(int))  );
      HANDLE_ERROR( cudaMemcpy( X_heads_dev, X_heads, n*sizeof(int), cudaMemcpyHostToDevice)  ); // Copy Data from dataArr to dataArr_dev

      double *dataArr_dev; // Data Array to Device
      HANDLE_ERROR( cudaMalloc((void**) &dataArr_dev,  n*d*sizeof(double))  );
      HANDLE_ERROR( cudaMemcpy( dataArr_dev, X, n*d*sizeof(double), cudaMemcpyHostToDevice)  ); // Copy Data from dataArr to dataArr_dev


      double *distances = (double *)malloc(n*sizeof(double));
      double *distances_dev;
      HANDLE_ERROR( cudaMalloc((void**) &distances_dev,  n*sizeof(double))  );

      int *X_indexes_dev_copy; // Data and Results Array to Device
      HANDLE_ERROR( cudaMalloc((void**) &X_indexes_dev_copy,  n*sizeof(int))  );


      dim3 grid;
      grid.x = ceil(n/(THREADS_PER_BLOCK*1.0));
      dim3 block;
      block.x = min(THREADS_PER_BLOCK,n);

      int deep = ceil(log( n ) / log( 2 ));

      struct timeb startCPU, endCPU;
      ftime(&startCPU);


      for(int k=0; k<=deep; k++)  {



        distanceWithIndexesGPU<<<grid,block>>>(dataArr_dev,X_indexes_dev,X_heads_dev,distances_dev,n,d);   // Cuda Calculate Distances
        cudaDeviceSynchronize();

        HANDLE_ERROR( cudaMemcpy(X_indexes_dev_copy,  X_indexes_dev,  n*sizeof(int), cudaMemcpyDeviceToDevice)  ); // Copy Data from dataArr to dataArr_dev

        my_sort<<<grid,block>>>(dataArr_dev,X_indexes_dev,X_indexes_dev_copy,X_heads_dev,distances_dev,array_n_dev,n);   // Cuda Calculate Distances

        cudaDeviceSynchronize();

      }


      ftime(&endCPU);
      printf("|%10d |", (int)(1000 * (endCPU.time - startCPU.time) + (endCPU.millitm - startCPU.millitm)) );


      HANDLE_ERROR( cudaMemcpy( X_heads, X_heads_dev, n*sizeof(int), cudaMemcpyDeviceToHost)  ); // Copy Data from dataArr to dataArr_dev
      HANDLE_ERROR( cudaMemcpy(distances, distances_dev, n*sizeof(double), cudaMemcpyDeviceToHost)  ); // Copy Data from dataArr to dataArr_dev
      HANDLE_ERROR( cudaMemcpy( X_indexes, X_indexes_dev, n*sizeof(int), cudaMemcpyDeviceToHost)  ); // Copy Data from dataArr to dataArr_dev




      for (int i = 0; i < n; i++)
      {
        int index = 0;
        index = X_indexes[i];
        tree[i].md = distances[index];
        tree[i].vp = (double *)malloc(d*sizeof(double));
        for (int j = 0; j < d; j++)
        tree[i].vp[j]= X[D(index,j,d)];
        tree[i].idx = index;
        tree[i].inner = NULL;
        tree[i].outer = NULL;

        for (int k = 0; k < n; k++) {
          if(X_heads[index]*2 == X_heads[X_indexes[k]])
          tree[i].inner = &tree[k];

          if(X_heads[index]*2 -1 == X_heads[X_indexes[k]])
          tree[i].outer = &tree[k];

        }
      }

      free(X_heads);
      free(X_indexes);
      free(distances);


      return tree;

}
