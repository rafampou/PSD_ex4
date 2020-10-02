#include "euclidean_distance.h"

#include <math.h>
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

__global__ void distanceSimple(double *X,double *distance,int n)
{
  int globalN = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ double dif[];

  if( globalN  > n )
    return;

  dif[threadIdx.x*blockDim.y + threadIdx.y] = (X[D(globalN,threadIdx.y,blockDim.y)] - X[D(n-1,threadIdx.y,blockDim.y)]);
  dif[threadIdx.x*blockDim.y + threadIdx.y] = dif[threadIdx.x*blockDim.y + threadIdx.y] * dif[threadIdx.x*blockDim.y + threadIdx.y];

  __syncthreads();
  if(threadIdx.y == 0)
  {
    for (int i = 1; i < blockDim.y; i++) {
      dif[threadIdx.x*blockDim.y] += dif[threadIdx.x*blockDim.y + i];
    }
    distance[globalN] = sqrt(dif[threadIdx.x*blockDim.y]);
  }

}

__global__ void distanceSimpleGPU(double *X,double *distance,int n,int d)
{
  int globalN = blockIdx.x * blockDim.x + threadIdx.x;

  if( globalN  > n )
    return;

  double temp = 0;
  for (int i = 0; i < d; i++)
    temp += (X[D(globalN,i,d)] - X[D(n-1,i,d)])*(X[D(globalN,i,d)] - X[D(n-1,i,d)]);

  distance[globalN] = sqrt(temp);


}

__global__ void distanceWithIndexesGPU(double *X,int *indexes,int *heads,double *distance,int n,int d)
{
  int globalN = blockIdx.x * blockDim.x + threadIdx.x;

  if( globalN  >= n )
    return;

  int index = indexes[globalN];
  int head = heads[index];
  if(head < 0)
  {
        return;
  }
  head = indexes[head];
  double temp = 0;
  for (int i = 0; i < d; i++)
    temp += pow(X[D(index,i,d)] - X[D(head,i,d)],2);
//    temp += (X[D(index,i,d)] - X[D(head,i,d)])*(X[D(index,i,d)] - X[D(head,i,d)]);

  distance[index] = sqrt(temp);


}

void distance(double *X,double *dinstance,int n,int d) {


  double *dataArr_dev; // Data Array to Device
  HANDLE_ERROR( cudaMalloc((void**) &dataArr_dev,  n*d*sizeof(double))  );

  double *distances_dev; // Data and Results Array to Device
  HANDLE_ERROR( cudaMalloc((void**) &distances_dev,  n*sizeof(double))  );

  HANDLE_ERROR( cudaMemcpy( dataArr_dev, X, n*d*sizeof(double), cudaMemcpyHostToDevice)  ); // Copy Data from dataArr to dataArr_dev

  distanceSimpleGPU<<<ceil(n/1024.0),1024>>>(dataArr_dev,distances_dev,n,d);   // Cuda Calculate Distances

  HANDLE_ERROR( cudaMemcpy(dinstance, distances_dev, n*sizeof(double), cudaMemcpyDeviceToHost)  ); // Copy Data from dataArr to dataArr_dev

  cudaDeviceSynchronize();

  HANDLE_ERROR( cudaFree(distances_dev) );
  HANDLE_ERROR( cudaFree(dataArr_dev) );

}

void distanceCompareCPU_GPU(double *X,int n,int d)
{

  printf("\nCompare %d x %d = %d distances\n",n,d,n*d );

  double *distanceCPU = (double *)malloc(n*sizeof(double));
  double *distancesGPU_1 = (double *)malloc(n*sizeof(double));
  double *distancesGPU_2 = (double *)malloc(n*sizeof(double));


  struct timeb startCPU, endCPU;
  ftime(&startCPU);
  for (int i = 0; i < n; i++) {
    distanceCPU[i]=0;
    for (int j = 0; j < d; j++) {
      distanceCPU[i] += (X[D(i,j,d)] - X[D(n-1,j,d)])*(X[D(i,j,d)] - X[D(n-1,j,d)]);
    }
    distanceCPU[i] = sqrt(distanceCPU[i]);
  }
  ftime(&endCPU);
  printf("Sequential time: %f msec\n", (1000.0 * (endCPU.time - startCPU.time) + (endCPU.millitm - startCPU.millitm)) );


  /*
  Calculate Distances from last point to all in CUDA
  */
  dim3 dimBlock_distances;
  dim3 dimGrid_distances;

  // Set kernel dimesions
  if ( (n*d) / THREADS_PER_BLOCK ) {
    dimBlock_distances.x = (int) (THREADS_PER_BLOCK / d);
    dimBlock_distances.y = d;
    dimGrid_distances.x =  ceil( (double)n /  dimBlock_distances.x )  ;
  }
  else
  {
    dimBlock_distances.x = n;
    dimBlock_distances.y = d;
    dimGrid_distances.x = 1;
  }


  double *dataArr_dev; // Data Array to Device
  HANDLE_ERROR( cudaMalloc((void**) &dataArr_dev,  n*d*sizeof(double))  );

  double *distances_dev; // Data and Results Array to Device
  HANDLE_ERROR( cudaMalloc((void**) &distances_dev,  n*sizeof(double))  );

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);



  HANDLE_ERROR( cudaMemcpy( dataArr_dev, X, n*d*sizeof(double), cudaMemcpyHostToDevice)  ); // Copy Data from dataArr to dataArr_dev
    cudaEventRecord(start);

  distanceSimple<<<dimGrid_distances,dimBlock_distances,dimBlock_distances.x*d*sizeof(double)>>>(dataArr_dev,distances_dev,n);   // Cuda Calculate Distances

  cudaDeviceSynchronize();
    cudaEventRecord(stop);
  HANDLE_ERROR( cudaMemcpy(distancesGPU_1, distances_dev, n*sizeof(double), cudaMemcpyDeviceToHost)  ); // Copy Data from dataArr to dataArr_dev


  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Paraller 1 time: %f msec\n",milliseconds );


  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  double *distances_dev1; // Data and Results Array to Device
  HANDLE_ERROR( cudaMalloc((void**) &distances_dev1,  n*sizeof(double))  );

  HANDLE_ERROR( cudaMemcpy( dataArr_dev, X, n*d*sizeof(double), cudaMemcpyHostToDevice)  ); // Copy Data from dataArr to dataArr_dev
    cudaEventRecord(start);
  distanceSimpleGPU<<<ceil(n/1024.0),1024>>>(dataArr_dev,distances_dev1,n,d);   // Cuda Calculate Distances
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
  HANDLE_ERROR( cudaMemcpy(distancesGPU_2, distances_dev1, n*sizeof(double), cudaMemcpyDeviceToHost)  ); // Copy Data from dataArr to dataArr_dev


  cudaEventSynchronize(stop);

  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Paraller 2 time: %f msec\n",milliseconds );

  HANDLE_ERROR( cudaFree(distances_dev) );

  int checkError = 0;
  for (int i = 0; i < n; i++) {
    if (distancesGPU_1[i] != distanceCPU[i] || distancesGPU_2[i] != distanceCPU[i])
    {
        checkError = 1;
        printf("%f %f != %f |",distancesGPU_1[i],distancesGPU_2[i],distanceCPU[i] );
        for (int j = 0; j < d; j++) {
          printf(" %f ",X[D(i,j,d)] );
        }
        printf("\n");
    }
    else
      printf("%f %f == %f \n",distancesGPU_1[i],distancesGPU_2[i],distanceCPU[i] );

  }
  free(distancesGPU_1);
  free(distancesGPU_2);
  free(distanceCPU);

  if (checkError) printf("\n Wrong Compare\n");
  else  printf("\n Correct Compare\n");




}
