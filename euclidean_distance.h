
__global__ void distanceSimple(double *X,double *distance,int n);
__global__ void distanceSimpleGPU(double *X,double *distance,int n,int d);
__global__ void distanceWithIndexesGPU(double *X,int *indexes,int *heads,double *distance,int n,int d);

void distanceCompareCPU_GPU(double *X,int n,int d);
void distance(double *X,double *dinstance,int n,int d);
