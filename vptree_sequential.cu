#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include "vptree.h"

//typedef struct vptree
//{
//    double *vp;
//    double md;
//    int idx;
//    struct vptree *inner, *outer;
//} vptree ;

#define SWAP(x, y) { double temp = x; x = y; y = temp; }

int partition(double *a, int left, int right, int pivotIndex)
{
	double pivot = a[pivotIndex];
	SWAP(a[pivotIndex], a[right]);
	int pIndex = left;
	int i;
	for (i = left; i < right; i++)
	{
		if (a[i] <= pivot)
		{
			SWAP(a[i], a[pIndex]);
			pIndex++;
		}
	}
	SWAP(a[pIndex], a[right]);
	return pIndex;
}

double quickselect(double *A, int left, int right, int k)
{
	if (left == right)
		return A[left];
	int pivotIndex = left + rand() % (right - left + 1);
	pivotIndex = partition(A, left, right, pivotIndex);
	if (k == pivotIndex)
		return A[k];
	else if (k < pivotIndex)
		return quickselect(A, left, pivotIndex - 1, k);
	else
		return quickselect(A, pivotIndex + 1, right, k);
}

void CalculateDistance(double *X, int n, int d, double *distance, double *distanceCopy)
{
    double distanceSquare = 0;
    for(int i = 0; i < n - 1; i++){
        for(int j = 0; j < d; j++){
            distanceSquare += pow(X[(n - 1)*d + j] - X[i*d + j], 2);
        }
        distance[i] = sqrt(distanceSquare);
        distanceCopy[i] = distance[i];
        distanceSquare = 0;
    }
}

vptree * MyBuildvp(double *X, int n, int d, int *idx)
{
    vptree *tree;
    double *distance, *distanceCopy, medianDistance;
    int nInner, nOuter;

    tree         = (vptree*) malloc(   sizeof(vptree));
    tree->vp     = (double*) malloc( d*sizeof(double));
    tree->inner  = (vptree*) malloc(   sizeof(vptree));
    tree->outer  = (vptree*) malloc(   sizeof(vptree));

    distance     = (double*) malloc( (n - 1)*sizeof(double));
    distanceCopy = (double*) malloc( (n - 1)*sizeof(double));

    nInner = 0; nOuter = 0;

    if ( n == 1 )
    {
       for( int i = 0; i < d; i++)
       {
          (tree->vp)[i] = X[i];
       }
       tree->idx   = idx[0];
       tree->md    = 0.0;
       tree->inner = NULL;
       tree->outer = NULL;

       return tree;
    }

    CalculateDistance(X, n, d, distance, distanceCopy);

    medianDistance = quickselect(distanceCopy, 0, n - 2, (int)(n/2 - 1));

    free(distanceCopy);

    for ( int i = 0; i < d; i++)
    {
      (tree->vp)[i] = X[(n - 1)*d + i];
    }
    tree->idx = idx[n - 1];
    tree->md = medianDistance;

    for ( int i = 0; i < n - 1; i++ )
    {
        if ( distance[i] <= medianDistance )
        {
           nInner++;
        }
        else
        {
           nOuter++;
        }
    }

    double *XInner, *XOuter;
    int *idxInner, *idxOuter, counterInner, counterOuter;

    XInner = (double*) malloc( nInner*d*sizeof(double));
    XOuter = (double*) malloc( nOuter*d*sizeof(double));
    idxInner  = (int*) malloc( nInner*sizeof(int));
    idxOuter  = (int*) malloc( nOuter*sizeof(int));
    counterInner = 0; counterOuter = 0;

    for ( int i = 0; i < n - 1; i++)
    {
        if ( distance[i] <= medianDistance )
        {
            for ( int j = 0; j < d; j++)
            {
               XInner[counterInner*d + j] = X[i*d + j];
            }
            idxInner[counterInner] = idx[i];
            counterInner++;
        }
        else
        {
            for(int j = 0; j < d; j++)
            {
               XOuter[counterOuter*d + j] = X[i*d + j];
            }
            idxOuter[counterOuter] = idx[i];
            counterOuter++;
        }
    }

    free(X);
    free(idx);
    free(distance);

    if ( nInner == 0 )
    {
       tree->inner = NULL;
    }
    else
    {
       tree->inner = MyBuildvp(XInner, nInner, d, idxInner);
    }


    if ( nOuter == 0)
    {
       tree->outer = NULL;
    }
    else
    {
       tree->outer = MyBuildvp(XOuter, nOuter, d, idxOuter);
    }


    return tree;
}

vptree * getInner(vptree * T){
    return T->inner;
}

vptree * getOuter(vptree * T){
    return T->outer;
}

double getMD(vptree * T){
    return T->md;
}

double * getVP(vptree * T){
    return T->vp;
}

int getIDX(vptree * T){
    return T->idx;
}

vptree * buildvp(double *X, int n, int d)
{
    // variables to hold execution time
    int *idx = (int *) malloc( n*sizeof(int));
    vptree *tree;

    for(int i = 0; i < n; i++){
        idx[i] = i;
    }


    tree = MyBuildvp(X,n,d,idx);




    return tree;
}

char *vptree_version( void ){
  return (char *)"sequential";
}
