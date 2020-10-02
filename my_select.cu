#include "my_select.h"
#include <stdio.h>



__global__ void my_sort(double *X,int *indexes,int *copy_indexes,int *heads,double *distances, int *array_n,int n_all){

  int idx = threadIdx.x +blockDim.x*blockIdx.x;

  if(idx >= n_all )
    return; //exit thread out of bound

  int my_index = copy_indexes[idx]; // find the index

  int my_head = heads[my_index];  // find the heads

  if (my_head < 0 )
    return;  //exit thread, it's a head

  int index_my_head = indexes[my_head]; // find index of head

  int n = array_n[my_index];   // n = lenght of array for idx point

  double my_dist = distances[my_index]; // distance for this point from head

  int midle = (int)(n-1)/2; // the midle of the tree

  int low_num  = n - 1 - midle; // the lower midle values without head
  int lower =  0;   // counter for lower pointers than me

 // main loop - find the midle point
  for(int i=my_head+1 ; (i< n + my_head) && ( lower <= low_num +1  ); i++){

    if (my_head != heads[copy_indexes[i]])
          return;

    double temp = distances[copy_indexes[i]];
    if (temp <= my_dist) lower++;

  }


  if( lower != low_num ) // if you 're not the midle please return and wait
    return;

// job only for the midles for each tree

    double median = my_dist; // median is my distance

    int id_lower= my_head + 1;  // the index for the lower than median distances
    int id_biger= my_head + 1 + low_num;  // the index for the higher than median distances
    int high_num = n - low_num - 1; // number of higher distances

    for(int i=my_head+1 ; i<n + my_head; i++) // for my head index to n points
    {
      int idx_i = copy_indexes[i];

      if(distances[idx_i] <= median){ // if is lower

        indexes[(id_lower++)] = idx_i;
        heads[idx_i] = my_head + 1;
        array_n[idx_i] = low_num;

      }else{  // if is higher

        indexes[(id_biger++)] = idx_i;
        heads[idx_i] = my_head + 1 + low_num;
        array_n[idx_i] = high_num;

      }

      distances[idx_i] = 0;

    }

    distances[indexes[my_head]] = median + 1e-8;
    heads[indexes[my_head + 1]] = heads[index_my_head]*2 ;

    if(high_num != 0 ) 
      heads[indexes[my_head + low_num + 1]] = heads[index_my_head]*2 -1 ;



}
