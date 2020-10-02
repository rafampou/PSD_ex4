#include "my_select.h"
#include <stdio.h>



__global__ void my_sort(double *X,int *indexes,int *copy_indexes,int *heads,double *distances, int *array_n,int n_all){

  int idx = threadIdx.x +blockDim.x*blockIdx.x;



  if(idx >= n_all )
    return; //exit thread out of bound

  int my_index = copy_indexes[idx];

  int my_head = heads[my_index];

  if (my_head < 0 )
    return;  //exit thread, it's a head


  int index_my_head = indexes[my_head];

  int n = array_n[my_index];   // n = lenght of array for idx point

  double my_dist = distances[my_index];

  int midle = (int)(n-1)/2;

  int low_num  = n - 1 - midle;
  int lower =  0;   // the lower midle values without head,



  for(int i=my_head+1 ; (i< n + my_head) && ( lower <= low_num +1  ); i++){

    if (my_head != heads[copy_indexes[i]])
          return;

    double temp = distances[copy_indexes[i]];
    if (temp <= my_dist) lower++;

  }

  if( lower != low_num )
    return;


    double median = my_dist;

    int id_lower= my_head + 1;
    int id_biger= my_head + 1 + low_num;
    int high_num = n - low_num - 1;

    for(int i=my_head+1 ; i<n + my_head; i++)
    {
      int idx_i = copy_indexes[i];

      if(distances[idx_i] <= median){

        indexes[(id_lower++)] = idx_i;
        heads[idx_i] = my_head + 1;
        array_n[idx_i] = low_num;

      }else{

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
