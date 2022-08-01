#include <stdio.h>
#include <math.h>

void distance_calc(int *train, int *test, int *distance);


int main() {   
    
//     int a[5] = {0,0,0,0,0};
//     a[3] = 3;
//     for (int i = 0; i < 5 ; i++){
//          printf("%d ", a[i]);
//     }
//     return 0;
}

void distance_calc(int *train, int *test, int *distance)
{
    int tid = 1;
    // printf(train[i] - test[i]);
    distance[tid] = pow((train[tid] - test[tid]), 2);
    printf("%f\n", pow(train[tid] - test[tid], 2));
    
}
