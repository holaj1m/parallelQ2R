#include </usr/local/cuda/include/cuda_runtime.h>
#include "../include/ptrKernel.h"
#include <cstdlib>
#include <iostream>


//DEFINE drand48 TO OBTAIN A RANDOM NUMBER
#define drand48() ((float)rand()/(float)RAND_MAX)


__global__ void configureInitialConditions(size_t size, int *statesPtr, int *neighborsPtr, int *evolutionPtr, double densityStatesAB, double densityStatesAC){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size) {
        // Create a variable that will determines if the state is or not zero
        int selectZero{1- int(drand48() + densityStatesAB)};

        // Initialize states randomly considering states density
        if(selectZero == 0){statesPtr[tid] = 0;}
        else{statesPtr[tid] = 1 - 2 * int(drand48() + densityStatesAC);}

        // As initial condition we impose that neighbors are equal to states for the first step
        neighborsPtr[tid] = statesPtr[tid];

        // Finally replace the garbage on evolution buffer
        evolutionPtr[tid] = 5;

        // Check for evelements out of range
        tid += blockDim.x * gridDim.x;
    }
}

// Display pointer
void displayPtr(size_t ptrSize, int *ptr){
    std::cout << "[" << ptr[0] << ", ";
    for(size_t i{}; i < ptrSize - 1; i++){
        std::cout << ptr[i] << ", ";
    }
    std::cout << ptr[ptrSize-1] << "]" << std::endl;
}
