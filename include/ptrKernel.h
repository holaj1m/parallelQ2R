#ifndef PTR_KERNEL_H
#define PTR_KERNEL_H

#include <stdio.h>
#include </usr/local/cuda/include/cuda_runtime.h>
#include <cstdlib>

__global__ void configureInitialConditions(size_t size, int *statesPtr, int *neighborsPtr, int *evolutionPtr, double densityStatesAB, double densityStatesAC);

void displayPtr(size_t ptrSize, int *ptr);

#endif