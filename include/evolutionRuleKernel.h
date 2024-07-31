#ifndef EVOLUTION_RULE_KERNEL_H
#define EVOLUTION_RULE_KERNEL_H

#include <stdio.h>
#include </usr/local/cuda/include/cuda_runtime.h>
#include <cstdlib>

__global__ void Q2RPottsRule(size_t size, int *statesPtr, int *neighborsPtr, int *evolutionPtr);

__global__ void computeEnergy(size_t size, int *statesPtr, int *neighborsPtr, int *partialEnergy);

#endif