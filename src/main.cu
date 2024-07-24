#include <iostream>
#include <cstdlib>

#include "../include/errorHandling.h"
#include "../include/ptrKernel.h"


int main(){

    // Size of the system
    size_t dimension{1024};

    // Pointers on host to allocate states of the system
    int *currentStates{nullptr}, *neighbors{nullptr}, *nextStates{nullptr};

    // Allocate memory on the CPU
    currentStates   = new int[dimension];
    neighbors       = new int[dimension];
    nextStates      = new int[dimension];

    //============================================================================
    //============================ D E V I C E ===================================
    // Pointers to handle device computations 
    int *d_currentStates{nullptr}, *d_neighbors{nullptr}, *d_nextStates{nullptr};

    // Allocate memory on the GPU
    HANDLE_ERROR(cudaMalloc((void**)&d_currentStates, dimension * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_neighbors, dimension * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_nextStates, dimension * sizeof(int)));

    // Density between states A and B, and between A and C.
    double densityStatesAB{0.035};
    double densityStatesAC{0.025};

    // Call the kernel to configure initial conditions
    configureInitialConditions<<<32,32>>>(dimension, d_currentStates, d_neighbors, d_nextStates, densityStatesAB, densityStatesAC);

    // Copy the results to the CPU
    HANDLE_ERROR(cudaMemcpy(currentStates, d_currentStates, dimension * sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(neighbors, d_neighbors, dimension * sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(nextStates, d_nextStates, dimension * sizeof(int), cudaMemcpyDeviceToHost));
    //============================================================================
    //============================== H O S T =====================================
    // Verify elements on pointers
    std::cout << "The elements stored on current states are: " << std::endl;
    displayPtr(10,currentStates);
    std::cout << "The elements stored on neighbors are: " << std::endl;
    displayPtr(10,neighbors);
    std::cout << "The elements stored on next states are: " << std::endl;
    displayPtr(10,nextStates);





    //============================================================================
    //============================ D E V I C E ===================================
    // Clean te allocated memory on GPU
    HANDLE_ERROR(cudaFree(d_currentStates));
    HANDLE_ERROR(cudaFree(d_neighbors));
    HANDLE_ERROR(cudaFree(d_nextStates));
    //============================================================================

    // Clean the allocated memory on GPU
    delete[] currentStates; currentStates = nullptr;
    delete[] neighbors; neighbors = nullptr;
    delete[] nextStates; nextStates = nullptr;

    return 0;
}