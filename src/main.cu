#include <iostream>
#include <cstdlib>

#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/curand.h>
#include </usr/local/cuda/include/curand_kernel.h>

#include "../include/errorHandling.h"
#include "../include/ptrKernel.h"
#include "../include/evolutionRuleKernel.h"


int main(){

    // Size of the system
    size_t dimension{10};

    // Pointers on host to allocate states of the system
    int *currentStates{nullptr}, *neighbors{nullptr}, *nextStates{nullptr};
    
    // Pointers on host to al

    // Allocate memory on the CPU
    currentStates   = new int[dimension];
    neighbors       = new int[dimension];
    nextStates      = new int[dimension];

    //============================================================================
    //============================ D E V I C E ===================================
    // Pointers to handle device computations 
    int *d_currentStates{nullptr}, *d_neighbors{nullptr}, *d_nextStates{nullptr};

    // Allocate memory on the GPU to handle states
    HANDLE_ERROR(cudaMalloc((void**)&d_currentStates, dimension * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_neighbors, dimension * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_nextStates, dimension * sizeof(int)));

    //-------------------- R A N D O M  N U M B E R S ------------------------------
    // Allocate memory on the GPU to generate initial condition
    float *d_randNumbers{nullptr};
    // Consider dimension random numbers
    HANDLE_ERROR(cudaMalloc((void**)&d_randNumbers, 2 * dimension * sizeof(float)));

    curandGenerator_t gen;
    // Create ther random number generator
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));
    // Set the seed
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,1234ULL));
    // Generate dimension random numbers
    CURAND_CALL(curandGenerateUniform(gen, d_randNumbers, 2 * dimension));
    // Destroy the generator
    CURAND_CALL(curandDestroyGenerator(gen));
    //------------------------------------------------------------------------------

    // Density between states A and B, and between A and C.
    double densityStatesAB{0.3};
    double densityStatesAC{0.3};

    // Call the kernel to configure initial conditions
    configureInitialConditions<<<1,10>>>(dimension, d_currentStates, d_neighbors, d_nextStates, densityStatesAB, densityStatesAC, d_randNumbers);

    

    // ---------------------- E V O L V E  T H E  S Y S T E M --------------------------
    //----------------------------------------------------------------------------------
    size_t time{10};
    for(size_t t{}; t < time; t++){
        // Apply the rule
        Q2RPottsRule<<<1,10>>>(dimension, d_currentStates, d_neighbors, d_nextStates);

        // Ensure kernel is complete before memcpy
        HANDLE_ERROR(cudaDeviceSynchronize()); 

        // VERIFICATION
        // Copy the results to the CPU
        HANDLE_ERROR(cudaMemcpy(currentStates, d_currentStates, dimension * sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(neighbors, d_neighbors, dimension * sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(nextStates, d_nextStates, dimension * sizeof(int), cudaMemcpyDeviceToHost));

        std::cout << "======================================================" << std::endl;
        std::cout << "============== T I M E  "<< t <<"  ===================" << std::endl;
        std::cout << "The elements stored on current states are: " << std::endl;
        displayPtr(dimension,currentStates);
        std::cout << "The elements stored on neighbors are: " << std::endl;
        displayPtr(dimension,neighbors);
        std::cout << "The elements stored on next states are: " << std::endl;
        displayPtr(dimension,nextStates); 

        // Transfer values from neighbors to current states
        HANDLE_ERROR(cudaMemcpy(d_currentStates, d_neighbors, dimension * sizeof(int), cudaMemcpyDeviceToDevice));

        // Transfer the values from next states to neighbors
        HANDLE_ERROR(cudaMemcpy(d_neighbors, d_nextStates, dimension * sizeof(int), cudaMemcpyDeviceToDevice));
    }


    // Copy the results to the CPU
    HANDLE_ERROR(cudaMemcpy(currentStates, d_currentStates, dimension * sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(neighbors, d_neighbors, dimension * sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(nextStates, d_nextStates, dimension * sizeof(int), cudaMemcpyDeviceToHost));
    //============================================================================
    //============================== H O S T =====================================
    // Verify elements on pointers
    /*std::cout << "The elements stored on current states are: " << std::endl;
    displayPtr(50,currentStates);
    std::cout << "The elements stored on neighbors are: " << std::endl;
    displayPtr(50,neighbors);
    std::cout << "The elements stored on next states are: " << std::endl;
    displayPtr(50,nextStates);*/

    //============================================================================
    //============================ D E V I C E ===================================
    // Clean te allocated memory on GPU
    HANDLE_ERROR(cudaFree(d_currentStates));
    HANDLE_ERROR(cudaFree(d_neighbors));
    HANDLE_ERROR(cudaFree(d_nextStates));

    HANDLE_ERROR(cudaFree(d_randNumbers));
    
    //============================================================================

    // Clean the allocated memory on GPU
    delete[] currentStates; currentStates = nullptr;
    delete[] neighbors; neighbors = nullptr;
    delete[] nextStates; nextStates = nullptr;

    return 0;
}