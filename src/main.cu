#include <iostream>
#include <cstdlib>
#include <cmath> 
#include <string>
#include <cstring>

#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/curand.h>
#include </usr/local/cuda/include/curand_kernel.h>

#include "../include/errorHandling.h"
#include "../include/ptrFunctions.h"
#include "../include/evolutionRuleKernel.h"

// Function that computes the minimum value of two input
#define imin(a,b) (a<b?a:b)

int main(){

    // Size of the system
    size_t dimension{10};

    // Define blocks and threads
    const size_t threadsPerBlock = 10;
    const size_t blocksPerGrid = 1;

    // Pointers on host to allocate states of the system
    int *currentStates{nullptr}, *neighbors{nullptr}, *nextStates{nullptr};

    // Pointers on host to save the initial condition
    int *initialCondStates{nullptr};
    int *initialCondNeigh{nullptr}; 
    
    // Allocate memory on the CPU
    currentStates   = new int[dimension];
    neighbors       = new int[dimension];
    nextStates      = new int[dimension];

    initialCondStates   = new int[dimension];
    initialCondNeigh    = new int[dimension];

    //================================================================
    //======================== P B C =================================
    // Pointers on host to allocate index of neighbors for each state
    int *ptrFirstNeighborRightIdx{nullptr}, *ptrSecondNeighborRightIdx{nullptr};
    int *ptrFirstNeighborLeftIdx{nullptr}, *ptrSecondNeighborLeftIdx{nullptr};

    // Pointers on host to store index of each neighbor for each state
    ptrFirstNeighborRightIdx   =   new int[dimension];
    ptrSecondNeighborRightIdx  =   new int[dimension];

    ptrFirstNeighborLeftIdx    =   new int[dimension];
    ptrSecondNeighborLeftIdx   =   new int[dimension];

    // Compute the idx of the neighbor for each position in the array
    idxPeriodicBoudaryCondition(dimension, ptrFirstNeighborRightIdx, ptrSecondNeighborRightIdx, ptrFirstNeighborLeftIdx, ptrSecondNeighborLeftIdx);

    //====================================================================================
    //==================== C O N F.  P A R A M E T E R S =================================
    
    // COmpute the total number of configurations
    double max{(pow(3,dimension) - 1) * 0.5};
    int maxConf{int(max)};
    int minConf{-1 * maxConf};

    // Variables to overwrite the máx. period
    int energyMaxP{0}, maxP{0};
    int *initialCondStatesMaxP{nullptr}, *initialCondNeighMaxP{nullptr};

    // Variables to overwrite the constrained máx. period
     int energyMaxPCons{0}, maxPCons{0};
    int *initialCondStatesMaxPCons{nullptr}, *initialCondNeighMaxPCons{nullptr};

    initialCondStatesMaxP       = new int[dimension];
    initialCondNeighMaxP        = new int[dimension];
    initialCondStatesMaxPCons   = new int[dimension];
    initialCondNeighMaxPCons    = new int[dimension];

    //============================================================================
    //============================ D E V I C E ===================================
    
    // Pointers to handle device computations 
    int *d_currentStates{nullptr}, *d_neighbors{nullptr}, *d_nextStates{nullptr};

    // Allocate memory on the GPU to handle states
    HANDLE_ERROR(cudaMalloc((void**)&d_currentStates, dimension * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_neighbors, dimension * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_nextStates, dimension * sizeof(int)));

    // Pointers to allocate index of neighbors for each state
    int *d_ptrFirstNeighborRightIdx{nullptr}, *d_ptrSecondNeighborRightIdx{nullptr};
    int *d_ptrFirstNeighborLeftIdx{nullptr}, *d_ptrSecondNeighborLeftIdx{nullptr};

    // Allocate memory on the GPU to handle index of neighbors
    HANDLE_ERROR(cudaMalloc((void**)&d_ptrFirstNeighborRightIdx, dimension * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_ptrSecondNeighborRightIdx, dimension * sizeof(int)));

    HANDLE_ERROR(cudaMalloc((void**)&d_ptrFirstNeighborLeftIdx, dimension * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_ptrSecondNeighborLeftIdx, dimension * sizeof(int)));

    // Copy the computed index neighbors from the host to the device
    HANDLE_ERROR(cudaMemcpy(d_ptrFirstNeighborRightIdx, ptrFirstNeighborRightIdx, dimension * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_ptrSecondNeighborRightIdx, ptrSecondNeighborRightIdx, dimension * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_ptrFirstNeighborLeftIdx, ptrFirstNeighborLeftIdx, dimension * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_ptrSecondNeighborLeftIdx, ptrSecondNeighborLeftIdx, dimension * sizeof(int), cudaMemcpyHostToDevice));


    // Variables to compute the energy
    int *partialEnergy{nullptr};

    partialEnergy = new int[blocksPerGrid];

    int *d_partialEnergy{nullptr};

    HANDLE_ERROR(cudaMalloc((void**)&d_partialEnergy, blocksPerGrid * sizeof(int)));

    //====================================================================================
    //======================== E V O L V E  T H E  S Y S T E M ===========================

    for(int numberStates{minConf}; numberStates <= maxConf; numberStates++){

        for(int numberNeighbors{numberStates}; numberNeighbors <= maxConf; numberNeighbors++){

            // Set the initial condition for the system
            decimalToTernary(dimension, numberStates, currentStates);
            decimalToTernary(dimension, numberNeighbors, neighbors);

            // Store the initial condition in the system
            memcpy(initialCondStates, currentStates, dimension * sizeof(int));
            memcpy(initialCondNeigh, neighbors, dimension * sizeof(int));

            // Copy the initial states and neighbors to the device
            HANDLE_ERROR(cudaMemcpy(d_currentStates, currentStates, dimension * sizeof(int), cudaMemcpyHostToDevice));
            HANDLE_ERROR(cudaMemcpy(d_neighbors, neighbors, dimension * sizeof(int), cudaMemcpyHostToDevice));

            // Compute the energy on device
            computeEnergy<<<blocksPerGrid,threadsPerBlock, threadsPerBlock * sizeof(int)>>>(dimension, d_currentStates, d_neighbors, d_ptrFirstNeighborRightIdx, d_ptrSecondNeighborRightIdx, d_ptrFirstNeighborLeftIdx, d_ptrSecondNeighborLeftIdx, d_partialEnergy);

            // copy computed partial energy on host
            HANDLE_ERROR(cudaMemcpy(partialEnergy, d_partialEnergy, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));

            // sum the partial energy to get the total one
            int energy{};
            for(size_t i{}; i < blocksPerGrid; i++){
                energy += partialEnergy[i];
            }
    
            // Variable that count the period of the configuration
            int period{};

            // EVOLUTION OF THE SYSTEM
    
            while(true){
                // Add a period
                period++;

                //displayPtr(dimension,currentStates);
                //displayPtr(dimension,neighbors);

                // Apply the rule
                Q2RPottsRule<<<blocksPerGrid,threadsPerBlock>>>(dimension, d_currentStates, d_neighbors, d_ptrFirstNeighborRightIdx, d_ptrSecondNeighborRightIdx, d_ptrFirstNeighborLeftIdx, d_ptrSecondNeighborLeftIdx, d_nextStates);

                // Ensure kernel is complete before memcpy
                HANDLE_ERROR(cudaDeviceSynchronize()); 

                // Copy the results evolved to the CPU
                HANDLE_ERROR(cudaMemcpy(currentStates, d_neighbors, dimension * sizeof(int), cudaMemcpyDeviceToHost));
                HANDLE_ERROR(cudaMemcpy(neighbors, d_nextStates, dimension * sizeof(int), cudaMemcpyDeviceToHost));
                HANDLE_ERROR(cudaMemcpy(nextStates, d_nextStates, dimension * sizeof(int), cudaMemcpyDeviceToHost));

                //displayPtr(dimension,nextStates);

                // Verify if the cycle was closed
                if(comparePtrs(dimension, initialCondStates, currentStates, initialCondNeigh, neighbors)){

                    // Overwrite the data if the current period is greater than the previous setted one
                    if(period > maxP){
                        energyMaxP  = energy;
                        maxP        = period;
                        memcpy(initialCondStatesMaxP, initialCondStates, dimension * sizeof(int));
                        memcpy(initialCondNeighMaxP, initialCondNeigh, dimension * sizeof(int));
                    }

                    // Overwrite the data if the current period is greater than the previous setted one for the constrained cond.
                    if(period > maxPCons && (numberStates == numberNeighbors)){
                        energyMaxPCons  = energy;
                        maxPCons        = period;
                        memcpy(initialCondStatesMaxPCons, initialCondStates, dimension * sizeof(int));
                        memcpy(initialCondNeighMaxPCons, initialCondNeigh, dimension * sizeof(int));
                    }

                    std::cout << "(" << numberStates << ", " << numberNeighbors << ")" << std::endl;

                    // Break the loop
                    break;
                }

                // Permute currentStates with neighbors and neighbors with nextStates to evolve one step on device

                // Transfer values from neighbors to current states
                HANDLE_ERROR(cudaMemcpy(d_currentStates, d_neighbors, dimension * sizeof(int), cudaMemcpyDeviceToDevice));
                // Transfer the values from next states to neighbors
                HANDLE_ERROR(cudaMemcpy(d_neighbors, d_nextStates, dimension * sizeof(int), cudaMemcpyDeviceToDevice));

            }

        }
    }

    //===============================================================
    //========================= F I L E S ===========================

    // Binary files to save the evolution of configuration in ternary configuration
    FILE* periodFile    = createBinOutput("maxP.bin");
    FILE* energyFile    = createBinOutput("energyMaxP.bin");
    FILE* statesFile    = createBinOutput("statesMaxP.bin");
    FILE* neighFile     = createBinOutput("neighMaxP.bin");


    // Verify outputs
    verifyBinaryOutput(periodFile);
    verifyBinaryOutput(energyFile);
    verifyBinaryOutput(statesFile);
    verifyBinaryOutput(neighFile);
    
    // Save the period and energy of the configuration
    fwrite(&maxP, sizeof(int), 1, periodFile);
    fwrite(&energyMaxP, sizeof(int), 1, energyFile);

    // Save the configuration of max. P
    fwrite(initialCondStatesMaxP, sizeof(int), dimension, statesFile);
    fwrite(initialCondNeighMaxP, sizeof(int), dimension, neighFile);

    // Close the files
    fclose(energyFile);
    fclose(periodFile);
    fclose(statesFile);
    fclose(neighFile);

    // Files for the constrained condition
    FILE* periodFileCons    = createBinOutput("maxPCons.bin");
    FILE* energyFileCons    = createBinOutput("energyMaxPCons.bin");
    FILE* statesFileCons    = createBinOutput("statesMaxPCons.bin");
    FILE* neighFileCons     = createBinOutput("neighMaxPCons.bin");

    // Verify outputs
    verifyBinaryOutput(periodFileCons);
    verifyBinaryOutput(energyFileCons);
    verifyBinaryOutput(statesFileCons);
    verifyBinaryOutput(neighFileCons);

    // Save the period and energy of the configuration constrained
    fwrite(&maxPCons, sizeof(int), 1, periodFileCons);
    fwrite(&energyMaxPCons, sizeof(int), 1, energyFileCons);

    // Save the configuration of max. P constrained
    fwrite(initialCondStatesMaxPCons, sizeof(int), dimension, statesFileCons);
    fwrite(initialCondNeighMaxPCons, sizeof(int), dimension, neighFileCons);

    // Close the files
    fclose(energyFileCons);
    fclose(periodFileCons);
    fclose(statesFileCons);
    fclose(neighFileCons);

    //===============================================================
    //================ C L E A N I N G  M E M O R Y =================

    //============================ D E V I C E ===================================
    // Clean te allocated memory on GPU
    HANDLE_ERROR(cudaFree(d_currentStates));
    HANDLE_ERROR(cudaFree(d_neighbors));
    HANDLE_ERROR(cudaFree(d_nextStates));

    HANDLE_ERROR(cudaFree(d_ptrFirstNeighborRightIdx));
    HANDLE_ERROR(cudaFree(d_ptrSecondNeighborRightIdx));

    HANDLE_ERROR(cudaFree(d_ptrFirstNeighborLeftIdx));
    HANDLE_ERROR(cudaFree(d_ptrSecondNeighborLeftIdx));

    HANDLE_ERROR(cudaFree(d_partialEnergy));
    
    //============================ H O S T ===================================
    // Clean the allocated memory on CPU
    delete[] currentStates; currentStates = nullptr;
    delete[] neighbors; neighbors = nullptr;
    delete[] nextStates; nextStates = nullptr;

    delete[] initialCondStates; initialCondStates   = NULL;
    delete[] initialCondNeigh; initialCondNeigh   = NULL;

    delete[] ptrFirstNeighborRightIdx;  ptrFirstNeighborRightIdx    = NULL;
    delete[] ptrFirstNeighborLeftIdx;   ptrFirstNeighborLeftIdx     = NULL;
    delete[] ptrSecondNeighborRightIdx; ptrSecondNeighborRightIdx   = NULL;
    delete[] ptrSecondNeighborLeftIdx;  ptrSecondNeighborLeftIdx    = NULL;

    delete[] initialCondStatesMaxP; initialCondStatesMaxP   = NULL;
    delete[] initialCondNeighMaxP; initialCondNeighMaxP   = NULL;

    delete[] initialCondStatesMaxPCons; initialCondStatesMaxPCons   = NULL;
    delete[] initialCondNeighMaxPCons; initialCondNeighMaxPCons   = NULL;

    return 0;
}