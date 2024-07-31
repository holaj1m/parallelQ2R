#include </usr/local/cuda/include/cuda_runtime.h>
#include <cstdlib>
#include <vector>


#include "../include/evolutionRuleKernel.h"

__global__ void Q2RPottsRule(size_t size, int *statesPtr, int *neighborsPtr, int *evolutionPtr){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size) {
        // Mod operation to obtain neighbors
        size_t firstNeighborRightIdx  =   (tid + 1) % size;
        size_t secondNeighborRightIdx =   (tid + 2) % size;

        size_t firstNeighborLeftIdx   =   (tid + size - 1) % size;
        size_t secondNeighborLeftIdx  =   (tid + size - 2) % size;

        // Initialize the current state and its neighbors
        int currentState{statesPtr[tid]};
        int firstNeighborRight{neighborsPtr[firstNeighborRightIdx]};
        int secondNeighborRight{neighborsPtr[secondNeighborRightIdx]};
        int firstNeighborLeft{neighborsPtr[firstNeighborLeftIdx]};
        int secondNeighborLeft{neighborsPtr[secondNeighborLeftIdx]};

        // Create an array to store the neighborhood
        int neighborhood[4] = {firstNeighborRight, secondNeighborRight, firstNeighborLeft, secondNeighborLeft};

        // Create variables to count the frequency of each state on neighborhood 
        int freqStateA{}, freqStateB{}, freqStateC{};

        //Count the frequency
        for(size_t element{}; element < 4; element++){
            switch (neighborhood[element])
            {
            case -1: freqStateA++; break;
            case 0: freqStateB++; break;
            default: freqStateC++;
            }
        }

        // Update the evolution state considering the different frequency among neighborhood
        if(freqStateA == 4 || freqStateB == 4 || freqStateC == 4){
        evolutionPtr[tid] = currentState;
        }

        else if(freqStateA < 3 && freqStateB < 3 && freqStateC < 3){

            if(freqStateA == 0 || freqStateB == 0 || freqStateC == 0){

                if(freqStateA == 2 && freqStateB == 2){
                    switch(currentState){
                        case -1: evolutionPtr[tid] = 0; break;
                        case 0: evolutionPtr[tid] = -1; break;
                        default: evolutionPtr[tid] = 1; break;
                    }
                }

                else if(freqStateA == 2 && freqStateC == 2){
                    switch(currentState){
                        case -1: evolutionPtr[tid] = 1; break;
                        case 1: evolutionPtr[tid] = -1; break;
                        default: evolutionPtr[tid] = 0; 
                    }
                }

                else if(freqStateB == 2 && freqStateC == 2){
                    switch(currentState){
                        case 0: evolutionPtr[tid] = 1; break;
                        case 1: evolutionPtr[tid] = 0; break;
                        default: evolutionPtr[tid] = -1;
                    }
                }

            }

            else{
                if(freqStateA == 2){
                    switch(currentState){
                        case -1: evolutionPtr[tid] = -1; break;
                        case 0: evolutionPtr[tid] = 1; break;
                        default: evolutionPtr[tid] = 0;
                    }
                }

                else if(freqStateB == 2){
                    switch(currentState){
                        case 0: evolutionPtr[tid] = 0; break;
                        case 1: evolutionPtr[tid] = -1; break;
                        default: evolutionPtr[tid] = 1; 
                    }
                }

                else if(freqStateC == 2){
                    switch(currentState){
                        case 1: evolutionPtr[tid] = 1; break;
                        case -1: evolutionPtr[tid] = 0; break;
                        default: evolutionPtr[tid] = -1;
                    }
                }

            }

        }
        else{evolutionPtr[tid] = currentState;}

        // Check for elements out of range
        tid += blockDim.x * gridDim.x;
    }
}



__global__ void computeEnergy(size_t size, int *statesPtr, int *neighborsPtr, int *partialEnergy){
    
    extern __shared__ int cache[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    // Create variable to store the contribution of the state to the total energy
    int localEnergy{};

    while(tid < size){
        // Mod operation to obtain neighbors
        size_t firstNeighborRightIdx  =   (tid + 1) % size;
        size_t secondNeighborRightIdx =   (tid + 2) % size;

        size_t firstNeighborLeftIdx   =   (tid + size - 1) % size;
        size_t secondNeighborLeftIdx  =   (tid + size - 2) % size;

        // Initialize the current state and its neighbors
        int currentState{statesPtr[tid]};
        int firstNeighborRight{neighborsPtr[firstNeighborRightIdx]};
        int secondNeighborRight{neighborsPtr[secondNeighborRightIdx]};
        int firstNeighborLeft{neighborsPtr[firstNeighborLeftIdx]};
        int secondNeighborLeft{neighborsPtr[secondNeighborLeftIdx]};

        // Create an array to store the neighborhood
        int neighborhood[4] = {firstNeighborRight, secondNeighborRight, firstNeighborLeft, secondNeighborLeft};

        
        //Count the repeated elements among the neighborhood
        for(size_t element{}; element < 4; element++){
            if(currentState == neighborhood[element]){
                localEnergy += 1;
            }
        }

        // Check for elements out of range
        tid += blockDim.x * gridDim.x;
    }

    // set the cache values
    cache[cacheIdx] = localEnergy;
    
    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIdx < i)
            cache[cacheIdx] += cache[cacheIdx + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIdx == 0){partialEnergy[blockIdx.x] = cache[0];}
        
}

