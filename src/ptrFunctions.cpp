#include "../include/ptrFunctions.h"
#include <cstdlib>
#include <iostream>


// Display pointer
void displayPtr(size_t ptrSize, int *ptr){
    std::cout << "[" << ptr[0] << ", ";
    for(size_t i{1}; i < ptrSize - 1; i++){
        std::cout << ptr[i] << ", ";
    }
    std::cout << ptr[ptrSize-1] << "]" << std::endl;
}

// Transform from decimal number to ternary base
void decimalToTernary(size_t size, int decimalNumber, int *ptrConf){
    
    int sign = (decimalNumber < 0) ? -1 : 1;
    decimalNumber = std::abs(decimalNumber);

    for(size_t power{}; power < size; power++){

        int remainder = decimalNumber % 3;
        decimalNumber /= 3;

        // Balance the ternary base to -1, 0 and 1
        if (remainder == 2 || remainder == -2) {
            remainder = -1;
            decimalNumber++;
        }

        // Fill the conf. from right to left
        ptrConf[size - 1 - power] = remainder * sign;
    }

    
}

// Compare the elements of two pointers
bool comparePtrs(size_t size, int *ptr1, int *ptr2, int *ptr3, int *ptr4){
    // If the two pointers have the same elements in the same order this function return true
    for(size_t cellIdx{}; cellIdx < size; cellIdx++){
        if(ptr1[cellIdx] != ptr2[cellIdx] || ptr3[cellIdx] != ptr4[cellIdx]){
            return false;
        }
    }
    return true;
}

// Compute neighbors index considering periodic boundary conditions
void idxPeriodicBoudaryCondition(const size_t &size, int *firstNeighborRightIdx, int *secondNeighborRightIdx, int *firstNeighborLeftIdx, int *secondNeighborLeftIdx){

    // Visit each cell on the state's array
    for(size_t idxCell{}; idxCell < size; idxCell++){
        // Mod operation to obtain neighbors
        firstNeighborRightIdx[idxCell]  =   (idxCell + 1) % size;
        secondNeighborRightIdx[idxCell] =   (idxCell + 2) % size;

        firstNeighborLeftIdx[idxCell]   =   (idxCell + size - 1) % size;
        secondNeighborLeftIdx[idxCell]  =   (idxCell + size - 2) % size;
    }
    

}

// Create a new binary file output
FILE* createBinOutput(const char *name){

    FILE* newFile = fopen(name, "wb");
    return newFile;
}

// Verify if the file was open properly
void verifyBinaryOutput(FILE* outFile){

    if(outFile == NULL){
        std::cerr << "Error creating binary output" << std::endl;
    }
}

