#ifndef PTR_FUNCTIONS_H
#define PTR_FUNCTIONS_H

#include <stdio.h>
#include <cstdlib>

void displayPtr(size_t ptrSize, int *ptr);

void decimalToTernary(size_t size, int decimalNumber, int *ptrConf);

bool comparePtrs(size_t size, int *ptr1, int *ptr2, int *ptr3, int *ptr4);

void idxPeriodicBoudaryCondition(const size_t &size, int *firstNeighborRightIdx, int *secondNeighborRightIdx, int *firstNeighborLeftIdx, int *secondNeighborLeftIdx);

FILE* createBinOutput(const char *name);

void verifyBinaryOutput(FILE* outFile);

#endif