#ifndef TRANSFORMER_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define TRANSFORMER_H

// System includes
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <thread>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cublas_v2.h>

#include "softmax.hpp"
#include "gemm.hpp"

using std::string;

typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

static __inline__ uint64_t gettime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (((uint64_t)tv.tv_sec) * 1000000 + ((uint64_t)tv.tv_usec));
}

static uint64_t usec;

__attribute__ ((noinline))  void begin_roi() {
  usec=gettime();
}

__attribute__ ((noinline))  void end_roi(string message)   {
//   usec=(gettime()-usec);
  std::cout << message << "elapsed (sec): " << (gettime() - usec)/1000000.0 << "\n";
}

#endif