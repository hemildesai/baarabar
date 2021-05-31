#ifndef TRANSFORMER_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define TRANSFORMER_H

// System includes
#include <stdio.h>
#include <assert.h>
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cublas_v2.h>

#include "softmax.hpp"
#include "gemm.hpp"

typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

#endif