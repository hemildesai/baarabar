#include "transformer.hpp"

void ConstantInit(float *data, int size, float val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(int argc, char **argv,
                   int block_size, const dim3 &dimsA,
                   const dim3 &dimsB) {
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = reinterpret_cast<float *>(malloc(mem_size_A));
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = reinterpret_cast<float *>(malloc(mem_size_B));

    // Initialize host memory
    const float valB = 0.01f;
    ConstantInit(h_A, size_A, 1.0f);
    ConstantInit(h_B, size_B, valB);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C = reinterpret_cast<float *>(malloc(mem_size_C));

    if (h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));

    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel
    if (block_size == 16) {
        MatrixMulCUDA<16> <<< grid, threads >>>(d_C, d_A, d_B,
                                                dimsA.x, dimsB.x);
    } else {
        MatrixMulCUDA<32> <<< grid, threads >>>(d_C, d_A, d_B,
                                                dimsA.x, dimsB.x);
    }

    printf("done\n");

    cudaDeviceSynchronize();

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    checkCudaErrors(cudaEventCreate(&start));

    cudaEvent_t stop;
    checkCudaErrors(cudaEventCreate(&stop));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    // Execute the kernel
    int nIter = 300;

    for (int j = 0; j < nIter; j++) {
        if (block_size == 16) {
            MatrixMulCUDA<16> <<< grid, threads >>>(d_C, d_A, d_B,
                                                    dimsA.x, dimsB.x);
        } else {
            MatrixMulCUDA<32> <<< grid, threads >>>(d_C, d_A, d_B,
                                                    dimsA.x, dimsB.x);
        }
    }

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
                               static_cast<double>(dimsA.y) *
                               static_cast<double>(dimsB.x);
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) /
                       (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops," \
        " WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y);

    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6;  // machine zero

    for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
        double abs_err = fabs(h_C[i] - (dimsA.x * valB));
        double dot_length = dimsA.x;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;

        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                   i, h_C[i], dimsA.x * valB, eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    printf("\nNOTE: The CUDA Samples are not meant for performance"\
           "measurements. Results may vary when GPU Boost is enabled.\n");

    if (correct) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}


/**
 * Program main
 */
int main(int argc, char **argv) {
    printf("[Linear Layer Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
            checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("  Note: Outer matrix dimensions of A & B matrices" \
               " must be equal.\n");

        exit(EXIT_SUCCESS);
    }

    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    int dev = findCudaDevice(argc, (const char **)argv);

    int block_size = 32;

    dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
    dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

    // width of Matrix A
    if (checkCmdLineFlag(argc, (const char **)argv, "wA")) {
        dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
    }

    // height of Matrix A
    if (checkCmdLineFlag(argc, (const char **)argv, "hA")) {
        dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
    }

    // width of Matrix B
    if (checkCmdLineFlag(argc, (const char **)argv, "wB")) {
        dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
    }

    // height of Matrix B
    if (checkCmdLineFlag(argc, (const char **)argv, "hB")) {
        dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
    }

    if (dimsA.x != dimsB.y) {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
               dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
                                               dimsB.x, dimsB.y);

    int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);

    exit(matrix_result);
}


// void initializeCUDA(int argc, char **argv, int &devID, int &iSizeMultiple, sMatrixSize &matrix_size)
// {
//     // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
//     cudaError_t error;
//     devID = 0;

//     devID = findCudaDevice(argc, (const char **)argv);

//     if (checkCmdLineFlag(argc, (const char **)argv, "sizemult"))
//     {
//         iSizeMultiple = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
//     }

//     iSizeMultiple = min(iSizeMultiple, 10);
//     iSizeMultiple = max(iSizeMultiple, 1);

//     cudaDeviceProp deviceProp;

//     error = cudaGetDeviceProperties(&deviceProp, devID);

//     if (error != cudaSuccess)
//     {
//         printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
//         exit(EXIT_FAILURE);
//     }

//     printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

//     int block_size = 32;

//     matrix_size.uiWA = 3 * block_size * iSizeMultiple;
//     matrix_size.uiHA = 4 * block_size * iSizeMultiple;
//     matrix_size.uiWB = 2 * block_size * iSizeMultiple;
//     matrix_size.uiHB = 3 * block_size * iSizeMultiple;
//     matrix_size.uiWC = 2 * block_size * iSizeMultiple;
//     matrix_size.uiHC = 4 * block_size * iSizeMultiple;

//     printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
//            matrix_size.uiHA, matrix_size.uiWA,
//            matrix_size.uiHB, matrix_size.uiWB,
//            matrix_size.uiHC, matrix_size.uiWC);

//     if( matrix_size.uiWA != matrix_size.uiHB ||
//         matrix_size.uiHA != matrix_size.uiHC ||
//         matrix_size.uiWB != matrix_size.uiWC)
//     {
//        printf("ERROR: Matrix sizes do not match!\n");
//        exit(-1);
//     }
// }

// int main(int argc, char **argv)
// {
//     printf("[Matrix Multiply CUBLAS] - Starting...\n");

//     int devID = 0, sizeMult = 5;
//     sMatrixSize matrix_size;

//     initializeCUDA(argc, argv, devID, sizeMult, matrix_size);

//     // int matrix_result = matrixMultiply(argc, argv, devID, matrix_size);

//     // return matrix_result;
//     return 0;
// }