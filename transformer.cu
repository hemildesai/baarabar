#include "transformer.hpp"
#include <unistd.h>

void ConstantInit(float *data, int size, float val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

void matrixInit(float* mat, int size) {
    for (int i = 0; i < size; i++)
    {
        mat[i] = (((float) rand() / (RAND_MAX)) - 0.5f) * 0.5;
    }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(int block_size, const dim3 &dimsA,
                   const dim3 &dimsB) {
    // Allocate host memory for matrices A and B
    unsigned int size_X = dimsA.x * dimsA.y;
    unsigned int mem_size_X = sizeof(float) * size_X;

    unsigned int size_S = dimsA.y * dimsA.y;
    unsigned int mem_size_S = sizeof(float) * size_S;

    unsigned int size_Wq = dimsB.x * dimsB.y;
    unsigned int mem_size_Wq = sizeof(float) * size_Wq;

    dim3 dimsC(dimsB.x, dimsA.y, 1);
    dim3 dimsCT(dimsA.y, dimsB.x, 1);
    unsigned int mem_size_Q = dimsC.x * dimsC.y * sizeof(float);

    float *h_X = reinterpret_cast<float *>(malloc(mem_size_X));
    float *h_Wq = reinterpret_cast<float *>(malloc(mem_size_Wq));
    float *h_Q = reinterpret_cast<float *>(malloc(mem_size_Q));
    if (h_Q == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host memory
    matrixInit(h_X, size_X);
    matrixInit(h_Wq, size_Wq);

    // Allocate device memory
    float *d_X, *d_Wq, *d_Q;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_X), mem_size_X));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_Wq), mem_size_Wq));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_Q), mem_size_Q));

    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_X, h_X, mem_size_X, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Wq, h_Wq, mem_size_Wq, cudaMemcpyHostToDevice));

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    printf("Computing Query using CUDA Kernel...\n");
    if (block_size == 16) {
        MatrixMulCUDA<16> <<< grid, threads >>>(d_Q, d_X, d_Wq,
                                                dimsA.x, dimsB.x);
    } else {
        MatrixMulCUDA<32> <<< grid, threads >>>(d_Q, d_X, d_Wq,
                                                dimsA.x, dimsB.x);
    }
    printf("done\n");

    // ******* K ********
    float *h_Wk = reinterpret_cast<float *>(malloc(mem_size_Wq));
    float *h_K = reinterpret_cast<float *>(malloc(mem_size_Q));

    if (h_K == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host memory
    matrixInit(h_Wk, size_Wq);

    float *d_Wk, *d_K;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_Wk), mem_size_Wq));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_K), mem_size_Q));

    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_Wk, h_Wk, mem_size_Wq, cudaMemcpyHostToDevice));

    printf("Computing Key using CUDA Kernel...\n");
    if (block_size == 16) {
        MatrixMulCUDA<16> <<< grid, threads >>>(d_K, d_X, d_Wk,
                                                dimsA.x, dimsB.x);
    } else {
        MatrixMulCUDA<32> <<< grid, threads >>>(d_K, d_X, d_Wk,
                                                dimsA.x, dimsB.x);
    }
    printf("done\n");

    // ******* V ********
    float *h_Wv = reinterpret_cast<float *>(malloc(mem_size_Wq));
    float *h_V = reinterpret_cast<float *>(malloc(mem_size_Q));

    if (h_V == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host memory
    matrixInit(h_Wv, size_Wq);

    float *d_Wv, *d_V;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_Wv), mem_size_Wq));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_V), mem_size_Q));

    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_Wv, h_Wv, mem_size_Wq, cudaMemcpyHostToDevice));

    printf("Computing Value using CUDA Kernel...\n");
    if (block_size == 16) {
        MatrixMulCUDA<16> <<< grid, threads >>>(d_V, d_X, d_Wv,
                                                dimsA.x, dimsB.x);
    } else {
        MatrixMulCUDA<32> <<< grid, threads >>>(d_V, d_X, d_Wv,
                                                dimsA.x, dimsB.x);
    }
    printf("done\n");

    // ******* K Transpose ********
    float *d_KT;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_KT), mem_size_Q));

    printf("Computing K Transpose using CUDA Kernel...\n");
    dim3 gT(dimsC.x / block_size, dimsC.y / block_size, 1);
    dim3 tT(block_size, block_size, 1);
    if (block_size == 16) {
        transpose<16> <<< gT, tT >>> (d_KT, d_K, dimsC.x, dimsC.y);
    } else {
        transpose<32> <<< gT, tT >>> (d_KT, d_K, dimsC.x, dimsC.y);
    }
    printf("done\n");


    // ******* Softmax ********
    float *h_S = reinterpret_cast<float *>(malloc(mem_size_S));

    if (h_S == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }
    float *d_S, *d_SS;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_S), mem_size_S));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_SS), mem_size_S));

    printf("Computing QK using CUDA Kernel...\n");

    dim3 tS(block_size, block_size);
    dim3 gS(dimsCT.x / threads.x, dimsC.y / threads.y);
    printf("dimsCT.x %d dimsC.y %d dimsC.x %d\n", dimsCT.x, dimsC.y, dimsC.x);
    if (block_size == 16) {
        MatrixMulCUDA<16> <<< gS, tS >>>(d_S, d_Q, d_KT,
                                                dimsC.x, dimsCT.x);
    } else {
        MatrixMulCUDA<32> <<< gS, tS >>>(d_S, d_Q, d_KT,
                                                dimsC.x, dimsCT.x);
    }

    printf("done\n");

    dim3 threads_ss(1, BLOCK_SIZE);
    dim3 grid_ss(ceil((precision) 32 / threads_ss.x), ceil((precision) 32 / threads_ss.y));
    Softmax <<< grid_ss, threads_ss >>> (d_S, d_SS, 32, 32);

    // cudaDeviceSynchronize();

    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(h_Q, d_Q, mem_size_Q, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_K, d_K, mem_size_Q, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_V, d_V, mem_size_Q, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_S, d_SS, mem_size_S, cudaMemcpyDeviceToHost));

    printf("h_Q[0] %f h_K[0] %f h_V[0] %f h_S[0][2] %f\n", h_Q[0], h_K[0], h_V[0], h_S[2]);
    for (int i = 64; i < 96; i++)
    {
        printf("h_S[%d] %f\n", i, h_S[i]);
    }


    // Clean up memory
    free(h_X);
    free(h_Wq);
    free(h_Q);
    free(h_Wk);
    free(h_K);
    free(h_Wv);
    free(h_V);
    checkCudaErrors(cudaFree(d_X));
    checkCudaErrors(cudaFree(d_Wq));
    checkCudaErrors(cudaFree(d_Q));
    checkCudaErrors(cudaFree(d_Wk));
    checkCudaErrors(cudaFree(d_K));
    checkCudaErrors(cudaFree(d_Wv));
    checkCudaErrors(cudaFree(d_V));

    printf("\nNOTE: The CUDA Samples are not meant for performance"\
           "measurements. Results may vary when GPU Boost is enabled.\n");

    return 0;
}


/**
 * Program main
 */
int main(int argc, char **argv) {
    printf("[Transformer Using CUDA] - Starting...\n");

    // if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
    //         checkCmdLineFlag(argc, (const char **)argv, "?")) {
    //     printf("Usage -device=n (n >= 0 for deviceID)\n");
    //     printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
    //     printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
    //     printf("  Note: Outer matrix dimensions of A & B matrices" \
    //            " must be equal.\n");

    //     exit(EXIT_SUCCESS);
    // }

    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    int dev = findCudaDevice(argc, (const char **)argv);

    int block_size = 32;

    dim3 dimsA(768, 32, 1);
    dim3 dimsB(64, 768, 1);

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
                                               dimsB.x, dimsB.y);

    int matrix_result = MatrixMultiply(block_size, dimsA, dimsB);

    // int block_size = 16;
    int n = 32;
    int D = 768;
    int dk = 64;

    printf("n %d D %d dk %d\n", n, D, dk);

    // MultiHeadAttention(n, D, dk, block_size);

    // exit(matrix_result);
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