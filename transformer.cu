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
 * Run Single Head Attention using CUDA
 */
void SingleHeadAttention(int block_size, const dim3 &dimsA,
                   const dim3 &dimsB, float *h_X) {
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

    // float *h_X = reinterpret_cast<float *>(malloc(mem_size_X));
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

    // ******* Q ********

    printf("Computing Query using CUDA Kernel...\n");
    if (block_size == 16) {
        MatrixMulCUDA<16> <<< grid, threads >>>(d_Q, d_X, d_Wq,
                                                dimsA.x, dimsB.x, 1);
    } else {
        MatrixMulCUDA<32> <<< grid, threads >>>(d_Q, d_X, d_Wq,
                                                dimsA.x, dimsB.x, 1);
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
                                                dimsA.x, dimsB.x, 1);
    } else {
        MatrixMulCUDA<32> <<< grid, threads >>>(d_K, d_X, d_Wk,
                                                dimsA.x, dimsB.x, 1);
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
                                                dimsA.x, dimsB.x, 1);
    } else {
        MatrixMulCUDA<32> <<< grid, threads >>>(d_V, d_X, d_Wv,
                                                dimsA.x, dimsB.x, 1);
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
                                                dimsC.x, dimsCT.x, 8);
    } else {
        MatrixMulCUDA<32> <<< gS, tS >>>(d_S, d_Q, d_KT,
                                                dimsC.x, dimsCT.x, 8);
    }

    printf("done\n");

    dim3 threads_ss(1, BLOCK_SIZE);
    dim3 grid_ss(ceil((precision) dimsA.y / threads_ss.x), ceil((precision) dimsA.y / threads_ss.y));
    Softmax <<< grid_ss, threads_ss >>> (d_S, d_SS, dimsA.y, dimsA.y);

    // ******* Softmax * V ********
    float *h_H = reinterpret_cast<float *>(malloc(mem_size_Q));

    if (h_H == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }
    float *d_H;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_H), mem_size_Q));
    dim3 threads_h(block_size, block_size);
    dim3 grid_h(dimsC.x / threads.x, dimsA.y / threads.y);
    if (block_size == 16) {
        MatrixMulCUDA<16> <<< gS, tS >>>(d_H, d_SS, d_V,
                                                dimsA.y, dimsC.x, 1);
    } else {
        MatrixMulCUDA<32> <<< gS, tS >>>(d_H, d_SS, d_V,
                                                dimsA.y, dimsC.x, 1);
    }

    cudaDeviceSynchronize();

    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(h_Q, d_Q, mem_size_Q, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_K, d_K, mem_size_Q, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_V, d_V, mem_size_Q, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_S, d_SS, mem_size_S, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_H, d_H, mem_size_Q, cudaMemcpyDeviceToHost));

    printf("h_Q[0] %f h_K[0] %f h_V[0] %f h_S[0][2] %f h_H[0][0] %f\n", h_Q[0], h_K[0], h_V[0], h_S[2], h_H[0]);

    // Clean up memory
    free(h_X);
    free(h_Wq);
    free(h_Q);
    free(h_Wk);
    free(h_K);
    free(h_Wv);
    free(h_V);
    free(h_H);
    free(h_S);
    checkCudaErrors(cudaFree(d_X));
    checkCudaErrors(cudaFree(d_Wq));
    checkCudaErrors(cudaFree(d_Q));
    checkCudaErrors(cudaFree(d_Wk));
    checkCudaErrors(cudaFree(d_K));
    checkCudaErrors(cudaFree(d_Wv));
    checkCudaErrors(cudaFree(d_V));
    checkCudaErrors(cudaFree(d_S));
    checkCudaErrors(cudaFree(d_SS));
    checkCudaErrors(cudaFree(d_H));

    printf("\nNOTE: The CUDA Samples are not meant for performance"\
           "measurements. Results may vary when GPU Boost is enabled.\n");
}


/**
 * Program main
 */
int main(int argc, char **argv) {
    printf("[Transformer Using CUDA] - Starting...\n");

    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    int dev = findCudaDevice(argc, (const char **)argv);

    int block_size = 32;

    dim3 dimsA(768, 32, 1);
    dim3 dimsB(64, 768, 1);

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
                                               dimsB.x, dimsB.y);

    unsigned int size_X = dimsA.x * dimsA.y;
    unsigned int mem_size_X = sizeof(float) * size_X;
    float *h_X = reinterpret_cast<float *>(malloc(mem_size_X));
    matrixInit(h_X, size_X);

    SingleHeadAttention(block_size, dimsA, dimsB, h_X);

    int n = 32;
    int D = 768;
    int dk = 64;

    printf("n %d D %d dk %d\n", n, D, dk);

    // MultiHeadAttention(n, D, dk, block_size);
}