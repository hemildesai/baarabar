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

    // printf("Computing Query using CUDA Kernel...\n");
    // begin_roi();
    if (block_size == 16) {
        MatrixMulCUDA<16> <<< grid, threads >>>(d_Q, d_X, d_Wq,
                                                dimsA.x, dimsB.x, 1);
    } else {
        MatrixMulCUDA<32> <<< grid, threads >>>(d_Q, d_X, d_Wq,
                                                dimsA.x, dimsB.x, 1);
    }
    // end_roi("Q time ");

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

    // printf("Computing Key using CUDA Kernel...\n");
    if (block_size == 16) {
        MatrixMulCUDA<16> <<< grid, threads >>>(d_K, d_X, d_Wk,
                                                dimsA.x, dimsB.x, 1);
    } else {
        MatrixMulCUDA<32> <<< grid, threads >>>(d_K, d_X, d_Wk,
                                                dimsA.x, dimsB.x, 1);
    }

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

    // printf("Computing Value using CUDA Kernel...\n");
    if (block_size == 16) {
        MatrixMulCUDA<16> <<< grid, threads >>>(d_V, d_X, d_Wv,
                                                dimsA.x, dimsB.x, 1);
    } else {
        MatrixMulCUDA<32> <<< grid, threads >>>(d_V, d_X, d_Wv,
                                                dimsA.x, dimsB.x, 1);
    }

    // ******* K Transpose ********
    float *d_KT;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_KT), mem_size_Q));

    // printf("Computing K Transpose using CUDA Kernel...\n");
    dim3 gT(dimsC.x / block_size, dimsC.y / block_size, 1);
    dim3 tT(block_size, block_size, 1);
    if (block_size == 16) {
        transpose<16> <<< gT, tT >>> (d_KT, d_K, dimsC.x, dimsC.y);
    } else {
        transpose<32> <<< gT, tT >>> (d_KT, d_K, dimsC.x, dimsC.y);
    }


    // ******* Softmax ********
    float *h_S = reinterpret_cast<float *>(malloc(mem_size_S));

    if (h_S == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }
    float *d_S, *d_SS;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_S), mem_size_S));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_SS), mem_size_S));

    // printf("Computing QK using CUDA Kernel...\n");

    dim3 tS(block_size, block_size);
    dim3 gS(dimsCT.x / threads.x, dimsC.y / threads.y);
    if (block_size == 16) {
        MatrixMulCUDA<16> <<< gS, tS >>>(d_S, d_Q, d_KT,
                                                dimsC.x, dimsCT.x, 8);
    } else {
        MatrixMulCUDA<32> <<< gS, tS >>>(d_S, d_Q, d_KT,
                                                dimsC.x, dimsCT.x, 8);
    }

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

    // printf("h_Q[0] %f h_K[0] %f h_V[0] %f h_S[0][2] %f h_H[0][0] %f\n\n\n", h_Q[0], h_K[0], h_V[0], h_S[2], h_H[0]);

    // Clean up memory
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
}

/**
 * Run Single Head Attention using CUDA
 */
void SingleHeadAttentionBreakdown(int block_size, const dim3 &dimsA,
                   const dim3 &dimsB, float *h_X) {
    begin_roi();
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
    matrixInit(h_Wq, size_Wq);

    // Allocate device memory
    float *d_X, *d_Wq, *d_Q;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_X), mem_size_X));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_Wq), mem_size_Wq));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_Q), mem_size_Q));

    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_X, h_X, mem_size_X, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Wq, h_Wq, mem_size_Wq, cudaMemcpyHostToDevice));
    end_roi("CUDA Malloc and Copy");
    begin_roi();

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // ******* Q ********

    // printf("Computing Query using CUDA Kernel...\n");
    // begin_roi();
    if (block_size == 16) {
        MatrixMulCUDA<16> <<< grid, threads >>>(d_Q, d_X, d_Wq,
                                                dimsA.x, dimsB.x, 1);
    } else {
        MatrixMulCUDA<32> <<< grid, threads >>>(d_Q, d_X, d_Wq,
                                                dimsA.x, dimsB.x, 1);
    }
    end_roi("Q time ");
    begin_roi();

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

    // printf("Computing Key using CUDA Kernel...\n");
    if (block_size == 16) {
        MatrixMulCUDA<16> <<< grid, threads >>>(d_K, d_X, d_Wk,
                                                dimsA.x, dimsB.x, 1);
    } else {
        MatrixMulCUDA<32> <<< grid, threads >>>(d_K, d_X, d_Wk,
                                                dimsA.x, dimsB.x, 1);
    }
    end_roi("K time ");
    begin_roi();

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

    // printf("Computing Value using CUDA Kernel...\n");
    if (block_size == 16) {
        MatrixMulCUDA<16> <<< grid, threads >>>(d_V, d_X, d_Wv,
                                                dimsA.x, dimsB.x, 1);
    } else {
        MatrixMulCUDA<32> <<< grid, threads >>>(d_V, d_X, d_Wv,
                                                dimsA.x, dimsB.x, 1);
    }
    end_roi("V time ");
    begin_roi();

    // ******* K Transpose ********
    float *d_KT;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_KT), mem_size_Q));

    // printf("Computing K Transpose using CUDA Kernel...\n");
    dim3 gT(dimsC.x / block_size, dimsC.y / block_size, 1);
    dim3 tT(block_size, block_size, 1);
    if (block_size == 16) {
        transpose<16> <<< gT, tT >>> (d_KT, d_K, dimsC.x, dimsC.y);
    } else {
        transpose<32> <<< gT, tT >>> (d_KT, d_K, dimsC.x, dimsC.y);
    }
    end_roi("K Transpose time ");
    begin_roi();


    // ******* Softmax ********
    float *h_S = reinterpret_cast<float *>(malloc(mem_size_S));

    if (h_S == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }
    float *d_S, *d_SS;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_S), mem_size_S));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_SS), mem_size_S));

    // printf("Computing QK using CUDA Kernel...\n");

    dim3 tS(block_size, block_size);
    dim3 gS(dimsCT.x / threads.x, dimsC.y / threads.y);
    if (block_size == 16) {
        MatrixMulCUDA<16> <<< gS, tS >>>(d_S, d_Q, d_KT,
                                                dimsC.x, dimsCT.x, 8);
    } else {
        MatrixMulCUDA<32> <<< gS, tS >>>(d_S, d_Q, d_KT,
                                                dimsC.x, dimsCT.x, 8);
    }

    dim3 threads_ss(1, BLOCK_SIZE);
    dim3 grid_ss(ceil((precision) dimsA.y / threads_ss.x), ceil((precision) dimsA.y / threads_ss.y));
    Softmax <<< grid_ss, threads_ss >>> (d_S, d_SS, dimsA.y, dimsA.y);
    end_roi("Softmax time ");
    begin_roi();

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
    end_roi("Softmax * V time ");
    begin_roi();

    // printf("h_Q[0] %f h_K[0] %f h_V[0] %f h_S[0][2] %f h_H[0][0] %f\n\n\n", h_Q[0], h_K[0], h_V[0], h_S[2], h_H[0]);

    // Clean up memory
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
    end_roi("Free up time ");
}


void MultiHeadAttentionSync(int block_size, const dim3 &dimsA, const dim3 &dimsB) {
    unsigned int size_X = dimsA.x * dimsA.y;
    unsigned int mem_size_X = sizeof(float) * size_X;
    float *h_X = reinterpret_cast<float *>(malloc(mem_size_X));
    matrixInit(h_X, size_X);
    SingleHeadAttention(block_size, dimsA, dimsB, h_X);

    begin_roi();

    for (int i = 0; i < 12; i++)
    {
        SingleHeadAttention(block_size, dimsA, dimsB, h_X);
    }
    end_roi("Multi Head Attention Synchronous Time ");
}

void MultiHeadAttentionAsync(int block_size, const dim3 &dimsA, const dim3 &dimsB) {
    unsigned int size_X = dimsA.x * dimsA.y;
    unsigned int mem_size_X = sizeof(float) * size_X;
    float *h_X = reinterpret_cast<float *>(malloc(mem_size_X));
    matrixInit(h_X, size_X);
    SingleHeadAttention(block_size, dimsA, dimsB, h_X);

    std::vector < std::thread > threads;

    for (int i = 0; i < 12; i++)
    {
      threads.push_back(std::thread (SingleHeadAttention, block_size, dimsA, dimsB, h_X));
    }

    begin_roi();
    for (auto & t:threads)
    {
        t.join ();
    }
    end_roi("Multi Head Attention Asynchronous Time ");

}

void ConcurrentQ(int block_size, const dim3 &dimsA,
                   const dim3 &dimsB, float *h_X) {
    // Allocate host memory for matrices A and B
    unsigned int size_X = dimsA.x * dimsA.y;
    unsigned int mem_size_X = sizeof(float) * size_X;

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
    matrixInit(h_Wq, size_Wq);

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // Allocate device memory
    float *d_X, *d_Wq, *d_Q;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_X), mem_size_X));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_Wq), mem_size_Wq));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_Q), mem_size_Q));

    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_X, h_X, mem_size_X, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Wq, h_Wq, mem_size_Wq, cudaMemcpyHostToDevice));

    MatrixMulCUDA<32> <<< grid, threads >>>(d_Q, d_X, d_Wq,
                                                    dimsA.x, dimsB.x, 1);

    int nkernels = 36;
    int nstreams = nkernels + 1;
    float elapsed_time;
    float kernel_time = 125;

    // allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));

    for (int i = 0; i < nstreams; i++)
    {
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }

    // create CUDA event handles
    cudaEvent_t start_event, stop_event;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));


    // the events are used for synchronization only and hence do not need to record timings
    // this also makes events not introduce global sync points when recorded which is critical to get overlap
    cudaEvent_t *kernelEvent;
    kernelEvent = (cudaEvent_t *) malloc(nkernels * sizeof(cudaEvent_t));

    for (int i = 0; i < nkernels; i++)
    {
        checkCudaErrors(cudaEventCreateWithFlags(&(kernelEvent[i]), cudaEventDisableTiming));
    }

    cudaEventRecord(start_event, 0);

    // queue nkernels in separate streams and record when they are done
    for (int i=0; i<nkernels; ++i)
    {
        MatrixMulCUDA<32> <<< grid, threads, 0, streams[i] >>>(d_Q, d_X, d_Wq,
                                                    dimsA.x, dimsB.x, 1);
        checkCudaErrors(cudaEventRecord(kernelEvent[i], streams[i]));

        // make the last stream wait for the kernel event to be recorded
        // checkCudaErrors(cudaStreamWaitEvent(streams[nstreams-1], kernelEvent[i],0));
    }

    // at this point the CPU has dispatched all work for the GPU and can continue processing other tasks in parallel

    // in this sample we just wait until the GPU is done
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));

    printf("Expected time for serial execution of %d kernels = %.5fus\n", nkernels, nkernels * kernel_time);
    printf("Expected time for concurrent execution of %d kernels = %.5fus\n", nkernels, kernel_time);
    printf("Measured time for sample = %fs\n", elapsed_time/1000.0f);
}
/**
 * Program main
 */
int main(int argc, char **argv) {
    printf("[Transformer Using CUDA] - Starting...\n");

    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    int dev = findCudaDevice(argc, (const char **)argv);

    int block_size = 16;

    dim3 dimsA(768, 128 * 16, 1);
    dim3 dimsB(64, 768, 1);

    printf("Embedding Dim %d, Single Head Dim %d Sentence Length %d\n", dimsA.x, dimsB.x,
                                               dimsA.y);

    printf("Hardware Concurrency %d threads\n", std::thread::hardware_concurrency());

    dimsA.y = 16 * 128;
    unsigned int size_X = dimsA.x * dimsA.y;
    unsigned int mem_size_X = sizeof(float) * size_X;
    float *h_X = reinterpret_cast<float *>(malloc(mem_size_X));
    matrixInit(h_X, size_X);
    // begin_roi();
    // SingleHeadAttention(block_size, dimsA, dimsB, h_X);
    // end_roi("Single head attention ");

    SingleHeadAttentionBreakdown(block_size, dimsA, dimsB, h_X);
    printf("\n\n");
    SingleHeadAttentionBreakdown(block_size, dimsA, dimsB, h_X);
    printf("\n\n");

    MultiHeadAttentionSync(block_size, dimsA, dimsB);
    MultiHeadAttentionAsync(block_size, dimsA, dimsB);

    ConcurrentQ(block_size, dimsA, dimsB, h_X);

    // MultiHeadAttention(n, D, dk, block_size);
}