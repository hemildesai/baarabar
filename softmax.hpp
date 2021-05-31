#ifndef SOFTMAX_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define SOFTMAX_H

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

#define precision float

#ifndef BLOCK_SIZE
     #define BLOCK_SIZE 64
#endif

__global__ void Softmax(precision *matrix, precision *out, int rows, int cols) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int rOut = 1 * bx + tx;
    int cOut = BLOCK_SIZE * by + ty;

    int outIdx = rOut * cols + cOut;

    int xStart = rOut * cols;

    precision exp_sum = 0.0;
    precision thread = 0.0;
    __shared__ precision thread_sum[BLOCK_SIZE];

    for (int i = 0; i < cols; i += BLOCK_SIZE)
    {
        if ((xStart + i + ty) < xStart + cols) {
            thread += expf(matrix[xStart + i + ty]);
        }
    }
    thread_sum[ty] = thread;
    __syncthreads();

    for (int j = 0; j < min(BLOCK_SIZE, cols); j++)
    {
        exp_sum += thread_sum[j];
    }

    if (rOut < rows && cOut < cols) {
        out[outIdx] = expf(matrix[outIdx])/exp_sum;
    }
}

#endif