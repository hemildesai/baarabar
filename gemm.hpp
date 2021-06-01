#ifndef GEMM_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define GEMM_H

// #ifndef BLOCK_SIZE
//      #define BLOCK_SIZE 16
// #endif

// https://github.com/JonathanWatkins/CUDA/blob/master/NvidiaCourse/Exercises/transpose/transpose.cu
template <int BLOCK_DIM> __global__ void transpose(float *odata, float *idata, int width, int height)
{
	__shared__ float block[BLOCK_DIM][BLOCK_DIM+1];

	// read the matrix tile into shared memory
        // load one element per thread from device memory (idata) and store it
        // in transposed order in block[][]
	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

        // synchronise to ensure all writes to block[][] have completed
	__syncthreads();

	// write the transposed matrix tile to global memory (odata) in linear order
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}



/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A,
                                                        float *B, int wA,
                                                        int wB, int divisor) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
            a <= aEnd;
            a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLK_SIZE][BLK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLK_SIZE][BLK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLK_SIZE * by + BLK_SIZE * bx;
    C[c + wB * ty + tx] = Csub/divisor;
}

__global__ void LinearTiled(
    float* out,
    float* x,
    float* w,
    int in_f,
    int out_f,
    int batch_size
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int rOut = BLOCK_SIZE * bx + tx;
    int cOut = BLOCK_SIZE * by + ty;

    int outIdx = rOut * out_f + cOut;

    int xStart = rOut * in_f;
    int wStart = cOut * in_f;

    float temp = 0;
    for (int i = 0; i < in_f; i+= BLOCK_SIZE)
    {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        if (xStart + i + ty < (xStart + in_f) && tx < batch_size && ty < in_f)
        {
            As[tx][ty] = x[xStart + i + ty];
        } else {
            As[tx][ty] = 0.0;
        }

        Bs[tx][ty] = w[wStart + i + tx];

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            temp += As[tx][k] * Bs[k][ty];
        }

        __syncthreads();
    }

    if (rOut < batch_size && cOut < out_f) {
        out[outIdx] = temp;
    }
}

#endif