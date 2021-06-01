#include "transformer.hpp"

void matrixInit(precision* mat, int size) {
    for (int i = 0; i < size; i++)
    {
        mat[i] = ((precision) rand() / (RAND_MAX)) - 0.5f;
    }
}

void verifyResult(precision *matrixGPU, precision *matrixCPU, int rows, int cols) {
    int valid=1;
    printf("Verifying results...\n");

    for (int i = 0; i < rows; i++)
    {
        precision row_sum = 0.0;
        for (int j = 0; j < cols; j++)
        {
            row_sum += exp(matrixCPU[i*cols + j]);
        }

        for (int j = 0; j < cols; j++)
        {
            precision cpu_val = exp(matrixCPU[i*cols + j])/row_sum;
            if (fabs(cpu_val - matrixGPU[i * cols + j]) > 1e-2) {
                printf("out[%d][%d] %f cpu: %f gpu: %f exp_sum %f\n", i, j, matrixCPU[i*cols+j], cpu_val, matrixGPU[i * cols + j], row_sum);
                valid = 0;
            }
        }

    }


    if (valid == 1) {
        printf("No errors...\nDone...\n");
    } else {
        printf("GPU function contained errors...\nDone...\n");
    }
}

void RunSoftmax(int rows, int cols) {
    unsigned int size = rows * cols;
    unsigned int mem_size = size * sizeof(precision);
    precision *matrix = reinterpret_cast<precision *>(malloc(mem_size));
    precision *out = reinterpret_cast<precision *>(malloc(mem_size));

    matrixInit(matrix, size);

    precision *d_matrix, *d_out;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_matrix), mem_size));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_out), mem_size));
    checkCudaErrors(cudaMemcpy(d_matrix, matrix, mem_size, cudaMemcpyHostToDevice));

    dim3 threads(1, BLOCK_SIZE);
    dim3 grid(ceil((precision) rows / threads.x), ceil((precision) cols / threads.y));

    printf("Computing result using Softmax on GPU...\n");
    Softmax<<< grid , threads >>> (d_matrix, d_out, rows, cols);

    checkCudaErrors(cudaMemcpy(out, d_out, mem_size, cudaMemcpyDeviceToHost));
    verifyResult(out, matrix, rows, cols);
}

/**
 * Program main
 */
int main(int argc, char **argv) {
    printf("[Softmax Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
            checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -rows=rows\n");
        printf("      -cols=cols\n");

        exit(EXIT_SUCCESS);
    }

    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    int dev = findCudaDevice(argc, (const char **)argv);

    int rows, cols;

    if (checkCmdLineFlag(argc, (const char **)argv, "row")) {
        rows = getCmdLineArgumentInt(argc, (const char **)argv, "row");
    } else {
        rows = 32;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "cols")) {
        cols = getCmdLineArgumentInt(argc, (const char **)argv, "cols");
    } else {
        cols = 32;
    }

    printf("rows: %d, cols: %d\n", rows, cols);

    RunSoftmax(rows, cols);
}
