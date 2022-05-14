#include "cuda_runtime.h"
#include "math_functions.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <thrust/extrema.h>
#include <stdio.h>
#include <crt/device_functions.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/timeb.h>
static const int blockSize = 1;
static const int totalBlock = 64;
static const int BLOCK_SIZE = 16;//the block size for matrix tilling
static const int BLOCK_SIZE2 = 32;//the block size for matrix tilling
//m*k is the layer size * channel size, k*n is the weight matrix
__host__ void sysUsecTime(void)
{
    struct timeb tv;
    struct tm* t;

    ftime(&tv);

    t = localtime(&tv.time);
    printf(", start at:%d-%d-%d %d:%d:%d.%ld\n", 1900 + t->tm_year, 1 + t->tm_mon, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec, tv.millitm);
}
__device__ float max(float a, float b, float c, float d) {
    return max(max(max(a, b), c), d);
}

__global__ void Relu(float* c, unsigned int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < n) {
        if (c[tid] < 0) {
            c[tid] = 0;
        }
        tid += blockDim.x * gridDim.x;
    }
}
__global__ void dense(float* fpMatrixA, float* fpMatrixB,
    float* fpMatrixC, float* bias, int m, int n, int k)
{
    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("the result is : %d  %d\n", nRow,nCol);
    float fCVal = 0.0f;
    for (int i = 0; i < k; i++)
    {
        //printf("the result is : %f\n", fpMatrixB[i]);
        fCVal += fpMatrixA[nRow * k + i] * fpMatrixB[i * n + nCol];
    }
    //printf("the result is : %f\n", fCVal);
    fpMatrixC[nRow * n + nCol] = fCVal + bias[nRow];
}
__global__ void dense2(float* fpMatrixA, float* fpMatrixB,
    float* fpMatrixC, float* bias, int m, int n, int k)
{
    float sum = 0.0f;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int l = 0; l < k; l++)
            {   
                sum += fpMatrixA[i * k + l] * fpMatrixB[l * n + j];
            }
            fpMatrixC[i * n + j] = sum + bias[i];
            sum = 0.0f;
        }
    }
}
void matrixMulCpu(float* fpMatrixA, float* fpMatrixB, float* fpMatrixC, float* bias,
    int m, int n, int k)
{
    float sum = 0.0f;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int l = 0; l < k; l++)
            {
                sum += fpMatrixA[i * k + l] * fpMatrixB[l * n + j];
            }
            fpMatrixC[i * n + j] = sum + bias[j];
            sum = 0.0f;
        }
    }
}
//m*k input , k*n weight matrix , m*n output matrix , m is the batch size
void gemm(float* fpMatrixA, float* fpMatrixB,
    float* fpMatrixC, float* bias,int m,int n, int k) {
    int dimx;
    int dimy;
    if (n >= 1024) {
        dimx = n / 64;
    }
    else {
        dimx = n / 4;
        if (dimx == 0) {
            dimx++;
        }
    }
    if (m >= 1024) {
        dimy = m / 32;
    }
    else {
        dimy = m / 4;
        if (dimy == 0) {
            dimy++;
        }
    }
    //m = 1 n = 4096 k = 25088
    //printf("the sum is :%d, %d,%d\n", m,dimx,dimy );
    dim3 dimBlock(dimx,  dimy);
    dim3 dimGrid(n/dimx, m/ dimy);
    //dim3 dimBlock(1, 64);
    //dim3 dimGrid(1, 64);
    if (m != 1000) {
        printf("I'm here\n");
        dense << <dimBlock, dimGrid >> > (fpMatrixA, fpMatrixB, fpMatrixC, bias, m, n, k);
        //dense2 << <1, 1 >> > (fpMatrixA, fpMatrixB, fpMatrixC, bias, m, n, k);
    }
    else {
        dim3 dimBlock(1, 100);
        dim3 dimGrid(1, 10);
        dense << <dimGrid,dimBlock >> > (fpMatrixA, fpMatrixB, fpMatrixC, bias, m, n, k);
        //dense2 << <1, 1 >> > (fpMatrixA, fpMatrixB, fpMatrixC, bias, m, n, k);

    }
    //dense2 << <1, 1 >> > (fpMatrixA, fpMatrixB, fpMatrixC, bias, m, n, k);

}
__global__ void batch_normalization(float* a, int rowsize, float* mean, float* variance, float* gamma, float* beta, float epsilon = 1e-5) {
    int idx = threadIdx.x;
    int bid = blockIdx.x;
    float sum = 0;
    //calculate mean

    for (int i = idx; i < rowsize; i += blockSize) {
        float a_hat = (a[rowsize * bid + i] - mean[bid]) / sqrt(variance[bid] + epsilon);
        //printf(" the number is %f %f\n", *(a + rowsize * bid + i), beta[bid]);
        float result = gamma[bid] * a_hat + beta[bid];
        *(a + rowsize * bid + i) = result;
    }
}
__global__ void Conv3(float* fpMatrixC, float* fpMatrixA, float* fpMatrixB, float* bias, int m, int k, int n) {
    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    float fCVal = 0.0f;

    __shared__ float shTileA[BLOCK_SIZE2][BLOCK_SIZE2];
    __shared__ float shTileB[BLOCK_SIZE2][BLOCK_SIZE2];

    int nIter = (k + BLOCK_SIZE2 - 1) / BLOCK_SIZE2;
    for (int i = 0; i < nIter; i++)
    {
        // load data from global memory to shared memory
        shTileA[threadIdx.y][threadIdx.x] = fpMatrixA[nRow * k + i * BLOCK_SIZE2 + threadIdx.x];
        shTileB[threadIdx.y][threadIdx.x] = fpMatrixB[(i * BLOCK_SIZE2 + threadIdx.y) * n + nCol];

        // sync to wait for all threads in one block to finish loading datas
        __syncthreads();

        // sub-matrix multiply
        for (int l = 0; l < BLOCK_SIZE2; l++)
        {
            fCVal += shTileA[threadIdx.y][l] * shTileB[l][threadIdx.x];
        }

        // sync to wait for all threads in one block to finish compute
        __syncthreads();
    }

    // store results into global memory
    fpMatrixC[nRow * n + nCol] = fCVal + bias[nRow];
}
__global__ void Conv(float* fpMatrixC, float* fpMatrixA, float* fpMatrixB, float* bias, int m, int k, int n) {
    int nRow = threadIdx.x;
    int nCol = blockIdx.x;
    //int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    //int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    float fCVal = 0.0f;
    for (int i = 0; i < k; i++)
    {
        fCVal += fpMatrixA[nRow * k + i] * fpMatrixB[i * n + nCol];
    }

    fpMatrixC[nRow * n + nCol] = fCVal + bias[nRow];
}
__global__ void Conv4(float* fpMatrixC, float* fpMatrixA, float* fpMatrixB, float* bias, int m, int k, int n)
{
    int row = blockIdx.y * blockDim.y * 2 + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float val[2] = { 0.0f };

    __shared__ float shTileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shTileB[BLOCK_SIZE][BLOCK_SIZE];

    int iter = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int i = 0; i < iter; i++)
    {
        // read data from global memory to shared memory
        shTileA[threadIdx.y][threadIdx.x] = fpMatrixA[row * k + i * BLOCK_SIZE + threadIdx.x];
        shTileA[threadIdx.y + 16][threadIdx.x] = fpMatrixA[(row + 16) * k + i * BLOCK_SIZE + threadIdx.x];

        shTileB[threadIdx.y][threadIdx.x] = fpMatrixB[(i * BLOCK_SIZE + threadIdx.y) * n + col];
        shTileB[threadIdx.y + 16][threadIdx.x] = fpMatrixB[(i * BLOCK_SIZE + threadIdx.y + 16) * n + col];

        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; j++)
        {
            val[0] += shTileA[threadIdx.y][j] * shTileB[j][threadIdx.x];
            val[1] += shTileA[threadIdx.y + 16][j] * shTileB[j][threadIdx.x];
        }

        __syncthreads();
    }

    fpMatrixC[row * n + col] = val[0]+bias[row];
    fpMatrixC[(row + 16) * n + col] = val[1] + bias[row+16];
}
__global__ void Conv2(float* fpMatrixC, float* fpMatrixA, float* fpMatrixB, float* bias, int m, int k, int n)
{
    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    float fCVal = 0.0f;

    __shared__ float shTileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shTileB[BLOCK_SIZE][BLOCK_SIZE];

    int nIter = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int i = 0; i < nIter; i++)
    {
        // load data from global memory to shared memory
        shTileA[threadIdx.y][threadIdx.x] = fpMatrixA[nRow * k + i * BLOCK_SIZE + threadIdx.x];
        shTileB[threadIdx.y][threadIdx.x] = fpMatrixB[(i * BLOCK_SIZE + threadIdx.y) * n + nCol];

        // sync to wait for all threads in one block to finish loading datas
        __syncthreads();

        // sub-matrix multiply
        for (int l = 0; l < BLOCK_SIZE; l++)
        {
            fCVal += shTileA[threadIdx.y][l] * shTileB[l][threadIdx.x];
        }

        // sync to wait for all threads in one block to finish compute
        __syncthreads();
    }

    // store results into global memory
    fpMatrixC[nRow * n + nCol] = fCVal + bias[nRow];
}

__global__ void softmax(float* out, float* in, int n)
{
    float sum = 0.0;
    for (int i = 0; i < n; ++i)
    {
        sum += exp(in[i]);
    }
    for (int i = 0; i < n; ++i) {
        out[i] = exp(in[i]) / sum;
    }
}

__global__ void maxpooling2d(float* out, float* in, int n, int channel)
{
    // n is the height/width of the feature map.
    for (int c = 0; c < channel; ++c)// for each channel
    {
        int newn = n / 2;
        for (int i = 0; i < newn;++i) {
            for (int j = 0;j < newn;++j) {
                out[newn * newn * c + i * newn + j] = max(in[n * n * c + n * i * 2 + 2 * j], in[n * n * c + n * i * 2 + 2 * j + 1], in[n * n * c + n * (i * 2 + 1) + 2 * j], in[n * n * c + n * (i * 2 + 1) + 2 * j + 1]);
            }
        }
    }
}

__global__ void changeform(float* in_map, int n, int in_channel, float* tmp_out)
{
    /*
    in_map : input feature map
    n: height/width of the input feature map
    in_channel: input feature map channel
    tmp_out: the resized feature map, which is used in the convolution operation
    */
    int step_table[9][2] = { {-1,-1},{-1,0},{-1,1},{0,-1},{0,0},{0,1},{1,-1},{1,0},{1,1} };
    //use the input feature map to get the matrix form
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < in_channel; ++k) {
                for (int r = 0;r < 9; ++r) {
                    int newi = i + step_table[r][0];
                    int newj = j + step_table[r][1];
                    tmp_out[(n * i + j) + (9 * k + r) * n * n] = (newi >= 0 && newj >= 0 && newi < n&& newj < n) ? in_map[n * n * k + newi * n + newj] : 0.0;
                }
            }
        }
    }
}

__global__ void Conv2d(float* out_map, float* kernels, int n, int in_channel, int out_channel, float* bias, float* tmp_out)
{
    /*
    kernels: weight of the conv kernel
    n: height/width of the input feature map
    in_channel: input feature map channel
    out_channel: output feature map channel
    bias: bias of the conv kernel
    tmp_out: the resized feature map, which is used in the convolution operation
    */
    for (int i = 0;i < out_channel;++i) {
        for (int j = 0;j < n * n;++j) {
            float sum = 0.0;
            for (int k = 0;k < 9 * in_channel;++k) {
                sum += kernels[i * 9 * in_channel + k] * tmp_out[j + k * n * n];
            }
            out_map[i * n * n + j] = sum + bias[i];
        }
    }
}

__host__ void loadweights(float* weight, char* weightpath, const char* path) {
    // load weights from given file
    char newpath[100] = { '\0' };
    strcat(newpath, weightpath);
    strcat(newpath, path);
    printf("%s\n", newpath);
    FILE* fp = fopen(newpath, "rb");
    fseek(fp, 0, SEEK_END);
    int fileSize = ftell(fp);
    int arraySize = fileSize / sizeof(float);
    fseek(fp, 0, SEEK_SET);
    fread(weight, sizeof(float), arraySize, fp);
    fclose(fp);
}

__host__ void loadimage(float* weight, const char* path) {
    // load weights from given file
    FILE* fp = fopen(path, "rb");
    fseek(fp, 0, SEEK_END);
    int fileSize = ftell(fp);
    int arraySize = fileSize / sizeof(float);
    fseek(fp, 0, SEEK_SET);
    fread(weight, sizeof(float), arraySize, fp);
    fclose(fp);
}


int main(int argc, char* argv[])
{
    float* feature;
    float* d_feature;
    printf("---------------------------------------------Start-----------------------------------------\n");
    char weightpath[100] = { '\0' };
    strcat(weightpath, argv[1]);
    char* imagefile = argv[2];
    const char* outputfile = argv[3];
    feature = (float*)malloc(sizeof(float) * (150528));
    loadimage(feature, imagefile);

    cudaMalloc((void**)&d_feature, sizeof(float) * (150528));
    cudaMemcpy(d_feature, feature, sizeof(float) * 150528, cudaMemcpyHostToDevice);
    //----------------------------------------Block 1-----------------------------------------------
    //----------------------------------------------------------------------------------------------
    //conv1-------------------------------------------------------------------------------------
    printf("Block1, Conv1");
    sysUsecTime();
    float* conv1_out, * conv1_w, * conv1_b;
    float* d_conv1_out, * d_conv1_w, * d_conv1_b, * d_tmp1;

    conv1_w = (float*)malloc(sizeof(float) * (1728));
    conv1_b = (float*)malloc(sizeof(float) * (64));
    conv1_out = (float*)malloc(sizeof(float) * (64 * 224 * 224));

    loadweights(conv1_w, weightpath, "features.0.weight.txt");
    loadweights(conv1_b, weightpath, "features.0.bias.txt");

    cudaMalloc((void**)&d_conv1_w, sizeof(float) * (1728));
    cudaMalloc((void**)&d_conv1_b, sizeof(float) * (64));
    cudaMalloc((void**)&d_conv1_out, sizeof(float) * (64 * 224 * 224));
    cudaMalloc((void**)&d_tmp1, sizeof(float) * (3 * 9 * 224 * 224));

    cudaMemcpy(d_conv1_w, conv1_w, sizeof(float) * 1728, cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1_b, conv1_b, sizeof(float) * 64, cudaMemcpyHostToDevice);
    changeform << <1, 1 >> > (d_feature, 224, 3, d_tmp1);
    printf("gemm");
    sysUsecTime();
    dim3 dimGrid(224, 8);
    dim3 dimBlock(224, 8);
    Conv << <224 * 224, 64 >> > (d_conv1_out, d_conv1_w, d_tmp1, d_conv1_b, 64, 9 * 3, 224 * 224);
    //Conv << <dimGrid,dimBlock>> > (d_conv1_out, d_conv1_w, d_tmp1, d_conv1_b, 64, 9 * 3, 224 * 224);

    cudaFree(d_conv1_w);
    cudaFree(d_conv1_b);
    cudaFree(d_feature);
    cudaFree(d_tmp1);
    free(conv1_w);
    free(conv1_b);
    free(feature);
    // bn1------------------------------------------------------------------------------------------
    printf("Block1, bn1");
    sysUsecTime();
    float* bn1_gamma, * bn1_beta, * bn1_mean, * bn1_var;
    float* d_bn1_gamma, * d_bn1_beta, * d_bn1_mean, * d_bn1_var;

    bn1_gamma = (float*)malloc(sizeof(float) * (64));
    bn1_beta = (float*)malloc(sizeof(float) * (64));
    bn1_mean = (float*)malloc(sizeof(float) * (64));
    bn1_var = (float*)malloc(sizeof(float) * (64));

    loadweights(bn1_gamma, weightpath, "features.1.weight.txt");
    loadweights(bn1_beta, weightpath, "features.1.bias.txt");
    loadweights(bn1_mean, weightpath, "features.1.running_mean.txt");
    loadweights(bn1_var, weightpath, "features.1.running_var.txt");

    cudaMalloc((void**)&d_bn1_gamma, sizeof(float) * (64));
    cudaMalloc((void**)&d_bn1_beta, sizeof(float) * (64));
    cudaMalloc((void**)&d_bn1_mean, sizeof(float) * (64));
    cudaMalloc((void**)&d_bn1_var, sizeof(float) * (64));

    cudaMemcpy(d_bn1_gamma, bn1_gamma, sizeof(float) * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn1_beta, bn1_beta, sizeof(float) * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn1_mean, bn1_mean, sizeof(float) * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn1_var, bn1_var, sizeof(float) * 64, cudaMemcpyHostToDevice);

    batch_normalization << <64, blockSize >> > (d_conv1_out, 224 * 224, d_bn1_mean, d_bn1_var, d_bn1_gamma, d_bn1_beta);
    cudaFree(d_bn1_beta);
    cudaFree(d_bn1_gamma);
    cudaFree(d_bn1_mean);
    cudaFree(d_bn1_var);
    free(bn1_beta);
    free(bn1_gamma);
    free(bn1_mean);
    free(bn1_var);
    //Relu------------------------------------------------------------------------------------------
    Relu << <100, 100 >> > (d_conv1_out, 224 * 224 * 64);
    //maxpool1--------------------------------------------------------------------------------------
    printf("Block1, maxpool1");
    sysUsecTime();
    float* pool1_out;
    float* d_pool1_out;
    pool1_out = (float*)malloc(sizeof(float) * (64 * 112 * 112));
    cudaMalloc((void**)&d_pool1_out, sizeof(float) * (64 * 112 * 112));
    maxpooling2d << <10, 100 >> > (d_pool1_out, d_conv1_out, 224, 64);
    cudaFree(d_conv1_out);
    free(conv1_out);
    cudaDeviceSynchronize();
    printf("--------------------------------------Block 1 End------------------------------------------\n");
    // ----------------------------------------Block 2-----------------------------------------------
    // ----------------------------------------------------------------------------------------------
    // conv2-----------------------------------------------------------------------------------------
    printf("Block2, Conv2");
    sysUsecTime();
    float* conv2_out, * conv2_w, * conv2_b;
    float* d_conv2_out, * d_conv2_w, * d_conv2_b, * d_tmp2;
    int h_conv = 112;
    int in_conv = 64;
    int kernel_num_conv = 128;

    conv2_w = (float*)malloc(sizeof(float) * (9 * in_conv * kernel_num_conv));
    conv2_b = (float*)malloc(sizeof(float) * (kernel_num_conv));
    conv2_out = (float*)malloc(sizeof(float) * (h_conv * h_conv * kernel_num_conv));

    loadweights(conv2_w, weightpath, "features.4.weight.txt");
    loadweights(conv2_b, weightpath, "features.4.bias.txt");

    cudaMalloc((void**)&d_conv2_w, sizeof(float) * (9 * in_conv * kernel_num_conv));
    cudaMalloc((void**)&d_conv2_b, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_conv2_out, sizeof(float) * (h_conv * h_conv * kernel_num_conv));
    cudaMalloc((void**)&d_tmp2, sizeof(float) * (in_conv * 9 * h_conv * h_conv));

    cudaMemcpy(d_conv2_w, conv2_w, sizeof(float) * (9 * in_conv * kernel_num_conv), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2_b, conv2_b, sizeof(float) * (kernel_num_conv), cudaMemcpyHostToDevice);
    changeform << <1, 1 >> > (d_pool1_out, h_conv, in_conv, d_tmp2);
    printf("gemm");
    sysUsecTime();
    printf("the size is %d,%d,%d\n", kernel_num_conv, 9 * in_conv, h_conv* h_conv);
    //128,576,12544
    dim3 dimGrid2(784, 8);
    dim3 dimBlock2(16, 16);
    //Conv << <h_conv * h_conv, kernel_num_conv >> > (d_conv2_out, d_conv2_w, d_tmp2, d_conv2_b, kernel_num_conv, 9 * in_conv, h_conv * h_conv);
    Conv2 << <dimGrid2, dimBlock2 >> > (d_conv2_out, d_conv2_w, d_tmp2, d_conv2_b, kernel_num_conv, 9 * in_conv, h_conv* h_conv);
    cudaFree(d_conv2_w);
    cudaFree(d_conv2_b);
    cudaFree(d_pool1_out);
    cudaFree(d_tmp2);
    free(conv2_w);
    free(conv2_b);
    free(pool1_out);
    // bn2------------------------------------------------------------------------------------------
    printf("Block2, bn2");
    sysUsecTime();
    float* bn2_gamma, * bn2_beta, * bn2_mean, * bn2_var;
    float* d_bn2_gamma, * d_bn2_beta, * d_bn2_mean, * d_bn2_var;

    bn2_gamma = (float*)malloc(sizeof(float) * (kernel_num_conv));
    bn2_beta = (float*)malloc(sizeof(float) * (kernel_num_conv));
    bn2_mean = (float*)malloc(sizeof(float) * (kernel_num_conv));
    bn2_var = (float*)malloc(sizeof(float) * (kernel_num_conv));

    loadweights(bn2_gamma, weightpath, "features.5.weight.txt");
    loadweights(bn2_beta, weightpath, "features.5.bias.txt");
    loadweights(bn2_mean, weightpath, "features.5.running_mean.txt");
    loadweights(bn2_var, weightpath, "features.5.running_var.txt");

    cudaMalloc((void**)&d_bn2_gamma, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_bn2_beta, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_bn2_mean, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_bn2_var, sizeof(float) * (kernel_num_conv));

    cudaMemcpy(d_bn2_gamma, bn2_gamma, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn2_beta, bn2_beta, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn2_mean, bn2_mean, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn2_var, bn2_var, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);

    batch_normalization << <128, blockSize >> > (d_conv2_out, h_conv * h_conv, d_bn2_mean, d_bn2_var, d_bn2_gamma, d_bn2_beta);
    cudaFree(d_bn2_beta);
    cudaFree(d_bn2_gamma);
    cudaFree(d_bn2_mean);
    cudaFree(d_bn2_var);
    free(bn2_beta);
    free(bn2_gamma);
    free(bn2_mean);
    free(bn2_var);
    //Relu------------------------------------------------------------------------------------------
    Relu << <10, 100 >> > (d_conv2_out, h_conv * h_conv * kernel_num_conv);
    //maxpool2--------------------------------------------------------------------------------------
    printf("Block2, pool2");
    sysUsecTime();
    float* pool2_out;
    float* d_pool2_out;
    pool2_out = (float*)malloc(sizeof(float) * (h_conv * h_conv * kernel_num_conv / 4));
    cudaMalloc((void**)&d_pool2_out, sizeof(float) * (h_conv * h_conv * kernel_num_conv / 4));
    maxpooling2d << <1, 1 >> > (d_pool2_out, d_conv2_out, h_conv, kernel_num_conv);
    cudaFree(d_conv2_out);
    free(conv2_out);
    cudaDeviceSynchronize();
    printf("--------------------------------------Block 2 End------------------------------------------\n");
    // ----------------------------------------Block 3-----------------------------------------------
    // ----------------------------------------------------------------------------------------------
    // conv3-----------------------------------------------------------------------------------------
    printf("Block3, Conv3");
    sysUsecTime();
    float* conv3_out, * conv3_w, * conv3_b;
    float* d_conv3_out, * d_conv3_w, * d_conv3_b, * d_tmp3;
    h_conv = 56;
    in_conv = 128;
    kernel_num_conv = 256;

    conv3_w = (float*)malloc(sizeof(float) * (9 * in_conv * kernel_num_conv));
    conv3_b = (float*)malloc(sizeof(float) * (kernel_num_conv));
    conv3_out = (float*)malloc(sizeof(float) * (h_conv * h_conv * kernel_num_conv));

    loadweights(conv3_w, weightpath, "features.8.weight.txt");
    loadweights(conv3_b, weightpath, "features.8.bias.txt");

    cudaMalloc((void**)&d_conv3_w, sizeof(float) * (9 * in_conv * kernel_num_conv));
    cudaMalloc((void**)&d_conv3_b, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_conv3_out, sizeof(float) * (h_conv * h_conv * kernel_num_conv));
    cudaMalloc((void**)&d_tmp3, sizeof(float) * (in_conv * 9 * h_conv * h_conv));

    cudaMemcpy(d_conv3_w, conv3_w, sizeof(float) * (9 * in_conv * kernel_num_conv), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv3_b, conv3_b, sizeof(float) * (kernel_num_conv), cudaMemcpyHostToDevice);
    changeform << <1, 1 >> > (d_pool2_out, h_conv, in_conv, d_tmp3);
    printf("gemm");
    sysUsecTime();
    printf("the size2 is %d,%d,%d\n", kernel_num_conv, 9 * in_conv, h_conv* h_conv);
    //256 112 3136
    dim3 dimGrid3(196, 16);
    dim3 dimBlock3(16, 16);
    //Conv << <h_conv * h_conv, kernel_num_conv >> > (d_conv3_out, d_conv3_w, d_tmp3, d_conv3_b, kernel_num_conv, 9 * in_conv, h_conv * h_conv);
    Conv2 << < dimGrid2 ,dimBlock2>> > (d_conv3_out, d_conv3_w, d_tmp3, d_conv3_b, kernel_num_conv, 9 * in_conv, h_conv* h_conv);
    cudaFree(d_conv3_w);
    cudaFree(d_conv3_b);
    cudaFree(d_pool2_out);
    cudaFree(d_tmp3);
    free(conv3_w);
    free(conv3_b);
    free(pool2_out);

    // bn3------------------------------------------------------------------------------------------
    printf("Block3, bn3");
    sysUsecTime();
    float* bn3_gamma, * bn3_beta, * bn3_mean, * bn3_var;
    float* d_bn3_gamma, * d_bn3_beta, * d_bn3_mean, * d_bn3_var;

    bn3_gamma = (float*)malloc(sizeof(float) * (kernel_num_conv));
    bn3_beta = (float*)malloc(sizeof(float) * (kernel_num_conv));
    bn3_mean = (float*)malloc(sizeof(float) * (kernel_num_conv));
    bn3_var = (float*)malloc(sizeof(float) * (kernel_num_conv));

    loadweights(bn3_gamma, weightpath, "features.9.weight.txt");
    loadweights(bn3_beta, weightpath, "features.9.bias.txt");
    loadweights(bn3_mean, weightpath, "features.9.running_mean.txt");
    loadweights(bn3_var, weightpath, "features.9.running_var.txt");

    cudaMalloc((void**)&d_bn3_gamma, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_bn3_beta, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_bn3_mean, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_bn3_var, sizeof(float) * (kernel_num_conv));

    cudaMemcpy(d_bn3_gamma, bn3_gamma, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn3_beta, bn3_beta, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn3_mean, bn3_mean, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn3_var, bn3_var, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);

    batch_normalization << <256, blockSize >> > (d_conv3_out, h_conv * h_conv, d_bn3_mean, d_bn3_var, d_bn3_gamma, d_bn3_beta);
    cudaFree(d_bn3_beta);
    cudaFree(d_bn3_gamma);
    cudaFree(d_bn3_mean);
    cudaFree(d_bn3_var);
    free(bn3_beta);
    free(bn3_gamma);
    free(bn3_mean);
    free(bn3_var);
    //Relu------------------------------------------------------------------------------------------
    Relu << <10, 100 >> > (d_conv3_out, h_conv * h_conv * kernel_num_conv);
    cudaDeviceSynchronize();
    printf("\n----------------------------------------------------------\n");
    // conv4-----------------------------------------------------------------------------------------
    printf("Block3, Conv4");
    sysUsecTime();
    h_conv = 56;
    in_conv = 256;
    kernel_num_conv = 256;

    float* conv4_out, * conv4_w, * conv4_b;
    float* d_conv4_out, * d_conv4_w, * d_conv4_b, * d_tmp4;


    conv4_w = (float*)malloc(sizeof(float) * (9 * in_conv * kernel_num_conv));
    conv4_b = (float*)malloc(sizeof(float) * (kernel_num_conv));
    conv4_out = (float*)malloc(sizeof(float) * (h_conv * h_conv * kernel_num_conv));

    loadweights(conv4_w, weightpath, "features.11.weight.txt");
    loadweights(conv4_b, weightpath, "features.11.bias.txt");

    cudaMalloc((void**)&d_conv4_w, sizeof(float) * (9 * in_conv * kernel_num_conv));
    cudaMalloc((void**)&d_conv4_b, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_conv4_out, sizeof(float) * (h_conv * h_conv * kernel_num_conv));
    cudaMalloc((void**)&d_tmp4, sizeof(float) * (in_conv * 9 * h_conv * h_conv));

    cudaMemcpy(d_conv4_w, conv4_w, sizeof(float) * (9 * in_conv * kernel_num_conv), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv4_b, conv4_b, sizeof(float) * (kernel_num_conv), cudaMemcpyHostToDevice);
    changeform << <1, 1 >> > (d_conv3_out, h_conv, in_conv, d_tmp4);
    printf("gemm");
    sysUsecTime();
    printf("the size3 is %d,%d,%d\n", kernel_num_conv, 9 * in_conv, h_conv* h_conv);
    //256 2304 3136
    dim3 dimGrid4(196, 16);
    dim3 dimBlock4(16, 16);
    //Conv << <h_conv * h_conv, kernel_num_conv >> > (d_conv4_out, d_conv4_w, d_tmp4, d_conv4_b, kernel_num_conv, 9 * in_conv, h_conv * h_conv);
    Conv2 << <dimGrid4, dimBlock4 >> > (d_conv4_out, d_conv4_w, d_tmp4, d_conv4_b, kernel_num_conv, 9 * in_conv, h_conv* h_conv);
    cudaFree(d_conv4_w);
    cudaFree(d_conv4_b);
    cudaFree(d_conv3_out);
    cudaFree(d_tmp4);
    free(conv4_w);
    free(conv4_b);
    free(conv3_out);
    // bn4------------------------------------------------------------------------------------------
    printf("Block3, bn4");
    sysUsecTime();
    float* bn4_gamma, * bn4_beta, * bn4_mean, * bn4_var;
    float* d_bn4_gamma, * d_bn4_beta, * d_bn4_mean, * d_bn4_var;

    bn4_gamma = (float*)malloc(sizeof(float) * (kernel_num_conv));
    bn4_beta = (float*)malloc(sizeof(float) * (kernel_num_conv));
    bn4_mean = (float*)malloc(sizeof(float) * (kernel_num_conv));
    bn4_var = (float*)malloc(sizeof(float) * (kernel_num_conv));

    loadweights(bn4_gamma, weightpath, "features.12.weight.txt");
    loadweights(bn4_beta, weightpath, "features.12.bias.txt");
    loadweights(bn4_mean, weightpath, "features.12.running_mean.txt");
    loadweights(bn4_var, weightpath, "features.12.running_var.txt");

    cudaMalloc((void**)&d_bn4_gamma, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_bn4_beta, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_bn4_mean, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_bn4_var, sizeof(float) * (kernel_num_conv));

    cudaMemcpy(d_bn4_gamma, bn4_gamma, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn4_beta, bn4_beta, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn4_mean, bn4_mean, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn4_var, bn4_var, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);

    batch_normalization << <256, blockSize >> > (d_conv4_out, h_conv * h_conv, d_bn4_mean, d_bn4_var, d_bn4_gamma, d_bn4_beta);
    cudaFree(d_bn4_beta);
    cudaFree(d_bn4_gamma);
    cudaFree(d_bn4_mean);
    cudaFree(d_bn4_var);
    free(bn4_beta);
    free(bn4_gamma);
    free(bn4_mean);
    free(bn4_var);
    //Relu------------------------------------------------------------------------------------------
    Relu << <10, 100 >> > (d_conv4_out, h_conv * h_conv * kernel_num_conv);
    cudaDeviceSynchronize();
    //maxpool3--------------------------------------------------------------------------------------
    printf("Block3, pool3");
    sysUsecTime();
    float* pool3_out;
    float* d_pool3_out;
    pool3_out = (float*)malloc(sizeof(float) * (h_conv * h_conv * kernel_num_conv / 4));
    cudaMalloc((void**)&d_pool3_out, sizeof(float) * (h_conv * h_conv * kernel_num_conv / 4));
    maxpooling2d << <1, 1 >> > (d_pool3_out, d_conv4_out, h_conv, kernel_num_conv);
    cudaFree(d_conv4_out);
    free(conv4_out);
    cudaDeviceSynchronize();
    printf("--------------------------------------Block 3 End------------------------------------------\n");
    // ----------------------------------------Block 4-----------------------------------------------
    // ----------------------------------------------------------------------------------------------
    // conv5-----------------------------------------------------------------------------------------
    printf("Block4, Conv5");
    sysUsecTime();
    float* conv5_out, * conv5_w, * conv5_b;
    float* d_conv5_out, * d_conv5_w, * d_conv5_b, * d_tmp5;
    h_conv = 28;
    in_conv = 256;
    kernel_num_conv = 512;

    conv5_w = (float*)malloc(sizeof(float) * (9 * in_conv * kernel_num_conv));
    conv5_b = (float*)malloc(sizeof(float) * (kernel_num_conv));
    conv5_out = (float*)malloc(sizeof(float) * (h_conv * h_conv * kernel_num_conv));

    loadweights(conv5_w, weightpath, "features.15.weight.txt");
    loadweights(conv5_b, weightpath, "features.15.bias.txt");

    cudaMalloc((void**)&d_conv5_w, sizeof(float) * (9 * in_conv * kernel_num_conv));
    cudaMalloc((void**)&d_conv5_b, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_conv5_out, sizeof(float) * (h_conv * h_conv * kernel_num_conv));
    cudaMalloc((void**)&d_tmp5, sizeof(float) * (in_conv * 9 * h_conv * h_conv));

    cudaMemcpy(d_conv5_w, conv5_w, sizeof(float) * (9 * in_conv * kernel_num_conv), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv5_b, conv5_b, sizeof(float) * (kernel_num_conv), cudaMemcpyHostToDevice);
    changeform << <1, 1 >> > (d_pool3_out, h_conv, in_conv, d_tmp5);
    printf("gemm");
    sysUsecTime();
    printf("the size4 is %d,%d,%d\n", kernel_num_conv, 9 * in_conv, h_conv* h_conv);
    //512 2304 784
    dim3 dimGrid5(49, 32);
    dim3 dimBlock5(16, 16);
    Conv << <h_conv * h_conv, kernel_num_conv >> > (d_conv5_out, d_conv5_w, d_tmp5, d_conv5_b, kernel_num_conv, 9 * in_conv, h_conv * h_conv);
    Conv2 << <dimGrid5, dimBlock5 >> > (d_conv5_out, d_conv5_w, d_tmp5, d_conv5_b, kernel_num_conv, 9 * in_conv, h_conv* h_conv);
    cudaFree(d_conv5_w);
    cudaFree(d_conv5_b);
    cudaFree(d_pool3_out);
    cudaFree(d_tmp5);
    free(conv5_w);
    free(conv5_b);
    free(pool3_out);

    // bn5------------------------------------------------------------------------------------------
    printf("Block4, bn5");
    sysUsecTime();
    float* bn5_gamma, * bn5_beta, * bn5_mean, * bn5_var;
    float* d_bn5_gamma, * d_bn5_beta, * d_bn5_mean, * d_bn5_var;

    bn5_gamma = (float*)malloc(sizeof(float) * (kernel_num_conv));
    bn5_beta = (float*)malloc(sizeof(float) * (kernel_num_conv));
    bn5_mean = (float*)malloc(sizeof(float) * (kernel_num_conv));
    bn5_var = (float*)malloc(sizeof(float) * (kernel_num_conv));

    loadweights(bn5_gamma, weightpath, "features.16.weight.txt");
    loadweights(bn5_beta, weightpath, "features.16.bias.txt");
    loadweights(bn5_mean, weightpath, "features.16.running_mean.txt");
    loadweights(bn5_var, weightpath, "features.16.running_var.txt");

    cudaMalloc((void**)&d_bn5_gamma, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_bn5_beta, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_bn5_mean, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_bn5_var, sizeof(float) * (kernel_num_conv));

    cudaMemcpy(d_bn5_gamma, bn5_gamma, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn5_beta, bn5_beta, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn5_mean, bn5_mean, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn5_var, bn5_var, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);

    batch_normalization << <512, blockSize >> > (d_conv5_out, h_conv * h_conv, d_bn5_mean, d_bn5_var, d_bn5_gamma, d_bn5_beta);
    cudaFree(d_bn5_beta);
    cudaFree(d_bn5_gamma);
    cudaFree(d_bn5_mean);
    cudaFree(d_bn5_var);
    free(bn5_beta);
    free(bn5_gamma);
    free(bn5_mean);
    free(bn5_var);
    //Relu------------------------------------------------------------------------------------------
    Relu << <10, 100 >> > (d_conv5_out, h_conv * h_conv * kernel_num_conv);
    cudaDeviceSynchronize();
    printf("\n----------------------------------------------------------\n");
    // conv6-----------------------------------------------------------------------------------------
    printf("Block4, Conv6");
    sysUsecTime();
    h_conv = 28;
    in_conv = 512;
    kernel_num_conv = 512;

    float* conv6_out, * conv6_w, * conv6_b;
    float* d_conv6_out, * d_conv6_w, * d_conv6_b, * d_tmp6;


    conv6_w = (float*)malloc(sizeof(float) * (9 * in_conv * kernel_num_conv));
    conv6_b = (float*)malloc(sizeof(float) * (kernel_num_conv));
    conv6_out = (float*)malloc(sizeof(float) * (h_conv * h_conv * kernel_num_conv));

    loadweights(conv6_w, weightpath, "features.18.weight.txt");
    loadweights(conv6_b, weightpath, "features.18.bias.txt");

    cudaMalloc((void**)&d_conv6_w, sizeof(float) * (9 * in_conv * kernel_num_conv));
    cudaMalloc((void**)&d_conv6_b, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_conv6_out, sizeof(float) * (h_conv * h_conv * kernel_num_conv));
    cudaMalloc((void**)&d_tmp6, sizeof(float) * (in_conv * 9 * h_conv * h_conv));

    cudaMemcpy(d_conv6_w, conv6_w, sizeof(float) * (9 * in_conv * kernel_num_conv), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv6_b, conv6_b, sizeof(float) * (kernel_num_conv), cudaMemcpyHostToDevice);
    changeform << <1, 1 >> > (d_conv5_out, h_conv, in_conv, d_tmp6);
    printf("gemm");
    sysUsecTime();
    printf("the size5 is %d,%d,%d\n", kernel_num_conv, 9 * in_conv, h_conv* h_conv);
    //512 46008 784
    dim3 dimGrid6(49, 32);
    dim3 dimBlock6(16, 16);
    Conv2 << <dimGrid6, dimBlock6 >> > (d_conv6_out, d_conv6_w, d_tmp6, d_conv6_b, kernel_num_conv, 9 * in_conv, h_conv* h_conv);

   // Conv << <h_conv * h_conv, kernel_num_conv >> > (d_conv6_out, d_conv6_w, d_tmp6, d_conv6_b, kernel_num_conv, 9 * in_conv, h_conv * h_conv);
    cudaFree(d_conv6_w);
    cudaFree(d_conv6_b);
    cudaFree(d_conv5_out);
    cudaFree(d_tmp6);
    free(conv6_w);
    free(conv6_b);
    free(conv5_out);
    // bn6------------------------------------------------------------------------------------------
    printf("Block4, bn6");
    sysUsecTime();
    float* bn6_gamma, * bn6_beta, * bn6_mean, * bn6_var;
    float* d_bn6_gamma, * d_bn6_beta, * d_bn6_mean, * d_bn6_var;

    bn6_gamma = (float*)malloc(sizeof(float) * (kernel_num_conv));
    bn6_beta = (float*)malloc(sizeof(float) * (kernel_num_conv));
    bn6_mean = (float*)malloc(sizeof(float) * (kernel_num_conv));
    bn6_var = (float*)malloc(sizeof(float) * (kernel_num_conv));

    loadweights(bn6_gamma, weightpath, "features.19.weight.txt");
    loadweights(bn6_beta, weightpath, "features.19.bias.txt");
    loadweights(bn6_mean, weightpath, "features.19.running_mean.txt");
    loadweights(bn6_var, weightpath, "features.19.running_var.txt");

    cudaMalloc((void**)&d_bn6_gamma, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_bn6_beta, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_bn6_mean, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_bn6_var, sizeof(float) * (kernel_num_conv));

    cudaMemcpy(d_bn6_gamma, bn6_gamma, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn6_beta, bn6_beta, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn6_mean, bn6_mean, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn6_var, bn6_var, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);

    batch_normalization << <512, blockSize >> > (d_conv6_out, h_conv * h_conv, d_bn6_mean, d_bn6_var, d_bn6_gamma, d_bn6_beta);
    cudaFree(d_bn6_beta);
    cudaFree(d_bn6_gamma);
    cudaFree(d_bn6_mean);
    cudaFree(d_bn6_var);
    free(bn6_beta);
    free(bn6_gamma);
    free(bn6_mean);
    free(bn6_var);
    //Relu------------------------------------------------------------------------------------------
    Relu << <10, 100 >> > (d_conv6_out, h_conv * h_conv * kernel_num_conv);
    cudaDeviceSynchronize();
    //maxpool4--------------------------------------------------------------------------------------
    printf("Block4, pool4");
    sysUsecTime();
    float* pool4_out;
    float* d_pool4_out;
    pool4_out = (float*)malloc(sizeof(float) * (h_conv * h_conv * kernel_num_conv / 4));
    cudaMalloc((void**)&d_pool4_out, sizeof(float) * (h_conv * h_conv * kernel_num_conv / 4));
    maxpooling2d << <1, 1 >> > (d_pool4_out, d_conv6_out, h_conv, kernel_num_conv);
    cudaFree(d_conv6_out);
    free(conv6_out);
    cudaDeviceSynchronize();
    printf("--------------------------------------Block 4 End------------------------------------------\n");

    // ----------------------------------------Block 5-----------------------------------------------
    // ----------------------------------------------------------------------------------------------
    // conv7-----------------------------------------------------------------------------------------
    printf("Block5, Conv7");
    sysUsecTime();
    float* conv7_out, * conv7_w, * conv7_b;
    float* d_conv7_out, * d_conv7_w, * d_conv7_b, * d_tmp7;
    h_conv = 14;
    in_conv = 512;
    kernel_num_conv = 512;

    conv7_w = (float*)malloc(sizeof(float) * (9 * in_conv * kernel_num_conv));
    conv7_b = (float*)malloc(sizeof(float) * (kernel_num_conv));
    conv7_out = (float*)malloc(sizeof(float) * (h_conv * h_conv * kernel_num_conv));

    loadweights(conv7_w, weightpath, "features.22.weight.txt");
    loadweights(conv7_b, weightpath, "features.22.bias.txt");

    cudaMalloc((void**)&d_conv7_w, sizeof(float) * (9 * in_conv * kernel_num_conv));
    cudaMalloc((void**)&d_conv7_b, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_conv7_out, sizeof(float) * (h_conv * h_conv * kernel_num_conv));
    cudaMalloc((void**)&d_tmp7, sizeof(float) * (in_conv * 9 * h_conv * h_conv));

    cudaMemcpy(d_conv7_w, conv7_w, sizeof(float) * (9 * in_conv * kernel_num_conv), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv7_b, conv7_b, sizeof(float) * (kernel_num_conv), cudaMemcpyHostToDevice);
    changeform << <1, 1 >> > (d_pool4_out, h_conv, in_conv, d_tmp7);
    printf("gemm");
    sysUsecTime();
    printf("the size6 is %d,%d,%d\n", kernel_num_conv, 9 * in_conv, h_conv* h_conv);
    //512 4608 196
    dim3 dimGrid7(49, 128);
    dim3 dimBlock7(4, 4);
    Conv3 << <dimGrid7, dimBlock7 >> > (d_conv7_out, d_conv7_w, d_tmp7, d_conv7_b, kernel_num_conv, 9 * in_conv, h_conv * h_conv);
    cudaFree(d_conv7_w);
    cudaFree(d_conv7_b);
    cudaFree(d_pool4_out);
    cudaFree(d_tmp7);
    free(conv7_w);
    free(conv7_b);
    free(pool4_out);

    // bn7------------------------------------------------------------------------------------------
    printf("Block5, bn7");
    sysUsecTime();
    float* bn7_gamma, * bn7_beta, * bn7_mean, * bn7_var;
    float* d_bn7_gamma, * d_bn7_beta, * d_bn7_mean, * d_bn7_var;

    bn7_gamma = (float*)malloc(sizeof(float) * (kernel_num_conv));
    bn7_beta = (float*)malloc(sizeof(float) * (kernel_num_conv));
    bn7_mean = (float*)malloc(sizeof(float) * (kernel_num_conv));
    bn7_var = (float*)malloc(sizeof(float) * (kernel_num_conv));

    loadweights(bn7_gamma, weightpath, "features.23.weight.txt");
    loadweights(bn7_beta, weightpath, "features.23.bias.txt");
    loadweights(bn7_mean, weightpath, "features.23.running_mean.txt");
    loadweights(bn7_var, weightpath, "features.23.running_var.txt");

    cudaMalloc((void**)&d_bn7_gamma, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_bn7_beta, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_bn7_mean, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_bn7_var, sizeof(float) * (kernel_num_conv));

    cudaMemcpy(d_bn7_gamma, bn7_gamma, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn7_beta, bn7_beta, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn7_mean, bn7_mean, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn7_var, bn7_var, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);

    batch_normalization << <512, blockSize >> > (d_conv7_out, h_conv * h_conv, d_bn7_mean, d_bn7_var, d_bn7_gamma, d_bn7_beta);
    cudaFree(d_bn7_beta);
    cudaFree(d_bn7_gamma);
    cudaFree(d_bn7_mean);
    cudaFree(d_bn7_var);
    free(bn7_beta);
    free(bn7_gamma);
    free(bn7_mean);
    free(bn7_var);
    //Relu------------------------------------------------------------------------------------------
    Relu << <10, 100 >> > (d_conv7_out, h_conv * h_conv * kernel_num_conv);
    cudaDeviceSynchronize();
    printf("\n----------------------------------------------------------\n");
    // conv8-----------------------------------------------------------------------------------------
    printf("Block5, Conv8");
    sysUsecTime();
    h_conv = 14;
    in_conv = 512;
    kernel_num_conv = 512;

    float* conv8_out, * conv8_w, * conv8_b;
    float* d_conv8_out, * d_conv8_w, * d_conv8_b, * d_tmp8;


    conv8_w = (float*)malloc(sizeof(float) * (9 * in_conv * kernel_num_conv));
    conv8_b = (float*)malloc(sizeof(float) * (kernel_num_conv));
    conv8_out = (float*)malloc(sizeof(float) * (h_conv * h_conv * kernel_num_conv));

    loadweights(conv8_w, weightpath, "features.25.weight.txt");
    loadweights(conv8_b, weightpath, "features.25.bias.txt");

    cudaMalloc((void**)&d_conv8_w, sizeof(float) * (9 * in_conv * kernel_num_conv));
    cudaMalloc((void**)&d_conv8_b, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_conv8_out, sizeof(float) * (h_conv * h_conv * kernel_num_conv));
    cudaMalloc((void**)&d_tmp8, sizeof(float) * (in_conv * 9 * h_conv * h_conv));

    cudaMemcpy(d_conv8_w, conv8_w, sizeof(float) * (9 * in_conv * kernel_num_conv), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv8_b, conv8_b, sizeof(float) * (kernel_num_conv), cudaMemcpyHostToDevice);
    changeform << <1, 1 >> > (d_conv7_out, h_conv, in_conv, d_tmp8);
    printf("gemm");
    sysUsecTime();
    printf("the size7 is %d,%d,%d\n", kernel_num_conv, 9 * in_conv, h_conv* h_conv);
    //512 4608 196
    dim3 dimGrid8(49, 128);
    dim3 dimBlock8(4, 4);
    Conv << <h_conv * h_conv, kernel_num_conv >> > (d_conv8_out, d_conv8_w, d_tmp8, d_conv8_b, kernel_num_conv, 9 * in_conv, h_conv * h_conv);
    //Conv3 << <dimGrid8, dimBlock8 >> > (d_conv8_out, d_conv8_w, d_tmp8, d_conv8_b, kernel_num_conv, 9 * in_conv, h_conv * h_conv);

    cudaFree(d_conv8_w);
    cudaFree(d_conv8_b);
    cudaFree(d_conv7_out);
    cudaFree(d_tmp8);
    free(conv8_w);
    free(conv8_b);
    free(conv7_out);
    // bn8------------------------------------------------------------------------------------------
    printf("Block5, bn8");
    sysUsecTime();
    float* bn8_gamma, * bn8_beta, * bn8_mean, * bn8_var;
    float* d_bn8_gamma, * d_bn8_beta, * d_bn8_mean, * d_bn8_var;

    bn8_gamma = (float*)malloc(sizeof(float) * (kernel_num_conv));
    bn8_beta = (float*)malloc(sizeof(float) * (kernel_num_conv));
    bn8_mean = (float*)malloc(sizeof(float) * (kernel_num_conv));
    bn8_var = (float*)malloc(sizeof(float) * (kernel_num_conv));

    loadweights(bn8_gamma, weightpath, "features.26.weight.txt");
    loadweights(bn8_beta, weightpath, "features.26.bias.txt");
    loadweights(bn8_mean, weightpath, "features.26.running_mean.txt");
    loadweights(bn8_var, weightpath, "features.26.running_var.txt");

    cudaMalloc((void**)&d_bn8_gamma, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_bn8_beta, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_bn8_mean, sizeof(float) * (kernel_num_conv));
    cudaMalloc((void**)&d_bn8_var, sizeof(float) * (kernel_num_conv));

    cudaMemcpy(d_bn8_gamma, bn8_gamma, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn8_beta, bn8_beta, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn8_mean, bn8_mean, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn8_var, bn8_var, sizeof(float) * kernel_num_conv, cudaMemcpyHostToDevice);

    batch_normalization << <512, blockSize >> > (d_conv8_out, h_conv * h_conv, d_bn8_mean, d_bn8_var, d_bn8_gamma, d_bn8_beta);
    cudaFree(d_bn8_beta);
    cudaFree(d_bn8_gamma);
    cudaFree(d_bn8_mean);
    cudaFree(d_bn8_var);
    free(bn8_beta);
    free(bn8_gamma);
    free(bn8_mean);
    free(bn8_var);
    //Relu------------------------------------------------------------------------------------------
    Relu << <10, 100 >> > (d_conv8_out, h_conv * h_conv * kernel_num_conv);
    cudaDeviceSynchronize();
    //maxpool5--------------------------------------------------------------------------------------
    printf("Block5, pool5");
    sysUsecTime();
    float* pool5_out;
    float* d_pool5_out;
    pool5_out = (float*)malloc(sizeof(float) * (h_conv * h_conv * kernel_num_conv / 4));
    cudaMalloc((void**)&d_pool5_out, sizeof(float) * (h_conv * h_conv * kernel_num_conv / 4));
    maxpooling2d << <1, 1 >> > (d_pool5_out, d_conv8_out, h_conv, kernel_num_conv);
    cudaMemcpy(pool5_out, d_pool5_out, sizeof(float) * (h_conv * h_conv * kernel_num_conv / 4), cudaMemcpyDeviceToHost);
    cudaFree(d_conv8_out);
    free(conv8_out);
    cudaDeviceSynchronize();
    printf("--------------------------------------Block 5 End------------------------------------------\n");
    printf("--------------------------------------Block 5 End------------------------------------------\n");
    //---------------------------------------------FC layer------------------------------------------
    // fc1-------------------------------------------------------------------------------------------
    printf("fc1");
    sysUsecTime();
    float* fc1_w, * fc1_b;
    float* d_fc1_out, * d_fc1_w, * d_fc1_b;

    fc1_w = (float*)malloc(sizeof(float) * (4096 * 512 * 7 * 7));
    fc1_b = (float*)malloc(sizeof(float) * (4096));

    loadweights(fc1_w, weightpath, "classifier.0.weight.txt");
    loadweights(fc1_b, weightpath, "classifier.0.bias.txt");

    cudaMalloc((void**)&d_fc1_w, sizeof(float) * (4096 * 512 * 7 * 7));
    cudaMalloc((void**)&d_fc1_b, sizeof(float) * (4096));
    cudaMalloc((void**)&d_fc1_out, sizeof(float) * (4096));
    cudaMemcpy(d_fc1_w, fc1_w, sizeof(float) * (4096 * 512 * 7 * 7), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_b, fc1_b, sizeof(float) * (4096), cudaMemcpyHostToDevice);
    // cudaMalloc((void**)&d_tmp1, sizeof(float) * (3 * 9 * 224 * 224));
    gemm(d_fc1_w, d_pool5_out, d_fc1_out, d_fc1_b, 4096, 1, 25088);
    cudaFree(d_fc1_w);
    cudaFree(d_fc1_b);
    free(fc1_b);
    free(fc1_w);
    // relu1-------------------------------------------------------------------------------------------
    Relu << <10, 100 >> > (d_fc1_out, 4096);
    printf("--------------------------------------FC layer 1 End------------------------------------------\n");

    // fc2-------------------------------------------------------------------------------------------
    printf("fc2");
    sysUsecTime();
    float* fc2_out, * fc2_w, * fc2_b;
    float* d_fc2_out, * d_fc2_w, * d_fc2_b;

    fc2_w = (float*)malloc(sizeof(float) * (4096 * 4096));
    fc2_b = (float*)malloc(sizeof(float) * (4096));
    fc2_out = (float*)malloc(sizeof(float) * (4096));

    loadweights(fc2_w, weightpath, "classifier.3.weight.txt");
    loadweights(fc2_b, weightpath, "classifier.3.bias.txt");

    cudaMalloc((void**)&d_fc2_w, sizeof(float) * (4096 * 4096));
    cudaMalloc((void**)&d_fc2_b, sizeof(float) * (4096));
    cudaMalloc((void**)&d_fc2_out, sizeof(float) * (4096));
    cudaMemcpy(d_fc2_w, fc2_w, sizeof(float) * (4096 * 4096), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_b, fc2_b, sizeof(float) * (4096), cudaMemcpyHostToDevice);
    gemm(d_fc2_w, d_fc1_out, d_fc2_out, d_fc2_b, 4096, 1, 4096);
    // relu1-------------------------------------------------------------------------------------------
    Relu << <10, 100 >> > (d_fc2_out, 4096);
    cudaMemcpy(fc2_out, d_fc2_out, sizeof(float) * (4096), cudaMemcpyDeviceToHost);
    cudaFree(d_fc2_w);
    cudaFree(d_fc2_b);
    cudaFree(d_fc1_out);
    free(fc2_b);
    free(fc2_w);
    printf("--------------------------------------FC layer 2 End------------------------------------------\n");

    // fc3-------------------------------------------------------------------------------------------
    printf("fc3");
    sysUsecTime();

    float* fc3_out, * fc3_w, * fc3_b, * softmax_out;
    float* d_fc3_out, * d_fc3_w, * d_fc3_b, * d_softmax_out;

    fc3_w = (float*)malloc(sizeof(float) * (4096 * 1000));
    fc3_b = (float*)malloc(sizeof(float) * (1000));
    fc3_out = (float*)malloc(sizeof(float) * (1000));
    loadweights(fc3_w, weightpath, "classifier.6.weight.txt");
    loadweights(fc3_b, weightpath, "classifier.6.bias.txt");

    cudaMalloc((void**)&d_fc3_w, sizeof(float) * (4096 * 1000));
    cudaMalloc((void**)&d_fc3_b, sizeof(float) * (1000));
    cudaMalloc((void**)&d_fc3_out, sizeof(float) * (1000));
    cudaMemcpy(d_fc3_w, fc3_w, sizeof(float) * (4096 * 1000), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc3_b, fc3_b, sizeof(float) * (1000), cudaMemcpyHostToDevice);
    gemm(d_fc3_w, d_fc2_out, d_fc3_out, d_fc3_b, 1000, 1, 4096);

    softmax_out = (float*)malloc(sizeof(float) * (1000));
    cudaMalloc((void**)&d_softmax_out, sizeof(float) * 1000);
    //printf("the feature 3 num is : %f\n", sum2);
    printf("--------------------------------------FC layer 3 End------------------------------------------\n");

    //softmax-----------------------------------------------------------------------------------------
    softmax << <1, 1 >> > (d_softmax_out, d_fc3_out, 1000);
    free(fc3_b);
    free(fc3_w);
    cudaFree(d_fc3_w);
    cudaFree(d_fc3_b);
    cudaFree(d_fc2_out);
    printf("--------------------------------------Softmax End------------------------------------------\n");

    float* tmp;
    tmp = (float*)malloc(sizeof(float) * (1000));
    loadimage(tmp, "./output/final_output.txt");
    cudaMemcpy(softmax_out, d_softmax_out, sizeof(float) * (1000), cudaMemcpyDeviceToHost);
    FILE* fp = NULL;
    fp = fopen(outputfile, "w+");
    for (int i = 0;i < 1000;++i) {
        fprintf(fp, "%f, ", softmax_out[i]);
        if (abs(tmp[i] - softmax_out[i]) > 1e-7) { printf("place:%d , true is: %f, now is: %f \n", i, tmp[i], softmax_out[i]); }
    }
    fprintf(fp, "\n");
    fclose(fp);
    printf("--------------------------------------Program End------------------------------------------\n");
    cudaFree(d_fc3_out);
    cudaFree(d_softmax_out);
    free(fc2_out);
    free(tmp);
    free(softmax_out);
    free(fc3_out);
    system("PAUSE");
    return 0;
}