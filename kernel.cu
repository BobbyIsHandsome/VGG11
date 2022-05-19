#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define INPUTSHAPE 3 * 244 * 244
#define OUTPUTSHAPE 1000
#define TESTNUM 1
#define ITERNUM 1
static const int BLOCK_SIZE = 16;//the block size for matrix tilling
static const int BLOCK_SIZE2 = 32;//the block size for matrix tilling
float inputArr[TESTNUM][INPUTSHAPE];
float benchOutArr[TESTNUM][OUTPUTSHAPE];
int batch_size = 1;
int data_size = 1;


void loadweights(float* weight, const char* path, int size) {
    FILE* fp = fopen(path, "rb");
    fseek(fp, 0, SEEK_SET);
    fread(weight, sizeof(float), size, fp);
    fclose(fp);
}

//load weights and bias of single layer
void initWeights(float* d_weights, float* d_bias, const char* weight_path, const char* bias_path, int weight_size, int bias_size)
{
    //load into memory
    float* conv_weights = new float[weight_size];
    float* conv_bias = new float[bias_size];
    loadweights(conv_weights, weight_path, weight_size);
    loadweights(conv_bias, bias_path, bias_size);

    //copy to gpu
    cudaMemcpy(d_weights, conv_weights, sizeof(float) * (weight_size), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, conv_bias, sizeof(float) * (bias_size), cudaMemcpyHostToDevice);

    //free space in memory
    free(conv_weights);
    free(conv_bias);

}




__global__ void Conv(float* fpMatrixC, float* fpMatrixA, float* fpMatrixB, float* bias, int m, int k, int n) {
    /*
        fpMatrixA: left matrix
        fpMatrixB: right matrix
        fpMatrixC: output matrix
        bias: bias
        m,n,k: A is m*k, B is k*n, C is m*n
    */
    int nRow = threadIdx.x;
    int nCol = blockIdx.x;
    float fCVal = 0.0;
    float* pta = &fpMatrixA[nRow * k];
    float* ptb = &fpMatrixB[nCol * k];
    for (int i = 0; i < k / 2; i++)
    {
        fCVal += (*pta) * (*ptb);
        ++pta;
        ++ptb;
        fCVal += (*pta) * (*ptb);
        ++pta;
        ++ptb;
    }
    if (k % 2 != 0) fCVal += (*pta) * (*ptb);
    fpMatrixC[nRow * n + nCol] = fCVal + bias[nRow];
}


__global__ void Conv2(float* fpMatrixC, float* fpMatrixA, float* fpMatrixB, float* bias, int m, int k, int n) {
    /*
        fpMatrixA: left matrix
        fpMatrixB: right matrix
        fpMatrixC: output matrix
        bias: bias
        m,n,k: A is m*k, B is k*n, C is m*n
    */
    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    float fCVal = 0.0f;

    __shared__ float shTileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shTileB[BLOCK_SIZE][BLOCK_SIZE];

    int nIter = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float* ptrA = &fpMatrixA[nRow * k + 0 * BLOCK_SIZE + threadIdx.x];
    float* ptrB = ptrB = &fpMatrixB[0 * BLOCK_SIZE + threadIdx.y + nCol * k];

    for (int i = 0; i < nIter; i++)
    {
        // load data from global memory to shared memory
        //shTileA[threadIdx.y][threadIdx.x] = fpMatrixA[nRow * k + i * BLOCK_SIZE + threadIdx.x];
        //shTileB[threadIdx.x][threadIdx.y] = fpMatrixB[i * BLOCK_SIZE + threadIdx.y  + nCol*k];
        shTileA[threadIdx.y][threadIdx.x] = *ptrA;
        if (i * BLOCK_SIZE + threadIdx.y + nCol * k >= n * k) {
            shTileB[threadIdx.x][threadIdx.y] = 0;
        }
        else {
            shTileB[threadIdx.x][threadIdx.y] = *ptrB;
        }
        // sync to wait for all threads in one block to finish loading datas
        __syncthreads();

        // sub-matrix multiply
        for (int l = 0; l < BLOCK_SIZE; l += 2)
        {
            fCVal += shTileA[threadIdx.y][l] * shTileB[threadIdx.x][l];
            fCVal += shTileA[threadIdx.y][l + 1] * shTileB[threadIdx.x][l + 1];
        }

        // sync to wait for all threads in one block to finish compute
        __syncthreads();
        i++;
        ptrA = ptrA + BLOCK_SIZE;
        ptrB = ptrB + BLOCK_SIZE;
        shTileA[threadIdx.y][threadIdx.x] = *ptrA;
        if (i * BLOCK_SIZE + threadIdx.y + nCol * k >= n * k) {
            shTileB[threadIdx.x][threadIdx.y] = 0;
        }
        else {
            shTileB[threadIdx.x][threadIdx.y] = *ptrB;
        }
        // sync to wait for all threads in one block to finish loading datas
        __syncthreads();

        // sub-matrix multiply
        for (int l = 0; l < BLOCK_SIZE; l += 2)
        {
            fCVal += shTileA[threadIdx.y][l] * shTileB[threadIdx.x][l];
            fCVal += shTileA[threadIdx.y][l + 1] * shTileB[threadIdx.x][l + 1];
        }

        // sync to wait for all threads in one block to finish compute
        __syncthreads();
        ptrA = ptrA + BLOCK_SIZE;
        ptrB = ptrB + BLOCK_SIZE;
    }

    // store results into global memory
    if (nCol < n) {
        fpMatrixC[nRow * n + nCol] = fCVal + bias[nRow];
    }
}

__global__ void Conv3(float* fpMatrixC, float* fpMatrixA, float* fpMatrixB, float* bias, int m, int k, int n) {
    /*
        fpMatrixA: left matrix
        fpMatrixB: right matrix
        fpMatrixC: output matrix
        bias: bias
        m,n,k: A is m*k, B is k*n, C is m*n
    */
    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    float fCVal = 0.0f;

    __shared__ float shTileA[BLOCK_SIZE2][BLOCK_SIZE2];
    __shared__ float shTileB[BLOCK_SIZE2][BLOCK_SIZE2];

    int nIter = (k + BLOCK_SIZE2 - 1) / BLOCK_SIZE2;
    float* ptrA = &fpMatrixA[nRow * k + 0 * BLOCK_SIZE2 + threadIdx.x];
    float* ptrB = ptrB = &fpMatrixB[0 * BLOCK_SIZE2 + threadIdx.y + nCol * k];

    for (int i = 0; i < nIter; i++)
    {
        // load data from global memory to shared memory
        //shTileA[threadIdx.y][threadIdx.x] = fpMatrixA[nRow * k + i * BLOCK_SIZE + threadIdx.x];
        //shTileB[threadIdx.x][threadIdx.y] = fpMatrixB[i * BLOCK_SIZE + threadIdx.y  + nCol*k];
        shTileA[threadIdx.y][threadIdx.x] = *ptrA;
        if (i * BLOCK_SIZE2 + threadIdx.y + nCol * k >= n * k) {
            shTileB[threadIdx.x][threadIdx.y] = 0;
        }
        else {
            shTileB[threadIdx.x][threadIdx.y] = *ptrB;
        }
        // sync to wait for all threads in one block to finish loading datas
        __syncthreads();

        // sub-matrix multiply
        for (int l = 0; l < BLOCK_SIZE2; l += 2)
        {
            fCVal += shTileA[threadIdx.y][l] * shTileB[threadIdx.x][l];
            fCVal += shTileA[threadIdx.y][l + 1] * shTileB[threadIdx.x][l + 1];
        }

        // sync to wait for all threads in one block to finish compute
        __syncthreads();
        i++;
        ptrA = ptrA + BLOCK_SIZE2;
        ptrB = ptrB + BLOCK_SIZE2;
        shTileA[threadIdx.y][threadIdx.x] = *ptrA;
        if (i * BLOCK_SIZE2 + threadIdx.y + nCol * k >= n * k) {
            shTileB[threadIdx.x][threadIdx.y] = 0;
        }
        else {
            shTileB[threadIdx.x][threadIdx.y] = *ptrB;
        }
        // sync to wait for all threads in one block to finish loading datas
        __syncthreads();

        // sub-matrix multiply
        for (int l = 0; l < BLOCK_SIZE2; l += 2)
        {
            fCVal += shTileA[threadIdx.y][l] * shTileB[threadIdx.x][l];
            fCVal += shTileA[threadIdx.y][l + 1] * shTileB[threadIdx.x][l + 1];
        }

        // sync to wait for all threads in one block to finish compute
        __syncthreads();
        ptrA = ptrA + BLOCK_SIZE2;
        ptrB = ptrB + BLOCK_SIZE2;
    }

    // store results into global memory
    if (nCol < n) {
        fpMatrixC[nRow * n + nCol] = fCVal + bias[nRow];
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
    //  use the input feature map to get the matrix form
    int step_table[9][2] = { {-1,-1},{-1,0},{-1,1},
                            {0,-1},{0,0},{0,1},
                            {1,-1},{1,0},{1,1} };
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < in_channel; ++k) {
                for (int r = 0;r < 9; ++r) {
                    int newi = i + step_table[r][0];
                    int newj = j + step_table[r][1];
                    tmp_out[9 * in_channel * (n * i + j) + 9 * k + r] = (newi >= 0 && newj >= 0 && newi < n&& newj < n) ? in_map[n * n * k + newi * n + newj] : 0.0;
                }
            }
        }
    }
}

__global__ void Relu(float* c, unsigned int n)
{
    /*
        c: input and output matrix
        n: length of c
    */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < n) {
        if (c[tid] < 0) {
            c[tid] = 0;
        }
        tid += blockDim.x * gridDim.x;
    }

}


__device__ float max(float a, float b, float c, float d)
{
    return max(max(max(a, b), c), d);
}


__global__ void MaxPool2D(float* out, float* in, int n, int channel)
{
    //  n is the height/width of the feature map.
    for (int c = 0; c < channel; ++c)
    {
        int newn = n / 2;
        for (int i = 0; i < newn;++i) {
            for (int j = 0;j < newn;++j) {
                out[newn * newn * c + i * newn + j] = max(in[n * n * c + n * i * 2 + 2 * j], in[n * n * c + n * i * 2 + 2 * j + 1], in[n * n * c + n * (i * 2 + 1) + 2 * j], in[n * n * c + n * (i * 2 + 1) + 2 * j + 1]);
            }
        }
    }
}

__global__ void dense(float* fpMatrixA, float* fpMatrixB,
    float* fpMatrixC, float* bias, int m, int n, int k)
{
    /*
        fpMatrixA: left matrix
        fpMatrixB: right matrix
        fpMatrixC: output matrix
        bias: bias
        m,n,k: A is m*k, B is k*n, C is m*n
    */
    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    float* pta = &fpMatrixA[nRow * k];
    float* ptb = &fpMatrixB[nCol];
    float fCVal = 0.0f;
    for (int i = 0; i < k; i++)
    {
        fCVal += (*pta) * (*ptb);
        ++pta;
        ptb += n;
    }
    fpMatrixC[nRow * n + nCol] = fCVal + bias[nRow];
}

//m*k input , k*n weight matrix , m*n output matrix , m is the batch size
void gemm(float* fpMatrixA, float* fpMatrixB,
    float* fpMatrixC, float* bias, int m, int n, int k) {
    /*
        fpMatrixA: left matrix
        fpMatrixB: right matrix
        fpMatrixC: output matrix
        bias: bias
        m,n,k: A is m*k, B is k*n, C is m*n
    */
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
    dim3 dimBlock(dimx, dimy);
    dim3 dimGrid(n / dimx, m / dimy);
    if (m != 1000) {
        dense << <dimBlock, dimGrid >> > (fpMatrixA, fpMatrixB, fpMatrixC, bias, m, n, k);
    }
    else {
        dim3 dimBlock(1, 100);
        dim3 dimGrid(1, 10);
        dense << <dimGrid, dimBlock >> > (fpMatrixA, fpMatrixB, fpMatrixC, bias, m, n, k);
    }
}

void initModel(float** weights, float** bias)
{
    int weight_size, bias_size;

    float* d_conv1_w;
    float* d_conv1_b;
    weight_size = 64 * 3 * 3 * 3;
    bias_size = 64;
    cudaMalloc((void**)&d_conv1_w, sizeof(float) * weight_size);
    cudaMalloc((void**)&d_conv1_b, sizeof(float) * bias_size);
    initWeights(d_conv1_w, d_conv1_b, "./vgg16_weights/features_0_weight.txt",
        "./vgg16_weights/features_0_bias.txt", weight_size, bias_size);

    float* d_conv2_w;
    float* d_conv2_b;
    weight_size = 64 * 64 * 3 * 3;
    bias_size = 64;
    cudaMalloc((void**)&d_conv2_w, sizeof(float) * weight_size);
    cudaMalloc((void**)&d_conv2_b, sizeof(float) * bias_size);
    initWeights(d_conv2_w, d_conv2_b, "./vgg16_weights/features_2_weight.txt", "./vgg16_weights/features_2_bias.txt", weight_size, bias_size);

    float* d_conv3_w;
    float* d_conv3_b;
    weight_size = 128 * 64 * 3 * 3;
    bias_size = 128;
    cudaMalloc((void**)&d_conv3_w, sizeof(float) * weight_size);
    cudaMalloc((void**)&d_conv3_b, sizeof(float) * bias_size);
    initWeights(d_conv3_w, d_conv3_b, "./vgg16_weights/features_5_weight.txt", "./vgg16_weights/features_5_bias.txt", weight_size, bias_size);
    printf("I'm here \n");

    float* d_conv4_w;
    float* d_conv4_b;
    weight_size = 128 * 128 * 3 * 3;
    bias_size = 128;
    cudaMalloc((void**)&d_conv4_w, sizeof(float) * weight_size);
    cudaMalloc((void**)&d_conv4_b, sizeof(float) * bias_size);
    initWeights(d_conv4_w, d_conv4_b, "./vgg16_weights/features_7_weight.txt", "./vgg16_weights/features_7_bias.txt", weight_size, bias_size);

    float* d_conv5_w;
    float* d_conv5_b;
    weight_size = 256 * 128 * 3 * 3;
    bias_size = 256;
    cudaMalloc((void**)&d_conv5_w, sizeof(float) * weight_size);
    cudaMalloc((void**)&d_conv5_b, sizeof(float) * bias_size);
    initWeights(d_conv5_w, d_conv5_b, "./vgg16_weights/features_10_weight.txt", "./vgg16_weights/features_10_bias.txt", weight_size, bias_size);

    float* d_conv6_w;
    float* d_conv6_b;
    weight_size = 256 * 256 * 3 * 3;
    bias_size = 256;
    cudaMalloc((void**)&d_conv6_w, sizeof(float) * weight_size);
    cudaMalloc((void**)&d_conv6_b, sizeof(float) * bias_size);
    initWeights(d_conv6_w, d_conv6_b, "./vgg16_weights/features_12_weight.txt", "./vgg16_weights/features_12_bias.txt", weight_size, bias_size);

    float* d_conv7_w;
    float* d_conv7_b;
    weight_size = 256 * 256 * 3 * 3;
    bias_size = 256;
    cudaMalloc((void**)&d_conv7_w, sizeof(float) * weight_size);
    cudaMalloc((void**)&d_conv7_b, sizeof(float) * bias_size);
    initWeights(d_conv7_w, d_conv7_b, "./vgg16_weights/features_14_weight.txt", "./vgg16_weights/features_14_bias.txt", weight_size, bias_size);

    float* d_conv8_w;
    float* d_conv8_b;
    weight_size = 512 * 256 * 3 * 3;
    bias_size = 512;
    cudaMalloc((void**)&d_conv8_w, sizeof(float) * weight_size);
    cudaMalloc((void**)&d_conv8_b, sizeof(float) * bias_size);
    initWeights(d_conv8_w, d_conv8_b, "./vgg16_weights/features_17_weight.txt", "./vgg16_weights/features_17_bias.txt", weight_size, bias_size);

    float* d_conv9_w;
    float* d_conv9_b;
    weight_size = 512 * 512 * 3 * 3;
    bias_size = 512;
    cudaMalloc((void**)&d_conv9_w, sizeof(float) * weight_size);
    cudaMalloc((void**)&d_conv9_b, sizeof(float) * bias_size);
    initWeights(d_conv9_w, d_conv9_b, "./vgg16_weights/features_19_weight.txt", "./vgg16_weights/features_19_bias.txt", weight_size, bias_size);

    float* d_conv10_w;
    float* d_conv10_b;
    weight_size = 512 * 512 * 3 * 3;
    bias_size = 512;
    cudaMalloc((void**)&d_conv10_w, sizeof(float) * weight_size);
    cudaMalloc((void**)&d_conv10_b, sizeof(float) * bias_size);
    initWeights(d_conv10_w, d_conv10_b, "./vgg16_weights/features_21_weight.txt", "./vgg16_weights/features_21_bias.txt", weight_size, bias_size);

    float* d_conv11_w;
    float* d_conv11_b;
    weight_size = 512 * 512 * 3 * 3;
    bias_size = 512;
    cudaMalloc((void**)&d_conv11_w, sizeof(float) * weight_size);
    cudaMalloc((void**)&d_conv11_b, sizeof(float) * bias_size);
    initWeights(d_conv11_w, d_conv11_b, "./vgg16_weights/features_24_weight.txt", "./vgg16_weights/features_24_bias.txt", weight_size, bias_size);

    float* d_conv12_w;
    float* d_conv12_b;
    weight_size = 512 * 512 * 3 * 3;
    bias_size = 512;
    cudaMalloc((void**)&d_conv12_w, sizeof(float) * weight_size);
    cudaMalloc((void**)&d_conv12_b, sizeof(float) * bias_size);
    initWeights(d_conv12_w, d_conv12_b, "./vgg16_weights/features_26_weight.txt", "./vgg16_weights/features_26_bias.txt", weight_size, bias_size);

    float* d_conv13_w;
    float* d_conv13_b;
    weight_size = 512 * 512 * 3 * 3;
    bias_size = 512;
    cudaMalloc((void**)&d_conv13_w, sizeof(float) * weight_size);
    cudaMalloc((void**)&d_conv13_b, sizeof(float) * bias_size);
    initWeights(d_conv13_w, d_conv13_b, "./vgg16_weights/features_28_weight.txt", "./vgg16_weights/features_28_bias.txt", weight_size, bias_size);


    float* d_fc1_w;
    float* d_fc1_b;
    weight_size = 4096 * 25088;
    bias_size = 4096;
    cudaMalloc((void**)&d_fc1_w, sizeof(float) * weight_size);
    cudaMalloc((void**)&d_fc1_b, sizeof(float) * bias_size);
    initWeights(d_fc1_w, d_fc1_b, "./vgg16_weights/classifier_0_weight.txt", "./vgg16_weights/classifier_0_bias.txt", weight_size, bias_size);

    float* d_fc2_w;
    float* d_fc2_b;
    weight_size = 4096 * 4096;
    bias_size = 4096;
    cudaMalloc((void**)&d_fc2_w, sizeof(float) * weight_size);
    cudaMalloc((void**)&d_fc2_b, sizeof(float) * bias_size);
    initWeights(d_fc2_w, d_fc2_b, "./vgg16_weights/classifier_3_weight.txt", "./vgg16_weights/classifier_3_bias.txt", weight_size, bias_size);

    float* d_fc3_w;
    float* d_fc3_b;
    weight_size = 1000 * 4096;
    bias_size = 1000;
    cudaMalloc((void**)&d_fc3_w, sizeof(float) * weight_size);
    cudaMalloc((void**)&d_fc3_b, sizeof(float) * bias_size);
    initWeights(d_fc3_w, d_fc3_b, "./vgg16_weights/classifier_6_weight.txt", "./vgg16_weights/classifier_6_bias.txt", weight_size, bias_size);


    weights[0] = d_conv1_w;
    weights[1] = d_conv2_w;
    weights[2] = d_conv3_w;
    weights[3] = d_conv4_w;
    weights[4] = d_conv5_w;
    weights[5] = d_conv6_w;
    weights[6] = d_conv7_w;
    weights[7] = d_conv8_w;
    weights[8] = d_conv9_w;
    weights[9] = d_conv10_w;
    weights[10] = d_conv11_w;
    weights[11] = d_conv12_w;
    weights[12] = d_conv13_w;
    weights[13] = d_fc1_w;
    weights[14] = d_fc2_w;
    weights[15] = d_fc3_w;


    bias[0] = d_conv1_b;
    bias[1] = d_conv2_b;
    bias[2] = d_conv3_b;
    bias[3] = d_conv4_b;
    bias[4] = d_conv5_b;
    bias[5] = d_conv6_b;
    bias[6] = d_conv7_b;
    bias[7] = d_conv8_b;
    bias[8] = d_conv9_b;
    bias[9] = d_conv10_b;
    bias[10] = d_conv11_b;
    bias[11] = d_conv12_b;
    bias[12] = d_conv13_b;
    bias[13] = d_fc1_b;
    bias[14] = d_fc2_b;
    bias[15] = d_fc3_b;
    printf("I'm here \n");

}



void inference(float* d_input, float* d_output, float* kernel[], float* bias[], int input_channels = 3, int output_channels = 1000)
{


    //printf("Running Block1\n");
    //convloution layer 1, in_channels =3 ,out_channels = 64, output_shape = 244
    //printf("Running Layer1\n");
    float* d_conv1_features, * d_tmp1;
    cudaMalloc((void**)&d_conv1_features, sizeof(float) * (64 * 244 * 244));
    cudaMalloc((void**)&d_tmp1, sizeof(float) * (3 * 9 * 244 * 244));
    changeform << <1, 1 >> > (d_input, 244, 3, d_tmp1);
    dim3 dimGrid(3721, 4);
    dim3 dimBlock(16, 16);
    //Conv2 << <dimGrid, dimBlock >> > (d_conv1_features, kernel[0], d_tmp1, bias[0], 64, 9 * 3, 244 * 244);
    Conv << <244 * 244, 64 >> > (d_conv1_features, kernel[0], d_tmp1, bias[0], 64, 9 * 3, 244 * 244);
    cudaFree(d_tmp1);
    //relu
    Relu << <64, 244 >> > (d_conv1_features, 244 * 244 * 64);


    //convolution layer 2, in_channels = 64, out_channels = 64, output_shape = 244
    //printf("Running Layer2\n");
    float* d_conv2_features, * d_tmp2;
    cudaMalloc((void**)&d_conv2_features, sizeof(float) * (64 * 244 * 244));
    cudaMalloc((void**)&d_tmp2, sizeof(float) * (64 * 9 * 244 * 244));
    changeform << <1, 1 >> > (d_conv1_features, 244, 64, d_tmp2);
    //out_channels = 64 =4*16, shape = 244 *244 = 16 *3721   
    dim3 dimGrid2(3721, 4);
    dim3 dimBlock2(16, 16);
    Conv2 << <dimGrid2, dimBlock2 >> > (d_conv2_features, kernel[1], d_tmp2, bias[1], 64, 9 * 64, 244 * 244);
    //Conv << <244 * 244, 64 >> > (d_conv2_features, kernel[1], d_tmp2, bias[1], 64, 9 * 64, 244 * 244);
    cudaFree(d_conv1_features);
    cudaFree(d_tmp2);

    //relu
    Relu << <64, 244 >> > (d_conv2_features, 244 * 244 * 64);

    //maxpool, channels = 64,input_shape = 244, output_shape = 122
    //printf("Running MaxPool\n");
    float* d_maxpool1_features;
    cudaMalloc((void**)&d_maxpool1_features, sizeof(float) * (64 * 122 * 122));
    MaxPool2D << <8, 8 >> > (d_maxpool1_features, d_conv2_features, 244, 64);
    cudaFree(d_conv2_features);


    //printf("Running Block2\n");
    //convolution layer 3, in_channels = 64, out_channels = 128, output_shape = 122
    //printf("Running Layer3\n");
    float* d_conv3_features, * d_tmp3;
    cudaMalloc((void**)&d_conv3_features, sizeof(float) * (128 * 122 * 122));
    cudaMalloc((void**)&d_tmp3, sizeof(float) * (64 * 9 * 122 * 122));
    changeform << <1, 1 >> > (d_maxpool1_features, 122, 64, d_tmp3);
    // 128 9*64 14884
    dim3 dimGrid3(931, 8);
    dim3 dimBlock3(16, 16);
    Conv2 << <dimGrid3, dimBlock3 >> > (d_conv3_features, kernel[2], d_tmp3, bias[2], 128, 9 * 64, 122 * 122);
    //Conv << <122 * 122, 128 >> > (d_conv3_features, kernel[2], d_tmp3, bias[2], 128, 9 * 64, 122 * 122);
    cudaFree(d_maxpool1_features);
    cudaFree(d_tmp3);

    //relu
    Relu << <128, 122 >> > (d_conv3_features, 122 * 122 * 128);

    //convolution layer 4, in_channels = 128, out_channels = 128, output_shape = 122
    //printf("Running Layer4\n");
    float* d_conv4_features, * d_tmp4;
    cudaMalloc((void**)&d_conv4_features, sizeof(float) * (128 * 122 * 122));
    cudaMalloc((void**)&d_tmp4, sizeof(float) * (128 * 9 * 122 * 122));
    changeform << <1, 1 >> > (d_conv3_features, 122, 128, d_tmp4);
    dim3 dimGrid4(931, 8);
    dim3 dimBlock4(16, 16);
    Conv2 << <dimGrid4, dimBlock4 >> > (d_conv4_features, kernel[3], d_tmp4, bias[3], 128, 9 * 128, 122 * 122);

    //Conv << <122 * 122, 128 >> > (d_conv4_features, kernel[3], d_tmp4, bias[3], 128, 9 * 128, 122 * 122);
    cudaFree(d_conv3_features);
    cudaFree(d_tmp4);

    //relu
    Relu << <128, 122 >> > (d_conv4_features, 122 * 122 * 128);

    //maxpool2, channels = 128, input_shape = 122, output_shape = 61
    //printf("Running Maxpool\n");
    float* d_maxpool2_features;
    cudaMalloc((void**)&d_maxpool2_features, sizeof(float) * (128 * 61 * 61));
    MaxPool2D << <8, 16 >> > (d_maxpool2_features, d_conv4_features, 122, 128);
    cudaFree(d_conv4_features);


    //printf("Running Block3\n");
    //convolution layer 8, in_channels = 128, out_channels = 256, output_shape = 61
    //printf("Running Layer5\n");
    float* d_conv5_features, * d_tmp5;
    cudaMalloc((void**)&d_conv5_features, sizeof(float) * (256 * 61 * 61));
    cudaMalloc((void**)&d_tmp5, sizeof(float) * (128 * 9 * 61 * 61));
    changeform << <1, 1 >> > (d_maxpool2_features, 61, 128, d_tmp5);
    dim3 dimGrid5(233, 16);
    dim3 dimBlock5(16, 16);
    //Conv << <61*61,256 >> > (d_conv5_features, kernel[4], d_tmp5, bias[4], 256, 9 * 128, 61*61);
    Conv2 << <dimGrid5, dimBlock5 >> > (d_conv5_features, kernel[4], d_tmp5, bias[4], 256, 9 * 128, 61 * 61);

    cudaFree(d_maxpool2_features);
    cudaFree(d_tmp5);

    //relu
    Relu << <256, 61 >> > (d_conv5_features, 61 * 61 * 256);

    //convolution layer 6, in_channels = 256, out_channels = 256, output_shape = 61
    //printf("Running Layer6\n");
    float* d_conv6_features, * d_tmp6;
    cudaMalloc((void**)&d_conv6_features, sizeof(float) * (256 * 61 * 61));
    cudaMalloc((void**)&d_tmp6, sizeof(float) * (256 * 9 * 61 * 61));
    changeform << <1, 1 >> > (d_conv5_features, 61, 256, d_tmp6);
    dim3 dimGrid6(233, 16);
    dim3 dimBlock6(16, 16);
    Conv2 << <dimGrid6, dimBlock6 >> > (d_conv6_features, kernel[5], d_tmp6, bias[5], 256, 9 * 256, 61 * 61);
    //Conv << <61 * 61, 256 >> > (d_conv6_features, kernel[5], d_tmp6, bias[5], 256, 9 * 256, 61 * 61);
    cudaFree(d_conv5_features);
    cudaFree(d_tmp6);

    //relu
    Relu << <256, 61 >> > (d_conv6_features, 61 * 61 * 256);

    //convolution layer 7, in_channels = 256, out_channels = 256, output_shape = 61
    //printf("Running Layer7\n");
    float* d_conv7_features, * d_tmp7;
    cudaMalloc((void**)&d_conv7_features, sizeof(float) * (256 * 61 * 61));
    cudaMalloc((void**)&d_tmp7, sizeof(float) * (256 * 9 * 61 * 61));
    changeform << <1, 1 >> > (d_conv6_features, 61, 256, d_tmp7);
    dim3 dimGrid7(233, 16);
    dim3 dimBlock7(16, 16);
    Conv2 << <dimGrid7, dimBlock7 >> > (d_conv7_features, kernel[6], d_tmp7, bias[6], 256, 9 * 256, 61 * 61);
    //Conv << <61 * 61, 256 >> > (d_conv7_features, kernel[6], d_tmp7, bias[6], 256, 9 * 256, 61 * 61);
    cudaFree(d_conv6_features);
    cudaFree(d_tmp7);

    //relu
    Relu << <256, 61 >> > (d_conv7_features, 61 * 61 * 256);

    //maxpool3,inchannels = 256, outchannels = 256, output_shape = 30
    //printf("Running Maxpool\n");
    float* d_maxpool3_features;
    cudaMalloc((void**)&d_maxpool3_features, sizeof(float) * (256 * 30 * 30));
    MaxPool2D << <16, 16 >> > (d_maxpool3_features, d_conv7_features, 61, 256);
    cudaFree(d_conv7_features);


    //printf("Running Block4\n");
    //convolution layer 8, in_channels = 256, out_channels = 512, output_shape = 30
    //printf("Running Layer8\n");
    float* d_conv8_features, * d_tmp8;
    cudaMalloc((void**)&d_conv8_features, sizeof(float) * (512 * 30 * 30));
    cudaMalloc((void**)&d_tmp8, sizeof(float) * (256 * 9 * 30 * 30));
    changeform << <1, 1 >> > (d_maxpool3_features, 30, 256, d_tmp8);
    dim3 dimGrid8(57, 32);
    dim3 dimBlock8(16, 16);
    Conv2 << <dimGrid8, dimBlock8 >> > (d_conv8_features, kernel[7], d_tmp8, bias[7], 512, 9 * 256, 30 * 30);

    //Conv << <30 * 30, 512 >> > (d_conv8_features, kernel[7], d_tmp8, bias[7], 512, 9 * 256, 30 * 30);
    cudaFree(d_maxpool3_features);
    cudaFree(d_tmp8);

    //relu
    Relu << <512, 30 >> > (d_conv8_features, 30 * 30 * 512);

    //convolution layer 9, in_channels = 512, out_channels = 512, output_shape = 30
    //printf("Running Layer9\n");
    float* d_conv9_features, * d_tmp9;
    cudaMalloc((void**)&d_conv9_features, sizeof(float) * (512 * 30 * 30));
    cudaMalloc((void**)&d_tmp9, sizeof(float) * (512 * 9 * 30 * 30));
    changeform << <1, 1 >> > (d_conv8_features, 30, 512, d_tmp9);
    dim3 dimGrid9(57, 32);
    dim3 dimBlock9(16, 16);
    Conv2 << <dimGrid9, dimBlock9 >> > (d_conv9_features, kernel[8], d_tmp9, bias[8], 512, 9 * 512, 30 * 30);
    //Conv << <30 * 30, 512 >> > (d_conv9_features, kernel[8], d_tmp9, bias[8], 512, 9 * 512, 30 * 30);
    cudaFree(d_conv8_features);
    cudaFree(d_tmp9);


    //relu
    Relu << <512, 30 >> > (d_conv9_features, 30 * 30 * 512);

    //convolution layer 10, in_channels = 512, out_channels = 512, output_shape = 30
    //printf("Running Layer10\n");
    float* d_conv10_features, * d_tmp10;
    cudaMalloc((void**)&d_conv10_features, sizeof(float) * (512 * 30 * 30));
    cudaMalloc((void**)&d_tmp10, sizeof(float) * (512 * 9 * 30 * 30));
    changeform << <1, 1 >> > (d_conv9_features, 30, 512, d_tmp10);
    dim3 dimGrid10(57, 32);
    dim3 dimBlock10(16, 16);
    Conv2 << <dimGrid10, dimBlock10 >> > (d_conv10_features, kernel[9], d_tmp10, bias[9], 512, 9 * 512, 30 * 30);

    //Conv << <30 * 30, 512 >> > (d_conv10_features, kernel[9], d_tmp10, bias[9], 512, 9 * 512, 30 * 30);
    cudaFree(d_conv9_features);
    cudaFree(d_tmp10);


    //relu
    Relu << <512, 30 >> > (d_conv10_features, 30 * 30 * 512);


    //maxpool4,inchannels = 512, outchannels = 512, output_shape = 15
    //printf("Running Maxpool\n");
    float* d_maxpool4_features;
    cudaMalloc((void**)&d_maxpool4_features, sizeof(float) * (512 * 15 * 15));
    MaxPool2D << <32, 16 >> > (d_maxpool4_features, d_conv10_features, 30, 512);
    cudaFree(d_conv10_features);

    //printf("Running Block5\n");
    //convolution layer 11, in_channels = 512, out_channels = 512, output_shape = 15
    //printf("Running Layer11\n");
    float* d_conv11_features, * d_tmp11;
    cudaMalloc((void**)&d_conv11_features, sizeof(float) * (512 * 15 * 15));
    cudaMalloc((void**)&d_tmp11, sizeof(float) * (512 * 9 * 15 * 15));
    changeform << <1, 1 >> > (d_maxpool4_features, 15, 512, d_tmp11);

    dim3 dimGrid11(15, 32);
    dim3 dimBlock11(16, 16);
    Conv2 << <dimGrid11, dimBlock11 >> > (d_conv11_features, kernel[10], d_tmp11, bias[10], 512, 9 * 512, 15 * 15);

    //Conv << <15 * 15, 512 >> > (d_conv11_features, kernel[10], d_tmp11, bias[10], 512, 9 * 512, 15 * 15);
    cudaFree(d_maxpool4_features);
    cudaFree(d_tmp11);

    //relu
    Relu << <512, 15 >> > (d_conv11_features, 15 * 15 * 512);

    //convolution layer 12, in_channels = 512, out_channels = 512, output_shape = 15
    //printf("Running Layer12\n");
    float* d_conv12_features, * d_tmp12;
    cudaMalloc((void**)&d_conv12_features, sizeof(float) * (512 * 15 * 15));
    cudaMalloc((void**)&d_tmp12, sizeof(float) * (512 * 9 * 15 * 15));
    changeform << <1, 1 >> > (d_conv11_features, 15, 512, d_tmp12);
    dim3 dimGrid12(15, 32);
    dim3 dimBlock12(16, 16);
    Conv2 << <dimGrid12, dimBlock12 >> > (d_conv12_features, kernel[11], d_tmp12, bias[11], 512, 9 * 512, 15 * 15);
    //Conv << <15 * 15, 512 >> > (d_conv12_features, kernel[11], d_tmp12, bias[11], 512, 9 * 512, 15 * 15);
    cudaFree(d_conv11_features);
    cudaFree(d_tmp12);

    //relu
    Relu << <512, 15 >> > (d_conv12_features, 15 * 15 * 512);

    //convolution layer 13, in_channels = 512, out_channels = 512, output_shape = 15
    //printf("Running Layer13\n");
    float* d_conv13_features, * d_tmp13;
    cudaMalloc((void**)&d_conv13_features, sizeof(float) * (512 * 15 * 15));
    cudaMalloc((void**)&d_tmp13, sizeof(float) * (512 * 9 * 15 * 15));
    changeform << <1, 1 >> > (d_conv12_features, 15, 512, d_tmp13);
    dim3 dimGrid13(15, 32);
    dim3 dimBlock13(16, 16);
    Conv2 << <dimGrid13, dimBlock13 >> > (d_conv13_features, kernel[12], d_tmp13, bias[12], 512, 9 * 512, 15 * 15);

    //Conv << <15 * 15, 512 >> > (d_conv13_features, kernel[12], d_tmp13, bias[12], 512, 9 * 512, 15 * 15);
    cudaFree(d_conv12_features);
    cudaFree(d_tmp13);

    //relu
    Relu << <512, 15 >> > (d_conv13_features, 15 * 15 * 512);

    //maxpool5,inchannels = 512, outchannels = 512, output_shape = 7
    //printf("Running Maxpool\n");
    float* d_maxpool5_features;
    cudaMalloc((void**)&d_maxpool5_features, sizeof(float) * (512 * 7 * 7));
    MaxPool2D << <32, 16 >> > (d_maxpool5_features, d_conv13_features, 15, 512);
    cudaFree(d_conv13_features);



    //printf("Running fc\n");
    //fc1 input_channels = 512 * 7 * 7, output_channels = 4096, kernel_shape = 4096 * (512 * 7 * 7)
    float* d_fc1_features;
    cudaMalloc((void**)&d_fc1_features, sizeof(float) * (4096));
    //Gemm<<<16,256>>>(d_maxpool5_features, d_fc1_features, kernel[13], bias[13], 25088, 4096);
    gemm(kernel[13], d_maxpool5_features, d_fc1_features, bias[13], 4096, 1, 25088);
    cudaFree(d_maxpool5_features);

    //relu  4096
    Relu << <64, 64 >> > (d_fc1_features, 4096);

    //fc2 input_channels = 4096, output_channels = 4096, kernel_shape = 4096 * 4096
    float* d_fc2_features;
    cudaMalloc((void**)&d_fc2_features, sizeof(float) * (4096));
    gemm(kernel[14], d_fc1_features, d_fc2_features, bias[14], 4096, 1, 4096);
    cudaFree(d_fc1_features);

    //relu  4096
    Relu << <64, 64 >> > (d_fc2_features, 4096);

    //fc3 input_channels = 4096, output_channels = 1000, kernel_shape = 1000 * 4096
    //float* d_output;
    //cudaMalloc((void**)&d_output, sizeof(float) * (1000));
    gemm(kernel[15], d_fc2_features, d_output, bias[15], 1000, 1, 4096);
    cudaFree(d_fc2_features);


}

void readInput(const char* filename)
{
    FILE* fp = NULL;
    fp = fopen(filename, "r");
    for (int i = 0; i < TESTNUM; i++)
        for (int j = 0; j < INPUTSHAPE; j++)
            fscanf(fp, "%f", &inputArr[i][j]);
}

void readOutput(const char* filename)
{
    FILE* fp = NULL;
    fp = fopen(filename, "r");
    for (int i = 0; i < TESTNUM; i++)
        for (int j = 0; j < OUTPUTSHAPE; j++)
            fscanf(fp, "%f", &benchOutArr[i][j]);
}

void checkOutput(float* out1, float* out2)
{
    float maxDiff = 0;
    for (int i = 0; i < OUTPUTSHAPE; i++)
    {
        maxDiff = (fabs(out1[i] - out2[i]) > maxDiff) ? fabs(out1[i] - out2[i]) : maxDiff;
        //printf("%f  %f\n", out1[i],out2[i]);
    }
    if (maxDiff > 1e-5)
    {
        printf("Output dismatch. MaxDiff is %.7f\n", maxDiff);
        //exit(-1);
    }
    printf("Output correct. MaxDiff is %.7f\n", maxDiff);
}


int main()
{

    readInput("./vgg16Input.txt");   // 读取输入
    readOutput("./vgg16Output.txt"); // 读取标准输出

    float* model_weights[16];
    float* model_bias[16];
    printf("initializing model\n");
    initModel(model_weights, model_bias); // 读取网络权重

    float sumTime = 0;
    for (int i = 0; i < TESTNUM; i++)
    {
        float inferOut[1000];
        float* d_input;
        float* d_output;
        cudaMalloc((void**)&d_input, sizeof(float) * (1 * INPUTSHAPE));
        cudaMalloc((void**)&d_output, sizeof(float) * (1000));
        cudaMemcpy(d_input, inputArr[i], sizeof(float) * (1 * INPUTSHAPE), cudaMemcpyHostToDevice);
        for (int j = 0; j < ITERNUM; j++)
        {
            printf("Running TESTNUM:%d and ITERNUM:%d \n", i, j);
            float Onetime;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);

            // 执行Inference

            inference(d_input, d_output, model_weights, model_bias);

            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&Onetime, start, stop);
            // 累加单次推理消耗时间
            sumTime += Onetime;
        }
        cudaMemcpy(inferOut, d_output, sizeof(float) * (1000), cudaMemcpyDeviceToHost);
        checkOutput(benchOutArr[i], inferOut);
    }
    printf("Average Time is: %f\n", (sumTime / TESTNUM / ITERNUM));
}