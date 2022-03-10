// Very minimal skeleton for the kernel

#include <stdio.h>

extern "C" __global__ void convolution_layer(double input_data[100][100], double filters[10][5][5], double conv_output[10][20][20]) {
    int layer_idx = blockIdx.x;     // 0 - 9
    int section_x = threadIdx.x;    // 0 - 19
    int section_y = threadIdx.y;    // 0 - 19

    // Compute dot product for the 5x5 region
    double dp = 0;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            dp += filters[layer_idx][i][j] * input_data[section_x*5 + i][section_y*5 + j];
        }
    }

    // Update output matrix
    conv_output[layer_idx][section_x][section_y] = dp;
}

extern "C" __global__ void relu_layer(double conv_output[10][20][20]) {
    int layer_idx = blockIdx.x;     // 0 - 9
    int section_x = threadIdx.x;    // 0 - 19
    int section_y = threadIdx.y;    // 0 - 19
    if (conv_output[layer_idx][section_x][section_y] < 0.0) {
        conv_output[layer_idx][section_x][section_y] = 0.0;
    }
}

extern "C" __global__ void output_layer_multiplication_for_single_output(double conv_output[10][20][20], double weights[10][4000], int output_number, double output_layer_temp_1[4000]) {
    output_layer_temp_1[blockIdx.x*400 + threadIdx.x*20 + threadIdx.y] = conv_output[blockIdx.x][threadIdx.x][threadIdx.y] * weights[output_number][blockIdx.x*400 + threadIdx.x*20 + threadIdx.y];
}

extern "C" __global__ void output_layer_add_1(double output_layer_temp_1[4000], double output_layer_temp_2[200]) {
    // 8 blocks, 25 threads per block
    double sum = 0;
    for (int i = 0; i < 20; i++) {
        sum += output_layer_temp_1[blockIdx.x*500 + threadIdx.x*20 + i];
    }
    output_layer_temp_2[blockIdx.x*25 + threadIdx.x] = sum;
}

extern "C" __global__ void output_layer_add_2(double output_layer_temp_2[200], double output_layer_temp_3[10]) {
    // 1 blocks, 10 threads per block
    double sum = 0;
    for (int i = 0; i < 20; i++) {
        sum += output_layer_temp_2[threadIdx.x*20 + i];
    }
    output_layer_temp_3[threadIdx.x] = sum;
}

extern "C" __global__ void output_layer_add_3(double output_layer_temp_3[10], double *output) {
    // 1 blocks, 1 threads per block
    double sum = 0;
    for (int i = 0; i < 10; i++) {
        sum += output_layer_temp_3[i];
    }
    *output = sum;
}
