// Very minimal skeleton for the kernel

#include <stdio.h>

__device__ void convolution_layer(double input_data[100][100], double filters[10][5][5], double conv_output[10][20][20]) {
    int layer_idx = blockIdx.x;
    int section_x = threadIdx.x;
    int section_y = threadIdx.y;

    // Compute dot product for the 5x5 region
    int dp = 0;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            dp += filters[layer_idx][i][j] * input_data[section_x*5 + i][section_y*5 + j];
        }
    }

    // Update output matrix
    conv_output[layer_idx][section_x][section_y] = dp;
}

__device__ void relu_layer(double conv_output[10][20][20]) {
    int layer_idx = blockIdx.x;
    int section_x = threadIdx.x;
    int section_y = threadIdx.y;
    if (conv_output[layer_idx][section_x][section_y] < 0.0) {
        conv_output[layer_idx][section_x][section_y] = 0.0;
    }
}

__device__ void output_layer(double conv_output[10][20][20], double weights[10][4000], double output_data[10]) {
    int layer_idx = blockIdx.x;
    int section_x = threadIdx.x;
    int section_y = threadIdx.y;
    for (int output_layer_idx = 0; output_layer_idx < 10; output_layer_idx++) {
        output_data[output_layer_idx] += conv_output[layer_idx][section_x][section_y] * weights[output_layer_idx][section_x + section_y*20 + layer_idx*400];
    }
}

extern "C" __global__ void compute(
        double input_data[100][100],
        double filters[10][5][5],
        double weights[10][4000],
        double output_data[10]
        ) {
    double conv_output[10][20][20];
    convolution_layer(input_data, filters, conv_output);
    relu_layer(conv_output);
    output_layer(conv_output, weights, output_data);
}