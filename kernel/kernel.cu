// Very minimal skeleton for the kernel

#include <stdio.h>

__device__ void convolution_layer(double input_data[100][100], double filters[10][5][5], double conv_output[10][20][20]) {
    int layer_idx = blockIdx.x;
    int section_x = threadIdx.x;
    int section_y = threadIdx.y;

    // Compute dot product for the 5x5 region
    double dp = 0;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            dp += filters[layer_idx][i][j] * input_data[section_x*5 + i][section_y*5 + j];
            //printf("%.2f * %.2f = %.2f\n", filters[layer_idx][i][j], input_data[section_x*5 + i][section_y*5 + j], dp);
        }
    }

    // Update output matrix
    conv_output[layer_idx][section_x][section_y] = dp;
    
    // printf("Layer: %d, section_x: %d, section_y: %d, %.2f\n", layer_idx, section_x, section_y, dp);
}

__device__ void relu_layer(double conv_output[10][20][20]) {
    int layer_idx = blockIdx.x;
    int section_x = threadIdx.x;
    int section_y = threadIdx.y;
    if (conv_output[layer_idx][section_x][section_y] < 0.0) {
        conv_output[layer_idx][section_x][section_y] = 0.0;
    }
    
    // printf("Layer: %d, section_x: %d, section_y: %d, %.2f\n", layer_idx, section_x, section_y, conv_output[layer_idx][section_x][section_y]);
}

__device__ void output_layer_multiply(double conv_output[10][20][20], double weights[10][4000], double output_layer_temp_1[10][4000]) {
    int layer_idx = blockIdx.x;
    int section_x = threadIdx.x;
    int section_y = threadIdx.y;
    for (int output_layer_idx = 0; output_layer_idx < 10; output_layer_idx++) {
        output_layer_temp_1[output_layer_idx][section_y + section_x*20 + layer_idx*400] = conv_output[layer_idx][section_x][section_y] * weights[output_layer_idx][section_y + section_x*20 + layer_idx*400];
    }
}

__device__ void output_layer_add_1(double output_layer_temp_1[10][4000], double output_layer_temp_2[10][200]) {
    int layer_idx = blockIdx.x;     // 0 - 9
    int section_x = threadIdx.x;    // 0 - 19
    int section_y = threadIdx.y;    // 0 - 19

    // A single block will be responsible for a single output, and then we will split up the summation into 200
    if (section_y )

    output_layer_temp_2[layer_idx] = 
}

extern "C" __global__ void compute(
        double input_data[100][100],
        double filters[10][5][5],
        double weights[10][4000],
        double output_data[10]
        ) {

    // int layer_idx = blockIdx.x;
    // int section_x = threadIdx.x;
    // int section_y = threadIdx.y;
    // printf("%d %d %d %.2f\n", layer_idx, section_x, section_y, input_data[0][0]);

    double conv_output[10][20][20];
    convolution_layer(input_data, filters, conv_output);
    relu_layer(conv_output);
    // Temporary matrices for parallelize summing portion of final dot product (divide and conquer strategy)
    double output_layer_temp_1[10][4000];
    double output_layer_temp_2[10][200];
    output_layer_multiply(conv_output, weights, output_layer_temp_1);
    output_layer_add_1(output_layer_temp_1, output_layer_temp_2);
    output_layer_add_2(output_data);
}