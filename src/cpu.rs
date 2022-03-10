// CPU implementation of the CNN. You must implement the CUDA version of this code.

// You should not need to modify this file.

use crate::cnn::*;

fn convolution_layer(input: &InputMatrix, conv_filters: &ConvLayer, outputs: &mut ConvOutput) {
    // Go through each convolution neuron
    for (filter, out) in conv_filters.0.iter().zip(outputs.0.iter_mut()) {
        // Divide the 100x100 input matrix into 5x5 regions. There are 20x20 such regions in the
        // matrix. Each convolution neuron does a dot product of its filter with each region, producing a
        // 20x20 output matrix of products
        for i in (0..INPUT_DIM).step_by(FILTER_DIM) {
            for j in (0..INPUT_DIM).step_by(FILTER_DIM) {
                // Dot product
                let prod: f64 = (0..FILTER_DIM)
                    .flat_map(move |x| {
                        (0..FILTER_DIM).map(move |y| input.0[i + x][j + y] * filter[x][y])
                    })
                    .sum();
                out[i / FILTER_DIM][j / FILTER_DIM] = prod;
            }
        }
    }
}

fn relu_layer(conv_out: &mut ConvOutput) {
    // Any value below 0 in the previous layer's output is changed to 0
    for matrix in conv_out.0.iter_mut() {
        for row in matrix {
            for val in row {
                if *val < 0.0 {
                    *val = 0.0;
                }
            }
        }
    }
}

fn output_layer(input: &ConvOutput, weights: &OutputLayer, output: &mut OutputVec) {
    // Go thru each output neuron
    for (weight, out) in weights.0.iter().zip(output.0.iter_mut()) {
        // Flatten the output of the previous layer into a 4000x1 vector, then dot product it with
        // the weight vector to produce a single value
        let flattened = input.0.iter().flat_map(|n| n.iter().flat_map(|r| r.iter()));
        let multiplied = flattened.zip(weight.iter()).map(|(a, b)| a * b);
        //let temp: Vec<f64> = multiplied.clone().into_iter().collect();
        // let small = Vec![];
        // for i in 0..
        // println!("{}", temp[2580]);
        // println!("{}", temp[2581]);
        // println!("{}", temp[2582]);
        // println!("{}", temp[2583]);
        // println!("{}", temp[2584]);
        // println!("{}", temp[2585]);
        // println!("{}", temp[2586]);
        // println!("{}", temp[2587]);
        // println!("{}", temp[2588]);
        // println!("{}", temp[2589]);
        // println!("{}", temp[2590]);
        // println!("{}", temp[2591]);
        // println!("{}", temp[2592]);
        // println!("{}", temp[2593]);
        // println!("{}", temp[2594]);
        // println!("{}", temp[2595]);
        // println!("{}", temp[2596]);
        // println!("{}", temp[2597]);
        // println!("{}", temp[2598]);
        // println!("{}", temp[2599]);
        // for value in multiplied.clone().into_iter() {
        //     println!("{}", value);
        // }
        let prod: f64 = multiplied.sum();
        // println!("{}", prod);
        *out = prod;
    }
}

pub fn compute(input: &InputMatrix, cnn: &Cnn) -> OutputVec {
    let mut conv_output = ConvOutput([[[0.0; CONV_OUT_DIM]; CONV_OUT_DIM]; CONV_LAYER_SIZE]);
    let mut output = OutputVec([0.0; OUT_LAYER_SIZE]);
    convolution_layer(&input, &cnn.conv_layer, &mut conv_output);
    relu_layer(&mut conv_output);
    output_layer(&conv_output, &cnn.output_layer, &mut output);
    output
}

