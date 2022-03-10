// This is the skeleton for the CUDA implementation

use std::convert::TryFrom;
use crate::cnn::*;
use rustacuda::function::BlockSize;
use rustacuda::launch;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;

// Fields need to be ordered this way so the DeviceBoxes are
// dropped before the Context. Otherwise the drop will panic.

pub struct CudaContext {
    conv_layer: DeviceBox<ConvLayer>,
    output_layer: DeviceBox<OutputLayer>,
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaContext {
    pub fn init(cnn: &Cnn) -> Result<Self, Box<dyn Error>> {
        rustacuda::init(CudaFlags::empty())?;
        let device = Device::get_device(0)?;
        let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
        let ptx = CString::new(include_str!("../kernel/kernel.ptx"))?;

        let cuda_context = CudaContext {
            conv_layer: DeviceBox::new(&cnn.conv_layer)?,
            output_layer: DeviceBox::new(&cnn.output_layer)?,
            module: Module::load_from_string(&ptx)?,
            stream: Stream::new(StreamFlags::DEFAULT, None)?,
            _context: _ctx
        };

        Ok(cuda_context)
    }

    pub fn compute(&mut self, input: &InputMatrix) -> Result<OutputVec, Box<dyn Error>> {
        let mut input_box = DeviceBox::new(input)?;
        let mut conv_output = DeviceBox::new(&ConvOutput([[[0.0f64; CONV_OUT_DIM]; CONV_OUT_DIM]; CONV_LAYER_SIZE]))?;
        let mut output_vec: Vec<f64> = vec![];
        let mut single_output = 0.0f64;
        let mut single_output_box = DeviceBox::new(&single_output)?;
        let module = &self.module;
        let stream = &self.stream;

        // Lets do grid size of 10 blocks, with 20x20 threads in each block
        let num_blocks = 10;
        let threads_per_block = BlockSize::xy(20, 20);

        unsafe {
            // Convolution Layer.
            let result = launch!(module.convolution_layer<<<num_blocks, threads_per_block.clone(), 0, stream>>>(
                input_box.as_device_ptr(),
                self.conv_layer.as_device_ptr(),
                conv_output.as_device_ptr()
            ));
            result?;

            // Relu Layer
            let result = launch!(module.relu_layer<<<num_blocks, threads_per_block.clone(), 0, stream>>>(conv_output.as_device_ptr()));
            result?;

            // Output Layer. One kernel call per output number
            for output_idx in 0..10 {
                let result = launch!(module.output_layer_for_single_output<<<num_blocks, threads_per_block.clone(), 0, stream>>>(
                    conv_output.as_device_ptr(),
                    self.output_layer.as_device_ptr(),
                    DeviceBox::new(&output_idx)?.as_device_ptr(),
                    single_output_box.as_device_ptr()
                ));
                result?;
                single_output_box.copy_to(&mut single_output)?;
                output_vec.push(single_output);
            }
        }

        stream.synchronize()?;

        Ok(OutputVec(<[f64; 10]>::try_from(output_vec).unwrap()))
    }
}
