// This is the skeleton for the CUDA implementation

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
            stream: Stream::new(StreamFlags::NON_BLOCKING, None)?,
            _context: _ctx
        };

        Ok(cuda_context)
    }

    pub fn compute(&mut self, input: &InputMatrix) -> Result<OutputVec, Box<dyn Error>> {
        // Lets do grid size of 10 blocks, with 20x20 threads in each block
        let num_blocks = 10;
        let threads_per_block = BlockSize::xy(20, 20);

        let mut input_box = DeviceBox::new(input)?;
        let mut output = OutputVec([0.0f64; OUT_LAYER_SIZE]);
        let mut output_box = DeviceBox::new(&output)?;
        let module = &self.module;
        let stream = &self.stream;
        unsafe {
            let result = launch!(module.compute<<<num_blocks, threads_per_block, 0, stream>>>(
                input_box.as_device_ptr(),
                self.conv_layer.as_device_ptr(),
                self.output_layer.as_device_ptr(),
                output_box.as_device_ptr()
            ));
            result?;
        }
        stream.synchronize()?;
        output_box.copy_to(&mut output)?;
        Ok(output)
    }
}
