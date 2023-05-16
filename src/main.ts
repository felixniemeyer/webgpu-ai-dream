import './style.css'

window.addEventListener('load', async () => {
  function fail(msg: string): never {
    const error = document.getElementById('error')
    error!.innerText = msg
    throw Error(msg);
  }

  if (!navigator.gpu) {
    fail('WebGPU is not supported')
  }

  const adapter = await navigator.gpu!.requestAdapter();
  if (!adapter) {
    fail("Couldn't request WebGPU adapter.");
  }

  const device = await adapter.requestDevice();
  if (!device) {
    fail("failed to get WebGPU device") 
  }

  console.log(device);

  const shaders = `
struct VertexOut {
  @builtin(position) position: vec4f, 
  @location(0) originalPosition: vec4f
}

@vertex
fn vertex_main(@location(0) position: vec4f) -> VertexOut
{
  var output: VertexOut;
  output.position = position;
  output.originalPosition = position;
  return output;
}

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4f 
{
  return vec4f(1.0, fragData.originalPosition[0] * 0.5 + 0.5, 0.0, 1.0);
}`

  const shaderModule = device.createShaderModule({
    code: shaders   
  });

  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  const context = canvas.getContext('webgpu');

  if(!context) {
    fail('WebGPU is supported, but something went wrong when initializing the context.')
  }

  const swapChainFormat = navigator.gpu.getPreferredCanvasFormat(); // 'bgra8unorm';
  console.log('swap chain format', swapChainFormat);
  
  context.configure({
    device: device,
    format: swapChainFormat,
    alphaMode: "premultiplied"
  });

    
  // screen quad
  const vertices = new Float32Array([
    -1, -1, 0, 1,
    -1,  1, 0, 1,
    1, -1, 0, 1,
    1,  1, 0, 1,
  ]);

  const vertexBuffer = device.createBuffer({
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
  });

  device.queue.writeBuffer(vertexBuffer, 0, vertices, 0, vertices.length) 

  const vertexBuffers: GPUVertexBufferLayout[] = [
    {
      attributes: [
        {
          shaderLocation: 0, 
          offset: 0, 
          format: "float32x4"
        }
      ], 
      arrayStride: 16, 
      stepMode: "vertex"
    }
  ]


  const renderPipelineDescriptor: GPURenderPipelineDescriptor = {
    vertex: {
      module: shaderModule,
      entryPoint: 'vertex_main', 
      buffers: vertexBuffers
    },
    fragment: {
      module: shaderModule,
      entryPoint: 'fragment_main', 
      targets: [{
        format: swapChainFormat
      }]
    },
    primitive: {
      topology: 'triangle-strip',
    }, 
    layout: "auto"
  };

  const renderPipeline = device.createRenderPipeline(renderPipelineDescriptor);

  const render = () => {
    const commandEncoder = device.createCommandEncoder();

    const textureView = context.getCurrentTexture().createView();

    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [{
        view: textureView,
        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }, 
        storeOp: 'store', 
        loadOp: 'clear'
      }]
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(renderPipeline);
    passEncoder.setVertexBuffer(0, vertexBuffer);
    passEncoder.draw(4, 1, 0, 0);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
  }

  render()
})

