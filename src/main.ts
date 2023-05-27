const brainSize = 64
const workgroupSize = 16
const workgroups = brainSize / workgroupSize

const keySize = 4; 

const brainTickShaderSrc = `
// texture in and out
@binding(0) @group(0) var current: texture_2d<f32>;
@binding(1) @group(0) var next: texture_storage_2d<r32float, write>;
@binding(2) @group(0) var weights: texture_2d<u32>;

const size = ${brainSize}; 
const workgroupSize = ${workgroupSize};

fn decomposeWeights(value: u32) -> vec4<f32> {


  var decomposed = vec4<u32>(
    (value & 0xFFu),
    (value >> 8u) & 0xFFu,
    (value >> 16u) & 0xFFu,
    (value >> 24u) & 0xFFu
  ); 

  return vec4<f32>(decomposed) / 128.0 - 1.0; 
}

@compute @workgroup_size(${workgroupSize}, ${workgroupSize}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  var weights = textureLoad(weights, gid.xy, 0);
  var sum = 0.;
  for(var x = 0; x < 4; x++) {
    var w = decomposeWeights(weights[x]);
    var u = i32(gid.x) + x - 1; 
    var v = i32(gid.y) - 1;

    var square = vec4<f32>(
      textureLoad(current, vec2<i32>(u, v - 1), 1).r,
      textureLoad(current, vec2(u, v), 1).r,
      textureLoad(current, vec2(u, v + 1), 1).r,
      textureLoad(current, vec2(u, v + 2), 1).r
    ); 
    sum += dot(square, w); 
  }
  // activation function tanh
  var activation = tanh(sum);

  var out = vec4<f32>(activation, 0, 0, 1);
  textureStore(next, gid.xy, out);
}
`

const drawBrainStateShaderSrc = `
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

@binding(0) @group(0) var currentBrainState: texture_2d<f32>;
@binding(1) @group(0) var previousBrainState: texture_2d<f32>;
@binding(2) @group(0) var brainStateSampler: sampler;

const size = ${brainSize};
const keySize = ${keySize};
const sideLength = size + 2 * keySize;

const pixelSize = 1.0 / f32(sideLength);

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4f 
{
  var uv = fragData.originalPosition.xy * 0.5 + 0.5;
  var value = textureLoad(currentBrainState, vec2<i32>(uv * size), 0).r;
  return vec4f(value, 0.5, 0.5, 1.0); 
}`


window.addEventListener('load', async () => {
  const bpm = 240

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

  console.log('WebGPU device:', device);

  // get canvas
  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  const canvasSize = Math.min(canvas.clientWidth, canvas.clientHeight);
  canvas.width = canvasSize
  canvas.height = canvasSize

  const context = canvas.getContext('webgpu');
  if(!context) {
    fail('WebGPU is supported, but something went wrong when initializing the context.')
  }

  const swapChainFormat = navigator.gpu.getPreferredCanvasFormat(); // 'bgra8unorm';
  console.log('Swap chain format:', swapChainFormat);
  
  context.configure({
    device: device,
    format: swapChainFormat,
    alphaMode: "premultiplied"
  });

  // initialize brain tick compute pipeline

  const brainTickShaderModule = device.createShaderModule({
    code: brainTickShaderSrc, 
  })

  const brainTickBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { // current activation state
        binding: 0, 
        visibility: GPUShaderStage.COMPUTE,
        texture: {
          sampleType: 'unfilterable-float',
          viewDimension: '2d',
          multisampled: false,
        }
      },
      { // next activation state
        binding: 1, 
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: {
          access: 'write-only',
          format: 'r32float',
          viewDimension: '2d',
        }
      },
      { // weights
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        texture: {
          sampleType: 'uint',
          viewDimension: '2d',
          multisampled: false,
        }
      }, 
    ]
  })


  // we need 3 activation state textures: previous, current, next

  const brainStates: GPUTexture[] = []
  const initialBrainStateBuffer = device.createBuffer({
    size: brainSize * brainSize * 4,
    usage: GPUBufferUsage.COPY_SRC, 
    mappedAtCreation: true,
  })
  const initialBrainState = new Float32Array(initialBrainStateBuffer.getMappedRange())
  for(let i = 0; i < brainSize * brainSize; i++) {
    initialBrainState[i] = Math.random() * 2.0 - 1.0
  }
  initialBrainStateBuffer.unmap()

  for(let i = 0; i < 3; i++) {
    brainStates[i] = device.createTexture({
      size: { width: brainSize, height: brainSize },
      format: 'r32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING
    }) 
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToTexture({
        buffer: initialBrainStateBuffer,
        bytesPerRow: brainSize * 4,
      }, { 
        texture: brainStates[i],
      },
      [brainSize, brainSize, 1]
    )
    device.queue.submit([commandEncoder.finish()])
  }

  const drawBrainStateShaderModule = device.createShaderModule({
    code: drawBrainStateShaderSrc   
  });

  // we need one weight texture, populated randomly
  const weights = device.createTexture({
    size: { width: brainSize, height: brainSize },
    format: 'rgba32uint',
    usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING
  }) 

  const stagingBuffer = device.createBuffer({
    size: brainSize * brainSize * 4 * 4,
    usage: GPUBufferUsage.COPY_SRC, 
    mappedAtCreation: true
  })
  const data = new Uint8Array(stagingBuffer.getMappedRange())
  for(let i = 0; i < brainSize * brainSize * 4 * 4; i ++) {
    data[i] = Math.trunc(Math.random() * 256)
  }
  stagingBuffer.unmap()

  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToTexture({
      buffer: stagingBuffer,
      bytesPerRow: brainSize * 4 * 4,
    }, { 
      texture: weights,
    },
    [brainSize, brainSize, 1]
  )
  device.queue.submit([commandEncoder.finish()])

  const brainTickBindGroups: GPUBindGroup[] = []

  // one bind group for each brainState texture combination
  // bind group 0: current = 0, next = 1 ...
  for(let i = 0; i < 3; i++) {
    brainTickBindGroups[i] = device.createBindGroup({
      layout: brainTickBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: brainStates[i].createView(), 
        },
        {
          binding: 1,
          resource: brainStates[(i + 1) % 3].createView(),
        },
        {
          binding: 2,
          resource: weights.createView(),
        }
      ]
    })
  }

  // create compute pipeline 
  const brainTickComputePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [brainTickBindGroupLayout]
    }), 
    compute: {
      entryPoint: 'main', 
      module: brainTickShaderModule
    }
  })


  // initialize brain state draw pipeline

  // create bind group
  const drawBrainStateBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { 
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {
          sampleType: 'unfilterable-float',
          viewDimension: '2d',
          multisampled: false,
        }
      }, 
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {
          sampleType: 'unfilterable-float',
          viewDimension: '2d',
          multisampled: false,
        }
      }
    ]
  })

  const dataTextureSampler = device.createSampler({
    addressModeU: 'clamp-to-edge', 
    addressModeV: 'clamp-to-edge', 
    magFilter: 'nearest', // TBD: could be errorneous
    minFilter: 'nearest', 
    mipmapFilter: 'nearest'
  })

  const drawBrainStateBindGroups: GPUBindGroup[] = []
  for(let i = 0; i < 3; i++) {
    drawBrainStateBindGroups[i] = device.createBindGroup({
      layout: drawBrainStateBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: brainStates[i].createView(),
        },
        {
          binding: 1,
          resource: brainStates[(i + 2) % 3].createView(),
        }
      ]
    })
  }

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

  const drawBrainStatePipelineDescriptor: GPURenderPipelineDescriptor = {
    vertex: {
      module: drawBrainStateShaderModule,
      entryPoint: 'vertex_main', 
      buffers: vertexBuffers
    },
    fragment: {
      module: drawBrainStateShaderModule,
      entryPoint: 'fragment_main', 
      targets: [{
        format: swapChainFormat
      }]
    },
    primitive: {
      topology: 'triangle-strip',
    }, 
    layout: device.createPipelineLayout({
      bindGroupLayouts: [drawBrainStateBindGroupLayout],
    }),
  };

  const renderPipeline = device.createRenderPipeline(drawBrainStatePipelineDescriptor);

  let currentBrainState = 0

  const brainTick = () => {
    // tick brain
    const brainTickCommandEncoder = device.createCommandEncoder();
    const brainTickPassEncoder = brainTickCommandEncoder.beginComputePass();
    brainTickPassEncoder.setPipeline(brainTickComputePipeline);
    brainTickPassEncoder.setBindGroup(0, brainTickBindGroups[currentBrainState]);
    brainTickPassEncoder.dispatchWorkgroups(workgroups, workgroups, 1); 
    brainTickPassEncoder.end();
    device.queue.submit([brainTickCommandEncoder.finish()]);
    console.log('brain tick')
    currentBrainState = (currentBrainState + 1) % 3
    // good luck!
  }

  setInterval(brainTick, 1000 * 60 / bpm)

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
    passEncoder.setBindGroup(0, drawBrainStateBindGroups[currentBrainState]);
    passEncoder.setVertexBuffer(0, vertexBuffer);
    passEncoder.draw(4, 1, 0, 0);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(render); // TBD stack grows because of fat arrow scope maintainance
  }

  render()
})

