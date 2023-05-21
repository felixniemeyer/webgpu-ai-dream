const brainSize = 64
const keySize = 4

const brainTickShaderSrc = `
// texture in and out
@binding(0) @group(0) var current: texture_2d<f32>;
@binding(1) @group(0) var next: texture_storage_2d<r32float, write>;
@binding(2) @group(0) var weights: texture_2d<f32>;

const size = ${brainSize}; 

@compute @workgroup_size(size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  var square = vec4<f32>(
    textureLoad(current, gid.xy, 1).r,
    textureLoad(current, gid.xy + vec2<u32>(1, 0), 1).r,
    textureLoad(current, gid.xy + vec2<u32>(0, 1), 1).r,
    textureLoad(current, gid.xy + vec2<u32>(1, 1), 1).r
  ); 
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
  if(fragData.originalPosition.x > 0.) {
    var rgb: vec3f = fragData.originalPosition.xyz * 0.5 + 0.5;
    var l = length(fragData.originalPosition.xyz); 
    var w = smoothstep(0.765, 0.77, l); 
    rgb = (1.0 - w) * vec3f(1.0, 1.0, 1.0) + w * rgb; 
    return vec4f(rgb, 1.0);
  } else {
    return vec4f(0.5, 0.5, 0.5, 1.0); 
  }
}`


window.addEventListener('load', async () => {
  const bpm = 120

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
  canvas.width = brainSize + keySize;
  canvas.height = brainSize + keySize;

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
          sampleType: 'unfilterable-float',
          viewDimension: '2d',
          multisampled: false,
        }
      }, 
    ]
  })


  // we need 3 activation state textures: previous, current, next

  const brainStates: GPUTexture[] = []
  for(let i = 0; i < 3; i++) {
    brainStates[i] = device.createTexture({
      size: { width: brainSize, height: brainSize },
      format: 'r32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING
    }) 
  }

  const drawBrainStateShaderModule = device.createShaderModule({
    code: drawBrainStateShaderSrc   
  });

  // we need one weight texture, populated randomly
  const weights = device.createTexture({
    size: { width: brainSize, height: brainSize },
    format: 'rgba32float',
    usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING
  }) 

  const stagingBuffer = device.createBuffer({
    size: brainSize * brainSize * 4 * 4,
    usage: GPUBufferUsage.COPY_SRC, 
    mappedAtCreation: true
  })
  const data = new Float32Array(stagingBuffer.getMappedRange())
  for(let i = 0; i < brainSize * brainSize * 4; i ++) {
    data[i] = Math.random() * 2 - 1
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
      },
      {
        binding: 2, 
        visibility: GPUShaderStage.FRAGMENT, 
        sampler: {
          type: 'non-filtering'
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
        },
        {
          binding: 2,
          resource: dataTextureSampler,
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
    layout: "auto"
  };

  const renderPipeline = device.createRenderPipeline(drawBrainStatePipelineDescriptor);

  let currentBrainState = 0

  const brainTick = () => {
    // tick brain
    const brainTickCommandEncoder = device.createCommandEncoder();
    const brainTickPassEncoder = brainTickCommandEncoder.beginComputePass();
    brainTickPassEncoder.setPipeline(brainTickComputePipeline);
    brainTickPassEncoder.setBindGroup(0, brainTickBindGroups[currentBrainState]);
    brainTickPassEncoder.dispatchWorkgroups(brainSize); 
    brainTickPassEncoder.end();
    console.log('brain tick')
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

