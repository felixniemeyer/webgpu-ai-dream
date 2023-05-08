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

  console.log(device);
})


