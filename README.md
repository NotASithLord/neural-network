# Neural Network

A working, rotatable seven-layer neural network. The WebGPU path trains 2 → 256 → 256 → 256 → 256 → 256 → 1: 1,283 neurons and 264,193 trainable parameters. Every hidden neuron is displayed with its live activation; the connecting lines are a sparse selection of real connections. The model learns a circular decision boundary from generated samples. Swift/WASM owns initialization, evaluation, and the complete CPU fallback (64 neurons per hidden layer). WGSL executes GPU forward passes and backpropagation; JavaScript connects browser APIs and renders the view.

## Run

Serve this directory over localhost or HTTPS and open index.html. The compiled core.wasm is included. Click Run simulation, drag to rotate, scroll to zoom, or double-click to reset the camera. Execution pauses when the page is hidden.

## Build and test

Use the official Swift 6.4 toolchain and matching WebAssembly SDK.

```sh
export SWIFT_BIN=/path/to/swift-toolchain/usr/bin
export WASI_SYSROOT=/path/to/swift-wasm-sdk/wasm32-unknown-wasip1/WASI.sdk
node build.mjs
node test.mjs
```

Tests verify loss reduction, validation accuracy, learning in all six trainable stages, finite initialization, and adaptive submission budgeting.

Bundled dependency licenses are in THIRD_PARTY_NOTICES.md.
