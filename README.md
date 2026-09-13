# Edge AI Quantization Benchmark

A small benchmark for comparing a floating-point PyTorch model with a dynamically
quantized CPU version. It reports serialized size and average inference latency, making
the deployment tradeoff visible before moving to ONNX Runtime or TensorRT.

## What it demonstrates

- Dynamic INT8 quantization for linear layers
- Model-size measurement from serialized state
- Warm-up and repeated latency timing
- A reproducible CPU-only benchmark

## Run

~~~bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m src.benchmark
~~~

Numbers depend on hardware and should be treated as local engineering measurements.
For a real deployment, compare accuracy on a held-out set and validate the target edge
runtime.

## License

Apache-2.0
