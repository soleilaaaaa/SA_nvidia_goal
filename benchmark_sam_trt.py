# Hands-On Lab: Segment Anything (SAM ViT-B) Inference Optimization with NVIDIA TensorRT 10

## Context & Motivation

As an industrial/generalist engineer exploring the NVIDIA software stack, I set up this practical lab to evaluate inference optimization on heavy foundation vision models. 

In production computer vision pipelines, Meta's Segment Anything Model (SAM ViT-B) is decoupled into an Image Encoder (Vision Transformer) and a lightweight Mask Decoder. The Vision Transformer backbone accounts for more than 90% of the total execution time, making it the primary operational bottleneck.

The objective of this proof of concept was to optimize the Vision Transformer pipeline using **NVIDIA TensorRT 10**, applying graph optimization, constant folding, and modern asynchronous execution interfaces (`execute_async_v3`).

---

## Benchmark Setup & Methodology

- **Hardware:** NVIDIA T4 GPU (Google Colab Runtime, CUDA 12)
- **Model:** SAM ViT-B (`sam_vit_b_01ec64.pth`)
- **Input Dimensions:** `(1, 3, 1024, 1024)`
- **Benchmarking Protocol:**
  - 15 warm-up iterations to initialize CUDA execution caches.
  - 50 continuous iterations measured via hardware CUDA events (`torch.cuda.Event`).
  - Hardware stream synchronization with zero-copy tensor address bindings.

---

## Results Summary

| Framework | Mean Latency | 95th Percentile (p95) | Throughput | Latency Reduction | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch (Baseline FP32)** | 395.85 ms | 405.38 ms | 2.53 FPS | Baseline | 1.0x |
| **NVIDIA TensorRT 10** | **304.93 ms** | **311.55 ms** | **3.28 FPS** | **-23.0%** | **x1.30** |

![Benchmark Results](benchmark_tensorrt_vs_pytorch.png)

---

## Engineering Takeaways

1. **Deterministic Latency:** The p95 latency decreased from 405.38 ms to 311.55 ms, demonstrating improved predictability under load.
2. **Operational Throughput:** Throughput increased by 0.75 FPS, allowing an additional ~65,000 processed frames per day on the same hardware instance.
3. **TensorRT 10 Migration:** Handled TensorRT 10 runtime transitions by leveraging direct tensor memory bindings via `context.set_tensor_address` and asynchronous execution streams.

---

## How to Run

1. Install dependencies:
   ```bash
   pip install torch torchvision onnx tensorrt opencv-python matplotlib
   pip install git+[https://github.com/facebookresearch/segment-anything.git](https://github.com/facebookresearch/segment-anything.git)
