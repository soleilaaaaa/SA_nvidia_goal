"""
Essai d'accélération d'inférence : Segment Anything (SAM ViT-B)
Optimisation du graphe via NVIDIA TensorRT 10 sur GPU NVIDIA.
Méthodologie de benchmark rigoureuse via CUDA Events matériels.
"""

import os
import urllib.request
import numpy as np
import cv2
import torch
import onnx
import tensorrt as trt
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry

# ---------------------------------------------------------
# 1. Préparation de l'environnement et du modèle
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "sam_vit_b_01ec64.pth"
image_path = "truck.jpg"

if not os.path.exists(checkpoint_path):
    print("Téléchargement des poids du modèle SAM ViT-B...")
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        checkpoint_path
    )

if not os.path.exists(image_path):
    print("Téléchargement de l'image de test...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg",
        image_path
    )

print("Chargement du modèle PyTorch...")
sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
sam.to(device=device)
encoder = sam.image_encoder
encoder.eval()

# Préparation du tenseur d'entrée (1, 3, 1024, 1024)
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized = cv2.resize(image_rgb, (1024, 1024))
mean = np.array([123.675, 116.28, 103.53])
std = np.array([58.395, 57.12, 57.375])
input_tensor = (resized - mean) / std
input_tensor = input_tensor.transpose((2, 0, 1))
input_tensor = np.expand_dims(input_tensor, axis=0)
dummy_input = torch.tensor(input_tensor, dtype=torch.float32, device="cuda")

# ---------------------------------------------------------
# 2. Mesure Baseline PyTorch (CUDA Events + Warm-up)
# ---------------------------------------------------------
print("\n--- Évaluation PyTorch Baseline (FP32) ---")
with torch.no_grad():
    for _ in range(15):
        _ = encoder(dummy_input)
torch.cuda.synchronize()

num_iterations = 50
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
pytorch_latencies = []

with torch.no_grad():
    for _ in range(num_iterations):
        start_event.record()
        _ = encoder(dummy_input)
        end_event.record()
        torch.cuda.synchronize()
        pytorch_latencies.append(start_event.elapsed_time(end_event))

pytorch_mean = float(np.mean(pytorch_latencies))
pytorch_p95 = float(np.percentile(pytorch_latencies, 95))
pytorch_fps = 1000.0 / pytorch_mean
print(f"PyTorch Mean Latency : {pytorch_mean:.2f} ms | p95 : {pytorch_p95:.2f} ms | Throughput : {pytorch_fps:.2f} FPS")

# ---------------------------------------------------------
# 3. Export du graphe vers ONNX
# ---------------------------------------------------------
onnx_file = "sam_encoder.onnx"
print(f"\nExportation du modèle vers {onnx_file}...")
torch.onnx.export(
    encoder,
    dummy_input,
    onnx_file,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["image_input"],
    output_names=["image_embeddings"]
)

# ---------------------------------------------------------
# 4. Compilation TensorRT 10
# ---------------------------------------------------------
engine_file = "sam_encoder_fp16.engine"
print(f"\nCompilation du moteur TensorRT ({engine_file})...")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network()
parser = trt.OnnxParser(network, TRT_LOGGER)
config = builder.create_builder_config()

# Allocation de l'espace de travail (2 Go)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)

with open(onnx_file, "rb") as model:
    parser.parse(model.read())

serialized_engine = builder.build_serialized_network(network, config)
with open(engine_file, "wb") as f:
    f.write(serialized_engine)
print("Moteur TensorRT compilé avec succès.")

# ---------------------------------------------------------
# 5. Benchmark Runtime TensorRT 10
# ---------------------------------------------------------
print("\n--- Évaluation NVIDIA TensorRT 10 ---")
runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()

input_tensor_gpu = dummy_input.contiguous()
output_tensor_gpu = torch.empty((1, 256, 64, 64), dtype=torch.float32, device="cuda").contiguous()

context.set_tensor_address("image_input", input_tensor_gpu.data_ptr())
context.set_tensor_address("image_embeddings", output_tensor_gpu.data_ptr())

cuda_stream = torch.cuda.current_stream().cuda_stream

for _ in range(15):
    context.execute_async_v3(stream_handle=cuda_stream)
torch.cuda.synchronize()

trt_latencies = []
for _ in range(num_iterations):
    start_event.record()
    context.execute_async_v3(stream_handle=cuda_stream)
    end_event.record()
    torch.cuda.synchronize()
    trt_latencies.append(start_event.elapsed_time(end_event))

trt_mean = float(np.mean(trt_latencies))
trt_p95 = float(np.percentile(trt_latencies, 95))
trt_fps = 1000.0 / trt_mean
speedup = pytorch_mean / trt_mean
latency_reduction = ((pytorch_mean - trt_mean) / pytorch_mean) * 100

print(f"TensorRT Mean Latency : {trt_mean:.2f} ms | p95 : {trt_p95:.2f} ms | Throughput : {trt_fps:.2f} FPS")
print(f"Facteur d'accélération : x{speedup:.2f} (Réduction de latence : -{latency_reduction:.1f}%)")

# ---------------------------------------------------------
# 6. Visualisation
# ---------------------------------------------------------
frameworks = ["PyTorch Standard (FP32)", "NVIDIA TensorRT 10"]
latencies = [pytorch_mean, trt_mean]
colors = ["#555555", "#76B900"]

plt.figure(figsize=(8, 5), dpi=300)
bars = plt.bar(frameworks, latencies, color=colors, width=0.45)
plt.ylabel("Latence moyenne d'inférence (ms) - Plus bas est meilleur", fontweight="bold")
plt.title("Benchmark Inférence SAM ViT-B (Image Encoder)\nPyTorch vs NVIDIA TensorRT", fontweight="bold")

for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., h + 10, f"{h:.1f} ms", ha="center", va="bottom", fontweight="bold")

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("benchmark_tensorrt_vs_pytorch.png")
print("Graphique sauvegardé sous 'benchmark_tensorrt_vs_pytorch.png'")
