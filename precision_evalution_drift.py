"""
Évaluation de l'intégrité numérique et de la dérive de précision.
Comparaison : PyTorch FP32 Baseline vs NVIDIA TensorRT 10 Engine
Sortie graphique : precision_drift_distribution.png (Histogramme logarithmique)
"""

import os
import urllib.request
import numpy as np
import cv2
import torch
import tensorrt as trt
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry

# ---------------------------------------------------------
# 1. Vérification matérielle et ressources
# ---------------------------------------------------------
assert torch.cuda.is_available(), "Erreur : Un GPU NVIDIA compatible CUDA est obligatoire."
device = "cuda"

checkpoint_path = "sam_vit_b_01ec64.pth"
image_path = "truck.jpg"
engine_file_path = "sam_encoder_fp16.engine"

assert os.path.exists(engine_file_path), (
    f"Fichier '{engine_file_path}' introuvable. "
    "Veuillez exécuter benchmark_sam_trt.py au préalable pour compiler le moteur TensorRT."
)

if not os.path.exists(checkpoint_path):
    print("Téléchargement des poids officiels SAM ViT-B...")
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

# ---------------------------------------------------------
# 2. Préparation du tenseur d'entrée
# ---------------------------------------------------------
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized = cv2.resize(image_rgb, (1024, 1024))
mean = np.array([123.675, 116.28, 103.53])
std = np.array([58.395, 57.12, 57.375])
input_tensor = (resized - mean) / std
input_tensor = input_tensor.transpose((2, 0, 1))
input_tensor = np.expand_dims(input_tensor, axis=0)
dummy_input = torch.tensor(input_tensor, dtype=torch.float32, device=device)

# ---------------------------------------------------------
# 3. Inférence Baseline PyTorch (FP32)
# ---------------------------------------------------------
print("Calcul de l'inférence de référence PyTorch...")
sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
sam.to(device=device)
encoder = sam.image_encoder
encoder.eval()

with torch.no_grad():
    pytorch_output = encoder(dummy_input)

# ---------------------------------------------------------
# 4. Inférence Runtime NVIDIA TensorRT 10
# ---------------------------------------------------------
print("Calcul de l'inférence TensorRT 10...")
trt_logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(trt_logger)

with open(engine_file_path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

input_tensor_gpu = dummy_input.contiguous()
output_tensor_gpu = torch.empty((1, 256, 64, 64), dtype=torch.float32, device=device).contiguous()

context.set_tensor_address("image_input", input_tensor_gpu.data_ptr())
context.set_tensor_address("image_embeddings", output_tensor_gpu.data_ptr())

cuda_stream = torch.cuda.current_stream().cuda_stream
context.execute_async_v3(stream_handle=cuda_stream)
torch.cuda.synchronize()

# ---------------------------------------------------------
# 5. Calcul des métriques de précision
# ---------------------------------------------------------
py_vec = pytorch_output.detach().cpu().numpy().flatten()
trt_vec = output_tensor_gpu.detach().cpu().numpy().flatten()

abs_diff = np.abs(py_vec - trt_vec)
max_error = float(np.max(abs_diff))
mean_error = float(np.mean(abs_diff))
rmse = float(np.sqrt(np.mean((py_vec - trt_vec) ** 2)))
cosine_sim = float(np.dot(py_vec, trt_vec) / (np.linalg.norm(py_vec) * np.linalg.norm(trt_vec)))

print("\n--- RAPPORT D'AUDIT QUALITÉ DU SIGNAL ---")
print(f"Similarité Cosinus             : {cosine_sim:.8f} / 1.00000000")
print(f"Écart quadratique moyen (RMSE) : {rmse:.6e}")
print(f"Erreur absolue moyenne (MAE)   : {mean_error:.6e}")
print(f"Erreur maximale ponctuelle     : {max_error:.6f}")

# ---------------------------------------------------------
# 6. Génération de l'histogramme unique
# ---------------------------------------------------------
plt.figure(figsize=(9, 6), dpi=300)

plt.hist(abs_diff, bins=70, color="#76B900", edgecolor="#222222", alpha=0.85, log=True)
plt.xlabel("Erreur absolue |y_pytorch - y_trt|", fontweight="bold", fontsize=11)
plt.ylabel("Nombre de valeurs (Échelle log)", fontweight="bold", fontsize=11)
plt.title("Distribution de l'écart numérique (1 048 576 points)", fontweight="bold", fontsize=13)
plt.grid(axis="y", linestyle="--", alpha=0.4)

metrics_box = (
    f"Similarité Cosinus : {cosine_sim:.6f}\n"
    f"RMSE : {rmse:.2e}\n"
    f"MAE : {mean_error:.2e}\n"
    f"Max Error : {max_error:.4f}"
)
plt.text(0.95, 0.95, metrics_box, transform=plt.gca().transAxes, ha="right", va="top", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFFFF", edgecolor="#76B900"))

plt.tight_layout()
output_fig = "precision_drift_distribution.png"
plt.savefig(output_fig)
plt.show()

print(f"\nGraphique sauvegardé sous '{output_fig}'.")
