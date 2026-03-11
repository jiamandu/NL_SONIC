"""
Compare model_decoder.onnx and model_decoder_native.pth
Input:  [1, 994] float32
Output: [1,  29] float32
"""

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

MODEL_DIR = "/home/jiamandu/projects/NL_SONIC/pretrained/model"
ONNX_PATH = f"{MODEL_DIR}/model_decoder.onnx"
PTH_PATH  = f"{MODEL_DIR}/model_decoder_native.pth"


class DecoderMLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── 1. Load models ────────────────────────────────────────────────────────────
print("Loading ONNX model …")
ort_sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
onnx_input_name = ort_sess.get_inputs()[0].name

print("Loading PT model …")
pt_model = DecoderMLP([994, 2048, 2048, 1024, 1024, 512, 512, 29])
pt_model.load_state_dict(torch.load(PTH_PATH, map_location="cpu"))
pt_model.eval()

# ── 2. Random test inputs ─────────────────────────────────────────────────────
np.random.seed(42)
N_TESTS = 5

print(f"\nRunning {N_TESTS} random input comparisons …\n")
print(f"{'Test':>5}  {'max_abs_diff':>14}  {'mean_abs_diff':>14}  {'match (atol=1e-4)':>18}")
print("-" * 60)

all_match = True
for i in range(N_TESTS):
    x_np = np.random.randn(1, 994).astype(np.float32)
    x_pt = torch.from_numpy(x_np)

    out_onnx = ort_sess.run(None, {onnx_input_name: x_np})[0]

    with torch.no_grad():
        out_pt_np = pt_model(x_pt).cpu().numpy()

    max_diff  = float(np.max(np.abs(out_onnx - out_pt_np)))
    mean_diff = float(np.mean(np.abs(out_onnx - out_pt_np)))
    match     = np.allclose(out_onnx, out_pt_np, atol=1e-4)
    all_match = all_match and match

    print(f"{i+1:>5}  {max_diff:>14.6f}  {mean_diff:>14.6f}  {'YES' if match else 'NO':>18}")

print("-" * 60)
print(f"\nOverall: {'ALL MATCH ✓' if all_match else 'MISMATCH ✗'}")

if not all_match:
    print("\n[Detail] Last test sample outputs:")
    print("  ONNX:", out_onnx.flatten()[:8], "…")
    print("  PT  :", out_pt_np.flatten()[:8], "…")
