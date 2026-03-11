"""
Convert model_decoder.pt (onnx2torch) → native PyTorch model
Output:
  model/model_decoder_native.pt   -- full model (torch.save(model))
  model/model_decoder_native.pth  -- state_dict only (recommended, no onnx2torch needed)

Network: SiLU-MLP  994→2048→2048→1024→1024→512→512→29
"""

import sys
from unittest.mock import MagicMock

# 只在这次转换时需要 onnx2torch
_tv = MagicMock()
for _mod in ["torchvision", "torchvision.ops", "torchvision.ops.nms",
             "torchvision._meta_registrations"]:
    sys.modules[_mod] = _tv

import torch
import torch.nn as nn
import numpy as np

MODEL_DIR = "model"
SRC  = f"{MODEL_DIR}/model_decoder.pt"
DST_MODEL = f"{MODEL_DIR}/model_decoder_native.pt"
DST_DICT  = f"{MODEL_DIR}/model_decoder_native.pth"

DIMS = [994, 2048, 2048, 1024, 1024, 512, 512, 29]


# ── 1. 定义原生网络 ───────────────────────────────────────────────────────────
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


# ── 2. 加载旧模型权重 ─────────────────────────────────────────────────────────
print("Loading onnx2torch model …")
src_model = torch.load(SRC, map_location="cpu")
src_sd = src_model.state_dict()

# ── 3. 权重映射（ONNX [in,out] → PyTorch Linear [out,in]）────────────────────
native = DecoderMLP(DIMS)
native_sd = native.state_dict()

n_layers = len(DIMS) - 1
for i in range(n_layers):
    w_key = f"initializers.onnx_initializer_{i * 2}"
    b_key = f"initializers.onnx_initializer_{i * 2 + 1}"
    pt_w  = f"net.{i * 2}.weight"   # Linear 在 Sequential 中的位置（每层 Linear+SiLU）
    pt_b  = f"net.{i * 2}.bias"

    native_sd[pt_w] = src_sd[w_key].T.contiguous()  # 转置
    native_sd[pt_b] = src_sd[b_key]

native.load_state_dict(native_sd)
native.eval()
print("Weights loaded.")

# ── 4. 验证输出一致 ───────────────────────────────────────────────────────────
print("Verifying …")
np.random.seed(0)
x = torch.from_numpy(np.random.randn(1, 994).astype(np.float32))

src_model.eval()
with torch.no_grad():
    out_src    = src_model(x)
    out_native = native(x)

if isinstance(out_src, (tuple, list)):
    out_src = out_src[0]

max_diff = (out_src - out_native).abs().max().item()
print(f"Max abs diff vs original: {max_diff:.2e}")
assert max_diff < 1e-5, "Mismatch! Check weight mapping."
print("Outputs match ✓")

# ── 5. 保存 ──────────────────────────────────────────────────────────────────
torch.save(native, DST_MODEL)
torch.save(native.state_dict(), DST_DICT)
print(f"\nSaved:")
print(f"  {DST_MODEL}  (full model, still needs DecoderMLP class)")
print(f"  {DST_DICT}   (state_dict only, 推荐使用)")
print("\n使用方法:")
print("  model = DecoderMLP([994,2048,2048,1024,1024,512,512,29])")
print("  model.load_state_dict(torch.load('model/model_decoder_native.pth'))")
print("  model.eval()")
