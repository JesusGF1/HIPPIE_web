"""
Export hippie_techcond_v1.ckpt → hippie_techcond_v1.dynamic.onnx

Run from HIPPIE_web/:
    python export_onnx.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

# Resolve paths relative to this script
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR    = os.path.join(SCRIPT_DIR, "..", "hippie_benchmarking")
CKPT_PATH    = os.path.join(BENCH_DIR, "huggingface", "hippie_techcond_v1.ckpt")
OUTPUT_PATH  = os.path.join(SCRIPT_DIR, "hippie_techcond_v1.dynamic.onnx")

sys.path.insert(0, BENCH_DIR)
sys.path.insert(0, os.path.join(BENCH_DIR, "hippie"))
sys.path.insert(0, os.path.join(BENCH_DIR, "scripts"))

from hippie.multimodal_model import MultiModalCVAE, ExperimentConfigs


# ---------------------------------------------------------------------------
# Load checkpoint (mirrors extract_embeddings.py::build_model)
# ---------------------------------------------------------------------------

def infer_model_dims(state_dict):
    src_key = next(k for k in state_dict if "source_embed" in k and "weight" in k)
    num_sources = state_dict[src_key].shape[0]
    cls_keys = [k for k in state_dict if "class_embed" in k and "weight" in k]
    num_classes = state_dict[cls_keys[0]].shape[0] if cls_keys else 2
    return num_sources, num_classes


def build_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]
    sd = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in sd.items()}

    num_sources, num_classes = infer_model_dims(sd)
    print(f"  num_sources={num_sources}, num_classes={num_classes}")

    model = MultiModalCVAE(
        modalities={"wave": 50, "isi": 100, "acg": 100},
        z_dim=30,
        num_sources=num_sources,
        num_classes=num_classes,
        num_super_regions=0,
        num_layers=0,
        config=ExperimentConfigs.class_decoder_source_bn_aug_reg(),
        backbone_base_width=64,
    )
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Thin wrapper: (wave, isi, acg, source_labels) → mu
# ---------------------------------------------------------------------------

class HIPPIEEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, wave, isi, acg, source_labels):
        data_dict = {"wave": wave, "isi": isi, "acg": acg}
        _, mu, _ = self.model.encode(
            data_dict, source_labels=source_labels, apply_dropout=False
        )
        return mu


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def main():
    print(f"Loading checkpoint: {CKPT_PATH}")
    model = build_model(CKPT_PATH)
    wrapper = HIPPIEEncoder(model)
    wrapper.eval()

    # Dummy inputs (batch=2 so dynamic axis is exercised during tracing)
    B = 2
    wave_d   = torch.zeros(B, 1, 50,  dtype=torch.float32)
    isi_d    = torch.zeros(B, 1, 100, dtype=torch.float32)
    acg_d    = torch.zeros(B, 1, 100, dtype=torch.float32)
    src_d    = torch.zeros(B,         dtype=torch.long)

    print(f"Exporting → {OUTPUT_PATH}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (wave_d, isi_d, acg_d, src_d),
            OUTPUT_PATH,
            input_names=["wave", "isi", "acg", "source_labels"],
            output_names=["hippie_out"],
            dynamic_axes={
                "wave":          {0: "batch"},
                "isi":           {0: "batch"},
                "acg":           {0: "batch"},
                "source_labels": {0: "batch"},
                "hippie_out":    {0: "batch"},
            },
            opset_version=14,
            do_constant_folding=True,
        )

    # Quick sanity check with onnxruntime
    print("Verifying with onnxruntime ...")
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(OUTPUT_PATH, sess_options=so, providers=["CPUExecutionProvider"])
    feed = {
        "wave":          np.zeros((3, 1, 50),  dtype=np.float32),
        "isi":           np.zeros((3, 1, 100), dtype=np.float32),
        "acg":           np.zeros((3, 1, 100), dtype=np.float32),
        "source_labels": np.zeros((3,),        dtype=np.int64),
    }
    out = sess.run(["hippie_out"], feed)[0]
    print(f"  Output shape: {out.shape}  (expected (3, 30))")
    assert out.shape == (3, 30), f"Unexpected shape: {out.shape}"
    print("Export OK.")


if __name__ == "__main__":
    main()
