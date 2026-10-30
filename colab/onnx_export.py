r"""Convert PyTorch RVC models to ONNX for the fast any-GPU engine (needs PyTorch + Applio: full app / dev only).

CLI:
    python src\onnx_export.py base [--weights fp16|int8|fp32]        -> <home>/models/contentvec.onnx, rmvpe.onnx, mel_basis.npy
    python src\onnx_export.py voice <voice dir> [--weights ...]      -> <voice dir>/voice.onnx, index.npy, meta.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
HOME = SRC.parent
APPLIO = next((p for p in (SRC / "applio", HOME / "applio") if p.exists()), SRC / "applio")
MODELS = HOME / "models"
OPSET = 17


def _applio():
    if str(APPLIO) not in sys.path:
        sys.path.insert(0, str(APPLIO))
    os.chdir(APPLIO)


# ------------------------------------------------------------------------------ weight storage
def compress_weights(path, mode="fp16"):
    """Shrink big weights on disk. Compute stays fp32 (works on every GPU); ONNX Runtime restores fp32 weights.
    fp16 : weights stored as float16 + Cast             (~half size, ~1e-4 relative weight error)
    int8 : weights stored as int8 per output channel + DequantizeLinear   (~quarter size, ~0.4% weight error)"""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    if mode == "fp32":
        return
    m = onnx.load(str(path))
    g = m.graph
    new_nodes, keep = [], []
    for init in g.initializer:
        n = int(np.prod(init.dims)) if init.dims else 1
        if init.data_type != TensorProto.FLOAT or n < 4096:
            keep.append(init)
            continue
        w = numpy_helper.to_array(init)
        if mode == "int8" and w.ndim >= 2:
            axis = 0
            red = tuple(range(1, w.ndim))
            scale = (np.abs(w).max(axis=red) / 127.0).astype(np.float32)
            scale[scale == 0] = 1e-12
            q = np.clip(np.round(w / scale.reshape((-1,) + (1,) * (w.ndim - 1))), -127, 127).astype(np.int8)
            qi = numpy_helper.from_array(q, init.name + "__q")
            si = numpy_helper.from_array(scale, init.name + "__s")
            zi = numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), init.name + "__z")
            keep += [qi, si, zi]
            new_nodes.append(helper.make_node("DequantizeLinear", [qi.name, si.name, zi.name], [init.name],
                                              axis=axis, name="dq_" + init.name))
        else:
            small = numpy_helper.from_array(w.astype(np.float16), init.name + "__h")
            keep.append(small)
            new_nodes.append(helper.make_node("Cast", [small.name], [init.name], to=TensorProto.FLOAT,
                                              name="cast_" + init.name))
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes + nodes)
    onnx.save(m, str(path))


def _export(model, args, path, inputs, outputs, dyn, weights):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, args, str(path), input_names=inputs, output_names=outputs, dynamic_axes=dyn,
                      opset_version=OPSET, do_constant_folding=True, dynamo=False)
    compress_weights(path, weights)
    print(f"  wrote {path.name} ({path.stat().st_size / 1e6:.0f} MB, weights {weights})", flush=True)


# ------------------------------------------------------------------------------ base models
def export_base(out=MODELS, weights=None):
    """Default storage (benchmarked): ContentVec fp16 (int8 adds audible-risk feature error),
    RMVPE int8 (pitch identical except 3 low-confidence frames in 780; saves 90 MB)."""
    import torch

    _applio()
    from rvc.lib.predictors.RMVPE import E2E, MelSpectrogram, N_MELS
    from rvc.lib.utils import load_embedding
    from rvc.realtime.pipeline import strip_parametrizations

    class Hub(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, audio):  # [1, N] 16 kHz -> [1, T, 768]
            return self.m(audio)["last_hidden_state"]

    out = Path(out)
    print("ContentVec (speech content)...", flush=True)
    hub = load_embedding("contentvec").eval().float()
    strip_parametrizations(hub)
    _export(Hub(hub), (torch.randn(1, 16000),), out / "contentvec.onnx", ["audio"], ["feats"],
            {"audio": {1: "n"}, "feats": {1: "t"}}, weights or "fp16")
    print("RMVPE (pitch)...", flush=True)
    m = E2E(4, 1, (2, 2))
    m.load_state_dict(torch.load(str(APPLIO / "rvc/models/predictors/rmvpe.pt"), map_location="cpu",
                                 weights_only=True))
    m.eval()
    _export(m, (torch.randn(1, N_MELS, 64),), out / "rmvpe.onnx", ["mel"], ["salience"],
            {"mel": {2: "t"}, "salience": {1: "t"}}, weights or "int8")
    np.save(out / "mel_basis.npy", MelSpectrogram(N_MELS, 16000, 1024, 160, None, 30, 8000).mel_basis.numpy())


# ------------------------------------------------------------------------------ voices
def _patch_layernorm():
    """Static layer-norm size (ONNX export can't handle x.size(-1)); doesn't modify Applio's files."""
    import torch
    from rvc.lib.algorithm.normalization import LayerNorm

    def fwd(self, x):
        x = x.transpose(1, -1)
        x = torch.nn.functional.layer_norm(x, (self.gamma.shape[0],), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)

    LayerNorm.forward = fwd


def load_synth(pth):
    import torch

    _applio()
    _patch_layernorm()
    from rvc.lib.algorithm.synthesizers import Synthesizer
    from rvc.realtime.pipeline import strip_parametrizations

    cpt = torch.load(str(pth), map_location="cpu", weights_only=True)
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    version, use_f0 = cpt.get("version", "v1"), cpt.get("f0", 1)
    vocoder = cpt.get("vocoder", "HiFi-GAN")
    if version != "v2" or not use_f0 or vocoder not in ("HiFi-GAN", "MRF HiFi-GAN"):
        raise ValueError(f"Only RVC v2 pitch-guided voices with the HiFi-GAN or MRF HiFi-GAN vocoder are supported "
                         f"(this one: {version}, f0={use_f0}, {vocoder}).")
    net = Synthesizer(*cpt["config"], use_f0=use_f0, text_enc_hidden_dim=768, vocoder=vocoder)
    net.load_state_dict(cpt["weight"], strict=False)
    strip_parametrizations(net)
    net.eval()
    return net, dict(sr=int(cpt["config"][-1]), version=version, vocoder=vocoder, speakers=int(cpt["config"][-3]))


def synth_module(net):
    import torch

    class Synth(torch.nn.Module):
        """Synthesizer with the real-time 'skip head' folded in (head is an input)."""

        def __init__(self, n):
            super().__init__()
            self.net = n

        def forward(self, feats, p_len, pitch, pitchf, sid, head):
            n = self.net
            g = n.emb_g(sid).unsqueeze(-1)
            m_p, logs_p, x_mask = n.enc_p(feats, pitch, p_len)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z_p, x_mask, nsff0 = z_p[:, :, head:], x_mask[:, :, head:], pitchf[:, head:]
            z = n.flow(z_p, x_mask, g=g, reverse=True)
            return torch.clamp(n.dec(z * x_mask, nsff0, g=g)[0, 0], -1.0, 1.0)

    return Synth(net)


def export_voice(voice_dir, weights="fp16", pth=None):
    import torch

    voice_dir = Path(voice_dir).resolve()
    pth = Path(pth) if pth else next(iter(sorted(voice_dir.glob("*.pth"))), None)
    if pth is None:
        raise FileNotFoundError(f"No .pth in {voice_dir}")
    print(f"Voice: {pth.name}", flush=True)
    net, meta = load_synth(pth)
    T = 100
    args = (torch.randn(1, T, 768), torch.tensor([T]), torch.randint(1, 255, (1, T)),
            torch.rand(1, T) * 300 + 80, torch.tensor([0]), torch.tensor(10))
    _export(synth_module(net), args, voice_dir / "voice.onnx", ["feats", "p_len", "pitch", "pitchf", "sid", "head"],
            ["audio"], {"feats": {1: "t"}, "pitch": {1: "t"}, "pitchf": {1: "t"}, "audio": {0: "n"}}, weights)
    idx = next(iter(sorted(voice_dir.glob("*.index"))), None)
    meta["has_index"] = False
    if idx is not None:
        import faiss

        ix = faiss.deserialize_index(np.fromfile(str(idx), dtype=np.uint8))  # unicode-safe path
        np.save(voice_dir / "index.npy", ix.reconstruct_n(0, ix.ntotal).astype(np.float16))
        meta["has_index"] = True
    meta["source"] = pth.name
    (voice_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return voice_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["base", "voice"])
    ap.add_argument("path", nargs="?")
    ap.add_argument("--weights", default=None, choices=["fp32", "fp16", "int8"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.what == "base":
        export_base(Path(a.out) if a.out else MODELS, a.weights)
    else:
        export_voice(Path(a.path).resolve(), a.weights or "fp16")
