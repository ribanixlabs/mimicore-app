"""Mimicore RVC v2 trainer. Runs INSIDE the Training Pack's own Python (PyTorch CUDA + patched Applio); started by
the Mimicore app (src-tauri/src/trainer.rs) with cwd = <pack>/applio.

    python mimicore_train.py --name NAME --dataset DIR --runs DIR --out DIR --epochs 250 --batch 8 [--tools DIR] [--fresh]

Steps (same as Mimicore's TrainJob): slice/clean audio -> ContentVec + RMVPE features -> train from the TITAN 48 kHz
base model -> faiss index -> export voice.onnx / index.npy / meta.json into --out (Mimicore's RVC voice format).
Progress for the app: lines starting with "@@" + JSON ({"step": n, "of": 5, "label": ...}, {"epoch": n}, {"done": dir}).
Re-running with the same --name continues from the last checkpoint; --fresh deletes that run folder first and starts
again from the base model (what a brand-new training of an existing name should pass).
Recordings that contain no clear speech (silence, noise, hum) are refused after slicing, before any GPU time is spent.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"  # Force headless matplotlib backend (prevents Colab inline backend ValueError)

NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # Windows: no console window; Linux (Colab cloud training): unused
SR = 48000
TITAN = Path("rvc/models/pretraineds/custom/TITAN_Medium_recommended_for_speech")
# Official RVC v2 pretrained base (IAHispano/Applio Resources, public): the TITAN medium base is fine-tuned from it,
# so the run stays compatible (same architecture, same 48 kHz). ~220 MB in total, downloaded once and cached.
# Pinned to ONE commit of the Hugging Face repo (never `main`) with exact sizes and SHA-256: these are PyTorch checkpoints (pickles), so a
# swapped file would be code execution. Same values as `SINGING_FILES` in src-tauri/src/trainer.rs.
RVC_V2_COMMIT = "3ff370b7023086f57e200ef86a9a251ddf4da472"
RVC_V2 = ("f0G48k.pth", "f0D48k.pth",
          "https://huggingface.co/IAHispano/Applio/resolve/" + RVC_V2_COMMIT + "/Resources/pretrained_v2/{}?download=true",
          [75465569, 142875703],
          ["b5d51f589cc3632d4eae36a315b4179397695042edc01d15312e1bddc2b764a4", "2269b73c7a4cf34da09aea99274dabf99b2ddb8a42cbfb065fb3c0aa9a2fc748"])
STEPS = ["Cleaning & slicing audio", "Extracting voice features", "Training", "Building voice index", "Exporting voice"]


def verified_base(path, size, sha256):
    """True when `path` is exactly the pinned file (size and SHA-256)."""
    try:
        if path.stat().st_size != size:
            return False
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(8 << 20), b""):
                h.update(block)
        return h.hexdigest() == sha256
    except OSError:
        return False


def say(**kw):
    print("@@" + json.dumps(kw), flush=True)


def step(n):
    say(step=n, of=len(STEPS), label=STEPS[n - 1])


OOM = ("out of memory", "bad allocation", "memory allocation failure", "cublas", "cudnn")
# Disk full (errno 28 / WinError 112), not VRAM: Applio's trainer exits 0 anyway, so it would look like an early stop.
DISK_FULL = ("no space left on device", "errno 28", "not enough space on the disk")


def run(args):
    oom = disk_full = False
    env = dict(os.environ, MPLBACKEND="Agg")
    p = subprocess.Popen([sys.executable, *map(str, args)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         creationflags=NO_WINDOW, env=env)
    buf = b""
    for ch in iter(lambda: p.stdout.read(1), b""):
        if ch not in (b"\n", b"\r"):
            buf += ch
            continue
        line = buf.decode("utf-8", "replace").strip()
        buf = b""
        if not line or "it/s]" in line or "s/it]" in line:
            continue
        low = line.lower()
        oom = oom or any(k in low for k in OOM)
        disk_full = disk_full or any(k in low for k in DISK_FULL)
        m = re.search(r"\| epoch=(\d+) \|", line)
        if m:
            say(epoch=int(m.group(1)))
        print(line, flush=True)
    tail = buf.decode("utf-8", "replace").strip()  # a final line with no trailing newline still gets classified
    if tail and "it/s]" not in tail and "s/it]" not in tail:
        low = tail.lower()
        oom = oom or any(k in low for k in OOM)
        disk_full = disk_full or any(k in low for k in DISK_FULL)
        print(tail, flush=True)
    p.wait()
    if disk_full:
        if os.name != "nt":
            raise SystemExit("Training ran out of disk space on the cloud machine. Free up space and start again.")
        raise SystemExit("Training ran out of disk space. Mimicore keeps checkpoints and logs in "
                         "%LOCALAPPDATA%\\Mimicore\\training on the system drive (C:). Free up at least 10 GB there, then "
                         "start again. With the same recordings it continues from the last saved epoch.")
    if oom:
        if os.name != "nt":
            raise SystemExit("Ran out of graphics memory on the cloud GPU. Start again: the next run uses a smaller batch.")
        raise SystemExit("Ran out of memory. Close other programs (games, browsers, the live voice), and make sure the "
                         "system drive has at least 10 GB free so Windows can grow its page file, then start again.")
    if p.returncode != 0:
        raise SystemExit(f"{args[0]} failed (exit {p.returncode})")


def checkpoints(run_dir, name):
    out = []
    for p in run_dir.glob(f"{name}_*e_*s.pth"):
        m = re.search(r"_(\d+)e_(\d+)s\.pth$", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)

# --- speech gate -------------------------------------------------------------------------------------------------
# Duration alone lets a silent or noise-only dataset through. After slicing we measure how much of the audio is
# VOICED speech: loud enough and periodic in the 70-400 Hz pitch range (white noise, hiss, rumble and silence are not).
MIN_VOICED_S = 30.0       # seconds of voiced speech needed (3 minutes of normal speech has about 70-90 s)
MIN_VOICED_FRACTION = 0.08  # ... and at least this share of the audio
MAX_ANALYSE_S = 1800.0    # look at the first 30 minutes only: enough to decide, quick on any PC
NOT_ENOUGH_SPEECH = ("We could not find enough clear speech in these recordings. Please record in a quiet room, "
                     "speak close to the microphone, and add more of your voice (about 3 minutes or more), then try again.")


def read_audio(path):
    """(mono float32 samples, sample rate). soundfile ships with the pack; scipy / plain 16-bit WAV are fallbacks."""
    import numpy as np

    try:
        import soundfile as sf

        x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except ImportError:
        try:
            from scipy.io import wavfile

            sr, x = wavfile.read(str(path))
            x = x.astype("float32") / (32768.0 if x.dtype.kind == "i" else 1.0)
        except ImportError:
            import wave

            with wave.open(str(path), "rb") as w:
                sr, ch = w.getframerate(), w.getnchannels()
                x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype("float32") / 32768.0
                x = x.reshape(-1, ch) if ch > 1 else x
    x = np.asarray(x, dtype="float32")
    return (x.mean(axis=1) if x.ndim > 1 else x), int(sr)


def voiced_stats(x, sr):
    """(voiced seconds, total seconds, pitch of each voiced frame in Hz) for a mono signal."""
    import numpy as np

    f = max(1, int(sr) // 16000)  # 16 kHz is plenty for pitch; cheap integer decimation of 32/48 kHz audio
    if f > 1:
        x = x[: len(x) // f * f].reshape(-1, f).mean(axis=1)
    sr = sr / f
    total = len(x) / sr
    frame, hop = int(0.04 * sr), int(0.02 * sr)
    if len(x) < frame * 2:
        return 0.0, total, np.zeros(0)
    x = np.asarray(x, dtype="float64") - float(np.mean(x))
    win = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop]
    db = 20 * np.log10(np.sqrt(np.mean(win ** 2, axis=1)) + 1e-9)
    active = db > max(-55.0, float(np.percentile(db, 95)) - 30.0)
    y = np.append(x[0], x[1:] - 0.97 * x[:-1])  # pre-emphasis: low-frequency rumble is not speech
    w = np.lib.stride_tricks.sliding_window_view(y, frame)[::hop] * np.hanning(frame)
    n = 1 << (2 * frame - 1).bit_length()
    r = np.fft.irfft(np.abs(np.fft.rfft(w, n, axis=1)) ** 2, n, axis=1)
    lo, hi = max(2, int(sr / 400)), min(frame - 2, int(sr / 70))
    wr = np.fft.irfft(np.abs(np.fft.rfft(np.hanning(frame), n)) ** 2, n)  # the window's own autocorrelation (bias)
    nr = r[:, lo - 1: hi + 2] / (r[:, :1] + 1e-12) / wr[lo - 1: hi + 2] * wr[0]  # normalised, bias-corrected, pitch lags
    mid = nr[:, 1:-1]
    k = np.argmax(mid, axis=1)
    peak = mid[np.arange(len(mid)), k]
    local = (peak > nr[np.arange(len(nr)), k]) & (peak >= nr[np.arange(len(nr)), k + 2])  # a real peak, not a slope
    voiced = active & local & (peak > 0.4)
    return float(voiced.sum()) * hop / sr, total, sr / (k[voiced] + lo)


def speech_check(files):
    """(ok, voiced seconds, analysed seconds). Never raises on unreadable files: those simply count as no speech."""
    import numpy as np

    voiced = total = 0.0
    pitch = []
    for p in files:
        if total >= MAX_ANALYSE_S:
            break
        try:
            x, sr = read_audio(p)
            v, t, f0 = voiced_stats(x, sr)
        except Exception as e:
            print(f"could not analyse {Path(p).name}: {e}", flush=True)
            continue
        voiced, total = voiced + v, total + t
        pitch.append(f0)
    f0 = np.concatenate(pitch) if pitch else np.zeros(0)
    # a single steady tone / hum is periodic but is not a voice: real speech moves around (> a quarter of a semitone)
    varied = len(f0) > 10 and float(np.std(np.log2(f0))) > 0.02
    ok = varied and voiced >= MIN_VOICED_S and voiced >= MIN_VOICED_FRACTION * max(total, 1e-9)
    return ok, voiced, total


def best_checkpoint(run_dir, cps):
    """Checkpoint with the lowest smoothed validation mel loss (late epochs can over-train). Falls back to the last."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        ea = EventAccumulator(str(run_dir / "eval"), size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        tag = next((t for t in ("loss/g/mel", "loss_avg_50/g/mel") if t in tags), None)
        if not tag or len(cps) < 2:
            return cps[-1]
        m = re.search(r"_(\d+)e_(\d+)s\.pth$", cps[-1][1].name)
        steps_per_epoch = int(m.group(2)) / max(1, int(m.group(1)))
        pts = [(e.step / steps_per_epoch, e.value) for e in ea.Scalars(tag)]
        if len(pts) < 5:
            return cps[-1]
        last = cps[-1][0]
        w = max(2.0, last * 0.04)  # smooth over ~4% of the run so single noisy epochs do not decide
        def score(ep):
            vals = [v for x, v in pts if ep - w <= x <= ep + w]
            return sum(vals) / len(vals) if vals else float("inf")
        # ignore the first fifth: the voice is still forming even if the loss dips
        cands = [c for c in cps if c[0] >= last * 0.2] or cps
        best = min(cands, key=lambda c: score(c[0]))
        print(f"best checkpoint: epoch {best[0]} (smoothed mel loss {score(best[0]):.3f}; last epoch {last}: {score(last):.3f})",
              flush=True)
        return best
    except Exception as e:
        print(f"checkpoint selection skipped: {e}", flush=True)
        return cps[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=0)  # 0 = about 20 checkpoints per run
    ap.add_argument("--fresh", action="store_true")  # forget any earlier run of this name and start from the base model
    ap.add_argument("--basis", choices=["titan", "rvcv2"], default="titan")  # base model to fine-tune from
    ap.add_argument("--bases-dir", default=str(Path.cwd() / "rvc" / "models" / "pretraineds" / "custom"))
    ap.add_argument("--tools", default=str(Path(sys.executable).resolve().parent.parent / "tools"))
    a = ap.parse_args()
    if a.save_every <= 0:
        a.save_every = max(5, a.epochs // 20)

    if not a.name or a.name in (".", "..") or Path(a.name).name != a.name:
        raise SystemExit("The voice name is not valid for a training folder.")
    runs = Path(a.runs).resolve()
    run_dir = runs / a.name
    # Where Applio reads/writes. APPLIO_LOGS_ROOT covers Applio up to 3.6.5 (train.py patch); 3.6.6+ moved to
    # APPLIO_LOGS_DIR (per-user config), APPLIO_CODE_ROOT tells it where the shipped logs/mute templates live, and
    # APPLIO_CONFIG_FILE keeps it out of the user's %APPDATA% (read-only/packed install): all four coords, only.
    applio = Path.cwd()
    os.environ["APPLIO_LOGS_ROOT"] = str(runs)
    os.environ["APPLIO_LOGS_DIR"] = str(runs)
    os.environ["APPLIO_CODE_ROOT"] = str(applio)
    os.environ["APPLIO_CONFIG_FILE"] = str(applio / "assets" / "config.json")
    cfg = applio / "assets" / "config.json"
    if not cfg.exists():  # else training falls back to fp32 precision (twice as slow); same as the Colab notebook
        tpl = applio / "assets" / "config_template.json"
        if tpl.exists():
            cfg.write_text(tpl.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            print(f"wrote {cfg.name} from the template")
    bases = Path(a.bases_dir).resolve()
    if a.basis == "rvcv2":
        g, d = bases / RVC_V2[0], bases / RVC_V2[1]
        if not (verified_base(g, RVC_V2[3][0], RVC_V2[4][0]) and verified_base(d, RVC_V2[3][1], RVC_V2[4][1])):  # also re-checks cached files
            bases.mkdir(parents=True, exist_ok=True)
            import urllib.request

            print("Downloading the official RVC v2 base (about 220 MB, once)...", flush=True)
            for f, size, want in zip(RVC_V2[:2], RVC_V2[3], RVC_V2[4]):
                dst = bases / f
                if dst.exists() and not verified_base(dst, size, want):
                    print(f"{f} is not the expected file; downloading it again.", flush=True)
                    dst.unlink()
                if not dst.exists():
                    url = RVC_V2[2].format(f)
                    part = dst.with_name(dst.name + ".part")
                    h = hashlib.sha256()
                    with urllib.request.urlopen(url) as r, open(part, "wb") as out:
                        got, total = 0, int(r.headers.get("Content-Length") or size)
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            got += len(chunk)
                            out.write(chunk)
                            h.update(chunk)
                            print(f"@@ {{\"label\": \"Downloading {f}: {got / 1e6:.0f} of {total / 1e6:.0f} MB\"}}", flush=True)
                    if part.stat().st_size != size:
                        part.unlink()
                        raise SystemExit(f"Download of {f} is incomplete; try again.")
                    if h.hexdigest() != want:
                        part.unlink()
                        raise SystemExit(f"Download of {f} failed its integrity check (not the expected file); nothing was kept.")
                    part.rename(dst)
    else:
        g, d = TITAN / "G-f048k-TITAN-Medium.pth", TITAN / "D-f048k-TITAN-Medium.pth"
        if not (g.exists() and d.exists()):
            raise SystemExit("The base model files are missing from the Training Pack - reinstall it.")
    if a.fresh and run_dir.exists() and run_dir != runs and runs in run_dir.parents:
        shutil.rmtree(run_dir, ignore_errors=True)
        if run_dir.exists():
            raise SystemExit("Could not clear the previous training of this voice (a file is in use). Close other "
                             "programs and try again.")
        print("Starting fresh: the earlier run of this name was cleared.", flush=True)
    if run_dir.exists():
        if any(run_dir.glob("G_*.pth")) or any(run_dir.glob("*_*e_*s.pth")):
            print("Continuing the earlier run of this name (pass --fresh to start over).", flush=True)
        # Continuing: keep learned weights (G_/D_, checkpoints) but rebuild everything derived from the recordings.
        for sub in ("sliced_audios", "sliced_audios_16k", "f0", "f0_voiced", "extracted"):
            shutil.rmtree(run_dir / sub, ignore_errors=True)
        for f in [run_dir / "filelist.txt", *run_dir.glob("*.index")]:
            f.unlink(missing_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    cores = str(max(1, (os.cpu_count() or 4) // 2))

    step(1)
    run(["rvc/train/preprocess/preprocess.py", run_dir, Path(a.dataset).resolve(), SR, cores, "Automatic", True, False,
         0.5, 3.0, 0.3, "pre"])
    sliced_dir = run_dir / "sliced_audios"
    sliced_files = list(sliced_dir.glob("*.wav")) if sliced_dir.is_dir() else []
    if not sliced_files:
        raise SystemExit("Preprocessing produced no sliced audio files. Make sure the dataset audio recordings are valid, "
                         "contain audible speech, and are not locked by another program.")
    ok, voiced, total = speech_check(sorted(sliced_files))
    print(f"speech check: {voiced:.0f} s of voiced speech in {total:.0f} s of audio", flush=True)
    if not ok:
        raise SystemExit(NOT_ENOUGH_SPEECH)
    step(2)
    run(["rvc/train/extract/extract.py", run_dir, "rmvpe", cores, "0", SR, "contentvec", "None", "2"])
    step(3)
    cache_gpu = False
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 6 * (1024**3):
            cache_gpu = True
    except Exception:
        pass
    run(["rvc/train/train.py", a.name, a.save_every, a.epochs, g, d, "0", a.batch, SR, True, True, cache_gpu, False,
         "HiFi-GAN", False])
    cps = checkpoints(run_dir, a.name)
    if not cps or cps[-1][0] < a.epochs - a.save_every:  # Applio exits 0 even when its worker died; tolerate save_every not dividing epochs
        last = cps[-1][0] if cps else 0
        raise SystemExit(f"Training stopped early (last saved epoch {last} of {a.epochs}). Common causes: not enough "
                         "video memory, or too little audio.")
    step(4)
    run(["rvc/train/process/extract_index.py", run_dir, "Auto"])
    step(5)
    out = Path(a.out).resolve()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    best_epoch, pth = best_checkpoint(run_dir, cps)
    shutil.copy2(pth, out / pth.name)
    idx = next(iter(sorted(run_dir.glob("*.index"))), None)
    if idx:
        shutil.copy2(idx, out / idx.name)
    sys.path.insert(0, a.tools)
    import onnx_export

    onnx_export.export_voice(out, "fp16", out / pth.name)
    meta = json.loads((out / "meta.json").read_text())
    meta.update(source="mimicore-trainer", version="v2", epochs=best_epoch, trained_epochs=cps[-1][0],
                base="rvc_v2_official" if a.basis == "rvcv2" else "titan_medium",
                # The faiss index holds the ContentVec features of the recordings (extract_index reads the extracted
                # features, never a checkpoint), so it is identical whichever epoch voice.onnx was exported from.
                index_source="recording_features", onnx_epoch=best_epoch)
    try:  # median pitch of the recordings: lets the app pitch-match text-to-speech for this voice
        import numpy as np

        f0 = np.concatenate([np.load(f).ravel() for f in (run_dir / "f0_voiced").glob("*.npy")])
        meta["f0_median"] = round(float(np.median(f0[f0 > 1])), 1)
    except Exception as e:
        print(f"pitch statistics skipped: {e}", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    for f in out.glob("*.pth"):  # the app needs only the ONNX export (+ index.npy)
        f.unlink()
    for f in out.glob("*.index"):
        f.unlink()
    say(done=str(out))


if __name__ == "__main__":
    main()
