"""Mimicore RVC v2 trainer. Runs INSIDE the Training Pack's own Python (PyTorch CUDA + patched Applio); started by
the Mimicore app (src-tauri/src/trainer.rs) with cwd = <pack>/applio.

    python mimicore_train.py --name NAME --dataset DIR --runs DIR --out DIR --epochs 250 --batch 8 [--tools DIR]

Steps (same as Mimicore's TrainJob): slice/clean audio -> ContentVec + RMVPE features -> train from the TITAN 48 kHz
base model -> faiss index -> export voice.onnx / index.npy / meta.json into --out (Mimicore's RVC voice format).
Progress for the app: lines starting with "@@" + JSON ({"step": n, "of": 5, "label": ...}, {"epoch": n}, {"done": dir}).
Re-running with the same --name continues from the last checkpoint.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # Windows: no console window; Linux (Colab cloud training): unused
SR = 48000
TITAN = Path("rvc/models/pretraineds/custom/TITAN_Medium_recommended_for_speech")
STEPS = ["Cleaning & slicing audio", "Extracting voice features", "Training", "Building voice index", "Exporting voice"]


def say(**kw):
    print("@@" + json.dumps(kw), flush=True)


def step(n):
    say(step=n, of=len(STEPS), label=STEPS[n - 1])


OOM = ("out of memory", "bad allocation", "memory allocation failure")


def run(args):
    oom = False
    p = subprocess.Popen([sys.executable, *map(str, args)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         creationflags=NO_WINDOW)
    buf = b""
    for ch in iter(lambda: p.stdout.read(1), b""):
        if ch not in (b"\n", b"\r"):
            buf += ch
            continue
        line = buf.decode("utf-8", "replace").strip()
        buf = b""
        if not line or "it/s]" in line or "s/it]" in line:
            continue
        oom = oom or any(k in line.lower() for k in OOM)
        m = re.search(r"\| epoch=(\d+) \|", line)
        if m:
            say(epoch=int(m.group(1)))
        print(line, flush=True)
    p.wait()
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
    ap.add_argument("--tools", default=str(Path(sys.executable).resolve().parent.parent / "tools"))
    a = ap.parse_args()
    if a.save_every <= 0:
        a.save_every = max(5, a.epochs // 20)

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
    g, d = TITAN / "G-f048k-TITAN-Medium.pth", TITAN / "D-f048k-TITAN-Medium.pth"
    if not (g.exists() and d.exists()):
        raise SystemExit("The base model files are missing from the Training Pack - reinstall it.")
    if run_dir.exists():
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
    step(2)
    run(["rvc/train/extract/extract.py", run_dir, "rmvpe", cores, "0", SR, "contentvec", "None", "2"])
    step(3)
    run(["rvc/train/train.py", a.name, a.save_every, a.epochs, g, d, "0", a.batch, SR, True, True, False, False,
         "HiFi-GAN", False])
    cps = checkpoints(run_dir, a.name)
    if not cps or cps[-1][0] < a.epochs:  # Applio's trainer exits 0 even when its worker crashed (e.g. out of memory)
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
    meta.update(source="mimicore-trainer", version="v2", epochs=best_epoch, trained_epochs=cps[-1][0])
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
