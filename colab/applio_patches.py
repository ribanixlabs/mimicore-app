r"""Small, idempotent patches to Applio so it runs from a read-only/packed install.

    python tools\applio_patches.py <applio dir>        (the build applies them to the pack copy; run on .\applio for dev)

1. Training runs go to APPLIO_LOGS_ROOT (our data root\runs) instead of <applio>\logs.
2. Absolute instead of cwd-relative paths in the training file list (works across drives).
3. Unicode-safe faiss index read/write (non-ASCII Windows user names).
4. pedalboard (GPLv3) is imported only when post-processing effects are requested (we never request them),
   so the Training Pack does not ship it.

Applio 3.6.6+ fixed 1 and 4 itself (a per-user config with an APPLIO_LOGS_DIR override, and a lazy pedalboard
import inside post_process_audio). Those patches are then simply skipped, so the same file works on the pinned
release and on newer checkouts. The unicode faiss patches (3) are REQUIRED: if their anchors move upstream, the
apply step must fail loudly rather than ship a pack that breaks non-ASCII Windows user names.
"""
import sys
from pathlib import Path

# (file, old, new, mode): mode "required" — a missing/changed anchor is a hard error because Mimicore still needs
# the fix; mode "legacy" — fixed differently upstream, skipping it is fine.
PATCHES = [
    ("rvc/train/process/extract_index.py",  # faiss opens files with narrow fopen: non-ASCII paths fail. Python IO.
     "    faiss.write_index(index_added, index_filepath_added)",
     "    with open(index_filepath_added, \"wb\") as _f:  # unicode-safe (faiss uses narrow fopen)\n"
     "        _f.write(faiss.serialize_index(index_added).tobytes())", "required"),
    ("rvc/infer/pipeline.py",  # the inference side reads the index back the same way
     "                index = faiss.read_index(file_index)",
     "                index = faiss.deserialize_index(np.fromfile(file_index, dtype=np.uint8))  # unicode-safe",
     "required"),
    ("rvc/train/train.py",  # logs on the data root (3.6.5-era layout; 3.6.6 honors APPLIO_LOGS_DIR itself)
     'experiment_dir = os.path.join(current_dir, "logs", model_name)',
     'experiment_dir = os.path.join(os.environ.get("APPLIO_LOGS_ROOT") or os.path.join(current_dir, "logs"), model_name)',
     "legacy"),
    ("rvc/train/train.py",
     'os.path.join(now_dir, "logs", model_name), topdown=False',
     'os.path.join(os.environ.get("APPLIO_LOGS_ROOT") or os.path.join(now_dir, "logs"), model_name), topdown=False',
     "legacy"),
    ("rvc/infer/infer.py",  # pedalboard (GPLv3): make the one inside post_process_audio self-contained first...
     "        board = Pedalboard()\n",
     "        from pedalboard import (Pedalboard, Chorus, Distortion, Reverb, PitchShift, Limiter, Gain, Bitcrush,\n"
     "                                Clipping, Compressor, Delay)\n\n        board = Pedalboard()\n", "lazy-board"),
    ("rvc/infer/infer.py",  # ...then drop the top-level import
     "from pedalboard import (\n    Pedalboard,\n    Chorus,\n    Distortion,\n    Reverb,\n    PitchShift,\n    Limiter,\n"
     "    Gain,\n    Bitcrush,\n    Clipping,\n    Compressor,\n    Delay,\n)\n",
     "# pedalboard (GPLv3) is imported lazily in post_process_audio - Mimicore never enables post-processing.\n",
     "legacy"),
]

# Exactly the 4-space-indented top-level import: in 3.6.6+ the same import lives, indented, inside the function.
TOP_PEDALBOARD = "from pedalboard import (\n    Pedalboard,"


def apply(applio_dir):
    root = Path(applio_dir)
    for rel, old, new, mode in PATCHES:
        f = root / rel
        s = f.read_text(encoding="utf-8")
        if new in s:
            continue
        if old in s:
            if mode == "lazy-board" and TOP_PEDALBOARD not in s:
                print(f"patch skipped - upstream already made pedalboard lazy: {rel}")
                continue
            f.write_text(s.replace(old, new, 1), encoding="utf-8")
        elif mode == "required":
            raise SystemExit(f"Applio patch does not apply ({rel}): upstream code changed")
        else:
            print(f"patch skipped - upstream handles this itself: {rel}")
    # relpath -> abspath in the training file list: 3.6.6+ writes absolute paths itself.
    f = root / "rvc/train/extract/preparing_files.py"
    s = f.read_text(encoding="utf-8")
    if "os.path.relpath(" in s:
        f.write_text(s.replace("os.path.relpath(", "os.path.abspath("), encoding="utf-8")
    return root


if __name__ == "__main__":
    print("patched", apply(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "applio"))
