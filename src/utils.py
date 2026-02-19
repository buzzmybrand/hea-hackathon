"""
utils.py
Utility functions: logging setup, memory helpers, plotting helpers, file save helpers.
"""
import os
import gc
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import psutil
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def get_mem_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

def force_cleanup(*vars_to_del, gc_gen: int = 2):
    for v in vars_to_del:
        try:
            del v
        except Exception:
            pass
    gc.collect(gc_gen)

def ensure_dirs(dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def save_show(fig, fname, out_dir="./plots", dpi=120):
    ensure_dirs([out_dir])
    path = Path(out_dir) / fname
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    log.info(f"Saved figure: {path}")
    return str(path)

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    log.info(f"Saved json: {path}")
    return path
