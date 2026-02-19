"""
data_loader.py
Data loading helpers for HRS Stata files with memory reduction.
"""
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from .utils import log, get_mem_gb, force_cleanup

def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["integer"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df

def load_hrs_data(path: str, wanted_cols=None, chunk_rows=50000, max_chunks=6, mem_limit_gb=12.0):
    """
    Load Stata file by chunks, only keeping wanted columns, and reduce memory usage.
    """
    log.info(f"Loading HRS data from: {path}")
    iterator = pd.read_stata(path, iterator=True, convert_categoricals=False)
    all_cols = list(iterator.variable_labels().keys())
    iterator.close()
    log.info(f"Total columns in file: {len(all_cols)}")
    wanted = set([c.lower() for c in (wanted_cols or all_cols)])
    available = [c for c in all_cols if c.lower() in wanted]
    iterator = pd.read_stata(path, columns=available or None, iterator=True, convert_categoricals=False, chunksize=chunk_rows)
    chunks = []
    n_loaded = 0
    pbar = tqdm(iterator, desc="Loading DTA chunks", unit="chunk", total=max_chunks)
    for chunk in pbar:
        if n_loaded >= max_chunks:
            log.info("Reached max_chunks limit — stopping load.")
            break
        if get_mem_gb() >= mem_limit_gb:
            log.warning(f"Memory limit reached ({get_mem_gb():.2f} GB) — stopping load.")
            break
        chunk.columns = [c.lower() for c in chunk.columns]
        chunk = reduce_mem_usage(chunk)
        chunks.append(chunk)
        n_loaded += 1
    iterator.close()
    pbar.close()
    if not chunks:
        raise RuntimeError("No data loaded — check file path or memory.")
    df = pd.concat(chunks, ignore_index=True)
    force_cleanup(chunks)
    df = reduce_mem_usage(df)
    log.info(f"Loaded DataFrame shape: {df.shape}")
    return df
