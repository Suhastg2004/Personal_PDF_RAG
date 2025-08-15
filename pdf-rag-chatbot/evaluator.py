# evaluator.py
import time
from typing import Dict, Any
import csv

def timed(fn):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = fn(*args, **kwargs)
        dt = time.time() - t0
        return out, dt
    return wrapper

def log_trace_csv(filename:str, row:Dict[str,Any]):
    header = list(row.keys())
    try:
        with open(filename, "x", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerow(row)
    except FileExistsError:
        with open(filename, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writerow(row)
