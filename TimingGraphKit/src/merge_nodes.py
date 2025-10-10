# -*- coding: utf-8 -*-
"""
Merge node_features and pins_timing at pin-level (node_key = inst/pin or port_name).
Usage:
  python merge_nodes.py --pins /path/to/b01.pins_timing.csv --nodes /path/to/node_features.csv \
                        --out /path/to/merged_nodes_inner_filtered.csv \
                        [--keep-internal-ff]

Outputs:
  - merged CSV (inner join on node_key)
  - also writes side reports near the merged file:
      only_in_node_features_*.csv
      only_in_pins_timing_*.csv
"""
import argparse
import pandas as pd
from pathlib import Path

def normalize_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = s.replace("\\$", "$").replace("\\[", "[").replace("\\]", "]").replace("\\/", "/")
    s = " ".join(s.split())
    return s

def split_inst_pin_from_combined(val: str):
    val = normalize_name(val)
    if "/" in val:
        inst, pin = val.split("/", 1)
        return inst, pin
    return "", val  # top-level port

def build_node_key(inst: str, pin: str):
    inst = normalize_name(inst)
    pin = normalize_name(pin)
    return f"{inst}/{pin}" if inst else pin

def is_internal_ff_pin(node_key: str) -> bool:
    return isinstance(node_key, str) and (node_key.endswith("/IQ") or node_key.endswith("/IQ_N"))

def main(args):
    pins = pd.read_csv(args.pins)
    nodes = pd.read_csv(args.nodes)
    pins_std = pins.copy(); pins_std.columns = [c.lower() for c in pins_std.columns]
    nodes_std = nodes.copy(); nodes_std.columns = [c.lower() for c in nodes_std.columns]

    if "pin" not in pins_std.columns:
        raise RuntimeError("Expected 'pin' column in pins_timing CSV.")
    if not {"inst","pin"}.issubset(nodes_std.columns):
        raise RuntimeError("Expected 'inst' and 'pin' columns in node_features CSV.")

    pins_std["inst"], pins_std["pin_only"] = zip(*pins_std["pin"].map(split_inst_pin_from_combined))
    pins_std["node_key"] = pins_std.apply(lambda r: build_node_key(r["inst"], r["pin_only"]), axis=1)
    nodes_std["node_key"] = nodes_std.apply(lambda r: build_node_key(r["inst"], r["pin"]), axis=1)

    pins_filt = pins_std.copy()
    nodes_filt = nodes_std.copy()
    if not args.keep_internal_ff:
        pins_filt = pins_filt[~pins_filt["node_key"].map(is_internal_ff_pin)].copy()
        nodes_filt = nodes_filt[~nodes_filt["node_key"].map(is_internal_ff_pin)].copy()

    merged_inner = nodes_filt.merge(pins_filt, on="node_key", how="inner", suffixes=("_nodefeat", "_timing"))
    only_in_nodes = nodes_filt[~nodes_filt["node_key"].isin(pins_filt["node_key"])].copy()
    only_in_pins = pins_filt[~pins_filt["node_key"].isin(nodes_filt["node_key"])].copy()

    out_path = Path(args.out)
    out_dir = out_path.parent
    tag = "full" if args.keep_internal_ff else "filtered"
    merged_path = out_path if out_path.suffix == ".csv" else out_dir / f"merged_nodes_inner_{tag}.csv"
    only_nodes_path = out_dir / f"only_in_node_features_{tag}.csv"
    only_pins_path = out_dir / f"only_in_pins_timing_{tag}.csv"

    merged_inner.to_csv(merged_path, index=False)
    only_in_nodes.to_csv(only_nodes_path, index=False)
    only_in_pins.to_csv(only_pins_path, index=False)

    print({
        "pins_rows_total": int(len(pins_std)),
        "nodes_rows_total": int(len(nodes_std)),
        "merged_inner_rows": int(len(merged_inner)),
        "only_in_nodes_rows": int(len(only_in_nodes)),
        "only_in_pins_rows": int(len(only_in_pins)),
        "merged_output": str(merged_path),
        "only_in_nodes_output": str(only_nodes_path),
        "only_in_pins_output": str(only_pins_path),
        "KEEP_INTERNAL_FF_PINS": bool(args.keep_internal_ff),
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pins", required=True, help="Path to b01.pins_timing.csv")
    parser.add_argument("--nodes", required=True, help="Path to node_features.csv")
    parser.add_argument("--out", required=True, help="Output CSV path (or folder)")
    parser.add_argument("--keep-internal-ff", dest="keep_internal_ff", action="store_true",
                        help="Keep internal FF pins (IQ/IQ_N). Default: drop them.")
    args = parser.parse_args()
    main(args)
