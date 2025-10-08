#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pinpos_to_csv.py
Usage:
  python pinpos_to_csv.py /path/to/config.json /path/to/output.csv
Minimal config.json needs:
{
  "lef_input": ["<tech.lef>", "<cell.lef>"],
  "def_input": "<design.def>"
}
"""
import sys, csv, math
import numpy as np
from dreamplace.ops.pin_pos.pin_pos import PinPos
def main():
    if len(sys.argv) < 3:
        print("Usage: python pinpos_to_csv.py <config.json> <out.csv>")
        sys.exit(2)

    cfg = sys.argv[1]
    out_csv = sys.argv[2]

    # Lazy imports so the file can be opened even without DreamPlace present
    import torch
    from dreamplace.Params import Params
    from dreamplace.PlaceDB import PlaceDB

    params = Params()
    params.load(sys.argv[1])  # only read the DEF/LEF file in this json file
    placedb = PlaceDB()
    placedb(params)

    pin_offset_x = torch.from_numpy(placedb.pin_offset_x).to(torch.float).cuda()
    pin_offset_y = torch.from_numpy(placedb.pin_offset_y).to(torch.float).cuda()
    pin2node_map = torch.from_numpy(placedb.pin2node_map).to(torch.long).cuda()
    flat_node2pin_map = torch.from_numpy(placedb.flat_node2pin_map).to(torch.int).cuda()
    flat_node2pin_start_map = torch.from_numpy(placedb.flat_node2pin_start_map).to(torch.int).cuda()

    pin_pos_instance = PinPos(
        pin_offset_x=pin_offset_x,
        pin_offset_y=pin_offset_y,
        pin2node_map=pin2node_map,
        flat_node2pin_map=flat_node2pin_map,
        flat_node2pin_start_map=flat_node2pin_start_map,
        num_physical_nodes=placedb.num_physical_nodes,
    )
    pin_pos_instance = pin_pos_instance.cuda()
    pos = torch.cat([torch.from_numpy(placedb.node_x).to(torch.float), 
                     torch.from_numpy(placedb.node_y).to(torch.float)]).cuda()
    pin_pos = pin_pos_instance(pos)
    
    # only use for checking the pin locations
    pin_pos = np.array(pin_pos.cpu())
    pin_x = pin_pos[:placedb.num_pins]
    pin_y = pin_pos[placedb.num_pins:]

    # Die area bounds
    def first_existing(*names):
        for n in names:
            if hasattr(placedb, n):
                return getattr(placedb, n)
        return None

    xl = first_existing("xl", "die_xl", "dieLeft", "die_lx")
    yl = first_existing("yl", "die_yl", "dieBottom", "die_ly")
    xh = first_existing("xh", "die_xh", "dieRight", "die_ux")
    yh = first_existing("yh", "die_yh", "dieTop", "die_uy")
    if None in (xl, yl, xh, yh):
        # Try DEF DIEAREA from raw text if exposed
        try:
            diearea = getattr_any(placedb, ["die_area"], default=None)
            if diearea:
                xl, yl, xh, yh = diearea
        except Exception:
            pass
    if None in (xl, yl, xh, yh):
        raise RuntimeError("Cannot determine die area bounds (xl,yl,xh,yh) from PlaceDB.")

    # Distances to edges
    # breakpoint()
    dist_left   = [px - xl for px in pin_x]
    dist_right  = [xh - px for px in pin_x]
    dist_bottom = [py - yl for py in pin_y]
    dist_top    = [yh - py for py in pin_y]

    def name_str_process(raw_str):
        raw_str = raw_str.decode("utf-8")
        return "".join(raw_str.split('\\'))

    # Optional: map pin to instance name (some versions expose pin_inst_map or use pin2node)
    rows = []
    num_pins = placedb.num_pins
    pin2node = placedb.pin2node_map
    pin_names = placedb.pin_names
    node_names = placedb.node_names
    for pid in range(num_pins):
        nid = pin2node[pid] if isinstance(pin2node, (list, tuple)) else int(pin2node[pid])
        inst_name = node_names[nid] if nid < len(node_names) else f"inst_{nid}"
        rows.append(
            [name_str_process(inst_name),
             name_str_process(pin_names[pid]),
             pin_x[pid], pin_y[pid],
             dist_left[pid], dist_right[pid],
             dist_bottom[pid], dist_top[pid]]
           )

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["inst", "pin", "x", "y",
                    "dist_left", "dist_right", "dist_bottom", "dist_top"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} pins -> {out_csv}")
    print(f"Die bounds: xl={placedb.rawdb.xl()}, yl={placedb.rawdb.yl()}, xh={placedb.rawdb.xh()}, yh={placedb.rawdb.yh()}")
    print(f"shift and scaled res xl:{placedb.xl}, yl:{placedb.yl}, xh:{placedb.xh}, yh:{placedb.yh}")


if __name__ == "__main__":
    main()
