#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os, re, sys
from collections import defaultdict
import pandas as pd

print(f"[EDGE-EXTRACT v1.2] running: {__file__}")

# ----------------------- OpenDB helpers -----------------------
def _tok(x):
    try: return x.name
    except Exception: return str(x)

def _io_token(io):
    s=_tok(io).upper().replace("DBIOTYPE::","")
    if "OUTPUT" in s or s=="OUT": return "OUTPUT"
    if "INPUT"  in s or s=="IN":  return "INPUT"
    if "INOUT"  in s:             return "INOUT"
    return s

def _sig_token(sig):
    s=_tok(sig).upper().replace("DBSIGTYPE::","")
    return s  # POWER/GROUND/SIGNAL/CLOCK/...

def load_pin_xy_and_nets(lefs, techlef, deffile):
    try:
        from openroad import Tech, Design
        import odb
    except Exception as e:
        raise RuntimeError("需要在 openroad -python 环境下运行，并能 import openroad, odb") from e

    tech = Tech()
    if techlef: tech.readLef(techlef)
    for lef in lefs: tech.readLef(lef)

    design = Design(tech)
    design.readDef(deffile)
    block = design.getBlock()
    if not block: raise RuntimeError(f"读取 DEF 失败：{deffile}")

    dbu = float(block.getDbUnitsPerMicron())

    # pin xy
    pin_xy = {}
    for bterm in block.getBTerms():
        name = bterm.getName()
        bpin = next(iter(bterm.getBPins() or []), None)
        if bpin:
            try:
                bbox = bpin.getBBox(); x,y = bbox.xMin(), bbox.yMin()
            except Exception:
                x=y=0
            pin_xy[("TOP", name)] = (x,y)
    for iterm in block.getITerms():
        inst = iterm.getInst().getName()
        mterm= iterm.getMTerm().getName()
        try:
            x,y = iterm.getAvgXY()
        except Exception:
            bbox = iterm.getBBox(); x,y = bbox.xMin(), bbox.yMin()
        pin_xy[(inst, mterm)] = (x,y)

    # nets and sigtypes
    net_conns = defaultdict(list)
    net_sigtypes = {}
    for net in block.getNets():
        n = net.getName()
        net_sigtypes[n] = _sig_token(net.getSigType())
        for it in net.getITerms():
            inst = it.getInst().getName()
            pin  = it.getMTerm().getName()
            io   = _io_token(it.getMTerm().getIoType())
            net_conns[n].append((inst, pin, io))
        for bt in net.getBTerms():
            name = bt.getName()
            io   = _io_token(bt.getIoType())
            net_conns[n].append(("TOP", name, io))

    inst_masters = {}
    for inst in block.getInsts():
        try: master = inst.getMaster().getName()
        except: master = ""
        inst_masters[inst.getName()] = master

    die = block.getDieArea()
    die_w = (die.xMax()-die.xMin())/dbu
    die_h = (die.yMax()-die.yMin())/dbu
    print(f"[INFO] DBU/um={dbu:.3f}, DIE={die_w:.2f}um x {die_h:.2f}um, nets={len(net_conns)}")
    return pin_xy, net_conns, dbu, inst_masters, net_sigtypes

# ----------------------- Liberty parsing -----------------------
_num_re = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

def _grab_block(s, start_idx):
    depth=0; i=start_idx
    while i < len(s):
        if s[i]=='{': depth+=1
        elif s[i]=='}':
            depth-=1
            if depth==0: return s[start_idx:i+1], i+1
        i+=1
    return None, i

def parse_liberty_simple(path):
    def _clean(s): return s.strip().strip('"').strip("'")
    txt=open(path,'r',encoding='utf-8',errors='ignore').read()
    txt=re.sub(r"/\*.*?\*/","",txt,flags=re.S)
    txt=re.sub(r"//.*?$","",txt,flags=re.M)

    templates={}
    for m in re.finditer(r"\blu_table_template\s*\(\s*([^)]+)\s*\)\s*{", txt):
        tname=_clean(m.group(1)); blk,_=_grab_block(txt, m.end()-1)
        def _nums(tag):
            mm=re.search(tag+r"\s*\(\s*([^)]+)\s*\)\s*;", blk, flags=re.S)
            if not mm: return []
            raw=mm.group(1).replace("\\"," ").replace("\n"," ")
            return [float(x) for x in re.findall(_num_re, raw)]
        templates[tname]={"index_1":_nums("index_1"),"index_2":_nums("index_2")}

    lib={}
    for cm in re.finditer(r"\bcell\s*\(\s*([^)]+)\s*\)\s*{", txt):
        cell=_clean(cm.group(1)); cblk,_=_grab_block(txt, cm.end()-1)
        cell_db={}
        for pm in re.finditer(r"\bpin\s*\(\s*([^)]+)\s*\)\s*{", cblk):
            to_pin=_clean(pm.group(1)); pblk,_=_grab_block(cblk, pm.end()-1)
            for tm in re.finditer(r"\btiming\s*\(\s*\)\s*{", pblk):
                tblk,_=_grab_block(pblk, tm.end()-1)
                rel=re.search(r"\brelated_pin\s*:\s*\"?([^\";]+)\"?\s*;", tblk)
                if not rel: continue
                frm=_clean(rel.group(1)); key=(frm,to_pin)
                lut={}
                for kind in ["cell_rise","cell_fall","rise_transition","fall_transition"]:
                    km=re.search(r"\b"+kind+r"\s*(\([^)]*\))?\s*{", tblk)
                    if not km: lut[kind]=None; continue
                    kblk,_=_grab_block(tblk, km.end()-1)
                    paren=km.group(1) or ""; tname=None
                    if paren:
                        mt=re.search(r"\btemplate\s*:\s*\"?([^\",)]+)\"?", paren)
                        if mt: tname=_clean(mt.group(1))
                        else:
                            cand=_clean(paren.strip("()").split(",")[0])
                            if cand and not re.fullmatch(_num_re, cand): tname=cand
                    def _nums_in(block, tag):
                        mm=re.search(tag+r"\s*\(\s*([^)]+)\s*\)\s*;", block, flags=re.S)
                        if not mm: return []
                        raw=mm.group(1).replace("\\"," ").replace("\n"," ")
                        return [float(x) for x in re.findall(_num_re, raw)]
                    idx1=_nums_in(kblk,"index_1"); idx2=_nums_in(kblk,"index_2")
                    if (not idx1 or not idx2) and tname and tname in templates:
                        if not idx1: idx1=templates[tname].get("index_1",[])
                        if not idx2: idx2=templates[tname].get("index_2",[])
                    vals=[]
                    for vm in re.finditer(r"\bvalues\s*\(\s*([^;]*?)\s*\)\s*;", kblk, flags=re.S):
                        raw=vm.group(1).replace("\\"," ").replace("\n"," ").replace("\""," ")
                        vals += [float(x) for x in re.findall(_num_re, raw)]
                    lut[kind]={"valid":1 if (idx1 and idx2 and vals) else 0,
                               "index_1":idx1 or [], "index_2":idx2 or [],
                               "values": vals or [], "rows": len(idx1) if idx1 else None,
                               "cols": len(idx2) if idx2 else None}
                cell_db[key]=lut
        lib[cell]=cell_db
    return lib

def _pack_list(lst): return " ".join(str(x) for x in lst) if lst else ""

def _emit_lut(row, pref, lut):
    if lut and lut.get("valid",0)==1:
        row[f"{pref}_valid"]=1
        row[f"{pref}_index1"]=_pack_list(lut["index_1"])
        row[f"{pref}_index2"]=_pack_list(lut["index_2"])
        row[f"{pref}_values"]=_pack_list(lut["values"])
    else:
        row[f"{pref}_valid"]=0; row[f"{pref}_index1"]=""; row[f"{pref}_index2"]=""; row[f"{pref}_values"]=""

def write_cell_edges_csv(path, arcs, lib_early, lib_late):
    rows=[]; eid=0
    for a in arcs:
        if a.get("type")!="cell": continue
        inst=a["inst"]; cell=a["cell"]; frm=a["from"]; to=a["to"]
        row={"edge_id":eid,"inst":inst,"cell":cell,"from_pin":frm,"to_pin":to}
        E=lib_early.get(cell,{}).get((frm,to), None)
        L=lib_late .get(cell,{}).get((frm,to), None)
        mapping=[
            ("late_delay_rise",   L,"cell_rise"),
            ("late_delay_fall",   L,"cell_fall"),
            ("late_slew_rise",    L,"rise_transition"),
            ("late_slew_fall",    L,"fall_transition"),
            ("early_delay_rise",  E,"cell_rise"),
            ("early_delay_fall",  E,"cell_fall"),
            ("early_slew_rise",   E,"rise_transition"),
            ("early_slew_fall",   E,"fall_transition"),
        ]
        for pref,librec,kind in mapping:
            _emit_lut(row, pref, librec.get(kind) if librec else None)
        rows.append(row); eid+=1
    df=pd.DataFrame(rows)
    if not df.empty: df["edge_id"]=range(len(df))
    df.to_csv(path, index=False)

def build_arcs_auto(inst_masters, lib_any, emit_json_path=None):
    arcs=[]; hit=miss=0
    for inst,cell in inst_masters.items():
        if not cell: miss+=1; continue
        arcmap=lib_any.get(cell)
        if not arcmap: miss+=1; continue
        hit+=1
        for (frm,to) in arcmap.keys():
            arcs.append({"type":"cell","inst":inst,"cell":cell,"from":frm,"to":to})
    if emit_json_path:
        os.makedirs(os.path.dirname(emit_json_path), exist_ok=True)
        with open(emit_json_path,"w") as f: json.dump(arcs,f,indent=2)
    print(f"[AUTO-ARCS] instances matched={hit}, missed={miss}, arcs={len(arcs)}")
    return arcs

# ----------------------- Net edges with filtering & max-sinks -----------------------
_DEFAULT_SKIP_REGEX = (
    r"(?i)^(VPWR|VGND|VNB|VPB|VDD|VSS|GND|VCC|VDDIO|VSSIO|AVDD|AVSS|VCCD\d*|VSSD\d*|VCCINT|VCCAUX|VREF|VBG)$"
)

def write_net_edges_csv(path, net_conns, pin_xy, dbu,
                        net_sigtypes,
                        skip_pg=True,
                        skip_regex_list=None,
                        max_sinks=0):
    """
    导出 driver->sink 边，过滤 POWER/GROUND 和黑名单，去自环去重，重排 edge_id。
    max_sinks>0 时，丢弃扇出超过该阈值的 net（避免巨网爆量）。
    """
    def _is_out(d): return str(d).upper() in ("OUT","OUTPUT")
    def _is_in(d):  return str(d).upper() in ("IN","INPUT")
    def _is_io(d):  return str(d).upper()=="INOUT"

    patterns=[re.compile(_DEFAULT_SKIP_REGEX)]
    if skip_regex_list:
        patterns += [re.compile(p) for p in skip_regex_list]
    def _skip_name(nm: str) -> bool:
        return any(p.search(nm) for p in patterns)

    rows=[]; eid=0
    n_all=len(net_conns); n_skipped=0; n_bigfan=0

    for net, terms in net_conns.items():
        # 1) 按 sigtype 过滤
        sigt=net_sigtypes.get(net,"SIGNAL")
        if skip_pg and sigt in ("POWER","GROUND"):
            n_skipped += 1; continue
        # 2) 名称黑名单
        if _skip_name(net):
            n_skipped += 1; continue

        drivers=[]; sinks=[]
        for inst,pin,d in terms:
            if inst=="TOP":
                if _is_in(d): drivers.append((inst,pin,d))      # PI=driver
                elif _is_out(d): sinks.append((inst,pin,d))     # PO=sink
                elif _is_io(d): drivers.append((inst,pin,d)); sinks.append((inst,pin,d))
            else:
                if _is_out(d): drivers.append((inst,pin,d))
                elif _is_in(d): sinks.append((inst,pin,d))
                elif _is_io(d): drivers.append((inst,pin,d)); sinks.append((inst,pin,d))

        if not drivers or not sinks:
            continue

        if max_sinks and len(sinks) > max_sinks:
            n_bigfan += 1
            continue

        seen=set()
        for di,dp,_ in drivers:
            x0=y0=None
            if (di,dp) in pin_xy: x0,y0=pin_xy[(di,dp)]
            for si,sp,_ in sinks:
                if di==si and dp==sp: continue
                key=(di,dp,si,sp)
                if key in seen: continue
                seen.add(key)

                x1=y1=None
                if (si,sp) in pin_xy: x1,y1=pin_xy[(si,sp)]
                if x0 is None or x1 is None:
                    dx=dy=0.0
                else:
                    dx=float(x1)-float(x0); dy=float(y1)-float(y0)

                rows.append({
                    "edge_id": eid,
                    "net": net,
                    "driver_inst": di, "driver_pin": dp,
                    "sink_inst": si,   "sink_pin":  sp,
                    "dx": dx, "dy": dy,
                    "dx_um": dx/float(dbu), "dy_um": dy/float(dbu),
                })
                eid+=1

    df=pd.DataFrame(rows)
    if df.empty:
        print(f"[WARN] net_edges empty after filtering (nets={n_all}, skipped={n_skipped}, bigfan={n_bigfan}).")
    else:
        df["edge_id"]=range(len(df))
        print(f"[NET] nets={n_all}, skipped={n_skipped}, bigfan_drop={n_bigfan}, edges={len(df)}")
    df.to_csv(path, index=False)

# ----------------------- CLI -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lef", nargs="+", required=True)
    ap.add_argument("--techlef", default=None)
    ap.add_argument("--def", dest="deffile", required=True)
    ap.add_argument("--lib_late", required=True)
    ap.add_argument("--lib_early", required=False)
    ap.add_argument("--arcs_json", default=None)
    ap.add_argument("--outdir", default="edge_features_out")

    ap.add_argument("--no-skip-pg", action="store_true", help="不过滤 POWER/GROUND（默认过滤）")
    ap.add_argument("--skip-net-regex", nargs="*", default=None, help="附加黑名单正则")
    ap.add_argument("--max-sinks", type=int, default=0, help="丢弃扇出超过阈值的 net（0=不限）")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("1) 读取 LEF/DEF ...")
    pin_xy, net_conns, dbu, inst_masters, net_sigtypes = load_pin_xy_and_nets(
        args.lef, args.techlef, args.deffile
    )

    print("2) 导出 net_edges.csv ...")
    write_net_edges_csv(
        os.path.join(args.outdir,"net_edges.csv"),
        net_conns, pin_xy, dbu,
        net_sigtypes=net_sigtypes,
        skip_pg=(not args.no_skip_pg),
        skip_regex_list=args.skip_net_regex,
        max_sinks=args.max_sinks
    )

    print("3) 解析 Liberty ...")
    lib_late  = parse_liberty_simple(args.lib_late)
    print("[DEBUG] lib_late cells parsed:", len(lib_late))
    lib_early = parse_liberty_simple(args.lib_early) if args.lib_early else lib_late

    print("4) 读取/枚举 arcs ...")
    if args.arcs_json and os.path.exists(args.arcs_json):
        with open(args.arcs_json,"r") as f:
            arcs = json.load(f)
    else:
        auto_arcs_json = os.path.join(args.outdir, "arcs_auto.json")
        arcs = build_arcs_auto(inst_masters, lib_late, emit_json_path=auto_arcs_json)

    if arcs:
        print("5) 导出 cell_edges.csv ...")
        write_cell_edges_csv(os.path.join(args.outdir,"cell_edges.csv"), arcs, lib_early, lib_late)
    else:
        print("SKIP: 无 arcs。")

    print("完成。")

if __name__ == "__main__":
    main()
