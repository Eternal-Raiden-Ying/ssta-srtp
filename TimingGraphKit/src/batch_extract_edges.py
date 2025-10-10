#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, csv, time, signal, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- 基本路径 ----
OR_BIN   = Path("/root/autodl-tmp/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad")
FLOW_DIR = Path(__file__).resolve().parent

PLATFORM   = "sky130hs"
LEF_MERGED = FLOW_DIR / "platforms/sky130hs/lef/sky130_fd_sc_hs_merged.lef"
TECHLEF    = FLOW_DIR / "platforms/sky130hs/lef/sky130_fd_sc_hs.tlef"
LIB_LATE   = FLOW_DIR / "platforms/sky130hs/lib/sky130_fd_sc_hs__ss_100C_1v60.lib"
LIB_EARLY  = FLOW_DIR / "platforms/sky130hs/lib/sky130_fd_sc_hs__ff_n40C_1v95.lib"

DESIGNS = [f"b{str(i).zfill(2)}" for i in range(1, 23)]  # b01..b22
LOGROOT = FLOW_DIR / "edge_batch_logs"
LOGROOT.mkdir(parents=True, exist_ok=True)

# ---- 并发/看门狗参数（按机子调）----
MAX_WORKERS  = 6        # 并发度，建议别太大（OpenDB/磁盘会是瓶颈）
HARD_TIMEOUT = 1800     # 强超时：s（进程组 SIGTERM→SIGKILL）
IDLE_TIMEOUT = 300      # 日志“静默超时”：s（无新日志即视为卡死）
RETRY_TIMES  = 1        # 失败重试次数（每个 design）

def _launch_openroad(cmd, log_path: Path, cwd: Path):
    """
    启动 openroad 进程（置于新进程组），stdout/err 写入 log。
    返回 (Popen, log_file_handle) —— 注意调用方负责关闭句柄。
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # 让 Python 端 stdout 及时刷新
    # 直接打开文件句柄，避免 stdout 管道阻塞
    f = open(log_path, "w")
    p = subprocess.Popen(
        cmd,
        stdout=f, stderr=subprocess.STDOUT,
        cwd=str(cwd), env=env,
        preexec_fn=os.setsid   # Linux: 新建进程组，便于整个组强杀
    )
    return p, f

def _wait_with_watchdogs(p: subprocess.Popen, logf: Path, t0: float):
    """
    等待子进程结束；若超出 HARD_TIMEOUT 或日志长时间无更新（IDLE_TIMEOUT），则强杀。
    返回 (status_str, elapsed_seconds)
    """
    last_update = t0
    while True:
        rc = p.poll()
        now = time.time()

        # 日志更新时间
        try:
            mtime = logf.stat().st_mtime
        except FileNotFoundError:
            mtime = now
        if mtime > last_update:
            last_update = mtime

        # 正常结束
        if rc is not None:
            return ("OK" if rc == 0 else f"RC{rc}"), now - t0

        # 强超时
        if now - t0 > HARD_TIMEOUT:
            os.killpg(p.pid, signal.SIGTERM); time.sleep(5)
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
            return "TIMEOUT", now - t0

        # 日志静默超时（大多数“CPU 空转卡住”都能靠这个踢掉）
        if now - last_update > IDLE_TIMEOUT:
            os.killpg(p.pid, signal.SIGTERM); time.sleep(5)
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
            return "IDLE_KILL", now - t0

        time.sleep(1.0)

def run_one(design: str):
    def_path = FLOW_DIR / f"results/{PLATFORM}/{design}/base/6_final.def"
    outdir   = FLOW_DIR / f"edge_feats/{PLATFORM}/{design}"
    outdir.mkdir(parents=True, exist_ok=True)
    logf     = LOGROOT / f"{design}.log"

    if not def_path.exists():
        logf.write_text(f"DEF missing: {def_path}\n")
        return design, "MISS_DEF", 0.0

    net_csv  = outdir / "net_edges.csv"
    cell_csv = outdir / "cell_edges.csv"
    if net_csv.exists() and net_csv.stat().st_size > 0 and \
       cell_csv.exists() and cell_csv.stat().st_size > 0:
        return design, "SKIP", 0.0

    # —— 关键改动：加 -exit，避免 openroad 悬挂在交互模式 ——
    cmd = [
        str(OR_BIN), "-exit", "-python", "extract_edge_features.py",
        "--lef",      str(LEF_MERGED),
        "--techlef",  str(TECHLEF),
        "--def",      str(def_path),
        "--lib_late", str(LIB_LATE),
        "--lib_early",str(LIB_EARLY),
        "--outdir",   str(outdir),
        # 体量控制：可按需追加参数（见 extract 的 CLI）
        "--max-sinks", "2000",          # 丢弃扇出>2000的巨网（可按需调整/删除）
        # "--no-skip-pg",               # 如需保留 PG 网则解开
        # "--skip-net-regex", "(?i)^reset$"  # 额外过滤示例
    ]

    # 可重试
    tries = 0
    while True:
        tries += 1
        t0 = time.time()
        p, fh = _launch_openroad(cmd, logf, FLOW_DIR)
        try:
            status, elapsed = _wait_with_watchdogs(p, logf, t0)
        finally:
            try: fh.close()
            except: pass

        # 如果生成了产物，标 OK（有些 RC!=0 但文件完整）
        if status != "OK":
            if net_csv.exists() and cell_csv.exists() and \
               net_csv.stat().st_size > 0 and cell_csv.stat().st_size > 0:
                status = "OK*"

        if status == "OK" or tries > RETRY_TIMES:
            return design, status, elapsed
        else:
            time.sleep(2.0)  # 间隔后重试一次

def main():
    rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(run_one, d) for d in DESIGNS]
        for fu in as_completed(futures):
            d, st, el = fu.result()
            print(f"{d:>3}: {st:9} {el:7.1f}s")
            rows.append((d, st, f"{el:.1f}s"))

    with open(LOGROOT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["design", "status", "elapsed"])
        for r in sorted(rows, key=lambda x: x[0]):
            w.writerow(r)

    print(f"\nLogs dir : {LOGROOT}")
    print(f"Tail all : tail -F -v {LOGROOT}/b*.log")
    print(f"Summary  : {LOGROOT}/summary.csv")

if __name__ == "__main__":
    main()
