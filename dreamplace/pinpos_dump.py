# python3 pinpos_dump.py /path/to/config.json /path/to/out.csv
import sys, csv
import torch
from dreamplace.Parameters import Params
from dreamplace.PlaceDB import PlaceDB
from dreamplace.ops.pin_pos.pin_pos import PinPos  # 模块名是 pin_pos（不是命令行）

cfg = sys.argv[1]; out_csv = sys.argv[2]

# 1) 读参数与数据库（不会启动优化）
params = Params()
params.load(cfg)
placedb = PlaceDB()
placedb(params)   # 构建数据库

# 2) 组装实例位置 pos（长度=2*num_nodes，前半 x 后半 y）
node_x = torch.tensor(placedb.node_x, dtype=torch.float32)
node_y = torch.tensor(placedb.node_y, dtype=torch.float32)
pos = torch.cat([node_x, node_y], dim=0)

# 3) 准备 pin 偏移与映射
pin_offset_x = torch.tensor(placedb.pin_offset_x, dtype=torch.float32)
pin_offset_y = torch.tensor(placedb.pin_offset_y, dtype=torch.float32)
pin2node_map = torch.tensor(placedb.pin2node_map, dtype=torch.int32)
pin2node_start = torch.tensor(placedb.pin2node_start, dtype=torch.int32)

# 4) 计算所有引脚坐标
pin_pos_op = PinPos(pin_offset_x, pin_offset_y, pin2node_map, pin2node_start)
pin_xy = pin_pos_op(pos)   # 形状: [2 * num_pins]
num_pins = placedb.num_pins
pin_x = pin_xy[:num_pins].tolist()
pin_y = pin_xy[num_pins:].tolist()

# 5) 回填 pin 对应的 inst/pin 名字（按你的版本取合适的字段）
# 下面给两种常见来源，择其一或打印 placedb 字段确认：
pin_names = getattr(placedb, "pin_names", None)
inst_of_pin = getattr(placedb, "pin2node", None) or placedb.pin2node_map
inst_names = getattr(placedb, "node_names", None)

rows = []
for pid in range(num_pins):
    inst_id = inst_of_pin[pid]
    inst = inst_names[inst_id] if inst_names else f"inst_{inst_id}"
    pin  = pin_names[pid] if pin_names else f"pin_{pid}"
    rows.append([inst, pin, pin_x[pid], pin_y[pid]])

with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["inst", "pin", "x", "y"])
    w.writerows(rows)

print(f"wrote {len(rows)} pins to {out_csv}")
