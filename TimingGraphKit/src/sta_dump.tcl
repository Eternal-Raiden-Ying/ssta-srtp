# ========== OpenSTA: 稳健导出关键路径到 path_sel.txt ==========

# 1) 基本加载（按你的路径）
set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -distance um
read_liberty -min /root/autodl-tmp/OpenROAD-flow-scripts/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__ff_n40C_1v95.lib
read_liberty -max /root/autodl-tmp/OpenROAD-flow-scripts/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__ss_100C_1v60.lib
read_verilog /root/autodl-tmp/OpenROAD-flow-scripts/flow/designs/sky130hd/jpeg_encoder/jpeg_encoder.synthesis_preroute.v
link_design jpeg_encoder
read_sdc  /root/autodl-tmp/OpenROAD-flow-scripts/flow/designs/sky130hd/jpeg_encoder/jpeg_encoder.cts_1.sdc
read_spef /root/autodl-tmp/OpenROAD-flow-scripts/flow/designs/sky130hd/jpeg_encoder/20-jpeg_encoder.spef

# 2) 可选：从 anchors.csv 取 anchor_at（第6列）
set ANCHOR_CSV "/root/autodl-tmp/TimingPredict/logs/anchors.csv"
set PRED_AT ""
if {[file exists $ANCHOR_CSV]} {
  set f [open $ANCHOR_CSV r]; set data [read $f]; close $f
  set lines [split $data "\n"]
  if {[llength $lines] > 1} {
    set last [lindex $lines end-1]
    if {$last eq ""} { set last [lindex $lines end-2] }
    set cols [split $last ","]
    if {[llength $cols] >= 6} { set PRED_AT [lindex $cols 5] }
  }
}

# 3) 抓“最差端点”列表 —— 轮训 3 种格式（JSON→END→默认）
proc get_endpoints_sorted {} {
  # 3.1 JSON（优先，最容易 parse）
  set endpoints {}
  set chk_json ""
  catch {
    with_output_to_variable chk_json {
      report_checks -path_delay max -sort_by_slack \
        -group_path_count 1 -endpoint_path_count 200 \
        -format json -digits 5 -no_line_splits
    }
  }
  if {$chk_json ne ""} {
    # 抓所有 "endpoint":"inst/pin"
    foreach {full ep} [regexp -all -inline {"endpoint"\s*:\s*"([^"]+)"} $chk_json] {
      lappend endpoints $ep
    }
    if {[llength $endpoints] > 0} { return $endpoints }
    # 落盘备查
    set f [open checks_json_dump.txt w]; puts $f $chk_json; close $f
  }

  # 3.2 END：每行通常直接出现 inst/pin；没有 "Endpoint:" 前缀
  set chk_end ""
  catch {
    with_output_to_variable chk_end {
      report_checks -path_delay max -sort_by_slack \
        -endpoint_path_count 200 -format end -digits 5 -no_line_splits
    }
  }
  if {$chk_end ne ""} {
    foreach line [split $chk_end "\n"] {
      # 抓第一个看起来像 inst/pin 的片段
      if {[regexp {([A-Za-z0-9_./\[\]-]+/[A-Za-z0-9_./\[\]-]+)} $line -> ep]} {
        lappend endpoints $ep
      }
    }
    if {[llength $endpoints] > 0} { return $endpoints }
    set f [open checks_end_dump.txt w]; puts $f $chk_end; close $f
  }

  # 3.3 默认文本：有 "Endpoint: <pin>" 行
  set chk_txt ""
  catch {
    with_output_to_variable chk_txt {
      report_checks -path_delay max -sort_by_slack -digits 5
    }
  }
  if {$chk_txt ne ""} {
    foreach line [split $chk_txt "\n"] {
      if {[regexp {^Endpoint:\s+(\S+)} $line -> ep]} {
        lappend endpoints $ep
      }
    }
    if {[llength $endpoints] > 0} { return $endpoints }
    set f [open checks_txt_dump.txt w]; puts $f $chk_txt; close $f
  }

  return {}
}

set endpoints [get_endpoints_sorted]
if {[llength $endpoints] == 0} {
  puts "FATAL: 仍未解析到端点；已把原始输出写入 checks_*_dump.txt。请把其中一个文件前50行发我。"
  exit 1
}

# 4) 选择端点：默认第一名；若给了 PRED_AT，则在前K个里做就近匹配
set sel_pin [lindex $endpoints 0]
if {$PRED_AT ne ""} {
  set K [expr {min(200, [llength $endpoints])}]
  set best_pin ""; set best_diff 1e99
  for {set i 0} {$i < $K} {incr i} {
    set ep [lindex $endpoints $i]
    set arr ""
    catch {
      with_output_to_variable ARR { report_arrival $ep }  ;# 你的版本支持 report_arrival
    }
    if {[info exists ARR]} {
      # 抓第一个浮点数作为到达时间
      if {[regexp {(-?\d+\.\d+)} $ARR -> arr]} {
        set d [expr {abs($arr - $PRED_AT)}]
        if {$d < $best_diff} { set best_diff $d; set best_pin $ep }
      }
      unset ARR
    }
  }
  if {$best_pin ne ""} { set sel_pin $best_pin }
}
puts "Selected endpoint: $sel_pin  (anchor_at=$PRED_AT)"

# 5) 导出该端点的一条完整路径
#    用 report_checks -to ... -format full（你的指令集明确支持）:contentReference[oaicite:2]{index=2}
report_checks -to $sel_pin -path_delay max -format full -digits 5 -no_line_splits > path_sel.txt

# 6) 顺手把该端点的 arrival/slack 打印出来核对
with_output_to_variable ARR2 { report_arrival $sel_pin }
with_output_to_variable SLK2 { report_slack   $sel_pin }
puts "Endpoint arrival ≈ $ARR2"
puts "Endpoint slack   ≈ $SLK2"

exit
