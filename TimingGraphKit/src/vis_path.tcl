# ====== 改成你的 LEF/DEF ======
read_lef /root/autodl-tmp/OpenROAD-flow-scripts/flow/platforms/sky130hd/lef/sky130_fd_sc_hd.tlef
read_lef /root/autodl-tmp/OpenROAD-flow-scripts/flow/platforms/sky130hd/lef/sky130_fd_sc_hd_merged.lef
read_def /root/autodl-tmp/OpenROAD-flow-scripts/flow/designs/sky130hd/jpeg_encoder/20-jpeg_encoder.def
# ==================================

proc read_pins_list {pathfile} {
  if {![file exists $pathfile]} { puts "ERR: $pathfile not found"; return {} }
  set f [open $pathfile r]; set pins {}
  while {[gets $f L] >= 0} {
    set L [string trim $L]
    if {$L ne ""} { lappend pins $L }
  }
  close $f
  return $pins
}
proc _inst {p} { return [regsub {/.+$} $p ""] }

# 从 pinA 出发，在它的网里找是否有属于 next_inst 的 pin —— 返回 dbNet*（对象）
proc _dbnet_to_next {pinA next_inst} {
  set pa [get_pins -hier $pinA]
  if {$pa eq ""} { return "" }
  # next_inst 的所有 pin 对象
  set nextpins ""
  if {[llength [info commands get_instances]]} {
    set nextpins [get_pins -of_objects [get_instances $next_inst]]
  } else {
    set nextpins [get_pins -of_objects [get_cells $next_inst]]
  }
  foreach n [get_nets -of_objects $pa] {
    foreach q [get_pins -of_objects $n] {
      if {[lsearch -exact $nextpins $q] >= 0} { return $n }
    }
  }
  return ""
}

# 轻量可视化
proc _clear_vis {} { catch { gui::clear_focus_nets }; catch { gui::clear_highlights }; catch { gui::clear_selections } }
proc _mark_inst {inst {group 0}} { catch { gui::highlight_inst $inst $group } }
proc _focus_dbnet {dbnet} { if {$dbnet ne ""} { catch { gui::focus_net $dbnet } } }
proc _fit_view {} {
  if {[llength [info commands gui::zoom_to_selection]]} { gui::zoom_to_selection } \
  elseif {[llength [info commands gui::fit]]} { gui::fit }
}

proc highlight_path_strong {pins {group 0}} {
  _clear_vis
  foreach p $pins { _mark_inst [_inst $p] $group }
  set kept 0
  for {set i 0} {$i < [llength $pins]-1} {incr i} {
    set a [lindex $pins $i]; set b [lindex $pins [expr {$i+1}]]
    set n [_dbnet_to_next $a [_inst $b]]
    if {$n ne ""} { _focus_dbnet $n; incr kept }
  }
  _fit_view
  puts "highlighted: [llength $pins] pins, $kept nets."
}



# ====== 调用：给出你导出的 path_pins.txt 路径 ======
set PINS [read_pins_list "./sta_out/path_pins.txt"]
puts "DATA pins count=[llength $PINS]"
# 高亮路径（你已有）
highlight_path_strong $PINS 0

# —— 高亮（你已有）——
highlight_path_strong $PINS 0

# —— 自动缩放（100%可用）——
if {[llength [info commands gui::zoom_to_selection]]} {
  gui::zoom_to_selection
} elseif {[llength [info commands gui::fit]]} {
  gui::fit
}

# 可选：打标签、截图
catch {
  gui::add_label 0 0 "START: [regsub {/.+$} [lindex $PINS 0] ""]"
  gui::add_label 0 0 "END:   [regsub {/.+$} [lindex $PINS end] ""]"
}
catch { save_image critpath.png }

