# ================= helpers & env =================
proc _getenv {n d} { return [expr {[info exists ::env($n)] ? $::env($n) : $d}] }

# 安全 debug 输出：避免双引号/[]/$ 触发替换
set ::DBG_NET   [expr {[info exists ::env(DEBUG_NET)]   ? $::env(DEBUG_NET)   : 0}]
set ::DBG_LIMIT [expr {[info exists ::env(DEBUG_LIMIT)] ? $::env(DEBUG_LIMIT) : 10}]
set ::DBG_SHOWN 0
proc _dbg {args} {
  if {$::DBG_NET && $::DBG_SHOWN < $::DBG_LIMIT} {
    incr ::DBG_SHOWN
    puts [join $args " "]
  }
}

# 基本参数
set TOP        [_getenv "TOP" ""]
if {$TOP eq ""} { puts "FATAL: set TOP=<top_module>"; exit 1 }

set PLATFORM   [_getenv "PLATFORM" ""]
set DESIGN     [_getenv "DESIGN" $TOP]
set FLOW_ROOT  [_getenv "FLOW_ROOT" ""]
set VERILOG    [_getenv "VERILOG" ""]
set SDC        [_getenv "SDC" ""]
set SPEF       [_getenv "SPEF" ""]
set LIB_MIN    [_getenv "LIB_MIN" ""]
set LIB_MAX    [_getenv "LIB_MAX" ""]
set OUTDIR     [_getenv "OUTDIR" "./dump_out/$DESIGN"]
set AUTO_BIND  [_getenv "AUTO_BIND_CLOCK" "0"]
set CLK_PER    [_getenv "CLK_PERIOD" "10.0"]
set SDF_DIGITS [_getenv "SDF_DIGITS" "9"]
set NET_ROOT_ZERO [_getenv "NET_ROOT_ZERO" "1"]
set PATH_ACCUM    [_getenv "PATH_ACCUM"    "1"]
set HOP_PICK      [_getenv "HOP_PICK"      "max_wire"] ;# max_wire|first

file mkdir $OUTDIR
puts [format "CONF: DESIGN=%s TOP=%s NET_ROOT_ZERO=%s PATH_ACCUM=%s OUTDIR=%s" \
              $DESIGN $TOP $NET_ROOT_ZERO $PATH_ACCUM $OUTDIR]

if {$VERILOG eq ""} {
  if {$FLOW_ROOT eq "" || $PLATFORM eq ""} {
    puts "FATAL: set VERILOG or {FLOW_ROOT, PLATFORM}"; exit 1
  }
  set base "$FLOW_ROOT/results/$PLATFORM/$DESIGN/base"
  set VERILOG "$base/6_final.v"
  if {$SDC eq ""}  { set SDC  "$base/6_final.sdc" }
  if {$SPEF eq ""} { set SPEF "$base/6_final.spef" }
}

# ================= load design =================
if {$LIB_MIN ne ""} { read_liberty -min $LIB_MIN }
if {$LIB_MAX ne ""} { read_liberty -max $LIB_MAX }

read_verilog $VERILOG
link_design $TOP
if {$SDC ne "" && [file exists $SDC]}  { read_sdc  $SDC }
if {$SPEF ne "" && [file exists $SPEF]} { read_spef $SPEF }

catch { set_propagated_clock [all_clocks] }

# 自动绑钟（仅兜底）
if {$AUTO_BIND} {
  set has_src 0
  foreach c [all_clocks] { if {[llength [get_property $c sources]]>0} { set has_src 1; break } }
  if {!$has_src} {
    set ck [all_registers -clock_pins]
    if {[llength $ck]>0} {
      create_clock -name AUTO_CLK -period $CLK_PER $ck
      catch { set_propagated_clock [get_clocks AUTO_CLK] }
      puts [format "INFO: AUTO_BIND_CLOCK on %d pins, period=%.1f ns" [llength $ck] $CLK_PER]
    }
  }
}

# ================= write augmented SDF =================
set sdf_file [file join $OUTDIR "${DESIGN}.aug.sdf"]
write_sdf -digits $SDF_DIGITS $sdf_file

# quick check
set has_arrivals 0
set fh0 [open $sdf_file r]
while {[gets $fh0 line] >= 0} {
  if {[string first "(ARRIVALTIMES" $line] >= 0} { set has_arrivals 1; break }
}
close $fh0
if {!$has_arrivals} {
  puts "FATAL: This sta build didn't emit (ARRIVALTIMES ...). Are you running OpenSTADump?"; exit 1
}

# ================= name helpers =================
proc _unescape_sdf_name {s} {
  set t $s
  regsub -all {\\\.} $t {.} t
  regsub -all {\\\$} $t {$} t
  regsub -all {\\\\} $t {\\} t
  return $t
}

# 先尝试端口，再尝试 pin，减少对顶层端口的“pin not found”告警
proc _canon_pin_full {name_in} {
  set candidates [list [_unescape_sdf_name $name_in] $name_in]
  foreach cand $candidates {
    # port
    set pobj ""; catch { set pobj [get_ports $cand] }
    if {$pobj ne ""} {
      set pfull ""; catch { set pfull [get_property $pobj full_name] }
      if {$pfull eq ""} { catch { set pfull [get_property $pobj name] } }
      if {$pfull ne ""} { return [list $pobj $pfull] }
    }
    # pin
    set obj ""; catch { set obj [get_pins $cand] }
    if {$obj ne ""} {
      set full ""; catch { set full [get_property $obj full_name] }
      if {$full eq ""} { catch { set full [get_property $obj name] } }
      if {$full ne ""} { return [list $obj $full] }
    }
  }
  return [list "" $name_in]
}

proc _strip_top {name} {
  if {$::TOP eq ""} { return $name }
  set prefix "$::TOP/"
  if {[string first $prefix $name] == 0} {
    return [string range $name [string length $prefix] end]
  }
  return $name
}

# ================= parse AT / SLEW / RAT =================
array set AT  {}
array set SLW {}
array set RAT {}

set NUM {[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?}
proc _parse_tuple_line {kind line num retname} {
  upvar 1 $retname R
  set fmt {^\s*\(%s\s+(\S+)\s+\(\s*(%s)\s*::\s*(%s)\s*\)\s+\(\s*(%s)\s*::\s*(%s)\s*\)\s*\)$}
  set pattern [format $fmt $kind $num $num $num $num]
  set m [regexp -inline -- $pattern $line]
  if {$m eq ""} { return 0 }
  set R(pin) [lindex $m 1]
  set R(er)  [lindex $m 2]
  set R(lr)  [lindex $m 3]
  set R(ef)  [lindex $m 4]
  set R(lf)  [lindex $m 5]
  return 1
}

set fh1 [open $sdf_file r]
set nAT 0; set nSL 0; set nRT 0
while {[gets $fh1 raw] >= 0} {
  set line [string trim $raw]
  if {[string first "(AT " $line] == 0} {
    array unset R
    if {[_parse_tuple_line "AT" $line $NUM R]} { set AT($R(pin))  [list $R(er) $R(ef) $R(lr) $R(lf)]; incr nAT }
  } elseif {[string first "(SLEW " $line] == 0} {
    array unset R
    if {[_parse_tuple_line "SLEW" $line $NUM R]} { set SLW($R(pin)) [list $R(er) $R(ef) $R(lr) $R(lf)]; incr nSL }
  } elseif {[string first "(RAT " $line] == 0} {
    array unset R
    if {[_parse_tuple_line "RAT" $line $NUM R]} { set RAT($R(pin)) [list $R(er) $R(ef) $R(lr) $R(lf)]; incr nRT }
  }
}
close $fh1
puts [format "INFO: parsed AT=%d SLEW=%d RAT=%d" $nAT $nSL $nRT]

# ================= parse INTERCONNECT → WIRE / PRED =================
array set ::WIRE {}         ;# key: "from->to" ; val: {er ef lr lf}
array set ::PRED_FROM {}    ;# key: to , val: from
array set ::PRED_W    {}    ;# key: to , val: {er ef lr lf}

proc _parse_inter_line {line num retname} {
  upvar 1 $retname R
  set fmt {^\s*\(INTERCONNECT\s+(\S+)\s+(\S+)\s+\(\s*(%s)\s*::\s*(%s)\s*\)\s+\(\s*(%s)\s*::\s*(%s)\s*\)\s*\)$}
  set pattern [format $fmt $num $num $num $num]
  set m [regexp -inline -- $pattern $line]
  if {$m eq ""} { return 0 }
  set from_raw [lindex $m 1]
  set to_raw   [lindex $m 2]
  lassign [_canon_pin_full $from_raw] _ from_norm
  if {$from_norm eq ""} { set from_norm [_unescape_sdf_name $from_raw] }
  lassign [_canon_pin_full $to_raw] _ to_norm
  if {$to_norm eq ""} { set to_norm [_unescape_sdf_name $to_raw] }
  set R(from) $from_norm
  set R(to)   $to_norm
  set R(er)   [lindex $m 3]
  set R(lr)   [lindex $m 4]
  set R(ef)   [lindex $m 5]
  set R(lf)   [lindex $m 6]
  return 1
}

set fh2 [open $sdf_file r]
set nIC 0
while {[gets $fh2 raw] >= 0} {
  set line [string trim $raw]
  if {[string first "(INTERCONNECT " $line] == 0} {
    array unset R
    if {[_parse_inter_line $line $NUM R]} {
      set tuple [list $R(er) $R(ef) $R(lr) $R(lf)]
      set k1 "$R(from)->$R(to)";                                           set ::WIRE($k1) $tuple
      set k2 "[_strip_top $R(from)]->[_strip_top $R(to)]";                  set ::WIRE($k2) $tuple
      set k3 "[_unescape_sdf_name $R(from)]->[_unescape_sdf_name $R(to)]"; set ::WIRE($k3) $tuple
      # to 的唯一前驱（网的驱动端是唯一的）
      if {![info exists ::PRED_FROM($R(to))]} {
        set ::PRED_FROM($R(to)) $R(from)
        set ::PRED_W($R(to))    $tuple
      }
      incr nIC
    }
  }
}
close $fh2
puts [format "INFO: parsed INTERCONNECT=%d" $nIC]
if {$::DBG_NET} {
  _dbg "DBG|wire|" "keys=$nIC" "(show up to 5)"
  set cnt 0
  foreach k [lsort [array names ::WIRE]] {
    _dbg "DBG|wire|" $k "=" [join $::WIRE($k) " "]
    incr cnt
    if {$cnt>=5} { break }
  }
}

# 反查：给定 to，返回 from 与 W
proc _pred_lookup {to_full ret_from ret_w} {
  upvar 1 $ret_from FROM $ret_w W
  set cand [list $to_full [_strip_top $to_full] [_unescape_sdf_name $to_full] \
                 [_unescape_sdf_name [_strip_top $to_full]]]
  foreach key $cand {
    if {[info exists ::PRED_FROM($key)]} {
      set FROM $::PRED_FROM($key)
      set W    $::PRED_W($key)
      return 1
    }
  }
  return 0
}

# ================= parse CELL/IOPATH → O2I & 边特征输出 =================
set EDGE_DIR [file join $OUTDIR "edge_features"]
file mkdir $EDGE_DIR
set arcs_csv [file join $EDGE_DIR "${DESIGN}.cell_arcs_delay.csv"]
set fa [open $arcs_csv w]
puts $fa "inst_path,cell_type,from_pin,to_pin,delay_early_rise,delay_early_fall,delay_late_rise,delay_late_fall"

array set ::O2I {}  ;# key: inst/out_pin_full -> list of inst/in_pin_full

proc _parse_iopath_line {line num retname} {
  upvar 1 $retname R
  set fmt {^\s*\(IOPATH\s+(\S+)\s+(\S+)\s+\(\s*(%s)\s*::\s*(%s)\s*\)\s+\(\s*(%s)\s*::\s*(%s)\s*\)\s*\)$}
  set pattern [format $fmt $num $num $num $num]
  set m [regexp -inline -- $pattern $line]
  if {$m eq ""} { return 0 }
  set R(from) [lindex $m 1]
  set R(to)   [lindex $m 2]
  set R(er)   [lindex $m 3]
  set R(lr)   [lindex $m 4]
  set R(ef)   [lindex $m 5]
  set R(lf)   [lindex $m 6]
  return 1
}

set fh3 [open $sdf_file r]
set in_cell 0
set cell_depth 0
set cur_inst ""
set cur_cell ""
set opens 0; set closes 0
while {[gets $fh3 raw] >= 0} {
  set line [string trim $raw]
  set opens  [llength [regexp -inline -all {\(} $raw]]
  set closes [llength [regexp -inline -all {\)} $raw]]

  if {!$in_cell} {
    if {[string first "(CELL" $line] == 0} {
      set in_cell 1
      set cell_depth [expr {$opens - $closes}]
      set cur_inst ""; set cur_cell ""
    }
    continue
  } else {
    if {[string first "(CELLTYPE" $line] == 0} {
      set m [regexp -inline -- {^\s*\(CELLTYPE\s+"([^"]+)"\s*\)$} $line]
      if {$m ne ""} { set cur_cell [lindex $m 1] }
    } elseif {[string first "(INSTANCE" $line] == 0} {
      set m [regexp -inline -- {^\s*\(INSTANCE\s+("?)([^)"]+)\1\s*\)$} $line]
      if {$m ne ""} { set cur_inst [_unescape_sdf_name [lindex $m 2]] }
    } elseif {[string first "(IOPATH " $line] == 0} {
      array unset R
      if {[_parse_iopath_line $line $NUM R]} {
        set out_full "$cur_inst/$R(to)"
        set in_full  "$cur_inst/$R(from)"
        if {![info exists ::O2I($out_full)]} { set ::O2I($out_full) [list] }
        if {[lsearch -exact $::O2I($out_full) $in_full] < 0} {
          lappend ::O2I($out_full) $in_full
        }
        set row [list $cur_inst $cur_cell $R(from) $R(to) $R(er) $R(ef) $R(lr) $R(lf)]
        puts $fa [join $row ","]
      }
    }
    set cell_depth [expr {$cell_depth + $opens - $closes}]
    if {$cell_depth <= 0} { set in_cell 0; set cell_depth 0; set cur_inst ""; set cur_cell "" }
  }
}
close $fh3
close $fa
puts "INFO: wrote $arcs_csv"

# ================= net delay to root (两套) =================
proc _net_to_root_ic {pin_name} {
  lassign [_canon_pin_full $pin_name] _ p0
  if {$p0 eq ""} { set p0 [_unescape_sdf_name $pin_name] }

  set er 0.0; set ef 0.0; set lr 0.0; set lf 0.0
  set hops 0
  set cur $p0

  while {$hops < 200000} {
    set FROM ""; set W ""
    if {![_pred_lookup $cur FROM W]} {
      if {$hops == 0} {
        if {$::NET_ROOT_ZERO} { return {0 0 0 0 ROOT 0} } else { return {NA NA NA NA ROOT 0} }
      }
      return [list [format %.9f $er] [format %.9f $ef] [format %.9f $lr] [format %.9f $lf] OK $hops]
    }
    lassign $W w_er w_ef w_lr w_lf
    foreach v {w_er w_ef w_lr w_lf} { if {[set $v] eq "" || [set $v] eq "NA"} { set $v 0.0 } }
    set er [expr {double($er)+double($w_er)}]
    set ef [expr {double($ef)+double($w_ef)}]
    set lr [expr {double($lr)+double($w_lr)}]
    set lf [expr {double($lf)+double($w_lf)}]
    incr hops
    _dbg "DBG|net| IC-STEP" $FROM "->" $cur "add={" $w_er,$w_ef,$w_lr,$w_lf "}"
    set cur $FROM
  }
  return {NA NA NA NA BROKEN 0}
}

proc _net_to_root_mh {pin_name} {
  lassign [_canon_pin_full $pin_name] _ p_sink
  if {$p_sink eq ""} { set p_sink [_unescape_sdf_name $pin_name] }

  set tF ""; set tW ""
  if {![_pred_lookup $p_sink tF tW]} {
    if {$::NET_ROOT_ZERO} { return {0 0 0 0 ROOT 0} } else { return {NA NA NA NA ROOT 0} }
  }

  set er 0.0; set ef 0.0; set lr 0.0; set lf 0.0
  set hops 0
  set seen {}
  set cur_sink $p_sink

  while {$hops < 200000} {
    # step A: sink <-wire- drv_out
    set drv ""; set W ""
    if {![_pred_lookup $cur_sink drv W]} { break }
    lassign $W w_er w_ef w_lr w_lf
    foreach v {w_er w_ef w_lr w_lf} { if {[set $v] eq "" || [set $v] eq "NA"} { set $v 0.0 } }
    set er [expr {double($er)+double($w_er)}]
    set ef [expr {double($ef)+double($w_ef)}]
    set lr [expr {double($lr)+double($w_lr)}]
    set lf [expr {double($lf)+double($w_lf)}]
    incr hops
    _dbg "DBG|mh| WIRE" $drv "->" $cur_sink " add={" $w_er,$w_ef,$w_lr,$w_lf "} HOPS=$hops SUM={" \
         [format %.9f $er] [format %.9f $ef] [format %.9f $lr] [format %.9f $lf] "}"

    # step B: 通过 IOPATH 抬一层（不计 cell 延时）
    set candidates {}
    foreach key [list $drv [_strip_top $drv] [_unescape_sdf_name $drv] [_unescape_sdf_name [_strip_top $drv]]] {
      if {[info exists ::O2I($key)]} {
        foreach inpin $::O2I($key) {
          if {[lsearch -exact $candidates $inpin] < 0} { lappend candidates $inpin }
        }
      }
    }
    if {[llength $candidates] == 0} { break }

    set next_sink ""; set best_w -1.0; set dbg_line ""
    foreach inp $candidates {
      set tF2 ""; set tW2 ""
      if {[_pred_lookup $inp tF2 tW2]} {
        lassign $tW2 te tf tr tl
        set w [expr {double($tr)+double($tl)}]
        if {$::HOP_PICK eq "first"} {
          set next_sink $inp
          set dbg_line [list $tF2 "->" $inp " w={" $te,$tf,$tr,$tl "}"]
          break
        } else {
          if {$w > $best_w} {
            set best_w $w
            set next_sink $inp
            set dbg_line [list $tF2 "->" $inp " w={" $te,$tf,$tr,$tl "}"]
          }
        }
      }
    }
    if {$next_sink eq ""} { break }
    _dbg "DBG|mh| IOPATH choose" {*}$dbg_line
    if {[lsearch -exact $seen $next_sink] >= 0} { break }
    lappend seen $next_sink
    set cur_sink $next_sink
  }

  return [list [format %.9f $er] [format %.9f $ef] [format %.9f $lr] [format %.9f $lf] OK $hops]
}

# ================= write pins csv =================
set out_csv [file join $OUTDIR "${DESIGN}.pins_timing.csv"]
set fo [open $out_csv w]
puts $fo "pin,at_early_rise,at_early_fall,at_late_rise,at_late_fall,slew_early_rise,slew_early_fall,slew_late_rise,slew_late_fall,rat_early_rise,rat_early_fall,rat_late_rise,rat_late_fall,net_to_root_early_rise,net_to_root_early_fall,net_to_root_late_rise,net_to_root_late_fall,net_to_root_tag,net_to_root_hops,is_endpoint,is_root,path_net_to_root_early_rise,path_net_to_root_early_fall,path_net_to_root_late_rise,path_net_to_root_late_fall,path_net_to_root_hops"

proc _get4 {arrName key} {
  upvar 1 $arrName A
  if {[info exists A($key)]} { return $A($key) }
  return {NA NA NA NA}
}

# pin 并集
set pins {}
foreach k [array names AT]  { lappend pins $k }
foreach k [array names SLW] { lappend pins $k }
foreach k [array names RAT] { lappend pins $k }
set pins [lsort -unique $pins]

foreach p $pins {
  lassign [_get4 AT  $p] at_er at_ef at_lr at_lf
  lassign [_get4 SLW $p] sl_er sl_ef sl_lr sl_lf
  lassign [_get4 RAT $p] rt_er rt_ef rt_lr rt_lf

  lassign [_canon_pin_full $p] _ p_full
  if {$p_full eq ""} { set p_full [_unescape_sdf_name $p] }

  # endpoint/root 判定
  set is_ep  [expr {[info exists RAT($p)] ? 1 : 0}]
  set tmpF ""; set tmpW ""
  set is_root [expr {[_pred_lookup $p_full tmpF tmpW] ? 0 : 1}]

  # 非路径版
  lassign [_net_to_root_ic $p_full] nd_er nd_ef nd_lr nd_lf nd_tag nd_hops

  # 路径版
  if {$PATH_ACCUM} {
    lassign [_net_to_root_mh $p_full] pnd_er pnd_ef pnd_lr pnd_lf _ pnd_hops
  } else {
    set pnd_er NA; set pnd_ef NA; set pnd_lr NA; set pnd_lf NA; set pnd_hops 0
  }

  set row [list $p_full $at_er $at_ef $at_lr $at_lf \
                 $sl_er $sl_ef $sl_lr $sl_lf \
                 $rt_er $rt_ef $rt_lr $rt_lf \
                 $nd_er $nd_ef $nd_lr $nd_lf $nd_tag $nd_hops $is_ep $is_root \
                 $pnd_er $pnd_ef $pnd_lr $pnd_lf $pnd_hops]
  puts $fo [join $row ","]
}
close $fo

# ================= summary =================
puts "INFO: wrote $arcs_csv"
puts "INFO: wrote $out_csv"
puts [format "STATS: inputs=%d  outputs=%d  regs_D=%d  clocks=%d" \
       [llength [all_inputs]] [llength [all_outputs]] \
       [llength [all_registers -data_pins]] [llength [all_clocks]]]
foreach c [all_clocks] {
  puts [format " - clock %s per=%s src=%s" [get_property $c name] [get_property $c period] [get_property $c sources]]
}

exit
