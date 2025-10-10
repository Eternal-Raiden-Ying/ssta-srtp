# Robust node pin capacitance extractor (OpenSTA/OpenROAD), no LEF/DEF.
# ENV: TOP, VERILOG, OUT_CSV, LIB_EARLY, LIB_LATE (or LIB), optional SDC

proc getenv {n {d ""}} { if {[info exists ::env($n)]} {return $::env($n)} {return $d} }
proc have {c} { expr {[llength [info commands $c]]>0} }

# --- alias shims (no elseif) ---
if {![have get_attribute]} {
  if {[have sta::get_attribute]} { interp alias {} get_attribute {} sta::get_attribute } else { if {[have get_attr]} { interp alias {} get_attribute {} get_attr } }
}
if {![have get_pins]}           { if {[have sta::get_pins]}           { interp alias {} get_pins {} sta::get_pins } }
if {![have get_ports]}          { if {[have sta::get_ports]}          { interp alias {} get_ports {} sta::get_ports } }
if {![have get_lib_pins]}       { if {[have sta::get_lib_pins]}       { interp alias {} get_lib_pins {} sta::get_lib_pins } }
if {![have get_lib_cells]}      { if {[have sta::get_lib_cells]}      { interp alias {} get_lib_cells {} sta::get_lib_cells } }
if {![have get_cells]}          { if {[have sta::get_cells]}          { interp alias {} get_cells {} sta::get_cells } }
if {![have define_corners]}     { if {[have sta::define_corners]}     { interp alias {} define_corners {} sta::define_corners } }
if {![have read_liberty]}       { if {[have sta::read_liberty]}       { interp alias {} read_liberty {} sta::read_liberty } }
if {![have read_verilog]}       { if {[have sta::read_verilog]}       { interp alias {} read_verilog {} sta::read_verilog } }
if {![have link_design]}        { if {[have sta::link_design]}        { interp alias {} link_design {} sta::link_design } }
if {![have read_sdc]}           { if {[have sta::read_sdc]}           { interp alias {} read_sdc {} sta::read_sdc } }
if {![have set_current_corner]} { if {[have sta::set_current_corner]} { interp alias {} set_current_corner {} sta::set_current_corner } }

proc GETATTR {obj attr {def ""}} {
  set v ""
  if {[have get_attribute]} { set v [get_attribute -quiet $obj $attr] }
  if {$v eq "" && [have sta::get_attribute]} { set v [sta::get_attribute -quiet $obj $attr] }
  if {$v eq "" && [have get_attr]} { set v [get_attr $obj $attr] }
  if {$v eq ""} { return $def } else { return $v }
}

proc leaf_of {full} { set leaf $full; if {[regexp {([^/]+)$} $full -> leaf]} {}; return $leaf }

# resolve libcell name/object to object
proc RESOLVE_LIBCELL {lc} {
  if {$lc eq ""} { return "" }
  # if already an object, try attribute to test
  set name [GETATTR $lc name ""]
  if {$name ne ""} { return $lc }
  set lst [get_lib_cells -quiet $lc]
  if {$lst eq ""} { return "" }
  return [lindex $lst 0]
}

# try to get libpin from design pin
proc LIBPIN_FROM_PIN {p} {
  # 1) direct attribute
  set lp [GETATTR $p lib_pin ""]
  if {$lp ne ""} {
    set obj [get_lib_pins -quiet $lp]
    if {$obj ne ""} { return [lindex $obj 0] }
    # maybe already object?
    set n [GETATTR $lp name ""]
    if {$n ne ""} { return $lp }
  }
  # 2) via instance -> libcell
  set full $p
  set inst ""; set leaf $full
  if {[regexp {(.+)/(.+)} $full -> inst leaf]} {}
  if {$inst eq ""} { return "" } ;# top port pin
  # prefer relation query; fallback to name
  set instObj ""
  set tmp [get_cells -of_objects $p -quiet]
  if {$tmp ne ""} { set instObj [lindex $tmp 0] } else {
    set tmp2 [get_cells -quiet $inst]
    if {$tmp2 ne ""} { set instObj [lindex $tmp2 0] }
  }
  if {$instObj eq ""} { return "" }
  set lc [GETATTR $instObj lib_cell ""]
  if {$lc eq ""} {
    set ref [GETATTR $instObj ref_name ""]
    if {$ref ne ""} { set lc $ref }
  }
  set lcobj [RESOLVE_LIBCELL $lc]
  if {$lcobj eq ""} { return "" }
  # search pin with same leaf name
  set lps [get_lib_pins -of_objects $lcobj -quiet]
  foreach x $lps {
    set n [GETATTR $x name ""]
    if {$n eq ""} { set n $x }
    if {[leaf_of $n] eq $leaf} { return $x }
  }
  return ""
}

proc CAPS_FROM_LIBPIN_CORNER {libpin corner} {
  set r 0.0; set f 0.0
  if {$libpin ne ""} {
    set_current_corner $corner
    set rcap [GETATTR $libpin rise_capacitance ""]
    if {$rcap eq ""} { set rcap [GETATTR $libpin capacitance ""] }
    set fcap [GETATTR $libpin fall_capacitance ""]
    if {$fcap eq ""} { set fcap [GETATTR $libpin capacitance ""] }
    if {$rcap eq ""} { set rcap 0.0 }
    if {$fcap eq ""} { set fcap 0.0 }
    set r [expr {double($rcap) * 1.0e15}]
    set f [expr {double($fcap) * 1.0e15}]
  }
  return [list $r $f]
}

# --- inputs & load ---
set TOP       [getenv TOP]
set VERILOG   [getenv VERILOG]
set OUT_CSV   [getenv OUT_CSV "pin_caps.csv"]
set LIB_EARLY [getenv LIB_EARLY]
set LIB_LATE  [getenv LIB_LATE]
set LIB       [getenv LIB]
if {$LIB_EARLY eq "" && $LIB_LATE eq ""} { set LIB_EARLY $LIB; set LIB_LATE $LIB }
if {$LIB_EARLY eq "" || $LIB_LATE eq ""} { puts "ERROR: set LIB or LIB_EARLY & LIB_LATE"; exit 1 }
set SDC   [getenv SDC]

define_corners early late
read_liberty -corner early $LIB_EARLY
read_liberty -corner late  $LIB_LATE
read_verilog $VERILOG
link_design $TOP
if {$SDC ne ""} { read_sdc $SDC }

# --- dump ---
set fp [open $OUT_CSV w]
puts $fp "design,inst,pin,full_pin,is_primary_io,direction,cap_early_rise_fF,cap_early_fall_fF,cap_late_rise_fF,cap_late_fall_fF"

# instance pins
foreach p [get_pins -hier *] {
  set full $p
  set inst ""; set leaf $full
  if {[regexp {(.+)/(.+)} $full -> inst leaf]} {}
  set dir [GETATTR $p direction ""]
  if {$dir eq ""} { set dir "NA" }

  set libpin [LIBPIN_FROM_PIN $p]
  set er [CAPS_FROM_LIBPIN_CORNER $libpin early]
  set lr [CAPS_FROM_LIBPIN_CORNER $libpin late]
  puts $fp "$TOP,$inst,$leaf,$full,0,$dir,[lindex $er 0],[lindex $er 1],[lindex $lr 0],[lindex $lr 1]"
}

# top ports (cap=0)
foreach prt [get_ports *] {
  set dir [GETATTR $prt direction "NA"]
  set name $prt
  puts $fp "$TOP,,${name},${name},1,$dir,0,0,0,0"
}

close $fp
exit
