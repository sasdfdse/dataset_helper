"""Linux(X11) 파일/폴더 선택 창의 목록을 세로 한 줄로 바꾼다.

Tk의 X11 파일 대화상자(askdirectory, askopenfilename 등)는 ::tk::IconList 위젯으로 항목을
여러 열로 늘어놓고 가로 스크롤바로 옆으로 넘긴다. 폴더가 많으면 찾기 불편하므로, 그 클래스의
배치/스크롤 메서드를 덮어써서 항목을 위에서 아래로 한 열에 늘어놓고, 세로 스크롤바와 마우스
휠(위/아래)로 오르내리게 한다. 대화상자 자체(경로 입력, 필터, 버튼 동작)는 그대로다.

install(root)은 Tk 루트를 만든 직후 한 번 부르면 된다. X11이 아니면(Windows/macOS 기본
대화상자) 아무것도 하지 않는다.
"""
from __future__ import annotations

import tkinter as tk

_TCL_PATCH = r"""
auto_load ::tk::IconList

oo::define ::tk::IconList {
    method Create {} {
        variable hull
        set sbar [ttk::scrollbar $hull.sbar -orient vertical -takefocus 0]
        catch {$sbar configure -highlightthickness 0}
        set canvas [canvas $hull.canvas -highlightthick 0 -takefocus 1 \
                        -width 400 -height 240 -background white]
        pack $sbar -side right -fill y -padx {0 2} -pady 2
        pack $canvas -side left -expand yes -fill both -padx {2 0} -pady 2

        $sbar configure -command [list $canvas yview]
        $canvas configure -yscrollcommand [list $sbar set]

        set maxIW 1
        set maxIH 1
        set maxTW 1
        set maxTH 1
        set numItems 0
        set noScroll 1
        set selection {}
        set index(anchor) ""
        set fill black

        bind $canvas <Configure>        [namespace code {my WhenIdle Arrange}]

        bind $canvas <Button-1>         [namespace code {my Btn1 %x %y}]
        bind $canvas <B1-Motion>        [namespace code {my Motion1 %x %y}]
        bind $canvas <B1-Leave>         [namespace code {my Leave1 %x %y}]
        bind $canvas <Control-Button-1> [namespace code {my CtrlBtn1 %x %y}]
        bind $canvas <Shift-Button-1>   [namespace code {my ShiftBtn1 %x %y}]
        bind $canvas <B1-Enter>         [list tk::CancelRepeat]
        bind $canvas <ButtonRelease-1>  [list tk::CancelRepeat]
        bind $canvas <Double-ButtonRelease-1> \
            [namespace code {my Double1 %x %y}]

        bind $canvas <Control-B1-Motion> {;}
        bind $canvas <Shift-B1-Motion>  [namespace code {my ShiftMotion1 %x %y}]

        # mouse wheel scrolls vertically
        bind $canvas <MouseWheel>       [namespace code {my MouseWheel %D}]
        bind $canvas <Button-4>         [namespace code {my MouseWheel 120}]
        bind $canvas <Button-5>         [namespace code {my MouseWheel -120}]

        bind $canvas <<PrevLine>>       [namespace code {my UpDown -1}]
        bind $canvas <<NextLine>>       [namespace code {my UpDown  1}]
        bind $canvas <<PrevChar>>       [namespace code {my UpDown -1}]
        bind $canvas <<NextChar>>       [namespace code {my UpDown  1}]
        bind $canvas <Prior>            [namespace code {my PageUpDown -1}]
        bind $canvas <Next>             [namespace code {my PageUpDown  1}]
        bind $canvas <Return>           [namespace code {my ReturnKey}]
        bind $canvas <Key>              [namespace code {my KeyPress %A}]
        bind $canvas <Alt-Key>          {# nothing}
        bind $canvas <Meta-Key>         {# nothing}
        bind $canvas <Control-Key>      {# nothing}

        bind $canvas <FocusIn>          [namespace code {my FocusIn}]
        bind $canvas <FocusOut>         [namespace code {my FocusOut}]

        return $w
    }

    method deleteall {} {
        $canvas delete all
        unset -nocomplain selected rect list itemList
        set maxIW 1
        set maxIH 1
        set maxTW 1
        set maxTH 1
        set numItems 0
        set noScroll 1
        set selection {}
        set index(anchor) ""
        $sbar set 0.0 1.0
        $canvas yview moveto 0
    }

    # one column, top to bottom
    method Arrange {} {
        if {![info exists list]} {
            if {[info exists canvas] && [winfo exists $canvas]} {
                set noScroll 1
                $sbar configure -command ""
            }
            return
        }

        set W [winfo width  $canvas]
        set H [winfo height $canvas]
        set pad [expr {[$canvas cget -highlightthickness]+[$canvas cget -bd]}]
        if {$pad < 2} {
            set pad 2
        }
        incr W [expr {$pad*-2}]
        incr H [expr {$pad*-2}]

        set dx [expr {$maxIW + $maxTW + 8}]
        if {$dx < $W} {
            set dx $W
        }
        set dy [expr {$maxTH > $maxIH ? $maxTH : $maxIH}]
        incr dy 2
        set shift [expr {$maxIW + 4}]
        $canvas configure -yscrollincrement $dy    ;# one scroll unit = one row

        set x [expr {$pad * 2}]
        set y $pad
        foreach sublist $list {
            lassign $sublist iTag tTag rTag iW iH tW tH
            set i_dy [expr {($dy - $iH)/2}]
            set t_dy [expr {($dy - $tH)/2}]
            $canvas coords $iTag $x [expr {$y + $i_dy}]
            $canvas coords $tTag [expr {$x + $shift}] [expr {$y + $t_dy}]
            $canvas coords $rTag $x $y [expr {$x+$dx}] [expr {$y+$dy}]
            incr y $dy
        }

        $canvas configure -scrollregion [list $pad $pad [expr {$x + $dx}] $y]
        if {$y <= $H + $pad} {
            $sbar configure -command ""
            $sbar set 0.0 1.0
            $canvas yview moveto 0
            set noScroll 1
        } else {
            $sbar configure -command [list $canvas yview]
            set noScroll 0
        }

        # single column, so left/right arrows move one item too
        set itemsPerColumn 1

        my DrawSelection
    }

    method see rTag {
        if {$noScroll} {
            return
        }
        set sRegion [$canvas cget -scrollregion]
        if {$sRegion eq ""} {
            return
        }
        if {$rTag < 0 || $rTag >= [llength $list]} {
            return
        }

        set bbox [$canvas bbox item$rTag]
        set pad [expr {[$canvas cget -highlightthickness]+[$canvas cget -bd]}]
        set y1 [expr {[lindex $bbox 1] - $pad}]
        set y2 [expr {[lindex $bbox 3] + $pad}]

        set cH [expr {[winfo height $canvas] - $pad*2}]
        set scrollH [expr {[lindex $sRegion 3]-[lindex $sRegion 1]}]
        set top [lindex $sRegion 1]
        set dispY [expr {$top + [lindex [$canvas yview] 0]*$scrollH}]
        set oldDispY $dispY

        if {($y2 - $dispY) > $cH} {
            set dispY [expr {$y2 - $cH}]
        }
        if {$y1 < $dispY} {
            set dispY $y1
        }

        if {$oldDispY != $dispY} {
            $canvas yview moveto [expr {double($dispY - $top) / double($scrollH)}]
        }
    }

    method AutoScan {} {
        if {![winfo exists $w]} return
        set x $oldX
        set y $oldY
        if {$noScroll} {
            return
        }
        if {$y >= [winfo height $canvas]} {
            $canvas yview scroll 1 units
        } elseif {$y < 0} {
            $canvas yview scroll -1 units
        } else {
            return
        }
        my Motion1 $x $y
        set ::tk::Priv(afterId) [after 50 [namespace code {my AutoScan}]]
    }

    method MouseWheel {amount} {
        if {$noScroll || $::tk_strictMotif} {
            return
        }
        # 3 rows per wheel notch (delta 120)
        if {$amount > 0} {
            $canvas yview scroll [expr {-3 * max(1, $amount / 120)}] units
        } else {
            $canvas yview scroll [expr {3 * max(1, -$amount / 120)}] units
        }
    }

    method PageUpDown {amount} {
        if {![info exists list]} {
            return
        }
        set dy [expr {$maxTH > $maxIH ? $maxTH : $maxIH}]
        incr dy 2
        set rows [expr {max(1, [winfo height $canvas] / $dy - 1)}]
        my UpDown [expr {$amount * $rows}]
    }
}
"""

_FLAG = "::dataset_helper_vertical_iconlist"


def install(root: tk.Misc) -> None:
    """X11에서 Tk 파일/폴더 대화상자의 목록을 세로 한 열 + 휠 스크롤로 바꾼다."""
    try:
        if root.tk.call("tk", "windowingsystem") != "x11":
            return
        if root.tk.call("info", "exists", _FLAG):
            return
        root.tk.eval(_TCL_PATCH)
        root.tk.call("set", _FLAG, 1)
    except tk.TclError:
        return      # unexpected Tk version: fall back to the stock dialog
