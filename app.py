"""
Singly Reinforced RCC Beam Designer
As per IS 456:2000 — Limit State Method (LSM)
Simply Supported Beam, UDL Loading
"""

import math
import datetime
import shutil
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

# ─── IS 456:2000 Constants ────────────────────────────────────────────────────

# Limiting neutral axis depth ratio xu,max/d  [IS 456:2000, Annex G, Cl. G-1.1]
# Derived from: xu_max/d = 700 / (1100 + 0.87 * fy)
XU_MAX_RATIO = {
    250: 0.531,
    415: 0.479,
    500: 0.456,
    550: 0.444,
}

# IS 456:2000, Table 19 — Design shear strength τc (N/mm²) of concrete
TAU_C_TABLE = [
    # (pt%, {fck: τc})
    (0.15, {15: 0.28, 20: 0.28, 25: 0.29, 30: 0.29, 35: 0.29, 40: 0.30}),
    (0.25, {15: 0.35, 20: 0.36, 25: 0.36, 30: 0.37, 35: 0.37, 40: 0.38}),
    (0.50, {15: 0.46, 20: 0.48, 25: 0.49, 30: 0.50, 35: 0.50, 40: 0.51}),
    (0.75, {15: 0.54, 20: 0.56, 25: 0.57, 30: 0.59, 35: 0.59, 40: 0.60}),
    (1.00, {15: 0.60, 20: 0.62, 25: 0.64, 30: 0.66, 35: 0.67, 40: 0.68}),
    (1.25, {15: 0.64, 20: 0.67, 25: 0.70, 30: 0.71, 35: 0.73, 40: 0.74}),
    (1.50, {15: 0.68, 20: 0.72, 25: 0.74, 30: 0.76, 35: 0.78, 40: 0.79}),
    (1.75, {15: 0.71, 20: 0.75, 25: 0.78, 30: 0.80, 35: 0.82, 40: 0.84}),
    (2.00, {15: 0.71, 20: 0.79, 25: 0.82, 30: 0.84, 35: 0.86, 40: 0.88}),
    (2.25, {15: 0.71, 20: 0.81, 25: 0.85, 30: 0.88, 35: 0.90, 40: 0.92}),
    (2.50, {15: 0.71, 20: 0.82, 25: 0.88, 30: 0.91, 35: 0.93, 40: 0.95}),
    (2.75, {15: 0.71, 20: 0.82, 25: 0.90, 30: 0.94, 35: 0.96, 40: 0.98}),
    (3.00, {15: 0.71, 20: 0.82, 25: 0.92, 30: 0.96, 35: 0.99, 40: 1.01}),
]

# IS 456:2000, Table 20 — Maximum shear stress τc,max (N/mm²)
TAU_C_MAX_TABLE = {15: 2.5, 20: 2.8, 25: 3.1, 30: 3.5, 35: 3.7, 40: 4.0}


# ─── Helper Functions ─────────────────────────────────────────────────────────

def get_tau_c(pt: float, fck: int) -> float:
    """Interpolate τc from IS 456:2000 Table 19."""
    fck_vals = [15, 20, 25, 30, 35, 40]
    fck_use = min(fck_vals, key=lambda x: abs(x - fck))
    table = TAU_C_TABLE
    if pt <= table[0][0]:
        return table[0][1][fck_use]
    if pt >= table[-1][0]:
        return table[-1][1][fck_use]
    for i in range(len(table) - 1):
        if table[i][0] <= pt <= table[i + 1][0]:
            p1, p2 = table[i][0], table[i + 1][0]
            t1 = table[i][1][fck_use]
            t2 = table[i + 1][1][fck_use]
            return t1 + (t2 - t1) * (pt - p1) / (p2 - p1)
    return 0.28


def get_tau_c_max(fck: int) -> float:
    """Return τc,max from IS 456:2000 Table 20."""
    fck_vals = list(TAU_C_MAX_TABLE.keys())
    fck_use = min(fck_vals, key=lambda x: abs(x - fck))
    return TAU_C_MAX_TABLE[fck_use]


# ─── Core Design Engine ───────────────────────────────────────────────────────

def design_beam(inp: dict):
    """
    Complete design of a singly reinforced RCC beam as per IS 456:2000.

    Parameters
    ----------
    inp : dict
        span      — Effective span (m)
        w_DL      — Dead load UDL (kN/m)
        w_LL      — Live load UDL (kN/m)
        fck       — Characteristic compressive strength of concrete (N/mm²)
        fy        — Yield strength of steel (N/mm²)
        b         — Beam width (mm)
        D         — Total depth (mm)
        cover     — Clear cover (mm)
        bar_dia   — Main bar diameter (mm)
        stir_dia  — Stirrup diameter (mm)

    Returns
    -------
    results : dict | None
        Key design values; None if design fails.
    lines : list[str]
        Report lines (plain text).
    """

    lines = []

    def add(txt=""):
        lines.append(txt)

    def hdr(title):
        add()
        add("=" * 72)
        add(f"  {title}")
        add("=" * 72)

    def step(label, formula, expression, value, unit=""):
        add(f"  ► {label}")
        add(f"      Formula     : {formula}")
        add(f"      Expression  : {expression}")
        add(f"      Value       : {value:.4f} {unit}")
        add()

    # ── Unpack ────────────────────────────────────────────────────────────────
    L        = inp["span"]       # m
    w_DL     = inp["w_DL"]      # kN/m
    w_LL     = inp["w_LL"]      # kN/m
    fck      = inp["fck"]       # N/mm²
    fy       = inp["fy"]        # N/mm²
    b        = inp["b"]         # mm
    D        = inp["D"]         # mm
    cover    = inp["cover"]     # mm
    bar_dia  = inp["bar_dia"]   # mm
    stir_dia = inp["stir_dia"]  # mm

    # Effective depth
    d = D - cover - stir_dia - bar_dia / 2.0

    # ── Report Header ─────────────────────────────────────────────────────────
    add("=" * 72)
    add("  DESIGN OF SINGLY REINFORCED RCC BEAM")
    add("  Reference: IS 456:2000 — Plain and Reinforced Concrete")
    add("  Method   : Limit State Method (LSM)")
    add(f"  Date     : {datetime.date.today()}")
    add("=" * 72)

    # ── Given Data ────────────────────────────────────────────────────────────
    hdr("GIVEN DATA")
    add(f"  Effective Span           L   = {L} m")
    add(f"  Dead Load (UDL)          w_DL = {w_DL} kN/m")
    add(f"  Live Load (UDL)          w_LL = {w_LL} kN/m")
    add(f"  Grade of Concrete             M{fck}   → fck = {fck} N/mm²")
    add(f"  Grade of Steel                Fe{fy}  → fy  = {fy} N/mm²")
    add(f"  Beam Width               b   = {b} mm")
    add(f"  Total Depth              D   = {D} mm")
    add(f"  Clear Cover              c'  = {cover} mm")
    add(f"  Main Bar Diameter        φ   = {bar_dia} mm")
    add(f"  Stirrup Diameter         φv  = {stir_dia} mm")

    # ── Step 1: Effective Depth ───────────────────────────────────────────────
    hdr("STEP 1 — EFFECTIVE DEPTH")
    step(
        "Effective Depth (d)  [IS 456:2000, Cl. 22.2]",
        "d = D − c' − φv − φ/2",
        f"d = {D} − {cover} − {stir_dia} − {bar_dia}/2",
        d, "mm",
    )

    # ── Step 2: Loads and Factored Moment ─────────────────────────────────────
    hdr("STEP 2 — FACTORED BENDING MOMENT")

    w_total = w_DL + w_LL
    add(f"  Total service load   w = w_DL + w_LL = {w_DL} + {w_LL} = {w_total} kN/m")
    add()

    M_service = w_total * L ** 2 / 8.0   # kN·m
    step(
        "Service Bending Moment (M)  [Simply Supported, UDL]",
        "M = w × L² / 8",
        f"M = {w_total} × {L}² / 8",
        M_service, "kN·m",
    )

    Mu = 1.5 * M_service * 1.0e6        # N·mm
    step(
        "Factored Bending Moment (Mu)  [IS 456:2000, Table 18, γf = 1.5]",
        "Mu = γf × M = 1.5 × M",
        f"Mu = 1.5 × {M_service:.4f} × 10⁶  N·mm",
        Mu / 1.0e6, "kN·m",
    )
    add(f"      → Mu = {Mu:.2f} N·mm")
    add()

    # Shear force (for use in Step 10)
    V_service = w_total * L / 2.0       # kN
    Vu = 1.5 * V_service * 1.0e3       # N

    # ── Step 3: Limiting Moment of Resistance ─────────────────────────────────
    hdr("STEP 3 — LIMITING MOMENT OF RESISTANCE")

    xu_max_ratio = XU_MAX_RATIO.get(fy, 0.479)
    xu_max = xu_max_ratio * d

    add("  [IS 456:2000, Annex G, Cl. G-1.1]")
    add(f"  Limiting neutral axis ratio for Fe{fy}:")
    add(f"    xu,max / d = 700 / (1100 + 0.87 × fy)")
    add(f"    xu,max / d = 700 / (1100 + 0.87 × {fy})")
    add(f"    xu,max / d = {xu_max_ratio}")
    add()

    step(
        "Limiting Neutral Axis Depth (xu,max)",
        "xu,max = (xu,max/d) × d",
        f"xu,max = {xu_max_ratio} × {d:.4f}",
        xu_max, "mm",
    )

    Mu_lim = 0.36 * fck * b * xu_max * (d - 0.42 * xu_max)  # N·mm
    step(
        "Limiting Moment of Resistance (Mu,lim)  [IS 456:2000, Annex G, Eq. G-1.1]",
        "Mu,lim = 0.36 × fck × b × xu,max × (d − 0.42 × xu,max)",
        (
            f"Mu,lim = 0.36 × {fck} × {b} × {xu_max:.2f}"
            f" × ({d:.2f} − 0.42 × {xu_max:.2f})"
        ),
        Mu_lim / 1.0e6, "kN·m",
    )
    add(f"      → Mu,lim = {Mu_lim:.2f} N·mm")
    add()

    # ── Step 4: Section Adequacy ──────────────────────────────────────────────
    hdr("STEP 4 — SECTION ADEQUACY CHECK")
    add(f"  Factored Moment   Mu      = {Mu / 1.0e6:.4f} kN·m")
    add(f"  Limiting Moment   Mu,lim  = {Mu_lim / 1.0e6:.4f} kN·m")
    add()

    if Mu > Mu_lim:
        add("  ✗ Mu > Mu,lim")
        add("    Section is INADEQUATE for singly reinforced beam design.")
        add("    → Increase total depth D, or adopt a doubly reinforced section.")
        add("    Design aborted. Revise inputs and re-run.")
        return None, lines

    add("  ✓ Mu ≤ Mu,lim")
    add("    Section is ADEQUATE — proceed with singly reinforced design.")
    add()

    # ── Step 5: Area of Tension Steel (Ast) ───────────────────────────────────
    hdr("STEP 5 — AREA OF TENSION STEEL (Ast)")

    add("  From IS 456:2000, Annex G, Cl. G-1.1a:")
    add("    Mu = 0.87 × fy × Ast × [d − (Ast × fy) / (fck × b)]")
    add()
    add("  Expanding and rearranging into standard quadratic form A·Ast² + B·Ast + C = 0:")
    add()
    add("    0.87×fy²         ")
    add("    ─────────  × Ast²  −  0.87×fy×d × Ast  +  Mu  =  0")
    add("     fck × b         ")
    add()

    A_q = (0.87 * fy ** 2) / (fck * b)
    B_q = -(0.87 * fy * d)
    C_q = Mu

    add(f"  Coefficients:")
    add(f"    A = 0.87 × fy² / (fck × b)")
    add(f"      = 0.87 × {fy}² / ({fck} × {b})")
    add(f"      = {A_q:.6f}")
    add()
    add(f"    B = −0.87 × fy × d")
    add(f"      = −0.87 × {fy} × {d:.4f}")
    add(f"      = {B_q:.4f}")
    add()
    add(f"    C = Mu = {C_q:.4f} N·mm")
    add()

    discriminant = B_q ** 2 - 4.0 * A_q * C_q
    add(f"  Discriminant  Δ = B² − 4AC")
    add(f"    Δ = ({B_q:.4f})² − 4 × {A_q:.6f} × {C_q:.4f}")
    add(f"    Δ = {B_q**2:.4f} − {4*A_q*C_q:.4f}")
    add(f"    Δ = {discriminant:.4f}")
    add()

    if discriminant < 0:
        add("  ✗ Discriminant < 0 → No real solution. Section requires revision.")
        return None, lines

    Ast1 = (-B_q - math.sqrt(discriminant)) / (2.0 * A_q)
    Ast2 = (-B_q + math.sqrt(discriminant)) / (2.0 * A_q)

    add(f"  Roots of quadratic:")
    add(f"    Ast  =  (−B ± √Δ) / (2A)")
    add()
    add(f"    Ast₁ = (−({B_q:.4f}) − √{discriminant:.4f}) / (2 × {A_q:.6f})")
    add(f"         = {Ast1:.4f} mm²    ← valid (smaller, under-reinforced)")
    add()
    add(f"    Ast₂ = (−({B_q:.4f}) + √{discriminant:.4f}) / (2 × {A_q:.6f})")
    add(f"         = {Ast2:.4f} mm²    ← exceeds balanced condition, reject")
    add()
    add(f"  → Required Ast  =  {Ast1:.4f} mm²")
    add()

    Ast_req = Ast1

    # ── Step 6: Min/Max Steel Check ───────────────────────────────────────────
    hdr("STEP 6 — MINIMUM AND MAXIMUM STEEL CHECK")

    Ast_min = 0.85 * b * d / fy
    Ast_max = 0.04 * b * D

    step(
        "Minimum Tension Reinforcement (Ast,min)  [IS 456:2000, Cl. 26.5.1.1 (a)]",
        "Ast,min / (b × d) = 0.85 / fy",
        f"Ast,min = 0.85 × {b} × {d:.4f} / {fy}",
        Ast_min, "mm²",
    )

    step(
        "Maximum Tension Reinforcement (Ast,max)  [IS 456:2000, Cl. 26.5.1.1 (b)]",
        "Ast,max = 0.04 × b × D",
        f"Ast,max = 0.04 × {b} × {D}",
        Ast_max, "mm²",
    )

    if Ast_req < Ast_min:
        add(f"  Ast,req ({Ast_req:.4f} mm²) < Ast,min ({Ast_min:.4f} mm²)")
        add(f"  → Adopt Ast = Ast,min = {Ast_min:.4f} mm²")
        Ast_design = Ast_min
    elif Ast_req > Ast_max:
        add(f"  ✗ Ast,req ({Ast_req:.4f} mm²) > Ast,max ({Ast_max:.4f} mm²)")
        add("    → Section is over-stressed. Revise beam dimensions.")
        return None, lines
    else:
        add(f"  ✓ Ast,min ≤ Ast,req ≤ Ast,max")
        add(f"    Ast,min = {Ast_min:.4f} mm²")
        add(f"    Ast,req = {Ast_req:.4f} mm²  ← govern")
        add(f"    Ast,max = {Ast_max:.4f} mm²")
        Ast_design = Ast_req
    add()

    # ── Step 7: Bar Selection ─────────────────────────────────────────────────
    hdr("STEP 7 — SELECTION OF TENSION STEEL BARS")

    a_bar = math.pi * bar_dia ** 2 / 4.0
    n_bars = math.ceil(Ast_design / a_bar)
    Ast_prov = n_bars * a_bar

    step(
        f"Area of one {bar_dia} mm diameter bar (a_bar)",
        "a_bar = π × φ² / 4",
        f"a_bar = π × {bar_dia}² / 4",
        a_bar, "mm²",
    )

    add(f"  Number of bars required = Ast,design / a_bar")
    add(f"    = {Ast_design:.4f} / {a_bar:.4f}")
    add(f"    = {Ast_design / a_bar:.4f}  →  Round up to  {n_bars} bars")
    add()

    step(
        "Ast Provided",
        "Ast,prov = n × (π × φ² / 4)",
        f"Ast,prov = {n_bars} × π × {bar_dia}² / 4",
        Ast_prov, "mm²",
    )

    add(f"  → Provide {n_bars} — {bar_dia} mm dia bars")
    add(f"     Ast provided = {Ast_prov:.4f} mm²  ≥  Ast required = {Ast_design:.4f} mm²  ✓")
    add()

    # ── Step 8: Actual Neutral Axis Depth ─────────────────────────────────────
    hdr("STEP 8 — ACTUAL NEUTRAL AXIS DEPTH")

    add("  Equilibrium of compression and tension forces:")
    add("    C = T")
    add("    0.36 × fck × b × xu  =  0.87 × fy × Ast,prov")
    add()
    xu_act = (0.87 * fy * Ast_prov) / (0.36 * fck * b)
    step(
        "Actual Neutral Axis Depth (xu)",
        "xu = (0.87 × fy × Ast,prov) / (0.36 × fck × b)",
        f"xu = (0.87 × {fy} × {Ast_prov:.4f}) / (0.36 × {fck} × {b})",
        xu_act, "mm",
    )

    add(f"  xu         = {xu_act:.4f} mm")
    add(f"  xu,max     = {xu_max:.4f} mm")
    add()
    if xu_act <= xu_max:
        add(f"  ✓ xu ({xu_act:.4f}) ≤ xu,max ({xu_max:.4f})")
        add("    → Under-reinforced section (ductile failure mode)  ✓")
    else:
        add(f"  ✗ xu ({xu_act:.4f}) > xu,max ({xu_max:.4f})")
        add("    → Over-reinforced section. Revise steel or depth.")
    add()

    # ── Step 9: Actual Moment of Resistance ───────────────────────────────────
    hdr("STEP 9 — ACTUAL MOMENT OF RESISTANCE")

    Mu_act = 0.87 * fy * Ast_prov * (d - (fy * Ast_prov) / (fck * b))
    step(
        "Actual Moment of Resistance (Mu,act)  [IS 456:2000, Annex G, Eq. G-1.1a]",
        "Mu,act = 0.87 × fy × Ast,prov × [d − (fy × Ast,prov) / (fck × b)]",
        (
            f"Mu,act = 0.87 × {fy} × {Ast_prov:.4f}"
            f" × [{d:.2f} − ({fy} × {Ast_prov:.4f}) / ({fck} × {b})]"
        ),
        Mu_act / 1.0e6, "kN·m",
    )
    add(f"      → Mu,act = {Mu_act:.2f} N·mm")
    add()
    add(f"  Factored moment Mu      = {Mu / 1.0e6:.4f} kN·m")
    add(f"  Actual MR      Mu,act   = {Mu_act / 1.0e6:.4f} kN·m")
    add()
    if Mu_act >= Mu:
        add("  ✓ Mu,act ≥ Mu  →  Section is SAFE in flexure")
    else:
        add("  ✗ Mu,act < Mu  →  Section is UNSAFE. Increase Ast.")
    add()

    # ── Step 10: Shear Design ─────────────────────────────────────────────────
    hdr("STEP 10 — DESIGN FOR SHEAR  [IS 456:2000, Cl. 40]")

    step(
        "Service Shear Force (V)  [Simply Supported, UDL]",
        "V = w × L / 2",
        f"V = {w_total} × {L} / 2",
        V_service, "kN",
    )

    step(
        "Factored Shear Force (Vu)  [IS 456:2000, Table 18, γf = 1.5]",
        "Vu = 1.5 × V",
        f"Vu = 1.5 × {V_service:.4f} × 1000",
        Vu / 1.0e3, "kN",
    )
    add(f"      → Vu = {Vu:.2f} N")
    add()

    tau_v = Vu / (b * d)
    step(
        "Nominal Shear Stress (τv)  [IS 456:2000, Cl. 40.1]",
        "τv = Vu / (b × d)",
        f"τv = {Vu:.2f} / ({b} × {d:.4f})",
        tau_v, "N/mm²",
    )

    pt = 100.0 * Ast_prov / (b * d)
    add(f"  Percentage of tension steel (pt):")
    add(f"    pt = 100 × Ast,prov / (b × d)")
    add(f"       = 100 × {Ast_prov:.4f} / ({b} × {d:.4f})")
    add(f"       = {pt:.4f} %")
    add()

    tau_c = get_tau_c(pt, fck)
    tau_c_max = get_tau_c_max(fck)

    add(f"  Design shear strength of concrete:")
    add(f"    τc = {tau_c:.4f} N/mm²")
    add(f"    [IS 456:2000, Table 19 — interpolated for pt = {pt:.4f}%, M{fck}]")
    add()
    add(f"  Maximum permissible shear stress:")
    add(f"    τc,max = {tau_c_max:.4f} N/mm²")
    add(f"    [IS 456:2000, Table 20 — for M{fck}]")
    add()

    # Stirrup area (2-legged)
    Asv = 2.0 * math.pi * stir_dia ** 2 / 4.0

    if tau_v > tau_c_max:
        add(f"  ✗ τv ({tau_v:.4f}) > τc,max ({tau_c_max:.4f}) N/mm²")
        add("    Beam cross-section must be revised (increase b or d).")
        shear_result = "SECTION INADEQUATE — revise"
    elif tau_v <= tau_c:
        add(f"  ✓ τv ({tau_v:.4f}) ≤ τc ({tau_c:.4f}) N/mm²")
        add("    Minimum/nominal stirrups required  [IS 456:2000, Cl. 40.3]")
        add()

        step(
            "Area of stirrup legs Asv (2-legged)",
            "Asv = 2 × (π × φv² / 4)",
            f"Asv = 2 × π × {stir_dia}² / 4",
            Asv, "mm²",
        )

        sv_min = (0.87 * fy * Asv) / (0.4 * b)
        sv_max = min(0.75 * d, 300.0)
        sv_adopt = min(sv_min, sv_max)

        step(
            "Minimum Stirrup Spacing  [IS 456:2000, Cl. 26.5.1.6]",
            "sv = 0.87 × fy × Asv / (0.4 × b)",
            f"sv = 0.87 × {fy} × {Asv:.4f} / (0.4 × {b})",
            sv_min, "mm",
        )
        add(f"  Maximum spacing = min(0.75d, 300 mm)")
        add(f"    = min(0.75 × {d:.2f}, 300) = min({0.75*d:.2f}, 300) = {sv_max:.0f} mm")
        sv_adopt = math.floor(min(sv_min, sv_max) / 5) * 5  # round down to 5 mm
        add(f"  Adopted sv = {sv_adopt:.0f} mm (rounded to nearest 5 mm)")
        add()
        add(f"  → Provide 2-legged {stir_dia} mm dia stirrups @ {sv_adopt:.0f} mm c/c")
        shear_result = f"Nominal: 2L-{stir_dia}φ @ {sv_adopt:.0f} mm c/c"
    else:
        add(f"  τc ({tau_c:.4f}) < τv ({tau_v:.4f}) ≤ τc,max ({tau_c_max:.4f}) N/mm²")
        add("  → Shear reinforcement is required  [IS 456:2000, Cl. 40.4]")
        add()

        Vus = Vu - tau_c * b * d   # N
        add(f"  Shear resisted by stirrups:")
        add(f"    Vus = Vu − τc × b × d")
        add(f"        = {Vu:.2f} − {tau_c:.4f} × {b} × {d:.4f}")
        add(f"        = {Vus:.4f} N")
        add()

        step(
            "Area of stirrup legs Asv (2-legged)",
            "Asv = 2 × (π × φv² / 4)",
            f"Asv = 2 × π × {stir_dia}² / 4",
            Asv, "mm²",
        )

        # Vus = 0.87 × fy × Asv × d / sv  [IS 456, Cl. 40.4(a)]
        sv_calc = (0.87 * fy * Asv * d) / Vus
        sv_max = min(0.75 * d, 300.0)
        sv_adopt = math.floor(min(sv_calc, sv_max) / 5) * 5

        step(
            "Stirrup Spacing (sv)  [IS 456:2000, Cl. 40.4(a)]",
            "Vus = 0.87 × fy × Asv × d / sv  →  sv = (0.87 × fy × Asv × d) / Vus",
            f"sv = (0.87 × {fy} × {Asv:.4f} × {d:.4f}) / {Vus:.4f}",
            sv_calc, "mm",
        )
        add(f"  Maximum spacing = min(0.75d, 300) = min({0.75*d:.2f}, 300) = {sv_max:.0f} mm")
        add(f"  Adopted sv = {sv_adopt:.0f} mm (rounded to nearest 5 mm)")
        add()
        add(f"  → Provide 2-legged {stir_dia} mm dia stirrups @ {sv_adopt:.0f} mm c/c")
        shear_result = f"Design: 2L-{stir_dia}φ @ {sv_adopt:.0f} mm c/c"

    add()

    # ── Step 11: Deflection Check ─────────────────────────────────────────────
    hdr("STEP 11 — DEFLECTION CHECK  [IS 456:2000, Cl. 23.2]")

    basic_ld = 20   # simply supported beam
    add(f"  Basic L/d ratio = {basic_ld}  [IS 456:2000, Cl. 23.2.1 — Simply Supported]")
    add()

    # Service stress in tension steel
    fs = 0.58 * fy * (Ast_design / Ast_prov)
    add("  Stress in tension steel at service load  [IS 456:2000, Cl. 23.2.1, Fig. 4]:")
    add(f"    fs = 0.58 × fy × (Ast,required / Ast,provided)")
    add(f"       = 0.58 × {fy} × ({Ast_design:.4f} / {Ast_prov:.4f})")
    add(f"       = {fs:.4f} N/mm²")
    add()

    # Modification factor MF from IS 456 Fig 4 (simplified: MF = 310/fs, max 2.0)
    MF = min(310.0 / fs, 2.0) if fs > 0 else 2.0
    add("  Modification Factor (MF) for tension reinforcement:")
    add("    MF = 310 / fs  (max 2.0)  [IS 456:2000, Fig. 4]")
    add(f"       = 310 / {fs:.4f}")
    add(f"       = {MF:.4f}  (capped at 2.0)")
    add()

    ld_perm = basic_ld * MF
    ld_actual = (L * 1000.0) / d

    step(
        "Permissible L/d Ratio",
        "L/d (permissible) = Basic L/d × MF",
        f"L/d = {basic_ld} × {MF:.4f}",
        ld_perm, "",
    )

    step(
        "Actual L/d Ratio",
        "L/d (actual) = L / d  (L in mm)",
        f"L/d = {L*1000:.0f} / {d:.4f}",
        ld_actual, "",
    )

    if ld_actual <= ld_perm:
        add(f"  ✓ Actual L/d ({ld_actual:.4f}) ≤ Permissible L/d ({ld_perm:.4f})")
        add("    → Deflection check SATISFIED")
        deflection_ok = True
    else:
        add(f"  ✗ Actual L/d ({ld_actual:.4f}) > Permissible L/d ({ld_perm:.4f})")
        add("    → Deflection check FAILS. Increase depth or reduce span.")
        deflection_ok = False
    add()

    # ── Step 12: Detailing ────────────────────────────────────────────────────
    hdr("STEP 12 — DETAILING CHECKS  [IS 456:2000, Cl. 26.3 & 26.5]")

    # Minimum width required
    b_min = n_bars * bar_dia + (n_bars - 1) * 25 + 2 * cover + 2 * stir_dia
    add("  Minimum beam width:")
    add("    b_min = n×φ + (n−1)×25 + 2×c' + 2×φv")
    add(f"          = {n_bars}×{bar_dia} + {n_bars-1}×25 + 2×{cover} + 2×{stir_dia}")
    add(f"          = {b_min:.0f} mm")
    add()
    if b >= b_min:
        add(f"  ✓ Provided b ({b} mm) ≥ b_min ({b_min:.0f} mm)  → Width adequate")
    else:
        add(f"  ✗ Provided b ({b} mm) < b_min ({b_min:.0f} mm)  → Increase width or add a second layer")
    add()

    # Side face reinforcement
    add("  Side Face Reinforcement  [IS 456:2000, Cl. 26.5.1.3]:")
    if D > 750:
        add(f"  D = {D} mm > 750 mm → Side face reinforcement required on each face")
        add("    Provide 10 mm dia bars at ≤ 300 mm spacing on each side face")
    else:
        add(f"  D = {D} mm ≤ 750 mm → Side face reinforcement NOT required")
    add()

    # Anchorage check (indicative)
    Ld = (bar_dia * fy) / (4 * 1.6 * 1.25 * math.sqrt(fck))   # development length
    add("  Indicative Development Length (Ld)  [IS 456:2000, Cl. 26.2.1]:")
    add("    Ld = (φ × σs) / (4 × τbd)")
    add(f"    τbd = 1.6 × 1.25 × √fck / 4  (approx. for deformed bars, IS 456 Cl. 26.2.1.1)")
    add(f"    Ld ≈ {Ld:.0f} mm")
    add()

    # ── Design Summary ────────────────────────────────────────────────────────
    add("=" * 72)
    add("  DESIGN SUMMARY")
    add("=" * 72)
    add(f"  Beam Cross-Section      : {b} mm × {D} mm")
    add(f"  Effective Depth (d)     : {d:.2f} mm")
    add(f"  Effective Span (L)      : {L} m")
    add(f"  Grade of Concrete       : M{fck}  (fck = {fck} N/mm²)")
    add(f"  Grade of Steel          : Fe{fy} (fy  = {fy} N/mm²)")
    add(f"  ─────────────────────────────────────────────────────────────────")
    add(f"  Factored Moment  Mu     : {Mu/1.0e6:.4f} kN·m")
    add(f"  Limiting Moment  Mu,lim : {Mu_lim/1.0e6:.4f} kN·m")
    add(f"  Utilisation Ratio       : {Mu/Mu_lim*100:.1f}%")
    add(f"  ─────────────────────────────────────────────────────────────────")
    add(f"  Ast Required            : {Ast_design:.2f} mm²")
    add(f"  Ast Provided            : {Ast_prov:.2f} mm²")
    add(f"  Tension Steel           : {n_bars} — {bar_dia} mm dia bars (bottom)")
    add(f"  Actual NA Depth  xu     : {xu_act:.2f} mm  (xu,max = {xu_max:.2f} mm)")
    add(f"  ─────────────────────────────────────────────────────────────────")
    add(f"  Shear Reinforcement     : {shear_result}")
    add(f"  ─────────────────────────────────────────────────────────────────")
    add(f"  Deflection Check        : {'PASS ✓' if deflection_ok else 'FAIL ✗'}")
    add(f"  L/d Actual / Permissible: {ld_actual:.2f} / {ld_perm:.2f}")
    add("=" * 72)
    add()
    add("  All calculations are in SI units: N, mm, N/mm², kN·m")
    add("  Reference: IS 456:2000 — Plain and Reinforced Concrete — Code of Practice")
    add()

    results = {
        "d": d,
        "Mu": Mu,
        "Mu_lim": Mu_lim,
        "Ast_req": Ast_design,
        "Ast_prov": Ast_prov,
        "n_bars": n_bars,
        "bar_dia": bar_dia,
        "xu_act": xu_act,
        "xu_max": xu_max,
        "Mu_act": Mu_act,
        "tau_v": tau_v,
        "tau_c": tau_c,
        "tau_c_max": tau_c_max,
        "shear_result": shear_result,
        "ld_actual": ld_actual,
        "ld_perm": ld_perm,
        "deflection_ok": deflection_ok,
        "pt": pt,
    }
    return results, lines


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

LATEX_CHAR_MAP = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
}

UNICODE_ASCII_MAP = {
    "—": "--",
    "–": "-",
    "−": "-",
    "×": "x",
    "·": ".",
    "²": "^2",
    "³": "^3",
    "≤": "<=",
    "≥": ">=",
    "→": "->",
    "✓": "PASS",
    "✗": "FAIL",
    "φ": "phi",
    "τ": "tau",
    "Δ": "Delta",
    "γ": "gamma",
    "σ": "sigma",
    "₁": "1",
    "₂": "2",
    "ₘ": "m",
    "ₛ": "s",
    "ₜ": "t",
    "ₙ": "n",
    "ₐ": "a",
    "ₗ": "l",
    "ₓ": "x",
    "ₚ": "p",
    "ₚ": "p",
    "ₖ": "k",
    "ₑ": "e",
    "ₒ": "o",
    "ₗ": "l",
    "₍": "(",
    "₎": ")",
    "⁰": "^0",
    "¹": "^1",
    "⁴": "^4",
    "⁵": "^5",
    "⁶": "^6",
    "⁷": "^7",
    "⁸": "^8",
    "⁹": "^9",
}


def escape_latex(text: str) -> str:
    """Escape text for safe inline LaTeX rendering."""
    escaped = []
    for char in text:
        escaped.append(LATEX_CHAR_MAP.get(char, char))
    return "".join(escaped)


def normalize_report_text(report_text: str) -> str:
    """Convert Unicode-heavy console text to pdflatex-safe ASCII."""
    normalized = report_text
    for src, dest in UNICODE_ASCII_MAP.items():
        normalized = normalized.replace(src, dest)
    return normalized.encode("ascii", errors="replace").decode("ascii")


def latex_safe_ascii(text: str) -> str:
    """Normalize Unicode symbols and escape LaTeX special characters."""
    return escape_latex(normalize_report_text(text))


def build_latex_report(report_text: str, inp: dict, results: dict) -> str:
    """Generate a standalone LaTeX report source for the beam design."""
    status = "PASS" if results["deflection_ok"] else "FAIL"
    utilization = results["Mu"] / results["Mu_lim"] * 100
    detailed_report = normalize_report_text(report_text)

    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{array}}
\usepackage{{booktabs}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{lmodern}}
\usepackage{{longtable}}
\usepackage{{fancyvrb}}

\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.6\baselineskip}}

\begin{{document}}

\begin{{center}}
{{\LARGE \textbf{{Singly Reinforced RCC Beam Design Report}}}}\\[4pt]
{{\large IS 456:2000 -- Limit State Method}}\\[4pt]
Simply Supported Beam under UDL\\[8pt]
Generated on {latex_safe_ascii(str(datetime.date.today()))}
\end{{center}}

\section*{{Input Summary}}

\begin{{tabular}}{{@{{}}ll@{{}}}}
\toprule
Parameter & Value \\
\midrule
Effective span, $L$ & ${inp['span']:.2f}\,\mathrm{{m}}$ \\
Dead load, $w_{{DL}}$ & ${inp['w_DL']:.2f}\,\mathrm{{kN/m}}$ \\
Live load, $w_{{LL}}$ & ${inp['w_LL']:.2f}\,\mathrm{{kN/m}}$ \\
Concrete grade, $f_{{ck}}$ & $\mathrm{{M}}{inp['fck']} \; (f_{{ck}} = {inp['fck']}\,\mathrm{{N/mm^2}})$ \\
Steel grade, $f_y$ & $\mathrm{{Fe}}{inp['fy']} \; (f_y = {inp['fy']}\,\mathrm{{N/mm^2}})$ \\
Beam width, $b$ & ${inp['b']}\,\mathrm{{mm}}$ \\
Overall depth, $D$ & ${inp['D']}\,\mathrm{{mm}}$ \\
Clear cover, $c'$ & ${inp['cover']}\,\mathrm{{mm}}$ \\
Main bar diameter, $\phi$ & ${inp['bar_dia']}\,\mathrm{{mm}}$ \\
Stirrup diameter, $\phi_v$ & ${inp['stir_dia']}\,\mathrm{{mm}}$ \\
\bottomrule
\end{{tabular}}

\section*{{Governing Equations}}

\begin{{align*}}
d &= D - c' - \phi_v - \frac{{\phi}}{{2}} \\
w &= w_{{DL}} + w_{{LL}} \\
M_u &= 1.5\left(\frac{{wL^2}}{{8}}\right) \\
x_{{u,\max}} &= \left(\frac{{x_{{u,\max}}}}{{d}}\right)d \\
M_{{u,\mathrm{{lim}}}} &= 0.36 f_{{ck}} b x_{{u,\max}}\left(d - 0.42x_{{u,\max}}\right) \\
\tau_v &= \frac{{V_u}}{{bd}} \\
p_t &= \frac{{100A_{{st}}}}{{bd}}
\end{{align*}}

\section*{{Design Summary}}

\begin{{tabular}}{{@{{}}ll@{{}}}}
\toprule
Item & Value \\
\midrule
Effective depth, $d$ & ${results['d']:.2f}\,\mathrm{{mm}}$ \\
Factored moment, $M_u$ & ${results['Mu'] / 1.0e6:.3f}\,\mathrm{{kN\cdot m}}$ \\
Limiting moment, $M_{{u,\mathrm{{lim}}}}$ & ${results['Mu_lim'] / 1.0e6:.3f}\,\mathrm{{kN\cdot m}}$ \\
Utilization ratio & {utilization:.1f}\% \\
Required steel, $A_{{st,req}}$ & ${results['Ast_req']:.2f}\,\mathrm{{mm^2}}$ \\
Provided steel, $A_{{st,prov}}$ & ${results['Ast_prov']:.2f}\,\mathrm{{mm^2}}$ \\
Tension reinforcement & ${results['n_bars']}~\mathrm{{bars}}~of~{inp['bar_dia']}\,\mathrm{{mm}}~\phi$ \\
Neutral axis depth, $x_u$ & ${results['xu_act']:.2f}\,\mathrm{{mm}}$ \\
Nominal shear stress, $\tau_v$ & ${results['tau_v']:.3f}\,\mathrm{{N/mm^2}}$ \\
Concrete shear strength, $\tau_c$ & ${results['tau_c']:.3f}\,\mathrm{{N/mm^2}}$ \\
Shear reinforcement & {latex_safe_ascii(results['shear_result'])} \\
Deflection check & {status} \\
Actual / permissible $L/d$ & {results['ld_actual']:.2f} / {results['ld_perm']:.2f} \\
\bottomrule
\end{{tabular}}

\section*{{Design Expressions}}

\begin{{align*}}
A_{{st,req}} &= {results['Ast_req']:.2f}\,\mathrm{{mm^2}} \\
A_{{st,prov}} &= {results['Ast_prov']:.2f}\,\mathrm{{mm^2}} \\
M_u &= {results['Mu'] / 1.0e6:.3f}\,\mathrm{{kN\cdot m}} \\
M_{{u,\mathrm{{lim}}}} &= {results['Mu_lim'] / 1.0e6:.3f}\,\mathrm{{kN\cdot m}} \\
x_u &= {results['xu_act']:.2f}\,\mathrm{{mm}} \\
\tau_v &= {results['tau_v']:.3f}\,\mathrm{{N/mm^2}} \\
\tau_c &= {results['tau_c']:.3f}\,\mathrm{{N/mm^2}}
\end{{align*}}

\section*{{Detailed Calculation Log}}

{{\small
\begin{{Verbatim}}
{detailed_report}
\end{{Verbatim}}
}}

\end{{document}}
"""


def generate_pdf_report(report_text: str, inp: dict, results: dict) -> bytes:
    """Compile a LaTeX report and return the generated PDF bytes."""
    if shutil.which("pdflatex") is None:
        raise RuntimeError("pdflatex is not available in the runtime environment.")

    tex_source = build_latex_report(report_text, inp, results)

    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        temp_path = Path(temp_dir)
        tex_file = temp_path / "beam_report.tex"
        pdf_file = temp_path / "beam_report.pdf"
        tex_file.write_text(tex_source, encoding="utf-8")

        command = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            tex_file.name,
        ]
        for _ in range(2):
            completed = subprocess.run(
                command,
                cwd=temp_path,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                log_snippet = completed.stdout[-2000:] + completed.stderr[-2000:]
                raise RuntimeError(f"LaTeX compilation failed.\n{log_snippet}")

        return pdf_file.read_bytes()


def latex_pdf_available() -> bool:
    """Return True when the LaTeX engine is available in the app runtime."""
    return shutil.which("pdflatex") is not None
  
def main():
    st.set_page_config(
        page_title="RCC Beam Designer | IS 456:2000",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🏗️ Singly Reinforced RCC Beam Design")
    st.caption(
        "**IS 456:2000 — Plain and Reinforced Concrete | Limit State Method | "
        "Simply Supported Beam | UDL Loading**"
    )
    st.markdown("---")

    if not latex_pdf_available():
        st.warning(
            "PDF export needs `pdflatex` in the runtime. "
            "For GitHub/Streamlit deployment, add the required TeX packages "
            "through `packages.txt`."
        )

    # ── Sidebar Inputs ────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📐 Design Inputs")

        with st.expander("📏 Geometry", expanded=True):
            L = st.number_input("Effective Span (L)  [m]", 1.0, 20.0, 5.0, 0.5)
            b = st.number_input("Beam Width (b)  [mm]", 100, 1000, 230, 10)
            D = st.number_input("Total Depth (D)  [mm]", 150, 2000, 450, 10)
            cover = st.number_input("Clear Cover (c')  [mm]", 15, 75, 25, 5)

        with st.expander("⚖️ Loading (UDL)", expanded=True):
            w_DL = st.number_input("Dead Load (w_DL)  [kN/m]", 0.0, 500.0, 15.0, 1.0)
            w_LL = st.number_input("Live Load (w_LL)  [kN/m]", 0.0, 500.0, 10.0, 1.0)

        with st.expander("🧱 Materials", expanded=True):
            fck = st.selectbox("Concrete Grade  [N/mm²]", [15, 20, 25, 30, 35, 40], index=2)
            fy  = st.selectbox("Steel Grade  [N/mm²]",    [250, 415, 500, 550], index=1)

        with st.expander("🔩 Reinforcement Bars", expanded=True):
            bar_dia  = st.selectbox("Main Bar Diameter (φ)  [mm]",  [8, 10, 12, 16, 20, 25, 28, 32], index=4)
            stir_dia = st.selectbox("Stirrup Diameter (φv)  [mm]",  [6, 8, 10, 12], index=1)

        run = st.button("🔨  DESIGN BEAM", type="primary", use_container_width=True)

    # ── Main Panel ────────────────────────────────────────────────────────────
    if not run:
        st.info("👈 Set the design parameters in the sidebar and press **DESIGN BEAM**.")
        with st.expander("📖 Scope of Design (Steps Covered)"):
            st.markdown(
                """
                | Step | Description | IS 456:2000 Clause |
                |------|-------------|-------------------|
                | 1 | Effective depth | Cl. 22.2 |
                | 2 | Factored bending moment (UDL, S.S.) | Table 18 |
                | 3 | Limiting moment of resistance (xu,max/d) | Annex G, Cl. G-1.1 |
                | 4 | Section adequacy (singly vs. doubly reinforced) | Annex G |
                | 5 | Area of tension steel — quadratic solution | Annex G, Eq. G-1.1a |
                | 6 | Min/max steel ratio checks | Cl. 26.5.1.1 |
                | 7 | Bar selection (number and diameter) | — |
                | 8 | Actual neutral axis depth | Annex G |
                | 9 | Actual moment of resistance | Annex G, Eq. G-1.1a |
                | 10 | Shear design (τv, τc, τc,max, stirrup spacing) | Cl. 40 |
                | 11 | Deflection check (L/d method, MF) | Cl. 23.2 |
                | 12 | Detailing — min width, side face reinf., dev. length | Cl. 26 |
                """
            )
        return

    # ── Run Design ────────────────────────────────────────────────────────────
    inp = dict(
        span=L, w_DL=w_DL, w_LL=w_LL,
        fck=fck, fy=fy, b=b, D=D,
        cover=cover, bar_dia=bar_dia, stir_dia=stir_dia,
    )

    with st.spinner("Running IS 456:2000 design calculations …"):
        results, report_lines = design_beam(inp)

    report_text = "\n".join(report_lines)

    # ── Failure Case ──────────────────────────────────────────────────────────
    if results is None:
        st.error(
            "⚠️ **Design could not be completed.** "
            "Review the report below for the reason, then adjust inputs."
        )
        st.subheader("Design Report")
        st.code(report_text, language="text")
        st.download_button(
            "⬇️ Download Report", report_text,
            file_name=f"RCC_Beam_{b}x{D}_FAILED.txt", mime="text/plain",
        )
        return

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    st.subheader("🔢 Key Design Values")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Effective Depth (d)", f"{results['d']:.0f} mm")
    c2.metric("Ast Required", f"{results['Ast_req']:.0f} mm²")
    c3.metric("Ast Provided", f"{results['Ast_prov']:.0f} mm²")
    c4.metric("Tension Steel", f"{results['n_bars']} — {results['bar_dia']}φ")

    c5, c6, c7, c8 = st.columns(4)
    util = results["Mu"] / results["Mu_lim"] * 100
    c5.metric("Mu / Mu,lim", f"{util:.1f}%", delta="Under-reinforced ✓" if util <= 100 else None)
    c6.metric("τv / τc  (N/mm²)", f"{results['tau_v']:.3f} / {results['tau_c']:.3f}")
    c7.metric("L/d  Actual / Perm.", f"{results['ld_actual']:.2f} / {results['ld_perm']:.2f}")
    c8.metric("Deflection", "PASS ✓" if results["deflection_ok"] else "FAIL ✗")

    st.markdown("---")

    # ── Detailed Report ────────────────────────────────────────────────────────
    st.subheader("📋 Detailed Step-by-Step Design Report")
    st.code(report_text, language="text")

    # ── Download ───────────────────────────────────────────────────────────────
    try:
        pdf_bytes = generate_pdf_report(report_text, inp, results)
    except RuntimeError as error:
        st.error(f"LaTeX PDF generation failed: {error}")
        st.download_button(
            label="⬇️ Download LaTeX Source (.tex)",
            data=build_latex_report(report_text, inp, results),
            file_name=f"RCC_Beam_Design_{b}x{D}_M{fck}_Fe{fy}.tex",
            mime="application/x-tex",
            use_container_width=True,
        )
    else:
        st.download_button(
            label="⬇️ Download Full Design Report (.pdf)",
            data=pdf_bytes,
            file_name=f"RCC_Beam_Design_{b}x{D}_M{fck}_Fe{fy}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.caption(
        "All values are in SI units (N, mm, N/mm², kN·m). "
        "Designed as per IS 456:2000 — Plain and Reinforced Concrete — Code of Practice."
    )


if __name__ == "__main__":
    main()
