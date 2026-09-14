"""Derive a shell-course schedule and steel weight from diameter and height.

Why this module exists
----------------------
v4's open problem #1 is that the model underprices the largest 5% of quotes by
13-17%, and its own diagnosis is correct: large tanks scale *superlinearly* and
nothing in the 42 exported columns explains it. v4's handoff concludes that the
fix is to get plate thickness or the shell course schedule into the export.

That conclusion treats plate thickness as data TBT has and we do not. It is not.
Shell thickness is not measured, it is *designed*, and the design rule is a
published one: the API 650 one-foot method plus a minimum-thickness table keyed
on diameter. Given D and H we can compute the schedule ourselves.

The payoff is not a better number for thickness. It is a better functional form.
Below the governing diameter the minimum-thickness table binds, every course is
3/16" regardless of head, and cost tracks surface area. Above it the one-foot
method binds, thickness grows with D*H, and cost tracks area x thickness, i.e.
D^2*H. That is a *kink* at a fixed physical threshold, and

    a ridge in [log D, log H, log D x log H]

is a smooth power law that cannot represent a kink at any price. Which is exactly
what v4 measured: its backbone alone runs -52% on the large segment. The tree
then has to learn the regime change from scratch, in the thin tail of the data
where it has the fewest rows to learn from.

So v5 hands the model log(shell steel weight) and the regime indicator directly.
The superlinear term stops being something to infer and becomes something we
computed.

How exact is this?
------------------
Not exact, and it does not need to be. These are code-*shaped* proxies: the
constants are representative rather than certified, real designs carry corrosion
allowances, annular rings, nozzle reinforcement and mill plate availability that
we cannot see, and TBT's engineers will not have designed any specific tank the
way this file does.

What matters is that the shape is right — the step at each diameter threshold,
the transition from area-governed to thickness-governed, the discrete course
count. A monotone transform of a feature is free to a tree, so being off by a
constant factor costs nothing. Being off by a *functional form* is what costs
13-17%, and that is the part this fixes.

Nothing here should be quoted to a customer or used as an engineering result.
"""
import numpy as np

from .config import (
    COURSE_FT, LB_PER_SQFT_IN, MIN_SHELL_T_IN, MIN_FLOOR_T_IN,
    DESIGN_STRESS_PSI, JOINT_EFFICIENCY, CORROSION_ALLOWANCE_IN,
    SPECIFIC_GRAVITY, ROOF_SELF_SUPPORTING_MAX_D, CRANE_CLASS_FT,
)

# Plate is ordered in 1/16" increments; a design thickness of 0.31" is bought as
# 5/16". The rounding is what makes the schedule discrete.
PLATE_STEP_IN = 1.0 / 16.0

# Enough courses for a 400 ft tank. Anything taller is not a tank.
MAX_COURSES = 50

PHYSICS_FEATURES = [
    "ph_shell_lb", "ph_floor_lb", "ph_roof_lb", "ph_total_lb",
    "ph_log_shell_lb", "ph_log_total_lb", "ph_lb_per_sqft",
    "ph_t_bottom", "ph_t_top", "ph_t_mean", "ph_t_design_raw",
    "ph_min_governed_frac", "ph_n_courses", "ph_n_plate_steps",
    "ph_wind_h1", "ph_wind_girders", "ph_slenderness",
    "ph_roof_columns", "ph_roof_area", "ph_crane_class",
    "ph_seismic_moment", "ph_hydro_head",
]


def min_shell_thickness(D):
    """API 650 5.6.1.1 minimum nominal shell thickness, a step function of D."""
    D = np.asarray(D, dtype=float)
    out = np.full(D.shape, MIN_SHELL_T_IN[-1][1], dtype=float)
    for lo_bound, t in reversed(MIN_SHELL_T_IN):
        out = np.where(D < lo_bound, t, out)
    return out


def _course_grid(H):
    """(n_rows, MAX_COURSES) arrays of course bottom elevation and course width.

    Courses are laid from the bottom up in COURSE_FT lifts; the top course is
    whatever height is left over. Widths are zero past the top of the tank, which
    is how rows with different course counts share one rectangular array.
    """
    H = np.asarray(H, dtype=float)[:, None]
    z = np.arange(MAX_COURSES, dtype=float)[None, :] * COURSE_FT   # bottom of course
    width = np.clip(H - z, 0.0, COURSE_FT)
    return z, width


def shell_schedule(D, H, freeboard_in=0.0):
    """Per-course design thickness, in inches, on the (n_rows, MAX_COURSES) grid.

    Returns (thickness, width, min_governed) where entries past the top of the
    tank are zero. `min_governed` marks courses where the minimum-thickness table
    binds rather than the one-foot method — the regime flag that matters.
    """
    D = np.asarray(D, dtype=float)
    H = np.asarray(H, dtype=float)
    fb = np.asarray(freeboard_in, dtype=float) / 12.0
    # Liquid height is the shell less freeboard. Freeboard is small and often
    # blank; treat a missing value as zero rather than dropping the row.
    H_liq = np.maximum(H - np.nan_to_num(fb, nan=0.0), 1.0)

    z, width = _course_grid(H)

    # One-foot method: design each course for the head 1 ft above its bottom.
    head = np.maximum(H_liq[:, None] - z - 1.0, 0.0)
    t_design = (2.6 * D[:, None] * head * SPECIFIC_GRAVITY
                / (DESIGN_STRESS_PSI * JOINT_EFFICIENCY)) + CORROSION_ALLOWANCE_IN

    t_min = min_shell_thickness(D)[:, None]
    min_governed = (t_design <= t_min) & (width > 0)
    t = np.maximum(t_design, t_min)

    # Buy plate in 1/16" increments.
    t = np.ceil(t / PLATE_STEP_IN) * PLATE_STEP_IN
    t = np.where(width > 0, t, 0.0)
    return t, width, min_governed


def compute(D, H, freeboard_in=0.0, Ss=None, deck_style=None):
    """All physics features for a batch of tanks. Returns a dict of 1-D arrays."""
    D = np.asarray(D, dtype=float)
    H = np.asarray(H, dtype=float)
    n = D.shape[0]

    t, width, min_gov = shell_schedule(D, H, freeboard_in)
    live = width > 0

    # Shell weight: sum over courses of (circumference x width x thickness).
    shell_lb = (np.pi * D[:, None] * width * t * LB_PER_SQFT_IN).sum(axis=1)

    floor_area = np.pi * (D / 2.0) ** 2
    floor_lb = floor_area * MIN_FLOOR_T_IN * LB_PER_SQFT_IN

    # Cone roof at 2:12 unless the deck is open-top. The structure allowance is a
    # flat multiplier: rafters and a compression ring roughly double plate weight.
    slope = np.arctan2(2.0, 12.0)
    roof_area = floor_area / np.cos(slope)
    open_top = np.zeros(n, dtype=bool)
    if deck_style is not None:
        open_top = np.array(["open" in str(x).lower() for x in deck_style], dtype=bool)
    roof_lb = np.where(open_top, 0.0, roof_area * (3.0 / 16.0) * LB_PER_SQFT_IN * 2.0)

    total_lb = shell_lb + floor_lb + roof_lb

    n_courses = live.sum(axis=1).astype(float)
    t_bottom = t[:, 0]
    # Thickness of the topmost live course.
    top_idx = np.maximum(n_courses.astype(int) - 1, 0)
    t_top = t[np.arange(n), top_idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        t_mean = np.where(live.sum(axis=1) > 0,
                          (t * width).sum(axis=1) / np.maximum(width.sum(axis=1), 1e-9),
                          0.0)
        min_gov_frac = min_gov.sum(axis=1) / np.maximum(n_courses, 1.0)

    # Distinct plate thicknesses in the schedule: a 1-step tank is a commodity
    # shell, a 6-step tank is an engineered one with a different erection cost.
    n_steps = np.array([len(np.unique(np.round(row[m] / PLATE_STEP_IN)))
                        if m.any() else 0
                        for row, m in zip(t, live)], dtype=float)

    # Raw one-foot-method thickness at the base, BEFORE the minimum is applied.
    # Its ratio to the minimum is the continuous version of "which regime".
    head_base = np.maximum(H - 1.0, 0.0)
    t_design_raw = (2.6 * D * head_base * SPECIFIC_GRAVITY
                    / (DESIGN_STRESS_PSI * JOINT_EFFICIENCY))

    # API 650 maximum unstiffened shell height, from the top-course thickness.
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(D > 0, t_top / D, 0.0)
        wind_h1 = 600000.0 * t_top * np.power(np.maximum(ratio, 1e-12), 1.5)
        wind_girders = np.maximum(np.ceil(H / np.maximum(wind_h1, 1e-6)) - 1.0, 0.0)
        slenderness = np.where(D > 0, H / D, 0.0)

    roof_columns = np.where(open_top, 0.0,
                            np.maximum(np.ceil(D / ROOF_SELF_SUPPORTING_MAX_D) - 1.0, 0.0))

    crane_class = np.zeros(n, dtype=float)
    for b in CRANE_CLASS_FT:
        crane_class += (H > b).astype(float)

    ss = np.zeros(n) if Ss is None else np.nan_to_num(np.asarray(Ss, dtype=float))
    # Ringwall overturning scales with seismic weight times lever arm.
    seismic_moment = ss * total_lb * H

    with np.errstate(invalid="ignore", divide="ignore"):
        shell_area = np.pi * D * H
        lb_per_sqft = np.where(shell_area > 0, shell_lb / shell_area, 0.0)

    return {
        "ph_shell_lb": shell_lb,
        "ph_floor_lb": floor_lb,
        "ph_roof_lb": roof_lb,
        "ph_total_lb": total_lb,
        "ph_log_shell_lb": np.log1p(shell_lb),
        "ph_log_total_lb": np.log1p(total_lb),
        "ph_lb_per_sqft": lb_per_sqft,
        "ph_t_bottom": t_bottom,
        "ph_t_top": t_top,
        "ph_t_mean": t_mean,
        "ph_t_design_raw": t_design_raw,
        "ph_min_governed_frac": min_gov_frac,
        "ph_n_courses": n_courses,
        "ph_n_plate_steps": n_steps,
        "ph_wind_h1": wind_h1,
        "ph_wind_girders": wind_girders,
        "ph_slenderness": slenderness,
        "ph_roof_columns": roof_columns,
        "ph_roof_area": np.where(open_top, 0.0, roof_area),
        "ph_crane_class": crane_class,
        "ph_seismic_moment": seismic_moment,
        "ph_hydro_head": head_base,
    }
