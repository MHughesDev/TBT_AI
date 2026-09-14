"""Column names, scope rules and tuning constants for v5.

Everything that is a *fact about the archive* lives here. Everything that is a
*choice* is named and given a reason, so the next person can tell the two apart.
"""
from pathlib import Path

VERSION = "v5.0"
PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
BUNDLE = ROOT / "tbt5_bundle.joblib"

# --------------------------------------------------------------- the archive
EPOCH = "2024-01-01"

# Present on every row. Processes, not options — they carry no scope flag.
ALWAYS = ["Material Price", "Fabrication Price"]

# Optional components, each governed by exactly one scope flag.
OPTIONAL = {
    "Construction Price": "IS_CONSTRUCTION",
    "Insulation Material Price": "IS_INSULATION",
    "Insulation Construction Price": "IS_INSULATION_ERECTION",
    "Freight Price": "IS_FREIGHT",
}
COMPONENTS = ALWAYS + list(OPTIONAL)
SCOPE = list(dict.fromkeys(OPTIONAL.values())) + ["IS_TAXABLE"]

TARGET = "Total Price"

# Never inputs. The first group are arithmetic identities with the target; the
# second are pricing-policy decisions made *during* quoting, not tank attributes.
# A model fed either looks excellent and can only score a quote already built.
BANNED = COMPONENTS + [
    TARGET, "Total Tax", "Proposal Total",
    "Margin (%)", "Contingency (%)", "Insulation Margin (%)",
    "Insulation Contingency (%)", "Commission (%)",
]

# Implausibility gate on $/sq-ft. Rows outside it are partial quotes missing
# scope lines. v3 established that these must be excluded from FITTING, not just
# from scoring — filtering evaluation alone measures the model more kindly
# without making it better.
PSF_LO, PSF_HI = 20.0, 250.0

GROUP_COL = "Quote #"
DATE_COL = "Due Date"

# ------------------------------------------------------------------- physics
# Steel: 490 lb/ft^3 => 40.833 lb per square foot per inch of thickness.
LB_PER_SQFT_IN = 490.0 / 12.0

# Shell plate courses are ordered in standard widths. 8 ft is the common mill
# plate width for this class of tank.
COURSE_FT = 8.0

# API 650 one-foot-method constants. See physics.py for the caveat about how
# exactly these should be read.
DESIGN_STRESS_PSI = 23200.0   # A36 design condition, 2/5 of tensile
JOINT_EFFICIENCY = 0.85       # spot radiography
CORROSION_ALLOWANCE_IN = 0.0625
SPECIFIC_GRAVITY = 1.0

# API 650 5.6.1.1 minimum nominal shell thickness by tank diameter. This step
# function is the whole reason v5 exists: below the governing threshold cost
# tracks surface area, above it cost tracks area x thickness. A smooth power law
# in D and H cannot represent the kink.
MIN_SHELL_T_IN = [
    (50.0, 3.0 / 16.0),
    (120.0, 4.0 / 16.0),
    (200.0, 5.0 / 16.0),
    (float("inf"), 6.0 / 16.0),
]

MIN_FLOOR_T_IN = 0.25
ROOF_SELF_SUPPORTING_MAX_D = 60.0   # beyond this a cone roof needs columns
CRANE_CLASS_FT = [30.0, 50.0, 80.0]  # lift-height bands

# ------------------------------------------------------------------- fitting
# Half-life in years for recency weighting. v4 established 1.0 as the best of
# several tried; there is no reason to relitigate it.
HALFLIFE_YEARS = 1.0

# Quantile grid for the conditional distribution. The decision layer needs the
# shape of the distribution, not just its centre — see decision.py.
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)

# LAD (quantile loss at 0.50) is the loss that matches the objective: for small
# errors APE ~= |log(yhat) - log(y)|, so absolute loss in log space is a
# first-order match to mean APE while squared loss is not. v4 measured absolute
# loss as "better p90, notably worse aggregate bias" and rejected it on the bias.
# v5 optimises mean APE explicitly, so it takes the trade.
GBM_BASE = dict(
    loss="absolute_error",
    learning_rate=0.03,
    max_iter=1200,
    min_samples_leaf=15,
    l2_regularization=1.0,
    random_state=0,
)
GBM_ALT = dict(
    loss="absolute_error",
    learning_rate=0.05,
    max_iter=700,
    min_samples_leaf=25,
    l2_regularization=0.3,
    max_leaf_nodes=63,
    random_state=1,
)

# Components get one smaller learner each rather than the full pair. Six of them
# are fitted per model and each is an easier target than the total — a single
# component is smoother, because the scope switch that makes the total lumpy has
# already been applied by selecting the rows. Keeping the retrain inside v4's
# two-minute budget matters more than the last hundredth of a point here:
# retraining monthly instead of quarterly is worth 0.6 points on its own, and a
# retrain nobody schedules because it takes twenty minutes is worth nothing.
GBM_COMPONENT = dict(
    loss="absolute_error",
    learning_rate=0.05,
    max_iter=600,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=2,
)

# Comparables engine.
CMP_K = 12
CMP_MIN_NEIGHBOURS = 3
CMP_HALFLIFE_YEARS = 1.5

# Blend weights are fitted out-of-fold to minimise mean APE rather than fixed.
# This is the starting point if fitting is skipped or degenerate.
BLEND_PRIOR = {"direct": 0.55, "components": 0.35, "comparables": 0.10}
