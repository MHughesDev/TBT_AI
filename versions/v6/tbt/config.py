"""Constants, column names, vocabularies and banned lists.

Every name here traces to a section of SPEC.md. Where a constant is
load-bearing the docstring says so and names what breaks.
"""
from __future__ import annotations

# ---------------------------------------------------------------- components
# SPEC 2.1. Order is fixed and is the order components are summed in.
COMPONENTS = [
    "Material Price",
    "Fabrication Price",
    "Construction Price",
    "Insulation Material Price",
    "Insulation Construction Price",
    "Freight Price",
]

# Components that are always present (processes, not options) -- SPEC 2.1.
ALWAYS_PRESENT = ["Material Price", "Fabrication Price"]

# Optional component -> the scope flag that gates it. SPEC 2.4.
OPTIONAL_GATE = {
    "Construction Price": "IS_CONSTRUCTION",
    "Insulation Material Price": "IS_INSULATION",
    "Insulation Construction Price": "IS_INSULATION_ERECTION",
    "Freight Price": "IS_FREIGHT",
}

SCOPE_FLAGS = [
    "IS_CONSTRUCTION",
    "IS_INSULATION",
    "IS_INSULATION_ERECTION",
    "IS_FREIGHT",
    "IS_TAXABLE",
]

# Scope flags that are features of the price models. IS_TAXABLE is deliberately
# excluded -- SPEC 4.3: tax status has no bearing on the pre-tax price of a tank,
# and letting the model see it invites learning customer type by proxy.
SCOPE_FEATURES = [
    "IS_CONSTRUCTION",
    "IS_INSULATION",
    "IS_INSULATION_ERECTION",
    "IS_FREIGHT",
]

TARGET = "Total Price"

# ------------------------------------------------------------------- leakage
# SPEC 4.5. Banned as features under any name, transformation or encoding.
# Enforced by name in features.build_matrix, which raises if any appears.
BANNED = [
    # 1. arithmetic identities of the target
    *COMPONENTS,
    "Total Tax",
    "Proposal Total",
    "Total Price",
    # 2. pricing-policy decisions applied during quoting
    "Margin (%)",
    "Contingency (%)",
    "Insulation Margin (%)",
    "Insulation Contingency (%)",
    "Commission (%)",
    # 3. post-hoc outcome
    "Status",
    # 4. calendar / rework count
    "Revision #",
    # 5. customer identity
    "Company Name",
    "Customer Name",
    # 6. sparse or unsupported identity columns
    "Project Name",
    "Sales Rep",
    "City",
    # 7. sequential identifier encoding time and customer
    "Quote #",
    # 8. derived from geometry, possibly hand-entered
    "Usable Capacity",
    # 9. tax status as a price feature
    "IS_TAXABLE",
]

# ------------------------------------------------------------------ features
NUMERIC_FEATURES_BASE = [
    "log_D",
    "log_H",
    "logD_x_logH",
    "t",
    "log_shell_area",
    "log_floor_area",
    "aspect",
    "freeboard_in",
    "log_quantity",
    "miles_tbt",
    "miles_gt",
    "Ss",
    "S1",
    "name_len",
    "name_words",
    "name_has_digits",
    "name_generic",
]

# Added when the corresponding open question adopts the term (SPEC 2.3).
NUMERIC_FEATURE_DRIFT = "t_x_logD"      # OPEN-1
NUMERIC_FEATURE_PHYSICS = "log_steel_lb_est"  # OPEN-2

CATEGORICAL_FEATURES_BASE = [
    "Material",
    "Use Type",
    "Deck Style",
    "Floor Style",
    "Country",
    "State",
    "Wage Type",
    "Bid Type",
    "use_family",
    "name_family",
]

CATEGORICAL_FEATURE_SALES_MANAGER = "Sales Manager"   # OPEN-5

# --------------------------------------------------------------- model setup
# SPEC 2.5. Starting hyperparameters; the search in B9 may replace variant 1.
GBM_VARIANTS = [
    dict(learning_rate=0.03, max_iter=1200, min_samples_leaf=15,
         l2_regularization=0.0, max_leaf_nodes=31, random_state=1),
    dict(learning_rate=0.05, max_iter=700, min_samples_leaf=25,
         l2_regularization=0.5, max_leaf_nodes=31, random_state=2),
]

RIDGE_ALPHA = 1.0
RECENCY_HALF_LIFE_YEARS = 1.0
# OPEN-4 RESOLVED: w = 1.0, pure component sum. The rule said take 0.7 unless
# 1.0 lands within 0.1 points on mean and 0.5 on top-5% bias, in which case take
# 1.0 for simplicity. On the measured run 1.0 was better on both, so the
# tie-break never had to be used. The direct model is still fitted and stored:
# it costs one seventh of the fit, it is the comparison that would reveal the
# component sum failing on an unusual tank, and re-weighting needs no refit.
# A consequence worth keeping: at w = 1.0 the breakdown sums EXACTLY to the
# point estimate, so there is no 'why doesn't this add up' question on the sheet.
BLEND_WEIGHT = 1.0

# SPEC 3.1 plausibility gate, $/sq-ft of shell area.
PSF_MIN, PSF_MAX = 20.0, 250.0

# SPEC 2.6 conformal.
CONFORMAL_MIN_GROUP = 150
BIG_QUANTILE = 0.95

# SPEC 2.7 retransformation factor clip. Outside this range is an alarm,
# not a correction to apply.
SMEAR_CLIP = (0.90, 1.25)

# SPEC 2.9 staleness thresholds in days.
STALE_DAYS = 45
STALE_HARD_DAYS = 120

# SPEC 2.9 time-feature cap at inference, in years past the last training row.
# LOAD-BEARING: a linear drift term extrapolated a year past the last training
# row is a guess about the market that nobody made.
T_CAP_YEARS = 0.25

EPOCH = "2024-01-01"

# SPEC 2.3 physics constants. NOT to be tuned to the data (SPEC 8.1 R15):
# the term's purpose is the shape; the ridge coefficient sets the scale.
STEEL_G = 1.0          # specific gravity of contents
STEEL_S = 20000.0      # allowable stress, psi
STEEL_E = 0.85         # joint efficiency
STEEL_LB_PER_SQFT_IN = 40.8   # lb per sq ft per inch of thickness
COURSE_HEIGHT_FT = 8.0
FLOOR_THICKNESS_IN = 0.25

# SPEC 6.6 / 7.5 Excel row cap.
UDF_ROW_CAP = 500
