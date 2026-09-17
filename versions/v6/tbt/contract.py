"""Input/output dataclasses, refusal tokens and warning codes.

SPEC 4.7 (inference input contract), 6.5 (negative acceptance), 7.5 (API).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Literal

# ------------------------------------------------------------ refusal tokens
# SPEC 6.5. Each returns the token and NO number.
TOKEN_SCOPE = "#SCOPE"
TOKEN_SCOPE_NEST = "#SCOPE_NEST"
TOKEN_SCOPE_WAGE = "#SCOPE_WAGE"
TOKEN_RANGE = "#RANGE"
TOKEN_UNKNOWN_MATERIAL = "#UNKNOWN:Material"
TOKEN_UNKNOWN_USETYPE = "#UNKNOWN:UseType"
TOKEN_QTY = "#QTY"
TOKEN_MODEL = "#MODEL"
TOKEN_TAXRATE = "#TAXRATE"
TOKEN_UNSUPPORTED = "#UNSUPPORTED"
TOKEN_BATCH = "#BATCH"

ALL_TOKENS = [
    TOKEN_SCOPE, TOKEN_SCOPE_NEST, TOKEN_SCOPE_WAGE, TOKEN_RANGE,
    TOKEN_UNKNOWN_MATERIAL, TOKEN_UNKNOWN_USETYPE, TOKEN_QTY, TOKEN_MODEL,
    TOKEN_TAXRATE, TOKEN_UNSUPPORTED, TOKEN_BATCH,
]

# ------------------------------------------------------------ warning codes
# SPEC 2.9 tier 2 and 3. These return a number AND a code that must render.
WARN_EXTRAP = "EXTRAP"
WARN_BIG = "BIG"
WARN_GEO = "GEO"
WARN_TAXRATE = "TAXRATE"
WARN_AR_BIAS = "AR_BIAS"
WARN_WAGE = "WAGE"
WARN_STALE = "STALE"
WARN_STALE_HARD = "STALE_HARD"
WARN_UNSEEN = "UNSEEN"        # rendered as UNSEEN:<field>

WARNING_TEXT = {
    WARN_EXTRAP: "geometry beyond the usual range",
    WARN_BIG: "large tank: treat as a floor",
    WARN_GEO: "location not seen in training",
    WARN_TAXRATE: "no tax rate on file for this state",
    WARN_AR_BIAS: "Argentina: model runs low here",
    WARN_WAGE: "wage type missing with erection included",
    WARN_STALE: "model is getting old; retrain",
    WARN_STALE_HARD: "model is badly stale; retrain before use",
    WARN_UNSEEN: "unseen value treated as missing",
}


class RefusalError(ValueError):
    """Raised instead of returning a number. SPEC 6.5.

    `code` is one of the TOKEN_* constants and is what renders on the sheet.
    """

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}{(': ' + detail) if detail else ''}")


@dataclass
class QuoteInput:
    """One tank line. SPEC 4.7.

    The five scope booleans are required and have no defaults. A default here
    is the single most likely good-faith regression in the whole system
    (SPEC 8.1 R1): a confidently wrong scope silently changes the price with
    nothing visible on the sheet.
    """
    diameter_ft: float
    height_ft: float
    material: str
    use_type: str
    # scope -- all five required, no defaults (LOAD-BEARING)
    construction: bool
    insulation: bool
    insulation_erection: bool
    freight: bool
    taxable: bool
    # optional descriptive
    freeboard_in: float = 0.0
    quantity: int = 1
    deck_style: str | None = None
    floor_style: str | None = None
    bid_type: str | None = None
    country: str | None = None
    state: str | None = None
    wage_type: str | None = None
    sales_manager: str | None = None
    miles_tbt: float | None = None
    miles_gt: float | None = None
    ss: float | None = None
    s1: float | None = None
    due_date: date | None = None
    tank_name: str | None = None

    def to_row(self) -> dict:
        """Map to archive column names for the feature builder."""
        return {
            "Diameter (ft)": self.diameter_ft,
            "Height (ft)": self.height_ft,
            "Freeboard (in)": self.freeboard_in,
            "Quantity": self.quantity,
            "Material": self.material,
            "Use Type": self.use_type,
            "Deck Style": self.deck_style,
            "Floor Style": self.floor_style,
            "Bid Type": self.bid_type,
            "Country": self.country,
            "State": self.state,
            "Wage Type": self.wage_type,
            "Sales Manager": self.sales_manager,
            "Miles to Site (From TBT)": self.miles_tbt,
            "Miles to Site (From GT)": self.miles_gt,
            "Ss": self.ss,
            "S1": self.s1,
            "Due Date": self.due_date,
            "Tank Name": self.tank_name,
            "IS_CONSTRUCTION": int(bool(self.construction)),
            "IS_INSULATION": int(bool(self.insulation)),
            "IS_INSULATION_ERECTION": int(bool(self.insulation_erection)),
            "IS_FREIGHT": int(bool(self.freight)),
            "IS_TAXABLE": int(bool(self.taxable)),
        }


@dataclass
class Estimate:
    """SPEC 7.5 return shape."""
    point: float                    # conditional median, per tank
    book: float                     # point * smearing factor for its band
    band80: tuple[float, float]
    band90: tuple[float, float]
    components: dict[str, float]    # rescaled to sum to point (SPEC 2.8)
    tax: float | None               # None when taxable and no rate on file
    proposal_total: float | None
    extended_point: float           # point * quantity
    tier: Literal["A", "B", "C"]
    group: str
    warnings: list[str] = field(default_factory=list)
    bundle_id: str = ""
    trained_through: date | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["band80_lo"], d["band80_hi"] = self.band80
        d["band90_lo"], d["band90_hi"] = self.band90
        d.pop("band80"), d.pop("band90")
        return d


@dataclass
class Check:
    """SPEC 7.5 / 7.3 -- the Phase 1 surface."""
    flag: Literal["OK", "LOW", "HIGH"]
    gap_pct: float                       # signed, as % of the estimator's own number
    reasons: list[tuple[str, float]]     # (component, signed pct gap), worst first
    tier: Literal["A", "B", "C"]
    warnings: list[str] = field(default_factory=list)


@dataclass
class ModelInfo:
    bundle_id: str
    trained_through: date | None
    age_days: int
    last_retrain_status: str
    s_rest: float
    s_top5: float
    n_train_rows: int
    bands: dict
    alarms: list[str] = field(default_factory=list)
