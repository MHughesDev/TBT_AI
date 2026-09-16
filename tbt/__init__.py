"""TBT tank pricing estimator.

Built to SPEC.md. The section references in each module are to that document
and are the authority on why a thing is shaped the way it is.
"""
from .contract import (Check, Estimate, ModelInfo, QuoteInput, RefusalError)
from .scoring import check, estimate, estimate_many, model_info
from .loader import LoadRejected, load_training, usable_mask
from .model import Bundle, fit_bundle, predict_frame
from .protocol import backtest, cadence_simulation
from .train import load_bundle, train

__version__ = "1.0.0"

__all__ = [
    "QuoteInput", "Estimate", "Check", "ModelInfo", "RefusalError",
    "estimate", "estimate_many", "check", "model_info",
    "load_training", "usable_mask", "LoadRejected",
    "Bundle", "fit_bundle", "predict_frame",
    "backtest", "cadence_simulation",
    "train", "load_bundle",
]
