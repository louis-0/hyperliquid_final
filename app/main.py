"""FastAPI advisor dashboard: a read-only, cost-aware view over the hlq package.

The verdict is honest: for a retail taker the default screen returns "do not deploy", and the
calculator lets a user enter their own fee tier, borrow, and drift to see when, and for whom,
the funding carry becomes deployable against the He et al. (2024) anchors. Read-only, no live
orders. Data is read from the local capture (override the root with HLQ_DATA_ROOT).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import calc, cohort
from hlq import data, features, signals, stats, timing
from hlq.costs import CostModel

BASE = Path(__file__).resolve().parent
COHORT_SNAPSHOT = BASE / "cohort_snapshot.json"
COST = CostModel()
CHASSIS = ["BTC", "ETH", "SOL", "HYPE"]
R11_BORROW_BPS_DAY = 1.0                 # central realistic borrow drag (Chapter 5, R11)
R11_DRIFT_BPS_DAY = 1.0                  # central realistic basis-drift drag

app = FastAPI(title="Cost-aware funding-carry advisor")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")


def _root() -> Path:
    return Path(os.environ.get("HLQ_DATA_ROOT") or data.DATA_ROOT)


def _funding_daily(coin: str, root: Path) -> pd.Series:
    pnl = signals.funding_carry_pnl(data.load_funding(coin, root=root))
    return stats.to_daily(pd.Series(pnl["pnl"].to_numpy(), index=pd.DatetimeIndex(pnl["ts"])))


def _basis_drift_daily(coin: str, root: Path) -> pd.Series:
    pnl = signals.basis_drift_pnl(data.load_funding(coin, root=root),
                                  data.load_spot(coin, root=root),
                                  data.load_marks(coin, root=root))
    return stats.to_daily(pd.Series(pnl["pnl"].to_numpy(), index=pd.DatetimeIndex(pnl["ts"])))


def _net_for(coin: str, fee_bps: float, borrow_bps: float, drift_bps: float, root: Path):
    """Net metrics for a coin under a user's cost stack, on the basis-drift series where spot
    exists, else the funding-only series. Returns None if the coin has no data."""
    try:
        daily, series = _basis_drift_daily(coin, root), "basis-drift"
    except FileNotFoundError:
        try:
            daily, series = _funding_daily(coin, root), "funding-only"
        except FileNotFoundError:
            return None
    m = calc.net_metrics(daily, fee_bps / 1e4, borrow_bps, drift_bps)
    m["coin"], m["series"] = coin, series
    m["verdict_class"] = m["verdict"].split(":", 1)[0].replace(" ", "-")
    return m


def _panel(root: Path):
    """Per-coin degradation ladder (funding-only -> basis-drift -> realistic net Sharpe), the
    macro regime, and the data as-of date, for the signal panel."""
    regime, asof = "unknown", None
    try:
        marks = data.load_marks("BTC", root=root)
        labels = signals.regime_label(marks)
        regime = str(labels.iloc[-1]) if len(labels) else "unknown"
        if len(marks):
            asof = marks["ts"].iloc[-1].date().isoformat()
    except FileNotFoundError:
        pass

    rows = []
    for coin in CHASSIS:
        try:
            funding = _funding_daily(coin, root)
        except FileNotFoundError:
            continue
        row = {"coin": coin, "funding_apr": stats.annualised_sharpe(funding)[1],
               "funding_only": calc.net_metrics(funding, COST.round_trip(), 0.0, 0.0)["net_sharpe"],
               "basis_drift": None, "realistic": None}
        try:
            drift = _basis_drift_daily(coin, root)
            row["basis_drift"] = calc.net_metrics(drift, COST.round_trip(), 0.0, 0.0)["net_sharpe"]
            row["realistic"] = calc.net_metrics(drift, COST.round_trip(),
                                                R11_BORROW_BPS_DAY, R11_DRIFT_BPS_DAY)["net_sharpe"]
        except FileNotFoundError:
            pass
        rows.append(row)
    return regime, asof, rows


_DRIVERS_CACHE: dict = {}


def _drivers(coin: str, root: Path, k: int = 5):
    """Top SHAP drivers of the funding-sign model for a coin, memoised per (coin, data root):
    training the model is the expensive step and the underlying data is a fixed snapshot. Returns
    None when the coin lacks the spot, perp, and funding history the feature matrix needs."""
    key = (coin, str(root))
    if key not in _DRIVERS_CACHE:
        try:
            df = features.build_features(data.load_funding(coin, root=root),
                                         data.load_spot(coin, root=root),
                                         data.load_marks(coin, root=root))
        except FileNotFoundError:
            _DRIVERS_CACHE[key] = None
            return None
        if len(df) < 200:
            _DRIVERS_CACHE[key] = None
            return None
        r = timing.sign_drivers(df, k=k)
        _DRIVERS_CACHE[key] = {
            "coin": coin,
            "drivers": [{"feature": f, "mean_abs_shap": v} for f, v in r["drivers"]],
            "auc": None if math.isnan(r["auc"]) else r["auc"],
            "acc": r["acc"], "naive_acc": r["naive_acc"], "n_test": r["n_test"],
        }
    return _DRIVERS_CACHE[key]


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    root = _root()
    regime, asof, rows = _panel(root)
    coins = [r["coin"] for r in rows] or ["BTC"]
    default = (_net_for("BTC", fee_bps=11.0, borrow_bps=R11_BORROW_BPS_DAY, drift_bps=R11_DRIFT_BPS_DAY, root=root)
               or _net_for(coins[0], fee_bps=11.0, borrow_bps=R11_BORROW_BPS_DAY, drift_bps=R11_DRIFT_BPS_DAY, root=root))
    return templates.TemplateResponse(request=request, name="index.html", context={
        "regime": regime, "asof": asof or "unknown", "rows": rows, "default": default,
        "coins": coins,
        "retail_anchor": calc.RETAIL_ANCHOR, "mm_anchor": calc.MM_ANCHOR,
        "cohort": cohort.load_snapshot(COHORT_SNAPSHOT),
    })


@app.get("/api/net")
def api_net(coin: str = "BTC", fee_bps: float = 11.0, borrow_bps: float = 1.0, drift_bps: float = 1.0):
    m = _net_for(coin, fee_bps, borrow_bps, drift_bps, _root())
    return m or {"error": f"no data for {coin}"}


@app.get("/api/drivers")
def api_drivers(coin: str = "BTC"):
    d = _drivers(coin, _root())
    return d or {"error": f"no funding-sign model for {coin} (needs spot, perp, and funding history)"}


@app.get("/api/cohort")
def api_cohort():
    return cohort.load_snapshot(COHORT_SNAPSHOT) or {"error": "cohort snapshot not built"}
