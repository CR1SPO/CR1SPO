"""Probability model + Kelly sizing.

The model: a log-normal random walk for BTC, with an optional small drift bias
from observed momentum. Outputs P(BTC_T > reference) where T = expiry.

This is the *right baseline*. It is NOT a magic predictor. Edge comes from
either (a) better volatility estimates than the market, (b) signals the market
hasn't fully priced (microstructure, cross-exchange lead-lag), or (c) the
market being slow to update on news. Don't expect to beat efficient pricing
without one of those.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from signals import Signals


SQRT_2 = math.sqrt(2.0)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


@dataclass
class Prediction:
    p_yes: float
    drift_per_sec: float
    sigma_per_sqrt_sec: float
    components: dict = field(default_factory=dict)


def predict_up_or_down(
    current_price: float,
    reference_price: float,
    seconds_remaining: float,
    signals: Signals,
    momentum_persistence: float = 0.10,
    sigma_floor: float = 1e-7,
) -> Prediction:
    """Estimate P(BTC at expiry > reference) under log-normal dynamics.

    log(P_T) ~ N(log(P_now) + mu*t, sigma^2 * t)
    P(P_T > ref) = Phi((log(P_now/ref) + mu*t) / (sigma*sqrt(t)))

    momentum_persistence: fraction of recent 1-min drift we project forward.
    Set to 0.0 to be a pure martingale model (recommended as a baseline).
    """
    if current_price <= 0 or reference_price <= 0:
        return Prediction(0.5, 0.0, 0.0, {"reason": "bad_input"})
    if seconds_remaining <= 0:
        return Prediction(
            1.0 if current_price > reference_price else 0.0,
            0.0,
            0.0,
            {"reason": "expired"},
        )

    sigma = signals.realized_vol_5m or signals.realized_vol_1m
    if sigma is None or sigma <= 0:
        sigma = sigma_floor
    sigma = max(sigma, sigma_floor)

    momentum_drift = 0.0
    if signals.ret_1m is not None and momentum_persistence > 0:
        momentum_drift = (signals.ret_1m / 60.0) * momentum_persistence

    mu = momentum_drift
    log_ratio = math.log(current_price / reference_price)
    denom = sigma * math.sqrt(seconds_remaining)
    z = (log_ratio + mu * seconds_remaining) / denom if denom > 0 else 0.0
    p_yes = norm_cdf(z)
    p_yes = min(max(p_yes, 1e-6), 1.0 - 1e-6)

    return Prediction(
        p_yes=p_yes,
        drift_per_sec=mu,
        sigma_per_sqrt_sec=sigma,
        components={
            "log_ratio": log_ratio,
            "z": z,
            "momentum_drift": momentum_drift,
        },
    )


def kelly_fraction_yes(p_model: float, yes_price: float) -> float:
    """Optimal Kelly fraction for buying YES at yes_price given true P=p_model.

    Binary contract: bet 1, win (1 - q)/q if YES, lose 1 if NO.
    f* = (p - q) / (1 - q)  (clipped at 0; never short via negative)
    """
    if yes_price >= 1.0 or yes_price <= 0.0:
        return 0.0
    f = (p_model - yes_price) / (1.0 - yes_price)
    return max(0.0, f)


def kelly_fraction_no(p_model: float, no_price: float) -> float:
    return kelly_fraction_yes(1.0 - p_model, no_price)
