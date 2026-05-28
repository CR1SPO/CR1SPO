"""Feature extraction from price tapes.

Pure functions on rolling price data. No I/O. No global state.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class PriceTick:
    ts: float
    price: float


@dataclass
class Signals:
    coinbase_price: Optional[float] = None
    binance_price: Optional[float] = None

    ret_30s: Optional[float] = None
    ret_1m: Optional[float] = None
    ret_3m: Optional[float] = None
    ret_5m: Optional[float] = None

    realized_vol_1m: Optional[float] = None
    realized_vol_5m: Optional[float] = None

    cb_binance_basis_bps: Optional[float] = None
    momentum_z: Optional[float] = None

    timestamp: Optional[float] = None


class PriceTape:
    """Rolling tape of (ts, price) ticks with bounded length."""

    def __init__(self, maxlen: int = 3600):
        self.ticks: deque[PriceTick] = deque(maxlen=maxlen)

    def add(self, ts: float, price: float) -> None:
        self.ticks.append(PriceTick(ts, price))

    def last(self) -> Optional[float]:
        return self.ticks[-1].price if self.ticks else None

    def last_ts(self) -> Optional[float]:
        return self.ticks[-1].ts if self.ticks else None

    def price_at_age(self, seconds_ago: float) -> Optional[float]:
        if not self.ticks:
            return None
        target = self.ticks[-1].ts - seconds_ago
        for tick in reversed(self.ticks):
            if tick.ts <= target:
                return tick.price
        return self.ticks[0].price

    def returns_over(self, seconds: float) -> Optional[float]:
        now_p = self.last()
        then_p = self.price_at_age(seconds)
        if now_p is None or then_p is None or then_p <= 0:
            return None
        return math.log(now_p / then_p)

    def realized_vol(self, window_sec: float, sample_sec: float = 5.0) -> Optional[float]:
        """Stddev of log returns sampled every sample_sec over past window_sec.

        Returns per-sqrt-second volatility (so total var over T seconds = sigma^2 * T).
        """
        if len(self.ticks) < 3:
            return None
        now_ts = self.ticks[-1].ts
        samples: list[float] = []
        offset = 0.0
        while offset <= window_sec:
            p = self.price_at_age(offset)
            if p is not None:
                samples.append(p)
            offset += sample_sec
        if len(samples) < 3:
            return None
        samples.reverse()
        rets = [
            math.log(samples[i + 1] / samples[i])
            for i in range(len(samples) - 1)
            if samples[i] > 0
        ]
        if len(rets) < 2:
            return None
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        sd = math.sqrt(var)
        return sd / math.sqrt(sample_sec)


def compute(cb_tape: PriceTape, bn_tape: PriceTape) -> Signals:
    s = Signals(timestamp=datetime.now(timezone.utc).timestamp())
    s.coinbase_price = cb_tape.last()
    s.binance_price = bn_tape.last()

    if s.coinbase_price and s.binance_price and s.binance_price > 0:
        s.cb_binance_basis_bps = (
            (s.coinbase_price - s.binance_price) / s.binance_price * 10000.0
        )

    s.ret_30s = cb_tape.returns_over(30)
    s.ret_1m = cb_tape.returns_over(60)
    s.ret_3m = cb_tape.returns_over(180)
    s.ret_5m = cb_tape.returns_over(300)

    s.realized_vol_1m = cb_tape.realized_vol(60, 5)
    s.realized_vol_5m = cb_tape.realized_vol(300, 10)

    if s.ret_1m is not None and s.realized_vol_5m and s.realized_vol_5m > 0:
        expected_sd_over_1m = s.realized_vol_5m * math.sqrt(60.0)
        if expected_sd_over_1m > 0:
            s.momentum_z = s.ret_1m / expected_sd_over_1m

    return s
