"""Replay a trader.py JSONL log to backtest config changes.

Reads ticks recorded by `trader.py --log ticks.jsonl`, simulates trading with
whatever min_edge / kelly_fraction / spread thresholds you pass, and reports
PnL, hit rate, and a few diagnostics.

Usage:
    python backtest.py ticks.jsonl --min-edge 0.04 --kelly-fraction 0.2

Important: this can only validate what you already logged. It cannot generate
synthetic prices, and it cannot prove forward-looking edge — it can only tell
you "given the prices I saw, would these parameters have done better?". Real
out-of-sample testing means letting paper mode run, *then* tuning, *then*
letting it run again on new data.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class BTPosition:
    market_id: str
    side: str
    shares: float
    entry: float


@dataclass
class BTResult:
    final_equity: float
    starting: float
    pnl: float
    n_trades: int
    n_wins: int
    n_losses: int
    n_markets: int


def iter_ticks(path: str) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def kelly_yes(p: float, q: float) -> float:
    if q <= 0 or q >= 1:
        return 0.0
    return max(0.0, (p - q) / (1 - q))


def simulate(
    path: str,
    starting: float = 1000.0,
    min_edge: float = 0.03,
    kelly_fraction: float = 0.25,
    max_position_pct: float = 0.10,
    max_spread: float = 0.04,
    min_secs: float = 30.0,
) -> BTResult:
    cash = starting
    position: Optional[BTPosition] = None
    n_trades = 0
    n_wins = 0
    n_losses = 0
    seen_markets: set[str] = set()
    last_yes_bid: dict[str, float] = {}
    last_market_id: Optional[str] = None
    last_secs: Optional[float] = None
    last_p_model: Optional[float] = None
    last_yes_ask: Optional[float] = None
    last_no_ask: Optional[float] = None

    for rec in iter_ticks(path):
        mid_ = rec.get("market_id")
        if mid_ is None:
            continue
        seen_markets.add(mid_)

        # market transition: settle on the last known YES_BID-like proxy.
        # We don't have a true outcome in the log unless the log was kept
        # past close, so we approximate settlement using the last yes_mid
        # before transition. If you want true settlement, extend trader.py
        # to log the settled outcomePrices when a market closes.
        if last_market_id is not None and mid_ != last_market_id and position is not None:
            settle_mark = last_yes_bid.get(last_market_id, 0.5)
            payout = settle_mark if position.side == "YES" else (1 - settle_mark)
            proceeds = position.shares * payout
            pnl = proceeds - position.shares * position.entry
            cash += proceeds
            if pnl > 0:
                n_wins += 1
            else:
                n_losses += 1
            position = None

        yes_bid = rec.get("yes_bid")
        yes_ask = rec.get("yes_ask")
        no_ask = 1 - yes_bid if yes_bid is not None else None  # rough proxy
        p_model = rec.get("p_model")
        secs = rec.get("secs")

        if yes_bid is not None:
            last_yes_bid[mid_] = yes_bid
        last_market_id = mid_
        last_secs = secs
        last_p_model = p_model
        last_yes_ask = yes_ask
        last_no_ask = rec.get("no_mid")  # use no_mid if no_ask not logged

        if position is not None:
            continue
        if p_model is None or secs is None or secs < min_secs:
            continue
        if yes_ask is None:
            continue

        yes_spread = (yes_ask - yes_bid) if (yes_bid is not None) else None
        if yes_spread is not None and yes_spread > max_spread:
            continue

        edge_yes = p_model - yes_ask
        side = None
        price = None
        if edge_yes >= min_edge:
            side, price = "YES", yes_ask

        if side is None:
            continue

        equity = cash  # no open pos
        kf = kelly_yes(p_model, price) if side == "YES" else kelly_yes(1 - p_model, price)
        dollars = min(kf * kelly_fraction * equity, equity * max_position_pct, cash)
        if dollars <= 1.0:
            continue
        shares = dollars / price
        cash -= dollars
        position = BTPosition(market_id=mid_, side=side, shares=shares, entry=price)
        n_trades += 1

    # close any final position at last mark
    if position is not None and last_market_id is not None:
        mark = last_yes_bid.get(last_market_id, 0.5)
        payout = mark if position.side == "YES" else (1 - mark)
        proceeds = position.shares * payout
        cash += proceeds
        pnl = proceeds - position.shares * position.entry
        if pnl > 0:
            n_wins += 1
        else:
            n_losses += 1

    return BTResult(
        final_equity=cash,
        starting=starting,
        pnl=cash - starting,
        n_trades=n_trades,
        n_wins=n_wins,
        n_losses=n_losses,
        n_markets=len(seen_markets),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("path")
    p.add_argument("--starting", type=float, default=1000.0)
    p.add_argument("--min-edge", type=float, default=0.03)
    p.add_argument("--kelly-fraction", type=float, default=0.25)
    p.add_argument("--max-position-pct", type=float, default=0.10)
    p.add_argument("--max-spread", type=float, default=0.04)
    p.add_argument("--min-secs", type=float, default=30.0)
    args = p.parse_args()

    r = simulate(
        args.path,
        starting=args.starting,
        min_edge=args.min_edge,
        kelly_fraction=args.kelly_fraction,
        max_position_pct=args.max_position_pct,
        max_spread=args.max_spread,
        min_secs=args.min_secs,
    )
    print(f"markets seen   : {r.n_markets}")
    print(f"trades         : {r.n_trades}")
    print(f"wins / losses  : {r.n_wins} / {r.n_losses}")
    if r.n_wins + r.n_losses:
        print(f"hit rate       : {r.n_wins/(r.n_wins+r.n_losses):.1%}")
    print(f"starting       : ${r.starting:,.2f}")
    print(f"final equity   : ${r.final_equity:,.2f}")
    print(f"pnl            : ${r.pnl:+,.2f}  ({r.pnl/r.starting:+.2%})")
    print()
    print("Caveat: settlement here is approximated from the last logged yes_bid")
    print("on each market. For true backtest accuracy, extend trader.py to log")
    print("settled outcomePrices on market resolution and use those instead.")


if __name__ == "__main__":
    main()
