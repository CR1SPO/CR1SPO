"""BTC x Polymarket paper trader.

Streams BTC from Coinbase + Binance, computes signals, predicts P(YES) on the
next Polymarket "bitcoin up or down" market, and paper-trades when there's
positive edge after spread + fractional Kelly sizing.

Live mode (--live + --pk) is NOT implemented here intentionally. Validate in
paper mode first. Then implement signing/order placement against the CLOB
exchange contracts. Don't skip the validation step.

Usage:
    python trader.py                          # paper trade, default config
    python trader.py --bankroll 5000
    python trader.py --min-edge 0.05          # only trade when edge >= 5c
    python trader.py --log ticks.jsonl        # log everything for backtest
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx
import websockets
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from signals import PriceTape, Signals, compute
from model import Prediction, predict_up_or_down, kelly_fraction_yes, kelly_fraction_no


COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
BINANCE_WS = "wss://stream.binance.com:9443/ws/btcusdt@trade"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
MARKET_SLUG = "bitcoin-up-or-down"


@dataclass
class Position:
    market_id: str
    market_question: str
    side: str            # "YES" or "NO"
    shares: float
    entry_price: float
    opened_at: float


@dataclass
class ClosedPosition:
    market_id: str
    market_question: str
    side: str
    shares: float
    entry_price: float
    payout_per_share: float
    pnl: float
    opened_at: float
    closed_at: float


@dataclass
class MarketView:
    market: dict
    yes_token: str
    no_token: str
    yes_mid: Optional[float] = None
    no_mid: Optional[float] = None
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    no_bid: Optional[float] = None
    no_ask: Optional[float] = None
    volume: Optional[float] = None
    reference_price: Optional[float] = None
    first_seen_at: Optional[float] = None
    prediction: Optional[Prediction] = None

    @property
    def yes_label(self) -> str:
        return _outcomes(self.market)[0]

    @property
    def no_label(self) -> str:
        return _outcomes(self.market)[1]

    def seconds_remaining(self) -> Optional[float]:
        end = self.market.get("endDate")
        if not end:
            return None
        try:
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except ValueError:
            return None
        return (end_dt - datetime.now(timezone.utc)).total_seconds()


@dataclass
class Config:
    bankroll: float = 1000.0
    min_edge: float = 0.03
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.10
    min_time_to_close: float = 30.0
    max_spread: float = 0.04
    paper: bool = True
    log_path: Optional[str] = None


@dataclass
class State:
    cfg: Config
    cb_tape: PriceTape = field(default_factory=lambda: PriceTape(3600))
    bn_tape: PriceTape = field(default_factory=lambda: PriceTape(3600))
    btc_price: Optional[float] = None
    btc_prev: Optional[float] = None
    btc_session_open: Optional[float] = None
    last_tick: Optional[datetime] = None

    signals: Signals = field(default_factory=Signals)
    market: Optional[MarketView] = None
    recent_market_ids: set = field(default_factory=set)

    cash: float = 0.0
    starting_cash: float = 0.0
    open_positions: list[Position] = field(default_factory=list)
    closed_positions: deque = field(default_factory=lambda: deque(maxlen=50))

    errors: deque = field(default_factory=lambda: deque(maxlen=5))
    actions: deque = field(default_factory=lambda: deque(maxlen=20))

    log_file = None

    def err(self, msg: str) -> None:
        self.errors.append(f"{datetime.now().strftime('%H:%M:%S')} {msg}")

    def log_action(self, msg: str) -> None:
        self.actions.append(f"{datetime.now().strftime('%H:%M:%S')} {msg}")

    def equity_estimate(self) -> float:
        equity = self.cash
        for p in self.open_positions:
            mark = p.entry_price
            if self.market and p.market_id == self.market.market.get("id"):
                m = self.market.yes_mid if p.side == "YES" else self.market.no_mid
                if m is not None:
                    mark = m
            equity += p.shares * mark
        return equity

    @property
    def n_wins(self) -> int:
        return sum(1 for c in self.closed_positions if c.pnl > 0)

    @property
    def n_losses(self) -> int:
        return sum(1 for c in self.closed_positions if c.pnl <= 0)


def _outcomes(market: dict) -> tuple[str, str]:
    raw = market.get("outcomes")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = None
    if isinstance(raw, list) and len(raw) >= 2:
        return str(raw[0]), str(raw[1])
    return "Yes", "No"


def _parse_token_ids(market: dict) -> tuple[Optional[str], Optional[str]]:
    raw = market.get("clobTokenIds") or market.get("clob_token_ids")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None, None
    if isinstance(raw, list) and len(raw) >= 2:
        return str(raw[0]), str(raw[1])
    return None, None


async def coinbase_stream(state: State) -> None:
    sub = json.dumps({
        "type": "subscribe",
        "product_ids": ["BTC-USD"],
        "channels": ["ticker"],
    })
    while True:
        try:
            async with websockets.connect(COINBASE_WS, ping_interval=20) as ws:
                await ws.send(sub)
                async for msg in ws:
                    data = json.loads(msg)
                    if data.get("type") != "ticker":
                        continue
                    price = float(data["price"])
                    ts = datetime.now(timezone.utc).timestamp()
                    state.cb_tape.add(ts, price)
                    state.btc_prev = state.btc_price
                    state.btc_price = price
                    if state.btc_session_open is None:
                        state.btc_session_open = price
                    state.last_tick = datetime.now(timezone.utc)
        except Exception as e:
            state.err(f"coinbase: {type(e).__name__}")
            await asyncio.sleep(3)


async def binance_stream(state: State) -> None:
    while True:
        try:
            async with websockets.connect(BINANCE_WS, ping_interval=20) as ws:
                async for msg in ws:
                    data = json.loads(msg)
                    if "p" not in data:
                        continue
                    price = float(data["p"])
                    ts = datetime.now(timezone.utc).timestamp()
                    state.bn_tape.add(ts, price)
        except Exception as e:
            state.err(f"binance: {type(e).__name__}")
            await asyncio.sleep(5)


async def find_market(client: httpx.AsyncClient) -> Optional[dict]:
    r = await client.get(
        f"{GAMMA_API}/markets",
        params={
            "closed": "false",
            "active": "true",
            "limit": 100,
            "order": "endDate",
            "ascending": "true",
        },
        timeout=10,
    )
    r.raise_for_status()
    for m in r.json():
        if MARKET_SLUG in (m.get("slug") or "").lower():
            return m
    return None


async def get_market_by_id(client: httpx.AsyncClient, market_id: str) -> Optional[dict]:
    r = await client.get(f"{GAMMA_API}/markets/{market_id}", timeout=10)
    if r.status_code != 200:
        return None
    return r.json()


async def fetch_book(client: httpx.AsyncClient, token_id: str):
    r = await client.get(f"{CLOB_API}/book", params={"token_id": token_id}, timeout=5)
    r.raise_for_status()
    book = r.json()
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    best_bid = float(bids[-1]["price"]) if bids else None
    best_ask = float(asks[-1]["price"]) if asks else None
    return best_bid, best_ask


async def fetch_mid(client: httpx.AsyncClient, token_id: str) -> Optional[float]:
    r = await client.get(f"{CLOB_API}/midpoint", params={"token_id": token_id}, timeout=5)
    if r.status_code != 200:
        return None
    try:
        return float(r.json()["mid"])
    except (KeyError, ValueError):
        return None


async def settle_resolved_market(
    state: State, client: httpx.AsyncClient, market_id: str
) -> None:
    """Fetch a resolved market and settle any open positions on it."""
    settled = await get_market_by_id(client, market_id)
    if not settled:
        state.err(f"could not settle market {market_id}")
        return
    raw_prices = settled.get("outcomePrices")
    if isinstance(raw_prices, str):
        try:
            raw_prices = json.loads(raw_prices)
        except json.JSONDecodeError:
            raw_prices = None
    if not raw_prices or len(raw_prices) < 2:
        state.err(f"no outcomePrices on {market_id}")
        return
    yes_payout = float(raw_prices[0])
    no_payout = float(raw_prices[1])

    still_open: list[Position] = []
    now_ts = datetime.now(timezone.utc).timestamp()
    for p in state.open_positions:
        if p.market_id != market_id:
            still_open.append(p)
            continue
        payout = yes_payout if p.side == "YES" else no_payout
        proceeds = p.shares * payout
        state.cash += proceeds
        pnl = proceeds - p.shares * p.entry_price
        closed = ClosedPosition(
            market_id=p.market_id,
            market_question=p.market_question,
            side=p.side,
            shares=p.shares,
            entry_price=p.entry_price,
            payout_per_share=payout,
            pnl=pnl,
            opened_at=p.opened_at,
            closed_at=now_ts,
        )
        state.closed_positions.appendleft(closed)
        state.log_action(
            f"SETTLE {p.side} {p.shares:.1f}sh @ {p.entry_price:.3f} -> {payout:.0f}  "
            f"pnl {pnl:+.2f}"
        )
    state.open_positions = still_open


def maybe_trade(state: State) -> None:
    """Decide whether to open a position on the current market."""
    m = state.market
    if m is None or m.prediction is None:
        return
    if any(p.market_id == m.market.get("id") for p in state.open_positions):
        return
    secs = m.seconds_remaining()
    if secs is None or secs < state.cfg.min_time_to_close:
        return

    p_model = m.prediction.p_yes
    equity = state.equity_estimate()
    cap = equity * state.cfg.max_position_pct

    yes_ask = m.yes_ask
    no_ask = m.no_ask
    yes_spread = (m.yes_ask - m.yes_bid) if (m.yes_ask and m.yes_bid) else None
    no_spread = (m.no_ask - m.no_bid) if (m.no_ask and m.no_bid) else None

    # Edge vs ask (the price we'd actually pay)
    yes_edge = (p_model - yes_ask) if yes_ask is not None else None
    no_edge = ((1 - p_model) - no_ask) if no_ask is not None else None

    side = None
    fill_price = None
    spread_ok = False
    if yes_edge is not None and yes_edge >= state.cfg.min_edge:
        if yes_spread is None or yes_spread <= state.cfg.max_spread:
            side, fill_price, spread_ok = "YES", yes_ask, True
    if no_edge is not None and no_edge >= state.cfg.min_edge:
        if no_spread is None or no_spread <= state.cfg.max_spread:
            if side is None or no_edge > yes_edge:
                side, fill_price, spread_ok = "NO", no_ask, True

    if side is None or fill_price is None or not spread_ok:
        return

    if side == "YES":
        kf = kelly_fraction_yes(p_model, fill_price)
    else:
        kf = kelly_fraction_no(p_model, fill_price)
    if kf <= 0:
        return

    dollars = min(kf * state.cfg.kelly_fraction * equity, cap, state.cash)
    if dollars <= 1.0:  # minimum trade
        return

    shares = dollars / fill_price
    state.cash -= dollars
    pos = Position(
        market_id=str(m.market.get("id")),
        market_question=m.market.get("question") or m.market.get("slug") or "",
        side=side,
        shares=shares,
        entry_price=fill_price,
        opened_at=datetime.now(timezone.utc).timestamp(),
    )
    state.open_positions.append(pos)
    state.log_action(
        f"OPEN  {side} {shares:.1f}sh @ {fill_price:.3f}  "
        f"p={p_model:.3f} edge={(p_model - fill_price if side=='YES' else (1-p_model) - fill_price):+.3f}  "
        f"${dollars:.0f}"
    )


def log_tick(state: State) -> None:
    if state.log_file is None:
        return
    m = state.market
    rec = {
        "ts": datetime.now(timezone.utc).timestamp(),
        "btc": state.btc_price,
        "bn": state.bn_tape.last(),
        "ret_1m": state.signals.ret_1m,
        "vol_5m": state.signals.realized_vol_5m,
        "basis_bps": state.signals.cb_binance_basis_bps,
        "market_id": m.market.get("id") if m else None,
        "market_slug": m.market.get("slug") if m else None,
        "yes_mid": m.yes_mid if m else None,
        "yes_bid": m.yes_bid if m else None,
        "yes_ask": m.yes_ask if m else None,
        "no_mid":  m.no_mid if m else None,
        "ref":     m.reference_price if m else None,
        "secs":    m.seconds_remaining() if m else None,
        "p_model": m.prediction.p_yes if (m and m.prediction) else None,
        "equity":  state.equity_estimate(),
    }
    state.log_file.write(json.dumps(rec) + "\n")
    state.log_file.flush()


async def polymarket_loop(state: State) -> None:
    async with httpx.AsyncClient() as client:
        loop = asyncio.get_event_loop()
        last_search = 0.0
        while True:
            try:
                now = loop.time()
                stale = state.market is None or (now - last_search) > 15
                if stale:
                    m = await find_market(client)
                    current_id = state.market.market.get("id") if state.market else None
                    if m and str(m.get("id")) != str(current_id):
                        if current_id:
                            await settle_resolved_market(state, client, str(current_id))
                        yes_t, no_t = _parse_token_ids(m)
                        if yes_t and no_t:
                            state.market = MarketView(
                                market=m,
                                yes_token=yes_t,
                                no_token=no_t,
                                reference_price=state.btc_price,
                                first_seen_at=datetime.now(timezone.utc).timestamp(),
                            )
                            state.log_action(
                                f"NEW market {m.get('slug')}  ref=${state.btc_price or 0:,.2f}"
                            )
                    elif m is None and state.market:
                        await settle_resolved_market(state, client, str(current_id))
                        state.market = None
                    last_search = now

                mv = state.market
                if mv is not None:
                    yes_mid, no_mid, yes_book, no_book = await asyncio.gather(
                        fetch_mid(client, mv.yes_token),
                        fetch_mid(client, mv.no_token),
                        fetch_book(client, mv.yes_token),
                        fetch_book(client, mv.no_token),
                        return_exceptions=True,
                    )
                    if isinstance(yes_mid, float):
                        mv.yes_mid = yes_mid
                    if isinstance(no_mid, float):
                        mv.no_mid = no_mid
                    if isinstance(yes_book, tuple):
                        mv.yes_bid, mv.yes_ask = yes_book
                    if isinstance(no_book, tuple):
                        mv.no_bid, mv.no_ask = no_book
                    try:
                        mv.volume = float(mv.market.get("volume") or 0)
                    except (TypeError, ValueError):
                        mv.volume = None

            except Exception as e:
                state.err(f"polymarket: {type(e).__name__}: {e}")

            await asyncio.sleep(2)


async def think_loop(state: State) -> None:
    """Recompute signals + prediction + trading decisions every second."""
    while True:
        try:
            state.signals = compute(state.cb_tape, state.bn_tape)
            mv = state.market
            if (
                mv is not None
                and state.btc_price is not None
                and mv.reference_price is not None
            ):
                secs = mv.seconds_remaining()
                if secs is not None and secs > 0:
                    mv.prediction = predict_up_or_down(
                        current_price=state.btc_price,
                        reference_price=mv.reference_price,
                        seconds_remaining=secs,
                        signals=state.signals,
                    )
                    maybe_trade(state)
            log_tick(state)
        except Exception as e:
            state.err(f"think: {type(e).__name__}: {e}")
        await asyncio.sleep(1.0)


# ----- UI -----

SPARK = " ▁▂▃▄▅▆▇█"


def sparkline(values, width: int = 50) -> str:
    if not values:
        return ""
    vs = list(values)[-width:]
    lo, hi = min(vs), max(vs)
    if hi == lo:
        return SPARK[4] * len(vs)
    return "".join(
        SPARK[int((v - lo) / (hi - lo) * (len(SPARK) - 1))] for v in vs
    )


def fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def render_btc(state: State) -> Panel:
    body = Text()
    if state.btc_price is None:
        body.append("\n  connecting to Coinbase...\n", style="dim")
        return Panel(body, title="BTC-USD spot", border_style="yellow")

    arrow, color = " ", "white"
    if state.btc_prev is not None:
        if state.btc_price > state.btc_prev:
            arrow, color = "▲", "bright_green"
        elif state.btc_price < state.btc_prev:
            arrow, color = "▼", "bright_red"
    body.append(f"\n  ${state.btc_price:,.2f} ", style=f"bold {color}")
    body.append(f"{arrow}\n", style=color)

    if state.btc_session_open is not None:
        delta = state.btc_price - state.btc_session_open
        pct = (delta / state.btc_session_open) * 100 if state.btc_session_open else 0
        c = "bright_green" if delta >= 0 else "bright_red"
        body.append(f"\n  session  {delta:+,.2f}  ({pct:+.3f}%)\n", style=c)

    prices = [t.price for t in list(state.cb_tape.ticks)[-60:]]
    if prices:
        body.append("\n  " + sparkline(prices) + "\n", style="cyan")

    bn = state.bn_tape.last()
    if bn:
        basis = state.signals.cb_binance_basis_bps
        b_color = "green" if (basis or 0) > 0 else "red"
        body.append(
            f"\n  binance ${bn:,.2f}   basis "
            f"{(basis or 0):+.1f}bps\n",
            style=f"dim {b_color}",
        )

    return Panel(body, title="BTC spot · Coinbase vs Binance", border_style="yellow")


def render_signals(state: State) -> Panel:
    s = state.signals
    tbl = Table(show_header=False, expand=True, pad_edge=False)
    tbl.add_column(style="bold")
    tbl.add_column(justify="right")

    def fmt(x, pct=False, dp=4):
        if x is None:
            return "—"
        if pct:
            return f"{x*100:+.{dp}f}%"
        return f"{x:+.{dp}f}"

    tbl.add_row("ret 30s", fmt(s.ret_30s, pct=True, dp=3))
    tbl.add_row("ret 1m",  fmt(s.ret_1m, pct=True, dp=3))
    tbl.add_row("ret 3m",  fmt(s.ret_3m, pct=True, dp=3))
    tbl.add_row("ret 5m",  fmt(s.ret_5m, pct=True, dp=3))
    tbl.add_row("vol 1m / √s",
                f"{s.realized_vol_1m:.6f}" if s.realized_vol_1m else "—")
    tbl.add_row("vol 5m / √s",
                f"{s.realized_vol_5m:.6f}" if s.realized_vol_5m else "—")
    tbl.add_row("momentum z", f"{s.momentum_z:+.2f}" if s.momentum_z else "—")
    tbl.add_row("cb-bn basis", f"{s.cb_binance_basis_bps:+.2f} bps"
                if s.cb_binance_basis_bps else "—")
    return Panel(tbl, title="signals", border_style="blue")


def render_market(state: State) -> Panel:
    mv = state.market
    if mv is None:
        return Panel(
            Text("\n  searching...\n", style="dim"),
            title="Polymarket",
            border_style="magenta",
        )

    secs = mv.seconds_remaining()
    cd = "—"
    cd_style = "yellow"
    if secs is not None:
        if secs <= 0:
            cd, cd_style = "RESOLVING", "red blink"
        else:
            h, rem = divmod(int(secs), 3600)
            m_, s_ = divmod(rem, 60)
            cd = f"{h:02d}:{m_:02d}:{s_:02d}"

    head = Text()
    head.append(mv.market.get("question") or mv.market.get("slug") or "", style="bold")
    head.append(f"\n  closes in {cd}", style=f"bold {cd_style}")
    if mv.reference_price is not None and state.btc_price is not None:
        delta = state.btc_price - mv.reference_price
        c = "bright_green" if delta > 0 else "bright_red" if delta < 0 else "white"
        head.append(
            f"\n  ref ${mv.reference_price:,.2f}    now ${state.btc_price:,.2f}    "
            f"Δ {delta:+,.2f}\n", style=c)

    tbl = Table(show_header=True, header_style="bold", expand=True, pad_edge=False)
    tbl.add_column("Side", style="bold")
    tbl.add_column("Bid",     justify="right")
    tbl.add_column("Ask",     justify="right")
    tbl.add_column("Model P", justify="right")
    tbl.add_column("Edge",    justify="right")

    def f(x): return f"{x:.3f}" if x is not None else "—"
    p_model = mv.prediction.p_yes if mv.prediction else None
    p_no = (1 - p_model) if p_model is not None else None

    def edge_yes():
        if p_model is None or mv.yes_ask is None: return None
        return p_model - mv.yes_ask
    def edge_no():
        if p_no is None or mv.no_ask is None: return None
        return p_no - mv.no_ask

    e_y, e_n = edge_yes(), edge_no()
    def edge_cell(e):
        if e is None: return Text("—")
        color = "bright_green" if e >= state.cfg.min_edge else (
            "yellow" if e > 0 else "dim red")
        return Text(f"{e:+.3f}", style=color)

    tbl.add_row(Text(mv.yes_label, style="bright_green"),
                f(mv.yes_bid), f(mv.yes_ask), f(p_model), edge_cell(e_y))
    tbl.add_row(Text(mv.no_label, style="bright_red"),
                f(mv.no_bid), f(mv.no_ask), f(p_no), edge_cell(e_n))

    extras = Text()
    if mv.prediction:
        extras.append(
            f"\n  σ={mv.prediction.sigma_per_sqrt_sec:.3e}/√s   "
            f"μ={mv.prediction.drift_per_sec:+.3e}/s\n",
            style="dim",
        )
    if mv.volume:
        extras.append(f"  volume {fmt_money(mv.volume)}\n", style="dim")

    return Panel(Group(head, tbl, extras), title="Polymarket", border_style="magenta")


def render_book(state: State) -> Panel:
    cfg = state.cfg
    equity = state.equity_estimate()
    pnl = equity - state.starting_cash
    color = "bright_green" if pnl >= 0 else "bright_red"

    head = Text()
    head.append("PAPER" if cfg.paper else "LIVE", style="bold black on yellow")
    head.append("  ")
    head.append(f"equity {fmt_money(equity)}  ", style=f"bold {color}")
    head.append(f"pnl {pnl:+,.2f}  ", style=color)
    head.append(f"cash {fmt_money(state.cash)}\n", style="dim")
    head.append(
        f"  config: edge≥{cfg.min_edge:.2f}  kelly={cfg.kelly_fraction}  "
        f"maxpos={cfg.max_position_pct:.0%}  spread≤{cfg.max_spread:.2f}\n",
        style="dim",
    )
    wins, losses = state.n_wins, state.n_losses
    total = wins + losses
    if total:
        head.append(
            f"  record: {wins}W / {losses}L  ({wins/total:.0%})\n",
            style="dim",
        )

    tbl = Table(title="open positions", show_header=True, header_style="bold", expand=True)
    tbl.add_column("Side")
    tbl.add_column("Shares", justify="right")
    tbl.add_column("Entry",  justify="right")
    tbl.add_column("Mark",   justify="right")
    tbl.add_column("uPnL",   justify="right")
    for p in state.open_positions:
        mark = p.entry_price
        if state.market and p.market_id == state.market.market.get("id"):
            m_ = state.market.yes_mid if p.side == "YES" else state.market.no_mid
            if m_ is not None:
                mark = m_
        upnl = (mark - p.entry_price) * p.shares
        c = "bright_green" if upnl >= 0 else "bright_red"
        tbl.add_row(
            Text(p.side, style="bright_green" if p.side == "YES" else "bright_red"),
            f"{p.shares:.1f}",
            f"{p.entry_price:.3f}",
            f"{mark:.3f}",
            Text(f"{upnl:+,.2f}", style=c),
        )
    if not state.open_positions:
        tbl.add_row("—", "—", "—", "—", "—")

    return Panel(Group(head, tbl), title="book", border_style="cyan")


def render_log(state: State) -> Panel:
    body = Text()
    if not state.actions:
        body.append("\n  no trades yet\n", style="dim")
    else:
        for line in list(state.actions)[-8:]:
            style = "bright_green" if "OPEN" in line else (
                "yellow" if "SETTLE" in line else "white")
            body.append(f"  {line}\n", style=style)
    return Panel(body, title="trade log", border_style="grey50")


def render(state: State) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="row1", ratio=2),
        Layout(name="row2", ratio=2),
        Layout(name="footer", size=4),
    )
    layout["row1"].split_row(
        Layout(name="btc", ratio=2),
        Layout(name="signals", ratio=1),
        Layout(name="market", ratio=3),
    )
    layout["row2"].split_row(
        Layout(name="book", ratio=3),
        Layout(name="log", ratio=2),
    )

    title = Text("  BTC × POLYMARKET PAPER TRADER  ", style="bold black on cyan")
    ts = Text(datetime.now().strftime(" %H:%M:%S "), style="dim")
    layout["header"].update(Panel(Align.center(Group(title, ts)), border_style="cyan"))
    layout["btc"].update(render_btc(state))
    layout["signals"].update(render_signals(state))
    layout["market"].update(render_market(state))
    layout["book"].update(render_book(state))
    layout["log"].update(render_log(state))

    foot = Text()
    if state.errors:
        for e in list(state.errors)[-2:]:
            foot.append(f"  {e}\n", style="red dim")
    else:
        foot.append("  feeds healthy\n", style="green dim")
    foot.append("  Ctrl-C to exit.  This is PAPER trading — no real money is at risk.",
                style="dim")
    layout["footer"].update(Panel(foot, title="status", border_style="grey50"))
    return layout


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--max-position-pct", type=float, default=0.10)
    parser.add_argument("--max-spread", type=float, default=0.04)
    parser.add_argument("--min-time", type=float, default=30.0,
                        help="min seconds-to-close to enter a new position")
    parser.add_argument("--log", type=str, default=None,
                        help="JSONL path to record ticks/signals/decisions")
    parser.add_argument("--refresh", type=float, default=0.25)
    parser.add_argument("--live", action="store_true",
                        help="NOT IMPLEMENTED. Paper mode only. Exits.")
    args = parser.parse_args()

    if args.live:
        sys.stderr.write(
            "Live mode is deliberately not implemented. Validate in paper mode for "
            "weeks, confirm positive PnL net of fees + slippage + withdrawal cost, "
            "then implement signing against the CLOB exchange.\n"
        )
        sys.exit(2)

    cfg = Config(
        bankroll=args.bankroll,
        min_edge=args.min_edge,
        kelly_fraction=args.kelly_fraction,
        max_position_pct=args.max_position_pct,
        max_spread=args.max_spread,
        min_time_to_close=args.min_time,
        paper=True,
        log_path=args.log,
    )
    state = State(cfg=cfg, cash=cfg.bankroll, starting_cash=cfg.bankroll)
    if args.log:
        state.log_file = open(args.log, "a", buffering=1)

    console = Console()

    async def ui() -> None:
        with Live(render(state), console=console, refresh_per_second=8, screen=True) as live:
            while True:
                await asyncio.sleep(args.refresh)
                live.update(render(state))

    try:
        await asyncio.gather(
            coinbase_stream(state),
            binance_stream(state),
            polymarket_loop(state),
            think_loop(state),
            ui(),
        )
    finally:
        if state.log_file:
            state.log_file.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
