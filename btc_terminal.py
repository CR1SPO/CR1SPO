"""BTC x Polymarket live terminal.

Streams BTC-USD from Coinbase and the soonest-ending "Bitcoin Up or Down"
Polymarket market, side-by-side with countdown and orderbook.
"""

import argparse
import asyncio
import json
from collections import deque
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


COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
MARKET_SLUG_HINT = "bitcoin-up-or-down"

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


class State:
    def __init__(self):
        self.btc_price: Optional[float] = None
        self.btc_prev: Optional[float] = None
        self.btc_open: Optional[float] = None
        self.btc_history: deque = deque(maxlen=60)
        self.last_tick: Optional[datetime] = None

        self.market: Optional[dict] = None
        self.yes_token: Optional[str] = None
        self.no_token: Optional[str] = None
        self.yes_mid: Optional[float] = None
        self.no_mid: Optional[float] = None
        self.yes_bid: Optional[float] = None
        self.yes_ask: Optional[float] = None
        self.no_bid: Optional[float] = None
        self.no_ask: Optional[float] = None
        self.volume: Optional[float] = None

        self.recent: deque = deque(maxlen=5)
        self.errors: deque = deque(maxlen=4)

    def err(self, msg: str):
        self.errors.append(f"{datetime.now().strftime('%H:%M:%S')} {msg}")


async def coinbase_stream(state: State):
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
                    state.btc_prev = state.btc_price
                    state.btc_price = price
                    if state.btc_open is None:
                        state.btc_open = price
                    state.btc_history.append(price)
                    state.last_tick = datetime.now(timezone.utc)
        except Exception as e:
            state.err(f"coinbase: {type(e).__name__}")
            await asyncio.sleep(3)


async def find_next_market(client: httpx.AsyncClient) -> Optional[dict]:
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
        slug = (m.get("slug") or "").lower()
        if MARKET_SLUG_HINT in slug:
            return m
    return None


def parse_token_ids(market: dict) -> tuple[Optional[str], Optional[str]]:
    raw = market.get("clobTokenIds") or market.get("clob_token_ids")
    if not raw:
        return None, None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None, None
    if isinstance(raw, list) and len(raw) >= 2:
        return str(raw[0]), str(raw[1])
    return None, None


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
    return float(r.json()["mid"])


async def polymarket_loop(state: State):
    async with httpx.AsyncClient() as client:
        last_refresh = 0.0
        loop = asyncio.get_event_loop()
        while True:
            try:
                now = loop.time()
                stale = state.market is None or (now - last_refresh) > 20
                if stale:
                    m = await find_next_market(client)
                    if m is None:
                        state.market = None
                    elif state.market is None or m.get("id") != state.market.get("id"):
                        if state.market is not None:
                            state.recent.appendleft(state.market)
                        state.market = m
                        state.yes_token, state.no_token = parse_token_ids(m)
                        state.yes_mid = state.no_mid = None
                        state.yes_bid = state.yes_ask = None
                        state.no_bid = state.no_ask = None
                    last_refresh = now

                if state.market and state.yes_token and state.no_token:
                    yes_mid, no_mid, yes_book, no_book = await asyncio.gather(
                        fetch_mid(client, state.yes_token),
                        fetch_mid(client, state.no_token),
                        fetch_book(client, state.yes_token),
                        fetch_book(client, state.no_token),
                        return_exceptions=True,
                    )
                    if isinstance(yes_mid, float):
                        state.yes_mid = yes_mid
                    if isinstance(no_mid, float):
                        state.no_mid = no_mid
                    if isinstance(yes_book, tuple):
                        state.yes_bid, state.yes_ask = yes_book
                    if isinstance(no_book, tuple):
                        state.no_bid, state.no_ask = no_book
                    try:
                        state.volume = float(state.market.get("volume") or 0)
                    except (TypeError, ValueError):
                        state.volume = None

            except Exception as e:
                state.err(f"polymarket: {type(e).__name__}")

            await asyncio.sleep(2)


def render_btc(state: State) -> Panel:
    body = Text()
    if state.btc_price is None:
        body.append("\nconnecting to Coinbase...\n", style="dim")
        return Panel(body, title="BTC-USD spot", border_style="yellow")

    arrow, color = " ", "white"
    if state.btc_prev is not None:
        if state.btc_price > state.btc_prev:
            arrow, color = "▲", "bright_green"
        elif state.btc_price < state.btc_prev:
            arrow, color = "▼", "bright_red"

    body.append("\n")
    body.append(f"  ${state.btc_price:,.2f} ", style=f"bold {color}")
    body.append(f"{arrow}\n", style=color)

    if state.btc_open is not None:
        delta = state.btc_price - state.btc_open
        pct = delta / state.btc_open * 100 if state.btc_open else 0
        d_color = "bright_green" if delta >= 0 else "bright_red"
        body.append(f"\n  session  {delta:+,.2f}  ({pct:+.3f}%)\n", style=d_color)

    if state.btc_history:
        body.append("\n  " + sparkline(state.btc_history) + "\n", style="cyan")

    if state.last_tick:
        age = (datetime.now(timezone.utc) - state.last_tick).total_seconds()
        body.append(f"\n  last tick {age:.1f}s ago\n", style="dim")

    return Panel(body, title="BTC-USD spot · Coinbase", border_style="yellow")


def parse_outcomes(market: dict) -> tuple[str, str]:
    raw = market.get("outcomes")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = None
    if isinstance(raw, list) and len(raw) >= 2:
        return str(raw[0]), str(raw[1])
    return "Yes", "No"


def countdown_str(market: dict) -> Optional[str]:
    end = market.get("endDate") or market.get("end_date_iso")
    if not end:
        return None
    try:
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return None
    remaining = (end_dt - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        return "RESOLVING"
    h, rem = divmod(int(remaining), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def render_market(state: State) -> Panel:
    if state.market is None:
        return Panel(
            Text("\n  searching for next BTC up/down market...\n", style="dim"),
            title="Polymarket",
            border_style="magenta",
        )

    m = state.market
    yes_label, no_label = parse_outcomes(m)

    header = Text(m.get("question") or m.get("slug") or "", style="bold")
    cd = countdown_str(m)
    cd_line = Text()
    if cd:
        style = "bold yellow" if cd != "RESOLVING" else "bold red blink"
        cd_line.append(f"\n  closes in {cd}\n", style=style)

    tbl = Table(show_header=True, header_style="bold", expand=True, pad_edge=False)
    tbl.add_column("Side", style="bold")
    tbl.add_column("Mid",     justify="right")
    tbl.add_column("Bid",     justify="right")
    tbl.add_column("Ask",     justify="right")
    tbl.add_column("Implied", justify="right")

    def fmt(p): return f"{p:.3f}" if p is not None else "—"
    def pct(p): return f"{p*100:5.1f}%" if p is not None else "    —"

    tbl.add_row(
        Text(yes_label, style="bright_green"),
        fmt(state.yes_mid), fmt(state.yes_bid), fmt(state.yes_ask),
        pct(state.yes_mid),
    )
    tbl.add_row(
        Text(no_label, style="bright_red"),
        fmt(state.no_mid), fmt(state.no_bid), fmt(state.no_ask),
        pct(state.no_mid),
    )

    extras = Text()
    if state.yes_bid is not None and state.yes_ask is not None:
        spread = state.yes_ask - state.yes_bid
        extras.append(f"\n  {yes_label} spread {spread:.3f}", style="dim")
    if state.volume is not None:
        extras.append(f"   ·   volume ${state.volume:,.0f}", style="dim")
    extras.append("\n")

    recent = Text()
    if state.recent:
        recent.append("\n  recent markets:\n", style="dim")
        for r in list(state.recent)[:3]:
            recent.append(f"    • {r.get('slug')}\n", style="dim")

    return Panel(
        Group(header, cd_line, tbl, extras, recent),
        title="Polymarket · next BTC up/down",
        border_style="magenta",
    )


def render(state: State) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=6),
    )
    layout["body"].split_row(Layout(name="btc"), Layout(name="market", ratio=2))

    title = Text("  BTC × POLYMARKET TERMINAL  ", style="bold black on cyan")
    ts = Text(datetime.now().strftime(" %H:%M:%S "), style="dim")
    layout["header"].update(Panel(Align.center(Group(title, ts)), border_style="cyan"))
    layout["btc"].update(render_btc(state))
    layout["market"].update(render_market(state))

    foot = Text()
    if state.errors:
        for e in list(state.errors)[-3:]:
            foot.append(f"  {e}\n", style="red dim")
    else:
        foot.append("  feeds healthy\n", style="green dim")
    foot.append("\n  Ctrl-C to exit.  data: Coinbase WS · Polymarket Gamma+CLOB", style="dim")
    layout["footer"].update(Panel(foot, title="status", border_style="grey50"))

    return layout


async def main():
    parser = argparse.ArgumentParser(description="BTC x Polymarket live terminal")
    parser.add_argument("--refresh", type=float, default=0.25,
                        help="UI refresh interval (seconds)")
    args = parser.parse_args()

    state = State()
    console = Console()

    async def ui():
        with Live(render(state), console=console, refresh_per_second=8, screen=True) as live:
            while True:
                await asyncio.sleep(args.refresh)
                live.update(render(state))

    await asyncio.gather(
        coinbase_stream(state),
        polymarket_loop(state),
        ui(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
