#!/usr/bin/env python3
"""
Perpetual XRP RSI bot for Coinbase Advanced (Railway-friendly)

- Every minute, fetch 1m candles for XRP-USD and compute RSI(14) on CLOSED bars only
- If RSI <= 30: place a MARKET BUY using 5% of available quote balance (USD by default)
- After the buy fills, place a GTC LIMIT SELL at +2% take-profit
- No stop loss
- Prints readable logs to stdout (Railway Logs)
"""

import os
import time
import uuid
import math
import signal
from decimal import Decimal, ROUND_DOWN
from typing import List, Dict
from datetime import datetime, timezone, timedelta

from coinbase.rest import RESTClient

# ========= Config via env =========
PRODUCT_ID        = os.getenv("PRODUCT_ID", "XRP-USD")
GRANULARITY       = os.getenv("GRANULARITY", "ONE_MINUTE")  # Coinbase enums
RSI_LEN           = int(os.getenv("RSI_LEN", "14"))
RSI_BUY_THRESH    = float(os.getenv("RSI_BUY_THRESH", "30"))
CANDLES_LIMIT     = int(os.getenv("CANDLES_LIMIT", "200"))  # buffer > 14
BAR_ALIGN_OFFSET  = int(os.getenv("BAR_ALIGN_OFFSET_SEC", "5"))  # wait 5s after bar close

ALLOCATION_PCT    = float(os.getenv("ALLOCATION_PCT", "0.05"))   # 5% of quote balance
TAKE_PROFIT_PCT   = float(os.getenv("TAKE_PROFIT_PCT", "0.02"))  # +2% TP
COOLDOWN_MIN      = int(os.getenv("COOLDOWN_MINUTES", "0"))      # optional cool-down after a buy

LOG_PREFIX        = os.getenv("LOG_PREFIX", "[xrp-rsi-bot]")

GRAN_SECONDS = {
    "ONE_MINUTE": 60, "FIVE_MINUTE": 300, "FIFTEEN_MINUTE": 900, "THIRTY_MINUTE": 1800,
    "ONE_HOUR": 3600, "TWO_HOUR": 7200, "FOUR_HOUR": 14400, "SIX_HOUR": 21600, "ONE_DAY": 86400,
}

# ========= Globals =========
client = RESTClient()  # expects COINBASE_API_KEY / COINBASE_API_SECRET in env
_shutting_down = False
_last_buy_time = None


def log(msg: str):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    print(f"{LOG_PREFIX} {now} | {msg}", flush=True)


def _handle(sig, _):
    global _shutting_down
    _shutting_down = True
    log(f"Received {sig}. Shutting down gracefully.")
signal.signal(signal.SIGTERM, _handle)
signal.signal(signal.SIGINT, _handle)


# ========= Indicators =========
def rsi(values: List[float], length: int = 14) -> float:
    """Wilder's RSI on a list of closes (must be oldest->newest)."""
    if len(values) < length + 1:
        return float("nan")

    # seed averages from first 'length' deltas
    gains, losses = 0.0, 0.0
    for i in range(1, length + 1):
        diff = values[i] - values[i - 1]
        gains += max(diff, 0.0)
        losses += max(-diff, 0.0)
    avg_gain = gains / length
    avg_loss = losses / length

    # Wilder smoothing
    for i in range(length + 1, len(values)):
        diff = values[i] - values[i - 1]
        gain = max(diff, 0.0)
        loss = max(-diff, 0.0)
        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ========= Coinbase helpers =========
def get_product_meta(pid: str) -> Dict[str, Decimal]:
    """Fetch increments and quote currency for rounding & balances."""
    p = client.get_product(product_id=pid)
    return {
        "price_inc": Decimal(p.price_increment),
        "base_inc":  Decimal(p.base_increment),
        "quote_inc": Decimal(p.quote_increment),
        "quote_ccy": p.quote_currency_id,
        # Not all SDK versions expose quote_min_size; guard if missing
        "quote_min": Decimal(getattr(p, "quote_min_size", "0")) if hasattr(p, "quote_min_size") else Decimal("0"),
    }


def round_to_inc(value: Decimal, inc: Decimal) -> Decimal:
    if inc <= 0:
        return value
    return (value / inc).to_integral_value(rounding=ROUND_DOWN) * inc


def get_quote_available(quote_ccy: str) -> Decimal:
    accs = client.get_accounts()
    for a in accs.accounts:
        if a.currency == quote_ccy:
            return Decimal(a.available_balance["value"])
    return Decimal("0")


def fetch_candles(pid: str, granularity: str, limit: int):
    """
    Fetch recent candles; return CLOSED candles sorted oldest->newest.
    Coinbase candle 'start' is a unix second timestamp.
    """
    seconds = GRAN_SECONDS.get(granularity, 60)
    end = int(time.time())
    start = end - seconds * (limit + 10)

    res = client.get(
        f"/api/v3/brokerage/products/{pid}/candles",
        params={"start": str(start), "end": str(end), "granularity": granularity, "limit": limit},
    )
    cands = res["candles"] if isinstance(res, dict) else res.candles

    def c_start(c): return int(c["start"]) if isinstance(c, dict) else int(c.start)
    cands = sorted(cands, key=c_start)

    # keep only CLOSED bars: last closed bar starts at end-seconds
    cutoff = end - seconds
    closed = [c for c in cands if c_start(c) <= cutoff]
    return closed


def latest_price(pid: str) -> Decimal:
    prod = client.get_product(product_id=pid)
    return Decimal(prod.price)


def place_market_buy_then_tp(pid: str, allocation_pct: float, tp_pct: float):
    """Place a market buy using quote_size, then a GTC limit sell at +tp_pct."""
    meta = get_product_meta(pid)
    quote_bal = get_quote_available(meta["quote_ccy"])
    if quote_bal <= 0:
        log(f"{pid} | No {meta['quote_ccy']} available; skipping.")
        return

    quote_notional = Decimal(str(allocation_pct)) * quote_bal
    if meta["quote_min"] and quote_notional < meta["quote_min"]:
        log(f"{pid} | Notional {quote_notional} < quote_min {meta['quote_min']}; skipping.")
        return

    qs = round_to_inc(quote_notional, meta["quote_inc"])
    if qs <= 0:
        log(f"{pid} | Quote size rounds to 0; skipping.")
        return

    coid_buy = str(uuid.uuid4())
    log(f"{pid} | BUY market ~{qs} {meta['quote_ccy']} (alloc {allocation_pct:.2%})")
    buy = client.market_order_buy(client_order_id=coid_buy, product_id=pid, quote_size=f"{qs.normalize():f}")
    if not getattr(buy, "success", True):
        log(f"{pid} | Buy failed: {getattr(buy, 'error_response', '')}")
        return

    buy_oid = buy.success_response.order_id if hasattr(buy, "success_response") else buy["success_response"]["order_id"]

    # Wait up to 60s for fill
    filled_base = Decimal("0")
    avg_price = Decimal("0")
    for _ in range(60):
        o = client.get_order(order_id=buy_oid)
        st = o.order.status
        if st in ("FILLED", "CANCELLED", "EXPIRED", "REJECTED"):
            break
        time.sleep(1)

    # Compute fills
    fills = client.list_fills(order_id=buy_oid, product_id=pid)
    f_list = fills.fills if hasattr(fills, "fills") else fills["fills"]
    if not f_list:
        log(f"{pid} | No fills returned (order {buy_oid})")
        return
    for f in f_list:
        size = Decimal(f.size)
        price = Decimal(f.price)
        filled_base += size
        avg_price += size * price
    if filled_base <= 0:
        log(f"{pid} | Filled base size is 0 (order {buy_oid})")
        return
    avg_price /= filled_base

    # Take-profit limit sell
    tp_price = avg_price * Decimal(1 + tp_pct)
    base_sz  = round_to_inc(filled_base, meta["base_inc"])
    limit_px = round_to_inc(tp_price, meta["price_inc"])

    payload = {
        "client_order_id": str(uuid.uuid4()),
        "product_id": pid,
        "side": "SELL",
        "order_configuration": {
            "limit_limit_gtc": {
                "base_size":  f"{base_sz.normalize():f}",
                "limit_price": f"{limit_px.normalize():f}",
                "post_only": False  # set True if you prefer maker-only
            }
        }
    }
    client.post("/api/v3/brokerage/orders", data=payload)
    log(f"{pid} | TP placed: sell {base_sz} @ {limit_px} (avg entry {avg_price})")


def should_buy(rsi_val: float) -> bool:
    return rsi_val <= RSI_BUY_THRESH


def sleep_until_next_closed_bar(seconds_per_bar: int, offset: int = 5):
    """
    Sleep so we act shortly after a new bar closes.
    Example: for 1m bars, with offset=5, we wake at hh:mm:05.
    """
    now = time.time()
    next_bar = (math.floor(now / seconds_per_bar) + 1) * seconds_per_bar
    wake = next_bar + max(0, offset)
    delay = max(1.0, wake - now)
    time.sleep(delay)


def main():
    global _last_buy_time
    log(f"Starting perpetual XRP RSI bot (RSI{RSI_LEN} <= {RSI_BUY_THRESH} => buy {ALLOCATION_PCT:.0%} notional; TP +{int(TAKE_PROFIT_PCT*100)}%; no SL)")
    seconds = GRAN_SECONDS.get(GRANULARITY, 60)

    # Product sanity
    try:
        meta = get_product_meta(PRODUCT_ID)
        log(f"{PRODUCT_ID} | increments price={meta['price_inc']} base={meta['base_inc']} quote={meta['quote_inc']} quote_ccy={meta['quote_ccy']}")
    except Exception as e:
        log(f"Failed to fetch product meta for {PRODUCT_ID}: {e}")
        return

    # Align to just after the next closed bar
    sleep_until_next_closed_bar(seconds, BAR_ALIGN_OFFSET)

    while not _shutting_down:
        try:
            # Fetch CLOSED candles and compute RSI on closed bars only
            candles = fetch_candles(PRODUCT_ID, GRANULARITY, CANDLES_LIMIT)
            closes = [float(c["close"]) if isinstance(c, dict) else float(c.close) for c in candles]
            r_closed = rsi(closes, RSI_LEN)

            # (Optional) also compute RSI including the forming bar for diagnostics
            # NOTE: fetch_candles already excluded partial; so r_incl == r_closed here.
            price = latest_price(PRODUCT_ID)
            log(f"{PRODUCT_ID} | 1m RSI{RSI_LEN} (closed)={r_closed:.2f} | price={price}")

            # Cool-down guard
            can_buy = should_buy(r_closed)
            if COOLDOWN_MIN > 0 and can_buy and _last_buy_time is not None:
                if datetime.now(timezone.utc) - _last_buy_time < timedelta(minutes=COOLDOWN_MIN):
                    can_buy = False
                    log(f"{PRODUCT_ID} | In cooldown ({COOLDOWN_MIN} min); skipping buy.")

            if can_buy:
                place_market_buy_then_tp(PRODUCT_ID, ALLOCATION_PCT, TAKE_PROFIT_PCT)
                _last_buy_time = datetime.now(timezone.utc)

        except Exception as e:
            log(f"Error: {e}")

        # Sleep until the next closed bar plus offset
        sleep_until_next_closed_bar(seconds, BAR_ALIGN_OFFSET)


if __name__ == "__main__":
    main()
