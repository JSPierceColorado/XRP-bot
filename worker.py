#!/usr/bin/env python3
"""
Perpetual XRP RSI+Trend bot — Every 15 minutes:

BUY when (15m):
  - RSI(14) < 30, and
  - SMA(60) < SMA(240)
Executes a MARKET BUY (no bracket).

SELL when (cross-timeframe):
  - RSI(14) >= 70, and
  - SMA60 (15m) > SMA60 (1h)
Executes a MARKET SELL of 5% of available XRP (configurable).

Logs are plain prints (suited for Railway).
Relies on Coinbase Advanced Trade API.
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

# ====== Config via env ======
PRODUCT_ID        = os.getenv("PRODUCT_ID", "XRP-USD")
GRANULARITY       = os.getenv("GRANULARITY", "FIFTEEN_MINUTE")   # 15m bars
RSI_LEN           = int(os.getenv("RSI_LEN", "14"))
RSI_BUY_THRESH    = float(os.getenv("RSI_BUY_THRESH", "30"))     # entry if RSI < 30
RSI_SELL_THRESH   = float(os.getenv("RSI_SELL_THRESH", "70"))     # exit if RSI >= 70

# Trend filters (15m buy uses 60 vs 240; sell uses 15m 60 vs 1h 60)
MA_FAST_LEN       = int(os.getenv("MA_FAST_LEN", "60"))
MA_SLOW_LEN       = int(os.getenv("MA_SLOW_LEN", "240"))

# Order params
ALLOCATION_PCT     = float(os.getenv("ALLOCATION_PCT", "0.05"))   # 5% of quote balance (buy)
SELL_ALLOCATION_PCT= float(os.getenv("SELL_ALLOCATION_PCT", "0.05"))  # 5% of base balance (sell)

# Deprecated by design change (kept for backwards compat; not used now)
TAKE_PROFIT_PCT   = float(os.getenv("TAKE_PROFIT_PCT", "0.05"))  # UNUSED
STOP_LOSS_PCT     = float(os.getenv("STOP_LOSS_PCT", "0.99"))    # UNUSED

COOLDOWN_MIN      = int(os.getenv("COOLDOWN_MINUTES", "0"))      # optional post-buy cooldown
BAR_ALIGN_OFFSET  = int(os.getenv("BAR_ALIGN_OFFSET_SEC", "5"))  # wait a few seconds after bar close
LOG_PREFIX        = os.getenv("LOG_PREFIX", "[xrp-rsi-bot]")

# Coinbase candle-window constraint: number of candles in [start, end) must be < 350
CANDLES_LIMIT     = int(os.getenv("CANDLES_LIMIT", "300"))

GRAN_SECONDS = {
    "ONE_MINUTE": 60, "FIVE_MINUTE": 300, "FIFTEEN_MINUTE": 900, "THIRTY_MINUTE": 1800,
    "ONE_HOUR": 3600, "TWO_HOUR": 7200, "FOUR_HOUR": 14400, "SIX_HOUR": 21600, "ONE_DAY": 86400,
}

# ====== Globals ======
client = RESTClient()  # needs COINBASE_API_KEY / COINBASE_API_SECRET env
_shutting_down = False
_last_buy_time = None

# ====== Utils & logging ======
def log(msg: str):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    print(f"{LOG_PREFIX} {now} | {msg}", flush=True)

def _handle(sig, _):
    global _shutting_down
    _shutting_down = True
    log(f"Received {sig}. Shutting down gracefully.")
signal.signal(signal.SIGTERM, _handle)
signal.signal(signal.SIGINT, _handle)

def _get(obj, key, default=None):
    if hasattr(obj, key):
        return getattr(obj, key, default)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default

def _get_in(obj, path, default=None):
    cur = obj
    for k in path:
        cur = _get(cur, k, default)
        if cur is default:
            return default
    return cur

# ====== Indicators ======
def rsi(values: List[float], length: int = 14) -> float:
    """Wilder's RSI on the provided series (oldest->newest)."""
    if len(values) < length + 1:
        return float("nan")
    gains = losses = 0.0
    for i in range(1, length + 1):
        d = values[i] - values[i - 1]
        gains  += max(d, 0.0)
        losses += max(-d, 0.0)
    avg_gain = gains / length
    avg_loss = losses / length
    for i in range(length + 1, len(values)):
        d = values[i] - values[i - 1]
        gain = max(d, 0.0)
        loss = max(-d, 0.0)
        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def sma(values: List[float], length: int) -> float:
    """Simple moving average of the last `length` values (latest closed bar)."""
    if length <= 0 or len(values) < length:
        return float("nan")
    return sum(values[-length:]) / length

# ====== Coinbase helpers ======
def get_product_meta(pid: str) -> Dict[str, Decimal]:
    p = client.get_product(product_id=pid)
    return {
        "price_inc": Decimal(p.price_increment),
        "base_inc":  Decimal(p.base_increment),
        "quote_inc": Decimal(p.quote_increment),
        "base_ccy":  p.base_currency_id,
        "quote_ccy": p.quote_currency_id,
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

def get_base_available(base_ccy: str) -> Decimal:
    accs = client.get_accounts()
    for a in accs.accounts:
        if a.currency == base_ccy:
            return Decimal(a.available_balance["value"])
    return Decimal("0")

def latest_price(pid: str) -> Decimal:
    prod = client.get_product(product_id=pid)
    return Decimal(prod.price)

def fetch_candles(pid: str, granularity: str, limit: int):
    """
    Return CLOSED candles oldest->newest in a single request, staying <350 bars.
    We cap `limit` at 300 to respect Coinbase's window rule and add no extra padding.
    """
    seconds = GRAN_SECONDS.get(granularity, 60)
    bars = max(1, min(int(limit), 300))  # hard cap to avoid 350+ error

    now = time.time()
    # Align to last CLOSED bar start
    last_closed_start = math.floor(now / seconds) * seconds - seconds

    # Build a window with exactly `bars` bars ending at the last closed bar.
    end = int(last_closed_start + seconds)     # exclusive-style top
    start = end - bars * seconds               # start so that we span `bars` bars

    res = client.get(
        f"/api/v3/brokerage/products/{pid}/candles",
        params={"start": str(start), "end": str(end), "granularity": granularity, "limit": bars},
    )
    cands = res["candles"] if isinstance(res, dict) else res.candles

    # Normalize and sort oldest -> newest, filter to CLOSED bars only
    def c_start(c): return int(c["start"]) if isinstance(c, dict) else int(c.start)
    cands = sorted(cands, key=c_start)
    candles_closed = [c for c in cands if c_start(c) <= last_closed_start]
    return candles_closed

# ====== Order flow (no bracket) ======
def place_market_buy(pid: str, allocation_pct: float):
    """Create a MARKET BUY (IOC) sized by % of quote balance. No attached TP/SL."""
    meta = get_product_meta(pid)
    quote_bal = get_quote_available(meta["quote_ccy"])
    if quote_bal <= 0:
        log(f"{pid} | No {meta['quote_ccy']} available; skipping BUY.")
        return

    quote_notional = Decimal(str(allocation_pct)) * quote_bal
    if meta["quote_min"] and quote_notional < meta["quote_min"]:
        log(f"{pid} | Notional {quote_notional} < quote_min {meta['quote_min']}; skipping BUY.")
        return
    qs = round_to_inc(quote_notional, meta["quote_inc"])
    if qs <= 0:
        log(f"{pid} | Quote size rounds to 0; skipping BUY.")
        return

    payload = {
        "client_order_id": str(uuid.uuid4()),
        "product_id": pid,
        "side": "BUY",
        "order_configuration": {
            "market_market_ioc": {  # market IOC
                "quote_size": f"{qs.normalize():f}"
            }
        }
    }
    log(f"{pid} | BUY market ~{qs} {meta['quote_ccy']}")
    resp = client.post("/api/v3/brokerage/orders", data=payload)
    success = _get(resp, "success", True)
    if not success:
        log(f"{pid} | BUY failed: {_get(resp, 'error_response')}")
        return
    oid = (_get_in(resp, ["success_response", "order_id"]) or
           _get_in(resp, ["success_response", "orderId"]) or
           _get(resp, "order_id") or _get(resp, "orderId"))
    log(f"{pid} | BUY order_id={oid}")

def place_market_sell_base_pct(pid: str, base_pct: float):
    """Create a MARKET SELL (IOC) sized by % of available base (XRP)."""
    meta = get_product_meta(pid)
    base_bal = get_base_available(meta["base_ccy"])
    if base_bal <= 0:
        log(f"{pid} | No {meta['base_ccy']} available; skipping SELL.")
        return

    base_size = round_to_inc(Decimal(str(base_pct)) * base_bal, meta["base_inc"])
    if base_size <= 0:
        log(f"{pid} | Base size rounds to 0; skipping SELL.")
        return

    payload = {
        "client_order_id": str(uuid.uuid4()),
        "product_id": pid,
        "side": "SELL",
        "order_configuration": {
            "market_market_ioc": {  # market IOC
                "base_size": f"{base_size.normalize():f}"
            }
        }
    }
    log(f"{pid} | SELL market ~{base_size} {meta['base_ccy']}")
    resp = client.post("/api/v3/brokerage/orders", data=payload)
    success = _get(resp, "success", True)
    if not success:
        log(f"{pid} | SELL failed: {_get(resp, 'error_response')}")
        return
    oid = (_get_in(resp, ["success_response", "order_id"]) or
           _get_in(resp, ["success_response", "orderId"]) or
           _get(resp, "order_id") or _get(resp, "orderId"))
    log(f"{pid} | SELL order_id={oid}")

# ====== Entry/Exit rules ======
def should_buy_15m(closes_15m: List[float]) -> bool:
    """BUY when RSI(14) < 30 AND SMA(60) < SMA(240) on latest CLOSED 15m bar."""
    r_closed = rsi(closes_15m, RSI_LEN)
    fast = sma(closes_15m, MA_FAST_LEN)
    slow = sma(closes_15m, MA_SLOW_LEN)

    cmp_txt = (fast < slow) if (not math.isnan(fast) and not math.isnan(slow)) else None
    log(f"{PRODUCT_ID} | BUY chk | 15m RSI{RSI_LEN}={r_closed:.2f} | SMA{MA_FAST_LEN}={fast:.6f} < SMA{MA_SLOW_LEN}={slow:.6f}? {cmp_txt}")

    if math.isnan(r_closed) or math.isnan(fast) or math.isnan(slow):
        return False
    return (r_closed < RSI_BUY_THRESH) and (fast < slow)

def should_sell_cross_tf(closes_15m: List[float], closes_1h: List[float]) -> bool:
    """SELL when RSI(14) >= 70 AND SMA60(15m) > SMA60(1h)."""
    r_closed = rsi(closes_15m, RSI_LEN)
    ma_15m = sma(closes_15m, MA_FAST_LEN)
    ma_1h  = sma(closes_1h,  MA_FAST_LEN)

    cmp_txt = (ma_15m > ma_1h) if (not math.isnan(ma_15m) and not math.isnan(ma_1h)) else None
    log(f"{PRODUCT_ID} | SELL chk | 15m RSI{RSI_LEN}={r_closed:.2f} | SMA60(15m)={ma_15m:.6f} > SMA60(1h)={ma_1h:.6f}? {cmp_txt}")

    if math.isnan(r_closed) or math.isnan(ma_15m) or math.isnan(ma_1h):
        return False
    return (r_closed >= RSI_SELL_THRESH) and (ma_15m > ma_1h)

def sleep_until_next_closed_bar(seconds_per_bar: int, offset: int = 5):
    now = time.time()
    next_bar = (math.floor(now / seconds_per_bar) + 1) * seconds_per_bar
    wake = next_bar + max(0, offset)
    delay = max(1.0, wake - now)
    time.sleep(delay)

# ====== Main loop ======
def main():
    global _last_buy_time
    log(f"Starting XRP bot (15m) | BUY: RSI<{RSI_BUY_THRESH}, SMA{MA_FAST_LEN}<SMA{MA_SLOW_LEN} | SELL: RSI>={RSI_SELL_THRESH}, SMA60(15m)>SMA60(1h)")
    seconds_15m = GRAN_SECONDS.get(GRANULARITY, 900)
    seconds_1h  = GRAN_SECONDS["ONE_HOUR"]

    # Meta once at start
    try:
        meta = get_product_meta(PRODUCT_ID)
        log(f"{PRODUCT_ID} | increments price={meta['price_inc']} base={meta['base_inc']} quote={meta['quote_inc']} base_ccy={meta['base_ccy']} quote_ccy={meta['quote_ccy']}")
    except Exception as e:
        log(f"Product meta error: {e}")
        return

    # Align to next closed 15m bar (+ small offset to ensure closure)
    sleep_until_next_closed_bar(seconds_15m, BAR_ALIGN_OFFSET)

    while not _shutting_down:
        try:
            # --- Data fetch ---
            candles_15m = fetch_candles(PRODUCT_ID, "FIFTEEN_MINUTE", max(300, CANDLES_LIMIT))
            closes_15m = [float(c["close"]) if isinstance(c, dict) else float(c.close) for c in candles_15m]

            # need at least 60 1h bars for SMA60(1h)
            candles_1h = fetch_candles(PRODUCT_ID, "ONE_HOUR", 100)
            closes_1h  = [float(c["close"]) if isinstance(c, dict) else float(c.close) for c in candles_1h]

            # --- Decisions ---
            do_sell = should_sell_cross_tf(closes_15m, closes_1h)
            do_buy  = should_buy_15m(closes_15m)

            # Optional cooldown applies only to buys
            if COOLDOWN_MIN > 0 and do_buy and _last_buy_time is not None:
                if datetime.now(timezone.utc) - _last_buy_time < timedelta(minutes=COOLDOWN_MIN):
                    do_buy = False
                    log(f"{PRODUCT_ID} | In cooldown ({COOLDOWN_MIN} min); skipping BUY.")

            # --- Actions ---
            # Prefer handling SELL first (reduce exposure), then consider BUY
            if do_sell:
                place_market_sell_base_pct(PRODUCT_ID, SELL_ALLOCATION_PCT)

            if do_buy:
                place_market_buy(PRODUCT_ID, ALLOCATION_PCT)
                _last_buy_time = datetime.now(timezone.utc)

        except Exception as e:
            log(f"Error: {e}")

        # Sleep to next closed 15m bar (plus small offset)
        sleep_until_next_closed_bar(seconds_15m, BAR_ALIGN_OFFSET)

if __name__ == "__main__":
    main()
