#!/usr/bin/env python3
"""
Perpetual XRP RSI+Trend bot — Every 15 minutes, buys when:
  - RSI(14) < 30, and
  - SMA(60) < SMA(240)
Executes a MARKET BUY with an attached TP/SL bracket:
  - Take profit +5% (limit)
  - Stop loss -99% (stop trigger)

Logs are plain prints (suited for Railway).
Relies on Coinbase Advanced Trade API + TriggerBracketGTC.
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

# Trend filter (simple MAs, no displacement)
MA_FAST_LEN       = int(os.getenv("MA_FAST_LEN", "60"))
MA_SLOW_LEN       = int(os.getenv("MA_SLOW_LEN", "240"))

# Order params
ALLOCATION_PCT    = float(os.getenv("ALLOCATION_PCT", "0.05"))   # 5% of quote balance
TAKE_PROFIT_PCT   = float(os.getenv("TAKE_PROFIT_PCT", "0.05"))  # +5% TP
STOP_LOSS_PCT     = float(os.getenv("STOP_LOSS_PCT", "0.99"))    # 99% below entry (very wide)

COOLDOWN_MIN      = int(os.getenv("COOLDOWN_MINUTES", "0"))      # optional post-fill cooldown
BAR_ALIGN_OFFSET  = int(os.getenv("BAR_ALIGN_OFFSET_SEC", "5"))  # wait a few seconds after bar close
LOG_PREFIX        = os.getenv("LOG_PREFIX", "[xrp-rsi-bot]")

GRAN_SECONDS = {
    "ONE_MINUTE": 60, "FIVE_MINUTE": 300, "FIFTEEN_MINUTE": 900, "THIRTY_MINUTE": 1800,
    "ONE_HOUR": 3600, "TWO_HOUR": 7200, "FOUR_HOUR": 14400, "SIX_HOUR": 21600, "ONE_DAY": 86400,
}

def required_bars() -> int:
    """
    Minimum history required for the indicators on the latest CLOSED bar.
    Add a cushion for stability.
    """
    need = max(MA_SLOW_LEN, MA_FAST_LEN, RSI_LEN + 1)
    return need + 60  # small cushion

# Allow override via env; otherwise ensure ample history
CANDLES_LIMIT     = int(os.getenv("CANDLES_LIMIT", str(max(400, required_bars()))))

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

def latest_price(pid: str) -> Decimal:
    prod = client.get_product(product_id=pid)
    return Decimal(prod.price)

def fetch_candles(pid: str, granularity: str, limit: int):
    """
    Return CLOSED candles oldest->newest.
    Uses Coinbase Advanced Trade GET /products/{product_id}/candles.
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
    cutoff = end - seconds  # last closed bar start
    return [c for c in cands if c_start(c) <= cutoff]

# ====== Order flow: market buy WITH attached TP/SL bracket ======
def place_market_buy_with_bracket(pid: str, allocation_pct: float, tp_pct: float, sl_pct: float):
    """
    Create a MARKET buy (IOC) with an attached TriggerBracketGTC:
      - TP at +tp_pct (limit)
      - SL at -sl_pct (stop trigger)
    Attached order size is inherited from the parent (omit size in the bracket).
    """
    meta = get_product_meta(pid)
    quote_bal = get_quote_available(meta["quote_ccy"])
    if quote_bal <= 0:
        log(f"{pid} | No {meta['quote_ccy']} available; skipping.")
        return

    # Notional & rounding
    quote_notional = Decimal(str(allocation_pct)) * quote_bal
    if meta["quote_min"] and quote_notional < meta["quote_min"]:
        log(f"{pid} | Notional {quote_notional} < quote_min {meta['quote_min']}; skipping.")
        return
    qs = round_to_inc(quote_notional, meta["quote_inc"])
    if qs <= 0:
        log(f"{pid} | Quote size rounds to 0; skipping.")
        return

    # Compute bracket prices from current price (market order → use latest)
    px = latest_price(pid)
    tp_px = round_to_inc(px * Decimal(1 + tp_pct), meta["price_inc"])
    sl_px = round_to_inc(px * Decimal(1 - sl_pct), meta["price_inc"])  # 99% below entry proxy

    payload = {
        "client_order_id": str(uuid.uuid4()),
        "product_id": pid,
        "side": "BUY",
        "order_configuration": {
            "market_market_ioc": {  # market IOC
                "quote_size": f"{qs.normalize():f}"
            }
        },
        "attached_order_configuration": {
            "trigger_bracket_gtc": {
                # omit size; inherits parent size
                "limit_price": f"{tp_px.normalize():f}",          # take-profit
                "stop_trigger_price": f"{sl_px.normalize():f}"    # stop (very low)
            }
        }
    }

    log(f"{pid} | BUY market ~{qs} {meta['quote_ccy']} with TP={tp_px} (+{int(tp_pct*100)}%), SL={sl_px} (-{int(sl_pct*100)}%)")
    resp = client.post("/api/v3/brokerage/orders", data=payload)

    success = _get(resp, "success", True)
    if not success:
        log(f"{pid} | Create order failed: {_get(resp, 'error_response')}")
        return

    oid = (
        _get_in(resp, ["success_response", "order_id"]) or
        _get_in(resp, ["success_response", "orderId"]) or
        _get(resp, "order_id") or
        _get(resp, "orderId")
    )
    log(f"{pid} | Parent order_id={oid} (bracket attached)")

# ====== Entry rule & timing ======
def should_buy(closes: List[float]) -> bool:
    """
    Entry only when RSI(14) < 30 AND SMA(60) < SMA(240) on the latest CLOSED bar.
    """
    r_closed = rsi(closes, RSI_LEN)
    fast = sma(closes, MA_FAST_LEN)
    slow = sma(closes, MA_SLOW_LEN)

    # Log details for visibility
    log(f"{PRODUCT_ID} | 15m RSI{RSI_LEN}={r_closed:.2f} | SMA{MA_FAST_LEN}={fast:.6f} < SMA{MA_SLOW_LEN}={slow:.6f}? {fast < slow if not math.isnan(fast) and not math.isnan(slow) else 'nan'}")

    if math.isnan(r_closed) or math.isnan(fast) or math.isnan(slow):
        return False
    return (r_closed < RSI_BUY_THRESH) and (fast < slow)

def sleep_until_next_closed_bar(seconds_per_bar: int, offset: int = 5):
    now = time.time()
    next_bar = (math.floor(now / seconds_per_bar) + 1) * seconds_per_bar
    wake = next_bar + max(0, offset)
    delay = max(1.0, wake - now)
    time.sleep(delay)

# ====== Main loop ======
def main():
    global _last_buy_time
    log(f"Starting XRP RSI+Trend bot (15m, RSI{RSI_LEN}< {RSI_BUY_THRESH}, SMA{MA_FAST_LEN}<SMA{MA_SLOW_LEN} → buy {ALLOCATION_PCT:.0%} with TP +{int(TAKE_PROFIT_PCT*100)}%, SL {int(STOP_LOSS_PCT*100)}% below)")
    seconds = GRAN_SECONDS.get(GRANULARITY, 900)

    # Meta once at start (helps surface increments/quote ccy early)
    try:
        meta = get_product_meta(PRODUCT_ID)
        log(f"{PRODUCT_ID} | increments price={meta['price_inc']} base={meta['base_inc']} quote={meta['quote_inc']} quote_ccy={meta['quote_ccy']}")
    except Exception as e:
        log(f"Product meta error: {e}")
        return

    # Align to next closed 15m bar (+ small offset to ensure closure)
    sleep_until_next_closed_bar(seconds, BAR_ALIGN_OFFSET)

    while not _shutting_down:
        try:
            candles = fetch_candles(PRODUCT_ID, GRANULARITY, CANDLES_LIMIT)
            closes = [float(c["close"]) if isinstance(c, dict) else float(c.close) for c in candles]

            # Decision
            can_buy = should_buy(closes)

            # Optional cooldown
            if COOLDOWN_MIN > 0 and can_buy and _last_buy_time is not None:
                if datetime.now(timezone.utc) - _last_buy_time < timedelta(minutes=COOLDOWN_MIN):
                    can_buy = False
                    log(f"{PRODUCT_ID} | In cooldown ({COOLDOWN_MIN} min); skipping buy.")

            if can_buy:
                place_market_buy_with_bracket(PRODUCT_ID, ALLOCATION_PCT, TAKE_PROFIT_PCT, STOP_LOSS_PCT)
                _last_buy_time = datetime.now(timezone.utc)

        except Exception as e:
            log(f"Error: {e}")

        # Sleep to next 15m closed bar (plus small offset)
        sleep_until_next_closed_bar(seconds, BAR_ALIGN_OFFSET)

if __name__ == "__main__":
    main()
