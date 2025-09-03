#!/usr/bin/env python3
"""
Perpetual XRP RSI+Trend bot — Every 15 minutes:

BUY when (15m):
  - RSI(14) < 30, and
  - SMA(60) < SMA(240)
Executes a MARKET BUY (no bracket).

SELL when (15m) **and profit gate**:
  - RSI(14) >= 70, and
  - SMA(60) > SMA(240), and
  - Current price >= (avg_entry_price * (1 + MIN_PROFIT_PCT))
Executes a MARKET SELL of 5% of available XRP.

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
PRODUCT_ID          = os.getenv("PRODUCT_ID", "XRP-USD")
GRANULARITY         = os.getenv("GRANULARITY", "FIFTEEN_MINUTE")
RSI_LEN             = int(os.getenv("RSI_LEN", "14"))
RSI_BUY_THRESH      = float(os.getenv("RSI_BUY_THRESH", "30"))
RSI_SELL_THRESH     = float(os.getenv("RSI_SELL_THRESH", "70"))

MA_FAST_LEN         = int(os.getenv("MA_FAST_LEN", "60"))
MA_SLOW_LEN         = int(os.getenv("MA_SLOW_LEN", "240"))

ALLOCATION_PCT      = float(os.getenv("ALLOCATION_PCT", "0.05"))        # 5% of quote balance for buys
SELL_ALLOCATION_PCT = float(os.getenv("SELL_ALLOCATION_PCT", "0.05"))   # 5% of base balance for sells
MIN_PROFIT_PCT      = float(os.getenv("MIN_PROFIT_PCT", "0.05"))        # +5% gate for sells

COOLDOWN_MIN        = int(os.getenv("COOLDOWN_MINUTES", "0"))
BAR_ALIGN_OFFSET    = int(os.getenv("BAR_ALIGN_OFFSET_SEC", "5"))
LOG_PREFIX          = os.getenv("LOG_PREFIX", "[xrp-rsi-bot]")

CANDLES_LIMIT       = int(os.getenv("CANDLES_LIMIT", "300"))

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
    """Wilder's RSI."""
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

def fetch_candles(pid: str, granularity: str, limit: int):
    seconds = GRAN_SECONDS.get(granularity, 60)
    bars = max(1, min(int(limit), 300))

    now = time.time()
    last_closed_start = math.floor(now / seconds) * seconds - seconds

    end = int(last_closed_start + seconds)
    start = end - bars * seconds

    res = client.get(
        f"/api/v3/brokerage/products/{pid}/candles",
        params={"start": str(start), "end": str(end), "granularity": granularity, "limit": bars},
    )
    cands = res["candles"] if isinstance(res, dict) else res.candles

    def c_start(c): return int(c["start"]) if isinstance(c, dict) else int(c.start)
    cands = sorted(cands, key=c_start)
    return [c for c in cands if c_start(c) <= last_closed_start]

# --- NEW: fills & cost basis -----------------------------------------------
def fetch_fills_for_product(pid: str, limit: int = 250):
    """
    Fetch recent fills for a product. We only need a reasonable window to
    estimate average entry of the current inventory.
    """
    try:
        res = client.get(
            "/api/v3/brokerage/orders/historical/fills",
            params={"product_id": pid, "limit": limit},
        )
        fills = res["fills"] if isinstance(res, dict) else getattr(res, "fills", [])
        # Sort ascending by time if available
        def f_time(f):
            return _get(f, "trade_time") or _get(f, "created_at") or _get(f, "time") or ""
        return sorted(fills, key=f_time)
    except Exception as e:
        log(f"{pid} | fills fetch error: {e}")
        return []

def compute_avg_cost_from_fills(fills) -> Decimal | None:
    """
    Moving-average inventory method:
      - On BUY: add units and cost = price * size
      - On SELL: remove units at current average cost
    Returns Decimal average cost for remaining inventory, or None if unknown/zero.
    """
    units = Decimal("0")
    cost  = Decimal("0")
    for f in fills:
        side = str(_get(f, "side", "")).upper()
        # Coinbase may use 'size' or 'base_size' for base quantity
        size_s  = _get(f, "size") or _get(f, "base_size") or _get_in(f, ["filled_size"]) or "0"
        price_s = _get(f, "price") or "0"
        try:
            qty   = Decimal(str(size_s))
            price = Decimal(str(price_s))
        except Exception:
            continue
        if qty <= 0 or price <= 0:
            continue

        if side == "BUY":
            units += qty
            cost  += qty * price
        elif side == "SELL":
            if units <= 0:
                # Nothing on book; skip consumption to avoid negative inventory
                continue
            # consume at average cost
            avg = cost / units if units > 0 else Decimal("0")
            consume = min(qty, units)
            cost  -= avg * consume
            units -= consume

    if units > 0:
        return (cost / units)
    return None

def is_min_profit_met(pid: str, min_profit_pct: float, current_price: Decimal) -> bool:
    fills = fetch_fills_for_product(pid)
    avg_cost = compute_avg_cost_from_fills(fills)
    if avg_cost is None or avg_cost <= 0:
        log(f"{pid} | Profit gate: unknown avg cost; skipping SELL for safety.")
        return False
    threshold = avg_cost * Decimal(str(1.0 + min_profit_pct))
    ok = current_price >= threshold
    log(f"{pid} | Profit gate: avg_cost={avg_cost:.6f} | price={current_price:.6f} | need≥{threshold:.6f} ? {ok}")
    return bool(ok)
# ---------------------------------------------------------------------------

# ====== Orders ======
def place_market_buy(pid: str, allocation_pct: float):
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
            "market_market_ioc": {"quote_size": f"{qs.normalize():f}"}
        }
    }
    log(f"{pid} | BUY market ~{qs} {meta['quote_ccy']}")
    resp = client.post("/api/v3/brokerage/orders", data=payload)
    if not _get(resp, "success", True):
        log(f"{pid} | BUY failed: {_get(resp, 'error_response')}")
    else:
        oid = (_get_in(resp, ["success_response", "order_id"])
               or _get_in(resp, ["success_response", "orderId"])
               or _get(resp, "order_id")
               or _get(resp, "orderId"))
        log(f"{pid} | BUY order_id={oid}")

def place_market_sell_base_pct(pid: str, base_pct: float):
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
            "market_market_ioc": {"base_size": f"{base_size.normalize():f}"}
        }
    }
    log(f"{pid} | SELL market ~{base_size} {meta['base_ccy']}")
    resp = client.post("/api/v3/brokerage/orders", data=payload)
    if not _get(resp, "success", True):
        log(f"{pid} | SELL failed: {_get(resp, 'error_response')}")
    else:
        oid = (_get_in(resp, ["success_response", "order_id"])
               or _get_in(resp, ["success_response", "orderId"])
               or _get(resp, "order_id")
               or _get(resp, "orderId"))
        log(f"{pid} | SELL order_id={oid}")

# ====== Rules ======
def should_buy_15m(closes: List[float]) -> bool:
    r_closed = rsi(closes, RSI_LEN)
    fast = sma(closes, MA_FAST_LEN)
    slow = sma(closes, MA_SLOW_LEN)
    cmp_txt = (fast < slow) if (not math.isnan(fast) and not math.isnan(slow)) else None
    log(f"{PRODUCT_ID} | BUY chk | RSI{RSI_LEN}={r_closed:.2f} | SMA{MA_FAST_LEN}={fast:.6f} < SMA{MA_SLOW_LEN}={slow:.6f}? {cmp_txt}")
    if math.isnan(r_closed) or math.isnan(fast) or math.isnan(slow):
        return False
    return (r_closed < RSI_BUY_THRESH) and (fast < slow)

def should_sell_15m_signal_only(closes: List[float]) -> bool:
    """Original technical SELL signal (no PnL)."""
    r_closed = rsi(closes, RSI_LEN)
    fast = sma(closes, MA_FAST_LEN)
    slow = sma(closes, MA_SLOW_LEN)
    cmp_txt = (fast > slow) if (not math.isnan(fast) and not math.isnan(slow)) else None
    log(f"{PRODUCT_ID} | SELL chk | RSI{RSI_LEN}={r_closed:.2f} | SMA{MA_FAST_LEN}={fast:.6f} > SMA{MA_SLOW_LEN}={slow:.6f}? {cmp_txt}")
    if math.isnan(r_closed) or math.isnan(fast) or math.isnan(slow):
        return False
    return (r_closed >= RSI_SELL_THRESH) and (fast > slow)

def sleep_until_next_closed_bar(seconds_per_bar: int, offset: int = 5):
    now = time.time()
    next_bar = (math.floor(now / seconds_per_bar) + 1) * seconds_per_bar
    wake = next_bar + max(0, offset)
    delay = max(1.0, wake - now)
    time.sleep(delay)

# ====== Main loop ======
def main():
    global _last_buy_time
    log(f"Starting XRP bot (15m) | BUY: RSI<{RSI_BUY_THRESH}, SMA60<SMA240 | SELL: RSI>={RSI_SELL_THRESH}, SMA60>SMA240 & profit≥{int(MIN_PROFIT_PCT*100)}%")
    seconds_15m = GRAN_SECONDS.get(GRANULARITY, 900)

    try:
        meta = get_product_meta(PRODUCT_ID)
        log(f"{PRODUCT_ID} | increments price={meta['price_inc']} base={meta['base_inc']} quote={meta['quote_inc']} base_ccy={meta['base_ccy']} quote_ccy={meta['quote_ccy']}")
    except Exception as e:
        log(f"Product meta error: {e}")
        return

    sleep_until_next_closed_bar(seconds_15m, BAR_ALIGN_OFFSET)

    while not _shutting_down:
        try:
            candles = fetch_candles(PRODUCT_ID, "FIFTEEN_MINUTE", CANDLES_LIMIT)
            closes = [float(c["close"]) if isinstance(c, dict) else float(c.close) for c in candles]
            if not closes:
                log(f"{PRODUCT_ID} | No candles; sleeping.")
                sleep_until_next_closed_bar(seconds_15m, BAR_ALIGN_OFFSET)
                continue

            do_sell_signal = should_sell_15m_signal_only(closes)
            do_buy         = should_buy_15m(closes)

            # Cooldown on buys
            if COOLDOWN_MIN > 0 and do_buy and _last_buy_time is not None:
                if datetime.now(timezone.utc) - _last_buy_time < timedelta(minutes=COOLDOWN_MIN):
                    do_buy = False
                    log(f"{PRODUCT_ID} | In cooldown ({COOLDOWN_MIN} min); skipping BUY.")

            # SELL with profit gate
            if do_sell_signal:
                current_price = Decimal(str(closes[-1]))
                if is_min_profit_met(PRODUCT_ID, MIN_PROFIT_PCT, current_price):
                    place_market_sell_base_pct(PRODUCT_ID, SELL_ALLOCATION_PCT)
                else:
                    log(f"{PRODUCT_ID} | SELL signal true but profit < {int(MIN_PROFIT_PCT*100)}%; skipping SELL.")

            # BUY unchanged
            if do_buy:
                place_market_buy(PRODUCT_ID, ALLOCATION_PCT)
                _last_buy_time = datetime.now(timezone.utc)

        except Exception as e:
            log(f"Error: {e}")

        sleep_until_next_closed_bar(seconds_15m, BAR_ALIGN_OFFSET)

if __name__ == "__main__":
    main()
