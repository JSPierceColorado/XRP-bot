#!/usr/bin/env python3
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
ALLOCATION_PCT    = float(os.getenv("ALLOCATION_PCT", "0.05"))   # 5% of quote balance
TAKE_PROFIT_PCT   = float(os.getenv("TAKE_PROFIT_PCT", "0.02"))  # 2% TP
RUN_INTERVAL_SEC  = int(os.getenv("RUN_INTERVAL_SEC", "60"))
RSI_LEN           = int(os.getenv("RSI_LEN", "14"))
CANDLES_LIMIT     = int(os.getenv("CANDLES_LIMIT", "200"))       # up to 350 per API
GRANULARITY       = os.getenv("GRANULARITY", "ONE_MINUTE")       # coinbase enums
COOLDOWN_MINUTES  = int(os.getenv("COOLDOWN_MINUTES", "0"))      # optional: min mins between buys

LOG_PREFIX        = os.getenv("LOG_PREFIX", "[xrp-rsi-bot]")

# ========= Globals =========
client = RESTClient()  # expects COINBASE_API_KEY / COINBASE_API_SECRET env vars
_shutting_down = False
_last_buy_time = None  # naive cooldown

def log(msg: str):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    print(f"{LOG_PREFIX} {now} | {msg}", flush=True)

def _handle(sig, _):
    global _shutting_down
    _shutting_down = True
    log(f"Received {sig}. Shutting down gracefully.")

signal.signal(signal.SIGTERM, _handle)
signal.signal(signal.SIGINT, _handle)

# ========= Helpers =========
def get_product_meta(pid: str) -> Dict[str, Decimal]:
    """Get increments and quote currency for rounding & balances."""
    p = client.get_product(product_id=pid)
    meta = {
        "price_inc": Decimal(p.price_increment),
        "base_inc": Decimal(p.base_increment),
        "quote_inc": Decimal(p.quote_increment),
        "quote_min": Decimal(getattr(p, "quote_min_size", "0")) if hasattr(p, "quote_min_size") else Decimal("0"),
        "quote_ccy": p.quote_currency_id,
    }
    return meta

def round_to_inc(value: Decimal, inc: Decimal) -> Decimal:
    if inc <= 0:
        return value
    q = (value / inc).to_integral_value(rounding=ROUND_DOWN) * inc
    return q

def get_quote_available(quote_ccy: str) -> Decimal:
    accs = client.get_accounts()
    for a in accs.accounts:
        if a.currency == quote_ccy:
            return Decimal(a.available_balance["value"])
    return Decimal("0")

def fetch_candles(pid: str, granularity: str, limit: int) -> List[Dict[str, str]]:
    """Fetch recent candles using REST; returns list of dicts with open/close/etc (strings)."""
    # Coinbase Advanced requires start/end timestamps (seconds). We'll fetch a tight window.
    end = int(datetime.now(timezone.utc).timestamp())
    # buffer window so we reliably get <limit> bars
    seconds = {
        "ONE_MINUTE": 60,
        "FIVE_MINUTE": 300,
        "FIFTEEN_MINUTE": 900,
        "THIRTY_MINUTE": 1800,
        "ONE_HOUR": 3600,
        "TWO_HOUR": 7200,
        "FOUR_HOUR": 14400,
        "SIX_HOUR": 21600,
        "ONE_DAY": 86400,
    }.get(granularity, 60)
    start = end - seconds * (limit + 10)

    res = client.get(
        f"/api/v3/brokerage/products/{pid}/candles",
        params={"start": str(start), "end": str(end), "granularity": granularity, "limit": limit},
    )
    return res["candles"] if isinstance(res, dict) and "candles" in res else res.candles  # SDK or raw dict

def rsi(values: List[float], length: int = 14) -> float:
    """Wilder's RSI; returns last RSI value or NaN if insufficient data."""
    if len(values) < length + 1:
        return float("nan")
    gains = []
    losses = []
    for i in range(1, length + 1):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_gain = sum(gains) / length
    avg_loss = sum(losses) / length

    if avg_loss == 0:
        return 100.0

    # Wilder's smoothing
    for i in range(length + 1, len(values)):
        diff = values[i] - values[i - 1]
        gain = max(diff, 0.0)
        loss = max(-diff, 0.0)
        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length

    rs = avg_gain / avg_loss if avg_loss != 0 else math.inf
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi_val)

def latest_price(pid: str) -> Decimal:
    prod = client.get_product(product_id=pid)
    return Decimal(prod.price)

def place_market_buy_then_tp(pid: str, allocation_pct: float, tp_pct: float):
    """Place a market buy using quote_size (pct of quote balance), then a GTC limit sell at +tp_pct."""
    meta = get_product_meta(pid)
    quote_bal = get_quote_available(meta["quote_ccy"])

    if quote_bal <= 0:
        log(f"{pid} | No {meta['quote_ccy']} available; skipping.")
        return

    quote_notional = Decimal(str(allocation_pct)) * quote_bal
    # honor product quote minimums & increments
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

    # Compute filled size & avg fill price
    fills = client.list_fills(order_id=buy_oid, product_id=pid)
    if hasattr(fills, "fills"):
        f_list = fills.fills
    else:
        f_list = fills["fills"]
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

    # TP Limit SELL at +tp_pct
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
                "post_only": False
            }
        }
    }
    client.post("/api/v3/brokerage/orders", data=payload)
    log(f"{pid} | TP placed: sell {base_sz} @ {limit_px} (avg entry {avg_price})")

def should_buy(rsi_val: float) -> bool:
    return rsi_val <= 30.0

def main():
    global _last_buy_time
    log("Starting perpetual XRP RSI bot (1m RSI<=30 => buy 5% notional; TP +2%; no SL)")
    # quick sanity check: product & increments
    try:
        meta = get_product_meta(PRODUCT_ID)
        log(f"{PRODUCT_ID} | increments price={meta['price_inc']} base={meta['base_inc']} quote={meta['quote_inc']} quote_ccy={meta['quote_ccy']}")
    except Exception as e:
        log(f"Failed to fetch product meta for {PRODUCT_ID}: {e}")
        return

    while not _shutting_down:
        try:
            candles = fetch_candles(PRODUCT_ID, GRANULARITY, CANDLES_LIMIT)
            closes = [float(c["close"]) if isinstance(c, dict) else float(c.close) for c in candles]
            r = rsi(closes, RSI_LEN)
            price = latest_price(PRODUCT_ID)
            log(f"{PRODUCT_ID} | 1m RSI{RSI_LEN}={r:.2f} | price={price}")

            can_buy = should_buy(r)

            # cooldown guard if configured
            if COOLDOWN_MINUTES > 0 and can_buy and _last_buy_time is not None:
                if datetime.now(timezone.utc) - _last_buy_time < timedelta(minutes=COOLDOWN_MINUTES):
                    can_buy = False
                    log(f"{PRODUCT_ID} | In cooldown window ({COOLDOWN_MINUTES} min); skipping buy.")

            if can_buy:
                place_market_buy_then_tp(PRODUCT_ID, ALLOCATION_PCT, TAKE_PROFIT_PCT)
                _last_buy_time = datetime.now(timezone.utc)
        except Exception as e:
            log(f"Error: {e}")

        # sleep to align roughly on minute boundaries
        time.sleep(RUN_INTERVAL_SEC)

if __name__ == "__main__":
    main()
