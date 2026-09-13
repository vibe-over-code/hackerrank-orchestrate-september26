"""Deterministic Phase 2 cash-flow ledger engine.

This module reconstructs a conservative 90-day balance forecast. It does not
choose a final payment recommendation yet; that belongs to Phase 4.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import calendar
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
CACHE = ROOT / ".cache" / "image_extractions.json"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATASET / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def money(value: str | float | int | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def d(value: str) -> date:
    return date.fromisoformat(value)


def add_months(day: date, months: int = 1) -> date:
    """Advance calendar-monthly commitments without drifting their payment day."""
    month_number = day.month - 1 + months
    year = day.year + month_number // 12
    month = month_number % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def load_image_amounts() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _message_amounts(text: str) -> list[tuple[float, str]]:
    pattern = r"\b(INR|IDR|USD|EUR|ZAR)\s*([0-9][0-9.,]*)"
    values = []
    for currency, raw in re.findall(pattern, text, flags=re.IGNORECASE):
        raw = raw.replace(",", "")
        try:
            values.append((float(raw), currency.upper()))
        except ValueError:
            pass
    return values


def _message_date(text: str) -> date | None:
    for raw in re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", text):
        try:
            return d(raw)
        except ValueError:
            pass
    return None


def _income_excluded_by_messages(event: dict[str, str], messages: list[dict[str, str]]) -> bool:
    """Return true only when a message explicitly disqualifies this income stream."""
    event_text = " ".join(
        str(event.get(key, "")) for key in ("category", "description", "event_type")
    ).lower()
    unavailable = (
        "pending", "not approved", "not-approved", "not withdrawable",
        "not-withdrawable", "unrealized", "unrealised", "belum disetujui",
        "masih menunggu", "tidak masuk pembayaran", "belum diperoleh",
    )
    stream_terms = {
        "payout": ("payout", "weekly", "earnings", "gig", "quickcrew", "app"),
        "bonus": ("bonus",),
        "commission": ("commission", "komisi"),
        "prize": ("prize", "reward", "hadiah"),
        "unrealized": ("unrealized", "unrealised", "gain", "keuntungan"),
    }
    for message in messages:
        text = message.get("message_text", "").lower()
        if not any(marker in text for marker in unavailable):
            continue
        for terms in stream_terms.values():
            if any(term in text for term in terms) and any(term in event_text for term in terms):
                return True
    return False


def _is_recurring_salary(event: dict) -> bool:
    """Keep only stable payroll-like income in the salary recurrence stream."""
    text = " ".join(
        str(event.get(key, "")) for key in ("description", "category", "event_type")
    ).lower()
    excluded = (
        "bonus", "commission", "komisi", "payout", "weekly", "earnings",
        "gig", "reward", "prize", "refund", "gain", "incentive",
    )
    if any(term in text for term in excluded):
        return False
    stable = ("salary", "payroll", "gaji", "wage", "employer")
    return any(term in text for term in stable)


def apply_message_amendments(
    request: dict[str, str], raw_events: list[dict[str, str]], messages: list[dict[str, str]],
    diagnostics: dict | None = None,
) -> list[dict[str, str]]:
    """Apply explicit, financially relevant message amendments before ledgering."""
    start = d(request["request_date"])
    relevant = [
        m for m in messages
        if m.get("user_id") == request["user_id"]
        and (not m.get("request_id") or m.get("request_id") == request["request_id"])
    ]
    if diagnostics is not None:
        diagnostics.update({
            "relevant_count": len(relevant),
            "synthetic_count": 0,
            "texts": [m.get("message_text", "")[:60] for m in relevant],
            "fired_rules": [],
        })
    events = [dict(e) for e in raw_events]
    synthetic: list[dict[str, str]] = []
    salary_number = 0
    for e in events:
        if e.get("category") == "salary" and e.get("amount"):
            salary_number += float(e["amount"])
    salary_count = sum(1 for e in events if e.get("category") == "salary" and e.get("amount"))
    salary_number = salary_number / max(1, salary_count)
    ended = False
    for message in relevant:
        text = message.get("message_text", "")
        lower = text.lower()
        amounts = _message_amounts(text)
        msg_date = _message_date(text)
        if "employment has ended" in lower or "seasonal contract has ended" in lower or "contract saat ini telah berakhir" in lower or "pendapatan ... telah berakhir" in lower:
            ended = True
            if diagnostics is not None:
                diagnostics["fired_rules"].append("employment_ended")
            continue
        if "increased" in lower or "naik menjadi" in lower or "remaining confirmed monthly salary" in lower or "sisa gaji bulanan" in lower or "confirmed base salary" in lower or "gaji pokok yang dikonfirmasi" in lower:
            if amounts:
                value, currency = amounts[0]
                if msg_date is None:
                    msg_date = start
                synthetic.append({"event_id": f"message_{message['message_id']}", "user_id": request["user_id"], "event_type": "income", "description": "message confirmed salary", "category": "salary", "direction": "credit", "amount": str(value), "currency": currency, "event_date": msg_date.isoformat(), "settlement_date": msg_date.isoformat(), "status": "scheduled", "linked_event_id": "", "flexibility": "fixed", "minimum_allowed_amount": ""})
                if diagnostics is not None:
                    diagnostics["fired_rules"].append("salary_increased_or_confirmed")
        elif "first salary" in lower or "gaji pertama" in lower or "salary of" in lower and "confirmed" in lower:
            if amounts and msg_date:
                value, currency = amounts[0]
                synthetic.append({"event_id": f"message_{message['message_id']}", "user_id": request["user_id"], "event_type": "income", "description": "message confirmed first salary", "category": "salary", "direction": "credit", "amount": str(value), "currency": currency, "event_date": msg_date.isoformat(), "settlement_date": msg_date.isoformat(), "status": "scheduled", "linked_event_id": "", "flexibility": "fixed", "minimum_allowed_amount": ""})
                if diagnostics is not None:
                    diagnostics["fired_rules"].append("first_salary")
        elif "regular salary for the next payroll" in lower or "gaji rutin untuk penggajian berikutnya" in lower:
            value, currency = amounts[0] if amounts else (salary_number, "")
            if value and msg_date:
                synthetic.append({"event_id": f"message_{message['message_id']}", "user_id": request["user_id"], "event_type": "income", "description": "message next salary", "category": "salary", "direction": "credit", "amount": str(value), "currency": currency, "event_date": msg_date.isoformat(), "settlement_date": msg_date.isoformat(), "status": "scheduled", "linked_event_id": "", "flexibility": "fixed", "minimum_allowed_amount": ""})
                if diagnostics is not None:
                    diagnostics["fired_rules"].append("regular_salary")
        elif "salary" in lower or "gaji" in lower:
            if amounts:
                value, currency = amounts[0]
                payment_date = msg_date or start
                synthetic.append({"event_id": f"message_{message['message_id']}", "user_id": request["user_id"], "event_type": "income", "description": "message payroll adjustment", "category": "salary", "direction": "credit", "amount": str(value), "currency": currency, "event_date": payment_date.isoformat(), "settlement_date": payment_date.isoformat(), "status": "scheduled", "linked_event_id": "", "flexibility": "fixed", "minimum_allowed_amount": ""})
                if diagnostics is not None:
                    diagnostics["fired_rules"].append("generic_salary")
        if "increases monthly rent by 12%" in lower or "menaikkan biaya sewa bulanan sebesar 12%" in lower:
            for event in events:
                if event.get("category") == "rent" and event.get("amount") and (not event.get("settlement_date") or d(event["settlement_date"]) >= start):
                    event["amount"] = str(float(event["amount"]) * 1.12)
        if "invoice payment" in lower or "pembayaran faktur" in lower or "client approved" in lower:
            if amounts and msg_date:
                value, currency = amounts[0]
                synthetic.append({"event_id": f"message_{message['message_id']}", "user_id": request["user_id"], "event_type": "income", "description": "confirmed invoice settlement", "category": "salary", "direction": "credit", "amount": str(value), "currency": currency, "event_date": msg_date.isoformat(), "settlement_date": msg_date.isoformat(), "status": "scheduled", "linked_event_id": "", "flexibility": "fixed", "minimum_allowed_amount": ""})
                if diagnostics is not None:
                    diagnostics["fired_rules"].append("invoice_payment")
    if ended:
        events = [e for e in events if not (e.get("category") == "salary" and e.get("event_date") and d(e["event_date"]) >= start)]
    if diagnostics is not None:
        diagnostics["synthetic_count"] = len(synthetic)
    return events + synthetic
def rate_for(rates: list[dict[str, str]], when: date, source: str, target: str) -> float:
    if source == target:
        return 1.0
    candidates = [
        r for r in rates
        if r["from_currency"] == source
        and r["to_currency"] == target
        and d(r["rate_date"]) <= when
    ]
    if not candidates:
        raise ValueError(f"no exchange rate {source}->{target} on {when}")
    chosen = max(candidates, key=lambda r: d(r["rate_date"]))
    return float(chosen["rate"])


def event_amount_currency(
    event: dict[str, str], image_amounts: dict[str, dict]
) -> tuple[float | None, str]:
    value = money(event.get("amount"))
    if value is not None:
        return value, event.get("currency", "")
    for item in image_amounts.values():
        if item.get("event_id") == event.get("event_id"):
            return money(item.get("amount")), item.get("currency", "")
    return None, event.get("currency", "")


def normalized_events(
    events: list[dict[str, str]],
    profile: dict[str, str],
    rates: list[dict[str, str]],
    image_amounts: dict[str, dict],
) -> list[dict]:
    home = profile["home_currency"]
    result = []
    for raw in events:
        status = raw["status"]
        direction = raw["direction"]
        if status in {"failed", "cancelled", "unrealized"} or direction == "non_cash":
            continue
        if direction == "credit" and status == "pending":
            continue
        amount, currency = event_amount_currency(raw, image_amounts)
        if amount is None:
            continue
        settlement = d(raw["settlement_date"] or raw["event_date"])
        amount *= rate_for(rates, settlement, currency, home)
        result.append({**raw, "date": settlement, "amount_home": amount})
    return result


def is_internal_transfer(event: dict, all_events: list[dict]) -> bool:
    if event["event_type"] not in {"expense", "income"}:
        return False
    opposite = "credit" if event["direction"] == "debit" else "debit"
    return any(
        other is not event
        and other["direction"] == opposite
        and other["date"] == event["date"]
        and abs(other["amount_home"] - event["amount_home"]) < 0.01
        and other["category"] in {event["category"], "transfer", "internal_transfer"}
        for other in all_events
    )


def median_amount(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def recurrence_templates(history: list[dict], start: date) -> list[dict]:
    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for event in history:
        if event["date"] >= start:
            continue
        if event["event_type"] in {"refund", "investment_purchase", "investment_sale"}:
            continue
        if event["category"] == "salary":
            continue
        groups[(event["direction"], event["category"], event["description"], event["flexibility"])].append(event)
    templates = []
    for key, rows in groups.items():
        rows.sort(key=lambda x: x["date"])
        if len(rows) < 2:
            continue
        gaps = [(b["date"] - a["date"]).days for a, b in zip(rows, rows[1:])]
        gap = median_amount(gaps)
        if not 20 <= gap <= 45:
            continue
        templates.append({
            "direction": key[0],
            "category": key[1],
            "description": key[2],
            "flexibility": key[3],
            "amount": median_amount([r["amount_home"] for r in rows[-12:]]),
            "gap": max(1, round(gap)),
            "last": rows[-1]["date"],
            "calendar_monthly": 27 <= gap <= 35 and max(x["date"].day for x in rows[-6:]) - min(x["date"].day for x in rows[-6:]) <= 3,
        })
    # Groceries and transport are variable essentials: forecast their recent
    # cadence and median amount by category, rather than by merchant text.
    variable_categories: set[str] = set()
    for category in ("groceries", "transport"):
        rows = [x for x in history if x["direction"] == "debit" and x["category"] == category]
        rows.sort(key=lambda x: x["date"])
        if len(rows) < 4:
            continue
        recent = rows[-12:]
        gaps = [(b["date"] - a["date"]).days for a, b in zip(recent, recent[1:])]
        gap = median_amount(gaps)
        if 5 <= gap <= 21:
            templates.append({
                "direction": "debit",
                "category": category,
                "description": "variable_essential",
                "flexibility": "fixed",
                "amount": median_amount([x["amount_home"] for x in recent]),
                "gap": max(1, round(gap)),
                "last": recent[-1]["date"],
                "calendar_monthly": False,
            })
            variable_categories.add(category)
    if variable_categories:
        templates = [
            t for t in templates
            if t["category"] not in variable_categories or t["description"] == "variable_essential"
        ]
    return templates


@dataclass
class Forecast:
    start: date
    balances: dict[date, float]
    minimum_balance: float

    def minimum_surplus_from(self, when: date) -> float:
        return min(
            self.balances[x] - self.minimum_balance
            for x in self.balances
            if x >= when
        )

    def payment_safe(self, when: date, amount: float) -> bool:
        return self.minimum_surplus_from(when) >= amount - 0.005


def build_forecast(
    request: dict[str, str], profile: dict[str, str], user_events: list[dict], rates: list[dict[str, str]], image_amounts: dict[str, dict], messages: list[dict[str, str]] | None = None,
    enable_recurrence: bool = True,
    enable_messages: bool = True,
) -> Forecast:
    start = d(request["request_date"])
    horizon = [start + timedelta(days=i) for i in range(91)]
    active_messages = messages or [] if enable_messages else []
    relevant_messages = [
        m for m in active_messages
        if m.get("user_id") == request["user_id"]
        and (not m.get("request_id") or m.get("request_id") == request["request_id"])
    ]
    amended_events = apply_message_amendments(request, user_events, active_messages)
    normalized = normalized_events(amended_events, profile, rates, image_amounts)
    normalized = [
        event for event in normalized
        if not _income_excluded_by_messages(event, relevant_messages)
    ]
    normalized = [e for e in normalized if not is_internal_transfer(e, normalized)]
    history = [e for e in normalized if e["date"] < start and e["status"] == "settled"]
    templates = recurrence_templates(history, start) if enable_recurrence else []
    flows: dict[date, float] = defaultdict(float)
    end = horizon[-1]
    for event in normalized:
        if start <= event["date"] <= end:
            flows[event["date"]] += event["amount_home"] if event["direction"] == "credit" else -event["amount_home"]
    real_debit_dates = {
        (event["category"], event["date"])
        for event in normalized
        if event["direction"] == "debit" and start <= event["date"] <= end
    }
    real_salary_dates = {
        event["date"]
        for event in normalized
        if event["direction"] == "credit"
        and event["category"] == "salary"
        and start <= event["date"] <= end
    }
    for template in templates:
        advance = add_months if template.get("calendar_monthly") else lambda day: day + timedelta(days=template["gap"])
        next_day = advance(template["last"])
        while next_day < start:
            next_day = advance(next_day)
        while next_day <= end:
            if any(
                category == template["category"]
                and abs((real_date - next_day).days) <= 3
                for category, real_date in real_debit_dates
            ):
                next_day = advance(next_day)
                continue
            sign = 1 if template["direction"] == "credit" else -1
            flows[next_day] += sign * template["amount"]
            next_day = advance(next_day)
    # Salary is projected only here.  Use the historical cadence and amount,
    # while allowing an explicit future/message salary to override the amount.
    salary_history = [
        e for e in normalized
        if e["category"] == "salary"
        and e["direction"] == "credit"
        and e["date"] < start
        and e["status"] == "settled"
        and _is_recurring_salary(e)
    ]
    salary_history.sort(key=lambda e: e["date"])
    salary_gaps = [
        (b["date"] - a["date"]).days
        for a, b in zip(salary_history, salary_history[1:])
    ]
    salary_gap = round(median_amount(salary_gaps)) if salary_gaps else 30
    salary_amount = median_amount([e["amount_home"] for e in salary_history[-3:]])
    salary_events = [
        e for e in normalized
        if e["category"] == "salary"
        and e["direction"] == "credit"
        and e["date"] >= start
        and e["status"] in {"scheduled", "pending", "settled"}
        and _is_recurring_salary(e)
    ]
    salary_override = next(
        (e for e in salary_events if str(e.get("event_id", "")).startswith("message_")),
        None,
    )
    if salary_override is not None:
        salary_amount = salary_override["amount_home"]
    if enable_recurrence and (salary_history or salary_events):
        anchor = min(salary_events, key=lambda e: e["date"])["date"] if salary_events else salary_history[-1]["date"] + timedelta(days=salary_gap)
        salary_advance = add_months if 27 <= salary_gap <= 35 else lambda day: day + timedelta(days=salary_gap)
        if not salary_events:
            anchor = salary_advance(salary_history[-1]["date"])
        if not salary_events and anchor <= end:
            flows[anchor] += salary_amount
        next_day = salary_advance(anchor)
        while next_day <= end:
            if any(abs((next_day - real_date).days) <= 3 for real_date in real_salary_dates):
                next_day = salary_advance(next_day)
                continue
            flows[next_day] += salary_amount
            next_day = salary_advance(next_day)
    balances = {}
    balance = float(profile["current_available_balance"])
    for day in horizon:
        balance += flows[day]
        balances[day] = balance
    return Forecast(start, balances, float(profile["minimum_balance_to_keep"]))


def amount_safe_to_pay(forecast: Forecast, requested: float) -> float:
    return round(max(0.0, min(requested, forecast.minimum_surplus_from(forecast.start))), 2)


def earliest_full_payment(forecast: Forecast, requested: float) -> date | None:
    for day in sorted(forecast.balances):
        if forecast.minimum_surplus_from(day) >= requested - 0.005:
            return day
    return None


def build_phase2_ledgers() -> dict[str, Forecast]:
    requests = read_csv("requests.csv")
    profiles = {r["user_id"]: r for r in read_csv("financial_profiles.csv")}
    events_by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
    for event in read_csv("financial_events.csv"):
        events_by_user[event["user_id"]].append(event)
    rates = read_csv("exchange_rates.csv")
    messages = read_csv("messages.csv")
    image_amounts = load_image_amounts()
    forecasts = {}
    for request in requests:
        forecasts[request["request_id"]] = build_forecast(
            request, profiles[request["user_id"]], events_by_user[request["user_id"]], rates, image_amounts, messages
        )
    return forecasts


def build_forecasts_for_requests(requests: list[dict[str, str]]) -> dict[str, Forecast]:
    profiles = {r["user_id"]: r for r in read_csv("financial_profiles.csv")}
    events_by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
    for event in read_csv("financial_events.csv"):
        events_by_user[event["user_id"]].append(event)
    rates = read_csv("exchange_rates.csv")
    image_amounts = load_image_amounts()
    messages = read_csv("messages.csv")
    return {
        request["request_id"]: build_forecast(
            request, profiles[request["user_id"]], events_by_user[request["user_id"]], rates, image_amounts, messages
        )
        for request in requests
    }


if __name__ == "__main__":
    samples = read_csv("sample_requests.csv")
    forecasts = build_forecasts_for_requests(samples + read_csv("requests.csv"))
    print(f"phase2_forecasts: {len(forecasts)}")
    for sample in samples[:5]:
        forecast = forecasts[sample["request_id"]]
        requested = float(sample["requested_amount"])
        safe = amount_safe_to_pay(forecast, requested)
        earliest = earliest_full_payment(forecast, requested)
        print(f"{sample['request_id']}: safe={safe}, expected={sample['amount_safe_to_pay']}, earliest={earliest}, expected_earliest={sample['earliest_date_for_full_payment']}")
