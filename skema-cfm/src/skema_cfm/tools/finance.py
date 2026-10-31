"""Read-only numeric logic; the model must not perform material calculations."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, InvalidOperation

from ..case import CaseSnapshot


def _euros(value: str) -> int:
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(
            f"Non-numeric monetary input in source record: {value}"
        ) from exc
    if number != number.to_integral_value():
        raise ValueError("This educational dataset represents whole euros only.")
    return int(number)


def _filter(rows: tuple[dict, ...], customer: str | None = None) -> list[dict]:
    return [
        dict(r)
        for r in rows
        if customer is None or r["customer"].lower() == customer.strip().lower()
    ]


def query_forecast(case: CaseSnapshot, customer: str | None = None) -> dict:
    records = _filter(case.forecast, customer)
    return {
        "period": case.period,
        "customer_filter": customer,
        "forecast_eur": sum(_euros(r["forecast_amount_eur"]) for r in records),
        "records": [
            {**row, "source_ref": f"cash_forecast.csv:{row['forecast_id']}"}
            for row in records
        ],
        "source": "cash_forecast.csv",
    }


def query_bank_receipts(case: CaseSnapshot, customer: str | None = None) -> dict:
    records = _filter(case.bank, customer)
    start, end = case.period.split(" to ")
    records = [r for r in records if start <= r["value_date"] <= end]
    unmatched = [
        r["bank_txn_id"]
        for r in records
        if not r.get("remittance_reference", "").strip()
    ]
    return {
        "period": case.period,
        "customer_filter": customer,
        "actual_eur": sum(_euros(r["amount_eur"]) for r in records),
        "records": [
            {**row, "source_ref": f"bank_transactions.csv:{row['bank_txn_id']}"}
            for row in records
        ],
        "unmatched_reference_ids": unmatched,
        "source": "bank_transactions.csv",
    }


def lookup_open_invoices(
    case: CaseSnapshot,
    customer: str | None = None,
    invoice_id: str | None = None,
) -> dict:
    records = _filter(case.ar, customer)
    if invoice_id:
        records = [
            r for r in records if r["invoice_id"].casefold() == invoice_id.casefold()
        ]
    return {
        "customer_filter": customer,
        "invoice_filter": invoice_id,
        "outstanding_eur": sum(_euros(r["open_amount_eur"]) for r in records),
        "records": [
            {**row, "source_ref": f"ar_open_items.csv:{row['invoice_id']}"}
            for row in records
        ],
        "source": "ar_open_items.csv",
    }


def reconcile_variance(case: CaseSnapshot, customer: str | None = None) -> dict:
    """Compare all receipts for a customer with forecast; *not* invoice matching."""
    forecast = defaultdict(int)
    actual = defaultdict(int)
    refs = defaultdict(lambda: {"forecast": [], "bank": []})
    for r in case.forecast:
        forecast[r["customer"]] += _euros(r["forecast_amount_eur"])
        refs[r["customer"]]["forecast"].append(f"cash_forecast.csv:{r['forecast_id']}")
    start, end = case.period.split(" to ")
    for r in case.bank:
        if start <= r["value_date"] <= end:
            actual[r["customer"]] += _euros(r["amount_eur"])
            refs[r["customer"]]["bank"].append(
                f"bank_transactions.csv:{r['bank_txn_id']}"
            )
    names = sorted(set(forecast) | set(actual))
    if customer:
        names = [name for name in names if name.lower() == customer.strip().lower()]
    rows = [
        {
            "customer": name,
            "forecast_eur": forecast[name],
            "actual_eur": actual[name],
            "variance_eur": actual[name] - forecast[name],
            "source_refs": refs[name],
        }
        for name in names
    ]
    unmatched = [
        r["bank_txn_id"]
        for r in case.bank
        if start <= r["value_date"] <= end
        and not r.get("remittance_reference", "").strip()
        and (customer is None or r["customer"].lower() == customer.strip().lower())
    ]
    return {
        "period": case.period,
        "customer_filter": customer,
        "forecast_eur": sum(row["forecast_eur"] for row in rows),
        "actual_eur": sum(row["actual_eur"] for row in rows),
        "variance_eur": sum(row["variance_eur"] for row in rows),
        "by_customer": rows,
        "unmatched_remittance_reference_ids": unmatched,
        "warning": "Customer-level comparison does not establish invoice-level attribution."
        if unmatched
        else "Compare customer-level amounts; invoice attribution requires remittance evidence.",
    }
