"""Immutable case snapshots and explicit, in-memory student testing scenarios."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib.resources import files

from .config import Scenario

CASE_PACKAGE = "skema_cfm.case_data"
TABLE_FILES = ("cash_forecast.csv", "bank_transactions.csv", "ar_open_items.csv")


@dataclass(frozen=True)
class EvidenceDocument:
    document_id: str
    title: str
    date: str
    filename: str
    text: str

    def to_dict(self) -> dict:
        return {
            "source_id": self.document_id,
            "title": self.title,
            "date": self.date,
            "filename": self.filename,
            "passage": self.text,
        }


@dataclass(frozen=True)
class CaseSnapshot:
    forecast: tuple[dict, ...]
    bank: tuple[dict, ...]
    ar: tuple[dict, ...]
    documents: tuple[EvidenceDocument, ...]
    scenario: Scenario
    annotations: tuple[str, ...] = ()

    @property
    def period(self) -> str:
        periods = {(row["period_start"], row["period_end"]) for row in self.forecast}
        if len(periods) != 1:
            raise ValueError("Forecast case must contain exactly one reporting period.")
        start, end = next(iter(periods))
        return f"{start} to {end}"


def _read_csv(name: str) -> tuple[dict, ...]:
    root = files(CASE_PACKAGE).joinpath("tables", name)
    with root.open("r", encoding="utf-8-sig", newline="") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def _load_documents() -> tuple[EvidenceDocument, ...]:
    folder = files(CASE_PACKAGE).joinpath("documents")
    docs = []
    for document in sorted(folder.iterdir(), key=lambda f: f.name):
        if not document.name.endswith(".txt"):
            continue
        text = document.read_text(encoding="utf-8")
        metadata = {}
        for line in text.splitlines():
            for key in ("Document ID", "Title", "Date"):
                if line.startswith(key + ":"):
                    metadata[key] = line.partition(":")[2].strip()
        if any(key not in metadata for key in ("Document ID", "Title", "Date")):
            raise ValueError(f"Malformed source document: {document.name}")
        docs.append(
            EvidenceDocument(
                document_id=metadata["Document ID"],
                title=metadata["Title"],
                date=metadata["Date"],
                filename=document.name,
                text=text,
            )
        )
    return tuple(docs)


def load_case(
    scenario: Scenario = Scenario.STANDARD,
    selected_documents: tuple[str, ...] | None = None,
) -> CaseSnapshot:
    """Load originals, applying *labelled* scenario overlays in memory only.

    DOC-08 is test-only: explicitly unavailable outside the conflict scenario.
    A scenario does not overwrite any CSV or supporting file on disk.
    """
    forecast = _read_csv("cash_forecast.csv")
    bank = _read_csv("bank_transactions.csv")
    ar = _read_csv("ar_open_items.csv")
    docs = _load_documents()
    annotations = []

    if scenario != Scenario.COBALT_CONFLICT:
        docs = tuple(d for d in docs if d.document_id != "DOC-08")
    else:
        annotations.append(
            "TEST SCENARIO: unverified DOC-08 is deliberately included; do not treat it as a corrected ledger."
        )

    if scenario == Scenario.MISSING_ACME:
        docs = tuple(d for d in docs if d.document_id != "DOC-02")
        annotations.append(
            "TEST SCENARIO: Acme remittance (DOC-02) is withheld; avoid pretending its contents are known."
        )

    if scenario == Scenario.UNMATCHED_BANK:
        bank = tuple(
            (
                {**row, "remittance_reference": ""}
                if row["bank_txn_id"] == "BT-004"
                else row
            )
            for row in bank
        )
        annotations.append(
            "TEST SCENARIO: BT-004 has no remittance reference. The bank receipt remains real in the fictional case; "
            "do not assert a verified invoice allocation without matching evidence."
        )

    if scenario == Scenario.TOOL_FAILURE:
        annotations.append(
            "TEST SCENARIO: first attempted tool call will fail (simulated tool outage)."
        )

    if selected_documents is not None:
        missing_ids = set(selected_documents) - {d.document_id for d in docs}
        if missing_ids:
            annotations.append(
                f"Unavailable or excluded requested documents: {sorted(missing_ids)}"
            )
        docs = tuple(doc for doc in docs if doc.document_id in selected_documents)

    return CaseSnapshot(
        forecast=forecast,
        bank=bank,
        ar=ar,
        documents=docs,
        scenario=scenario,
        annotations=tuple(annotations),
    )


def list_case_document_ids(scenario: Scenario) -> tuple[str, ...]:
    return tuple(d.document_id for d in load_case(scenario).documents)
