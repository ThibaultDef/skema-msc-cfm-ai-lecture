"""Offline tests after pip install -e '.[test]'; no API key or external model needed."""

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage
from skema_cfm.agents.hybrid_agent import (
    HybridAgentSettings,
    run_hybrid_agent,
)
from skema_cfm.case import load_case
from skema_cfm.config import Scenario
from skema_cfm.tools.finance import reconcile_variance


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages)
        return AIMessage(content=next(self.responses))


class StubChild:
    def __init__(self, reply):
        self.reply = reply
        self.requests = []

    def invoke(self, state, config=None):
        self.requests.append(state)
        return self.reply


def _finance_child():
    reconciled = reconcile_variance(load_case())
    assert reconciled["variance_eur"] == -90000
    return StubChild(
        {
            "messages": [
                AIMessage(
                    content="Customer variances checked; verify documentary causes."
                )
            ],
            "calls_used": 1,
            "stop_reason": "completed",
            "trace": [
                {
                    "kind": "tool_call",
                    "tool": "reconcile_variance",
                    "status": "ok",
                    "arguments": {},
                    "result": reconciled,
                }
            ],
        }
    )


def _rag_child():
    doc = load_case().documents
    document = next(d for d in doc if d.document_id == "DOC-02").to_dict()
    return StubChild(
        {
            "messages": [
                AIMessage(content="Acme stated approval remains pending [DOC-02].")
            ],
            "calls_used": 1,
            "stop_reason": "completed",
            "trace": [
                {
                    "kind": "tool_call",
                    "tool": "search_documents",
                    "status": "ok",
                    "arguments": {"query": "Acme approval"},
                    "result": {"passages": [document]},
                }
            ],
            "retrieved": [document],
            "evidence_review": {"source_ids": ["DOC-02"]},
        }
    )


def test_two_real_subgraph_adapters_shared_budget_and_evidence():
    model = FakeModel(
        [
            '{"next_agent":"tools_agent","task":"reconcile all customer variances"}',
            '{"next_agent":"rag_agent","task":"search Acme remittance"}',
            "The period shows a EUR -90,000 variance; [cash_forecast.csv:F-001] and "
            "[bank_transactions.csv:BT-001] support the figures, while Acme reports pending approval [DOC-02].",
        ]
    )
    finance, rag = _finance_child(), _rag_child()
    result = run_hybrid_agent(
        "Explain the cash variance",
        HybridAgentSettings(),
        load_case(),
        model,
        tools_graph=finance,
        rag_graph=rag,
    )
    assert result.stop_reason == "completed"
    assert result.tool_calls_used == 2
    assert result.visited_agents == ["tools_agent", "rag_agent"]
    assert len(finance.requests) == len(rag.requests) == 1
    assert "retrieved" not in finance.requests[0]
    assert "finance_tool_results" not in rag.requests[0]
    assert result.retrieved_passages[0]["source_id"] == "DOC-02"
    assert result.finance_tool_results[0]["result"]["variance_eur"] == -90000
    assert any(
        t["agent"] == "tools_agent" and t["kind"] == "tool_call" for t in result.trace
    )
    assert any(
        t["agent"] == "rag_agent" and t["kind"] == "tool_call" for t in result.trace
    )


def test_premature_finish_is_overridden_by_coverage_policy():
    model = FakeModel(
        [
            '{"next_agent":"finish","task":""}',
            '{"next_agent":"finish","task":""}',
            "[cash_forecast.csv:F-001] [DOC-02] Under analyst review.",
        ]
    )
    result = run_hybrid_agent(
        "Investigate full cash variance",
        HybridAgentSettings(),
        load_case(),
        model,
        tools_graph=_finance_child(),
        rag_graph=_rag_child(),
    )
    assert result.visited_agents == ["tools_agent", "rag_agent"]
    assert any(t["kind"] == "supervisor_override" for t in result.trace)


def test_forged_document_citation_is_rejected():
    model = FakeModel(
        [
            '{"next_agent":"rag_agent","task":"Acme"}',
            '{"next_agent":"tools_agent","task":"variance"}',
            "The outcome is confirmed by [DOC-08] and [cash_forecast.csv:F-001].",
        ]
    )
    result = run_hybrid_agent(
        "Investigate receipts",
        HybridAgentSettings(),
        load_case(),
        model,
        tools_graph=_finance_child(),
        rag_graph=_rag_child(),
    )
    assert result.stop_reason == "finalization_failed"
    assert "did not pass the source check" in result.answer


def test_write_request_denied_before_model_or_tools():
    model = FakeModel([])
    finance, rag = _finance_child(), _rag_child()
    result = run_hybrid_agent(
        "Update the forecast for Acme",
        HybridAgentSettings(),
        load_case(),
        model,
        tools_graph=finance,
        rag_graph=rag,
    )
    assert result.stop_reason == "out_of_scope_request"
    assert result.tool_calls_used == 0
    assert not finance.requests and not rag.requests and not model.prompts


def test_quotas_cannot_exceed_global_limit():
    with pytest.raises(ValueError):
        HybridAgentSettings(
            max_tool_calls=6, finance_tool_budget=4, rag_search_budget=3
        )


def test_disabled_document_cannot_be_reintroduced_in_missing_case():
    case = load_case(Scenario.MISSING_ACME)
    assert "DOC-02" not in {d.document_id for d in case.documents}


def test_no_api_key_in_export():
    model = FakeModel(
        [
            '{"next_agent":"tools_agent","task":"reconcile"}',
            '{"next_agent":"rag_agent","task":"document"}',
            "[cash_forecast.csv:F-001] [DOC-02] Documentation needs human review.",
        ]
    )
    result = run_hybrid_agent(
        "Cash variance",
        HybridAgentSettings(api_key="TOP_SECRET_KEY"),
        load_case(),
        model,
        tools_graph=_finance_child(),
        rag_graph=_rag_child(),
    )
    assert "TOP_SECRET_KEY" not in str(result.to_dict())
