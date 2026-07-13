"""Phase 5 Research Agent tests: tool choice, evidence merge, budgets."""

from __future__ import annotations

from time import sleep

from scholar_agent.agents.researcher import (
    ResearchAgent,
    ResearchAgentConfig,
    build_research_subgraph,
    hits_to_evidence,
)
from scholar_agent.agents.state import merge_evidence
from scholar_agent.ids import make_chunk_id, make_evidence_id
from scholar_agent.models.base import QueryType
from scholar_agent.models.evidence import EvidenceItem, EvidenceLedger
from scholar_agent.models.planning import SubQuestion, SubQuestionStatus
from scholar_agent.models.retrieval import RetrievalHit, RetrievalResult
from scholar_agent.models.routing import RetrievalPolicy
from scholar_agent.retrieval.router import policy_to_tool_modes, recommend_policy
from scholar_agent.retrieval.tools import RetrievalToolkit


class FakeToolkit(RetrievalToolkit):
    """Toolkit that records modes and returns canned hits (no indexes)."""

    def __init__(self, *, has_graph: bool = True) -> None:
        # Bypass parent index requirements
        self.store = None  # type: ignore[assignment]
        self.dense = None
        self.sparse = None
        self.graph = object() if has_graph else None  # truthy for router has_graph
        self.reranker = None  # type: ignore[assignment]
        self.dense_top_k = 12
        self.sparse_top_k = 12
        self.fused_top_k = 20
        self.rerank_top_k = 8
        self.rrf_k = 60
        self.calls: list[str] = []

    def search(
        self, query: str, *, mode: str = "hybrid_rerank", k: int | None = None, filters=None
    ) -> RetrievalResult:  # type: ignore[override]
        self.calls.append(mode)
        # Distinct chunk per mode so multi-tool passes can accumulate evidence
        text = f"Evidence for '{query}' via {mode}. Self-RAG and CRAG are related methods."
        hit = RetrievalHit(
            chunk_id=make_chunk_id("paper_demo", page_start=1, page_end=1, text=text + mode),
            paper_id="paper_demo",
            text=text,
            page_start=1,
            page_end=2,
            section="Method",
            score=0.9,
            retrieval_method=mode,
        )
        method = (
            mode if mode in {"dense", "sparse", "hybrid", "hybrid_rerank", "graph"} else "hybrid"
        )
        return RetrievalResult(query=query, method=method, hits=[hit])  # type: ignore[arg-type]


class SlowFakeToolkit(FakeToolkit):
    def search(self, *args, **kwargs) -> RetrievalResult:  # type: ignore[no-untyped-def,override]
        sleep(0.005)
        return super().search(*args, **kwargs)


def _sq(qid: str, question: str, qtype: QueryType) -> SubQuestion:
    return SubQuestion(
        id=qid,
        question=question,
        query_type=qtype,
        required_evidence=["passages"],
        status=SubQuestionStatus.PENDING,
    )


def test_research_agent_chooses_different_tools_by_query_type() -> None:
    toolkit = FakeToolkit(has_graph=True)
    agent = ResearchAgent(
        toolkit,  # type: ignore[arg-type]
        config=ResearchAgentConfig(max_tool_calls_per_pass=2, allow_policy_override=False),
    )

    semantic = agent.research_sub_question(
        _sq("sq_sem", "What is retrieval-augmented generation conceptually?", QueryType.SEMANTIC)
    )
    comparison = agent.research_sub_question(
        _sq("sq_cmp", "Compare Self-RAG versus CRAG", QueryType.COMPARISON)
    )
    relational = agent.research_sub_question(
        _sq("sq_rel", "Which datasets does Self-RAG evaluate on?", QueryType.RELATIONAL)
    )

    assert semantic.routing.recommended_policy == RetrievalPolicy.DENSE
    assert semantic.actions[0].tool_name.startswith("dense")

    assert comparison.routing.recommended_policy == RetrievalPolicy.HYBRID_PLUS_GRAPH
    cmp_tools = [a.tool_name for a in comparison.actions]
    assert any("hybrid" in t for t in cmp_tools)
    assert any("graph" in t for t in cmp_tools)

    assert relational.routing.recommended_policy == RetrievalPolicy.GRAPH
    assert relational.actions[0].tool_name.startswith("graph")

    # Different first tools across types
    first_tools = {
        semantic.actions[0].tool_name,
        comparison.actions[0].tool_name,
        relational.actions[0].tool_name,
    }
    assert len(first_tools) >= 2


def test_duplicate_evidence_is_merged() -> None:
    hit = RetrievalHit(
        chunk_id="chunk_same",
        paper_id="paper_x",
        text="Same span about Self-RAG retrieval.",
        page_start=3,
        page_end=3,
        score=0.5,
        retrieval_method="dense",
    )
    hit2 = hit.model_copy(update={"score": 0.9, "retrieval_method": "hybrid", "rerank_score": 0.95})
    a = hits_to_evidence([hit], run_id="run_1", sub_question_id="sq_1")
    ledger = EvidenceLedger(items=a)
    b = hits_to_evidence([hit2], run_id="run_1", sub_question_id="sq_1", existing=ledger)
    # second conversion sees duplicate key → no new items
    assert b == []
    # merge_evidence still prefers higher scores if both present
    high = EvidenceItem(
        evidence_id=make_evidence_id(
            run_id="run_1",
            chunk_id="chunk_same",
            evidence_text="Same span about Self-RAG retrieval.",
            sub_question_id="sq_1",
        ),
        sub_question_id="sq_1",
        claim="c",
        evidence_text="Same span about Self-RAG retrieval.",
        paper_id="paper_x",
        chunk_id="chunk_same",
        page_start=3,
        page_end=3,
        retrieval_method="hybrid",
        retrieval_score=0.9,
        rerank_score=0.95,
    )
    low = high.model_copy(
        update={"retrieval_score": 0.1, "rerank_score": 0.1, "retrieval_method": "dense"}
    )
    # force different evidence_id to test merge by dedupe key
    low = low.model_copy(update={"evidence_id": low.evidence_id + "_low"})
    merged = merge_evidence([low], high)
    assert len(merged) == 1
    assert merged[0].retrieval_score == 0.9


def test_tool_budget_cannot_be_exceeded() -> None:
    toolkit = FakeToolkit(has_graph=True)
    agent = ResearchAgent(
        toolkit,  # type: ignore[arg-type]
        config=ResearchAgentConfig(
            max_tool_calls_per_pass=1,
            max_evidence_per_sub_question=8,
            allow_policy_override=True,
        ),
    )
    # Comparison wants hybrid+graph (2 modes) but budget is 1
    result = agent.research_sub_question(
        _sq("sq_b", "Compare Self-RAG versus CRAG", QueryType.COMPARISON)
    )
    assert result.tool_call_count <= 1
    assert len(result.actions) <= 1
    assert result.terminated_reason in {
        "tool_budget_exhausted",
        "completed",
        "evidence_budget_reached",
    }


def test_iteration_budget_cannot_be_exceeded() -> None:
    toolkit = FakeToolkit(has_graph=True)
    agent = ResearchAgent(
        toolkit,  # type: ignore[arg-type]
        config=ResearchAgentConfig(
            max_tool_calls_per_pass=4,
            max_iterations_per_pass=1,
            max_evidence_per_sub_question=8,
            allow_policy_override=False,
        ),
    )
    result = agent.research_sub_question(
        _sq("sq_iter", "Compare Self-RAG versus CRAG", QueryType.COMPARISON)
    )
    assert result.iteration_count == 1
    assert result.tool_call_count == 1
    assert result.terminated_reason == "iteration_budget_exhausted"


def test_finishing_exactly_at_budget_is_not_reported_exhausted() -> None:
    toolkit = FakeToolkit(has_graph=True)
    agent = ResearchAgent(
        toolkit,  # type: ignore[arg-type]
        config=ResearchAgentConfig(
            max_tool_calls_per_pass=2,
            max_iterations_per_pass=2,
            max_evidence_per_sub_question=8,
            allow_policy_override=False,
        ),
    )
    result = agent.research_sub_question(
        _sq("sq_exact", "Compare Self-RAG versus CRAG", QueryType.COMPARISON)
    )
    assert result.tool_call_count == 2
    assert result.iteration_count == 2
    assert result.terminated_reason == "completed"


def test_latency_budget_stops_additional_tools() -> None:
    toolkit = SlowFakeToolkit(has_graph=True)
    agent = ResearchAgent(
        toolkit,  # type: ignore[arg-type]
        config=ResearchAgentConfig(
            max_tool_calls_per_pass=4,
            max_iterations_per_pass=4,
            max_evidence_per_sub_question=8,
            max_latency_ms=1,
            allow_policy_override=False,
        ),
    )
    result = agent.research_sub_question(
        _sq("sq_slow", "Compare Self-RAG versus CRAG", QueryType.COMPARISON)
    )
    assert result.tool_call_count == 1
    assert result.terminated_reason == "latency_budget_exhausted"


def test_evidence_budget_cannot_be_exceeded() -> None:
    toolkit = FakeToolkit(has_graph=False)
    agent = ResearchAgent(
        toolkit,  # type: ignore[arg-type]
        config=ResearchAgentConfig(
            max_tool_calls_per_pass=4,
            max_evidence_per_sub_question=1,
            allow_policy_override=False,
        ),
    )
    result = agent.research_sub_question(_sq("sq_e", "What is RAPTOR?", QueryType.SEMANTIC))
    assert len(result.evidence) <= 1


def test_parallel_multi_subquestion_merges_ledger() -> None:
    toolkit = FakeToolkit(has_graph=True)
    agent = ResearchAgent(
        toolkit,  # type: ignore[arg-type]
        config=ResearchAgentConfig(max_tool_calls_per_pass=1, allow_policy_override=False),
    )
    run = agent.research_many(
        [
            _sq("sq_1", "What is Self-RAG?", QueryType.SEMANTIC),
            _sq("sq_2", "Compare Self-RAG and CRAG", QueryType.COMPARISON),
        ],
        original_query="Self-RAG overview and comparison",
        parallel=True,
    )
    assert len(run.passes) == 2
    assert run.tool_call_count == sum(p.tool_call_count for p in run.passes)
    # Events include start/finish and tool selections
    types = {e.event_type.value for e in run.events}
    assert "run_started" in types
    assert "tool_selected" in types or "tool_result" in types
    assert "run_finished" in types


def test_duplicate_questions_are_not_parallelized() -> None:
    agent = ResearchAgent(
        FakeToolkit(has_graph=False),  # type: ignore[arg-type]
        config=ResearchAgentConfig(max_tool_calls_per_pass=1),
    )
    run = agent.research_many(
        [
            _sq("sq_1", "What is Self-RAG?", QueryType.SEMANTIC),
            _sq("sq_2", "  what IS self-rag? ", QueryType.SEMANTIC),
        ],
        original_query="duplicate work",
        parallel=True,
    )
    assert run.parallel is False
    assert run.events[0].payload["parallel_requested"] is True
    assert run.events[0].payload["parallel_safe"] is False


def test_budget_events_emitted() -> None:
    toolkit = FakeToolkit(has_graph=True)
    agent = ResearchAgent(
        toolkit,  # type: ignore[arg-type]
        config=ResearchAgentConfig(max_tool_calls_per_pass=1, allow_policy_override=False),
    )
    result = agent.research_sub_question(_sq("sq_x", "Compare A versus B", QueryType.COMPARISON))
    # hybrid_plus_graph wants 2 tools; with budget 1 should hit tool budget or complete after 1
    assert result.tool_call_count == 1
    assert any(
        e.event_type.value in {"budget_hit", "terminated", "tool_result"} for e in result.events
    )


def test_tool_result_trace_contains_sources_pages_and_scores() -> None:
    agent = ResearchAgent(
        FakeToolkit(has_graph=False),  # type: ignore[arg-type]
        config=ResearchAgentConfig(allow_policy_override=False),
    )
    result = agent.research_sub_question(
        _sq("sq_trace", "What is RAG conceptually?", QueryType.SEMANTIC)
    )
    tool_event = next(event for event in result.events if event.event_type.value == "tool_result")
    hit = tool_event.payload["hits"][0]
    assert hit["chunk_id"]
    assert hit["paper_id"] == "paper_demo"
    assert hit["page_start"] == 1
    assert hit["score"] == 0.9


def test_research_subgraph_preserves_events_and_state_budgets() -> None:
    agent = ResearchAgent(
        FakeToolkit(has_graph=True),  # type: ignore[arg-type]
        config=ResearchAgentConfig(
            max_tool_calls_per_pass=4,
            max_iterations_per_pass=4,
            max_evidence_per_sub_question=8,
            allow_policy_override=False,
        ),
    )
    sub_question = _sq(
        "sq_graph",
        "Compare Self-RAG versus CRAG",
        QueryType.COMPARISON,
    )
    routing = recommend_policy(
        sub_question.question,
        query_type=sub_question.query_type,
        has_graph=True,
    )
    graph = build_research_subgraph(agent)
    state = graph.invoke(
        {
            "run_id": "run_subgraph",
            "sub_question": sub_question.model_dump(mode="json"),
            "routing": routing.model_dump(mode="json"),
            "tool_call_count": 0,
            "iteration": 0,
            "max_tool_calls": 4,
            "max_iterations": 1,
            "max_evidence": 1,
            "evidence": [],
            "actions": [],
            "events": [],
            "modes_queue": policy_to_tool_modes(routing.recommended_policy),
        }
    )
    assert state["tool_call_count"] == 1
    assert state["iteration"] == 1
    assert len(state["evidence"]) <= 1
    assert state["terminated_reason"] in {
        "iteration_budget_exhausted",
        "evidence_budget_reached",
    }
    event_types = [event["event_type"] for event in state["events"]]
    assert "decision" in event_types
    assert "tool_selected" in event_types
    assert "tool_result" in event_types
    assert "terminated" in event_types


def test_token_budget_caps_evidence_and_emits_structured_event() -> None:
    agent = ResearchAgent(
        FakeToolkit(has_graph=False),  # type: ignore[arg-type]
        config=ResearchAgentConfig(
            max_total_tokens_per_pass=10,
            max_tool_calls_per_pass=2,
            allow_policy_override=False,
        ),
    )
    result = agent.research_sub_question(_sq("sq_tokens", "What is RAPTOR?", QueryType.SEMANTIC))
    assert result.token_usage.total_tokens <= 10
    assert result.terminated_reason == "token_budget_exhausted"
    assert any(event.event_type.value == "budget_hit" for event in result.events)


def test_research_run_aggregates_token_usage() -> None:
    agent = ResearchAgent(
        FakeToolkit(has_graph=False),  # type: ignore[arg-type]
        config=ResearchAgentConfig(
            max_total_tokens_per_pass=100,
            max_tool_calls_per_pass=1,
            allow_policy_override=False,
        ),
    )
    result = agent.research_many(
        [
            _sq("sq_t1", "What is RAPTOR?", QueryType.SEMANTIC),
            _sq("sq_t2", "What is CRAG?", QueryType.SEMANTIC),
        ],
        original_query="Two questions",
        parallel=False,
    )
    assert result.token_usage.total_tokens == sum(
        research_pass.token_usage.total_tokens for research_pass in result.passes
    )
    assert result.token_usage.total_tokens > 0
