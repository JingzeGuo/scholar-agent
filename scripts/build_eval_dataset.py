#!/usr/bin/env python3
"""Build the frozen Phase-8 evaluation dataset from the processed corpus.

Produces:
  data/evaluation/questions.jsonl
  data/evaluation/reference_evidence.jsonl
  data/evaluation/frozen_split.json

Re-running is deterministic given the same processed artifacts. Gold labels are
derived from paper_id + keyword match over chunks so evaluation stays offline.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROCESSED = REPO / "data" / "processed"
OUT = REPO / "data" / "evaluation"

# (qid, type, question, arxiv_ids, keywords_per_paper, reference_claims, graph_expected, notes)
# types: factual | keyword | comparison | relational | unanswerable
RAW: list[tuple[str, str, str, list[str], list[list[str]], list[str], bool, str]] = [
    # --- 10 factual ---
    (
        "q_f01",
        "factual",
        "What is Retrieval-Augmented Generation (RAG) as introduced by Lewis et al.?",
        ["2005.11401"],
        [["retrieval-augmented", "parametric", "non-parametric"]],
        [
            "RAG combines parametric memory of a pre-trained seq2seq model with non-parametric retrieved documents."
        ],
        False,
        "Single-paper definition from Lewis et al. 2020.",
    ),
    (
        "q_f02",
        "factual",
        "What architecture does Dense Passage Retrieval (DPR) use for open-domain QA?",
        ["2004.04906"],
        [["dense passage", "dual", "encoder"]],
        ["DPR uses dual-encoder dense representations for questions and passages with MIPS retrieval."],
        False,
        "DPR dual-encoder fact.",
    ),
    (
        "q_f03",
        "factual",
        "What is Sentence-BERT (SBERT) designed to produce?",
        ["1908.10084"],
        [["sentence-bert", "siamese", "sentence embeddings"]],
        ["SBERT modifies BERT with siamese/triplet networks to produce semantically meaningful sentence embeddings."],
        False,
        "SBERT purpose.",
    ),
    (
        "q_f04",
        "factual",
        "What problem does BEIR evaluate for information retrieval models?",
        ["2104.08663"],
        [["beir", "zero-shot", "heterogen"]],
        ["BEIR is a heterogeneous benchmark for zero-shot evaluation of information retrieval models."],
        False,
        "BEIR scope.",
    ),
    (
        "q_f05",
        "factual",
        "What is REALM's key idea for language model pre-training?",
        ["2002.08909"],
        [["realm", "retrieval", "pre-training"]],
        ["REALM augments language model pre-training with a learned retrieval step over a textual knowledge corpus."],
        False,
        "REALM retrieval-augmented pretraining.",
    ),
    (
        "q_f06",
        "factual",
        "What does Fusion-in-Decoder (FiD) do with retrieved passages?",
        ["2007.01282"],
        [["fusion", "decoder", "passage"]],
        ["FiD encodes retrieved passages independently and fuses them in the decoder for open-domain QA."],
        False,
        "FiD fusion strategy.",
    ),
    (
        "q_f07",
        "factual",
        "What is REPLUG's approach to retrieval-augmented black-box LMs?",
        ["2301.12652"],
        [["replug", "black-box", "retrieval"]],
        ["REPLUG treats the language model as a black box and prepends retrieved documents to improve generation."],
        False,
        "REPLUG black-box RAG.",
    ),
    (
        "q_f08",
        "factual",
        "What is Atlas designed for in few-shot learning?",
        ["2208.03299"],
        [["atlas", "few-shot", "retrieval"]],
        ["Atlas is a retrieval-augmented language model aimed at few-shot learning including QA."],
        False,
        "Atlas few-shot RAG.",
    ),
    (
        "q_f09",
        "factual",
        "What does MTEB benchmark?",
        ["2210.07316"],
        [["mteb", "embedding", "benchmark"]],
        ["MTEB is a massive text embedding benchmark covering diverse embedding tasks."],
        False,
        "MTEB definition.",
    ),
    (
        "q_f10",
        "factual",
        "What is Contriever's training approach for dense retrieval?",
        ["2112.09118"],
        [["contriever", "unsupervised", "contrastive"]],
        ["Contriever learns dense retrievers via unsupervised contrastive learning."],
        False,
        "Contriever unsupervised dense IR.",
    ),
    # --- 10 keyword ---
    (
        "q_k01",
        "keyword",
        "What are reflection tokens in Self-RAG?",
        ["2310.11511"],
        [["reflection", "self-rag", "retrieve"]],
        ["Self-RAG uses reflection tokens to decide when to retrieve and to critique generation quality."],
        False,
        "Exact terminology: reflection tokens.",
    ),
    (
        "q_k02",
        "keyword",
        "What does HyDE stand for and how does it help dense retrieval?",
        ["2212.10496"],
        [["hyde", "hypothetical", "document"]],
        ["HyDE means Hypothetical Document Embeddings; it generates a hypothetical answer document for zero-shot dense retrieval."],
        False,
        "Exact acronym HyDE.",
    ),
    (
        "q_k03",
        "keyword",
        "What is the 'lost in the middle' phenomenon for long-context language models?",
        ["2307.03172"],
        [["lost in the middle", "middle", "context"]],
        ["Models often use information at the beginning or end of long contexts better than information placed in the middle."],
        False,
        "Exact phrase lost in the middle.",
    ),
    (
        "q_k04",
        "keyword",
        "What is FLARE / active retrieval augmented generation?",
        ["2305.06983"],
        [["flare", "active retrieval", "forward"]],
        ["FLARE actively retrieves when the model needs additional information during generation."],
        False,
        "FLARE / active RAG term.",
    ),
    (
        "q_k05",
        "keyword",
        "What is MultiHop-RAG designed to benchmark?",
        ["2401.15391"],
        [["multihop", "multi-hop", "retrieval-augmented"]],
        ["MultiHop-RAG benchmarks retrieval-augmented generation for multi-hop questions."],
        False,
        "Exact benchmark name MultiHop-RAG.",
    ),
    (
        "q_k06",
        "keyword",
        "What is RA-DIT?",
        ["2310.01352"],
        [["ra-dit", "dual instruction", "retrieval-augmented"]],
        ["RA-DIT is Retrieval-Augmented Dual Instruction Tuning for aligning LLMs with retrieval."],
        False,
        "Exact method name RA-DIT.",
    ),
    (
        "q_k07",
        "keyword",
        "What is RQ-RAG?",
        ["2404.00610"],
        [["rq-rag", "refine", "queries"]],
        ["RQ-RAG learns to refine queries for retrieval-augmented generation."],
        False,
        "Exact method name RQ-RAG.",
    ),
    (
        "q_k08",
        "keyword",
        "What does the RAGAs framework evaluate?",
        ["2309.15217"],
        [["ragas", "faithfulness", "evaluation"]],
        ["RAGAs provides automated evaluation metrics for retrieval-augmented generation such as faithfulness."],
        False,
        "Exact product name RAGAs.",
    ),
    (
        "q_k09",
        "keyword",
        "What is LightRAG?",
        ["2410.05779"],
        [["lightrag", "graph", "retrieval-augmented"]],
        ["LightRAG is a simple and fast graph-empowered retrieval-augmented generation system."],
        False,
        "Exact system name LightRAG.",
    ),
    (
        "q_k10",
        "keyword",
        "What is HippoRAG inspired by?",
        ["2405.14831"],
        [["hipporag", "hippocamp", "memory"]],
        ["HippoRAG is neurobiologically inspired by hippocampal memory indexing for long-term memory RAG."],
        False,
        "Exact name HippoRAG + inspiration.",
    ),
    # --- 15 comparison ---
    (
        "q_c01",
        "comparison",
        "Compare Self-RAG and CRAG: how do their corrective or critique mechanisms differ?",
        ["2310.11511", "2401.15884"],
        [["self-rag", "reflection"], ["corrective", "crag", "evaluator"]],
        [
            "Self-RAG uses reflection tokens to retrieve on demand and critique generations.",
            "CRAG evaluates retrieved documents and triggers corrective retrieval when quality is low.",
        ],
        False,
        "Cross-paper Self-RAG vs CRAG.",
    ),
    (
        "q_c02",
        "comparison",
        "How do ReAct and Tree of Thoughts differ in structuring intermediate reasoning?",
        ["2210.03629", "2305.10601"],
        [["react", "reasoning and acting"], ["tree of thoughts", "deliberate"]],
        [
            "ReAct interleaves verbal reasoning traces with actions/tool use.",
            "Tree of Thoughts explores multiple deliberative thought branches rather than a single chain.",
        ],
        False,
        "ReAct vs ToT.",
    ),
    (
        "q_c03",
        "comparison",
        "Compare dense passage retrieval (DPR) with unsupervised Contriever.",
        ["2004.04906", "2112.09118"],
        [["dense passage", "dpr"], ["contriever", "unsupervised"]],
        [
            "DPR is typically trained with supervised question-passage pairs.",
            "Contriever trains dense retrievers with unsupervised contrastive learning.",
        ],
        False,
        "DPR vs Contriever training.",
    ),
    (
        "q_c04",
        "comparison",
        "How does GraphRAG (From Local to Global) differ from standard vector RAG?",
        ["2404.16130", "2005.11401"],
        [["graph rag", "community", "global"], ["retrieval-augmented", "parametric"]],
        [
            "GraphRAG builds a knowledge graph and uses community summaries for global questions.",
            "Classic RAG retrieves text chunks via dense/sparse retrieval without an explicit corpus graph.",
        ],
        True,
        "GraphRAG vs classic RAG.",
    ),
    (
        "q_c05",
        "comparison",
        "Compare Toolformer and ReAct for tool use in language models.",
        ["2302.04761", "2210.03629"],
        [["toolformer", "api"], ["react", "acting"]],
        [
            "Toolformer self-supervises API call insertion during LM training.",
            "ReAct prompts models to interleave reasoning and acting with tools at inference time.",
        ],
        False,
        "Toolformer vs ReAct.",
    ),
    (
        "q_c06",
        "comparison",
        "How do HyDE and query rewriting for RAG differ as query enhancement strategies?",
        ["2212.10496", "2305.14283"],
        [["hyde", "hypothetical"], ["query rewriting", "retrieval-augmented"]],
        [
            "HyDE generates a hypothetical document embedding to query dense indexes.",
            "Query rewriting reformulates the user question for better retrieval-augmented LLM performance.",
        ],
        False,
        "HyDE vs query rewrite.",
    ),
    (
        "q_c07",
        "comparison",
        "Compare AutoGen and MetaGPT as multi-agent frameworks.",
        ["2308.08155", "2308.00352"],
        [["autogen", "multi-agent conversation"], ["metagpt", "sop", "multi-agent"]],
        [
            "AutoGen enables multi-agent conversation applications with customizable agents.",
            "MetaGPT encodes SOPs for multi-agent collaboration inspired by software engineering roles.",
        ],
        False,
        "AutoGen vs MetaGPT.",
    ),
    (
        "q_c08",
        "comparison",
        "How do BEIR and MTEB differ in what they evaluate?",
        ["2104.08663", "2210.07316"],
        [["beir", "retrieval"], ["mteb", "embedding"]],
        [
            "BEIR focuses on heterogeneous zero-shot information retrieval benchmarks.",
            "MTEB evaluates text embedding models across many embedding tasks.",
        ],
        False,
        "BEIR vs MTEB.",
    ),
    (
        "q_c09",
        "comparison",
        "Compare FiD and classic RAG generation patterns for using multiple passages.",
        ["2007.01282", "2005.11401"],
        [["fusion-in-decoder", "fusion"], ["retrieval-augmented", "generation"]],
        [
            "FiD encodes passages separately and fuses in the decoder.",
            "RAG-style models condition generation on retrieved documents as non-parametric memory.",
        ],
        False,
        "FiD vs RAG generation.",
    ),
    (
        "q_c10",
        "comparison",
        "How does LightRAG relate to GraphRAG-style graph-empowered RAG?",
        ["2410.05779", "2404.16130"],
        [["lightrag", "graph"], ["graph rag", "community"]],
        [
            "LightRAG emphasizes simple/fast graph-empowered RAG.",
            "From Local to Global GraphRAG emphasizes hierarchical community summaries for global sensemaking.",
        ],
        True,
        "LightRAG vs GraphRAG.",
    ),
    (
        "q_c11",
        "comparison",
        "Compare Reflexion and ReAct for agent improvement loops.",
        ["2303.11366", "2210.03629"],
        [["reflexion", "verbal reinforcement"], ["react", "reasoning"]],
        [
            "Reflexion uses verbal reinforcement / self-reflection memory to improve agents.",
            "ReAct focuses on interleaving reasoning and acting during a trajectory.",
        ],
        False,
        "Reflexion vs ReAct.",
    ),
    (
        "q_c12",
        "comparison",
        "How do G-Retriever and HippoRAG differ as graph-oriented RAG methods?",
        ["2402.07630", "2405.14831"],
        [["g-retriever", "graph"], ["hipporag", "memory"]],
        [
            "G-Retriever targets retrieval-augmented generation over textual graphs.",
            "HippoRAG uses neurobiologically inspired indexing for multi-hop RAG memory.",
        ],
        True,
        "G-Retriever vs HippoRAG.",
    ),
    (
        "q_c13",
        "comparison",
        "Compare RAGAs and RAGBench as evaluation resources for RAG systems.",
        ["2309.15217", "2407.11005"],
        [["ragas", "evaluation"], ["ragbench", "benchmark"]],
        [
            "RAGAs offers automated metrics for RAG evaluation such as faithfulness.",
            "RAGBench is an explainable benchmark for retrieval-augmented generation.",
        ],
        False,
        "RAGAs vs RAGBench.",
    ),
    (
        "q_c14",
        "comparison",
        "How do HuggingGPT and Toolformer differ in orchestrating tools/models?",
        ["2303.17580", "2302.04761"],
        [["hugginggpt", "chatgpt"], ["toolformer", "tools"]],
        [
            "HuggingGPT uses ChatGPT as a controller to plan and call specialist models.",
            "Toolformer teaches a single LM to call APIs via self-supervised training.",
        ],
        False,
        "HuggingGPT vs Toolformer.",
    ),
    (
        "q_c15",
        "comparison",
        "Compare RETRO and Atlas as retrieval-augmented language models.",
        ["2112.04426", "2208.03299"],
        [["retro", "retriev"], ["atlas", "few-shot"]],
        [
            "RETRO improves LMs by retrieving from trillions of tokens at scale.",
            "Atlas focuses on few-shot learning with retrieval-augmented LMs.",
        ],
        False,
        "RETRO vs Atlas.",
    ),
    # --- 10 relational / multi-hop ---
    (
        "q_r01",
        "relational",
        "Which paper introduces reflection tokens for retrieve-on-demand generation, and how does that relate to corrective document evaluation in CRAG?",
        ["2310.11511", "2401.15884"],
        [["reflection", "self-rag"], ["corrective", "crag"]],
        [
            "Self-RAG introduces reflection tokens for retrieve-on-demand and critique.",
            "CRAG adds an evaluator over retrieved documents for corrective retrieval.",
        ],
        True,
        "Multi-hop Self-RAG→CRAG relation.",
    ),
    (
        "q_r02",
        "relational",
        "How does ReAct's reasoning-acting loop connect to multi-agent conversation systems like AutoGen?",
        ["2210.03629", "2308.08155"],
        [["react", "acting"], ["autogen", "multi-agent"]],
        [
            "ReAct shows single-agent reasoning+acting with tools.",
            "AutoGen extends collaboration to multi-agent conversations for applications.",
        ],
        True,
        "ReAct to AutoGen hop.",
    ),
    (
        "q_r03",
        "relational",
        "How does classic RAG (Lewis et al.) underpin later GraphRAG global query answering?",
        ["2005.11401", "2404.16130"],
        [["retrieval-augmented generation"], ["graph rag", "global"]],
        [
            "Lewis et al. establish RAG with parametric + non-parametric memory.",
            "GraphRAG builds on retrieval-augmented generation with graph communities for global questions.",
        ],
        True,
        "RAG foundation → GraphRAG.",
    ),
    (
        "q_r04",
        "relational",
        "What retrieval challenge does 'lost in the middle' imply for long retrieved contexts in RAG pipelines?",
        ["2307.03172", "2005.11401"],
        [["lost in the middle", "middle"], ["retrieval-augmented"]],
        [
            "Models underuse evidence placed in the middle of long contexts.",
            "RAG systems that concatenate many retrieved passages can suffer from this position bias.",
        ],
        False,
        "Lost-in-middle implications for RAG.",
    ),
    (
        "q_r05",
        "relational",
        "How do unsupervised Contriever retrievers relate to BEIR zero-shot evaluation goals?",
        ["2112.09118", "2104.08663"],
        [["contriever", "unsupervised"], ["beir", "zero-shot"]],
        [
            "Contriever trains dense retrievers without labeled pairs.",
            "BEIR stresses zero-shot transfer across heterogeneous retrieval tasks.",
        ],
        False,
        "Contriever ↔ BEIR zero-shot.",
    ),
    (
        "q_r06",
        "relational",
        "How does Toolformer's API-calling ability relate to HuggingGPT's model orchestration?",
        ["2302.04761", "2303.17580"],
        [["toolformer", "api"], ["hugginggpt", "tasks"]],
        [
            "Toolformer learns when/how to call APIs from a single LM.",
            "HuggingGPT uses an LLM controller to dispatch tasks to many expert models.",
        ],
        True,
        "Tool use lineage Toolformer→HuggingGPT.",
    ),
    (
        "q_r07",
        "relational",
        "How does MultiHop-RAG connect multi-hop questions to retrieval-augmented generation evaluation?",
        ["2401.15391", "2312.10997"],
        [["multihop", "multi-hop"], ["retrieval-augmented generation", "survey"]],
        [
            "MultiHop-RAG provides a benchmark for multi-hop RAG questions.",
            "RAG surveys frame retrieval, generation, and augmentation stages that multi-hop settings stress.",
        ],
        True,
        "Multi-hop RAG evaluation link.",
    ),
    (
        "q_r08",
        "relational",
        "How does active retrieval (FLARE) address limitations of one-shot retrieve-then-read RAG?",
        ["2305.06983", "2005.11401"],
        [["active retrieval", "flare"], ["retrieval-augmented"]],
        [
            "FLARE retrieves adaptively during generation when more information is needed.",
            "Classic RAG often retrieves once before generation.",
        ],
        False,
        "FLARE vs one-shot RAG.",
    ),
    (
        "q_r09",
        "relational",
        "How do hallucination surveys motivate automated RAG evaluation metrics like those in RAGAs?",
        ["2311.05232", "2309.15217"],
        [["hallucination"], ["ragas", "faithfulness"]],
        [
            "Hallucination surveys document LLM fabrication risks.",
            "RAGAs automates faithfulness-style checks for RAG answers grounded in retrieved context.",
        ],
        False,
        "Hallucination → RAGAs motivation.",
    ),
    (
        "q_r10",
        "relational",
        "How does Voyager's lifelong skill acquisition relate to multi-agent systems that scale the number of agents?",
        ["2305.16291", "2402.05120"],
        [["voyager", "lifelong", "minecraft"], ["more agents", "ensemble"]],
        [
            "Voyager is an open-ended embodied agent that continuously acquires skills.",
            "More Agents Is All You Need studies performance gains from ensembling many agents.",
        ],
        True,
        "Agent scaling / lifelong agents hop.",
    ),
    # --- 5 unanswerable ---
    (
        "q_u01",
        "unanswerable",
        "What is the exact GPU price schedule used inside OpenAI's private 2026 training cluster for GPT-6?",
        [],
        [],
        ["The corpus does not contain proprietary 2026 OpenAI cluster pricing."],
        False,
        "Out-of-corpus proprietary ops detail.",
    ),
    (
        "q_u02",
        "unanswerable",
        "Which arXiv paper in this corpus proves P=NP using retrieval-augmented generation?",
        [],
        [],
        ["No paper in the corpus claims a P=NP proof via RAG."],
        False,
        "False claim; unanswerable/refuse.",
    ),
    (
        "q_u03",
        "unanswerable",
        "What is the recommended dosage of amoxicillin for pediatric RAG-induced fever according to Self-RAG?",
        ["2310.11511"],
        [["self-rag"]],
        ["Self-RAG is an NLP method paper; it does not provide medical dosing guidance."],
        False,
        "Medical question not supported by NLP paper.",
    ),
    (
        "q_u04",
        "unanswerable",
        "What was the closing stock price of GraphRAG Inc. on NASDAQ yesterday?",
        [],
        [],
        ["The literature corpus has no live financial market prices."],
        False,
        "Live finance not in corpus.",
    ),
    (
        "q_u05",
        "unanswerable",
        "According to the corpus, who won the 2028 ACM Turing Award for inventing LightRAG?",
        [],
        [],
        ["No 2028 Turing Award result exists in this corpus."],
        False,
        "Future event; unanswerable.",
    ),
]


def load_papers() -> dict[str, dict]:
    by_ax: dict[str, dict] = {}
    for line in (PROCESSED / "papers.jsonl").open():
        p = json.loads(line)
        ax = p.get("arxiv_id")
        if ax:
            by_ax[str(ax)] = p
    return by_ax


def load_chunks() -> dict[str, list[dict]]:
    by_pid: dict[str, list[dict]] = defaultdict(list)
    for line in (PROCESSED / "chunks.jsonl").open():
        c = json.loads(line)
        by_pid[c["paper_id"]].append(c)
    return by_pid


def pick_chunk(chunks: list[dict], keywords: list[str]) -> dict | None:
    if not chunks:
        return None
    kws = [k.lower() for k in keywords]
    scored: list[tuple[int, dict]] = []
    for c in chunks:
        text = c.get("text", "").lower()
        score = sum(1 for k in kws if k in text)
        # Prefer early pages slightly
        score = score * 10 - int(c.get("page_start", 99))
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    best_score, best = scored[0]
    if best_score <= -50 and kws:
        # fallback: first substantial early chunk
        early = sorted(chunks, key=lambda c: (c["page_start"], -len(c.get("text", ""))))
        return early[0]
    return best


def main() -> None:
    if not (PROCESSED / "papers.jsonl").is_file():
        raise SystemExit("processed papers missing; run ingest first")
    papers = load_papers()
    chunks = load_chunks()
    OUT.mkdir(parents=True, exist_ok=True)

    questions: list[dict] = []
    refs: list[dict] = []

    for (
        qid,
        qtype,
        question,
        arxiv_ids,
        kw_lists,
        claims,
        graph_expected,
        notes,
    ) in RAW:
        required_papers: list[str] = []
        gold_items: list[dict] = []
        for i, ax in enumerate(arxiv_ids):
            p = papers.get(ax)
            if p is None:
                raise SystemExit(f"paper arxiv {ax} missing for {qid}")
            pid = p["paper_id"]
            required_papers.append(pid)
            kws = kw_lists[i] if i < len(kw_lists) else []
            ch = pick_chunk(chunks.get(pid, []), kws)
            if ch is None:
                raise SystemExit(f"no chunks for {pid} ({qid})")
            gold_items.append(
                {
                    "paper_id": pid,
                    "chunk_id": ch["chunk_id"],
                    "page_start": ch["page_start"],
                    "page_end": ch["page_end"],
                    "relevance": 1.0,
                    "snippet": re.sub(r"\s+", " ", ch["text"])[:280],
                }
            )
            refs.append(
                {
                    "question_id": qid,
                    "paper_id": pid,
                    "chunk_id": ch["chunk_id"],
                    "page_start": ch["page_start"],
                    "page_end": ch["page_end"],
                    "relevance": 1.0,
                    "keywords": kws,
                }
            )

        questions.append(
            {
                "question_id": qid,
                "question": question,
                "question_type": qtype,
                "reference_answer": " ".join(claims),
                "reference_claims": claims,
                "required_paper_ids": required_papers,
                "required_chunk_ids": [g["chunk_id"] for g in gold_items],
                "gold_evidence": gold_items,
                "acceptable_alternate_paper_ids": [],
                "graph_reasoning_expected": graph_expected,
                "unanswerable": qtype == "unanswerable",
                "annotation_notes": notes,
            }
        )

    # Type counts check
    counts: dict[str, int] = defaultdict(int)
    for q in questions:
        counts[q["question_type"]] += 1
    expected = {
        "factual": 10,
        "keyword": 10,
        "comparison": 15,
        "relational": 10,
        "unanswerable": 5,
    }
    if dict(counts) != expected or len(questions) != 50:
        raise SystemExit(f"bad dataset composition: {dict(counts)} n={len(questions)}")

    q_path = OUT / "questions.jsonl"
    r_path = OUT / "reference_evidence.jsonl"
    with q_path.open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    with r_path.open("w", encoding="utf-8") as fh:
        for r in refs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    material = q_path.read_bytes() + b"\n" + r_path.read_bytes()
    fingerprint = sha256(material).hexdigest()
    split = {
        "name": "scholaragent_eval_v1",
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "n_questions": 50,
        "type_counts": expected,
        "question_ids": [q["question_id"] for q in questions],
        "fingerprint_sha256": fingerprint,
        "frozen": True,
        "notes": (
            "Frozen before final system tuning. Keep out of development prompts/fixtures. "
            "Gold chunk IDs resolved from data/processed at freeze time."
        ),
    }
    s_path = OUT / "frozen_split.json"
    s_path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {q_path} ({len(questions)} questions)")
    print(f"wrote {r_path} ({len(refs)} evidence rows)")
    print(f"wrote {s_path} fingerprint={fingerprint[:16]}…")


if __name__ == "__main__":
    main()
