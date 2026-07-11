"""Seed aliases and known entities for RAG/agent literature.

Used by the staged entity resolver (exact alias stage) and offline extractors.
"""

from __future__ import annotations

from scholar_agent.models.graph import EntityType

# surface (lowercase key) -> (canonical_name, entity_type)
SEED_ALIASES: dict[str, tuple[str, EntityType]] = {
    # Methods / systems
    "rag": ("Retrieval-Augmented Generation", EntityType.METHOD),
    "retrieval-augmented generation": ("Retrieval-Augmented Generation", EntityType.METHOD),
    "retrieval augmented generation": ("Retrieval-Augmented Generation", EntityType.METHOD),
    "self-rag": ("Self-RAG", EntityType.METHOD),
    "self rag": ("Self-RAG", EntityType.METHOD),
    "selfrag": ("Self-RAG", EntityType.METHOD),
    "crag": ("Corrective RAG", EntityType.METHOD),
    "corrective rag": ("Corrective RAG", EntityType.METHOD),
    "corrective retrieval augmented generation": ("Corrective RAG", EntityType.METHOD),
    "dpr": ("Dense Passage Retrieval", EntityType.METHOD),
    "dense passage retrieval": ("Dense Passage Retrieval", EntityType.METHOD),
    "bm25": ("BM25", EntityType.METHOD),
    "colbert": ("ColBERT", EntityType.METHOD),
    "colbertv2": ("ColBERTv2", EntityType.METHOD),
    "contriever": ("Contriever", EntityType.METHOD),
    "splade": ("SPLADE", EntityType.METHOD),
    "hyde": ("HyDE", EntityType.METHOD),
    "raptor": ("RAPTOR", EntityType.METHOD),
    "hipporag": ("HippoRAG", EntityType.METHOD),
    "lightrag": ("LightRAG", EntityType.METHOD),
    "graphrag": ("GraphRAG", EntityType.METHOD),
    "graph rag": ("GraphRAG", EntityType.METHOD),
    "react": ("ReAct", EntityType.METHOD),
    "toolformer": ("Toolformer", EntityType.METHOD),
    "reflexion": ("Reflexion", EntityType.METHOD),
    "flare": ("FLARE", EntityType.METHOD),
    "realm": ("REALM", EntityType.METHOD),
    "retro": ("RETRO", EntityType.METHOD),
    "atlas": ("Atlas", EntityType.METHOD),
    "fid": ("Fusion-in-Decoder", EntityType.METHOD),
    "fusion-in-decoder": ("Fusion-in-Decoder", EntityType.METHOD),
    "sbert": ("Sentence-BERT", EntityType.METHOD),
    "sentence-bert": ("Sentence-BERT", EntityType.METHOD),
    "bge": ("BGE", EntityType.METHOD),
    "e5": ("E5", EntityType.METHOD),
    # Datasets
    "natural questions": ("Natural Questions", EntityType.DATASET),
    "nq": ("Natural Questions", EntityType.DATASET),
    "triviaqa": ("TriviaQA", EntityType.DATASET),
    "hotpotqa": ("HotpotQA", EntityType.DATASET),
    "hotpot qa": ("HotpotQA", EntityType.DATASET),
    "ms marco": ("MS MARCO", EntityType.DATASET),
    "msmarco": ("MS MARCO", EntityType.DATASET),
    "beir": ("BEIR", EntityType.DATASET),
    "mteb": ("MTEB", EntityType.DATASET),
    "kilt": ("KILT", EntityType.DATASET),
    "wikipedia": ("Wikipedia", EntityType.DATASET),
    "multihop-rag": ("MultiHop-RAG", EntityType.DATASET),
    "ragbench": ("RAGBench", EntityType.DATASET),
    # Tasks
    "open-domain qa": ("Open-Domain QA", EntityType.TASK),
    "open domain qa": ("Open-Domain QA", EntityType.TASK),
    "open-domain question answering": ("Open-Domain QA", EntityType.TASK),
    "question answering": ("Question Answering", EntityType.TASK),
    "multi-hop qa": ("Multi-Hop QA", EntityType.TASK),
    "multi hop qa": ("Multi-Hop QA", EntityType.TASK),
    "fact verification": ("Fact Verification", EntityType.TASK),
    "retrieval": ("Information Retrieval", EntityType.TASK),
    "information retrieval": ("Information Retrieval", EntityType.TASK),
    # Metrics
    "em": ("Exact Match", EntityType.METRIC),
    "exact match": ("Exact Match", EntityType.METRIC),
    "f1": ("F1", EntityType.METRIC),
    "ndcg": ("nDCG", EntityType.METRIC),
    "ndcg@10": ("nDCG@10", EntityType.METRIC),
    "mrr": ("MRR", EntityType.METRIC),
    "recall@k": ("Recall@k", EntityType.METRIC),
    "recall": ("Recall", EntityType.METRIC),
    "precision": ("Precision", EntityType.METRIC),
    "ragas": ("RAGAS", EntityType.METRIC),
    "faithfulness": ("Faithfulness", EntityType.METRIC),
    # Organizations
    "openai": ("OpenAI", EntityType.ORGANIZATION),
    "google": ("Google", EntityType.ORGANIZATION),
    "meta": ("Meta", EntityType.ORGANIZATION),
    "facebook ai": ("Meta", EntityType.ORGANIZATION),
    "microsoft": ("Microsoft", EntityType.ORGANIZATION),
    "deepmind": ("DeepMind", EntityType.ORGANIZATION),
    "anthropic": ("Anthropic", EntityType.ORGANIZATION),
    "huggingface": ("Hugging Face", EntityType.ORGANIZATION),
    "hugging face": ("Hugging Face", EntityType.ORGANIZATION),
}


# Acronym expansions used in stage 2 of the resolver
ACRONYM_MAP: dict[str, str] = {
    "rag": "retrieval-augmented generation",
    "dpr": "dense passage retrieval",
    "crag": "corrective rag",
    "sbert": "sentence-bert",
    "nq": "natural questions",
    "em": "exact match",
    "mrr": "mean reciprocal rank",
    "ndcg": "normalized discounted cumulative gain",
    "ir": "information retrieval",
    "odqa": "open-domain qa",
    "llm": "large language model",
    "llms": "large language models",
}
