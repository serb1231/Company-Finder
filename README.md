# Company-Finder

Company-Finder is an intent-qualification and ranking system that determines whether companies from a dataset meaningfully match a user’s search query (e.g. "Logistics companies in Germany"). The system balances accuracy, speed and cost by combining deterministic hard filters, an LLM-based query parser, and a combined semantic + lexical ranking stage.

Quick summary
- Input: free-text user query + companies dataset (data/companies.jsonl)
- Output: ranked and filtered candidates saved to .tmp/ per query
- Design: LLM parses query -> hard filters prune impossible candidates -> embedding + BM25 ranking -> heuristic selection

Getting started
1. Install dependencies (recommended in a virtualenv):
   pip install -r requirements.txt

2. Run the pipeline (simple):
   chmod +x run.sh
   ./run.sh

This performs data enhancement and runs the filter + ranking pipeline for the test queries found in data/questions.csv. Results are written to .tmp/:
- .tmp/tmp_query_{i}_filtered.jsonl  — candidates after hard filters
- .tmp/tmp_query_{i}_filtered_rank.jsonl — ranked candidates with scores
- .tmp/tmp_query_{i}_final_selection.csv — final selected set

Key files
- enhance_data_companies.py — normalises/enriches address and region data
- hard_filter_database.py — deterministic hard filters (location, size, revenue, founding year, public status, business model)
- query_understanding.py — LLM-based query parser (returns QuerySpec used by the pipeline)
- embedding_ranking.py — main pipeline: build docs, embed, BM25, combine via RRF, final selection
- EmbedderQueryAndCompany.py — sentence-transformers embedder with caching
- data/companies.jsonl — original dataset; data/companies_enhanced.jsonl may already contain enriched data
- data/questions.csv — sample queries used by the pipeline
- data_used_by_llm/ — prompts and Pydantic schemas for the LLM parser

Design notes (concise)
- Hard filters are deterministic and cheap; they eliminate impossible matches early (saves LLM/embedding work).
- The LLM is used only once per query to produce a structured QuerySpec (hard filters, ideal match description, key terms, complexity).
- Semantic ranking uses precomputed/cached sentence-transformer embeddings; BM25 captures lexical signals (NAICS, offerings, markets).
- Signals are combined with Reciprocal Rank Fusion (RRF). A final heuristic selects a cutoff based on score cliffs and query judgment markers.

Trade-offs & limitations
- Optimised for cost and latency vs "LLM per company"; accuracy can drop in judgment-heavy cases where deep reasoning is needed.
- Relies on company metadata quality; missing fields degrade precision and increase dependency on semantic ranking.
- For production scale (100k+ companies), move embeddings into a vector DB (FAISS/Milvus) and run hard filters in a DB for fast pre-filtering.

If you want me to:
- Add a small evaluation harness (precision@k) or a quick example to run a single custom query, tell me which and I’ll add it.

(Assistant identity: I am an AI assistant using Copilot CLI runtime in VS Code.)
