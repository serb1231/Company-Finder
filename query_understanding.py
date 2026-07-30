from __future__ import annotations

from pathlib import Path
from typing import List

import ollama
from pydantic import ValidationError

from read_questions import load_questions
from cache_query import CacheQuery

from data_used_by_llm.schemas import QuerySpec, HardFilters, Complexity, KeyTerm, KeyTermList

from data_used_by_llm.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_FOR_KEY_WORDS

MODEL = "gemma2:9b"

class QueryParser:
    # define the model and the cache location

    def __init__(self, model: str = MODEL, querycache: CacheQuery = None):
        self.model = model
        self.cache = querycache

    def parse(self, query: str) -> QuerySpec:
        # in case it were cached before we return it
        cached = self.cache.cache_get(query)
        if cached is not None:
            return cached

        # call the LLM on the query
        spec = self._call_llm(query)

        extracted_terms = self._call_llm_keywords(spec.ideal_match_description)

        spec.key_terms = extracted_terms

        self.cache.cache_put(query, spec)
        return spec


    def _call_llm(self, query: str, retries: int = 2) -> QuerySpec:
        last_err: Exception | None = None
        for _ in range(retries + 1):
            try:
                # every conversation with a LLM starts this way
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Query: {query}"},
                    ],
                    # structured outputs: constrain generation to our schema (Json)
                    format=QuerySpec.model_json_schema(),
                    # deterministic results for a query
                    options={"temperature": 0},
                )
                return QuerySpec.model_validate_json(
                    response["message"]["content"])
            except (ValidationError, KeyError, Exception) as e:
                last_err = e
                continue

        return QuerySpec(
            hard_filters=HardFilters(),
            naics_keywords=[],
            ideal_match_description=query,
            key_terms=[],
            complexity=Complexity.judgment,
            reasoning=f"Parser failed ({last_err}); degraded to semantic-only.",
        )

    def _call_llm_keywords(self, big_description: str, retries: int = 2) -> List[KeyTerm]:
        last_err: Exception | None = None
        for _ in range(retries + 1):
            try:
                # every conversation with a LLM starts this way
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_FOR_KEY_WORDS},
                        {"role": "user", "content": f"Query: {big_description}"},
                    ],
                    # structured outputs: constrain generation to our schema (Json)
                    format=KeyTermList.model_json_schema(),
                    # deterministic results for a query
                    options={"temperature": 0},
                )
                parsed_list = KeyTermList.model_validate_json(
                    response["message"]["content"])

                return parsed_list.keyTerms

            except (ValidationError, KeyError, Exception) as e:
                last_err = e
                continue

        return [KeyTerm(term="", weight=0.0)]  # Return a default KeyTerm if all retries fail


if __name__ == "__main__":
    cache = CacheQuery(Path(".query_cache"), MODEL)
    parser = QueryParser(model=MODEL, querycache=cache)

    for item in load_questions():
        number, question = item["number"], item["question"]
        spec = parser.parse(question)
        print(f"\n=== {number}. {question}")
        print(spec.model_dump_json(indent=2))