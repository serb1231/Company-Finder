from pathlib import Path
import hashlib

from pydantic import ValidationError

from data_used_by_llm.schemas import QuerySpec


# Cache the results as users ask multiple times the same
# question
CACHE_DIR = Path(".query_cache")

from data_used_by_llm.prompts import SYSTEM_PROMPT

VERSION = hashlib.md5(
    (SYSTEM_PROMPT + str(QuerySpec.model_json_schema())).encode()
).hexdigest()[:8]

class CacheQuery:
    def __init__(self, cache_dir: Path = CACHE_DIR, model: str = ""):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.model = model

    # get the key for hashing a query
    def cache_key(self, query: str) -> Path:
        h = hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]
        return self.cache_dir / f"{self.model}:{VERSION}:{h}.json"


    # get the query
    def cache_get(self, query: str) -> QuerySpec | None:
        path = self.cache_key(query)
        if path.exists():
            try:
                return QuerySpec.model_validate_json(path.read_text())
            except ValidationError:
                path.unlink(missing_ok=True)
        return None


    # put the query in the cache
    def cache_put(self, query: str, spec: QuerySpec) -> None:
        self.cache_key(query).write_text(spec.model_dump_json(indent=2))

    def cache_remove(self, query: str, spec: QuerySpec) -> None:
        self.cache_key(query).unlink()


if __name__ == '__main__':
    cache = CacheQuery()
    query = "companies supplying packaging for cosmetics brands"
    spec = cache.cache_get(query)
    if spec is None:
        print("Not cached, creating a dummy spec")
        spec = QuerySpec(
            hard_filters={},
            naics_keywords=["packaging", "cosmetics"],
            ideal_match_description="Manufactures packaging materials such as bottles, jars, tubes and boxes. Sells B2B to consumer goods and beauty brands.",
            key_terms=[],
            complexity="structured",
            reasoning="Dummy spec for testing."
        )
        cache.cache_put(query, spec)
    else:
        print("Cached spec found:")
    print(spec.model_dump_json(indent=2))

    # now remove
    cache.cache_remove(query, spec)