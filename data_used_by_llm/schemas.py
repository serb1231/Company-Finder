from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

from data_used_by_llm.vocab import BusinessModel

class BusinessModels(BaseModel):
    required: list[BusinessModel] = Field(default_factory=list)
    preferred: list[BusinessModel] = Field(default_factory=list)


# Structured -> means that we get only hard filters
# from the query, and we can skip LLM later
# Judgment -> means that we let the LLM reason about everything
# Hybrid -> combination between structured and Judgment
class Complexity(str, Enum):
    structured = "structured"   # pure filters -> skip LLM qualification later
    hybrid = "hybrid"           # filters + semantics -> normal cascade
    judgment = "judgment"       # heavy reasoning -> widen the LLM gray zone


# TODO: verify / define the list inside another file in case it extends
# town?
class HardFilters(BaseModel):
    countries: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    min_employees: Optional[int] = None
    max_employees: Optional[int] = None
    min_revenue: Optional[float] = None
    max_revenue: Optional[float] = None
    founded_after: Optional[int] = None
    founded_before: Optional[int] = None
    is_public: Optional[bool] = None


# this will define the relevance for the terms inside
# the query (more like an attention layer) as some terms
# are dependent of others (Company makes PRODUCTS for cosmetics)
class KeyTerm(BaseModel):
    term: str
    weight: float = Field(ge=0.0, le=1.0)


# objects that get returned after a LLM call
class QuerySpec(BaseModel):
    hard_filters: HardFilters
    business_models: BusinessModels = Field(default_factory=BusinessModels)
    # business_models: list[BusinessModel] = Field(default_factory=list)
    naics_keywords: list[str] = Field(default_factory=list)
    ideal_match_description: str
    key_terms: list[KeyTerm] = Field(default_factory=list)
    complexity: Complexity
    reasoning: str
